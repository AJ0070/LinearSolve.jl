module LinearSolveMUMPSExt

using MUMPS, LinearSolve
using SparseArrays, LinearAlgebra
using LinearSolve: MUMPSFactorization, LinearVerbosity
using LinearSolve.SciMLBase
using SciMLLogging: verbosity_to_bool

# MUMPS supports Float32, Float64, ComplexF32, ComplexF64.
# Map an arbitrary eltype to the closest supported MUMPS type.
function _mumps_T(A, b)
    T = promote_type(eltype(A), eltype(b))
    if T <: Complex
        return T <: Union{ComplexF32, ComplexF64} ? T : ComplexF64
    else
        return T <: Union{Float32, Float64} ? T : Float64
    end
end

# Choose the appropriate real cntl array for the MUMPS arithmetic type.
# Float32/ComplexF32 → single precision; Float64/ComplexF64 → double precision.
_mumps_cntl(::Type{T}) where {T <: Union{Float32, ComplexF32}} = MUMPS.default_cntl32[:]
_mumps_cntl(::Type{T}) where {T}                               = MUMPS.default_cntl64[:]

LinearSolve.needs_concrete_A(::MUMPSFactorization) = true

function LinearSolve.init_cacheval(
        alg::MUMPSFactorization,
        A,
        b,
        u,
        Pl,
        Pr,
        maxiters::Int,
        abstol,
        reltol,
        verbose::Union{LinearVerbosity, Bool},
        assumptions::LinearSolve.OperatorAssumptions
    )
    T = _mumps_T(A, b)

    # Build integer control array, suppressing all output by default.
    icntl = if alg.icntl !== nothing
        alg.icntl[:]
    else
        icntl_work = MUMPS.default_icntl[:]
        # Redirect error/diagnostic/global-info streams and set print level
        # to silent unless the user has requested verbosity.
        should_print = if verbose isa Bool
            verbose
        else
            verbosity_to_bool(verbose.mumps_verbosity)
        end
        if !should_print
            icntl_work[1] = 0   # error messages stream  → suppressed
            icntl_work[2] = 0   # diagnostics/stats/warnings stream → suppressed
            icntl_work[3] = 0   # global info on host stream → suppressed
            icntl_work[4] = 0   # output level (0 = no output)
        end
        icntl_work
    end

    # Build real control array.
    cntl = alg.cntl !== nothing ? alg.cntl[:] : _mumps_cntl(T)

    # Mumps{T}(sym, icntl, cntl; par=…) is the standard convenience constructor.
    mumps = MUMPS.Mumps{T}(alg.sym, icntl, cntl; par = alg.par)
    return mumps
end

function SciMLBase.solve!(
        cache::LinearSolve.LinearCache,
        alg::MUMPSFactorization;
        kwargs...
    )
    A = convert(AbstractMatrix, cache.A)
    b = cache.b
    u = cache.u
    mumps = cache.cacheval

    if cache.isfresh
        # MUMPS.jl's mumps_factorize! skips re-factorization when the job state
        # indicates the object has already been factored or solved.  Reset the job
        # field so the next factorize! call runs ANALYZE_FACTOR unconditionally.
        mumps.job = MUMPS.INITIALIZE
        MUMPS.factorize!(mumps, A)
        cache.isfresh = false
    else
        # After a previous solve, mumps.job == MUMPS.SOLVE (part of SOLVE_JOBS).
        # mumps_solve! silently returns nothing when job ∈ SOLVE_JOBS, which
        # would skip the solve entirely for a repeated RHS.  Reset to FACTOR
        # (∈ ONLY_FACTORED) so the next solve! call executes unconditionally.
        mumps.job = MUMPS.FACTOR
    end

    x = MUMPS.solve(mumps, b)
    copyto!(u, x)

    return SciMLBase.build_linear_solution(alg, cache.u, nothing, cache)
end

end

