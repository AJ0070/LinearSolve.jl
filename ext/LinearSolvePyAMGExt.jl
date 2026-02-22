module LinearSolvePyAMGExt

using LinearSolve, LinearSolvePyAMG, SparseArrays
using LinearSolve: LinearCache, LinearVerbosity, OperatorAssumptions
using SciMLBase: SciMLBase, ReturnCode

function LinearSolve.init_cacheval(
        alg::PyAMGJL, A, b, u, Pl, Pr, maxiters::Int, abstol, reltol,
        verbose::Union{LinearVerbosity, Bool}, assumptions::OperatorAssumptions
    )
    @assert size(A, 1) == size(A, 2) "PyAMG requires a square matrix"

    # PyAMG.jl constructors require a SparseMatrixCSC
    Asp = A isa SparseMatrixCSC ? A : sparse(A)

    if alg.solver === :ruge_stuben
        return LinearSolvePyAMG.RugeStubenSolver(Asp; alg.kwargs...)
    else # :smoothed_aggregation
        return LinearSolvePyAMG.SmoothedAggregationSolver(Asp; alg.kwargs...)
    end
end

function SciMLBase.solve!(cache::LinearCache, alg::PyAMGJL; kwargs...)
    if cache.isfresh
        cache.cacheval = LinearSolve.init_cacheval(
            alg, cache.A, cache.b, cache.u, cache.Pl, cache.Pr,
            cache.maxiters, cache.abstol, cache.reltol, cache.verbose,
            cache.assumptions
        )
        cache.isfresh = false
    end

    amg = cache.cacheval
    tol = cache.reltol
    maxiter = cache.maxiters

    x = LinearSolvePyAMG.solve(amg, cache.b; tol = tol, maxiter = maxiter, kwargs...)
    copyto!(cache.u, x)

    return SciMLBase.build_linear_solution(
        alg, cache.u, nothing, cache; retcode = ReturnCode.Success
    )
end

end
