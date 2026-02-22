"""
    LinearSolvePyAMG

A minimal Julia wrapper for the Python [pyamg](https://pyamg.readthedocs.io) algebraic
multigrid library, used as a sub-package by LinearSolve.jl. Requires a Python environment
with `pyamg` and `scipy` installed.

Based on [cortner/PyAMG.jl](https://github.com/cortner/PyAMG.jl), modernised to use
PythonCall.jl instead of the legacy PyCall.jl.
"""
module LinearSolvePyAMG

export RugeStubenSolver, SmoothedAggregationSolver, AMGSolver,
    solve, aspreconditioner

using LinearAlgebra
using SparseArrays
using PythonCall

const pyamg = PythonCall.pynew()
const scipy_sparse = PythonCall.pynew()

function __init__()
    PythonCall.pycopy!(pyamg, pyimport("pyamg"))
    return PythonCall.pycopy!(scipy_sparse, pyimport("scipy.sparse"))
end

# ── Type hierarchy ────────────────────────────────────────────────────────────

struct RugeStuben end
struct SmoothedAggregation end

"""
    AMGSolver{T}

Wraps a Python pyamg solver object together with the original sparse matrix and any
default keyword arguments for the solve step.
"""
mutable struct AMGSolver{T}
    po::Py                      # Python AMG solver object
    id::T
    kwargs::NamedTuple
    A::SparseMatrixCSC
end

const RugeStubenSolver = AMGSolver{RugeStuben}
const SmoothedAggregationSolver = AMGSolver{SmoothedAggregation}

# ── Internal: Julia SparseMatrixCSC → scipy sparse CSR ───────────────────────

function py_csr(A::SparseMatrixCSC)
    # scipy expects 0-based integer index arrays
    data = A.nzval
    indices = A.rowval .- 1
    indptr = A.colptr .- 1
    m, n = size(A)
    csc = scipy_sparse.csc_matrix((data, indices, indptr), shape = (m, n))
    return csc.tocsr()
end

# ── Constructors ──────────────────────────────────────────────────────────────

"""
    RugeStubenSolver(A::SparseMatrixCSC; kwargs...)

Build a Ruge-Stüben AMG hierarchy for the sparse matrix `A`. Keyword arguments are
forwarded to `pyamg.ruge_stuben_solver`.
"""
function RugeStubenSolver(A::SparseMatrixCSC; kwargs...)
    po = pyamg.ruge_stuben_solver(py_csr(A); kwargs...)
    return AMGSolver(po, RugeStuben(), NamedTuple(kwargs), A)
end

"""
    SmoothedAggregationSolver(A::SparseMatrixCSC; kwargs...)

Build a smoothed-aggregation AMG hierarchy for the sparse matrix `A`. Keyword arguments
are forwarded to `pyamg.smoothed_aggregation_solver`.
"""
function SmoothedAggregationSolver(A::SparseMatrixCSC; kwargs...)
    po = pyamg.smoothed_aggregation_solver(py_csr(A); kwargs...)
    return AMGSolver(po, SmoothedAggregation(), NamedTuple(kwargs), A)
end

# ── Solve ─────────────────────────────────────────────────────────────────────

"""
    solve(amg::AMGSolver, b; tol, maxiter, accel, kwargs...)

Solve the linear system using the prebuilt AMG hierarchy. Keyword arguments override any
defaults stored in `amg.kwargs` and are forwarded to the Python solver.
"""
function solve(amg::AMGSolver, b::AbstractVector; kwargs...)
    x = amg.po.solve(b; amg.kwargs..., kwargs...)
    return pyconvert(Vector{Float64}, x)
end

# ── Preconditioner interface ──────────────────────────────────────────────────

struct AMGPreconditioner
    po::Py
    A::SparseMatrixCSC
end

"""
    aspreconditioner(amg::AMGSolver; kwargs...)

Return an `AMGPreconditioner` wrapping a single V-cycle of the AMG hierarchy, suitable
for use as a left preconditioner in iterative solvers.
"""
aspreconditioner(amg::AMGSolver; kwargs...) =
    AMGPreconditioner(amg.po.aspreconditioner(; kwargs...), amg.A)

Base.:(\)(M::AMGPreconditioner, b::AbstractVector) =
    pyconvert(Vector{Float64}, M.po.matvec(b))

Base.:(*)(M::AMGPreconditioner, x::AbstractVector) = M.A * x

LinearAlgebra.ldiv!(x, M::AMGPreconditioner, b) = copyto!(x, M \ b)
LinearAlgebra.mul!(b, M::AMGPreconditioner, x) = mul!(b, M.A, x)

# ── Convenience: `\` and `*` on AMGSolver itself ─────────────────────────────

Base.:(\)(amg::AMGSolver, b::AbstractVector) = solve(amg, b; amg.kwargs...)
Base.:(*)(amg::AMGSolver, x::AbstractVector) = amg.A * x

LinearAlgebra.ldiv!(x, amg::AMGSolver, b) = copyto!(x, amg \ b)

end
