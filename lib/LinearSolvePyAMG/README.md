# LinearSolvePyAMG.jl

A minimal Julia wrapper for the Python [PyAMG](https://pyamg.readthedocs.io) algebraic
multigrid library, bundled as a sub-package of
[LinearSolve.jl](https://github.com/SciML/LinearSolve.jl).

This package replaces the unmaintained [cortner/PyAMG.jl](https://github.com/cortner/PyAMG.jl)
(which was never migrated from the old METADATA.jl registry to the Julia General registry)
by embedding equivalent functionality directly in the LinearSolve.jl repository. It uses
[PythonCall.jl](https://github.com/JuliaPy/PythonCall.jl) instead of the legacy PyCall.jl.

## Requirements

- A Python environment with `pyamg` and `scipy` installed.
  - With [CondaPkg.jl](https://github.com/JuliaPy/CondaPkg.jl): add `pyamg` to your `CondaPkg.toml`.
  - With pip: `pip install pyamg scipy`.

## Usage via LinearSolve.jl

The recommended way to use this package is through LinearSolve.jl:

```julia
using LinearSolve, LinearSolvePyAMG, SparseArrays

A = sprand(500, 500, 0.02) + 20I
b = rand(500)

prob = LinearProblem(A, b)

# Ruge-Stüben AMG (default)
sol = solve(prob, PyAMGJL())

# Smoothed aggregation AMG
sol = solve(prob, PyAMGJL(:smoothed_aggregation))
```

## Standalone Usage

```julia
using LinearSolvePyAMG, SparseArrays

A = sprand(500, 500, 0.02) + 20I
b = rand(500)

# Build the AMG hierarchy once
amg = RugeStubenSolver(A)

# Solve (reuse hierarchy for multiple right-hand sides)
x = solve(amg, b; tol = 1e-6, maxiter = 100)

# Use as a preconditioner
M = aspreconditioner(amg)
# M \ b applies one V-cycle
```

## Solver Types

| Constructor | Python backend |
|---|---|
| `RugeStubenSolver(A; kwargs...)` | `pyamg.ruge_stuben_solver` |
| `SmoothedAggregationSolver(A; kwargs...)` | `pyamg.smoothed_aggregation_solver` |

Constructor keyword arguments are forwarded directly to the Python solver (e.g. `strength`,
`presmoother`, `postsmoother`, `max_levels`).

## Solve Keyword Arguments

`solve(amg, b; kwargs...)` forwards all keyword arguments to the Python `.solve()` method:

| Keyword | Description |
|---|---|
| `tol` | Relative residual tolerance (default: `1e-5`) |
| `maxiter` | Maximum number of iterations |
| `accel` | Acceleration method, e.g. `"cg"` or `"gmres"` |
| `cycle` | Cycle type: `"V"` (default), `"W"`, `"F"`, `"AMLI"` |
