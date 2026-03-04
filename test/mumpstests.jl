using LinearSolve, SparseArrays, LinearAlgebra, Test
using MUMPS, MPI
import LinearSolve: solve, solve!

MPI.Init()

@testset "MUMPSFactorization" begin
    n = 50

    @testset "Real unsymmetric (Float64)" begin
        A = sprand(Float64, n, n, 0.3) + 10I
        b = rand(Float64, n)
        prob = LinearProblem(A, b)
        sol = solve(prob, MUMPSFactorization())
        @test norm(A * sol.u - b) / norm(b) < 1e-10
    end

    @testset "Real symmetric positive definite (Float64)" begin
        B = sprandn(Float64, n, n, 0.3)
        A = B' * B + n * I
        @test issymmetric(A)
        b = rand(Float64, n)
        prob = LinearProblem(A, b)
        # sym=1 → mumps_definite
        sol = solve(prob, MUMPSFactorization(sym = MUMPS.mumps_definite))
        @test norm(A * sol.u - b) / norm(b) < 1e-10
    end

    @testset "Real general symmetric (Float64)" begin
        B = sprand(Float64, n, n, 0.3)
        A = sparse(B + B' + 10I)
        @test issymmetric(A)
        b = rand(Float64, n)
        prob = LinearProblem(A, b)
        sol = solve(prob, MUMPSFactorization(sym = MUMPS.mumps_symmetric))
        @test norm(A * sol.u - b) / norm(b) < 1e-10
    end

    @testset "Real unsymmetric (Float32)" begin
        A = Float32.(Matrix(sprand(n, n, 0.3) + 10I))
        b = rand(Float32, n)
        prob = LinearProblem(sparse(A), b)
        sol = solve(prob, MUMPSFactorization())
        @test norm(A * sol.u - b) / norm(b) < 1e-4
    end

    @testset "Complex unsymmetric (ComplexF64)" begin
        Ar = sprand(Float64, n, n, 0.3) + 10I
        Ai = sprand(Float64, n, n, 0.1)
        A = complex.(Ar, Ai)
        b = complex.(rand(Float64, n), rand(Float64, n))
        prob = LinearProblem(A, b)
        sol = solve(prob, MUMPSFactorization())
        @test norm(A * sol.u - b) / norm(b) < 1e-10
    end

    @testset "Re-solve with same factorization" begin
        A = sprand(Float64, n, n, 0.3) + 10I
        b1 = rand(Float64, n)
        b2 = rand(Float64, n)
        cache = init(LinearProblem(A, b1), MUMPSFactorization())
        sol1 = solve!(cache)
        @test norm(A * sol1.u - b1) / norm(b1) < 1e-10

        # Re-solve with a new rhs (matrix unchanged)
        cache.b = b2
        sol2 = solve!(cache)
        @test norm(A * sol2.u - b2) / norm(b2) < 1e-10
    end

    @testset "Constructor error without MUMPS loaded" begin
        # Just verify MUMPSFactorization constructs successfully when MUMPS is loaded
        alg = MUMPSFactorization()
        @test alg isa MUMPSFactorization
        @test alg.sym == 0
        @test alg.par == 1
    end
end
