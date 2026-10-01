using Test, LinearAlgebra, SparseArrays, TimerOutputs, HYPRE
BLAS.set_num_threads(1)
HYPRE.Init()
const to = TimerOutput()
include("../src/utils/amg_utils.jl")

@testset "HYPRE reports setup and solve diagnostics for every call" begin
    # Three interleaved displacement components, matching NumFunctions=3.
    nodes = 20
    K = kron(spdiagm(-1 => -ones(nodes-1), 0 => 3ones(nodes), 1 => -ones(nodes-1)),
             sparse(Matrix{Float64}(I, 3, 3)))
    expected = [sin.(collect(1:size(K, 1))), cos.(collect(1:size(K, 1)))]
    solutions = Vector{Vector{Float64}}()
    log = mktemp() do _, io
        redirect_stdout(io) do
            for x in expected
                push!(solutions, solve_lse_hypre(K, K*x))
            end
        end
        seekstart(io)
        read(io, String)
    end

    for (u, x) in zip(solutions, expected)
        @test norm(K*u - K*x) / norm(K*x) <= 1e-3
        @test u ≈ x rtol=1e-3
    end
    lines = filter(line -> startswith(line, "[hypre]"), split(log, '\n'))
    @test length(lines) == length(expected)
    for line in lines
        @test occursin("PCG+BoomerAMG", line)
        its = match(r"its=(\d+)", line)
        @test its !== nothing && parse(Int, its[1]) > 0
        for label in ("relres", "setup(PCG+AMG)", "solve", "total")
            value = split(split(line, label * "="; limit=2)[2])[1]
            number = parse(Float64, rstrip(value, 's'))
            @test isfinite(number) && number >= 0
        end
    end
    # Setup and the Krylov iterations also remain in the aggregate timer.
    timer_text = sprint(show, to)
    @test occursin("hypre_setup", timer_text)
    @test occursin("hypre_solve", timer_text)
end
