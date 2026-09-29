using Test, Random, LinearAlgebra
include("../src/density_projection.jl")

@testset "Complete-force density volume projection" begin
    rng = MersenneTwister(382)
    total_iterations = Int[]
    for n in (2,31,1000), scale in (0.01,1.,1000.), target in (0.001,0.03,0.3,0.98,1.)
        volumes = exp.(4randn(rng,n))
        trial = 7 .+ scale*randn(rng,n) # deliberately outside the old force bracket
        density = similar(trial)
        iterations = project_density_volume!(density,trial,volumes,target,0.001)
        push!(total_iterations,iterations)
        @test all(x->0.001 <= x <= 1.,density)
        @test abs(dot(volumes,density)/sum(volumes)-target) <= 1e-8
        # Independent bisection reference of the constrained explicit update.
        lo,hi = minimum(trial)-1,maximum(trial)-0.001
        for _ in 1:100
            shift = (lo+hi)/2
            mean = dot(volumes,clamp.(trial .- shift,0.001,1.))/sum(volumes)
            mean > target ? (lo=shift) : (hi=shift)
        end
        @test maximum(abs.(density-clamp.(trial .- (lo+hi)/2,0.001,1.))) < 1e-5
    end
    @test project_density_volume!(zeros(3),fill(0.3,3),ones(3),0.3,0.001) == 1
    @test_throws ArgumentError project_density_volume!(zeros(2),ones(2),ones(2),0.,0.001)
    @test_throws ErrorException project_density_volume!(zeros(2),[NaN,1.],ones(2),0.3,0.001)
    @info "Projection iterations" maximum(total_iterations)
end
