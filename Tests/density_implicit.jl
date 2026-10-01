using Test, SparseArrays, LinearAlgebra
BLAS.set_num_threads(1)
include("../src/density_implicit.jl")
include("../src/density_projection.jl")
include("../src/density_timestepping.jl")

@testset "Implicit diffusion damps a stiff mode in one step" begin
    R = sparse(100.0 .* [-1 1; 1 -1])
    cache = DensityImplicitCache(R,1.0,1.0,ones(2); beta_in_operator=true)
    rhs = [0.8,0.2]
    result = similar(rhs)
    info = implicit_density_solve!(result,rhs,cache; rtol=1e-12)
    @test result ≈ [0.5+0.3/201,0.5-0.3/201] atol=1e-12
    @test info.relative_residual < 1e-12
    @test info.iterations <= 2
    @test sum(result) ≈ sum(rhs)
    @test maximum(abs.(rhs + R*rhs)) > 1 # A full explicit step is unstable here.
    @test density_substeps(R,1.0,1.0; beta_in_operator=true,update_mode=:implicit) == (1,200.0)
    @test density_substeps(R,1.0,1.0; beta_in_operator=true) == (96,200.0)

    # Reuse the same matrix/workspace for a different rhs; in-place solves work.
    saved_matrix = copy(cache.scaled_matrix)
    workspace = cache.workspace
    fill!(rhs,0.4)
    implicit_density_solve!(rhs,rhs,cache; rtol=1e-12)
    @test rhs ≈ fill(0.4,2) atol=1e-12
    @test cache.scaled_matrix == saved_matrix
    @test cache.workspace === workspace
    implicit_density_solve!(result,zeros(2),cache)
    @test iszero(result)

    cache32 = DensityImplicitCache(SparseMatrixCSC{Float64,Int32}(R),1.0,1.0,ones(2);
        beta_in_operator=true)
    implicit_density_solve!(result,[0.8,0.2],cache32; rtol=1e-12)
    @test result ≈ [0.5+0.3/201,0.5-0.3/201] atol=1e-12
end

@testset "Unequal cell volumes and a uniform multiplier response" begin
    R = sparse([-60.0 60.0; 20.0 -20.0])
    volumes = [1.0,3.0]
    cache = DensityImplicitCache(R,15.0,1.0,ones(2); beta_in_operator=true)
    result = zeros(2)
    rhs = [0.8,0.4]
    implicit_density_solve!(result,rhs,cache; rtol=1e-12)
    @test result ≈ [0.5+0.9/19,0.5-0.3/19] atol=1e-12
    @test dot(volumes,result) ≈ dot(volumes,rhs)

    # The one-solve multiplier identity needs a constant nullspace, not symmetry.
    A = Matrix(I-R/15)
    for lambda in (-2.3,0.0,1.7)
        @test A \ (rhs .- lambda/15) ≈ result .- lambda/15 atol=1e-12
    end
end

@testset "Split projection enforces bounds and physical volume" begin
    volumes = [1.0,2.0,3.0]
    K = [2.0 -2.0 0.0; -2.0 5.0 -3.0; 0.0 -3.0 3.0]
    R = sparse(-Diagonal(1 ./ volumes)*K)
    cache = DensityImplicitCache(R,1.0,1.0,ones(3); beta_in_operator=true)
    unbounded = [1.4,0.7,-0.4]
    rhs = (I-R)*unbounded
    result = zeros(3)
    implicit_density_solve!(result,rhs,cache; rtol=1e-12)
    @test result ≈ unbounded atol=1e-12
    project_density_volume!(result,copy(result),volumes,0.45,0.05)
    @test result ≈ [1.0,0.775,0.05] atol=1e-10
    @test dot(volumes,result)/sum(volumes) ≈ 0.45 atol=1e-10
end

@testset "Legacy beta scaling, cache freshness, and zero regularization" begin
    L = sparse([-2.0 2.0; 5.0 -5.0])
    rhs = [0.7,0.2]
    result = zeros(2)
    h = [0.25,1.0]
    cache = DensityImplicitCache(L,15.0,2.0,h)
    implicit_density_solve!(result,rhs,cache; rtol=1e-12)
    @test result ≈ (I-Diagonal(2 .* h.^2 .* 2 ./ 15)*L) \ rhs atol=1e-12
    @test validate_implicit_cache(cache,L,15.0,2.0,h) === cache
    @test_throws ArgumentError validate_implicit_cache(cache,copy(L),15.0,2.0,h)
    @test_throws ArgumentError validate_implicit_cache(cache,L,16.0,2.0,h)
    @test_throws ArgumentError validate_implicit_cache(cache,L,15.0,1.0,h)
    @test_throws ArgumentError validate_implicit_cache(cache,L,15.0,2.0,2h)
    @test_throws ArgumentError validate_implicit_cache(cache,L,15.0,2.0,h; beta_in_operator=true)

    # Rebuild after a same-size operator, beta, eta, or geometry change.
    for (newL,eta,beta,sizes) in ((2L,15.0,2.0,h),(L,30.0,2.0,h),
                                 (L,15.0,1.0,h),(L,15.0,2.0,2h))
        fresh = DensityImplicitCache(newL,eta,beta,sizes)
        implicit_density_solve!(result,rhs,fresh; rtol=1e-12)
        @test result ≈ (I-Diagonal(2 .* sizes.^2 .* beta ./ eta)*newL) \ rhs atol=1e-12
    end
    zero_beta = DensityImplicitCache(L,15.0,0.0,h)
    implicit_density_solve!(result,rhs,zero_beta; rtol=1e-12)
    @test result ≈ rhs atol=1e-12
end

@testset "Invalid input and nonconvergence do not change the result" begin
    R = sparse([-1.0 1.0; 1.0 -1.0])
    h = ones(2)
    @test_throws ArgumentError DensityImplicitCache(R,0.0,1.0,h)
    @test_throws ArgumentError DensityImplicitCache(R,1.0,-1.0,h)
    @test_throws ArgumentError DensityImplicitCache(R,1.0,1.0,[NaN,1.0])
    @test_throws DimensionMismatch DensityImplicitCache(R,1.0,1.0,ones(3))
    @test_throws ErrorException DensityImplicitCache(sparse([NaN 0.0; 0.0 1.0]),1.0,1.0,h)
    @test_throws ErrorException DensityImplicitCache(-R,0.5,1.0,h; beta_in_operator=true)
    @test_throws ArgumentError density_substeps(R,1.0,1.0; update_mode=:invalid)

    cache = DensityImplicitCache(R,1.0,1.0,h; beta_in_operator=true)
    result = [0.3,0.6]
    original = copy(result)
    @test_throws ErrorException implicit_density_solve!(result,[NaN,1.0],cache)
    @test result == original
    @test_throws ErrorException implicit_density_solve!(result,[0.8,0.2],cache; rtol=1e-12,itmax=1)
    @test result == original

    # Positive diagonal alone does not establish invertibility: incompatible rhs.
    singular = DensityImplicitCache(-R,2.0,1.0,h; beta_in_operator=true)
    @test_throws ErrorException implicit_density_solve!(result,[1.0,-1.0],singular)
    @test result == original
end
