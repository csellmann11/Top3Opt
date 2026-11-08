"""
Correctness of the volume-preserving density projection and its adaptive slope.

The finite-difference references resolve the volume-preserving threshold anew
after every input perturbation. This checks its implicit derivative, not just
the derivative of a Heaviside function with a frozen threshold.

Standalone: julia --project=. "Tests/regularization/projection.jl"
"""

using Test, LinearAlgebra
isdefined(@__MODULE__, :DensityProjection) || include(joinpath(@__DIR__,"..","..","src","regularization","projection.jl"))

function _projection_test_state(V; rho_min=0.0, beta=6.0)
    options = DensityProjectionOptions(enabled=true, beta_initial=beta)
    return DensityProjection(options, V; rho_min=rho_min)
end

# Independent scalar reference, used at moderate beta where the standard
# tanh expression does not suffer subtraction underflow.
function _reference_projection(x, V, beta; rho_min=0.0)
    s = (x .- rho_min) ./ (1 - rho_min)
    target = dot(V, s)
    h(z, eta) = (tanh(beta * eta) + tanh(beta * (z - eta))) /
                (tanh(beta * eta) + tanh(beta * (1 - eta)))
    lo, hi = 0.0, 1.0
    for _ in 1:100
        eta = (lo + hi) / 2
        mass = sum(V[i] * h(s[i], eta) for i in eachindex(s))
        if mass > target
            lo = eta
        else
            hi = eta
        end
    end
    eta = (lo + hi) / 2
    return rho_min .+ (1 - rho_min) .* h.(s, eta)
end

function _projection_fd_jacobian(x, V, beta; rho_min=0.0, epsilon=2e-6)
    J = zeros(length(x), length(x))
    for j in eachindex(x)
        xp, xm = copy(x), copy(x)
        xp[j] += epsilon
        xm[j] -= epsilon
        J[:, j] = (_reference_projection(xp, V, beta; rho_min=rho_min) .-
                   _reference_projection(xm, V, beta; rho_min=rho_min)) ./ (2epsilon)
    end
    return J
end

@testset "density projection — identity and storage" begin
    x = [0.0, 0.13, 0.43, 0.91, 1.0]
    out = similar(x)
    projection_apply!(out, nothing, x)
    @test out == x
    projection_apply!(x, nothing, x)
    @test x == out
    gradient = [1.0, -3.0, 0.0, 0.2, 7.0]
    expected = copy(gradient)
    projection_pullback!(gradient, nothing)
    @test gradient == expected
    @test !projection_advance!(nothing, 1.0)

    V = [0.7, 1.3, 2.1, 0.4, 1.5]
    p = _projection_test_state(V)
    derivative, threshold_derivative, weights = p.derivative, p.threshold_derivative, p.weights
    for shift in (0.0, 0.02, -0.01)
        xx = clamp.(x .+ shift, 0.0, 1.0)
        projection_apply!(out, p, xx)
        @test p.derivative === derivative
        @test p.threshold_derivative === threshold_derivative
        @test p.weights === weights
        @test dot(V, out) ≈ dot(V, xx) atol=2e-11 rtol=0.0
    end
end

@testset "density projection — mass and implicit-threshold chain rule" begin
    V = [0.4, 1.8, 0.7, 2.3, 1.1, 0.9]
    s = [0.05, 0.19, 0.37, 0.54, 0.72, 0.96]
    g = [1.7, -0.4, 0.9, -2.1, 0.3, 1.2]
    for floor in (0.0, 0.15), beta in (1.0, 6.0, 24.0)
        x = floor .+ (1 - floor) .* s
        p = _projection_test_state(V; rho_min=floor, beta=beta)
        out = similar(x)
        projection_apply!(out, p, x)
        reference = _reference_projection(x, V, beta; rho_min=floor)
        @test out ≈ reference atol=2e-11 rtol=2e-11
        @test dot(V, out) ≈ dot(V, x) atol=2e-11 rtol=0.0
        @test all(floor .<= out .<= 1.0)
        @test maximum(abs, out .- x) > 1e-3

        J = _projection_fd_jacobian(x, V, beta; rho_min=floor)
        pulled = copy(g)
        projection_pullback!(pulled, p)
        @test pulled ≈ (transpose(J) * (V .* g)) ./ V atol=2e-7 rtol=2e-7
        # A fitted threshold couples different cells. A diagonal-only
        # Heaviside derivative must not accidentally pass this test.
        @test norm(J - Diagonal(diag(J))) > 1e-2
        @test norm(pulled - p.derivative .* g) > 1e-2

        # In the area metric, physical mass has the constant gradient one.
        mass_gradient = ones(length(V))
        projection_pullback!(mass_gradient, p)
        @test mass_gradient ≈ ones(length(V)) atol=2e-11 rtol=2e-11
        normalized = (out .- floor) ./ (1 - floor)
        expected_grayness = 4dot(V, normalized .* (1 .- normalized)) / sum(V)
        @test p.grayness ≈ expected_grayness atol=2e-12 rtol=2e-12
    end
end

@testset "density projection — endpoints and uniform fields" begin
    V = [0.5, 2.0, 1.0, 3.0]
    for floor in (0.0, 0.1), beta in (1.0, 20.0)
        p = _projection_test_state(V; rho_min=floor, beta=beta)
        for level in (floor, floor + 0.37 * (1 - floor), 1.0)
            x = fill(level, length(V))
            out = similar(x)
            projection_apply!(out, p, x)
            @test out ≈ x atol=2e-11 rtol=2e-11
            gradient = ones(length(V))
            projection_pullback!(gradient, p)
            @test gradient ≈ ones(length(V)) atol=2e-11 rtol=2e-11
        end
        binary = [floor, 1.0, 1.0, floor]
        out = similar(binary)
        projection_apply!(out, p, binary)
        @test out ≈ binary atol=2e-14 rtol=0.0
        @test p.grayness ≈ 0.0 atol=2e-14
    end
end

@testset "density projection — steep slopes and aliasing" begin
    # Every literal P_eta underflows at beta=1000, but its normalized ratio
    # has a finite limit. In this symmetric case the mass component of any
    # input perturbation remains and the contrast component is flattened.
    p = _projection_test_state([1.0, 1.0]; beta=1000.0)
    x = [0.001, 0.999]
    projection_apply!(x, p, x)
    @test x ≈ [0.0, 1.0] atol=1e-12
    g = [2.0, 4.0]
    projection_pullback!(g, p)
    @test g ≈ [3.0, 3.0] atol=1e-10
    @test all(isfinite, p.threshold_derivative)
    @test any(!iszero, p.threshold_derivative)
    V = [0.4, 1.8, 0.7, 2.3, 1.1, 0.9]
    for beta in (128.0, 512.0, 1000.0)
        p = _projection_test_state(V; beta)
        x = [0.05, 0.19, 0.37, 0.54, 0.72, 0.96]
        target = dot(V, x)
        projection_apply!(x, p, x)
        @test dot(V, x) ≈ target atol=2e-11
        g = ones(length(V))
        projection_pullback!(g, p)
        @test g ≈ ones(length(V)) atol=1e-10
    end
end

@testset "density projection — settings validation" begin
    @test DensityProjection(DensityProjectionOptions(), [1.0]) === nothing
    for options in (DensityProjectionOptions(beta_initial=0.0),
                    DensityProjectionOptions(growth_factor=NaN),
                    DensityProjectionOptions(max_growth_rate=-1.0),
                    DensityProjectionOptions(gray_tolerance=0.0),
                    DensityProjectionOptions(volume_tolerance=Inf),
                    DensityProjectionOptions(max_threshold_iterations=0))
        @test_throws ArgumentError validate_density_projection_options(options)
    end
    @test_throws ArgumentError _projection_test_state([1.0, 0.0])
    @test_throws ArgumentError _projection_test_state([1.0]; rho_min=1.0)
    p = _projection_test_state([1.0, 1.0])
    @test_throws DimensionMismatch projection_apply!(zeros(1), p, [0.2, 0.5])
    @test_throws ArgumentError projection_apply!(zeros(2), p, [0.2, NaN])
    @test_throws ArgumentError projection_apply!(zeros(2), p, [0.2, 1.1])
end

@testset "density projection — adaptive continuation" begin
    function fresh()
        # Fix the policy parameters for these analytic expectations. Production
        # defaults may be tuned without changing the continuation equation.
        options = DensityProjectionOptions(enabled=true,beta_initial=1.0,
            growth_factor=5e-4,max_growth_rate=0.3)
        p = DensityProjection(options,ones(4))
        projection_apply!(zeros(4), p, [0.15, 0.35, 0.6, 0.9])
        @test p.grayness > 0.01
        return p
    end

    p = fresh()
    @test !projection_advance!(p, 100.0)
    @test p.beta == 1.0
    @test projection_advance!(p, 99.0)
    @test p.beta ≈ 1.0 + 5e-4 * 99.5

    p = fresh()
    @test !projection_advance!(p, -100.0)
    @test projection_advance!(p, -101.0)
    @test p.beta ≈ 1.0 + 5e-4 * 100.5

    p = fresh()
    projection_advance!(p, 100.0)
    @test !projection_advance!(p, 101.0)
    @test p.beta == 1.0
    @test !projection_advance!(p, NaN)
    @test !projection_advance!(p, Inf)
    @test p.beta == 1.0

    p = fresh()
    projection_advance!(p, 100.0)
    @test projection_advance!(p, 100.0 - 1e-8)
    @test p.beta ≈ 1.3

    p = fresh()
    projection_advance!(p, 1.0)
    @test !projection_advance!(p, -5.0; reset=true)
    @test p.beta == 1.0
    @test !projection_advance!(p, -5.1)
    @test projection_advance!(p, -5.2)
    @test p.beta ≈ 1.0 + 5e-4 * 5.15 / 0.1

    p = fresh()
    projection_advance!(p, 2.0)
    @test projection_advance!(p, 2.0)
    @test p.beta ≈ 1.3
    @test p.beta_history ≈ [1.0, 1.0]
    @test length(p.gray_history) == 2

    p = fresh()
    projection_advance!(p, 0.0)
    @test !projection_advance!(p, 0.0)
    @test p.beta == 1.0

    p = fresh()
    projection_advance!(p, 1.0)
    @test !projection_advance!(p, 1.0; allow_advance=false)
    @test p.beta == 1.0

    p = fresh()
    projection_advance!(p, 1.0)
    projection_apply!(zeros(4), p, [0.0, 0.0, 1.0, 1.0])
    @test !projection_advance!(p, 0.99999)
    @test p.beta == 1.0
end

@testset "Adaptive cell weights and roundoff-only bounds correction" begin
    area = [0.5, 1.0, 1.5]
    p = _projection_test_state(area; rho_min=0.001)
    projection_apply!(zeros(3), p, [0.05, 0.4, 0.9])
    projection_advance!(p, 2.0)
    projection_advance!(p, 1.99)
    beta, previous, history = p.beta, p.previous_objective, copy(p.beta_history)
    weights, derivative, threshold = p.weights, p.derivative, p.threshold_derivative
    @test projection_rebuild!(nothing, [1.0]) === nothing
    @test projection_converged(nothing)
    for next_area in ([0.1, 0.2, 0.3, 0.4], [0.4, 0.6], [0.8, 0.2])
        @test projection_rebuild!(p, next_area) === p
        @test p.weights === weights && p.derivative === derivative && p.threshold_derivative === threshold
        @test p.weights ≈ next_area ./ sum(next_area)
        @test p.beta == beta && p.previous_objective == previous && p.beta_history == history
        @test_throws ErrorException projection_pullback!(ones(length(next_area)), p)
        x = collect(range(0.2, 0.8; length=length(next_area)))
        out = similar(x)
        projection_apply!(out, p, x)
        @test dot(next_area, out) ≈ dot(next_area, x) atol=2e-12
        g = ones(length(x))
        projection_pullback!(g, p)
        @test g ≈ ones(length(x)) atol=1e-12
    end
    @test_throws ArgumentError projection_rebuild!(p, Float64[])
    @test_throws ArgumentError projection_rebuild!(p, [1.0, 0.0])
    @test_throws ArgumentError projection_rebuild!(p, [Inf, 1.0])

    p = _projection_test_state([1.0, 2.0, 3.0, 4.0]; rho_min=0.001)
    x = [prevfloat(0.001), 0.2, 0.9, nextfloat(1.0)]
    expected = clamp.(x, 0.001, 1.0)
    projection_apply!(x, p, x)
    @test minimum(x) >= 0.001 && maximum(x) <= 1.0
    @test dot(p.weights, x) ≈ dot(p.weights, expected) atol=2e-12
    @test_throws ArgumentError projection_apply!(x, p, [0.001-1e-8, 0.2, 0.9, 1.0])
    @test_throws ArgumentError projection_apply!(x, p, [0.001, 0.2, 0.9, 1+1e-8])

    binary = [prevfloat(0.001), 1.0, nextfloat(1.0), 0.001]
    projection_apply!(x, p, binary)
    @test x == [0.001, 1.0, 1.0, 0.001]
    @test p.grayness == 0.0
end

function projection_allocation(n)
    area = ones(n)
    p = _projection_test_state(area)
    x = collect(range(0.05, 0.95; length=n))
    out = similar(x)
    gradient = ones(n)
    projection_apply!(out, p, x)
    projection_pullback!(gradient, p)
    projection_apply!(out, p, x)
    projection_pullback!(gradient, p)
    return @allocated begin
        projection_apply!(out, p, x)
        projection_pullback!(gradient, p)
    end
end

@testset "Projection workspace reuse" begin
    small = projection_allocation(20)
    large = projection_allocation(20_000)
    @test large <= small + 64
    @test large < 1024
end
