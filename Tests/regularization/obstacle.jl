# Double-obstacle term of the gradient penalty, its imbalance measure and continuation,
# and the imbalance of the Helmholtz design step. Uses kernel_faces() from kernels.jl.

"""Uniform chain of cubes with edge h: volume h^3, face area h^2, beta = h^2 (beta0 = 1/2).
Every face has t/|K| = 1 and kappa = beta/h^2 = 1, so g is the normalized gamma itself."""
cube_chain(n;h=1.0) = TPFAFaces(fill(h^3,n),collect(1:n-1),collect(2:n),fill(h^2,n-1),
    fill(h/2,n-1),fill(h/2,n-1))

"""A planar mesh-aligned interface between a solid and a void layer, relaxed under the
uniform force (absorbed by the volume multiplier) for `steps` outer iterations."""
function relax_interface(g,integrator;steps=4000,lower=0.001,h=1.0)
    faces = cube_chain(8;h)
    x = [1.0,1.0,1.0,0.8,1+lower-0.8,lower,lower,lower]
    rho = dot(faces.area,x)/sum(faces.area)
    cache = DensityCache(faces,h^2)
    cache.obstacle = g
    options = DensityOptions(;integrator,scaling=:mean)
    for _ in 1:steps
        advance_density!(x,zeros(8),p->fill!(p,1.0),cache;options,eta0=5.0,rho,lower)
    end
    return x,rho,faces
end

@testset "Zero obstacle weight and the balance measurement leave the update unchanged" begin
    faces = kernel_faces()
    initial = [0.2,0.4,0.6,0.3,0.5]
    rho = dot(faces.area,initial)/sum(faces.area)
    weights = [1.0,20.0,3.0,2.0,1.0]
    for scaling in GRADIENT_SCALINGS, integrator in (:explicit,:semi_implicit), diag in (false,true),
            move_limit in (Inf,0.05)
        allowed_update(scaling,integrator) || continue
        results = map(((nothing,false),(0.0,false),(0.0,true))) do (g,measure_balance)
            x, cache = copy(initial), DensityCache(faces,0.02)
            g === nothing || (cache.obstacle = g)
            options = DensityOptions(;scaling,integrator,diag,move_limit,measure_balance,substeps=2)
            for _ in 1:4
                advance_density!(x,zeros(5),p->(@. p = 1+weights*x^2),cache;options,eta0=4.0,rho,lower=0.001)
            end
            x, cache.balance
        end
        @test results[1][1] == results[2][1] == results[3][1]
        @test isnan(results[1][2].imbalance) && isfinite(results[3][2].imbalance)
    end
end

@testset "Planar interface: gray below g = 2, binary above" begin
    lower = 0.001
    center, half = (1+lower)/2, (1-lower)/2
    # Stationary interface cell between a bulk cell and its mirror image:
    # (half - 3u) + g*u = 0 for the deviation u from the center.
    for h in (1.0,0.1)
        x,rho,faces = relax_interface(1.5,:explicit;h)
        @test x[4] ≈ center+half/(3-1.5) atol=1e-8
        @test x[5] ≈ center-half/(3-1.5) atol=1e-8
        @test x[1:3] == ones(3) && x[6:8] == fill(lower,3)
        @test dot(faces.area,x)/sum(faces.area) ≈ rho atol=1e-9
        x,rho,faces = relax_interface(2.5,:explicit;h)
        @test x == [1.0,1.0,1.0,1.0,lower,lower,lower,lower]
        @test dot(faces.area,x)/sum(faces.area) ≈ rho atol=1e-9
    end
    # The semi-implicit split shifts the stationary value, not the two regimes.
    x,_ = relax_interface(1.0,:semi_implicit)
    @test lower+0.05 < x[5] < x[4] < 0.95
    x,_ = relax_interface(4.0,:semi_implicit)
    @test x == [1.0,1.0,1.0,1.0,lower,lower,lower,lower]
end

@testset "Obstacle continuation: imbalance budget, gray stop and cap" begin
    @test_throws ArgumentError ObstacleContinuation(max_weight=0.0)
    @test_throws ArgumentError ObstacleContinuation(max_weight=NaN)
    @test_throws ArgumentError ObstacleContinuation(imbalance_tolerance=0.0)
    @test_throws ArgumentError ObstacleContinuation(gray_tolerance=1.0)
    balance(e;F=1.0,D=0.5,U=0.25) = (imbalance=e,force=F,diffusion=D,obstacle_unit=U,free_fraction=0.1)
    o = ObstacleContinuation()
    @test (o.weight,o.max_weight,o.imbalance_tolerance,o.gray_tolerance) == (0.0,Inf,0.1,0.01)
    @test !obstacle_advance!(o,balance(0.1),0.5)        # at the tolerance: no budget
    @test !obstacle_advance!(o,balance(0.3),0.5) && o.weight == 0
    @test obstacle_advance!(o,balance(0.0),0.5)
    @test o.weight ≈ 0.1*(1.0+0.5)/0.25                  # the full budget from g = 0
    g = o.weight
    @test obstacle_advance!(o,balance(0.04),0.5)
    @test o.weight ≈ g+0.06*(1.0+0.5+g*0.25)/0.25
    g = o.weight
    @test !obstacle_advance!(o,balance(0.0;U=0.0),0.5)   # no free cells
    @test !obstacle_advance!(o,balance(NaN),0.5) && o.weight == g
    @test !obstacle_advance!(o,balance(0.0),0.5;allow_advance=false) && o.weight == g
    @test !obstacle_converged(o)
    @test !obstacle_advance!(o,balance(0.0),0.005) && obstacle_converged(o) && o.weight == g
    capped = ObstacleContinuation(max_weight=0.2)
    @test obstacle_advance!(capped,balance(0.0),0.5) && capped.weight == 0.2
    @test obstacle_converged(capped) && obstacle_converged(nothing)
    @test !obstacle_advance!(capped,balance(0.0),0.5)
end

@testset "Explicit imbalance: the net term is F + D + g*U on the free cells" begin
    faces = cube_chain(8;h=0.5)
    lower = 0.001
    start = [1.0,0.9,0.75,0.6,0.45,0.3,0.1,lower]
    rho = dot(faces.area,start)/sum(faces.area)
    beta = 2*0.25                                     # 2*beta0*h^2 with beta0 = 1: kappa = 2
    norm2(v,free) = sqrt(sum(faces.area[free].*v[free].^2))
    for scaling in GRADIENT_SCALINGS, diag in (false,true)
        x = copy(start)
        cache = DensityCache(faces,beta)
        cache.obstacle = 0.7
        force(p) = (@. p = (2-x)*(9-(1:8))) # falls with x: diag measures a curvature
        options = DensityOptions(;scaling,diag,substeps=1,measure_balance=true,diagnostics=true)
        # With diag, the first call only stores the secant entry.
        diag && advance_density!(x,zeros(8),force,cache;options,eta0=20.0,rho,lower)
        old = copy(x)
        p = force(zeros(8))
        result = advance_density!(x,zeros(8),force,cache;options,eta0=20.0,rho,lower)
        row = only(result.records)
        b = cache.balance
        @test diag == cache.damping_active
        @test 0 <= b.imbalance <= 1+1e-12 && row.imbalance == b.imbalance
        @test b.obstacle_unit > 0 && 0 < b.free_fraction < 1
        # Recompute N, F, D and U of the single explicit substep on its free cells.
        s = row.p_avg
        l = row.lambda/s
        boost = cache.beta_relative
        F = @. (p/s-l)/boost
        D = -(cache.stiffness*old)./faces.area
        U = @. beta/0.25*(old-(1+lower)/2)
        r = diag ? cache.damping./(s*20.0*boost) : zeros(8)
        N = @. (x-old)*(1+r)*20.0
        free = @. lower < x < 1
        @test count(free) >= 3
        @test N[free] ≈ (F+D+0.7U)[free] rtol=1e-9
        @test b.imbalance ≈ norm2(N,free)/(norm2(F,free)+norm2(D,free)+0.7norm2(U,free)) rtol=1e-10
        @test b.obstacle_unit ≈ norm2(U,free) rtol=1e-12
    end
    # Without the measurement option the field keeps its initial value.
    other = DensityCache(faces,beta)
    advance_density!(copy(start),zeros(8),p->(@. p = 1+(1:8)),other;options=DensityOptions(),eta0=20.0,rho,lower)
    @test isnan(other.balance.imbalance)
end

@testset "Filter imbalance: |F - l| over |F| + |l|*sqrt(V_free)" begin
    faces = kernel_faces()
    initial = [0.2,0.4,0.6,0.3,0.5]
    lower = 0.001
    rho = dot(faces.area,initial)/sum(faces.area)
    for enabled in (false,true), limit in (Inf,0.05)
        cache = HelmholtzCache(faces,ones(5),0.2)
        design,physical = copy(initial),similar(initial)
        projection = DensityProjection(DensityProjectionOptions(enabled=enabled),faces.area;rho_min=lower)
        map_filtered_design!(physical,design,cache,projection)
        options = DensityOptions(regularization=:helmholtz,move_limit=limit,measure_balance=true,diagnostics=true)
        result = advance_filtered_design!(design,physical,zeros(5),p->(@. p = 1+[1.0,20.0,3.0,2.0,1.0]*physical^2),
            cache;options,eta0=15.0,rho,lower,density_projection=projection)
        row = only(result.records)
        b = cache.balance
        s = row.p_avg
        l = row.lambda/s
        F = cache.q./s
        N = (design.-initial).*15.0
        box = isfinite(limit) ? move_box!(cache,initial,lower,options,1.0) : nothing
        free = [lower < design[i] < 1 && (box === nothing || box.lower[i] < design[i] < box.upper[i]) for i in 1:5]
        @test count(free) >= 2
        @test N[free] ≈ (F.-l)[free] rtol=1e-9
        A = faces.area[free]
        @test b.imbalance ≈ sqrt(sum(A.*N[free].^2))/(sqrt(sum(A.*F[free].^2))+abs(l)*sqrt(sum(A))) rtol=1e-10
        @test row.imbalance == b.imbalance && b.free_fraction ≈ sum(A)/sum(faces.area)
        @test b.diffusion == 0 && b.obstacle_unit == 0
    end
    # A uniform design force is balanced by the multiplier alone.
    cache = HelmholtzCache(faces,ones(5),0.2)
    design,physical = copy(initial),similar(initial)
    map_filtered_design!(physical,design,cache)
    advance_filtered_design!(design,physical,zeros(5),p->fill!(p,2.0),cache;
        options=DensityOptions(regularization=:helmholtz,measure_balance=true),eta0=15.0,rho,lower)
    @test cache.balance.imbalance < 1e-10
end
