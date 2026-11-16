# Modular gradient-penalty update: scaling x integrator x diag, the move limit,
# and the removed option names. Uses kernel_faces() from kernels.jl.
const GRADIENT_SCALINGS = (:mean,:local_max,:local_lambda)
allowed_update(scaling,integrator) = !(scaling == :local_lambda && integrator == :semi_implicit)

"""Force of a quadratic objective per cell, p = f0 - k*(x - xstar) (positive = grow)."""
quadratic_force(f0,k,xstar) = (p,x) -> (@. p = f0-k*(x-xstar))

"""Move-limit box of the specification, built independently of `move_box!`."""
function reference_box(old,lower,limit,p)
    lo = [min(max(lower,max(x^p-limit,0.0)^(1/p)),x) for x in old]
    hi = [max(min(1.0,(x^p+limit)^(1/p)),x) for x in old]
    return (lower=lo,upper=hi)
end

"""Cells on a box bound strictly tighter than [lower, 1]."""
recount_move_limited(x,box,lower) = count(i -> (box.lower[i] > lower && x[i] == box.lower[i]) ||
    (box.upper[i] < 1 && x[i] == box.upper[i]),eachindex(x))

@testset "Removed names and forbidden combinations are rejected" begin
    faces = kernel_faces()
    x = [0.2,0.4,0.6,0.3,0.5]
    rho = dot(faces.area,x)/sum(faces.area)
    run(options) = advance_density!(copy(x),ones(5),identity,DensityCache(faces,0.02);
        options,eta0=15.0,rho,lower=0.001)
    for old in (:mean,:local_max,:local_lambda)
        @test_throws ArgumentError run(DensityOptions(regularization=old))
        @test_throws ArgumentError validate_density_options(DensityOptions(regularization=old),15.0)
    end
    for options in (DensityOptions(integrator=:implicit),DensityOptions(scaling=:helmholtz),
            DensityOptions(scaling=:local_lambda,integrator=:semi_implicit),
            DensityOptions(scaling=:local_lambda,integrator=:semi_implicit,diag=true),
            DensityOptions(regularization=:helmholtz,diag=true),
            DensityOptions(regularization=:helmholtz,scaling=:local_max),
            DensityOptions(regularization=:helmholtz,integrator=:semi_implicit),
            DensityOptions(move_limit=0.0),DensityOptions(move_limit=-1.0),
            DensityOptions(move_limit=NaN),DensityOptions(simp_exponent=0.0))
        @test_throws ArgumentError validate_density_options(options,15.0)
    end
    @test_throws ArgumentError run(DensityOptions(scaling=:local_lambda,integrator=:semi_implicit))
    @test_throws ArgumentError run(DensityOptions(regularization=:helmholtz))
    # Defaults are the former default (:mean, explicit, no damping, no move limit).
    defaults = DensityOptions()
    @test (defaults.regularization,defaults.scaling,defaults.integrator,defaults.diag,defaults.move_limit) ==
        (:gradient,:mean,:explicit,false,Inf)
end

@testset "Every allowed scaling x integrator x diag conserves volume" begin
    faces = kernel_faces()
    initial = [0.2,0.4,0.6,0.3,0.5]
    lower = 0.001
    rho = dot(faces.area,initial)/sum(faces.area)
    refresh! = quadratic_force([2.0,9.0,3.0,1.0,4.0],[3.0,40.0,8.0,2.0,6.0],[0.3,0.5,0.4,0.6,0.2])
    for scaling in GRADIENT_SCALINGS, integrator in (:explicit,:semi_implicit), diag in (false,true),
            enabled in (false,true)
        allowed_update(scaling,integrator) || continue
        cache = DensityCache(faces,0.02)
        design,physical = copy(initial),copy(initial)
        projection = DensityProjection(DensityProjectionOptions(enabled=enabled,beta_initial=4.0),
            faces.area;rho_min=lower)
        projection_apply!(physical,projection,design)
        options = DensityOptions(;scaling,integrator,diag,substeps=2)
        force = zeros(5)
        damped = false
        for _ in 1:6
            advance_density!(design,force,p -> refresh!(p,physical),cache;options,eta0=4.0,rho,lower,
                physical,density_projection=projection)
            damped |= cache.damping_active
            @test dot(faces.area,design)/sum(faces.area) ≈ rho atol=2e-10
            @test dot(faces.area,physical)/sum(faces.area) ≈ rho atol=2e-10
            @test all(lower-1e-12 .<= design .<= 1+1e-12)
        end
        @test damped == diag
        @test all(isfinite,design)
    end
end

@testset "diag without history is bitwise the undamped update" begin
    faces = kernel_faces()
    initial = [0.2,0.4,0.6,0.3,0.5]
    rho = dot(faces.area,initial)/sum(faces.area)
    weights = [1.0,20.0,3.0,2.0,1.0]
    for scaling in GRADIENT_SCALINGS, integrator in (:explicit,:semi_implicit), enabled in (false,true)
        allowed_update(scaling,integrator) || continue
        results = map((false,true)) do diag
            cache = DensityCache(faces,0.02)
            design,physical = copy(initial),copy(initial)
            projection = DensityProjection(DensityProjectionOptions(enabled=enabled),faces.area;rho_min=0.001)
            projection_apply!(physical,projection,design)
            options = DensityOptions(;scaling,integrator,diag,substeps=3)
            advance_density!(design,zeros(5),p -> (@. p = 1+weights*physical^2),cache;options,
                eta0=15.0,rho,lower=0.001,physical,density_projection=projection)
            (;design,physical,cache)
        end
        @test results[1].design == results[2].design
        @test results[1].physical == results[2].physical
        @test !results[2].cache.damping_active && results[2].cache.secant_valid
        # After a reset (projection beta step) the next call is again undamped.
        cache = results[2].cache
        reset_secant!(cache)
        @test !cache.secant_valid && all(iszero,cache.damping)
    end
end

@testset "Damped explicit and semi-implicit steps match the closed form" begin
    faces = kernel_faces()
    x0 = [0.2,0.4,0.6,0.3,0.5]
    lower,eta0 = 0.001,15.0
    rho = dot(faces.area,x0)/sum(faces.area)
    k = [3.0,40.0,8.0,0.0,6.0]
    refresh! = quadratic_force([2.0,9.0,3.0,1.0,4.0],k,[0.3,0.5,0.4,0.6,0.2])
    for scaling in GRADIENT_SCALINGS, integrator in (:explicit,:semi_implicit)
        allowed_update(scaling,integrator) || continue
        options = DensityOptions(;scaling,integrator,diag=true,substeps=1,diagnostics=true,
            implicit_tolerance=1e-14)
        cache = DensityCache(faces,0.02)
        x = copy(x0)
        advance_density!(x,zeros(5),p -> refresh!(p,x),cache;options,eta0,rho,lower)
        x1 = copy(x)
        # Second entry: secant curvature D = EMA(max(-dp/dx,0)) = k/2 (exact for a quadratic).
        p1 = refresh!(zeros(5),x1)
        q_scale = driving_scale(x1,p1,faces.area,lower).value
        frozen = DensityCache(faces,0.02)
        update_coefficients!(frozen,p1,q_scale,options)
        boost = frozen.beta_relative
        result = advance_density!(x,zeros(5),p -> refresh!(p,x),cache;options,eta0,rho,lower)
        D = cache.damping
        @test D ≈ k/2 rtol=1e-12
        @test cache.damping_active
        K = diffusion_matrix(faces,fill(0.02,5))
        q = p1/q_scale
        Dn = D/q_scale # normalized like q = p/s
        diffusion = K*x1 ./ faces.area
        if scaling == :local_lambda
            l = only(result.records).lambda/q_scale
            b = max.(boost,abs(l)/2)
            @test cache.beta_relative ≈ b
            expected = clamp.(x1 .+ (q .- l .- b.*diffusion)./(eta0.*b .+ Dn),lower,1.0)
        elseif integrator == :explicit
            # dx = dt*(q - l - b*(K x)/area)/(eta*b + D/s), dt = 1, l from the volume constraint.
            y = x1 .+ (q .- boost.*diffusion)./(eta0.*boost .+ Dn)
            expected = similar(x1)
            project_volume!(expected,y,faces.area,rho,lower;weights=1 ./(eta0.*boost .+ Dn),tolerance=1e-13)
        else
            # (eta*b + D/s)(x - x1) = q - l - b*(K x)/area, divided by b, times area/eta.
            m = 1 .+ Dn./(eta0.*boost)
            A = Diagonal(faces.area.*m)+K/eta0
            y = A\(faces.area.*(m.*x1 .+ q./(eta0.*boost)))
            w = A\(faces.area./boost)
            expected = similar(x1)
            project_volume!(expected,y,faces.area,rho,lower;weights=w,tolerance=1e-13)
        end
        @test x ≈ expected atol=2e-10
        @test dot(faces.area,x)/sum(faces.area) ≈ rho atol=2e-10
        @test only(result.records).damping_max ≈ maximum(k)/2
    end
end

@testset "Damping uses the full outer step, not the substep length" begin
    # beta = 0 removes diffusion and the force is frozen at the outer entry, so n
    # substeps are n volume-projected steps of length 1/n with viscosity eta0 + D/s
    # (T = 1). The diluted form eta0 + D*dt/s would weaken the damping to D/n.
    faces = kernel_faces()
    x0 = [0.4,0.5,0.6,0.45,0.55]
    lower,eta0,n = 0.001,4.0,4
    rho = dot(faces.area,x0)/sum(faces.area)
    k = [3.0,40.0,8.0,0.0,6.0]
    force! = quadratic_force([0.5,2.0,1.0,0.3,0.8],k,[0.45,0.5,0.55,0.5,0.5])
    # With a move limit each replayed substep is clamped to its per-cell box.
    for integrator in (:explicit,:semi_implicit), limit in (Inf,0.01)
        options = DensityOptions(;integrator,diag=true,substeps=n,implicit_tolerance=1e-14,move_limit=limit)
        cache = DensityCache(faces,0.0)
        x,entry = copy(x0),zeros(5)
        frozen! = p -> copyto!(p,entry)
        force!(entry,x)
        advance_density!(x,zeros(5),frozen!,cache;options,eta0,rho,lower)
        x1 = copy(x)
        force!(entry,x1)
        result = advance_density!(x,zeros(5),frozen!,cache;options,eta0,rho,lower)
        D = cache.damping
        @test cache.damping_active
        @test D ≈ k/2 rtol=1e-12
        function replay(T)
            y,trial = copy(x1),similar(x1)
            clipped = 0
            for _ in 1:n
                s = driving_scale(y,entry,faces.area,lower).value
                viscosity = eta0 .+ D.*T./s
                @. trial = y+(entry/s)/(n*viscosity)
                box = isfinite(limit) ? reference_box(y,lower,limit/n,3.0) : nothing
                project_volume!(y,trial,faces.area,rho,lower;weights=1 ./viscosity,tolerance=1e-13,box)
                box === nothing || (clipped += recount_move_limited(y,box,lower))
            end
            return y,clipped
        end
        (full,clipped),(diluted,_) = replay(1.0),replay(1/n)
        @test all(lower .< x .< 1)
        @test x ≈ full atol=1e-9
        isfinite(limit) ? (@test clipped > 0 && result.move_limited == clipped) :
            (@test maximum(abs,full-diluted) > 1e-3)
        @test dot(faces.area,x)/sum(faces.area) ≈ rho atol=2e-10
    end
end

@testset "Secant damping removes the two-cycle of a stiff single-cell objective" begin
    # One tiny cell carries a quadratic objective, p_1 = c - k*(x_1 - xstar); the
    # others have the constant force c, so lambda ≈ c and s ≈ c. Without damping the
    # map e <- (1 - k/(c*eta0)) e has |factor| = 2 > 1: x_1 two-cycles between
    # the bounds. With damping the viscosity eta0 + D/s, D -> k, makes it monotone.
    n = 10
    area = [1e-6; ones(n-1)]
    faces = TPFAFaces(area,collect(1:n-1),collect(2:n),ones(n-1),fill(0.5,n-1),fill(0.5,n-1))
    c,k,xstar,eta0 = 1.0,3.0,0.5,1.0
    for integrator in (:explicit,:semi_implicit)
        trajectories = map((false,true)) do diag
            cache = DensityCache(faces,0.0)
            x = fill(0.5,n); x[1] = 0.45
            refresh! = p -> (fill!(p,c); p[1] = c-k*(x[1]-xstar); p)
            options = DensityOptions(;integrator,diag,substeps=1)
            history = [x[1]]
            for _ in 1:12
                advance_density!(x,zeros(n),refresh!,cache;options,eta0,rho=0.5,lower=0.001)
                push!(history,x[1])
            end
            history .- xstar
        end
        plain,damped = trajectories
        @test all(sign(plain[i]) != sign(plain[i+1]) for i in 1:length(plain)-1)
        @test all(abs.(plain[2:end]) .>= abs(plain[1]))
        @test abs(plain[end]-plain[end-2]) < 1e-6 && abs(plain[end]-plain[end-1]) > 0.5
        # The first step has no history (same overshoot); afterwards the error
        # decreases monotonically, without sign changes from the third entry on.
        @test damped[2] == plain[2]
        @test all(abs(damped[i+1]) < abs(damped[i]) for i in 2:length(damped)-1)
        @test all(sign(damped[i]) == sign(damped[i+1]) for i in 3:length(damped)-1)
        @test abs(damped[end]) < 1e-7
    end
end

"""Every allowed update as (regularization, scaling, integrator, diag)."""
const MOVE_LIMIT_UPDATES = [[(:gradient,scaling,integrator,diag) for scaling in GRADIENT_SCALINGS
    for integrator in (:explicit,:semi_implicit) for diag in (false,true) if allowed_update(scaling,integrator)];
    [(:helmholtz,:mean,:explicit,false)]]

"""Outer iterations of one update with the substep start states (the final design appended)."""
function move_limit_run(update,limit,enabled;outer=3,substeps=3,simp_exponent=3.0,diagnostics=true)
    regularization,scaling,integrator,diag = update
    faces = kernel_faces()
    initial = [0.2,0.4,0.6,0.3,0.5]
    lower = 0.001
    rho = dot(faces.area,initial)/sum(faces.area)
    weights = [1.0,20.0,3.0,2.0,1.0]
    design,physical = copy(initial),copy(initial)
    projection = DensityProjection(DensityProjectionOptions(enabled=enabled),faces.area;rho_min=lower)
    options = DensityOptions(;regularization,scaling,integrator,diag,substeps,move_limit=limit,
        diagnostics,simp_exponent)
    entries = Vector{Float64}[]
    refresh! = p -> (push!(entries,copy(design)); @. p = 1+weights*physical^2)
    if regularization == :helmholtz
        eta0 = 1.0
        cache = HelmholtzCache(faces,ones(5),0.2)
        map_filtered_design!(physical,design,cache,projection)
        step! = () -> advance_filtered_design!(design,physical,zeros(5),refresh!,cache;
            options,eta0,rho,lower,density_projection=projection)
    else
        eta0 = 2.0
        cache = DensityCache(faces,0.02)
        projection_apply!(physical,projection,design)
        step! = () -> advance_density!(design,zeros(5),refresh!,cache;
            options,eta0,rho,lower,physical,density_projection=projection)
    end
    outers = map(1:outer) do _
        empty!(entries)
        result = step!()
        (entries=[copy(entries);[copy(design)]],physical=copy(physical),result)
    end
    return (;outers,design,physical,faces,rho,lower,eta0)
end

@testset "Move limit: per-cell SIMP-scaled box in every update" begin
    for update in MOVE_LIMIT_UPDATES, enabled in (false,true)
        limit = first(update) == :helmholtz ? 0.005 : 0.02
        run = move_limit_run(update,limit,enabled)
        (;faces,rho,lower,eta0) = run
        clipped = 0
        for (;entries,physical,result) in run.outers
            @test length(entries) == length(result.records)+1
            for (i,row) in enumerate(result.records)
                old,x = entries[i],entries[i+1]
                @test maximum(abs.(x.^3 .- old.^3)) <= limit*row.dt*(1+1e-12)
                box = reference_box(old,lower,limit*row.dt,3.0)
                @test row.move_limited == recount_move_limited(x,box,lower)
                @test row.eta == eta0*row.p_avg # eta is never raised by the move limit
            end
            @test result.move_limited == sum(row.move_limited for row in result.records)
            @test dot(faces.area,entries[end])/sum(faces.area) ≈ rho atol=2e-10
            @test dot(faces.area,physical)/sum(faces.area) ≈ rho atol=2e-10
            @test all(lower .<= entries[end] .<= 1)
            clipped += result.move_limited
        end
        @test clipped > 0
    end
    # The measure follows simp_exponent.
    run = move_limit_run(first(MOVE_LIMIT_UPDATES),0.01,false;simp_exponent=1.0)
    for (;entries,result) in run.outers, (i,row) in enumerate(result.records)
        @test maximum(abs.(entries[i+1] .- entries[i])) <= 0.01*row.dt*(1+1e-12)
    end
    @test sum(outer.result.move_limited for outer in run.outers) > 0
end

@testset "Move limit Inf and 1e300 are bitwise the unlimited update" begin
    for update in MOVE_LIMIT_UPDATES, enabled in (false,true), diagnostics in (false,true)
        unlimited,huge = (move_limit_run(update,limit,enabled;diagnostics) for limit in (Inf,1e300))
        @test unlimited.design == huge.design && unlimited.physical == huge.physical
        @test all(a.entries == b.entries && a.physical == b.physical for (a,b) in zip(unlimited.outers,huge.outers))
        @test all(outer.result.move_limited == 0 for outer in (unlimited.outers...,huge.outers...))
    end
end

@testset "Move limit clamps the unlimited trial to the box in the volume bisection" begin
    faces = kernel_faces()
    x0 = [0.2,0.4,0.6,0.3,0.5]
    lower,eta0,limit = 0.001,2.0,0.02
    rho = dot(faces.area,x0)/sum(faces.area)
    force = [1.0,20.0,3.0,2.0,1.0]
    K = diffusion_matrix(faces,fill(0.02,5))
    s = driving_scale(x0,force,faces.area,lower).value
    box = reference_box(x0,lower,limit,3.0)
    for integrator in (:explicit,:semi_implicit)
        x = copy(x0)
        options = DensityOptions(;integrator,substeps=1,move_limit=limit,implicit_tolerance=1e-14,diagnostics=true)
        result = advance_density!(x,force,identity,DensityCache(faces,0.02);options,eta0,rho,lower)
        @test only(result.records).dt == 1
        # The box acts after the (semi-implicit) solve, where [lower, 1] acts without it.
        y = integrator == :explicit ? x0 .+ (force./s .- K*x0./faces.area)./eta0 :
            (Diagonal(faces.area)+K/eta0)\(faces.area.*(x0 .+ force./(s*eta0)))
        expected = similar(x0)
        project_volume!(expected,y,faces.area,rho,lower;box,tolerance=1e-13)
        @test x ≈ expected atol=1e-9
        @test result.move_limited == recount_move_limited(x,box,lower) > 0
        free = copy(x0)
        advance_density!(free,force,identity,DensityCache(faces,0.02);
            options=DensityOptions(;integrator,substeps=1,implicit_tolerance=1e-14),eta0,rho,lower)
        @test maximum(abs.(free.^3 .- x0.^3)) > limit
    end
    # A box that cannot reach the volume target in one step is an error naming the move limit.
    for scaling in (:mean,:local_lambda)
        failure = try
            advance_density!(copy(x0),force,identity,DensityCache(faces,0.02);
                options=DensityOptions(;scaling,move_limit=1e-3,substeps=1),eta0=15.0,rho=rho+0.05,lower)
            nothing
        catch err
            err
        end
        @test failure isa ErrorException && occursin("move limit",failure.msg)
    end
end

@testset "Helmholtz move limit" begin
    faces = kernel_faces()
    initial = [0.2,0.4,0.6,0.3,0.5]
    lower = 0.001
    rho = dot(faces.area,initial)/sum(faces.area)
    for enabled in (false,true)
        designs = map((Inf,0.005)) do limit
            cache = HelmholtzCache(faces,ones(5),0.2)
            design,physical = copy(initial),copy(initial)
            projection = DensityProjection(DensityProjectionOptions(enabled=enabled),faces.area;rho_min=lower)
            map_filtered_design!(physical,design,cache,projection)
            options = DensityOptions(regularization=:helmholtz,move_limit=limit,substeps=2,diagnostics=true)
            entries = Vector{Float64}[]
            refresh! = p -> (push!(entries,copy(design)); @. p = 1+[1.0,20.0,3.0,2.0,1.0]*physical^2)
            result = advance_filtered_design!(design,physical,zeros(5),refresh!,cache;options,eta0=1.0,rho,lower,
                density_projection=projection)
            push!(entries,copy(design))
            @test isfinite(limit) ? result.move_limited > 0 : result.move_limited == 0
            isfinite(limit) && @test all(maximum(abs.(entries[i+1].^3 .- entries[i].^3)) <= limit*0.5*(1+1e-12) for i in 1:2)
            # The box is rebuilt from the latent design at each substep start.
            isfinite(limit) && @test result.move_limited == sum(recount_move_limited(entries[i+1],
                reference_box(entries[i],lower,limit*0.5,3.0),lower) for i in 1:2)
            @test dot(faces.area,design)/sum(faces.area) ≈ rho atol=2e-10
            design
        end
        @test designs[1] != designs[2]
    end
end

function damped_allocations(n,scaling,integrator,enabled)
    faces = TPFAFaces(ones(n),collect(1:n-1),collect(2:n),ones(n-1),fill(0.5,n-1),fill(0.5,n-1))
    cache = DensityCache(faces,0.01)
    design,physical = fill(0.3,n),fill(0.3,n)
    projection = DensityProjection(DensityProjectionOptions(enabled=enabled),faces.area;rho_min=0.001)
    projection_apply!(physical,projection,design)
    refresh! = p -> (@. p = 1+0.5sin(1:n)+2physical)
    options = DensityOptions(;scaling,integrator,diag=true,move_limit=0.05,substeps=1)
    force = zeros(n)
    step! = () -> advance_density!(design,force,refresh!,cache;options,
        eta0=15.0,rho=0.3,lower=0.001,physical,density_projection=projection)
    step!(); step!(); step!()
    return @allocated step!()
end

@testset "Damped, move-limited updates reuse mesh-sized storage" begin
    for scaling in GRADIENT_SCALINGS, integrator in (:explicit,:semi_implicit), enabled in (false,true)
        allowed_update(scaling,integrator) || continue
        small = damped_allocations(20,scaling,integrator,enabled)
        large = damped_allocations(2000,scaling,integrator,enabled)
        @test large <= small+1024
        @test large < 8192
    end
end
