# Unequal cell volumes, a split face and an isolated cell exercise the actual
# finite-volume algebra without coupling these unit tests to a mesh generator.
function kernel_faces()
    TPFAFaces([1.0,2.0,0.5,3.0,4.0],[1,1,2,3],[2,2,3,4],
        [0.4,0.6,1.0,0.7],[0.5,0.5,0.8,0.25],[0.8,0.8,0.25,1.0])
end

@testset "Zero gradient coefficient preserves the mechanical update" begin
    faces = kernel_faces()
    initial = [0.2,0.4,0.6,0.3,0.5]
    force = [1.0,20.0,3.0,2.0,1.0]
    rho = dot(faces.area,initial)/sum(faces.area)
    reference = similar(initial)
    scale = driving_scale(initial,force,faces.area,0.001).value
    project_volume!(reference,initial+force/(15scale),faces.area,rho,0.001)
    for integrator in (:explicit,:semi_implicit), method in (:mean,:local_max)
        cache = DensityCache(faces,0.0)
        design = copy(initial)
        advance_density!(design,force,identity,cache;
            options=DensityOptions(;integrator,regularization=method),eta0=15.0,rho,lower=0.001)
        @test design ≈ reference atol=2e-12
        @test all(iszero,nonzeros(cache.stiffness))
        @test nnz(cache.stiffness) > 5 # Structural face entries remain cached.
    end
    cache = DensityCache(faces,[0.1,0.0,0.2,0.3,0.4])
    @test cache.stiffness[1,2] == cache.stiffness[2,3] == 0
end

@testset "Box-volume Newton agrees with independent bisection" begin
    volumes = [0.1,2.0,0.03,5.0,0.7]
    for trial in ([-2.,4.,0.5,-0.2,1.5],fill(0.3,5),[2.,2.,-3.,-3.,0.5]), rho in (0.001,0.1,0.6,1.0)
        result = similar(trial)
        project_volume!(result,trial,volumes,rho,0.001;tolerance=1e-12)
        lo,hi = minimum(trial)-1,maximum(trial)-0.001
        for _ in 1:100
            middle = (lo+hi)/2
            mass = dot(volumes,clamp.(trial.-middle,0.001,1.0))/sum(volumes)
            mass > rho ? (lo=middle) : (hi=middle)
        end
        expected = clamp.(trial.-(lo+hi)/2,0.001,1.0)
        @test result ≈ expected atol=2e-10
        @test dot(volumes,result)/sum(volumes) ≈ rho atol=1e-12
    end
end

@testset "Conservative TPFA and direct-neighbor local beta" begin
    faces = kernel_faces()
    baseline = [0.01,0.02,0.005,0.03,0.04]
    cache = DensityCache(faces,baseline)
    force = [1.0,20.0,3.0,2.0,1.0]
    opts = DensityOptions(regularization=:local_max)
    @test update_coefficients!(cache,force,2.0,opts)
    @test cache.base_beta == baseline
    @test cache.beta_relative == [10.0,10.0,10.0,1.5,1.0]
    expected = diffusion_matrix(faces,baseline.*cache.beta_relative)
    @test cache.stiffness ≈ expected
    @test issymmetric(expected)
    @test expected*ones(5) ≈ zeros(5) atol=1e-15
    @test minimum(eigvals(Symmetric(Matrix(expected)))) >= -1e-14
    @test expected[5,5] == 0
    reverse_faces = TPFAFaces(faces.area,reverse(faces.left),reverse(faces.right),
        reverse(faces.length),reverse(faces.distance_left),reverse(faces.distance_right))
    reversed = DensityCache(reverse_faces,baseline)
    update_coefficients!(reversed,force,2.0,opts)
    @test reversed.beta_relative == cache.beta_relative
    @test reversed.stiffness ≈ expected
    @test !update_coefficients!(cache,force,2.0,opts)
    @test update_coefficients!(cache,force,2.0,DensityOptions())
    @test cache.stiffness ≈ diffusion_matrix(faces,baseline)
    @test cache.solver === nothing # Explicit updates need no implicit workspace.
end

@testset "Gradient variants satisfy the frozen equation, with optional projection" begin
    faces = kernel_faces()
    initial = [0.2,0.4,0.6,0.3,0.5]
    baseline = [0.01,0.02,0.005,0.03,0.04]
    force = [1.0,20.0,3.0,2.0,1.0]
    lower,eta0 = 0.001,15.0
    rho = dot(faces.area,initial)/sum(faces.area)
    for method in (:mean,:local_max), integrator in (:explicit,:semi_implicit), enabled in (false,true)
        cache = DensityCache(faces,baseline)
        design,physical = copy(initial),copy(initial)
        projection = DensityProjection(DensityProjectionOptions(enabled=enabled,beta_initial=4.0),
            faces.area;rho_min=lower)
        projection_apply!(physical,projection,design)
        pulled = copy(force)
        projection_pullback!(pulled,projection)
        scale = driving_scale(initial,pulled,faces.area,lower).value
        opts = DensityOptions(;integrator,regularization=method,substeps=1,
            implicit_tolerance=1e-13,diagnostics=true)
        update_coefficients!(cache,pulled,scale,opts)
        K = copy(cache.stiffness)
        trial = integrator == :explicit ? initial+(pulled/scale-K*initial./faces.area)/eta0 :
            (Diagonal(faces.area)+K/eta0)\(faces.area.*(initial+pulled/(eta0*scale)))
        expected = similar(initial)
        project_volume!(expected,trial,faces.area,rho,lower)
        result = advance_density!(design,copy(force),identity,cache;options=opts,
            eta0,rho,lower,physical,density_projection=projection)
        @test design ≈ expected atol=2e-12
        @test dot(faces.area,physical)/sum(faces.area) ≈ rho atol=2e-10
        @test all(lower .<= physical .<= 1)
        @test length(result.records) == 1
        if integrator == :semi_implicit
            @test norm(cache.system*cache.trial-cache.rhs)/norm(cache.rhs) < 1e-12
            system,solver = cache.system,cache.solver
            @test !prepare_density_system!(cache,1.0,eta0)
            @test prepare_density_system!(cache,0.5,eta0)
            @test cache.system === system && cache.solver === solver
        end
    end
end

@testset "Helmholtz invariants and unequal-volume adjoint" begin
    faces = kernel_faces()
    cache = HelmholtzCache(faces,[0.8,1.0,0.3,2.0,1.0],0.3)
    design = [0.15,0.4,0.8,0.65,0.3]
    physical = similar(design)
    apply_filter!(physical,design,cache)
    @test cache.radius2 == fill(0.3^2,5)
    @test cache.system*ones(5) ≈ faces.area
    @test minimum(design)-1e-12 <= minimum(physical) <= maximum(physical) <= maximum(design)+1e-12
    @test dot(faces.area,physical) ≈ dot(faces.area,design) atol=2e-12
    @test physical[5] ≈ design[5]
    weights,direction = [2.0,0.7,4.0,1.1,0.2],[0.2,-0.1,0.4,-0.2,0.3]
    for enabled in (false,true)
        projection = DensityProjection(DensityProjectionOptions(enabled=enabled,beta_initial=4.0),
            faces.area;rho_min=0.001)
        map_filtered_design!(physical,design,cache,projection)
        g = weights.*physical
        projection_pullback!(g,projection)
        q = similar(g)
        apply_filter!(q,g,cache)
        function objective(a)
            # Independent direct solve and threshold bisection from projection tests.
            filtered = cache.system\(faces.area.*a)
            field = enabled ? _reference_projection(filtered,faces.area,4.0;rho_min=0.001) : filtered
            return dot(faces.area,weights.*field.^2)/2
        end
        epsilon = 1e-6
        fd = (objective(design+epsilon*direction)-objective(design-epsilon*direction))/(2epsilon)
        @test dot(faces.area.*q,direction) ≈ fd rtol=2e-7 atol=2e-9
        @test dot(faces.area,physical) ≈ dot(faces.area,design) atol=2e-11
    end
    apply_filter!(physical,ones(5),cache)
    @test physical ≈ ones(5) atol=2e-12
    @test_throws ArgumentError advance_filtered_design!(design,design,ones(5),identity,cache;
        eta0=15.0,rho=0.3,lower=0.001)
    solver,rhs = cache.solver,cache.rhs
    old_system = copy(cache.system)
    faces.length[1] *= 2
    rebuild_filter!(cache,faces,ones(5),0.3)
    @test cache.solver === solver && cache.rhs === rhs
    @test cache.system != old_system
end

@testset "Zero-force stationarity and explicit stability" begin
    faces = kernel_faces()
    initial = [0.2,0.4,0.6,0.3,0.5]
    rho = dot(faces.area,initial)/sum(faces.area)
    for method in (:mean,:local_max,:helmholtz), integrator in (:explicit,:semi_implicit)
        method == :helmholtz && integrator == :semi_implicit && continue
        cache = method == :helmholtz ? HelmholtzCache(faces,ones(5),0.2) : DensityCache(faces,0.02)
        design,physical = copy(initial),copy(initial)
        opts = DensityOptions(;integrator,regularization=method)
        if method == :helmholtz
            map_filtered_design!(physical,design,cache)
            oldphysical = copy(physical)
            advance_filtered_design!(design,physical,zeros(5),identity,cache;
                options=opts,eta0=15.0,rho,lower=0.001)
            @test physical ≈ oldphysical atol=2e-12
        else
            advance_density!(design,zeros(5),identity,cache;options=opts,
                eta0=15.0,rho,lower=0.001,physical)
        end
        @test design == initial
    end
    cache = DensityCache(faces,5.0)
    design = copy(initial)
    result = advance_density!(design,[1.,20.,3.,2.,1.],identity,cache;
        options=DensityOptions(regularization=:local_max,diagnostics=true),eta0=1.0,rho,lower=0.001)
    @test length(result.records) > 1
    @test all(row.diffusion_cfl <= 0.95+1e-12 for row in result.records)
    @test last(result.records).pseudo_time ≈ 1.0
    @test dot(faces.area,design)/sum(faces.area) ≈ rho atol=2e-10
end

@testset "Diagnostics and snapshots preserve all density update paths" begin
    faces = kernel_faces()
    initial = [0.2,0.4,0.6,0.3,0.5]
    rho = dot(faces.area,initial)/sum(faces.area)
    weights = [1.0,20.0,3.0,2.0,1.0]
    for method in (:mean,:local_max,:helmholtz), integrator in (:explicit,:semi_implicit), enabled in (false,true)
        method == :helmholtz && integrator == :semi_implicit && continue
        results = map(1:3) do mode
            cache = method == :helmholtz ? HelmholtzCache(faces,ones(5),0.2) : DensityCache(faces,0.02)
            design,physical = copy(initial),copy(initial)
            projection = DensityProjection(DensityProjectionOptions(enabled=enabled),faces.area;rho_min=0.001)
            opts = DensityOptions(;integrator,regularization=method,substeps=3,diagnostics=mode==2)
            force = zeros(5)
            refresh! = p -> (@. p = 1+weights*physical^2)
            if method == :helmholtz
                map_filtered_design!(physical,design,cache,projection)
                result = advance_filtered_design!(design,physical,force,refresh!,cache;
                    options=opts,eta0=15.0,rho,lower=0.001,collect_fields=mode==3,
                    density_projection=projection)
            else
                projection_apply!(physical,projection,design)
                result = advance_density!(design,force,refresh!,cache;
                    options=opts,eta0=15.0,rho,lower=0.001,physical,collect_fields=mode==3,
                    density_projection=projection)
            end
            (;cache,design,physical,result)
        end
        @test results[1].design == results[2].design == results[3].design
        @test results[1].physical == results[2].physical == results[3].physical
        @test results[1].result.changed == results[2].result.changed == results[3].result.changed
        @test results[1].cache.diagnostic_work === nothing
        @test isempty(results[1].result.records) && isempty(results[1].cache.fields)
        @test length(results[2].result.records) == 3
        @test !isempty(results[3].cache.fields)
        @test results[3].cache.fields.design_after == results[3].design
    end
end

function kernel_allocations(n,method,integrator,enabled)
    faces = TPFAFaces(ones(n),collect(1:n-1),collect(2:n),ones(n-1),fill(0.5,n-1),fill(0.5,n-1))
    cache = method == :helmholtz ? HelmholtzCache(faces,ones(n),0.1) : DensityCache(faces,0.01)
    design,physical = fill(0.3,n),fill(0.3,n)
    force = [1+0.5sin(i) for i in 1:n]
    projection = DensityProjection(DensityProjectionOptions(enabled=enabled),faces.area;rho_min=0.001)
    opts = DensityOptions(;integrator,regularization=method,substeps=1)
    if method == :helmholtz
        map_filtered_design!(physical,design,cache,projection)
        step! = () -> advance_filtered_design!(design,physical,force,identity,cache;options=opts,
            eta0=15.0,rho=0.3,lower=0.001,density_projection=projection)
    else
        projection_apply!(physical,projection,design)
        step! = () -> advance_density!(design,force,identity,cache;options=opts,
            eta0=15.0,rho=0.3,lower=0.001,physical,density_projection=projection)
    end
    step!(); step!()
    return @allocated step!()
end

@testset "Steady mesh updates reuse mesh-sized storage" begin
    for method in (:mean,:local_max,:helmholtz), integrator in (:explicit,:semi_implicit), enabled in (false,true)
        method == :helmholtz && integrator == :semi_implicit && continue
        small = kernel_allocations(20,method,integrator,enabled)
        large = kernel_allocations(2000,method,integrator,enabled)
        @test large <= small+1024
        @test large < 8192
    end
end
