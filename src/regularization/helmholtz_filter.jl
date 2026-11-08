function HelmholtzCache(faces::TPFAFaces, h::AbstractVector, radius::Real)
    n = length(faces.area)
    cache = HelmholtzCache(faces,zeros(n),TPFAAssemblyWorkspace(),spzeros(n,n),Diagonal(zeros(n)),
        CgWorkspace(n,n,Vector{Float64}),zeros(n),zeros(n),zeros(n),zeros(n),zeros(n),
        Float64[],Float64[],zeros(n),falses(n),nothing,(;))
    return rebuild_filter!(cache,faces,h,radius)
end

"""Reuse face arrays and dense work across remeshing; rebuild only the sparse operator.
The Krylov workspace is replaced only when the number of active cells changes.
"""
function rebuild_filter!(cache::HelmholtzCache,faces::TPFAFaces,h,radius)
    validate_filter_mesh(faces,h,radius)
    cache.faces = faces
    resize!(cache.radius2,length(faces.area))
    @. cache.radius2 = regularization_coefficient(radius^2,h)
    assemble_filter_operator!(cache)
    resize_filter_workspace!(cache,length(faces.area))
    return cache
end

function validate_filter_mesh(faces,h,radius)
    n = length(faces.area)
    n > 0 && length(h) == n || throw(DimensionMismatch("Filter mesh"))
    isfinite(radius) && radius > 0 || throw(ArgumentError("Filter radius must be finite and positive"))
    all(x -> isfinite(x) && x > 0,h) || throw(ArgumentError("Invalid cell diameter"))
    all(x -> isfinite(x) && x > 0,faces.area) || throw(ArgumentError("Invalid cell area"))
    return nothing
end

"""Assemble B=M+K with zero boundary flux and its Jacobi preconditioner."""
function assemble_filter_operator!(cache)
    faces = cache.faces
    cache.system = diffusion_matrix(faces,cache.radius2,cache.assembly)
    resize!(cache.preconditioner.diag,length(faces.area))
    for i in eachindex(faces.area)
        k = stored_index(cache.system,i,i)
        nonzeros(cache.system)[k] += faces.area[i]
        cache.preconditioner.diag[i] = inv(nonzeros(cache.system)[k])
    end
    return nothing
end

"""Retain dense work and CG storage across remeshing; discard stale sensitivities."""
function resize_filter_workspace!(cache,n)
    cache.solver.n != n && (cache.solver=CgWorkspace(n,n,Vector{Float64}))
    resize!(cache.solver.z,n) # Provision Krylov's otherwise lazily allocated vector.
    for v in (cache.rhs,cache.q,cache.old,cache.design_before,cache.trial,cache.activity_reference,cache.changed)
        resize!(v,n)
    end
    for v in (cache.filtered,cache.filtered_driving)
        isempty(v) || resize!(v,n) # Only active projection needs these fields.
    end
    fill!(cache.q,0.0)
    cache.fields = (;)
    if cache.diagnostic_work !== nothing
        work = cache.diagnostic_work
        for v in (work.product,work.trial,work.projected,work.filtered_before)
            resize!(v,n)
        end
    end
    return nothing
end

"""
Apply F = (M + K)⁻¹ M in place. The input is a cell value (or a sensitivity
per unit area), not an area-integrated Euclidean gradient. Because B is
symmetric, the same solve is the adjoint in the area inner product.
Aliasing input and output is allowed: the right-hand side is assembled before
solving. Start CG from zero so its relative tolerance refers to this right-hand
side, even if the force becomes tiny or vanishes between optimization steps.
"""
function apply_filter!(output, input, cache::HelmholtzCache; tolerance=1e-12)
    length(output) == length(input) == length(cache.rhs) || throw(DimensionMismatch("Filter vectors"))
    @. cache.rhs = cache.faces.area*input
    cg!(cache.solver,cache.system,cache.rhs;
        M=cache.preconditioner,atol=0.0,rtol=tolerance,history=false)
    cache.solver.stats.solved || error("Helmholtz solve failed: $(cache.solver.stats.status)")
    copyto!(output,cache.solver.x)
    return cache.solver.stats.niter
end

"""Map persistent design to physical density, including the optional projection.
Keep the unprojected filter output only when projection needs it; the identity
mapping filters directly into chi. Call after initial setup and every remesh.
"""
function map_filtered_design!(chi,design,cache::HelmholtzCache,density_projection=nothing;
        tolerance=1e-12)
    filtered = density_projection === nothing ? chi : resize!(cache.filtered,length(design))
    iterations = apply_filter!(filtered,design,cache;tolerance)
    projection_apply!(chi,density_projection,filtered)
    return iterations
end

"""
One unit of explicit evolution of the persistent design a:
    B q = M P'(filtered)' p;  a⁺ = project(a + dt*q/(eta0*s));
    B filtered⁺ = M a⁺;  chi⁺ = P(filtered⁺).
P is the volume-preserving density projection, or the identity when disabled.
Here p = -dJ/dchi per unit area; s is its adjoint's grayness-weighted mean
in design space. eta controls only the step. No gradient-energy penalty is added.

On entry chi must equal P(F*a) on this mesh. The driver enforces this
after each conservative mesh transfer, before computing mechanics. By default
there is one design step per mechanical solve. Optional extra substeps retain
the caller's frozen displacement, just as in the mean-beta reference update.
"""
function advance_filtered_design!(design,chi,p,refresh_driving!::F,cache::HelmholtzCache;
        options=DensityOptions(integrator=:explicit,regularization=:helmholtz),
        eta0,rho,lower,collect_fields::Bool=false,density_projection=nothing) where F
    steps = validate_filter_update(design,chi,p,cache,options,eta0,lower)
    settings = (;options,eta0,rho,lower,density_projection,dt=1.0/steps)
    work = prepare_filter_update!(cache,design,chi,p,settings,collect_fields)
    return evolve_filtered_design!(design,chi,p,refresh_driving!,cache,work,settings,steps)
end

function evolve_filtered_design!(design,chi,p,refresh!::F,cache,work::W,settings::S,steps) where {F,W,S}
    for substep in 1:steps
        step = filtered_design_substep!(design,chi,p,refresh!,cache,work,settings,substep,steps)
        observe_filter_step!(cache,design,chi,p,work,settings,step,substep,steps)
    end
    return (changed=cache.changed,records=work.records)
end

function validate_filter_update(design,chi,p,cache,options,eta0,lower)
    validate_density_options(options,eta0,1)
    options.regularization == :helmholtz || throw(ArgumentError("HelmholtzCache requires regularization=:helmholtz"))
    length(design) == length(chi) == length(p) == length(cache.rhs) || throw(DimensionMismatch("Design vectors"))
    design === chi && throw(ArgumentError("Helmholtz requires separate persistent design and physical vectors"))
    invalid = findfirst(x -> !(isfinite(x) && lower <= x <= 1),design)
    invalid === nothing || throw(ArgumentError(
        "Design before density update is outside [$lower, 1]: design[$invalid] = $(design[invalid])"))
    steps = max(options.substeps,1)
    steps <= options.max_substeps || throw(ArgumentError("Too many design substeps"))
    return steps
end

function prepare_filter_update!(cache,design,chi,p,settings,collect_fields)
    (;options,density_projection) = settings
    fill!(cache.changed,false)
    copyto!(cache.activity_reference,chi)
    cache.fields = (;)
    records = options.diagnostics ? NamedTuple[] : ()
    if options.diagnostics && cache.diagnostic_work === nothing
        cache.diagnostic_work = FilterDiagnostics(length(chi))
    end
    filtered = density_projection === nothing ? chi : cache.filtered
    length(filtered) == length(design) || throw(ArgumentError(
        "Call map_filtered_design! before advancing a projected Helmholtz design"))
    driving = density_projection === nothing ? p : resize!(cache.filtered_driving,length(p))
    outer = collect_fields ? (physical=copy(chi),design=copy(design),filtered=copy(filtered),
        driving=similar(p),threshold=projection_threshold(density_projection)) : nothing
    return (;filtered,driving,outer,records)
end

"""Pull back sensitivity through projection/filter, then take the explicit design trial."""
function filtered_design_trial!(cache,design,p,work,settings)
    (;options,eta0,lower,dt,density_projection) = settings
    timer = diagnostic_timer(options)
    projected_driving!(work.driving,p,density_projection)
    adjoint_iterations = apply_filter!(cache.q,work.driving,cache;tolerance=options.filter_tolerance)
    adjoint_seconds = diagnostic_seconds(options,timer)
    scale = driving_scale(design,cache.q,cache.faces.area,lower)
    step = dt/(eta0*scale.value)
    @. cache.trial = design+step*cache.q
    return (;scale,adjoint_iterations,adjoint_seconds)
end

"""Enforce design constraints and map the accepted design to physical density."""
function project_filtered_trial!(cache,design,chi,settings)
    (;options,rho,lower,density_projection) = settings
    timer = diagnostic_timer(options)
    projection = project_volume!(design,cache.trial,cache.faces.area,rho,lower;tolerance=options.volume_tolerance)
    projection_seconds = diagnostic_seconds(options,timer)
    timer = diagnostic_timer(options)
    forward_iterations = map_filtered_design!(chi,design,cache,density_projection;tolerance=options.filter_tolerance)
    forward_seconds = diagnostic_seconds(options,timer)
    mark_physical_activity!(cache.changed,chi,cache.activity_reference,options)
    return (;projection,projection_seconds,forward_iterations,forward_seconds)
end

function filtered_design_substep!(design,chi,p,refresh!::F,cache,work,settings,substep,steps) where F
    (;options,density_projection) = settings
    start_time = diagnostic_timer(options)
    threshold = projection_threshold(density_projection)
    (options.diagnostics || work.outer !== nothing) && copyto!(cache.old,chi)
    options.diagnostics && copyto!(cache.diagnostic_work.filtered_before,work.filtered)
    filtered_before = work.outer !== nothing && substep == steps ? copy(work.filtered) : nothing
    copyto!(cache.design_before,design)
    refresh!(p)
    work.outer !== nothing && substep == 1 && copyto!(work.outer.driving,p)
    trial = filtered_design_trial!(cache,design,p,work,settings)
    accepted = project_filtered_trial!(cache,design,chi,settings)
    return (;start_time,threshold,filtered_before,trial,accepted)
end

function observe_filter_step!(cache,design,chi,p,work,settings,step,substep,steps)
    (;options,eta0,rho,lower,dt,density_projection) = settings
    trial,accepted = step.trial,step.accepted
    if options.diagnostics
        push!(work.records,filter_record!(cache.diagnostic_work,cache,design,chi,p,
            trial.scale,accepted.projection,options;eta0,rho,lower,substep,dt,
            adjoint_iterations=trial.adjoint_iterations,adjoint_seconds=trial.adjoint_seconds,
            forward_iterations=accepted.forward_iterations,forward_seconds=accepted.forward_seconds,
            projection_seconds=accepted.projection_seconds,start_time=step.start_time,
            filtered=work.filtered,filtered_driving=work.driving,density_projection))
    end
    work.outer !== nothing && substep == steps && save_filter_snapshot!(cache,design,chi,p,work,settings,step)
    return nothing
end

function save_filter_snapshot!(cache,design,chi,p,work,settings,step)
    outer = work.outer
    cache.fields = (chi_before_outer=outer.physical,design_before_outer=outer.design,
        density_projection_metadata(settings.density_projection)...,
        density_projection_threshold_before=step.threshold,density_projection_threshold_outer=outer.threshold,
        driving_force_outer=outer.driving,chi_before=copy(cache.old),chi_after=copy(chi),
        design_before=copy(cache.design_before),design_after=copy(design),
        filtered_before_outer=outer.filtered,filtered_before=step.filtered_before,filtered_after=copy(work.filtered),
        driving_force=copy(p),filtered_driving_force=copy(work.driving),
        design_driving_force=copy(cache.q),radius2=copy(cache.radius2))
    return nothing
end
