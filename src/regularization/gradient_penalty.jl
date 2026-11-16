"""Choose a substep that ends at the outer time boundary and satisfies explicit CFL.

At most `options.max_substeps` substeps are taken: the remaining time is spread
over the remaining substeps when the request or the CFL bound would need more.
An explicit step may then exceed the bound; `evolve_gradient_design!` reports it.
"""
function gradient_step_size(cache,options,eta0,nominal_dt,elapsed,substep)
    rate = options.integrator == :explicit || options.diagnostics ? diffusion_rate(cache,eta0) : 0.0
    remaining = 1.0-elapsed
    dt = min(nominal_dt,remaining)
    options.integrator == :explicit && rate > 0 && (dt=min(dt,0.95/rate))
    dt = max(dt,remaining/(options.max_substeps-substep+1))
    return dt,rate
end

"""Form the unconstrained explicit or semi-implicit gradient-penalty trial.
Implicit diffusion reuses its matrix, preconditioner and CG workspace.

The local boost b (`cache.beta_relative`, ones for `:mean`) divides only the
driving force, see `update_coefficients!`; diffusion uses the baseline K:

    explicit:       chi = old + dt/eta0 * (p/(scale*b) - K*old/area)
    semi-implicit:  (M + dt/eta0*K) chi = M*(old + dt/eta0*p/(scale*b))

With active secant damping (r = D/(s*eta0*b), see `update_secant_damping!`)

    explicit:       chi = old + (dt/eta0 * (p/(scale*b) - K*old/area))/(1+r)
    semi-implicit:  (M*diag(1+r) + dt/eta0*K) chi = M*((1+r)*old + dt/eta0*p/(scale*b)).

Without history (D = 0) the undamped expressions are evaluated unchanged.
"""
function gradient_trial!(cache,driving,scale,dt,eta0,options)
    area, old, boost = cache.faces.area, cache.old, cache.beta_relative
    a = dt/eta0
    system_seconds = solve_seconds = 0.0
    system_updated = false
    damped = cache.damping_active
    damped && damping_ratio!(cache,scale,eta0)
    if options.integrator == :semi_implicit
        timer = options.diagnostics ? time_ns() : UInt64(0)
        system_updated = prepare_density_system!(cache,dt,eta0)
        system_seconds = options.diagnostics ? (time_ns()-timer)*1e-9 : 0.0
        if damped
            @. cache.rhs = area*((1+cache.damping_ratio)*old+(a/boost)*(driving/scale))
        else
            @. cache.rhs = area*(old+(a/boost)*(driving/scale))
        end
        timer = options.diagnostics ? time_ns() : UInt64(0)
        solve_density_system!(cache,options)
        solve_seconds = options.diagnostics ? (time_ns()-timer)*1e-9 : 0.0
    else
        mul!(cache.rhs,cache.stiffness,old)
        if damped
            @. cache.trial = old+((a/boost)*(driving/scale)-a*(cache.rhs/area))/(1+cache.damping_ratio)
        else
            @. cache.trial = (old+(a/boost)*(driving/scale))-a*(cache.rhs/area)
        end
    end
    return (;system_updated,system_seconds,solve_seconds)
end

"""
Advance one unit of optimization time. The callback refreshes p at each substep
with the mechanical displacement held fixed by the caller. Semi-implicit means
one unconstrained diffusion solve, then box/volume projection, as in the thesis.
It is a split update, not the exact constrained quadratic minimizer.

When density projection is active, chi is the persistent design and physical
must be its separate projected field on entry. The callback evaluates mechanics
sensitivities at physical; their exact pullback drives the design update. Both
the gradient penalty and the box/volume constraint act on the design. Without
projection, physical defaults to chi and the mapping is the identity.

Diagnostics are opt-in. `collect_fields=true` captures only the outer input and
last substep for a scheduled snapshot; no snapshot copies are made otherwise.
The returned `changed` mask belongs to the cache and is reused by the next call.
It records physical-density excursions from the outer iteration's initial field,
using the configured activity tolerance with a numerical-error floor.
"""
function advance_density!(chi,p,refresh_driving!::F,cache::DensityCache;
        options=DensityOptions(),eta0,rho,lower,legacy_substeps=1,
        collect_fields::Bool=false,physical=chi,density_projection=nothing) where F
    validate_gradient_update(chi,physical,p,cache,options,eta0,lower,legacy_substeps,density_projection)
    settings = (;options,eta0,rho,lower,density_projection)
    work = prepare_gradient_update!(cache,chi,physical,p,settings,collect_fields)
    nominal_dt = 1.0/(options.substeps == 0 ? legacy_substeps : options.substeps)
    return evolve_gradient_design!(chi,physical,p,refresh_driving!,cache,work,settings,nominal_dt)
end

# Specialize the loop after selecting optional observation storage. This barrier
# keeps the diagnostic/snapshot type unions outside the numerical substeps.
function evolve_gradient_design!(chi,physical,p,refresh!::F,cache,work::W,settings::S,nominal_dt) where {F,W,S}
    options = settings.options
    elapsed = 0.0
    cfl_excess = 0.0
    eta_increases = 0
    eta_factor = 1.0 # largest eta/eta0 of the move limit in this outer iteration
    for substep in 1:options.max_substeps
        step = gradient_substep!(chi,physical,p,refresh!,cache,work,settings,
            substep,nominal_dt,elapsed)
        last_substep = elapsed+step.dt >= 1-8eps(Float64)
        observe_gradient_step!(cache,chi,physical,p,work,settings,step,substep,elapsed,last_substep)
        elapsed += step.dt
        eta_increases += step.eta_increases
        eta_factor = max(eta_factor,step.eta/settings.eta0)
        options.integrator == :explicit && (cfl_excess=max(cfl_excess,step.dt*step.rate/0.95))
        if last_substep
            cfl_excess > 1+1e-12 && println("[density] explicit update capped at ",
                options.max_substeps," substeps; step is ",round(cfl_excess,sigdigits=3),
                "x the stability limit (",options.scaling == :local_lambda ?
                "raise max_substeps or eta0; local_lambda is explicit only" :
                "use integrator=semi_implicit to avoid this",")")
            return (changed=cache.changed,records=work.records,eta_increases,eta_factor)
        end
    end
    error("Density update exceeded $(options.max_substeps) substeps at time $elapsed")
end

function validate_gradient_update(chi,physical,p,cache,options,eta0,lower,legacy_substeps,projection)
    validate_density_options(options,eta0,legacy_substeps)
    options.regularization == :gradient ||
        throw(ArgumentError("Helmholtz requires a persistent design and HelmholtzCache"))
    length(chi) == length(physical) == length(p) == length(cache.faces.area) ||
        throw(DimensionMismatch("Density vectors"))
    projection !== nothing && physical === chi && throw(ArgumentError(
        "Active density projection requires separate persistent design and physical vectors"))
    all(x -> isfinite(x) && lower-1e-12 <= x <= 1+1e-12,chi) ||
        throw(ArgumentError("Initial density outside bounds"))
    return nothing
end

"""Prepare reusable update storage and optional outer-iteration snapshots."""
function prepare_gradient_update!(cache,chi,physical,p,settings,collect_fields)
    (;options,density_projection) = settings
    fill!(cache.changed,false)
    copyto!(cache.activity_reference,physical)
    cache.fields = (;)
    if (options.diagnostics || collect_fields) && cache.diagnostic_work === nothing
        cache.diagnostic_work = DiagnosticWorkspace(length(chi))
    end
    outer = collect_fields ? (physical=copy(physical),design=copy(chi),driving=similar(p),
        threshold=projection_threshold(density_projection)) : nothing
    driving = density_projection === nothing ? p : resize!(cache.design_driving,length(p))
    records = options.diagnostics ? NamedTuple[] : ()
    return (;outer,driving,records)
end

"""Refresh sensitivity, freeze diffusion, advance design, and update physical activity."""
function gradient_substep!(chi,physical,p,refresh!::F,cache,work,settings,substep,nominal_dt,elapsed) where F
    (;options,eta0,rho,lower,density_projection) = settings
    start_time = diagnostic_timer(options)
    threshold = projection_threshold(density_projection)
    copyto!(cache.old,chi)
    (options.diagnostics || work.outer !== nothing) && copyto!(cache.diagnostic_work.physical_before,physical)
    refresh!(p)
    work.outer !== nothing && substep == 1 && copyto!(work.outer.driving,p)
    projected_driving!(work.driving,p,density_projection)
    # The secant reads consecutive outer entry states, not the substeps.
    options.diag && substep == 1 && update_secant_damping!(cache,cache.old,work.driving)
    scale = driving_scale(cache.old,work.driving,cache.faces.area,lower)
    timer = diagnostic_timer(options)
    update_coefficients!(cache,work.driving,scale.value,options)
    coefficient_seconds = diagnostic_seconds(options,timer)
    if scale.force_max == 0
        # Both force-dependent beta and eta vanish. Retain the 3D stationary
        # convention instead of manufacturing diffusion through a scale floor.
        dt,rate = min(nominal_dt,1.0-elapsed),0.0
        copyto!(cache.trial,cache.old)
        trial = (system_updated=false,system_seconds=0.0,solve_seconds=0.0)
        timer = diagnostic_timer(options)
        projection = project_volume!(chi,cache.trial,cache.faces.area,rho,lower;
            tolerance=options.volume_tolerance)
        projection_seconds = diagnostic_seconds(options,timer)
        eta,eta_increases = eta0,0
    else
        dt,rate = gradient_step_size(cache,options,eta0,nominal_dt,elapsed,substep)
        (;trial,projection,eta,eta_increases,projection_seconds) =
            move_limited_step!(chi,cache,work,scale.value,dt,settings)
    end
    timer = diagnostic_timer(options)
    projection_apply!(physical,density_projection,chi)
    projection_seconds += diagnostic_seconds(options,timer)
    mark_physical_activity!(cache.changed,physical,cache.activity_reference,options)
    return (;dt,rate,scale,projection,threshold,start_time,coefficient_seconds,projection_seconds,
        trial,eta,eta_increases)
end

"""Most retries of one substep under the move limit before giving up."""
const MOVE_LIMIT_ATTEMPTS = 50

"""SIMP-scaled design change max_i |chi_i^p - old_i^p| of a substep."""
function simp_move(chi,old,exponent)
    move = 0.0
    @inbounds for i in eachindex(chi,old)
        move = max(move,abs(chi[i]^exponent-old[i]^exponent))
    end
    return move
end

"""Trial and volume projection (bisection) of one substep, with the move limit.

With the SIMP exponent p (`options.simp_exponent`) and m = max_i |chi_i^p - old_i^p|,
a substep of length dt with m > move_limit*dt is repeated from the same `old`
with eta <- eta*1.01*m/(move_limit*dt). The change is close to linear in dt/eta,
so one or two repetitions usually suffice. eta restarts from eta0 in every
substep, and eta is the only quantity that changes: driving force, scale,
diffusion operator, damping D and boost stay fixed (the floored projection of
`:local_lambda` raised the boost with the rejected multiplier, so it is reset).
Each repetition rebuilds M + dt/eta*K for the semi-implicit integrator. The
substep length dt is not changed. `move_limit = Inf` is the unlimited update.
"""
function move_limited_step!(chi,cache,work,scale,dt,settings)
    (;options,eta0,rho,lower) = settings
    limit = options.move_limit*dt
    eta = tried = eta0
    for attempt in 1:MOVE_LIMIT_ATTEMPTS
        tried = eta
        trial = gradient_trial!(cache,work.driving,scale,dt,eta,options)
        # `:local_lambda` (explicit only) puts the current |lambda| into the boost.
        floored = options.scaling == :local_lambda
        weights = floored ? nothing : volume_response!(cache,options)
        timer = diagnostic_timer(options)
        projection = floored ?
            project_volume_floored!(chi,cache,work.driving,scale,dt/eta,rho,lower;
                tolerance=options.volume_tolerance,damping=cache.damping_active ? 1/(scale*eta) : 0.0) :
            project_volume!(chi,cache.trial,cache.faces.area,rho,lower;
                tolerance=options.volume_tolerance,weights)
        projection_seconds = diagnostic_seconds(options,timer)
        move = isfinite(limit) ? simp_move(chi,cache.old,options.simp_exponent) : 0.0
        move <= limit && return (;trial,projection,eta,eta_increases=attempt-1,projection_seconds)
        eta *= 1.01*move/limit
        floored && update_coefficients!(cache,work.driving,scale,options)
    end
    error("Move limit $(options.move_limit) not reached after $MOVE_LIMIT_ATTEMPTS attempts ",
        "(last tried eta/eta0 = $(tried/eta0))")
end

function observe_gradient_step!(cache,chi,physical,p,work,settings,step,substep,elapsed,last_substep)
    (;options,rho,lower,density_projection) = settings
    snapshot = work.outer !== nothing && last_substep
    options.diagnostics || snapshot || return nothing
    diagnostic = cache.diagnostic_work::DiagnosticWorkspace
    regularization_forces!(diagnostic,cache,chi)
    if options.diagnostics
        # diffusion_cfl at the accepted eta (rate ~ 1/eta); damping, which lowers
        # the effective rate further by 1/(1+r_i), is not included (upper bound).
        rate = step.eta == settings.eta0 ? step.rate : step.rate*(settings.eta0/step.eta)
        push!(work.records,diagnostic_record!(diagnostic,cache,chi,work.driving,step.scale,step.projection,options;
            eta0=step.eta,eta_increases=step.eta_increases,rho,lower,substep,elapsed,dt=step.dt,
            rate,start_time=step.start_time,
            coefficient_seconds=step.coefficient_seconds,projection_seconds=step.projection_seconds,
            step.trial...,physical,physical_driving=p,density_projection))
    end
    snapshot && save_gradient_snapshot!(cache,chi,physical,p,work,settings,step)
    return nothing
end

function save_gradient_snapshot!(cache,chi,physical,p,work,settings,step)
    outer,diagnostic = work.outer,cache.diagnostic_work
    projection,s = settings.density_projection,step.scale.value
    cache.fields = (chi_before_outer=outer.physical,driving_force_outer=outer.driving,
        density_projection_metadata(projection)...,
        density_projection_threshold_before=step.threshold,density_projection_threshold_outer=outer.threshold,
        design_before_outer=outer.design,chi_before=copy(diagnostic.physical_before),chi_after=copy(physical),
        driving_force=copy(p),design_before=copy(cache.old),design_after=copy(chi),
        design_driving_force=copy(work.driving),regularization_field=projection === nothing ? "physical" : "design",
        beta=s.*cache.base_beta.*cache.beta_relative,beta_boost=copy(cache.beta_relative),
        regularization_force_before=s.*diagnostic.reg_before,regularization_force_after=s.*diagnostic.reg_after)
    return nothing
end
