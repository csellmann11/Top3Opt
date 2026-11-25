"""Original grayness-weighted mean, with explicit binary/zero-force fallbacks."""
function driving_scale(chi, p, area, lower)
    denom = weighted_p = area_p = total = pmax = 0.0
    for i in eachindex(chi,p,area)
        isfinite(p[i]) || throw(ArgumentError("Nonfinite density driving force"))
        w = area[i]*max((chi[i]-lower)*(1-chi[i]),0.0)
        denom += w
        weighted_p += w*p[i]
        area_p += area[i]*p[i]
        total += area[i]
        pmax = max(pmax,abs(p[i]))
    end
    fallback = denom <= eps(Float64)*total
    raw = fallback ? area_p/total : weighted_p/denom
    floor = eps(Float64)*max(pmax,1.0)
    return (value=max(abs(raw),floor), raw=raw, force_max=pmax, fallback=fallback,
            floored=abs(raw)<floor, gray_weight=denom/total)
end

@inline volume_weight(::Nothing,i) = 1.0
@inline volume_weight(weights,i) = weights[i]

"""Bounds of cell i: the density bounds [lower, 1], or the move-limit box (see `move_box!`)."""
@inline cell_bounds(::Nothing,lower,i) = (lower,1.0)
@inline cell_bounds(box,lower,i) = (box.lower[i],box.upper[i])

"""Move-limit box of a substep of length dt around its start state `old`.

With p = `options.simp_exponent` and the limit m = move_limit*dt, cell i may move to

    lo_i = max(lower, max(old_i^p - m, 0)^(1/p)),   hi_i = min(1, (old_i^p + m)^(1/p)),

so |x_i^p - old_i^p| <= m inside the box. Rounding is guarded by
lo_i <= old_i <= hi_i, so `old` always lies in its box. The bounds are written
into the cache-owned `move_lower`/`move_upper`; `nothing` (no box) for
`move_limit = Inf`, so the projections keep their scalar bounds unchanged.
"""
function move_box!(cache,old,lower,options,dt)
    limit = options.move_limit*dt
    isfinite(limit) || return nothing
    p = options.simp_exponent
    lo, hi = resize!(cache.move_lower,length(old)), resize!(cache.move_upper,length(old))
    @inbounds for i in eachindex(old,lo,hi)
        x = old[i]
        xp = x^p
        lo[i] = min(max(lower,max(xp-limit,0.0)^(1/p)),x)
        hi[i] = max(min(1.0,(xp+limit)^(1/p)),x)
    end
    return (lower=lo,upper=hi)
end

"""Cells at a box bound strictly tighter than the density bound [lower, 1]."""
function count_move_limited(x,box,lower)
    box === nothing && return 0
    count = 0
    @inbounds for i in eachindex(x,box.lower,box.upper)
        lo, hi = box.lower[i], box.upper[i]
        ((lo > lower && x[i] == lo) || (hi < 1 && x[i] == hi)) && (count += 1)
    end
    return count
end

"""Area-metric projection: chi_i=clamp(y_i-shift*w_i, lower, 1), at prescribed volume.
The positive `weights` w_i are the response of cell i to a unit volume multiplier;
`nothing` is the uniform shift w_i = 1. A move-limit `box` (see `move_box!`)
replaces [lower, 1] by the per-cell bounds [lo_i, hi_i], as MMA move limits.
"""
function project_volume!(chi, y, area, rho, lower; tolerance=1e-10, weights=nothing, box=nothing)
    0 <= lower < 1 && lower <= rho <= 1 || throw(ArgumentError("Require 0 <= lower < 1 and lower <= rho <= 1"))
    tolerance > 0 || throw(ArgumentError("Projection tolerance must be positive"))
    all(isfinite, y) || throw(ArgumentError("Nonfinite density before projection"))
    weights === nothing || all(x -> isfinite(x) && x > 0, weights) ||
        throw(ArgumentError("Volume weights must be finite and positive"))
    box === nothing || length(box.lower) == length(box.upper) == length(y) ||
        throw(DimensionMismatch("Move-limit box"))
    total = sum(area)
    # Shifts that put every cell at the upper (lo) or lower (hi) bound.
    lo, hi = Inf, -Inf
    for i in eachindex(y)
        w = volume_weight(weights,i)
        lower_i, upper_i = cell_bounds(box,lower,i)
        lo = min(lo,(y[i]-upper_i)/w)
        hi = max(hi,(y[i]-lower_i)/w)
    end
    if rho == lower || rho == 1
        fill!(chi,rho)
        return (shift=rho == lower ? hi : lo,iterations=0,error=0.0)
    end
    shift = clamp(0.0,lo,hi)
    # The piecewise-linear mass equation is solved exactly by Newton whenever
    # its active set is unchanged. A bracket makes bound crossings reliable.
    for iteration in 0:100
        mass,free_volume = projected_mass!(chi,y,area,shift,lower,weights,box)
        volume = mass/total
        abs(volume-rho) <= tolerance && return (shift=shift,iterations=iteration,error=volume-rho)
        volume > rho ? (lo=shift) : (hi=shift)
        candidate = free_volume > 0 ? shift+(mass-rho*total)/free_volume : NaN
        shift = isfinite(candidate) && lo < candidate < hi ? candidate : lo/2+hi/2
    end
    error("Volume projection failed: error=$(dot(area,chi)/total-rho)",box === nothing ? "" :
        " (the move limit may make the volume unreachable in one step)")
end

"""Evaluate projected mass and its active-set derivative in one cell pass."""
function projected_mass!(chi,y,area,shift,lower,weights=nothing,box=nothing)
    mass = free_volume = 0.0
    @inbounds for i in eachindex(chi,y,area)
        w = volume_weight(weights,i)
        lower_i, upper_i = cell_bounds(box,lower,i)
        trial = y[i]-shift*w
        value = clamp(trial,lower_i,upper_i)
        chi[i] = value
        mass += area[i]*value
        lower_i < trial < upper_i && (free_volume += area[i]*w)
    end
    return mass,free_volume
end

"""Explicit `scaling=:local_lambda`/`:local_free` projection with the current |lambda| in the boost.

With the normalized multiplier l, q = p/scale, a = dt/eta0 and r the frozen peak
boost (`cache.beta_relative`, see `update_coefficients!`), the boost is
b_i(l) = max(r_i, |l|/2) and

    chi_i(l) = clamp(z_i + a*(q_i - l)/b_i(l), lower, 1),

where z = trial - a*q/r is the explicit diffusion part. b is evaluated inside
the bisection, so the accepted state is the fixed point of boost and multiplier.
For q >= 0 the volume is continuous and decreasing in l >= 0, and the pull
toward void saturates at 2a: the bracket is therefore grown from l = 0 by
doubling in the direction of the volume excess instead of being taken from the
all-at-bound shifts. With signed q or l < 0
monotonicity is not guaranteed; bisection keeps a sign change and an
unreachable target is an error. The final boost
max(r_i, |l|/2) is written back to `cache.beta_relative`. The returned shift a*l
matches the standard projection, so diagnostics read the same lambda.

With secant damping, `damping = 1/(scale*eta)` > 0, the viscosity eta*b_i(l) + D_i/scale
replaces eta*b_i(l) and the diffusion no longer cancels from the boost:

    chi_i(l) = clamp(old_i + a*(q_i - l - b_i(l)*(K old)_i/area_i)/(b_i(l) + D_i*damping), lower, 1),

with K*old taken from `cache.rhs` (explicit trial; K*old - G with the obstacle term,
see `add_obstacle!`). `damping = 0` is the undamped form above.
A move-limit `box` (see `move_box!`) replaces [lower, 1] in every clamp; the
bracket search is unchanged.

`push_boost` (`cache.push_boost` for `:local_free`, `nothing` otherwise) is the boost
without floor of a cell that gains material: b_i(l) = max(push_i, |l|/2) where q_i > l,
and max(r_i, |l|/2) as above otherwise. The force vanishes at the switch q_i = l, so
the volume stays continuous and nonincreasing in l >= 0 for q >= 0. This holds without secant
damping only: in the damped form b_i(l) also multiplies the diffusion term, so chi_i(l) jumps
at q_i = l where push_i != r_i, and the bisection can end on such a jump with an error.
"""
function project_volume_floored!(chi,cache,driving,scale,a,rho,lower;tolerance=1e-10,damping=0.0,box=nothing,
        push_boost=nothing)
    0 <= lower < 1 && lower <= rho <= 1 || throw(ArgumentError("Require 0 <= lower < 1 and lower <= rho <= 1"))
    tolerance > 0 || throw(ArgumentError("Projection tolerance must be positive"))
    trial, relative, area = cache.trial, cache.beta_relative, cache.faces.area
    all(isfinite, trial) || throw(ArgumentError("Nonfinite density before projection"))
    # A box end fixes the design, as in `project_volume!`; the multiplier is then
    # not unique and is reported as 0 with the frozen peak boost.
    if rho == lower || rho == 1
        fill!(chi,rho)
        return (shift=0.0,iterations=0,error=0.0)
    end
    total = sum(area)
    # Captures only variables that are never reassigned here, so nothing is boxed.
    volume_at!(l) = (damping > 0 ?
        damped_floored_volume!(chi,cache,driving,scale,a,lower,l,damping,box,push_boost) :
        floored_volume!(chi,trial,relative,driving,area,scale,a,lower,l,box,push_boost))/total
    volume = volume_at!(0.0)
    abs(volume-rho) <= tolerance && return (shift=0.0,iterations=0,error=volume-rho)
    # Grow the bracket away from l = 0 until the volume crosses rho.
    direction = volume > rho ? 1.0 : -1.0
    near, far = 0.0, direction
    iterations = 0
    while (volume_at!(far) > rho) == (direction > 0)
        iterations += 1
        iterations <= 200 || error("Floored volume projection: no bracket for rho=$rho ",
            "(volume target unreachable in one step, each cell moves at most 2*dt/eta0",
            box === nothing ? ")" : "; the move limit may make the volume unreachable in one step)")
        near, far = far, 2far
    end
    # Invariant: volume(lo) > rho >= volume(hi), also without monotonicity.
    lo, hi = minmax(near,far)
    l = far
    for iteration in 1:200
        l = lo+(hi-lo)/2
        volume = volume_at!(l)
        iterations += 1
        abs(volume-rho) <= tolerance && break
        hi-lo <= 4eps(max(abs(lo),abs(hi))) && break
        volume > rho ? (lo=l) : (hi=l)
    end
    volume = volume_at!(l)
    abs(volume-rho) <= tolerance || error("Floored volume projection failed: error=$(volume-rho)",
        box === nothing ? "" : " (the move limit may make the volume unreachable in one step)")
    boost_floor = LOCAL_MAX_BOOST_SCALE*abs(l)
    @inbounds for i in eachindex(relative,driving)
        relative[i] = floored_boost(relative,push_boost,i,boost_floor,driving[i]/scale,l)
    end
    return (shift=a*l,iterations=iterations,error=volume-rho)
end

# Boost b_i(l) = max(r_i, |l|/2); with a push boost (`:local_free`) max(push_i, |l|/2) where q_i > l.
@inline floored_boost(relative,::Nothing,i,boost_floor,q,l) = max(relative[i],boost_floor)
@inline floored_boost(relative,push,i,boost_floor,q,l) = max(q > l ? push[i] : relative[i],boost_floor)

"""Mass of chi_i(l) = clamp(z_i + a*(q_i - l)/b_i(l), lower, 1) (`floored_boost`), written into chi."""
function floored_volume!(chi,trial,relative,driving,area,scale,a,lower,l,box=nothing,push_boost=nothing)
    boost_floor = LOCAL_MAX_BOOST_SCALE*abs(l)
    mass = 0.0
    @inbounds for i in eachindex(chi,trial,relative,driving,area)
        lower_i, upper_i = cell_bounds(box,lower,i)
        q = driving[i]/scale
        z = trial[i]-a*q/relative[i]
        chi[i] = clamp(z+a*(q-l)/floored_boost(relative,push_boost,i,boost_floor,q,l),lower_i,upper_i)
        mass += area[i]*chi[i]
    end
    return mass
end

"""Mass of the damped floored update (see `project_volume_floored!`), written into chi."""
function damped_floored_volume!(chi,cache,driving,scale,a,lower,l,damping,box=nothing,push_boost=nothing)
    boost_floor = LOCAL_MAX_BOOST_SCALE*abs(l)
    old, Kold, relative, D, area = cache.old, cache.rhs, cache.beta_relative, cache.damping, cache.faces.area
    mass = 0.0
    @inbounds for i in eachindex(chi,old,Kold,relative,D,driving,area)
        lower_i, upper_i = cell_bounds(box,lower,i)
        q = driving[i]/scale
        b = floored_boost(relative,push_boost,i,boost_floor,q,l)
        chi[i] = clamp(old[i]+a*(q-l-b*(Kold[i]/area[i]))/(b+D[i]*damping),lower_i,upper_i)
        mass += area[i]*chi[i]
    end
    return mass
end

"""Grayness index and actual area fractions, with theta=(chi-lower)/(1-lower)."""
function grayness_metrics(chi,area,lower)
    total = sum(area)
    mean_theta = grayness = near_void = near_solid = 0.0
    for i in eachindex(chi,area)
        theta = (chi[i]-lower)/(1-lower)
        mean_theta += area[i]*theta
        grayness += area[i]*4theta*(1-theta)
        theta <= 0.05 && (near_void += area[i])
        theta >= 0.95 && (near_solid += area[i])
    end
    mean_theta /= total; grayness /= total
    near_void /= total; near_solid /= total
    uniform_grayness = 4mean_theta*(1-mean_theta)
    return (grayness=grayness,
        grayness_uniform_ratio=uniform_grayness>eps(Float64) ? grayness/uniform_grayness : 0.0,
        intermediate_area_fraction_05_95=1-near_void-near_solid,
        near_void_area_fraction_05=near_void,near_solid_area_fraction_95=near_solid)
end

function validate_density_options(options::DensityOptions, eta0)
    (;regularization,scaling,integrator) = options
    regularization in (:mean,:local_max,:local_lambda,:local_free) && throw(ArgumentError(
        "regularization=:$regularization was removed; use regularization=:gradient, scaling=:$regularization"))
    regularization in (:gradient,:helmholtz) || throw(ArgumentError(
        "Use regularization=:gradient or :helmholtz"))
    scaling in (:mean,:local_max,:local_lambda,:local_free) || throw(ArgumentError(
        "Use scaling=:mean, :local_max, :local_lambda, or :local_free"))
    integrator in (:explicit,:semi_implicit) || throw(ArgumentError(
        "Unknown density integrator $integrator; use :explicit or :semi_implicit"))
    scaling in (:local_lambda,:local_free) && integrator != :explicit &&
        throw(ArgumentError("$scaling needs the current multiplier in the boost; select integrator=:explicit"))
    if regularization == :helmholtz
        integrator == :explicit || throw(ArgumentError(
            "Helmholtz uses an explicit design step and elliptic filter solves; select integrator=:explicit"))
        scaling == :mean || throw(ArgumentError("Helmholtz has no local boost; select scaling=:mean"))
        options.diag && throw(ArgumentError("Secant damping (diag) is implemented for the gradient penalty only"))
    end
    options.move_limit > 0 || throw(ArgumentError("move_limit must be positive (Inf disables it)"))
    isfinite(options.simp_exponent) && options.simp_exponent > 0 ||
        throw(ArgumentError("simp_exponent must be finite and positive"))
    isfinite(eta0) && eta0 > 0 || throw(ArgumentError("eta0 must be finite and positive"))
    options.substeps >= 0 || throw(ArgumentError("Invalid density substeps"))
    options.max_substeps > 0 || throw(ArgumentError("Invalid maximum substeps"))
    isfinite(options.filter_tolerance) && 0 < options.filter_tolerance < 1 ||
        throw(ArgumentError("Invalid filter solve tolerance"))
    isfinite(options.implicit_tolerance) && 0 < options.implicit_tolerance < 1 ||
        throw(ArgumentError("Invalid implicit solve tolerance"))
    options.implicit_max_iterations > 0 || throw(ArgumentError("Invalid implicit iteration limit"))
    isfinite(options.volume_tolerance) && 0 < options.volume_tolerance < 1 ||
        throw(ArgumentError("Invalid volume tolerance"))
    isfinite(options.activity_tolerance) && 0 <= options.activity_tolerance <= 1 ||
        throw(ArgumentError("Density activity tolerance must lie in [0, 1]"))
end
