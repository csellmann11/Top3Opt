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

"""Area-metric projection: chi_i=clamp(y_i-shift*w_i, lower, 1), at prescribed volume.
The positive `weights` w_i are the response of cell i to a unit volume multiplier;
`nothing` is the uniform shift w_i = 1.
"""
function project_volume!(chi, y, area, rho, lower; tolerance=1e-10, weights=nothing)
    0 <= lower < 1 && lower <= rho <= 1 || throw(ArgumentError("Require 0 <= lower < 1 and lower <= rho <= 1"))
    tolerance > 0 || throw(ArgumentError("Projection tolerance must be positive"))
    all(isfinite, y) || throw(ArgumentError("Nonfinite density before projection"))
    weights === nothing || all(x -> isfinite(x) && x > 0, weights) ||
        throw(ArgumentError("Volume weights must be finite and positive"))
    total = sum(area)
    # Shifts that put every cell at the upper (lo) or lower (hi) bound.
    lo, hi = Inf, -Inf
    for i in eachindex(y)
        w = volume_weight(weights,i)
        lo = min(lo,(y[i]-1.0)/w)
        hi = max(hi,(y[i]-lower)/w)
    end
    if rho == lower || rho == 1
        fill!(chi,rho)
        return (shift=rho == lower ? hi : lo,iterations=0,error=0.0)
    end
    shift = clamp(0.0,lo,hi)
    # The piecewise-linear mass equation is solved exactly by Newton whenever
    # its active set is unchanged. A bracket makes bound crossings reliable.
    for iteration in 0:100
        mass,free_volume = projected_mass!(chi,y,area,shift,lower,weights)
        volume = mass/total
        abs(volume-rho) <= tolerance && return (shift=shift,iterations=iteration,error=volume-rho)
        volume > rho ? (lo=shift) : (hi=shift)
        candidate = free_volume > 0 ? shift+(mass-rho*total)/free_volume : NaN
        shift = isfinite(candidate) && lo < candidate < hi ? candidate : lo/2+hi/2
    end
    error("Volume projection failed: error=$(dot(area,chi)/total-rho)")
end

"""Evaluate projected mass and its active-set derivative in one cell pass."""
function projected_mass!(chi,y,area,shift,lower,weights=nothing)
    mass = free_volume = 0.0
    @inbounds for i in eachindex(chi,y,area)
        w = volume_weight(weights,i)
        trial = y[i]-shift*w
        value = clamp(trial,lower,1.0)
        chi[i] = value
        mass += area[i]*value
        lower < trial < 1.0 && (free_volume += area[i]*w)
    end
    return mass,free_volume
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

function validate_density_options(options::DensityOptions, eta0, legacy_substeps)
    options.integrator in (:explicit,:semi_implicit) || throw(ArgumentError("Unknown density integrator"))
    options.regularization in (:mean,:local_max,:helmholtz) || throw(ArgumentError(
        "Use regularization=:mean, :local_max, or :helmholtz"))
    options.regularization == :helmholtz && options.integrator != :explicit &&
        throw(ArgumentError("Helmholtz uses an explicit design step and elliptic filter solves; select integrator=:explicit"))
    isfinite(eta0) && eta0 > 0 || throw(ArgumentError("eta0 must be finite and positive"))
    options.substeps >= 0 && legacy_substeps > 0 || throw(ArgumentError("Invalid density substeps"))
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
