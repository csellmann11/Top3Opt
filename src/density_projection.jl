"""
    project_density_volume!(density, trial, volumes, target, lower; tolerance=1e-8)

Find the scalar shift giving `density = clamp.(trial .- shift, lower, 1)`
with the requested volume-weighted mean. This is the explicit Euler multiplier
solve, with the complete driving-plus-regularization force already in `trial`.
A safeguarded Newton step uses the volume of the currently unclipped cells;
bisection handles changes of active set and fully clipped trial states.
"""
function project_density_volume!(density, trial, volumes, target, lower;
        tolerance=1e-8, max_iterations=100)
    lower <= target <= 1 || throw(ArgumentError("Infeasible target density: $target"))
    all(isfinite,trial) || error("Nonfinite unconstrained density update")
    total_volume = sum(volumes)
    isfinite(total_volume) && total_volume > 0 || error("Invalid total design volume")
    if target == lower || target == 1
        fill!(density,target)
        return 0
    end
    smallest,largest = extrema(trial)
    # At lo every cell is solid; at hi every cell is at the lower density bound.
    lo = smallest-1
    hi = largest-lower
    shift = clamp(0.0,lo,hi)
    for iteration in 1:max_iterations
        mass = 0.0
        free_volume = 0.0
        for i in eachindex(density,trial,volumes)
            value = trial[i]-shift
            clipped = clamp(value,lower,1.0)
            density[i] = clipped
            mass += volumes[i]*clipped
            if lower < value < 1
                free_volume += volumes[i]
            end
        end
        residual = mass-target*total_volume
        abs(residual) <= tolerance*total_volume && return iteration
        if residual > 0
            lo = shift
        else
            hi = shift
        end
        candidate = free_volume > 0 ? shift+residual/free_volume : NaN
        shift = isfinite(candidate) && lo < candidate < hi ? candidate : lo+(hi-lo)/2
    end
    error("Density volume projection did not converge in $max_iterations iterations")
end
