# Double-obstacle term of the gradient penalty: the sharpening of the gradient-penalty
# flow (it has no density projection), its imbalance measure and its continuation.

"""Add `factor`*G to `target`, with G the integrated force of the double-obstacle term.

The regularization energy of a cell gets the concave part gamma_K/2*(chi-lower)*(1-chi)
next to the gradient penalty (double-obstacle potential of Modica-Mortola type). Its
force per unit volume, positive = grow, is gamma_K*(chi_K - c) with c = (1+lower)/2: it
pushes a cell away from the middle toward the bound it is closer to, where the box of
the volume projection holds it. The weight is tied to the regularization,

    gamma_K = g*kappa_K,   kappa_K = beta_K/h_K^2,   h_K = cbrt(|K|),
    G_K = |K|*gamma_K*(chi_K - c) = g*beta_K*h_K*(chi_K - c),

with g = `cache.obstacle`. kappa_K is the normalized diffusion per face of a uniform
grid (t_f/|K| = 2*beta0 on cubes for beta_K = 2*beta0*h_K^2), so g is mesh independent:
a planar interface has the continuum width pi*h/sqrt(g), and a mesh-aligned one becomes
binary at g = 2 (its lateral face neighbors carry no flux, as in 2D). The term is
boosted together with beta (`:local_max`), so like the diffusion it is not divided by
the boost.

Explicit in both integrators: the term is anti-diffusive, and an implicit negative
reaction would amplify without bound as dt/eta0*gamma approaches 1. g = 0 adds nothing.
"""
function add_obstacle!(target,cache,chi,lower,factor)
    g = cache.obstacle
    g == 0 && return target
    beta, volume, center = cache.base_beta, cache.faces.area, (1+lower)/2
    @inbounds for i in eachindex(target,chi,beta,volume)
        target[i] += factor*g*beta[i]*cbrt(volume[i])*(chi[i]-center)
    end
    return target
end

"""Obstacle force per unit volume g*kappa_i*(x_i - c) of cell i (see `add_obstacle!`)."""
@inline obstacle_force(cache,x,lower,i) =
    cache.obstacle*cache.base_beta[i]/cbrt(cache.faces.area[i])^2*(x[i]-(1+lower)/2)

"""Imbalance of the substep just taken, written to `cache.balance`.

On the free cells (strictly inside [lower, 1] and inside the move-limit box) the
normalized update has the terms, with q = p/scale, l = shift/a the normalized volume
multiplier (a = dt/eta0) and b the boost of the accepted state,

    F_i = (q_i - l)/b_i,   D_i = -(K chi)_i/|K_i|,   g*U_i,   U_i = kappa_i*(old_i - c),

(chi = old for `:explicit`, the new state for `:semi_implicit`), and the net term the
step actually took, N_i = (chi_i - old_i)*(1+r_i)/a with r_i = D_i/(scale*eta0*b_i) the
damping ratio (0 without damping). For the explicit step N = F + D + g*U on the free
cells. With volume-weighted L2 norms over the free cells,

    imbalance = |N| / (|F| + |D| + g*|U|),

1 = nothing balanced, 0 = equilibrium; independent of eta0, dt and the force scale.
`obstacle_unit` = |U| is the change of the net term per unit increase of g. Without
free cells the imbalance is NaN and |U| = 0. Uses `cache.volume_weight` as scratch:
the volume projection that reads it is done, and the diagnostics still need `rhs`.
"""
function measure_balance!(cache,chi,driving,scale,dt,eta0,shift,lower,box,options)
    area, old, boost, beta = cache.faces.area, cache.old, cache.beta_relative, cache.base_beta
    kchi = cache.volume_weight
    mul!(kchi,cache.stiffness,options.integrator == :semi_implicit ? chi : old)
    a = dt/eta0
    l = shift/a
    damped = cache.damping_active
    g, center = cache.obstacle, (1+lower)/2
    total = free = net2 = f2 = d2 = u2 = 0.0
    @inbounds for i in eachindex(chi,old,area)
        total += area[i]
        lower < chi[i] < 1 || continue
        box === nothing || box.lower[i] < chi[i] < box.upper[i] || continue
        # `:local_lambda` raises the boost in its projection; r uses the final one.
        r = damped ? cache.damping[i]/(scale*eta0*boost[i]) : 0.0
        net = (chi[i]-old[i])*(1+r)/a
        force = (driving[i]/scale-l)/boost[i]
        diffusion = -kchi[i]/area[i]
        unit = beta[i]/cbrt(area[i])^2*(old[i]-center)
        free += area[i]
        net2 += area[i]*net^2; f2 += area[i]*force^2
        d2 += area[i]*diffusion^2; u2 += area[i]*unit^2
    end
    F, D, U = sqrt(f2), sqrt(d2), sqrt(u2)
    gross = F+D+g*U
    cache.balance = (imbalance=gross > 0 ? sqrt(net2)/gross : NaN,force=F,diffusion=D,
        obstacle_unit=U,free_fraction=free/total)
    return cache.balance
end

"""
Continuation of the double-obstacle weight g (`cache.obstacle`) by the imbalance budget.

g starts at zero, so the structure forms under the plain gradient penalty. With the
imbalance e of the last substep and the norms |F|, |D|, |U| over its free cells
(`measure_balance!`), an increase dg adds dg*U to the net term. The largest increase
that keeps the imbalance below e_tol = `imbalance_tolerance` is, to first order,

    dg = max(0, e_tol - e) * (|F| + |D| + g*|U|) / |U|,

capped at `max_weight`. g holds while the design is out of balance (e >= e_tol), in
particular while cells run to a bound after the last increase. It stops for good once
the grayness of the design is at most `gray_tolerance`, and does not grow without free
cells (|U| = 0: a larger g would change nothing).
"""
mutable struct ObstacleContinuation
    weight::Float64
    max_weight::Float64
    imbalance_tolerance::Float64
    gray_tolerance::Float64
    grayness::Float64
end

function ObstacleContinuation(;max_weight=Inf,imbalance_tolerance=0.1,gray_tolerance=0.01)
    !isnan(max_weight) && max_weight > 0 ||
        throw(ArgumentError("The obstacle weight cap must be positive (Inf: no cap)"))
    0 < imbalance_tolerance < 1 && 0 < gray_tolerance < 1 ||
        throw(ArgumentError("Obstacle imbalance and gray tolerances must lie in (0, 1)"))
    return ObstacleContinuation(0.0,max_weight,imbalance_tolerance,gray_tolerance,NaN)
end

obstacle_converged(::Nothing) = true
obstacle_converged(o::ObstacleContinuation) = o.grayness <= o.gray_tolerance || o.weight >= o.max_weight

"""Observe one outer iteration (`balance` of its last substep, grayness of the design)
and set the weight of the next one. Returns whether the weight grew."""
function obstacle_advance!(o::ObstacleContinuation,balance,grayness;allow_advance=true)
    o.grayness = grayness
    (!allow_advance || obstacle_converged(o)) && return false
    (balance.obstacle_unit > 0 && isfinite(balance.imbalance)) || return false
    budget = max(0.0,o.imbalance_tolerance-balance.imbalance)
    step = budget*(balance.force+balance.diffusion+o.weight*balance.obstacle_unit)/balance.obstacle_unit
    next = min(o.max_weight,o.weight+step)
    advanced = next > o.weight
    o.weight = next
    return advanced
end
