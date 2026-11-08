# Adapted from csellmann11/NewtonKrylovPlasticity, branch small_strain_plasticity,
# src/core/density_projection.jl at commit 1a6c8e47be785b18d7b3e3216e03f7701da18d6e.
# The pullback below uses GTTO area-metric
# gradients (force per unit area), rather than integrated cell derivatives.

"""
Settings for the optional smooth density projection. Only `enabled` is exposed
by the runners; edit these defaults to change the continuation policy.

The beta update follows Dunning & Wein (2025), Eqs. (6)–(7),
https://doi.org/10.1007/s00158-025-03968-2. Absolute objective magnitude makes
the progress ratio valid for either sign (our minimized strain energy is positive).
The threshold is solved for volume preservation instead of fixing eta=0.5:
the existing design box/volume correction therefore also enforces physical volume.
"""
Base.@kwdef struct DensityProjectionOptions
    enabled                  ::Bool    = false
    beta_initial             ::Float64 = 1.0
    growth_factor            ::Float64 = 1e-3
    max_growth_rate          ::Float64 = 0.5
    gray_tolerance           ::Float64 = 0.01
    volume_tolerance         ::Float64 = 1e-12
    max_threshold_iterations ::Int     = 80
end

function validate_density_projection_options(o::DensityProjectionOptions)
    for v in (o.beta_initial, o.growth_factor, o.max_growth_rate)
        isfinite(v) && v > 0 || throw(ArgumentError("projection growth settings must be finite and positive"))
    end
    isfinite(o.gray_tolerance) && 0 < o.gray_tolerance < 1 ||
        throw(ArgumentError("projection gray_tolerance must lie in (0, 1)"))
    isfinite(o.volume_tolerance) && 0 < o.volume_tolerance < 1 ||
        throw(ArgumentError("projection volume_tolerance must lie in (0, 1)"))
    o.max_threshold_iterations > 0 ||
        throw(ArgumentError("projection max_threshold_iterations must be positive"))
    return o
end

"""Run-local projection state; mesh-sized derivative buffers are reused.

For t=(rho-rho_min)/(1-rho_min), the physical density is
rho_min+(1-rho_min) P(t; beta, eta). Eta satisfies sum(V*P)=sum(V*t).
The floor normalization preserves both material endpoints and volume.
`nothing` is the exact identity, including its pullback and continuation.
"""
mutable struct DensityProjection
    options              ::DensityProjectionOptions
    weights              ::Vector{Float64}  # normalized cell volumes
    rho_min              ::Float64
    beta                 ::Float64
    eta                  ::Float64
    derivative           ::Vector{Float64}  # partial P / partial t
    threshold_derivative ::Vector{Float64}  # partial P / partial eta, common scale removed
    grayness             ::Float64
    previous_objective   ::Float64
    beta_history         ::Vector{Float64}  # beta used for each completed iteration
    gray_history         ::Vector{Float64}  # final candidate, before beta advances
end

function DensityProjection(o::DensityProjectionOptions, V::Vector{Float64}; rho_min::Float64=0.0)
    validate_density_projection_options(o)
    o.enabled || return nothing
    isfinite(rho_min) && 0 <= rho_min < 1 ||
        throw(ArgumentError("projection rho_min must lie in [0, 1)"))
    !isempty(V) && all(v -> isfinite(v) && v > 0, V) && isfinite(sum(V)) ||
        throw(ArgumentError("projection needs finite positive cell volumes"))
    return DensityProjection(o, V ./ sum(V), rho_min, o.beta_initial, 0.5,
                             ones(length(V)), zeros(length(V)), NaN, NaN,
                             Float64[], Float64[])
end

"""Update cell weights and reuse derivative buffers after an adaptive mesh change.

The continuation state (beta, previous objective and scalar histories) is retained.
Call projection_apply! with the transferred field before using its pullback again.
"""
projection_rebuild!(::Nothing, area) = nothing
function projection_rebuild!(p::DensityProjection, area::AbstractVector{<:Real})
    isempty(area) && throw(ArgumentError("projection needs a nonempty mesh"))
    total = 0.0
    @inbounds for value in area
        isfinite(value) && value > 0 ||
            throw(ArgumentError("projection needs finite positive cell areas"))
        total += value
    end
    isfinite(total) || throw(ArgumentError("projection cell area sum must be finite"))
    resize!(p.weights, length(area))
    resize!(p.derivative, length(area))
    resize!(p.threshold_derivative, length(area))
    @inbounds for i in eachindex(area)
        p.weights[i] = area[i] / total
        p.derivative[i] = NaN
        p.threshold_derivative[i] = NaN
    end
    p.grayness = NaN
    return p
end

# Stable hyperbolic forms retain derivatives even when tanh rounds to +/-1.
@inline _projection_logcosh(x) = abs(x) + log1p(exp(-2abs(x))) - log(2.0)
@inline _projection_logsinh(x) = x + log(-expm1(-2x)) - log(2.0)

@inline _projection_shape(beta, eta, log_sinh_beta) =
    (; beta, eta, log_sinh_beta, log_beta=log(beta),
       log_cosh_eta=_projection_logcosh(beta * eta),
       log_cosh_one_eta=_projection_logcosh(beta * (1 - eta)))

@inline function _projection_point(t, shape)
    (; beta, eta, log_sinh_beta, log_beta, log_cosh_eta, log_cosh_one_eta) = shape
    c = _projection_logcosh(beta * (t - eta))
    d = exp(log_beta + log_cosh_eta + log_cosh_one_eta - log_sinh_beta - 2c)
    t == 0 && return (0.0, d, -Inf)
    t == 1 && return (1.0, d, -Inf)
    a = _projection_logsinh(beta * t)
    value = clamp(exp(a - log_sinh_beta + log_cosh_one_eta - c), 0.0, 1.0)
    log_minus_e = log_beta + a + _projection_logsinh(beta * (1 - t)) - log_sinh_beta - 2c
    return (value, d, log_minus_e)
end

projection_apply!(out, ::Nothing, input) = (out === input || copyto!(out, input); out)
projection_pullback!(gradient, ::Nothing) = gradient
projection_advance!(::Nothing, objective; reset=false, allow_advance=true) = false
projection_converged(::Nothing) = true
projection_converged(p::DensityProjection) = p.grayness <= p.options.gray_tolerance

"""Apply projection and cache its derivatives. `out === input` is supported.

Each threshold trial reads the filtered field, with no filter or FE solves.
At eta=0 projection increases every interior density; at eta=1 it decreases
every interior density. This brackets the volume-preserving threshold.
"""
function projection_apply!(out::Vector{Float64}, p::DensityProjection, input::Vector{Float64})
    length(out) == length(input) == length(p.weights) ||
        throw(DimensionMismatch("projection field and cell volumes differ"))
    target,interior = projection_target(p,input)
    interior || return project_binary_field!(out,p,input)
    isfinite(p.beta) && p.beta > 0 || throw(ArgumentError("projection beta must be finite and positive"))
    log_sinh_beta = _projection_logsinh(p.beta)
    p.eta = solve_projection_threshold(p,input,target,log_sinh_beta)
    cache_projected_field!(out,p,input,_projection_shape(p.beta,p.eta,log_sinh_beta))
    return out
end

"""Validate the input and obtain the normalized physical-volume target."""
function projection_target(p,input)
    lo,width = p.rho_min,1-p.rho_min
    target,interior = 0.0,false
    @inbounds for i in eachindex(input)
        x = input[i]
        isfinite(x) && lo-1e-12 <= x <= 1+1e-12 ||
            throw(ArgumentError("projection density must lie in [rho_min, 1]"))
        t = clamp((x-lo)/width,0.0,1.0)
        target += p.weights[i]*t
        interior |= 0 < t < 1
    end
    return target,interior
end

"""Binary fields have a nonunique threshold; retain the identity tangent convention."""
function project_binary_field!(out,p,input)
    @inbounds for i in eachindex(input)
        out[i] = clamp(input[i],p.rho_min,1.0)
    end
    fill!(p.derivative,1.0)
    fill!(p.threshold_derivative,0.0)
    p.eta,p.grayness = 0.5,0.0
    return out
end

function projection_mass_slope(p,input,shape)
    lo,width = p.rho_min,1-p.rho_min
    mass,slope = 0.0,0.0
    @inbounds for i in eachindex(input)
        t = clamp((input[i]-lo)/width,0.0,1.0)
        value,_,log_minus_e = _projection_point(t,shape)
        mass += p.weights[i]*value
        slope -= p.weights[i]*exp(log_minus_e)
    end
    return mass,slope
end

"""Safeguarded scalar Newton solve; rejected steps bisect the volume bracket."""
function solve_projection_threshold(p,input,target,log_sinh_beta)
    left,right = 0.0,1.0
    eta = clamp(p.eta,left,right)
    for iteration in 1:p.options.max_threshold_iterations
        shape = _projection_shape(p.beta,eta,log_sinh_beta)
        mass,slope = projection_mass_slope(p,input,shape)
        residual = mass-target
        abs(residual) <= p.options.volume_tolerance && return eta
        residual > 0 ? (left=eta) : (right=eta)
        candidate = eta-residual/slope
        eta = isfinite(candidate) && left < candidate < right ? candidate : (left+right)/2
    end
    error("density projection threshold did not reach the volume tolerance")
end

"""Evaluate the physical field, derivatives and grayness in one allocation-free pass."""
function cache_projected_field!(out,p,input,shape)
    lo,width = p.rho_min,1-p.rho_min
    gray,max_log_e = 0.0,-Inf
    @inbounds for i in eachindex(input)
        t = clamp((input[i]-lo)/width,0.0,1.0)
        value,derivative,log_minus_e = _projection_point(t,shape)
        out[i] = clamp(lo+width*value,lo,1.0)
        p.derivative[i] = derivative
        p.threshold_derivative[i] = log_minus_e
        max_log_e = max(max_log_e,log_minus_e)
        gray += p.weights[i]*4value*(1-value)
    end
    normalize_threshold_derivative!(p,max_log_e)
    p.grayness = gray
    return nothing
end

"""Remove a common logarithmic scale: the implicit pullback uses only derivative ratios."""
function normalize_threshold_derivative!(p,max_log_e)
    isfinite(max_log_e) || error("density projection threshold derivative is numerically unresolved")
    @inbounds for i in eachindex(p.threshold_derivative)
        p.threshold_derivative[i] = -exp(p.threshold_derivative[i]-max_log_e)
    end
    return nothing
end

"""Exact implicit-threshold chain rule, in place, before the filter transpose.

The input is a gradient per unit area: delta J = sum(area .* gradient .* delta rho).
With d=P_t and e=P_eta (a common scaling cancels), differentiating
sum(w*P)=sum(w*t) gives the area-metric pullback
    g_i <- d_i*g_i + (1-d_i)*sum(w .* g .* e)/sum(w .* e).
The reference code accepts integrated cell derivatives and therefore has an
additional outer cell weight. Do not copy that factor into this area-metric form.
The rank-one term differentiates the volume-preserving threshold; in particular
the physical volume gradient (the constant one field) pulls back to itself.
"""
function projection_pullback!(gradient::Vector{Float64}, p::DensityProjection)
    length(gradient) == length(p.weights) || throw(DimensionMismatch("projection gradient size"))
    numerator, denominator = 0.0, 0.0
    @inbounds for i in eachindex(gradient)
        e = p.threshold_derivative[i]
        numerator += p.weights[i] * gradient[i] * e
        denominator += p.weights[i] * e
    end
    isfinite(denominator) || error("density projection derivatives need a forward refresh")
    if denominator == 0
        # Explicit saturated identity convention (see forward); reject a
        # numerically unresolved interior threshold instead of a false gradient.
        all(==(1.0), p.derivative) && return gradient
        error("density projection threshold derivative is numerically unresolved")
    end
    correction = numerator / denominator
    @inbounds for i in eachindex(gradient)
        d = p.derivative[i]
        gradient[i] = d * gradient[i] + (1 - d) * correction
    end
    return gradient
end

"""Pull a physical driving force back into a reusable design-space buffer.
The identity mapping may alias its input; an active projection uses separate storage.
"""
function projected_driving!(output,input,projection)
    output !== input && copyto!(output,input)
    return projection_pullback!(output,projection)
end

"""Observe one completed outer iteration and prepare beta for the next one.

Negative objectives still improve when they DECREASE. Equal nonzero values
use the capped limit of the paper's rule; a zero scale or nonfinite value
cannot supply a progress ratio. SIMP continuation clears the baseline so
objectives from different penalization laws are not compared.
"""
function projection_advance!(p::DensityProjection, objective::Real; reset=false, allow_advance=true)
    push!(p.beta_history, p.beta)
    push!(p.gray_history, p.grayness)
    previous = p.previous_objective
    current = Float64(objective)
    p.previous_objective = reset || !isfinite(current) ? NaN : current
    (reset || !allow_advance || !isfinite(previous) || !isfinite(current) || projection_converged(p)) && return false
    improvement = previous - current
    scale = abs(previous) / 2 + abs(current) / 2
    (improvement < 0 || scale == 0) && return false
    cap = p.options.max_growth_rate * p.beta
    increment = improvement == 0 ? cap : min(p.options.growth_factor * (scale / improvement), cap)
    next = p.beta + increment
    isfinite(next) || error("density projection beta overflow")
    next > p.beta || return false
    p.beta = next
    return true
end
