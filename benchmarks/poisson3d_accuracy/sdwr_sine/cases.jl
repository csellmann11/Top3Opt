"""Smooth unit-cube eigenmode with homogeneous Dirichlet values."""
function sine_problem(modes::NTuple{3,Int})
    all(>(0),modes) || throw(ArgumentError("positive integer frequencies required"))
    eigenvalue = pi^2*sum(abs2,modes)
    exact(x) = prod(sinpi(modes[d]*x[d]) for d in 1:3)
    gradient(x) = ntuple(d->pi*modes[d]*cospi(modes[d]*x[d])*
        prod(sinpi(modes[j]*x[j]) for j in 1:3 if j != d),3)
    rhs(x) = eigenvalue*exact(x)
    name = Symbol("sine_"*join(modes,"_"))
    return (;name,domain=:cube,modes,eigenvalue,exact,gradient,rhs,
        reference_energy=eigenvalue/16,reference_uncertainty=0.0)
end

function sine_goal(name::Symbol)
    name in (:integral_u,:integral_u2) && return sdwr_goal(name,:cube)
    name == :energy || throw(ArgumentError("unknown sine goal $name"))
    return (;name,derivative_value=(x,u)->0.0,
        derivative_gradient=(x,gradient)->gradient)
end

function sine_reference(problem,name::Symbol)
    name == :integral_u && return prod((1-cospi(k))/(pi*k) for k in problem.modes)
    name == :integral_u2 && return 1/8
    name == :energy && return problem.reference_energy
    throw(ArgumentError("unknown sine goal $name"))
end

"""Independent symbols of the original uniform-cube operator.

The cube's Walsh modes diagonalize K and its fan energy matrix H; H=K-S/2.
These formulas are for validation only: the experiment uses sparse solves.
"""
function sine_discrete_symbols(modes,n)
    h = 1/n
    theta = ntuple(d->pi*modes[d]*h,3)
    t = sin.(theta./2).^2
    c = 1 .- t
    K = 4h*(sum(t)-prod(t))
    S = 8h*(t[1]*t[2]+t[1]*t[3]+t[2]*t[3]-2prod(t))
    H = K-S/2
    M = h^3*(prod(c)+sum(t[d]*prod(c[j] for j in 1:3 if j!=d) for d in 1:3)/3+
        sum(c[d]*prod(t[j] for j in 1:3 if j!=d) for d in 1:3)/6+prod(t)/10)
    A = sin.(theta)./theta
    C = (sin.(theta./2)./(theta./2)).^2
    projected_load = h^3*(sum(C[d]*prod(A[j] for j in 1:3 if j!=d) for d in 1:3)-2prod(A))
    amplitude = pi^2*sum(abs2,modes)*projected_load/K
    return (;K,S,H,M,projected_load,amplitude)
end

"""Leading effectivity for this operator, projected load and fan target.

Let lambda=sum(k_i^2), Q=sum(k_i^4), P=sum_{i<j}(k_i*k_j)^2.
The nodal amplitude is 1-h^2*P/(6lambda)+O(h^4), and the mixed
stabilization symbol is h^5*P/2+O(h^7). Fan moment/energy expansions
then give these limits. They are analytic predictions, not fitted factors.
"""
function sine_effectivity_limit(modes,name::Symbol)
    squares = modes.^2
    Q = sum(abs2,squares)
    P = squares[1]*squares[2]+squares[1]*squares[3]+squares[2]*squares[3]
    name in (:integral_u,:integral_u2) && return 6P/(Q+4P)
    name == :energy && return 9P/(Q+7P)
    throw(ArgumentError("unknown sine goal $name"))
end

"""Reference energy and its load from the original explicit tetrahedral fan."""
function sine_full_energy(mesh,fan,U)
    load = zeros(length(U))
    value = 0.0
    for tet in eachindex(fan.volumes)
        ids = fan.tet_nodes[:,tet]
        G,volume = fan.gradients[tet],fan.volumes[tet]
        gradient = G*U[ids]
        value += volume*dot(gradient,gradient)/2
        load[ids] .+= volume*(transpose(G)*gradient)
    end
    return (;value,load=sdwr_reconstruction_transpose(mesh,fan,load))
end
