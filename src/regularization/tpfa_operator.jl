function DensityCache(faces::TPFAFaces, beta0::Union{Real,AbstractVector})
    n = length(faces.area)
    n > 0 || throw(ArgumentError("The density mesh is empty"))
    nf = length(faces.left)
    all(length(v) == nf for v in (faces.right, faces.length,
        faces.distance_left, faces.distance_right)) || throw(DimensionMismatch("Face geometry"))
    for v in (faces.area, faces.length, faces.distance_left, faces.distance_right)
        all(x -> isfinite(x) && x > 0, v) || throw(ArgumentError("Nonpositive/nonfinite TPFA geometry"))
    end
    all(1 <= i <= n && 1 <= j <= n && i != j for (i,j) in zip(faces.left,faces.right)) ||
        throw(ArgumentError("Invalid TPFA face connectivity"))
    # The caller owns the baseline policy (in 3D: 2*beta0*h_i^2, h_i = cbrt(cell volume)).
    b = beta0 isa Real ? fill(Float64(beta0),n) : Float64.(beta0)
    length(b) == n || throw(DimensionMismatch("Baseline regularization coefficient"))
    all(x -> isfinite(x) && x >= 0,b) || throw(ArgumentError("beta0 must be finite and nonnegative"))
    K = diffusion_matrix(faces,b)
    diagonal_indices = [stored_index(K,i,i) for i in 1:n]
    face_indices = [(stored_index(K,faces.left[f],faces.right[f]),
        stored_index(K,faces.right[f],faces.left[f])) for f in 1:nf]
    return DensityCache(faces,b,ones(n),K,diagonal_indices,face_indices,
        nothing,Diagonal(Float64[]),nothing,true,NaN,NaN,false,
        Float64[],Float64[],Float64[],Float64[],false,false,
        zeros(n),zeros(n),zeros(n),zeros(n),Float64[],Float64[],Float64[],zeros(n),falses(n),nothing,(;))
end

"""Harmonic face coefficient, including an insulating zero-coefficient cell."""
@inline tpfa_conductance(measure,dl,dr,bi,bj) =
    bi == 0 || bj == 0 ? 0.0 : measure/(dl/bi+dr/bj)

function stored_index(A, row, col)
    for k in nzrange(A,col)
        rowvals(A)[k] == row && return k
    end
    error("Missing TPFA matrix entry ($row,$col)")
end

"""Positive semidefinite graph stiffness K; the flux operator is -M^-1 K.
Retain every diagonal in the pattern, including isolated cells.
"""
function diffusion_matrix(faces::TPFAFaces, beta::AbstractVector,work=TPFAAssemblyWorkspace())
    n = length(faces.area)
    length(beta) == n || throw(DimensionMismatch("beta"))
    all(x -> isfinite(x) && x >= 0, beta) || throw(ArgumentError("beta must be finite and nonnegative"))
    nf = length(faces.left)
    entries = n+2nf
    rows = resize!(work.rows,entries)
    cols = resize!(work.cols,entries)
    vals = resize!(work.values,entries)
    fill!(vals,0.0)
    for i in 1:n
        rows[i] = cols[i] = i
    end
    for f in 1:nf
        i, j = faces.left[f], faces.right[f]
        t = tpfa_conductance(faces.length[f],faces.distance_left[f],faces.distance_right[f],beta[i],beta[j])
        isfinite(t) && t >= 0 || throw(ArgumentError("Invalid face conductance at face $f"))
        k = n+2*f-1
        rows[k] = i; cols[k] = j; vals[k] = -t
        rows[k+1] = j; cols[k+1] = i; vals[k+1] = -t
        vals[i] += t; vals[j] += t
    end
    resize!(work.last_touch,n)
    resize!(work.csr_rowptr,n+1)
    resize!(work.csr_cols,entries)
    resize!(work.csr_values,entries)
    return SparseArrays.sparse!(rows,cols,vals,n,n,+,work.last_touch,
        work.csr_rowptr,work.csr_cols,work.csr_values,rows,cols,vals)
end

"""Fixed calibration of the `:local_max` boost against `:mean`.

With the boost max(1, m_i) alone, `:local_max` acts like `:mean` with roughly twice
beta0 and eta0, because the boost exceeds one along most interfaces. The factor 1/2
compensates this on average, so both schemes can use the same beta0 and eta0. It is a
constant, not a field statistic: the normalized force (p_i/scale - lambda)/b_i of a cell
with p_i >= lambda is bounded by 1/LOCAL_MAX_BOOST_SCALE = 2 for every force field.
The value was calibrated on 2D benchmarks.

With the baseline coefficient beta_i = 2*beta0*h_i^2 on a uniform grid, a 0->1 density
jump across one face gives the normalized diffusion beta_i/h_i^2 = 2*beta0 in the solid
cell against a normalized force of at most 2, so for beta0 > 1 such an unresolved
interface cannot be stationary.
"""
const LOCAL_MAX_BOOST_SCALE = 0.5

"""Normalized boost floor of `:local_lambda`; only guards a zero force against zero."""
const LOCAL_LAMBDA_GUARD = 1e-12

"""Freeze the local boost for one density substep (`options.scaling`).

For `:local_max`, with the mean force `scale`,

    m_i = max(abs(p[j])) / scale,   j = i or a direct TPFA face neighbor,
    b_i = LOCAL_MAX_BOOST_SCALE * max(1, m_i).

`:local_lambda` drops the mean-force floor 1 and uses the current normalized
volume multiplier l = lambda/scale instead,

    b_i(l) = LOCAL_MAX_BOOST_SCALE * max(m_i, |l|, LOCAL_LAMBDA_GUARD),

so `scale` cancels from the update and |(p_i/scale - l)/b_i| <= 2 for p >= 0
and l >= 0 (<= 4 for signed p or l < 0).
Here only the peak part max(m_i, GUARD) is frozen; the |l| floor needs the
current multiplier and is applied by `project_volume_floored!` (explicit only).

The boost scales the regularization and the viscosity of a cell together,
beta_i = b_i * beta0_i and eta_i = b_i * eta0 (both times `scale`):

    eta0 * b_i * dchi_i/dt = p_i/scale - lambda - gamma_i - b_i * (K chi)_i / area_i

with K the baseline operator. Dividing by b_i,

    eta0 * dchi_i/dt = (p_i/scale - lambda - gamma_i)/b_i - (K chi)_i / area_i,

shows that the boost only divides the driving force and the multipliers of a
cell. The stationary balance is that of a locally boosted beta, but diffusion
keeps the baseline operator: the explicit stability limit and the implicit
system are those of `:mean`, so a single force peak no longer dictates the time
step and the boost needs no cap. Neither the stiffness matrix nor the implicit
system depends on the boost. This is not a differentiated, fixed objective in
p(chi).

The existing face graph supplies neighbors. Read each neighbor's original p
so a single traversal cannot propagate a peak beyond its direct neighbors.
The trial buffer is free until the subsequent density solve/projection.
"""
function update_coefficients!(cache::DensityCache, p, scale, options)
    relative, candidate = cache.beta_relative, cache.trial
    if options.scaling in (:local_max,:local_lambda)
        floor = options.scaling == :local_max ? 1.0 : LOCAL_LAMBDA_GUARD
        for i in eachindex(p)
            candidate[i] = max(floor,abs(p[i])/scale)
        end
        faces = cache.faces
        for f in eachindex(faces.left)
            i, j = faces.left[f], faces.right[f]
            candidate[i] = max(candidate[i],abs(p[j])/scale)
            candidate[j] = max(candidate[j],abs(p[i])/scale)
        end
        candidate .*= LOCAL_MAX_BOOST_SCALE
    else
        # The same cache can switch back from local to uniform regularization.
        all(isone,relative) && return false
        fill!(candidate,1.0)
    end
    candidate == relative && return false
    copyto!(relative,candidate)
    return true
end

"""Secant reaction damping (`diag`): update the measured curvature at an outer entry.

With the design x and the design driving force p (positive = wants to grow,
after the projection pullback) at the entries k-1 and k of two consecutive
outer iterations, dx = x^k - x^(k-1), dp = p^k - p^(k-1), N_i the cell i
and its face neighbors and m = <dx^2> the volume-weighted mean square step over
the whole domain,

    D_new,i = sum_{j in N_i} A_j*max(-dp_j*dx_j, 0) / sum_{j in N_i} A_j*(dx_j^2 + m)
              (0 if the denominator is zero, i.e. nothing moved),
    D_i    <- D_new,i/2 + D_i/2                                  (all cells),

The global term m keeps the fit of a neighborhood that barely moved from
blowing up: its force change then stems mostly from the rest of the design, and
the plain ratio froze gray regions and made the design mesh dependent. A
neighborhood of rms step theta*sqrt(m) keeps the fraction theta^2/(theta^2 + 1)
of its fit, half at the mean step.

with D = 0 without history. Without the inner clamp D_new is the least-squares
solution of the secant equation D*dx_j = -dp_j over N_i, the curvature along the
step measured on the neighborhood. A per-cell secant -dp_i/dx_i also contains
the neighbors' influence on p_i and is unbounded for a cell that barely moved;
neighboring cells then get very different viscosities and a smooth force
produces a cell-scale pattern. The weights dx_j^2 give a cell that dominates
the movement of its neighborhood its own secant, and a cell that barely moved
the one of its neighbors. Each term is clamped at zero before the sum: damping
is only meant for a force that falls as the cell grows, and a cell whose force
rises with its step would otherwise cancel the damping of its neighborhood.
A neighbor sharing several faces with cell i counts once per face.

D_i >= 0 estimates how fast the force falls when the neighborhood of cell i
grows. It enters as an additional per-cell viscosity over the full
outer step T = 1: the viscosity eta*b_i becomes eta*b_i + D_i/s (s the force
scale, q = p/s), i.e. with r_i = D_i/(s*eta*b_i)

    eta*b_i*(1 + r_i) * dx_i/dt = q_i - lambda - b_i*(K x)_i/area_i.

T, not the substep length, multiplies D because the force is frozen over the
sweep; the substep length would dilute the damping to D/n. A cell that would
overshoot a quadratic force maximum (two-cycle for dt*k/eta > 2) is thereby
moved by the secant (Newton) step at most. The right-hand side is unchanged, so
stationary points depend neither on D nor on eta.
"""
function update_secant_damping!(cache::DensityCache,x,p)
    n = length(x)
    if length(cache.damping) != n
        for v in (cache.damping,cache.damping_ratio,cache.secant_design,cache.secant_force)
            resize!(v,n)
        end
        cache.secant_valid = false
    end
    active = false
    if cache.secant_valid
        D, x0, p0 = cache.damping, cache.secant_design, cache.secant_force
        # Secant over the cell and its face neighbors: numerator
        # sum A_j*max(-dp_j*dx_j,0), denominator sum A_j*(dx_j^2 + m). Both buffers
        # are free until the trial of this substep.
        faces, numerator, denominator = cache.faces, cache.trial, cache.rhs
        area = faces.area
        step2 = total = 0.0
        @inbounds for i in eachindex(x,x0,area)
            dx = x[i]-x0[i]
            step2 += area[i]*dx*dx
            total += area[i]
        end
        m = step2/total
        @inbounds for i in eachindex(x,p,x0,p0,numerator,denominator)
            dx = x[i]-x0[i]
            numerator[i] = area[i]*max(-(p[i]-p0[i])*dx,0.0)
            denominator[i] = area[i]*(dx*dx+m)
        end
        # Each face once, both directions; x0/p0 still hold the previous entry.
        @inbounds for f in eachindex(faces.left)
            i, j = faces.left[f], faces.right[f]
            dxi, dxj = x[i]-x0[i], x[j]-x0[j]
            ni = area[i]*max(-(p[i]-p0[i])*dxi,0.0)
            nj = area[j]*max(-(p[j]-p0[j])*dxj,0.0)
            numerator[i] += nj; denominator[i] += area[j]*(dxj*dxj+m)
            numerator[j] += ni; denominator[j] += area[i]*(dxi*dxi+m)
        end
        @inbounds for i in eachindex(D,numerator,denominator)
            measured = denominator[i] > 0 ? numerator[i]/denominator[i] : 0.0
            D[i] = 0.5measured+0.5D[i]
            active |= D[i] > 0
        end
    else
        fill!(cache.damping,0.0)
    end
    copyto!(cache.secant_design,x)
    copyto!(cache.secant_force,p)
    cache.secant_valid, cache.damping_active = true, active
    return cache
end

"""Forget the secant history (projection beta step); D = 0 until two new entries."""
function reset_secant!(cache::DensityCache)
    fill!(cache.damping,0.0)
    cache.secant_valid = cache.damping_active = false
    return cache
end
reset_secant!(cache) = cache

"""Per-cell damping ratio r_i = D_i/(s*eta*b_i) of the current substep."""
damping_ratio!(cache::DensityCache,scale,eta) =
    (@. cache.damping_ratio = cache.damping/(scale*eta*cache.beta_relative); cache.damping_ratio)

"""Refresh M+(dt/eta0)K and its Jacobi preconditioner only when needed.
CG uses a reusable workspace, avoiding fill-in from a 3D sparse factorization.
The extra matrix and solver vectors are created only for an implicit update.

With active secant damping the mass term carries the per-cell viscosity,
M*diag(1+r) + (dt/eta0)K with r = `cache.damping_ratio`; r depends on the
substep's scale, boost and eta, so this system is rebuilt on every call, and
once more on the first undamped call afterwards.
"""
function prepare_density_system!(cache::DensityCache, dt, eta0)
    damped = cache.damping_active
    if cache.system_dirty || cache.system_dt != dt || cache.system_eta0 != eta0 ||
            damped || cache.system_damped
        if cache.system === nothing
            cache.system = copy(cache.stiffness)
            n = length(cache.faces.area)
            resize!(cache.preconditioner.diag,n)
            cache.solver = CgWorkspace(n,n,Vector{Float64})
            resize!(cache.solver.z,n)
        end
        values = nonzeros(cache.system)
        kvals = nonzeros(cache.stiffness)
        a = dt/eta0
        for k in eachindex(values)
            values[k] = a*kvals[k]
        end
        for i in eachindex(cache.faces.area)
            mass = damped ? cache.faces.area[i]*(1+cache.damping_ratio[i]) : cache.faces.area[i]
            values[cache.diagonal_indices[i]] += mass
            cache.preconditioner.diag[i] = inv(values[cache.diagonal_indices[i]])
        end
        cache.system_dt, cache.system_eta0 = dt, eta0
        cache.system_dirty, cache.system_damped = false, damped
        return true
    end
    return false
end

"""Solve the frozen diffusion equation into preallocated trial storage."""
function solve_density_system!(cache,options)
    solver = cache.solver::CgWorkspace{Float64,Float64,Vector{Float64}}
    cg!(solver,cache.system,cache.rhs;M=cache.preconditioner,atol=0.0,
        rtol=options.implicit_tolerance,itmax=options.implicit_max_iterations,history=false)
    solver.stats.solved || error("Implicit density solve failed: $(solver.stats.status)")
    copyto!(cache.trial,solver.x)
    return solver.stats.niter
end

"""Response of the unconstrained trial to a unit volume multiplier.

A uniform multiplier lambda moves cell i by -lambda*w_i. Without a boost w = 1
(`nothing`), the plain uniform shift. With the boost the explicit response is
w_i = 1/b_i; the semi-implicit one is the diffusion solve of that load,
(M + dt/eta0*K) w = M/b, so the multiplier stays uniform in the solved equation.
With secant damping the viscosity eta0*b_i*(1+r_i) gives w_i = 1/(b_i*(1+r_i))
explicitly and (M*diag(1+r) + dt/eta0*K) w = M/b semi-implicitly.
The implicit system must already be prepared for the current substep.
"""
function volume_response!(cache::DensityCache,options)
    relative, damped = cache.beta_relative, cache.damping_active
    !damped && all(isone,relative) && return nothing
    weights = cache.volume_weight
    if options.integrator == :semi_implicit
        @. weights = cache.faces.area/relative
        solver = cache.solver::CgWorkspace{Float64,Float64,Vector{Float64}}
        cg!(solver,cache.system,weights;M=cache.preconditioner,atol=0.0,
            rtol=options.implicit_tolerance,itmax=options.implicit_max_iterations,history=false)
        solver.stats.solved || error("Implicit volume-response solve failed: $(solver.stats.status)")
        copyto!(weights,solver.x)
    elseif damped
        @. weights = 1/(relative*(1+cache.damping_ratio))
    else
        @. weights = 1/relative
    end
    return weights
end

"""Largest explicit diffusion rate of the baseline operator, independent of the boost."""
function diffusion_rate(cache, eta0)
    values = nonzeros(cache.stiffness)
    rate = 0.0
    for i in eachindex(cache.faces.area)
        rate = max(rate,values[cache.diagonal_indices[i]]/cache.faces.area[i])
    end
    return rate/eta0
end
