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
    # The caller owns the baseline policy (in 3D: 2*beta0*h_i^2).
    b = beta0 isa Real ? fill(Float64(beta0),n) : Float64.(beta0)
    length(b) == n || throw(DimensionMismatch("Baseline regularization coefficient"))
    all(x -> isfinite(x) && x >= 0,b) || throw(ArgumentError("beta0 must be finite and nonnegative"))
    K = diffusion_matrix(faces,b)
    diagonal_indices = [stored_index(K,i,i) for i in 1:n]
    face_indices = [(stored_index(K,faces.left[f],faces.right[f]),
        stored_index(K,faces.right[f],faces.left[f])) for f in 1:nf]
    return DensityCache(faces,b,ones(n),K,diagonal_indices,face_indices,
        nothing,Diagonal(Float64[]),nothing,true,NaN,NaN,
        zeros(n),zeros(n),zeros(n),Float64[],zeros(n),falses(n),nothing,(;))
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

"""Freeze the normalized coefficient for one density substep.

For `:local_max`, beta_hat[i] = beta0[i] * max(1, max(abs(p[j])) / scale),
where j is the cell itself or a direct TPFA face neighbor. The physical beta
is scale * beta_hat; eta remains eta0 * scale. This is the former adaptive
regularization rule, not a differentiated, fixed objective in p(chi).

The existing face graph supplies neighbors. Read each neighbor's original p
so a single traversal cannot propagate a peak beyond its direct neighbors.
The trial buffer is free until the subsequent density solve/projection.
"""
function update_coefficients!(cache::DensityCache, p, scale, options)
    relative, candidate = cache.beta_relative, cache.trial
    if options.regularization == :local_max
        for i in eachindex(p)
            candidate[i] = max(1.0,abs(p[i])/scale)
        end
        faces = cache.faces
        for f in eachindex(faces.left)
            i, j = faces.left[f], faces.right[f]
            candidate[i] = max(candidate[i],abs(p[j])/scale)
            candidate[j] = max(candidate[j],abs(p[i])/scale)
        end
    else
        # The same cache can switch back from local to uniform regularization.
        all(isone,relative) && return false
        fill!(candidate,1.0)
    end
    candidate == relative && return false
    copyto!(relative,candidate)
    values = nonzeros(cache.stiffness)
    fill!(values,0.0)
    faces = cache.faces
    for f in eachindex(faces.left)
        i, j = faces.left[f], faces.right[f]
        bi = cache.base_beta[i]*relative[i]
        bj = cache.base_beta[j]*relative[j]
        conductance = tpfa_conductance(faces.length[f],faces.distance_left[f],faces.distance_right[f],bi,bj)
        isfinite(conductance) && conductance >= 0 ||
            throw(ArgumentError("Invalid local face conductance at face $f"))
        ij, ji = cache.face_indices[f]
        values[ij] -= conductance
        values[ji] -= conductance
        values[cache.diagonal_indices[i]] += conductance
        values[cache.diagonal_indices[j]] += conductance
    end
    cache.system_dirty = true
    return true
end

"""Refresh M+(dt/eta0)K and its Jacobi preconditioner only when needed.
CG uses a reusable workspace, avoiding fill-in from a 3D sparse factorization.
The extra matrix and solver vectors are created only for an implicit update.
"""
function prepare_density_system!(cache::DensityCache, dt, eta0)
    if cache.system_dirty || cache.system_dt != dt || cache.system_eta0 != eta0
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
            values[cache.diagonal_indices[i]] += cache.faces.area[i]
            cache.preconditioner.diag[i] = inv(values[cache.diagonal_indices[i]])
        end
        cache.system_dt, cache.system_eta0 = dt, eta0
        cache.system_dirty = false
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

function diffusion_rate(cache, eta0)
    values = nonzeros(cache.stiffness)
    rate = 0.0
    for i in eachindex(cache.faces.area)
        rate = max(rate,values[cache.diagonal_indices[i]]/cache.faces.area[i])
    end
    return rate/eta0
end
