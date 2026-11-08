"""Six geometric quadratic interpolation defects, without a polynomial fit.

Rows are P2 nodal values (fan vertices followed by edge midpoints), not
hierarchical coefficients. I_V uses exactly the original VEM-to-fan map.
"""
function compact_local_defects(mesh,fan,geometry)
    P = hierarchical_prolongation(mesh,fan,geometry)
    nv,nd = geometry.vertex_count,size(geometry.K,1)
    xyz = Matrix{Float64}(undef,3,nd)
    xyz[:,1:nv] .= fan.points
    for (edge,(a,b)) in enumerate(geometry.edges)
        xyz[:,nv+edge] .= (fan.points[:,a]+fan.points[:,b])/2
    end
    q = hcat(xyz[1,:].^2,xyz[2,:].^2,xyz[3,:].^2,
        xyz[1,:].*xyz[2,:],xyz[1,:].*xyz[3,:],xyz[2,:].*xyz[3,:])
    interpolated = P*q[1:fan.vertex_count,:]
    for (edge,(a,b)) in enumerate(geometry.edges)
        interpolated[nv+edge,:] .= (interpolated[a,:]+interpolated[b,:])/2
    end
    incident_edges = [Int[] for _ in 1:nv]
    for (edge,(a,b)) in enumerate(geometry.edges)
        push!(incident_edges[a],edge)
        push!(incident_edges[b],edge)
    end
    return (;P,defects=q-interpolated,incident_edges)
end

"""P2 interpolation of Lambda_i*(q-I_V q), converted to hierarchical form.

The optional last column is Lambda_i itself. Before physical-boundary masking,
summing any curvature column over original vertices gives exactly q-I_V q.
No artificial patch-boundary values are independently prescribed.
"""
function compact_local_basis(geometry,geometric,node;include_hat=false,mask_boundary=true)
    nv,P = geometry.vertex_count,geometric.P
    vertices = P.rowval[nzrange(P,node)]
    lambda = Dict(vertex=>P.nzval[p] for (vertex,p) in zip(vertices,nzrange(P,node)))
    edges = sort!(unique!(reduce(vcat,(geometric.incident_edges[v] for v in vertices);init=Int[])))
    ids = vcat(vertices,nv .+ edges)
    B = zeros(length(ids),6+include_hat)
    for (row,vertex) in enumerate(vertices)
        B[row,1:6] .= lambda[vertex].*geometric.defects[vertex,:]
        include_hat && (B[row,7] = lambda[vertex])
    end
    for (row,edge) in enumerate(edges)
        a,b = geometry.edges[edge]
        la,lb = get(lambda,a,0.0),get(lambda,b,0.0)
        B[length(vertices)+row,1:6] .= ((la+lb).*geometric.defects[nv+edge,:] .-
            la.*geometric.defects[a,:] .- lb.*geometric.defects[b,:])./2
    end
    retained = findall(row -> (!mask_boundary || !geometry.boundary[ids[row]]) &&
        any(!iszero,B[row,:]),eachindex(ids))
    return ids[retained],B[retained,:]
end

"""Energy-orthonormal columns spanning a small supplied local basis.

Diagonal equilibration makes rank decisions independent of column units.
The filter removes dependent functions; it never fills missing information
by a point fit. The actual operator defines the local energy and coefficients.
"""
function compact_local_energy_basis(A,ids,B;rank_tolerance=1e-11)
    isempty(ids) && return zeros(0,0)
    G = transpose(B)*A[ids,ids]*B
    diagonal = diag(G)
    active = findall(>(eps()*maximum(abs,diagonal)),diagonal)
    isempty(active) && return zeros(length(ids),0)
    scaling = 1 ./sqrt.(diagonal[active])
    normalized = Symmetric((scaling.*G[active,active]).*transpose(scaling))
    decomposition = eigen(normalized)
    retained = findall(>(rank_tolerance*maximum(decomposition.values)),decomposition.values)
    isempty(retained) && return zeros(length(ids),0)
    transform = (scaling.*decomposition.vectors[:,retained])./
        transpose(sqrt.(decomposition.values[retained]))
    return B[:,active]*transform
end

"""Prepare six/seven-mode blocks on ORIGINAL VEM-vertex patches.

This clarity prototype restricts the already assembled hierarchical matrix;
it is not a low-memory production implementation. Only tiny local Gram
matrices are diagonalized. No global/coarse correction matrix is factored.
"""
function compact_local_blocks(mesh,fan,geometry;include_hat=false,rank_tolerance=1e-11)
    geometric = compact_local_defects(mesh,fan,geometry)
    blocks = NamedTuple[]
    ranks = zeros(Int,fan.vertex_count)
    max_energy_defect = 0.0
    for node in 1:fan.vertex_count
        ids,B = compact_local_basis(geometry,geometric,node;include_hat)
        Q = compact_local_energy_basis(geometry.K,ids,B;rank_tolerance)
        rank = size(Q,2)
        ranks[node] = rank
        rank == 0 && continue
        image_ids = sort!(unique!([geometry.K.rowval[p] for id in ids
            for p in nzrange(geometry.K,id)]))
        image = geometry.K[image_ids,ids]*Q
        max_energy_defect = max(max_energy_defect,
            norm(transpose(Q)*geometry.K[ids,ids]*Q-I,Inf))
        push!(blocks,(;node,ids,basis=Q,image_ids,image))
    end
    statistics = (;patch_count=fan.vertex_count,active_patches=length(blocks),
        local_coefficients=sum(ranks),maximum_rank=maximum(ranks),
        minimum_rank=minimum(ranks),rank_histogram=[count(==(r),ranks) for r in 0:(6+include_hat)],
        include_hat,max_energy_defect,rank_tolerance)
    return (;blocks,statistics)
end

"""Symmetric multiplicative local block updates of one actual residual.

The global fine vector/matrix here is solely an accuracy harness. Each update
is a rank <= 7 energy projection; overlap enters through the updated residual.
There is no global/coarse solve, fitted damping, or problem-specific weight.
"""
function compact_local_sweeps(A,b,blocks;checkpoints=(1,2,4),free=axes(A,1))
    x,r = zeros(length(b)),copy(b)
    output = NamedTuple[]
    denominator = max(norm(b[free]),eps())
    for step in 1:maximum(checkpoints)
        for ordering in (eachindex(blocks),reverse(eachindex(blocks)))
            for index in ordering
                block = blocks[index]
                coefficient = transpose(block.basis)*r[block.ids]
                x[block.ids] .+= block.basis*coefficient
                r[block.image_ids] .-= block.image*coefficient
            end
        end
        if step in checkpoints
            # Independent residual calculation also checks update bookkeeping.
            true_residual = b-A*x
            drift = norm(r[free]-true_residual[free])/denominator
            push!(output,(;step,x=copy(x),relative_residual=norm(true_residual[free])/denominator,
                residual_drift=drift))
        end
    end
    return output
end

"""Variational goal readout of assembled overlapping local corrections."""
function compact_local_estimates(geometry,residuals,collection;checkpoints=(1,2,4),
        remainder=delta->0.0,energy_goal=false)
    us = compact_local_sweeps(geometry.K,residuals.bu,collection.blocks;
        checkpoints,free=geometry.free)
    zs = compact_local_sweeps(geometry.K,residuals.bz,collection.blocks;
        checkpoints,free=geometry.free)
    results = NamedTuple[]
    for (su,sz) in zip(us,zs)
        du,dz = su.x,sz.x
        raw = energy_goal ? residuals.primal_work+dot(residuals.bu,du)-
            dot(du,geometry.K*du)/2 : residuals.consistency+
            dot(residuals.bu,dz)+dot(residuals.bz,du)-dot(du,geometry.K*dz)
        quadratic_correction = energy_goal ? 0.0 : remainder(du)
        push!(results,(;method=:compact_local,steps=su.step,
            estimate=raw+quadratic_correction,raw_estimate=raw,quadratic_correction,
            primal_residual=su.relative_residual,dual_residual=sz.relative_residual,
            residual_drift=max(su.residual_drift,sz.residual_drift)))
    end
    return results
end
