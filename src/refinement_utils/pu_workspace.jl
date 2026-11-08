"""Scratch reused across node patches and mesh rebuilds; no patch factors are cached."""
mutable struct PU3DWorkspace
    keys::Dict{Tuple{Int,Int},Int}
    masks::Vector{SVector{3,Bool}}
    maps::Vector{SVector{10,Int}}
    dofs::Vector{SVector{3,Int}}
    radial_slots::Vector{Int}
    radial_nodes::Vector{Int}
    matrix::Matrix{Float64}
    rhs::Vector{Float64}
    weighted::Vector{Float64}
    ndofs::Int
    max_used::Int
end
PU3DWorkspace() = PU3DWorkspace(Dict{Tuple{Int,Int},Int}(),SVector{3,Bool}[],
    SVector{10,Int}[],SVector{3,Int}[],Int[],Int[],zeros(0,0),Float64[],Float64[],0,0)

struct PU3DCache{G,L}
    geometry::G
    loads::L
    offsets::Vector{Int}
    incidence::Vector{Tuple{Int,Int}}
    fixed::Vector{SVector{3,Bool}}
    face_fixed::Vector{SVector{3,Bool}}
    edge_fixed::Dict{Tuple{Int,Int},SVector{3,Bool}}
    vertex_dofs::Vector{SVector{3,Int}}
    face_weights::Vector{Vector{Float64}}
    values::Vector{SVector{3,Float64}}
    stress::Vector{SMatrix{3,3,Float64,9}}
    energy::Vector{Float64}
    projected_energy::Vector{Float64}
    scale::Vector{Float64}
    marking_weight::Vector{Float64}
    nodal::Vector{Float64}
    element_correction::Vector{Float64}
    indicators::Vector{Float64}
    workspaces::Vector{PU3DWorkspace}
    lambda::Float64
    mu::Float64
end

function pu3d_incidence(geometry)
    offsets = ones(Int,length(geometry.node_points)+1)
    for t in geometry.tets, node in t.nodes
        offsets[node+1] += 1
    end
    for i in 2:length(offsets)
        offsets[i] += offsets[i-1]-1
    end
    incidence = Vector{Tuple{Int,Int}}(undef,offsets[end]-1)
    cursor = copy(offsets)
    for (index,t) in enumerate(geometry.tets), hat in 1:4
        node = t.nodes[hat]
        incidence[cursor[node]] = (index,hat)
        cursor[node] += 1
    end
    return offsets,incidence
end

function pu3d_fixed_values(geometry,cv,ch,boundary_data)
    all(iszero,values(ch.d_bcs)) || throw(ArgumentError(
        "PU-DWR strain-energy estimation requires homogeneous prescribed displacements"))
    fixed = fill(PU3D_FREE,length(geometry.node_points))
    dofs = [SVector{3,Int}(cv.dh.dof_mapping[id]) for id in geometry.vertex_ids]
    for i in eachindex(dofs)
        fixed[i] = SVector{3,Bool}(ntuple(k -> haskey(ch.d_bcs,dofs[i][k]),3))
    end
    face_fixed,edge_fixed = pu3d_auxiliary_constraints(geometry,cv,boundary_data)
    nv = length(geometry.vertex_ids)
    for face in eachindex(face_fixed)
        fixed[nv+face] = face_fixed[face]
    end
    return fixed,dofs,face_fixed,edge_fixed
end

"""Face-center interpolation preserving the mean of the face VEM projection."""
function pu3d_face_weights(geometry,cv)
    weights = Vector{Vector{Float64}}(undef,length(geometry.face_ids))
    nv = length(geometry.vertex_ids)
    for (face,id) in enumerate(geometry.face_ids)
        fd,ids = cv.facedata_col[id],geometry.face_nodes[face]
        result = [3fd.ΠsL2[1,j] for j in eachindex(ids)]
        center = geometry.node_points[nv+face]
        area = Ju3VEM.VEMGeo.get_area(fd)
        for i in eachindex(ids)
            j = mod1(i+1,length(ids))
            a,b = geometry.node_points[ids[i]],geometry.node_points[ids[j]]
            fraction = norm(cross(a-center,b-center))/(2area)
            result[i] -= fraction
            result[j] -= fraction
        end
        weights[face] = result
    end
    return weights
end

"""Linear mesh setup: geometry, node incidence, loads and interpolation weights.

Patch numbering, stiffness and factorization are computed on demand. Pass the
previous cache to reuse worker scratch after frequent mesh changes.
"""
function build_pu_dwr_cache(cv,states,eldata,ch,pars;
        boundary_data=PU3DBoundaryData(),rhs_fun=ZeroBodyForce(),previous=nothing)
    isfinite(pars.λ) && isfinite(pars.μ) && pars.μ > 0 && 3pars.λ+2pars.μ > 0 ||
        throw(ArgumentError("PU-DWR requires a positive isotropic elastic material"))
    geometry = build_pu_geometry(cv,states,eldata)
    offsets,incidence = pu3d_incidence(geometry)
    fixed,dofs,face_fixed,edge_fixed = pu3d_fixed_values(geometry,cv,ch,boundary_data)
    loads = pu3d_build_loads(geometry,cv,ch,boundary_data,rhs_fun,eldata)
    n,ne,nt = length(geometry.node_points),length(geometry.element_ids),length(geometry.tets)
    workspaces = previous === nothing ? [PU3DWorkspace() for _ in 1:Threads.nthreads()] : previous.workspaces
    for w in workspaces
        old_length=length(w.radial_slots)
        if old_length < n
            resize!(w.radial_slots,n)
            fill!(@view(w.radial_slots[old_length+1:n]),0)
        end
    end
    return PU3DCache(geometry,loads,offsets,incidence,fixed,face_fixed,edge_fixed,dofs,
        pu3d_face_weights(geometry,cv),fill(zero(SVector{3,Float64}),n),
        fill(zero(SMatrix{3,3,Float64,9}),nt),zeros(nt),zeros(ne),zeros(ne),
        zeros(ne),zeros(n),zeros(ne),zeros(ne),workspaces,pars.λ,pars.μ)
end

@inline function pu3d_local_key(t,i)
    i <= 4 && return (t.nodes[i],t.nodes[i])
    a,b = PU3D_P2_EDGES[i-4]
    return minmax(t.nodes[a],t.nodes[b])
end

@inline function pu3d_local_fixed(c,t,i,hat)
    # The face opposite the star node is an artificial patch boundary, except
    # when it coincides with the physical boundary (the cell-center patch).
    physical = c.geometry.face_cells[t.face][2] == 0
    artificial = !(hat == 4 && physical)
    if i <= 4
        artificial && i != hat && return PU3D_FIXED
        return c.fixed[t.nodes[i]]
    end
    a,b = PU3D_P2_EDGES[i-4]
    artificial && a != hat && b != hat && return PU3D_FIXED
    # Entity metadata distinguishes a prescribed edge/face from isolated
    # supports at its endpoints. Propagate an edge constraint to every incident
    # tetrahedron, including tetrahedra from an otherwise natural face.
    a == 1 && b == 2 && return get(c.edge_fixed,minmax(t.nodes[1],t.nodes[2]),PU3D_FREE)
    b == 3 && return c.face_fixed[t.face]
    return PU3D_FREE
end

function pu3d_patch_number_generic!(w,c,node)
    empty!(w.keys)
    empty!(w.masks)
    first,last = c.offsets[node],c.offsets[node+1]-1
    resize!(w.maps,last-first+1)
    for (localtet,index) in enumerate(first:last)
        ti,hat = c.incidence[index]
        t = c.geometry.tets[ti]
        w.maps[localtet] = SVector{10,Int}(ntuple(Val(10)) do i
            key = pu3d_local_key(t,i)
            slot = get(w.keys,key,0)
            mask = pu3d_local_fixed(c,t,i,hat)
            if slot == 0
                slot = length(w.masks)+1
                w.keys[key] = slot
                push!(w.masks,mask)
            else
                w.masks[slot] = w.masks[slot] .| mask
            end
            return slot
        end)
    end
    resize!(w.dofs,length(w.masks))
    n = 0
    for i in eachindex(w.masks)
        mask = w.masks[i]
        a = mask[1] ? 0 : (n += 1)
        b = mask[2] ? 0 : (n += 1)
        d = mask[3] ? 0 : (n += 1)
        w.dofs[i] = SA[a,b,d]
    end
    w.ndofs = n
    w.max_used = max(w.max_used,n)
    if size(w.matrix,1) < n
        capacity = max(n,ceil(Int,1.25size(w.matrix,1)))
        w.matrix = zeros(capacity,capacity)
        resize!(w.rhs,capacity)
        resize!(w.weighted,capacity)
    end
    @inbounds for j in 1:n
        w.rhs[j] = w.weighted[j] = 0.0
        for i in j:n
            w.matrix[i,j] = 0.0
        end
    end
    return first:last
end

# A vertex/face star has only its central P2 vertex and incident edge midpoints
# free on its artificial boundary. Interior cell stars have the same structure.
# Full numbering is needed only when a cell star reaches a natural boundary.
@inline function pu3d_radial_star(c,node,incidence)
    node <= length(c.geometry.vertex_ids)+length(c.geometry.face_ids) && return true
    for index in incidence
        ti,_ = c.incidence[index]
        face = c.geometry.tets[ti].face
        c.geometry.face_cells[face][2] == 0 && c.face_fixed[face] != PU3D_FIXED && return false
    end
    return true
end

@inline function pu3d_radial_dofs!(w,mask)
    n = w.ndofs
    a = mask[1] ? 0 : (n += 1)
    b = mask[2] ? 0 : (n += 1)
    d = mask[3] ? 0 : (n += 1)
    w.ndofs = n
    push!(w.dofs,SA[a,b,d])
    return length(w.dofs)
end

@inline function pu3d_radial_slot!(w,c,t,shape,hat)
    a,b = PU3D_P2_EDGES[shape-4]
    other = t.nodes[a == hat ? b : a]
    slot = w.radial_slots[other]
    if slot == 0
        slot = pu3d_radial_dofs!(w,pu3d_local_fixed(c,t,shape,hat))
        w.radial_slots[other] = slot
        push!(w.radial_nodes,other)
    end
    return slot
end

function pu3d_patch_number!(w,c,node)
    first,last = c.offsets[node],c.offsets[node+1]-1
    incidence = first:last
    pu3d_radial_star(c,node,incidence) || return pu3d_patch_number_generic!(w,c,node)
    @inbounds for other in w.radial_nodes
        w.radial_slots[other] = 0
    end
    empty!(w.radial_nodes)
    empty!(w.masks)
    empty!(w.dofs)
    w.ndofs = 0
    push!(w.dofs,SA[0,0,0]) # Shared slot for all constrained outer shapes.
    push!(w.dofs,SA[0,0,0]) # Number the central vertex last to avoid early fill.
    resize!(w.maps,length(incidence))
    @inbounds for (localtet,index) in enumerate(incidence)
        ti,hat = c.incidence[index]
        t = c.geometry.tets[ti]
        # Only the three radial P2 edges are unknown on an artificial star.
        # Explicit maps avoid visiting seven constrained shapes and avoid
        # capturing the central DOF counters in a tuple-construction closure.
        if hat == 1
            edge_a = pu3d_radial_slot!(w,c,t,5,1)
            edge_b = pu3d_radial_slot!(w,c,t,6,1)
            edge_c = pu3d_radial_slot!(w,c,t,7,1)
            w.maps[localtet] = SA[2,1,1,1,edge_a,edge_b,edge_c,1,1,1]
        elseif hat == 2
            edge_a = pu3d_radial_slot!(w,c,t,5,2)
            edge_b = pu3d_radial_slot!(w,c,t,8,2)
            edge_c = pu3d_radial_slot!(w,c,t,9,2)
            w.maps[localtet] = SA[1,2,1,1,edge_a,1,1,edge_b,edge_c,1]
        elseif hat == 3
            edge_a = pu3d_radial_slot!(w,c,t,6,3)
            edge_b = pu3d_radial_slot!(w,c,t,8,3)
            edge_c = pu3d_radial_slot!(w,c,t,10,3)
            w.maps[localtet] = SA[1,1,2,1,1,edge_a,1,edge_b,1,edge_c]
        else
            edge_a = pu3d_radial_slot!(w,c,t,7,4)
            edge_b = pu3d_radial_slot!(w,c,t,9,4)
            edge_c = pu3d_radial_slot!(w,c,t,10,4)
            w.maps[localtet] = SA[1,1,1,2,1,1,edge_a,1,edge_b,edge_c]
        end
    end
    n = w.ndofs
    mask = c.fixed[node]
    a = mask[1] ? 0 : (n += 1)
    b = mask[2] ? 0 : (n += 1)
    d = mask[3] ? 0 : (n += 1)
    w.dofs[2] = SA[a,b,d]
    w.ndofs = n
    w.max_used = max(w.max_used,n)
    if size(w.matrix,1) < n
        capacity = max(n,ceil(Int,1.25size(w.matrix,1)))
        w.matrix = zeros(capacity,capacity)
        resize!(w.rhs,capacity)
        resize!(w.weighted,capacity)
    end
    @inbounds for j in 1:n
        w.rhs[j] = w.weighted[j] = 0.0
        for i in j:n
            w.matrix[i,j] = 0.0
        end
    end
    return incidence
end
