using Ju3VEM

isdefined(@__MODULE__, :PU3DCache) || include("pu_dwr.jl")

"""Inverse-distance average of the element stabilization at the nodes, indexed by node id."""
function project_states_to_nodes(
    eldata_col::Dict{Int,<:ElData},
    cv::CellValues{D,U},
    states::DesignVarInfo) where {D,U}

    nodes       = cv.mesh.topo.nodes
    node_sums   = zeros(Float64,length(nodes))
    node_states = zeros(Float64,length(nodes))
    e2s = states.el_id_to_state_id
    for (el_id,el_data) in eldata_col
        γ_stab = el_data.γ_stab
        bc     = states.x_vec[e2s[el_id]]
        for node_id in el_data.node_ids
            d = norm(nodes[node_id].coords - bc)
            node_states[node_id] += γ_stab/d
            node_sums[node_id]   += 1/d
        end
    end
    @inbounds for node_id in eachindex(node_states)
        node_sums[node_id] > 0 && (node_states[node_id] /= node_sums[node_id])
    end
    return node_states
end

# Σ_i ((P u - u)_i)² * weight(node of i) * hvol for one element. P couples only equal
# components, so the sum over the other components (structural zeros) is skipped.
function element_error(u::AbstractVector{Float64},el_data::ElData,
        dof_mapping::Dict{Int,SVector{U,Int}},node_states::Vector{Float64}) where {U}
    node_ids = el_data.node_ids
    proj     = el_data.proj
    error    = 0.0
    @inbounds for a in eachindex(node_ids)
        dofs_a = dof_mapping[node_ids[a]]
        weight = node_states[node_ids[a]]
        for c in 1:U
            du = 0.0
            for b in eachindex(node_ids)
                du += proj[a,b] * u[dof_mapping[node_ids[b]][c]]
            end
            du -= u[dofs_a[c]]
            error += du^2 * weight * el_data.hvol
        end
    end
    return error
end

function estimate_element_error(
    u::AbstractVector{Float64},
    states::DesignVarInfo,
    cv::CellValues{D,U},
    eldata_col::Dict{Int,<:ElData}
    ) where {D,U}

    node_states = project_states_to_nodes(eldata_col,cv,states)
    dof_mapping = cv.dh.dof_mapping

    # Elements are independent: evaluate them on all threads, then fill the Dict.
    elements = collect(eldata_col)
    errors   = Vector{Float64}(undef,length(elements))
    Threads.@threads for k in eachindex(elements)
        errors[k] = element_error(u,elements[k].second,dof_mapping,node_states)
    end
    element_error_col = sizehint!(Dict{Int,Float64}(),length(elements))
    for k in eachindex(elements)
        element_error_col[elements[k].first] = errors[k]
    end
    return element_error_col
end





function mark_elements_for_adaption(
    cv::CellValues{D,U},
    element_error::Union{Dict{Int,Float64},AbstractVector{<:Real}},
    states       ::DesignVarInfo,
    state_changed::AbstractVector{<:Real},
    max_ref_level::Int,
    no_coarsening_marker::AbstractVector{Bool},
    density_marking::Bool,
    upper_error_bound::Float64 = 16.0,
    lower_error_bound::Float64 = 0.25;
    vertex_scaling::Bool = true
    ) where {D,U}

    topo = cv.mesh.topo
    e2s = states.el_id_to_state_id

    ref_marker    = zeros(Bool,length(get_volumes(topo)))
    coarse_marker = copy(ref_marker)

    m_error = mean(values(element_error))
     



    for (el_id,state_id) in e2s

        n_vertices = length(get_volume_node_ids(topo,el_id))
        refinement_bound = vertex_scaling ? lower_error_bound * 4*n_vertices : upper_error_bound
        error  = element_error isa AbstractVector ? element_error[state_id] : element_error[el_id]
        element = get_volumes(topo)[el_id]
        ref_level = element.refinement_level
        dχi    = state_changed[state_id]
 
        has_parent = element.parent_id != 0

        # Cached activity includes physical changes from filtering/projection.
        density_active = !iszero(dχi) && density_marking
        if (density_active || error > m_error * refinement_bound) &&  ref_level < max_ref_level
            ref_marker[el_id] = true
        elseif error < m_error * lower_error_bound && has_parent && dχi == 0.0
            (el_id <= length(no_coarsening_marker) && no_coarsening_marker[el_id]) && continue
            coarse_marker[el_id] = true
        end
    end

    # remove coarse marker from children, if not all childs are marked for coarseing 
    for el_id in eachindex(coarse_marker)
        element = get_volumes(topo)[el_id] 
        is_root(element) && continue 

        child_ids = element.childs
        if any(id -> !coarse_marker[id], child_ids)
            coarse_marker[child_ids] .= false
        end
    end

    return ref_marker, coarse_marker
end


using Ju3VEM.VEMGeo: _refine!,_coarsen!


function resize_with_default!(vec::Vector{T}, n::Int, default::T) where T
    current_length = length(vec)
    resize!(vec, n)
    for i in current_length+1:n
        vec[i] = default
    end
    return vec
end

function recycle_mesh!(mesh::Mesh{D,ET}) where {D,ET<:ElType{1}}
    int_coords_connect = mesh.int_coords_connect
    n_edges = length(get_edges(mesh.topo)) 
    n_areas = length(get_areas(mesh.topo))
    n_volumes = length(get_volumes(mesh.topo))
    resize_with_default!(int_coords_connect[2], n_edges, Int[])
    resize_with_default!(int_coords_connect[3], n_areas, Int[])
    resize_with_default!(int_coords_connect[4], n_volumes, Int[])
    empty!(mesh.node_sets)
    empty!(mesh.edge_sets)
    empty!(mesh.face_sets)
    empty!(mesh.volume_sets)
    return mesh
end


"""
    update_facedata(mesh, old) -> Dict

Face data for the adapted mesh. A root face that already existed and still has the
same nodes (no edge of it was split or merged) keeps its `FaceData` object; a face
whose node list changed keeps its geometry integrals and only gets a new projector;
only new faces are computed from scratch. Reused entries are exactly what a full
rebuild would compute, since they depend only on the face's nodes.
"""
function update_facedata(mesh::Mesh{3,StandardEl{K}}, old::Dict{Int,FD}) where {K,FD}
    topo = mesh.topo
    K_MAX = max(2*K-1, 2)
    facedata_col = sizehint!(Dict{Int,FD}(), length(old))
    scratch = Ju3VEM.VEMGeo.FlattenVecs{3,Int}()
    for face in RootIterator{3}(topo)
        previous = get(old, face.id, nothing)
        if previous === nothing
            facedata_col[face.id] = Ju3VEM.VEMUtils.h1_projectors!(face.id, mesh,
                Ju3VEM.VEMGeo.precompute_face_monomials(face.id, topo, Val(K_MAX)))
            continue
        end
        foreach(empty!, scratch.v.args)
        Ju3VEM.VEMUtils.get_iterative_area_node_ids!(scratch, face, mesh)
        facedata_col[face.id] = scratch.v.args == previous.face_node_ids.v.args ? previous :
            Ju3VEM.VEMUtils.h1_projectors!(face.id, mesh, previous.dΩ)
    end
    return facedata_col
end

function adapt_mesh(cv::CellValues{D,U},
    coarse_marker::AbstractVector{Bool},
    ref_marker::AbstractVector{Bool}) where {D,U}

    topo = cv.mesh.topo
    @timeit to "refine_coarsen" for element in RootIterator{4}(topo)
        if coarse_marker[element.id]
            _coarsen!(element,topo)
        elseif ref_marker[element.id]
            _refine!(element,topo)
        end
    end

    # Same result as CellValues{U}(recycle_mesh!(cv.mesh)), but only faces that
    # are new or whose nodes changed get their projector recomputed.
    mesh = recycle_mesh!(cv.mesh)
    @timeit to "dof_handler" dh = Ju3VEM.VEMUtils.DofHandler{U}(mesh)
    @timeit to "facedata" facedata_col = update_facedata(mesh, cv.facedata_col)
    return CellValues(mesh, cv.vnm, facedata_col, cv.volume_data, dh)
end
