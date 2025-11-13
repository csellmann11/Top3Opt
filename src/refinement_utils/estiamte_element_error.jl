using Ju3VEM

function project_states_to_nodes(
    eldata_col::Dict{Int,<:ElData},
    cv::CellValues{D,U},
    states::DesignVarInfo) where {D,U}

    node_sums   = zeros(Float64,length(cv.mesh.topo.nodes))
    node_states = Dict{Int,Float64}()
    e2s = states.el_id_to_state_id
    for (el_id,el_data) in eldata_col
        sid      = e2s[el_id]
        γ_stab   = el_data.γ_stab
        bc       = states.x_vec[sid]
        node_ids = el_data.node_ids
        
        for node_id in node_ids
            weight =  get!(node_states,node_id,0.0)
            node    = cv.mesh.topo.nodes[node_id]
            d       = norm(node.coords - bc)
            weight += γ_stab/d
            node_states[node_id] = weight
            node_sums[node_id] += 1/d
        end
    end
    for node_id in keys(node_states)
        node_states[node_id] /= node_sums[node_id]
    end

    return node_states
end

function estimate_element_error(
    u::AbstractVector{Float64},
    states::DesignVarInfo,
    cv::CellValues{D,U},
    eldata_col::Dict{Int,<:ElData}
    ) where {D,U}

    element_error = Dict{Int,Float64}()

    node_states = project_states_to_nodes(eldata_col,cv,states)

    for (el_id,el_data) in eldata_col

        # dofs = el_data.dofs
        node_ids = el_data.node_ids
        error = @no_escape begin 
            node_ids = el_data.node_ids 
            dofs = @alloc(Int,length(node_ids)*U)
            get_dofs!(dofs,cv.dh,node_ids)
            uel  = @view u[dofs]
            proj = stretch(el_data.proj,Val(3))
            node_ids = el_data.node_ids 
            hvol     = el_data.hvol
            error = 0.0 
            for i in axes(proj,1)
                du = 0.0 
                for j in axes(proj,2)
                    du += proj[i,j] * uel[j]
                end  
                du -= uel[i]
                n_count = ceil(Int,i/U)
                error += du^2 * node_states[node_ids[n_count]] * hvol
            end
            error 
        end
        element_error[el_id] = error
    end


    return element_error
end



# function estimate_element_error(
#     u::AbstractVector{Float64},
#     eldata_col::Dict{Int,<:ElData}
#     ) 

#     element_error = Dict{Int,Float64}()

    

#     for (el_id,el_data) in eldata_col

#         dofs = el_data.dofs
#         uel  = @view u[dofs]
#         proj = el_data.proj 
#         hvol = el_data.hvol
#         diff = stretch((I-proj),Val(3))*uel
#         error = el_data.γ_stab * diff' * diff * hvol
#         # error = diff' * diff
#         element_error[el_id] = error
#     end

#     return element_error
# end


function mark_elements_for_adaption(
    cv::CellValues{D,U},
    element_error::Dict{Int,Float64},
    states       ::DesignVarInfo,
    state_changed::AbstractVector{Float64},
    max_ref_level::Int,
    no_coarsening_marker::Vector{Bool},
    state_neights_col::AbstractVector{Vector{Int32}},
    upper_error_bound::Float64 = 16.0,
    lower_error_bound::Float64 = 0.25
    ) where {D,U}

    topo = cv.mesh.topo
    e2s = states.el_id_to_state_id

    ref_marker    = zeros(Bool,length(get_volumes(topo)))
    coarse_marker = copy(ref_marker)

    m_error = mean(values(element_error))
     



    for (el_id,state_id) in e2s

        n_vertices = length(get_volume_node_ids(topo,el_id))
        upper_error_bound = lower_error_bound * 4*n_vertices
        error  = element_error[el_id]
        element = get_volumes(topo)[el_id]
        ref_level = element.refinement_level
        dχi    = state_changed[state_id]
 
        has_parent = element.parent_id != 0

        if (dχi > 0.0 || error > m_error * upper_error_bound) &&  ref_level < max_ref_level 
            ref_marker[el_id] = true
        elseif error < m_error * lower_error_bound && has_parent && dχi == 0.0
            (el_id <= length(no_coarsening_marker) && no_coarsening_marker[el_id]) && continue
            coarse_marker[el_id] = true
        end
    end

    # for (state_id,neighs) in enumerate(state_neights_col)
    #     el_id = get_el_id(states,state_id)
    #     coarse_marker[el_id] || continue
    #     for neigh_id in neighs
    #         if neigh_id < 0
    #             continue
    #         end
    #         neight_el_id = get_el_id(states,neigh_id)
    #         if ref_marker[neight_el_id]
    #             coarse_marker[el_id] = false
    #             break
    #         end
    #     end
    # end

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



function adapt_mesh(cv::CellValues{D,U},
    coarse_marker::Vector{Bool},
    ref_marker::Vector{Bool}) where {D,U}

    topo = cv.mesh.topo
    for element in RootIterator{4}(topo)
        if coarse_marker[element.id]
            _coarsen!(element,topo)
        elseif ref_marker[element.id]
            _refine!(element,topo)
        end
    end

    return CellValues{U}(Mesh(topo,StandardEl{1}()))
end