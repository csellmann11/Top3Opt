using Ju3VEM

function estimate_element_error(
    u::AbstractVector{Float64},
    eldata_col::Dict{Int,<:ElData}
    ) 

    element_error = Dict{Int,Float64}()

    for (el_id,el_data) in eldata_col

        dofs = el_data.dofs
        uel  = @view u[dofs]
        proj = el_data.proj 
        diff = stretch((I-proj),Val(3))*uel
        error = el_data.γ_stab * diff' * diff
        element_error[el_id] = error
    end

    return element_error
end


function mark_elements_for_adaption(
    cv::CellValues{D,U},
    element_error::Dict{Int,Float64},
    states       ::TopStates,
    state_changed::Vector{Bool},
    max_ref_level::Int;
    upper_error_bound::Float64 = 4.0,
    lower_error_bound::Float64 = 1/8
    ) where {D,U}

    topo = cv.mesh.topo
    e2s = states.el_id_to_state_id

    ref_marker    = zeros(Bool,length(get_volumes(topo)))
    coarse_marker = copy(ref_marker)

    m_error = mean(values(element_error))

    for (el_id,state_id) in e2s

        error  = element_error[el_id]
        element = get_volumes(topo)[el_id]
        ref_level = element.refinement_level
        dχi    = state_changed[state_id]

        has_parent = element.parent_id != 0
#dχi != 0 || 
        if (error > m_error * upper_error_bound) &&  ref_level < max_ref_level 
            ref_marker[el_id] = true
        elseif error < m_error * lower_error_bound && has_parent
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