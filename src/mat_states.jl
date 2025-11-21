using Ju3VEM.VEMGeo: is_active_root


struct SimPars{H<:Helmholtz}
    mat_law::H
    λ::Float64
    μ::Float64 
    
    χmin::Float64
    η0::Float64 
    β0::Float64
    ρ_init::Float64
end


@inline function Ψlin_totopt(∇u::M,λ,μ,χ) where {M<:AbstractMatrix}
    ε = 1/2*(∇u + ∇u')
    W = λ/2 * tr(ε)^2 + μ*tr(ε*ε)
    return W*χ[1]^3
end

"""
    DesignVarInfo{D}

A struct representing the states of elements in a topology optimization problem.

# Fields
- `χ_vec::FlattenVecs{2,Float64}`: A flattened vector storing the state values (χ) for both inner and ghost states.
- `x_vec::Vector{SVector{D,Float64}}`: A vector storing the coordinates of the centroids for inner states.
- `el_id_to_state_id::Dict{Int32,Int32}`: A dictionary mapping element IDs to their corresponding state IDs.

# Type Parameters
- `D`: The dimension of the problem (e.g., 2 for 2D problems).

# Notes
- The `χ_vec` field uses `FlattenVecs` to efficiently store both inner and ghost states.
- Only the coordinates of inner states are stored in `x_vec`.
- The `el_id_to_state_id` dictionary allows for quick lookup of state IDs given an element ID.
"""
struct DesignVarInfo{D}
    χ_vec::Vector{Float64} 
    x_vec::Vector{SVector{D,Float64}}
    area_vec::Vector{Float64} #TODO: rename to elsize_vec
    h_vec::Vector{Float64} 

    el_id_to_state_id::OrderedDict{Int32,Int32}
end

function get_el_id(states::DesignVarInfo,state_id::Integer)
    return states.el_id_to_state_id.keys[state_id]
end
function get_state_id(states::DesignVarInfo,el_id::Integer)
    return states.el_id_to_state_id[el_id]
end


function fill_states!(states::DesignVarInfo{D},cv::CellValues{D,U},ρ_init) where {D,U}

    topo = cv.mesh.topo 
    for (state_id,element) in enumerate(RootIterator{D+1}(topo))
        vol_data = precompute_volume_monomials(element.id,topo,cv.facedata_col,Val(1))
        bc                     = vol_data.vol_bc
        hvol                   = vol_data.hvol 
        area                   = vol_data.integrals[1]

        states.χ_vec[state_id]               = ρ_init
        states.x_vec[state_id]               = bc
        states.area_vec[state_id]            = area
        states.h_vec[state_id]               = hvol
        states.el_id_to_state_id[element.id] = state_id
    end
end

function resize_states!(states::DesignVarInfo{D},
    n_states::Integer) where {D}
    
    resize!(states.χ_vec,n_states)
    resize!(states.x_vec,n_states)
    resize!(states.area_vec,n_states)
    resize!(states.h_vec,n_states)
    empty!(states.el_id_to_state_id)
end


function DesignVarInfo{D}(cv::CellValues{D,U},ρ_init) where {D,U}


    mesh = cv.mesh
    topo = mesh.topo

    n_states = length(RootIterator{D+1}(topo))

    χ_vec              = Vector{Float64}(undef,n_states)
    x_vec              = Vector{SVector{D,Float64}}(undef, n_states)
    area_vec           = Vector{Float64}(undef,n_states)
    h_vec              = Vector{Float64}(undef,n_states)
    el_id_to_state_id  = OrderedDict{Int32,Int32}()

    states = DesignVarInfo(χ_vec,x_vec,area_vec,h_vec,el_id_to_state_id)
    fill_states!(states,cv,ρ_init)

    return states
end



function update_states_after_mesh_adaption!(states::DesignVarInfo,
    cv::CellValues{D,U},
    ref_marker::Vector{Bool},
    coarse_marker::Vector{Bool}) where {D,U}	

    topo      = cv.mesh.topo
    e2s_old   = copy(states.el_id_to_state_id)
    χ_vec_old = copy(states.χ_vec)

    n_states = length(RootIterator{4}(topo))
    resize_states!(states,n_states)
    fill_states!(states,cv,0.0) # here rho init is just a dummy

    for (el_id,state_id) in states.el_id_to_state_id
        element = get_volumes(topo)[el_id]
        parent_id = element.parent_id 
        child_ids = element.childs 

        #Info: element comes from refinement
        if parent_id != 0 && ref_marker[parent_id]
            parents_old_state_id = e2s_old[parent_id]
            χ_parent             = χ_vec_old[parents_old_state_id]
            # element gets parent density 
            states.χ_vec[state_id] = χ_parent
        elseif !isempty(child_ids) && any(coarse_marker[child_ids])
            #Info: element comes from coarseing 

            χ_sum = 0.0 
            for child_id in child_ids 
                χ_sum += χ_vec_old[e2s_old[child_id]]
            end
            states.χ_vec[state_id] = χ_sum / length(child_ids)

        else #INFO: element is not new, just state_id changed
            states.χ_vec[state_id] = χ_vec_old[e2s_old[element.id]] #!FIX
        end
    end
    return states
end







function measure_of_nondiscreteness(
    states::DesignVarInfo,
    pars::SimPars)


    ∑Ω  = 0.0 
    ∑Ωχ = 0.0 
    χmin = pars.χmin
    for (χ,el_size) in zip(states.χ_vec,states.area_vec) 
        ∑Ω  += el_size
        ∑Ωχ += el_size * (1 - χ)*(χ-χmin)
    end
    return 4∑Ωχ/∑Ω
end