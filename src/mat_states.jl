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
    TopStates{D}

A struct representing the states of elements in a topology optimization problem.

# Fields
- `χ_vec::FlattenVecs{2,Float64}`: A flattened vector storing the state values (χ) for both inner and ghost states.
- `x_vec::Vector{SVector{D,Float64}}`: A vector storing the coordinates of the centroids for inner states.
- `el_id_to_state_id::Dict{Int,Int}`: A dictionary mapping element IDs to their corresponding state IDs.

# Type Parameters
- `D`: The dimension of the problem (e.g., 2 for 2D problems).

# Notes
- The `χ_vec` field uses `FlattenVecs` to efficiently store both inner and ghost states.
- Only the coordinates of inner states are stored in `x_vec`.
- The `el_id_to_state_id` dictionary allows for quick lookup of state IDs given an element ID.
"""
struct TopStates{D}
    χ_vec::Vector{Float64} 
    x_vec::Vector{SVector{D,Float64}}
    area_vec::Vector{Float64} #TODO: rename to elsize_vec
    h_vec::Vector{Float64} 

    el_id_to_state_id::Dict{Int,Int}
end



function fill_states!(states::TopStates{D},cv::CellValues{D,U},ρ_init) where {D,U}

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

function resize_states!(states::TopStates{D},
    n_states::Int) where {D}
    
    resize!(states.χ_vec,n_states)
    resize!(states.x_vec,n_states)
    resize!(states.area_vec,n_states)
    resize!(states.h_vec,n_states)
    empty!(states.el_id_to_state_id)
end


function TopStates{D}(cv::CellValues{D,U},ρ_init) where {D,U}


    mesh = cv.mesh
    topo = mesh.topo

    n_states = length(RootIterator{D+1}(topo))

    χ_vec              = Vector{Float64}(undef,n_states)
    x_vec              = Vector{SVector{D,Float64}}(undef, n_states)
    area_vec           = Vector{Float64}(undef,n_states)
    h_vec              = Vector{Float64}(undef,n_states)
    el_id_to_state_id  = Dict{Int,Int}()

    states = TopStates(χ_vec,x_vec,area_vec,h_vec,el_id_to_state_id)
    fill_states!(states,cv,ρ_init)

    return states
end



# function update_states_after_mesh_adaption!(states::TopStates,
#     cv::CellValues{D,U},
#     ref_marker::Vector{Bool},
#     coarse_marker::Vector{Bool}) where {D,U}	

#     topo      = cv.mesh.topo
#     e2s_old   = copy(states.el_id_to_state_id)
#     χ_vec_old = copy(states.χ_vec)

#     n_states = length(RootIterator{4}(topo))
#     resize_states!(states,n_states)
#     fill_states!(states,cv,0.0) # here rho init is just a dummy

#     e2s = states.el_id_to_state_id

#     for element in RootIterator{4}(topo) 
#         parent_id = element.parent_id 
#         child_ids = element.childs 

#         #Info: element comes from refinement
#         if parent_id != 0 && ref_marker[parent_id]
#             parents_old_state_id = e2s_old[parent_id]
#             χ_parent             = χ_vec_old[parents_old_state_id]
#             # element gets parent density 
#             states.χ_vec[e2s[element.id]] = χ_parent
#         elseif !isempty(child_ids) && all(coarse_marker[child_ids])
#             #Info: element comes from coarseing 

#             χ_sum = 0.0 
#             for child_id in child_ids 
#                 χ_sum += χ_vec_old[e2s_old[child_id]]
#             end
#             states.χ_vec[e2s[element.id]] = χ_sum / length(child_ids)

#         else #INFO: element is not new, just state_id changed
#             states.χ_vec[e2s[element.id]] = χ_vec_old[e2s_old[element.id]]
#         end
#     end
#     return states
# end


function handle_state_change_for_refinement!(
    parent_id::Int, 
    cv::CellValues{D,U},
    eldata_col::Dict{Int},
    states::TopStates{D},
    χ_vec_old::Vector{Float64},
    e2s_old::Dict{Int,Int},
    node_states::Dict{Int,Float64}) where {D,U} 

    parent = get_volumes(cv.mesh.topo)[parent_id]
    node_ids = eldata_col[parent_id].node_ids 
    χ_nodes_vec = [node_states[node_id] for node_id in node_ids]
    proj_s      = stretch(eldata_col[parent_id].proj_s,Val(1))
    base        = get_base(BaseInfo{3,1,1}())
    χ_π         = sol_proj(base,χ_nodes_vec,proj_s)

    parent_state_id = e2s_old[parent_id] 
    parent_χ        = χ_vec_old[parent_state_id]
    parent_vol      = eldata_col[parent_id].volume
    parent_mass     = parent_χ * parent_vol
    parent_bc       = states.x_vec[parent_state_id]
    parent_h        = states.h_vec[parent_state_id]
    child_state_ids  = [states.el_id_to_state_id[child_id] for child_id in parent.childs]

    child_mass = 0.0
    iter = 0
    λ_trial = 0.0
    λ_lower, λ_upper = -100.0,100.0

    while abs(parent_mass - child_mass) > 1e-8
        iter += 1 
        child_mass = 0.0 

        for child_state_id in child_state_ids  
            bc = states.x_vec[child_state_id]
            chi_pi = χ_π((bc - parent_bc)/parent_h)
            χ_child = clamp(chi_pi - λ_trial,1e-03,1.0)
            states.χ_vec[child_state_id] = χ_child
            child_mass += χ_child * states.area_vec[child_state_id]
        end

        child_mass > parent_mass ? λ_lower = λ_trial : λ_upper = λ_trial 
        λ_trial = (λ_lower + λ_upper)/2.0
        if iter > 1000
            error("Max iterations reached")
            break
        end
    end
end


function update_states_after_mesh_adaption!(states::TopStates,
    cv::CellValues{D,U},
    eldata_col::Dict{Int},
    ref_marker::Vector{Bool},
    coarse_marker::Vector{Bool}) where {D,U}	

    topo      = cv.mesh.topo
    e2s_old   = copy(states.el_id_to_state_id)
    χ_vec_old = copy(states.χ_vec)
    
    # node_sums   = zeros(Float64,length(cv.mesh.topo.nodes))
    # node_states = Dict{Int,Float64}()
    # for (el_id,el_data) in eldata_col
    #     sid      = e2s_old[el_id]
    #     χ_el     = χ_vec_old[sid]
    #     bc       = states.x_vec[sid]
    #     node_ids = el_data.node_ids
        
    #     for node_id in node_ids
    #         weight =  get!(node_states,node_id,0.0)
    #         node    = cv.mesh.topo.nodes[node_id]
    #         d       = node.coords - bc
    #         weight += χ_el/norm(d)
    #         node_states[node_id] = weight
    #         node_sums[node_id] += 1/norm(d)
    #     end
    # end
    # for node_id in keys(node_states)
    #     node_states[node_id] /= node_sums[node_id]
    # end


    n_states = length(RootIterator{4}(topo))
    resize_states!(states,n_states)
    fill_states!(states,cv,0.0) # here rho init is just a dummy

    e2s = states.el_id_to_state_id

    for element in RootIterator{4}(topo) 
        parent_id = element.parent_id 
        child_ids = element.childs 

        #Info: element comes from refinement
        if parent_id != 0 && ref_marker[parent_id]
            parents_old_state_id = e2s_old[parent_id]
            χ_parent             = χ_vec_old[parents_old_state_id]
            # element gets parent density 
            states.χ_vec[e2s[element.id]] = χ_parent
            # handle_state_change_for_refinement!(parent_id,cv,eldata_col,states,χ_vec_old,e2s_old,node_states)
            # ref_marker[child_ids] .= false
        elseif !isempty(child_ids) && any(coarse_marker[child_ids])
            #Info: element comes from coarseing 

            χ_sum = 0.0 
            for child_id in child_ids 
                χ_sum += χ_vec_old[e2s_old[child_id]]
            end
            states.χ_vec[e2s[element.id]] = χ_sum / length(child_ids)

        else #INFO: element is not new, just state_id changed
            states.χ_vec[e2s[element.id]] = χ_vec_old[e2s_old[element.id]]
        end
    end
    return states
end









function measure_of_nondiscreteness(
    states::TopStates,
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