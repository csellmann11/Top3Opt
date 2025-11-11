using Ju3VEM.VEMUtils: static_matmul

"""
    mirror_across_face(bc, p0, n)

Compute the mirrored position of a barycenter `bc` across a boundary face.

# Arguments
- `bc`: Current element barycenter (center point to be mirrored)
- `p0`: A point on the boundary face
- `n`: Normalized normal vector of the boundary face

# Returns
- The mirrored barycenter position on the "virtual" neighbor side

# Mathematical Description
The function mirrors point `bc` across the plane defined by point `p0` and normal `n`.
It computes the signed distance from `bc` to the plane and reflects it across.
"""
function mirror_across_face(bc, p0, n)
    # Compute signed distance from bc to the plane
    d = dot(p0 - bc, n)
    # Mirror the barycenter across the face
    return bc + 2 * d * n
end


@inline weight_factor(d,β) = exp(-0.5*(4d/√(β))^2)


function compute_d_mat(
    state_id  ::Int,
    el_neighs ::AbstractVector{Int},
    face_to_vols ::Dict{Int,Vector{Int}},
    topo::Topology{D},
    states::TopStates{D},
    sim_pars::SimPars) where D

    # hmin,hmax = extrema(states.h_vec)
    # β = 2*hmax^2 * sim_pars.β0
    
    n_neighs  = length(el_neighs)
    h0         = states.h_vec[state_id]
    bc         = states.x_vec[state_id]
    e2s        = states.el_id_to_state_id

    A          =  FixedSizeMatrix{Float64}(undef,n_neighs,9)

    w_vec     = FixedSizeVector{Float64}(undef,n_neighs)
    n_state_ids = FixedSizeVector{Int}(undef,n_neighs)

    for (n_count,n_id) in enumerate(el_neighs) 
         
         
        if n_id < 0 # we are on the boundary 
            face_node_ids = get_area_node_ids(topo,abs(n_id))
            face_el_id = face_to_vols[abs(n_id)] |> only 
            state_id_face = e2s[face_el_id]
            bc_vol_normal = states.x_vec[state_id_face]
            n_state_ids[n_count] = state_id_face

            _,_,n_unsigned,_,p0 = Ju3VEM.VEMGeo.get_plane_parameters(@view(topo.nodes[face_node_ids]))
            n = Ju3VEM.VEMGeo.get_outward_normal(bc_vol_normal,n_unsigned,p0)
            # Compute the mirrored neighbor barycenter
            n_bc = mirror_across_face(bc_vol_normal, p0, n)
            λ_scale = 1.0
        else
            n_state_id = e2s[n_id]
            h_n = states.h_vec[n_state_id]
            n_bc = states.x_vec[n_state_id]
            λ_scale = h0*inv(0.5*(h0+h_n))
        end

        dx,dy,dz = λ_scale*(n_bc - bc)
        w_vec[n_count] = 1.0#weight_factor(norm(n_bc - bc),β)
        A[n_count,:] .= (dx,dy,dz,0.5*dx^2,dx*dy,dx*dz,0.5*dy^2,dy*dz,0.5*dz^2)
    end


    W_A = w_vec .*A
    AT_A = static_matmul(A',W_A,Val((9,9)))

    # if cond(AT_A) > 1e9 
    #     display(bc)
    #     throw("Condition number of AT_A is too high")

    # end

    invAT_A = inv(AT_A)
    res  = FixedSizeVector{Float64}(undef,n_neighs)

    for i in axes(A, 1)
        res[i] = zero(eltype(res))
        for j in axes(A, 2)
            res[i] += (invAT_A[4,j] + invAT_A[7,j] + invAT_A[9,j]) * W_A[i,j]
        end
    end
    return res, n_state_ids
end




function compute_laplace_operator_mat(
    topo::Topology{D},
    elneighs_col::Dict,
    face_to_vols::Dict{Int,Vector{Int}},
    states::TopStates{D},
    sim_pars::SimPars
    ) where D


    n_entries = sum(1 + length(local_neighs) for local_neighs in values(elneighs_col))

    e2s = states.el_id_to_state_id

    rows     = Vector{Int}(undef,n_entries)
    cols     = Vector{Int}(undef,n_entries)
    vals     = Vector{Float64}(undef,n_entries)

    sparse_mat_counter = 1 


    for element in RootIterator{D+1}(topo)
        state_id = e2s[element.id]
        local_neighs = elneighs_col[element.id]

        d_row, n_state_ids = compute_d_mat(
            state_id,local_neighs,face_to_vols,topo,states,sim_pars)

        dΔ_sum =  0.0 

        for (n_count,(n_id,d_row_i)) in enumerate(zip(local_neighs,d_row))
            rows[sparse_mat_counter] = state_id
            cols[sparse_mat_counter] = n_id < 0 ? n_state_ids[n_count] : e2s[n_id]
            vals[sparse_mat_counter] = d_row_i

            dΔ_sum += d_row_i
            sparse_mat_counter += 1
        end

        rows[sparse_mat_counter] = state_id
        cols[sparse_mat_counter] = state_id
        vals[sparse_mat_counter] = -dΔ_sum
        sparse_mat_counter += 1
    end


    laplace_operator = sparse(rows,cols,vals,length(states.χ_vec),length(states.χ_vec))
    return laplace_operator
end