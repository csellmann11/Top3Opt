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


function find_distance_to_boundary(
    vol_id, 
    topo, 
    x0::SVector{3,Float64}, 
    d::SVector{3,Float64})

    min_rel_dist = typemax(Float64)

    face_ids = get_volume_area_ids(topo, vol_id)
    
    @inbounds for face_id in face_ids
        node_ids = get_area_node_ids(topo, face_id)
        
        length(node_ids) < 3 && continue
        
        p1 = topo.nodes[node_ids[1]]
        p2 = topo.nodes[node_ids[2]]
        p3 = topo.nodes[node_ids[3]]
        
        e1 = p2 - p1
        e2 = p3 - p1
        n = e1 × e2
        
        denom = n ⋅ d
        abs(denom) < eps(Float64) && continue
        
        t = (n ⋅ (p1 - x0)) / denom
        
        # Accept both positive AND negative t, take minimum absolute value
        abs_t = abs(t)
        if abs_t > eps(Float64) && abs_t < min_rel_dist
            min_rel_dist = abs_t
        end
    end
    
    return min_rel_dist
end



function get_location(
    n_id::Integer,
    state_id::Integer,
    x0::SVector{3,Float64},
    topo::Topology{3},
    b_face_id_to_state_id::Dict{Int32,Int32},
    states::DesignVarInfo{3},
    laplace_rescale::Bool = true
    )
    x = if n_id < 0 # we are on the boundary 
        face_node_ids = get_area_node_ids(topo,abs(n_id))
        state_id_face = b_face_id_to_state_id[n_id]
        bc_vol_normal = states.x_vec[state_id_face]

        _,_,n_unsigned,_,p0 = Ju3VEM.VEMGeo.get_plane_parameters(@view(topo.nodes[face_node_ids]))
        n = Ju3VEM.VEMGeo.get_outward_normal(bc_vol_normal,n_unsigned,p0)
        # Compute the mirrored neighbor barycenter
        mirror_across_face(bc_vol_normal, p0, n)
    else
        
        _x  = states.x_vec[n_id]
        _x = if laplace_rescale
            e0id = get_el_id(states,state_id) 
            e1id = get_el_id(states,n_id) 
            rd1  = find_distance_to_boundary(e0id,topo,x0,_x - x0)
            rd2  = find_distance_to_boundary(e1id,topo,_x,x0 - _x)
            gap_rel = (1 - rd1 - rd2)
            (2rd1+gap_rel) * (_x - x0) + x0   
            # (2*max(rd1,rd2)+gap_rel) * (_x - x0) + x0   
        else    
            _x
        end
    end
    return x
end



function compute_d_mat!(
    res_cache::CachedVector{Float64},
    state_id  ::Int,
    local_neighs ::AbstractVector{Int32},
    b_face_id_to_state_id ::Dict{Int32,Int32},
    topo::Topology{D},
    states::DesignVarInfo{D},
    laplace_rescale::Bool = true) where D

    
    n_neighs   = length(local_neighs)
    h0         = states.h_vec[state_id]
    bc         = states.x_vec[state_id]

    setsize!(res_cache,(n_neighs,))

    @no_escape begin
        A     = @alloc(Float64,n_neighs,6)
        w_vec = @alloc(Float64,n_neighs)
        W_A   = @alloc(Float64,n_neighs,6)

        for (n_count,n_id) in enumerate(local_neighs) 
            
            y = get_location(n_id,state_id,bc,topo,b_face_id_to_state_id,states,laplace_rescale)

            dx,dy,dz = (y-bc)/h0
            w_vec[n_count] = 1.0#weight_factor(norm(dist_vec),2*hmin^2)
     
            A[n_count,:] .= (dx,dy,dz,0.5*dx^2,0.5*dy^2,0.5*dz^2)
        end


        W_A .= w_vec .*A
        AT_A = static_matmul(A',W_A,Val((6,6)))

 
        reg_α   = 1e-04*sum(AT_A[i,i] for i in 1:6)
        invAT_A = inv(AT_A + reg_α*I)
        res     = res_cache.array

        for i in axes(A, 1)
            res[i] = zero(eltype(res))
            for j in axes(A, 2)
                # res[i] += (invAT_A[4,j] + invAT_A[7,j] + invAT_A[9,j]) * W_A[i,j] / h0^2
                res[i] += (invAT_A[4,j] + invAT_A[5,j] + invAT_A[6,j]) * W_A[i,j] / h0^2
            end
        end
    end
    return res_cache
end


function compute_d_mat_full!(
    res_cache::CachedVector{Float64},
    state_id  ::Int,
    local_neighs ::AbstractVector{Int32},
    b_face_id_to_state_id ::Dict{Int32,Int32},
    topo::Topology{D},
    states::DesignVarInfo{D},
    laplace_rescale::Bool = true) where D

    
    n_neighs   = length(local_neighs)
    h0         = states.h_vec[state_id]
    bc         = states.x_vec[state_id]

    setsize!(res_cache,(n_neighs,))

    @no_escape begin
        A     = @alloc(Float64,n_neighs,9)
        w_vec = @alloc(Float64,n_neighs)
        W_A   = @alloc(Float64,n_neighs,9)

        for (n_count,n_id) in enumerate(local_neighs) 
            
            y = get_location(n_id,state_id,bc,topo,b_face_id_to_state_id,states,laplace_rescale)

            dx,dy,dz = (y-bc)/h0
            w_vec[n_count] = 1.0#weight_factor(norm(dist_vec),2*hmin^2)
     
            A[n_count,:] .= (dx,dy,dz,0.5*dx^2,dx*dy,0.5*dy^2,dx*dz,dy*dz,0.5*dz^2)
        end


        W_A .= w_vec .*A
        AT_A = static_matmul(A',W_A,Val((9,9)))

 
        reg_α   = 1e-04*sum(AT_A[i,i] for i in 1:9)
        invAT_A = inv(AT_A + reg_α*I)
        res     = res_cache.array

        for i in axes(A, 1)
            res[i] = zero(eltype(res))
            for j in axes(A, 2)
                # res[i] += (invAT_A[4,j] + invAT_A[7,j] + invAT_A[9,j]) * W_A[i,j] / h0^2
                res[i] += (invAT_A[4,j] + invAT_A[6,j] + invAT_A[9,j]) * W_A[i,j] / h0^2
            end
        end
    end
    return res_cache
end




function compute_laplace_operator_mat(
    topo::Topology{D},
    state_neights_col::FixedSizeVector{Vector{Int32}},
    b_face_id_to_state_id::Dict{Int32,Int32},
    states::DesignVarInfo{D},
    laplace_rescale::Bool = true
    ) where D


    n_entries = sum(1 + length(local_neighs) for local_neighs in values(state_neights_col))

    e2s = states.el_id_to_state_id

    rows     = Vector{Int}(undef,n_entries)
    cols     = Vector{Int}(undef,n_entries)
    vals     = Vector{Float64}(undef,n_entries)
    res_cache = CachedVector(Float64)

    sparse_mat_counter = 1 


    # for element in RootIterator{D+1}(topo)
    for element_id in e2s.keys
        # state_id = e2s[element.id]
        state_id = get_state_id(states,element_id)
        local_neighs = state_neights_col[state_id]

        compute_d_mat_full!(res_cache,
            state_id,local_neighs,b_face_id_to_state_id,topo,states,laplace_rescale)

        dΔ_sum =  0.0 

        for (n_id,d_row_i) in zip(local_neighs,res_cache.array)
            rows[sparse_mat_counter] = state_id
            cols[sparse_mat_counter] = n_id < 0 ? b_face_id_to_state_id[n_id] : n_id
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