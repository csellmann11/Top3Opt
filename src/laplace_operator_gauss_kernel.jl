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

@inline function gauss_kernel(
    x::StaticVector{D,Float64},
    y::StaticVector{D,Float64};
    shape_parameter = 0.1) where D

    return exp(-(shape_parameter*norm(x-y))^2)
end

Base.@propagate_inbounds function polynomial_kernel(
    x::StaticVector{D,Float64},
    x0::StaticVector{D,Float64},
    h ::Float64,
    idx::Integer
    ) where D

    @boundscheck ((1 <= idx <= 10) || throw(BoundsError(x,1:10)))

    d = x#(x - x0) / h
 
    if idx == 1
        return 1.0 
    elseif idx == 2 
        return d[1]
    elseif idx == 3
        return d[2]
    elseif idx == 4
        return d[3]
    elseif idx == 5
        return d[1]^2
    elseif idx == 6
        return d[1]*d[2]
    elseif idx == 7
        return d[2]^2
    elseif idx == 8
        return d[1]*d[3]
    elseif idx == 9
        return d[2]*d[3]
    elseif idx == 10
        return d[3]^2
    end 
end

Base.@propagate_inbounds function polynomial_laplacian_kernel(h::Float64,idx::Integer)

    @boundscheck ((1 <= idx <= 10) || throw(BoundsError(idx,1:10)))

    if idx == 5 || idx == 7 || idx == 10 
        return 2.0#/h^2 
    else
        return 0.0
    end
end

@inline function laplacian_gauss_kernel(x::StaticVector{D,Float64}, 
    y::StaticVector{D,Float64}; shape_parameter=0.1) where D
    r = norm(x - y)
    ε = shape_parameter
    return ε^2 * (4*ε^2*r^2 - 2*D) * exp(-ε^2 * r^2)
end


Base.@kwdef struct LaplaceKernelSettings
    # Kernel identifier. Supported: :gaussian, :polyharmonic, :wendland
    kind::Symbol = :polyharmonic
    # Scaling factor used to derive the Gaussian inverse length scale from the local spacing.
    scale_factor::Float64 = 1 / 32
    # Support radius multiplier for compact kernels such as Wendland.
    support_factor::Float64 = 1300.5
    # Relative smoothing radius used by the polyharmonic kernel to keep the matrix well-conditioned.
    polyharmonic_epsilon::Float64 = 1e-3
    # Diagonal Tikhonov regularisation that stabilises very small neighborhood systems.
    diag_regularization::Float64 = 1e-10
end

@inline function _estimate_reference_length(
    nodes::Vector{SVector{D,Float64}},
    x0::SVector{D,Float64},
    hmin::Float64,
    h0::Float64
) where D
    total = 0.0
    count = 0
    for node in nodes
        dist = norm(node - x0)
        dist <= eps(Float64) && continue
        total += dist
        count += 1
    end

    if count == 0
        return max(h0, hmin)
    end

    avg = total / count
    return max(avg, hmin)
end

@inline function _kernel_parameters(
    settings::LaplaceKernelSettings,
    reference_length::Float64
)
    ref = max(reference_length, eps(Float64))
    if settings.kind === :gaussian
        ε = settings.scale_factor / ref
        return (; ε = ε)
    elseif settings.kind === :polyharmonic
        δ = max(settings.polyharmonic_epsilon * ref, eps(Float64))
        return (; δ = δ)
    elseif settings.kind === :wendland
        ρ = max(settings.support_factor * ref, eps(Float64))
        return (; ρ = ρ)
    else
        throw(ArgumentError("Unsupported kernel kind $(settings.kind). Available kinds are :gaussian, :polyharmonic and :wendland."))
    end
end

@inline function _kernel_value(
    settings::LaplaceKernelSettings,
    r::Float64,
    params
)
    if settings.kind === :gaussian
        ε = params.ε
        return exp(-(ε * r)^2)
    elseif settings.kind === :polyharmonic
        δ = params.δ
        return (r^2 + δ^2)^(1.5)
    elseif settings.kind === :wendland
        ρ = params.ρ
        if r >= ρ
            return 0.0
        end
        q = r / ρ
        s = 1 - q
        return s^4 * (4q + 1)
    else
        throw(ArgumentError("Unsupported kernel kind $(settings.kind)."))
    end
end

@inline function _kernel_laplacian(
    settings::LaplaceKernelSettings,
    r::Float64,
    D::Int,
    params
)
    if settings.kind === :gaussian
        ε = params.ε
        return ε^2 * (4 * ε^2 * r^2 - 2 * D) * exp(-ε^2 * r^2)
    elseif settings.kind === :polyharmonic
        δ = params.δ
        s = sqrt(r^2 + δ^2)
        if r <= eps(Float64)
            return 3 * D * δ
        end
        fpp = 3 * (2r^2 + δ^2) / s
        return fpp + (D - 1) / r * (3r * s)
    elseif settings.kind === :wendland
        ρ = params.ρ
        if r >= ρ
            return 0.0
        end
        q = r / ρ
        s = 1 - q
        return (-20 / ρ^2) * s^2 * (D - (D + 3) * q)
    else
        throw(ArgumentError("Unsupported kernel kind $(settings.kind)."))
    end
end


function find_distance_to_boundary(
    vol_id::Integer, 
    topo::Topology{3}, 
    x0::SVector{3,Float64}, 
    d::SVector{3,Float64}
    )
    min_rel_dist = typemax(Float64)

    face_ids = get_volume_area_ids(topo, vol_id)

    norm(d) < eps(Float64) && return 0.0
    
    @inbounds for face_id in face_ids
        node_ids = get_area_node_ids(topo, face_id)
        
        length(node_ids) < 3 && continue
        
        p1 = topo.nodes[node_ids[1]]
        p2 = topo.nodes[node_ids[2]]
        p3 = topo.nodes[node_ids[3]]

        n = (p2 - p1) × (p3 - p1)
 
        denom = n ⋅ d
        abs(denom) < eps(Float64) && continue
        
        t = (n ⋅ (p1 - x0)) / denom
        
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
    states::DesignVarInfo{3}
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
        # if norm(_x - x0) < eps(Float64)
        #     _x
        # else
        #     h0  = states.h_vec[state_id] 
        #     h1  = states.h_vec[n_id]
        #     _xx = _x - x0
        #     2*h0*_xx/(h0+h1) + x0
        # end

        e0id = get_el_id(states,state_id) 
        e1id = get_el_id(states,n_id) 
        rd1  = find_distance_to_boundary(e0id,topo,x0,_x - x0)
        rd2  = find_distance_to_boundary(e1id,topo,_x,x0 - _x)
        gap_rel = (1 - rd1 - rd2)
        (2rd1+gap_rel) * (_x - x0) + x0       
        
        _x
    end
    return x
end


function compute_d_mat_poly!(
    res_cache::CachedVector{Float64},
    state_id  ::Int,
    local_neighs ::AbstractVector{Int32},
    b_face_id_to_state_id ::Dict{Int32,Int32},
    topo::Topology{D},
    states::DesignVarInfo{D}
    ) where D


    
    n_neighs  = length(local_neighs) 
    x0        = states.x_vec[state_id]
    h0 = states.h_vec[state_id]

    setsize!(res_cache,(n_neighs,))

    @no_escape begin
        A     = @alloc(Float64,n_neighs,10)
        b     = MVector{10,Float64}(undef)
        

        for i in 1:10 
            
            b[i] = polynomial_laplacian_kernel(h0,i)

            for (j,ni_id) in enumerate(local_neighs) 
                y = get_location(ni_id,state_id,x0,topo,b_face_id_to_state_id,states)
                A[j,i] = polynomial_kernel(y,x0,h0,i)
            end
        end
        # M = static_matmul(A',A,Val((10,10)))
        # _temp = inv(M) * b 
        # mul!(res_cache.array,A,_temp)
        # res = A * inv(M) * b

        res_cache.array .= qr(A)'\b
    end
    return res_cache
end



@inline function _min_neighbor_spacing(
    local_neighs::AbstractVector{Int32},
    states::DesignVarInfo,
    state_id::Int
)
    min_h = states.h_vec[state_id]
    for ni in local_neighs
        ni > 0 || continue
        min_h = min(min_h, states.h_vec[ni])
    end
    return min_h
end

function compute_d_mat_gauss!(
    res_cache::CachedVector{Float64},
    state_id  ::Int,
    local_neighs ::AbstractVector{Int32},
    b_face_id_to_state_id ::Dict{Int32,Int32},
    topo::Topology{D},
    states::DesignVarInfo{D},
    kernel_settings::LaplaceKernelSettings = LaplaceKernelSettings()
    ) where D

    n_neighs = length(local_neighs)
    setsize!(res_cache, (n_neighs,))

    
    n_neighs == 0 && return res_cache

    x0 = states.x_vec[state_id]
    h0 = states.h_vec[state_id]
    hmin = _min_neighbor_spacing(local_neighs, states, state_id)
    kernel_settings = LaplaceKernelSettings(kind=:polyharmonic,scale_factor=1/16,polyharmonic_epsilon = 1)

    nodes = Vector{SVector{D,Float64}}(undef, n_neighs)

    @no_escape begin
        A = @alloc(Float64, n_neighs, n_neighs)
        b = @alloc(Float64, n_neighs)

        for (i, ni_id) in enumerate(local_neighs)
            nodes[i] = get_location(ni_id, state_id, x0, topo, b_face_id_to_state_id, states)
        end

        reference_length = _estimate_reference_length(nodes, x0, hmin, h0)
        kernel_params = _kernel_parameters(kernel_settings, reference_length)
        diag_entry = _kernel_value(kernel_settings, 0.0, kernel_params) + 0*kernel_settings.diag_regularization

        for i in 1:n_neighs
            xi = nodes[i]
            A[i, i] = diag_entry
            b[i] = _kernel_laplacian(kernel_settings, norm(xi - x0), D, kernel_params)

            @inbounds for j in (i+1):n_neighs
                val = _kernel_value(kernel_settings, norm(xi - nodes[j]), kernel_params)
                A[i, j] = val
                A[j, i] = val
            end
        end
        if state_id == 1
            @show cond(A)
        end

        res_cache.array .= A \ b
    end

    return res_cache
end



function compute_laplace_operator_mat_gauss(
    topo::Topology{D},
    state_neights_col::FixedSizeVector{Vector{Int32}},
    b_face_id_to_state_id::Dict{Int32,Int32},
    states::DesignVarInfo{D};
    kernel_settings::LaplaceKernelSettings = LaplaceKernelSettings()
    ) where D


    n_entries = sum(length(local_neighs) for local_neighs in values(state_neights_col))

    e2s = states.el_id_to_state_id

    rows     = Vector{Int}(undef,n_entries)
    cols     = Vector{Int}(undef,n_entries)
    vals     = Vector{Float64}(undef,n_entries)
    res_cache = CachedVector(Float64)

    sparse_mat_counter = 1 

    for element_id in e2s.keys 
        state_id = get_state_id(states,element_id)
        local_neighs = state_neights_col[state_id]

        compute_d_mat_gauss!(res_cache,
                state_id,local_neighs,b_face_id_to_state_id,topo,states,kernel_settings)


        for (n_id,d_row_i) in zip(local_neighs,res_cache.array)
            rows[sparse_mat_counter] = state_id
            cols[sparse_mat_counter] = n_id < 0 ? b_face_id_to_state_id[n_id] : n_id
            vals[sparse_mat_counter] = d_row_i

            sparse_mat_counter += 1
        end
    end


    laplace_operator = sparse(rows,cols,vals,length(states.χ_vec),length(states.χ_vec))
    return laplace_operator
end