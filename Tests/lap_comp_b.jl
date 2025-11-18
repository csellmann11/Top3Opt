using Ju3VEM


function mirror_across_face(bc, p0, n)
    # Compute signed distance from bc to the plane
    d = dot(p0 - bc, n)
    # Mirror the barycenter across the face
    return bc + 2 * d * n
end


function project_states_to_nodes(
    cv::CellValues{D,U},
    states::DesignVarInfo) where {D,U}

    node_sums   = zeros(Float64,length(cv.mesh.topo.nodes))
    node_states = zeros(Float64,length(cv.mesh.topo.nodes))
    e2s = states.el_id_to_state_id

    projectors  = Dict{Int,FixedSizeMatrix{Float64}}()
    el_node_ids = Dict{Int,FixedSizeVector{Int}}()


    for (el_id,sid) in e2s
        bc       = states.x_vec[sid]
        χi       = states.χ_vec[sid]
        reinit!(el_id,cv)
        proj_s, _ = create_volume_vem_projectors(
            el_id,cv.mesh,cv.volume_data,cv.facedata_col,cv.vnm)
        
        projectors[el_id] = proj_s

        node_ids = cv.vnm.map.keys
        el_node_ids[el_id] = FixedSizeVector(node_ids)
        
        for node_id in node_ids
            weight =  node_states[node_id]
            node    = cv.mesh.topo.nodes[node_id]
            d       = norm(node - bc)
            weight += χi/d^2
            node_states[node_id] = weight
            node_sums[node_id] += 1/d^2
        end
    end
    for node_id in eachindex(node_states)
        node_states[node_id] /= node_sums[node_id]
    end

    return node_states, projectors, el_node_ids
end





function comp_lap(
    cv::CellValues,
    states::DesignVarInfo)

    topo = cv.mesh.topo
    χvec = states.χ_vec
    fdc  = cv.facedata_col


    face_neighs = Dict{Int,SVector{2,Int}}()
    for element in RootIterator{4}(topo)


        Ju3VEM.VEMGeo.iterate_volume_areas(
            topo,element.id) do face,_
                face_id = face.id
                neighs_vec = get(face_neighs,face_id,SA[0,0])
                vec = if neighs_vec[1] == 0 
                    neighs_vec = SA[element.id,0]
                else
                    neighs_vec = SA[neighs_vec[1],element.id]
                end
                face_neighs[face_id] = vec
            end
    end

    node_states, projectors, el_node_ids = project_states_to_nodes(cv,states)

    Δχv = zero(χvec)

    base = get_base(BaseInfo{3,1,1}())
    # compute the avarage laplacian as :
    # 1/V * ∫ Δχ dV = 1/V * ∫ ∇χ ⋅ n dS = 
    for element in RootIterator{4}(topo)

        lap = Ref(0.0)
        state_id = get_state_id(states,element.id)
        x0       = states.x_vec[state_id]
        χ0       = χvec[state_id]
        # vol      = volume_of_topo_volume(topo,element.id)
        proj     = projectors[element.id] 
        node_ids = el_node_ids[element.id]
        χvec_el  = @view node_states[node_ids]
        χpi      = sol_proj(base,χvec_el,stretch(proj,Val(1)))
        h        = states.h_vec[state_id]
        

        
        Ju3VEM.VEMGeo.iterate_volume_areas(
            topo,element.id) do face,_
                face_id = face.id
                local_neighs = face_neighs[face_id]

                # normal gradient at boundary is 0 => just continue here
                local_neighs[2] == 0 && return nothing 

                el_id_n    = local_neighs[1] == element.id ? local_neighs[2] : local_neighs[1]
                state_id_n = get_state_id(states,el_id_n) 

                face_bc2d    = get_bc(cv.facedata_col[face_id])

                face_node_ids = get_area_node_ids(topo,face_id)

                plane = Ju3VEM.VEMGeo.D2FaceParametrization(@view(topo.nodes[face_node_ids]))
                n_unsigned = plane.n; p0 = plane.p0
                face_bc = Ju3VEM.VEMGeo.project_to_3d(face_bc2d,plane)
                n = Ju3VEM.VEMGeo.get_outward_normal(x0,n_unsigned,p0)
                grad_χpi = ∇x(χpi,h,(face_bc-x0)/h)
                normal_deriv_integral = dot(grad_χpi,n) * get_area(fdc[face_id])
                
                
                # Δχv[state_id_n] -= normal_deriv_integral/2
                # Δχv[state_id]   += normal_deriv_integral/2
       
                # xn         = states.x_vec[state_id_n] 

                # face_bc    = get_bc(cv.facedata_col[face_id])
                



                state_id_n = get_state_id(states,el_id_n)
                xn = states.x_vec[state_id_n]
                χn = χvec[state_id_n]
                r = norm(xn - x0)
                face_size      = get_area(fdc[face_id])
                

                face_node_ids = get_area_node_ids(topo,face_id)
                _,_,n_unsigned,_,p0 = Ju3VEM.VEMGeo.get_plane_parameters(@view(topo.nodes[face_node_ids]))
                n = Ju3VEM.VEMGeo.get_outward_normal(x0,n_unsigned,p0)
     
                xn_mirror = mirror_across_face(x0,p0,n)
                x0_mirror = mirror_across_face(xn,p0,-n)

                neigh_node_ids = el_node_ids[el_id_n]
                proj_neigh = projectors[el_id_n]
                χvec_neigh = @view node_states[neigh_node_ids]
                χpi_neigh = sol_proj(base,χvec_neigh,stretch(proj_neigh,Val(1)))
                h_neigh = states.h_vec[state_id_n]
                χn = χpi_neigh((xn_mirror-xn)/h_neigh)

                χ0_mirror = χpi((x0_mirror-x0)/h)
                # if isnan(r) || isnan(face_size) || r ≈ 0.0
                #     @show r 
                # end
                r = norm(xn_mirror - x0)
                normal_deriv_integral = (χn - χ0_mirror) / r * face_size
                Δχv[state_id] += normal_deriv_integral

            end

        # @show lap[]
        # Δχv[state_id] =  lap[]/vol
    end

    for element in RootIterator{4}(topo)
        vol      = volume_of_topo_volume(topo,element.id)
        state_id = get_state_id(states,element.id)
        Δχv[state_id] /= vol
    end

    return Δχv
end






