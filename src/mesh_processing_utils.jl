function find_maximal_cell_diameter(topo::Topology{3})
    maximal_diameter = -Inf

    for element in RootIterator{4}(topo)
        vol_node_ids = get_volume_node_ids(topo, element.id)

        diameter = max_node_distance(topo.nodes, vol_node_ids)
        maximal_diameter = max(maximal_diameter, diameter)
    end
    return maximal_diameter
end

function find_minimal_cell_diameter(topo::Topology{3})
    minimal_diameter = Inf

    for element in RootIterator{4}(topo)
        vol_node_ids = get_volume_node_ids(topo, element.id)

        diameter = max_node_distance(topo.nodes, vol_node_ids)
        minimal_diameter = min(minimal_diameter, diameter)
    end
    return minimal_diameter
end


function permute_coord_dimensions(mesh::Mesh{D,ET}, perm::SVector{D,Int}) where {D,ET}

    nodes = mesh.topo.nodes
    for i in eachindex(nodes)
        node = nodes[i]
        nodes[i] = Node(node.id, node.coords[perm])
    end
    return Mesh(mesh.topo, StandardEl{1}())
end


function face_area(nodes::AbstractVector{<:SVector{3}})
    n = length(nodes)
    area_vec = sum(i -> cross(nodes[i], nodes[mod1(i + 1, n)]), 1:n)
    return 0.5 * norm(area_vec)
end

function face_area(node_ids::AbstractVector{Int},topo::Topology{3})
    nodes = topo.nodes
    n = length(node_ids)
    area_vec = sum(i -> cross(nodes[node_ids[i]], nodes[node_ids[mod1(i + 1, n)]]), 1:n)
    return 0.5 * norm(area_vec)
end

# function remove_short_edges(mesh::Mesh{2})
#     topo = mesh.topo

#     edge_to_el = Dict{Int,Vector{Int}}()
#     for element in RootIterator{3}(topo)

#         Ju3VEM.VEMGeo.iterate_element_edges(
#             topo,element.id) do _,edge_id,_
#                 push!(edge_to_el[edge_id], element.id)
#         end
#     end

#     for (edge_id,adj_els) in edge_to_el
#         n1_id,n2_id = get_edge_node_ids(topo, edge_id) 
#         n1  = topo.nodes[n1_id]
#         n2  = topo.nodes[n2_id]
#         len = norm(n1.coords - n2.coords)

#         short_edge = false
#         for el_id in adj_els 
#             area_node_ids = get_area_node_ids(topo, el_id) 
#             if length(area_node_ids) == 3 
#                 short_edge = false 
#                 break
#             end
#             area = face_area(get_area_node_ids(topo, el_id), topo)
#             #edge is to short if len^2 < area/8
#             if len^2 < area/8
#                 short_edge = true
#             end
#         end
#     end



#     return Mesh(topo, StandardEl{Ju3VEM.VEMUtils.get_order(mesh)}())
# end









function refine_sets(cv::CellValues{3},
    sets_to_refine::T,
    MAX_REF_LEVEL::Int) where T<:Tuple 

    mesh = cv.mesh
    topo = mesh.topo

    fdc = cv.facedata_col

    ref_marker = zeros(Bool,length(get_volumes(topo)))

    n_els = length(get_volumes(topo))
    
    for element in RootIterator{4}(topo)

        Ju3VEM.VEMGeo.iterate_volume_areas(
            fdc,topo,element.id) do face,fd,_ 

                vertex_ids = fd.face_node_ids |> get_first
                for set_func in sets_to_refine
                    if any(set_func(topo.nodes[vid]) for vid in vertex_ids)
                        ref_marker[element.id] = true
                        return nothing
                    end
                end
        
                # ref_marker[element.id] && return nothing
                # for set_name in sets_to_refine 
                #     set = get(mesh.face_sets,set_name,nothing)

                #     set === nothing && continue 
                #     if face.id in set
                #         ref_marker[element.id] = true
                #         return nothing
                #     end
                # end

                # vertex_ids = fd.face_node_ids |> get_first
                # for (i,vid) in enumerate(vertex_ids)
                #     i_p1 = get_next_idx(vertex_ids,i)
                #     vid_p1 = vertex_ids[i_p1]
                #     edge = get_edge(vid,vid_p1,topo)
                #     for set_name in sets_to_refine 
                #         set = get(mesh.edge_sets,set_name,nothing)

                        
                #         if set !== nothing && edge.id in set
                #             ref_marker[element.id] = true
                #             return nothing
                #         end

                #         set = get(mesh.node_sets,set_name,nothing)
                #         if set !== nothing && vid in set
                #             ref_marker[element.id] = true
                #             return nothing
                #         end
                #     end
                # end
        end
    end

    for element in RootIterator{4}(topo)
        
        if element.refinement_level >= MAX_REF_LEVEL 
            ref_marker[element.id] = false
            continue
        end
        ref_marker[element.id] && _refine!(element,topo)
    end

  
    for _ in 3:MAX_REF_LEVEL 
        ref_marker = resize!(ref_marker,length(get_volumes(topo)))

        for element in RootIterator{4}(topo)
            element.id <= n_els && continue 
            _refine!(element,topo)
            ref_marker[element.id] = true
        end

    end

    mesh = Mesh(topo,StandardEl{1}())
    cv = CellValues{3}(mesh)

    return cv,ref_marker
end


function mark_elements_which_are_part_of_sets(cv::CellValues{3}, ch::ConstraintHandler{3})
    mesh = cv.mesh
    topo = mesh.topo
    forbid_coarsening = zeros(Bool,length(get_volumes(topo)))

    for element in RootIterator{4}(topo)
        
        reinit!(element.id,cv)
        dofs  = Ju3VEM.VEMUtils.get_cell_dofs(cv)
        forbidden = false

        for dof in dofs
            continue #! remove
            if haskey(ch.n_bcs,dof)
                forbidden = true
                break
            end
            if haskey(ch.d_bcs,dof)
                forbidden = true
                break
            end
        end

        forbid_coarsening[element.id] = forbidden
    end

    return forbid_coarsening
end


function get_sets_to_refine(b_case::Symbol)
    if b_case == :MBB_sym
        return ( x -> (0 ≤ x[1] ≤ 0.251) && x[3] ≈ 1.0,)
        # return (x -> x[1] ≈ 0.0 && x[3] ≈ 1.0,)
    elseif b_case == :Cantilever_sym
        return (x -> x[1] ≈ 2.0 && (0.4 ≤ x[3] ≤ 0.6) && (0.4 ≤ x[2] ≤ 0.6),)
    elseif b_case == :Bending_Beam_sym
        return (x -> x[1] ≈ 3.0 && x[3] ≈ 0.0,)
    elseif b_case == :simple_lever
        return (x -> x[1] ≈ 3.0 && x[3] >= 2.5 && x[2] >= 0.4,)
    elseif b_case == :pressure_plate
        return (x -> x[1] > 2.8 && x[2] > 2.8 && x[3] ≈ 1.0,)
    else
        ()
    end
end


function create_constraint_handler(cv::CellValues{3}, b_case::Symbol)
    mesh = cv.mesh
    ch = ConstraintHandler{U}(mesh)

    if b_case == :MBB_sym

        add_face_set!(mesh, "symmetry_bc", x -> x[1] ≈ 0.0)
        add_face_set!(mesh, "symmetry_bc_2", x -> x[2] ≈ 0.5)
        add_node_set!(mesh, "roller_bearing", x -> x[1] ≈ 3.0 && x[3] ≈ 0.0 && x[2] ≈ 0.0)
        add_face_set!(mesh, "middle_traction", x -> (0 ≤ x[1] ≤ 0.251) && x[3] ≈ 1.0)
        # add_edge_set!(mesh, "middle_traction", x -> (x[1] ≈ 0.0) && x[3] ≈ 1.0)

        
        add_dirichlet_bc!(ch, cv.dh, cv.facedata_col, "symmetry_bc", x -> SA[0.0], c_dofs=SA[1])
        add_dirichlet_bc!(ch, cv.dh, cv.facedata_col, "symmetry_bc_2", x -> SA[0.0], c_dofs=SA[2])
        add_dirichlet_bc!(ch, cv.dh, "roller_bearing", x -> SA[0.0, 0.0], c_dofs=SA[2, 3])
        add_neumann_bc!(ch, cv.dh, cv.facedata_col, "middle_traction", x -> SA[0.0, 0.0, -1.0])
        # add_neumann_bc!(ch, cv.dh, "middle_traction", x -> SA[0.0, 0.0, -1.0])

    elseif b_case == :Cantilever_sym 

        add_face_set!(mesh, "symmetry_bc", x -> x[2] ≈ 0.5)
        add_face_set!(mesh,"left_clamp", x -> x[1] ≈ 0.0)
        add_face_set!(mesh,"right_traction", x -> x[1] ≈ 2.0 && (0.4 ≤ x[3] ≤ 0.6) && (0.4 ≤ x[2] ≤ 0.6))

        add_dirichlet_bc!(ch, cv.dh, cv.facedata_col, "symmetry_bc", x -> SA[0.0], c_dofs=SA[2])
        add_dirichlet_bc!(ch, cv.dh, cv.facedata_col, "left_clamp", x -> SA[0.0,0.0,0.0], c_dofs=SA[1,2,3])
        add_neumann_bc!(ch, cv.dh, cv.facedata_col, "right_traction", x -> SA[0.0, 0.0, -1.0])
    elseif b_case == :Bending_Beam_sym
        add_face_set!(mesh, "symmetry_bc", x -> x[2] ≈ 0.5)
        add_node_set!(mesh, "left_clamp1", x -> x[1] ≈ 0.0 && x[2] ≈ 0.0 && x[3] ≈ 0.0)
        add_node_set!(mesh, "left_clamp2", x -> x[1] ≈ 0.0 && x[2] ≈ 0.0 && x[3] ≈ 1.0)
        add_edge_set!(mesh, "right_traction", x -> x[1] ≈ 3.0 && x[3] ≈ 0.0)
        # add_face_set!(mesh, "right_traction", x -> x[1] ≈ 3.0 && x[3] <= 0.1)

        add_dirichlet_bc!(ch, cv.dh, cv.facedata_col, "symmetry_bc", x -> SA[0.0], c_dofs=SA[2])
        add_dirichlet_bc!(ch, cv.dh, "left_clamp1", x -> SA[0.0,0.0,0.0], c_dofs=SA[1,2,3])
        add_dirichlet_bc!(ch, cv.dh, "left_clamp2", x -> SA[0.0,0.0,0.0], c_dofs=SA[1,2,3])
        add_neumann_bc!(ch, cv.dh, "right_traction", x -> SA[0.0, 0.0, -1.0])
        # add_neumann_bc!(ch, cv.dh, cv.facedata_col, "right_traction", x -> SA[0.0, 0.0, -100.0])^
    elseif b_case == :simple_lever
        add_node_set!(mesh, "bottom_clamp1", x -> x[1] ≈ 0.0 && x[2] ≈ 0.0 && x[3] ≈ 0.0)
        add_node_set!(mesh, "bottom_clamp2", x -> x[1] ≈ 1.0 && x[2] ≈ 0.0 && x[3] ≈ 0.0)
        add_face_set!(mesh, "right_traction", x -> x[1] ≈ 3.0 && x[3] >= 2.5 && x[2] >= 0.4)
        add_face_set!(mesh, "symmetry_bc", x -> x[2] ≈ 0.5)
        add_face_set!(mesh, "second_traction", x -> x[1] < 1.0 && x[3] ≈ 3.0)

        add_dirichlet_bc!(ch, cv.dh, "bottom_clamp1", x -> SA[0.0,0.0,0.0], c_dofs=SA[1,2,3])
        add_dirichlet_bc!(ch, cv.dh, "bottom_clamp2", x -> SA[0.0,0.0,0.0], c_dofs=SA[1,2,3])
        add_dirichlet_bc!(ch, cv.dh, cv.facedata_col, "symmetry_bc", x -> SA[0.0], c_dofs=SA[2])
        add_neumann_bc!(ch, cv.dh, cv.facedata_col, "right_traction", x -> SA[0.0, 0.0, -1.0])
        add_neumann_bc!(ch, cv.dh, cv.facedata_col, "second_traction", x -> SA[0.0, 0.0, -1.0])
    elseif b_case == :pressure_plate
        add_node_set!(mesh,"bottom_clamp", x -> x[1] ≈ 0.0 && x[2] ≈ 0.0 && x[3] ≈ 0.0)
        add_face_set!(mesh,"symmetry_bc1", x -> x[1] ≈ 3.0)
        add_face_set!(mesh,"symmetry_bc2", x -> x[2] ≈ 3.0)
        add_face_set!(mesh,"pressure", x -> x[1] > 2.8 && x[2] > 2.8 && x[3] ≈ 1.0)

        add_dirichlet_bc!(ch, cv.dh, "bottom_clamp", x -> SA[0.0], c_dofs=SA[3])
        add_dirichlet_bc!(ch, cv.dh, cv.facedata_col, "symmetry_bc1", x -> SA[0.0], c_dofs=SA[1])
        add_dirichlet_bc!(ch, cv.dh, cv.facedata_col, "symmetry_bc2", x -> SA[0.0], c_dofs=SA[2])
        add_neumann_bc!(ch, cv.dh, cv.facedata_col, "pressure", x -> SA[0.0, 0.0, -1.0])
    else
        error("Invalid boundary value problem: $b_case")
    end
    return ch
end


function clear_up_topo!(
    topo::Topology{3}
    )

    do_face_coarse_markers = ones(Bool,length(get_areas(topo)))
    do_edge_coarse_markers = ones(Bool,length(get_edges(topo)))
    for element in RootIterator{4}(topo)

        Ju3VEM.VEMGeo.iterate_volume_areas(
            topo,element.id) do face 

                face_ref_level = face.refinement_level
                do_coarse = do_face_coarse_markers[face.id] && (face_ref_level > element.refinement_level)

                do_face_coarse_markers[face.id] = do_coarse
                

                Ju3VEM.VEMGeo.iterate_element_edges(
                    topo,face.id) do _,edge_id,_ 

                        edge_ref_level = get_edges(topo)[edge_id].refinement_level
                        do_coarse = do_edge_coarse_markers[edge_id] && (edge_ref_level > element.refinement_level)
                        do_edge_coarse_markers[edge_id] = do_coarse
                        
                end   
        end
    end

    for face in RootIterator{3}(topo)
        do_coarse = do_face_coarse_markers[face.id]
        do_coarse || continue
        _coarsen!(face, topo)
    end
    for edge in RootIterator{2}(topo)
        do_coarse = do_edge_coarse_markers[edge.id]
        do_coarse || continue
        _coarsen!(edge, topo)
    end
end