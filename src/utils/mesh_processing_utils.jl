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




function refine_sets(mesh::Mesh{3},
    sets_to_refine::T,
    MAX_REF_LEVEL::Int) where T<:Tuple

    topo = mesh.topo


    ref_marker = zeros(Bool, length(get_volumes(topo)))

    n_els = length(get_volumes(topo))

    for element in RootIterator{4}(topo)

        Ju3VEM.VEMGeo.iterate_volume_areas(
            topo, element.id) do face, _

            Ju3VEM.VEMGeo.iterate_element_edges(
                topo, face.id) do n1_id, _, _
                node = topo.nodes[n1_id]
                if any(set_func(node) for set_func in sets_to_refine)
                    ref_marker[element.id] = true
                end
                nothing
            end
        end
    end

    for element in RootIterator{4}(topo)

        if element.refinement_level >= MAX_REF_LEVEL
            ref_marker[element.id] = false
            continue
        end
        ref_marker[element.id] && _refine!(element, topo)
    end


    for _ in 3:MAX_REF_LEVEL
        ref_marker = resize!(ref_marker, length(get_volumes(topo)))

        for element in RootIterator{4}(topo)
            element.id <= n_els && continue
            _refine!(element, topo)
            ref_marker[element.id] = true
        end

    end

    mesh = Mesh(topo, StandardEl{1}())

    return mesh, ref_marker
end


function mark_elements_which_are_part_of_sets(cv::CellValues{3}, ch::ConstraintHandler{3})
    mesh = cv.mesh
    topo = mesh.topo
    forbid_coarsening = zeros(Bool, length(get_volumes(topo)))

    for element in RootIterator{4}(topo)

        reinit!(element.id, cv)
        dofs = Ju3VEM.VEMUtils.get_cell_dofs(cv)
        forbidden = false

        for dof in dofs
            continue #! remove
            if haskey(ch.n_bcs, dof)
                forbidden = true
                break
            end
            if haskey(ch.d_bcs, dof)
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
        return (x -> (0 ≤ x[1] ≤ 0.251) && x[3] ≈ 1.0,)
        # return (x -> x[1] ≈ 0.0 && x[3] ≈ 1.0,)
    elseif b_case == :Cantilever_sym
        return (x -> x[1] ≈ 2.0 && (0.4 ≤ x[3] ≤ 0.6) && (0.4 ≤ x[2] ≤ 0.6),)
    elseif b_case == :Bending_Beam_sym
        return (x -> x[1] ≈ 3.0 && x[3] ≈ 0.0,)
    elseif b_case == :simple_lever
        return (x -> x[1] ≈ 3.0 && x[3] >= 2.5 && x[2] >= 0.4,)
    elseif b_case == :pressure_plate
        return (x -> x[1] > 2.8 && x[2] > 2.8 && x[3] ≈ 1.0,)
    elseif b_case == :L_cantilever
        return (x ->(1.749 ≤ x[1] ≤ 2.0) && x[2] ≈ 1.0 && x[3] >= 0.374,)
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
        add_face_set!(mesh, "left_clamp", x -> x[1] ≈ 0.0)
        add_face_set!(mesh, "right_traction", x -> x[1] ≈ 2.0 && (0.4 ≤ x[3] ≤ 0.6) && (0.4 ≤ x[2] ≤ 0.6))

        add_dirichlet_bc!(ch, cv.dh, cv.facedata_col, "symmetry_bc", x -> SA[0.0], c_dofs=SA[2])
        add_dirichlet_bc!(ch, cv.dh, cv.facedata_col, "left_clamp", x -> SA[0.0, 0.0, 0.0], c_dofs=SA[1, 2, 3])
        add_neumann_bc!(ch, cv.dh, cv.facedata_col, "right_traction", x -> SA[0.0, 0.0, -1.0])
    elseif b_case == :Bending_Beam_sym
        add_face_set!(mesh, "symmetry_bc", x -> x[2] ≈ 0.5)
        add_node_set!(mesh, "left_clamp1", x -> x[1] ≈ 0.0 && x[2] ≈ 0.0 && x[3] ≈ 0.0)
        add_node_set!(mesh, "left_clamp2", x -> x[1] ≈ 0.0 && x[2] ≈ 0.0 && x[3] ≈ 1.0)
        add_edge_set!(mesh, "right_traction", x -> x[1] ≈ 3.0 && x[3] ≈ 0.0)
        # add_face_set!(mesh, "right_traction", x -> x[1] ≈ 3.0 && x[3] <= 0.1)

        add_dirichlet_bc!(ch, cv.dh, cv.facedata_col, "symmetry_bc", x -> SA[0.0], c_dofs=SA[2])
        add_dirichlet_bc!(ch, cv.dh, "left_clamp1", x -> SA[0.0, 0.0, 0.0], c_dofs=SA[1, 2, 3])
        add_dirichlet_bc!(ch, cv.dh, "left_clamp2", x -> SA[0.0, 0.0, 0.0], c_dofs=SA[1, 2, 3])
        add_neumann_bc!(ch, cv.dh, "right_traction", x -> SA[0.0, 0.0, -1.0])
        # add_neumann_bc!(ch, cv.dh, cv.facedata_col, "right_traction", x -> SA[0.0, 0.0, -100.0])^
    elseif b_case == :simple_lever
        add_node_set!(mesh, "bottom_clamp1", x -> x[1] ≈ 0.0 && x[2] ≈ 0.0 && x[3] ≈ 0.0)
        add_node_set!(mesh, "bottom_clamp2", x -> x[1] ≈ 1.0 && x[2] ≈ 0.0 && x[3] ≈ 0.0)
        add_face_set!(mesh, "right_traction", x -> x[1] ≈ 3.0 && x[3] >= 2.5 && x[2] >= 0.4)
        add_face_set!(mesh, "symmetry_bc", x -> x[2] ≈ 0.5)
        add_face_set!(mesh, "second_traction", x -> x[1] < 1.0 && x[3] ≈ 3.0)

        add_dirichlet_bc!(ch, cv.dh, "bottom_clamp1", x -> SA[0.0, 0.0, 0.0], c_dofs=SA[1, 2, 3])
        add_dirichlet_bc!(ch, cv.dh, "bottom_clamp2", x -> SA[0.0, 0.0, 0.0], c_dofs=SA[1, 2, 3])
        add_dirichlet_bc!(ch, cv.dh, cv.facedata_col, "symmetry_bc", x -> SA[0.0], c_dofs=SA[2])
        add_neumann_bc!(ch, cv.dh, cv.facedata_col, "right_traction", x -> SA[0.0, 0.0, -1.0])
        add_neumann_bc!(ch, cv.dh, cv.facedata_col, "second_traction", x -> SA[0.0, 0.0, -1.0])
    elseif b_case == :pressure_plate
        add_node_set!(mesh, "bottom_clamp", x -> x[1] ≈ 0.0 && x[2] ≈ 0.0 && x[3] ≈ 0.0)
        add_face_set!(mesh, "symmetry_bc1", x -> x[1] ≈ 3.0)
        add_face_set!(mesh, "symmetry_bc2", x -> x[2] ≈ 3.0)
        add_face_set!(mesh, "pressure", x -> x[1] > 2.8 && x[2] > 2.8 && x[3] ≈ 1.0)

        add_dirichlet_bc!(ch, cv.dh, "bottom_clamp", x -> SA[0.0], c_dofs=SA[3])
        add_dirichlet_bc!(ch, cv.dh, cv.facedata_col, "symmetry_bc1", x -> SA[0.0], c_dofs=SA[1])
        add_dirichlet_bc!(ch, cv.dh, cv.facedata_col, "symmetry_bc2", x -> SA[0.0], c_dofs=SA[2])
        add_neumann_bc!(ch, cv.dh, cv.facedata_col, "pressure", x -> SA[0.0, 0.0, -1.0])
    elseif b_case == :L_cantilever
        add_face_set!(mesh, "symmetry_bc", x -> x[3] ≈ 0.5)
        add_node_set!(mesh, "top_clamp1", x -> x[2] ≈ 2.0 && x[3] ≈ 0.0 && x[1] ≈ 0.0)
        add_node_set!(mesh, "top_clamp2", x -> x[2] ≈ 2.0 && x[3] ≈ 0.0 && x[1] ≈ 1.0)
        add_face_set!(mesh, "traction", x -> (1.749 ≤ x[1] ≤ 2.0) && x[2] ≈ 1.0 && x[3] >= 0.374)

        add_dirichlet_bc!(ch, cv.dh, cv.facedata_col, "symmetry_bc", x -> SA[0.0], c_dofs=SA[3])
        add_dirichlet_bc!(ch, cv.dh, "top_clamp1", x -> SA[0.0, 0.0], c_dofs=SA[1, 2])
        add_dirichlet_bc!(ch, cv.dh, "top_clamp2", x -> SA[0.0, 0.0], c_dofs=SA[1, 2])
        add_neumann_bc!(ch, cv.dh, cv.facedata_col, "traction", x -> SA[0.0, -1.0, 0.0])
    else
        error("Invalid boundary value problem: $b_case")
    end
    return ch
end


function clear_up_topo!(
    topo::Topology{3})

    do_face_coarse_markers = ones(Bool, length(get_areas(topo)))
    do_edge_coarse_markers = ones(Bool, length(get_edges(topo)))
    for element in RootIterator{4}(topo)

        Ju3VEM.VEMGeo.iterate_volume_areas(
            topo, element.id) do face, _

            face_ref_level = face.refinement_level
            do_coarse = do_face_coarse_markers[face.id] && (face_ref_level > element.refinement_level)

            do_face_coarse_markers[face.id] = do_coarse


            Ju3VEM.VEMGeo.iterate_element_edges(
                topo, face.id) do _, edge_id, _

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






function face_area(node_ids::AbstractVector{Int}, topo::Topology{2})
    nodes = topo.nodes
    n = length(node_ids)
    area_sum = sum(i -> nodes[node_ids[i]][1] * nodes[node_ids[mod1(i + 1, n)]][2] -
                        nodes[node_ids[i]][2] * nodes[node_ids[mod1(i + 1, n)]][1], 1:n)
    return 0.5 * abs(area_sum)
end


using Ju3VEM.VEMGeo: get_area_edge_ids

function create_dense_node_id_map(topo::Topology{D}) where {D}
    node_id_map = FixedSizeVector{Int32}(undef,length(topo.nodes))
    dense_node_id = 1
    for node in topo.nodes
        is_active(node) || continue
        node_id_map[node.id] = dense_node_id
        dense_node_id += 1
    end
    return node_id_map
end


function remove_short_edges(topo::Topology{2})

    edge_to_faces   = Dict{Int,Vector{Int}}()
    node_to_faces   = Dict{Int,Vector{Int}}() 
    are_short_edges = ones(Bool,length(get_edges(topo)))

    for element in RootIterator{3}(topo)

        vertex_ids = get_area_node_ids(topo,element.id)
        area       = face_area(vertex_ids,topo)

        last_edge_marked = Ref(false)
        last_edge_seen   = Ref(0)
        first_edge_seen  = Ref(0)

        Ju3VEM.VEMGeo.iterate_element_edges(
            topo, element.id
        ) do _n1_id, edge_id, _

            push!(get!(edge_to_faces,edge_id,Int[]),element.id)
            push!(get!(node_to_faces,_n1_id,Int[]),element.id)

            if first_edge_seen[] == 0
                first_edge_seen[] = edge_id
            end
            last_edge_seen[] = edge_id

            n1_id,n2_id = get_edge_node_ids(topo,edge_id)

            n1 = topo.nodes[n1_id]; n2 = topo.nodes[n2_id]
            edge_len = norm(n1-n2)

            is_short_edge = edge_len^2 < 1/8 * area
            if last_edge_marked[] || length(vertex_ids) <= 3
                is_short_edge = false 
            end
            last_edge_marked[] = is_short_edge
            are_short_edges[edge_id] = (are_short_edges[edge_id] && is_short_edge)
        end


        if are_short_edges[first_edge_seen[]] && are_short_edges[last_edge_seen[]]
            are_short_edges[first_edge_seen[]] = false
        end
    end

    n_removed_edges = sum(are_short_edges)
    println("will remove $n_removed_edges")

    is_boundary_node = zeros(Bool,length(topo.nodes))
    for (edge_id,adj_faces) in edge_to_faces
        length(adj_faces) == 2 && continue 
        n1_id,n2_id = get_edge_node_ids(topo,edge_id)
        is_boundary_node[n1_id] = true 
        is_boundary_node[n2_id] = true
    end

    counter = 0
    removed_nodes = Int64[]
    for (edge_id,adj_faces) in edge_to_faces
        are_short_edges[edge_id] || continue

        counter += 1
        
        n1_id,n2_id = get_edge_node_ids(topo,edge_id)
        
        if is_boundary_node[n1_id] 
            n1_id,n2_id = n2_id,n1_id 
        end

        push!(removed_nodes,n1_id)


        # remove n1_id from face nodes 
        # remove edge from face edges 
        for adj_face_id in adj_faces
            node_ids = get_area_node_ids(topo,adj_face_id) |> collect
            idx_node      = findfirst(id -> id == n1_id, node_ids)

            if idx_node === nothing 
                @show n1_id,n2_id,adj_face_id
                @show topo.nodes[n1_id]
                continue 
            end
            popat!(node_ids,idx_node)
            topo.connectivity[1,3][adj_face_id] = FixedSizeVector(node_ids)

        end
        

        # remove n1 from elements whihc dont share the edge, just the node 
        # add n2 to those elements #
        for adj_face_id in unique!(node_to_faces[n1_id])
            adj_face_id in adj_faces && continue # face shares an edge 
            node_ids = get_area_node_ids(topo,adj_face_id)

            idx = findfirst(id -> id == n1_id, node_ids)
            idx === nothing && continue 
            node_ids[idx] = n2_id
        end

        Ju3VEM.VEMGeo.deactivate!(topo.nodes[n1_id])
    end

    node_id_map = create_dense_node_id_map(topo)
    
    topo_new = Topology{2}()

    # add_node!.(topo.nodes,Ref(topo_new))
    # append!(topo_new.nodes,topo.nodes)
    for node in topo.nodes
        is_active(node) || continue
        add_node!(node.coords,topo_new)
    end

    for element in RootIterator{3}(topo)
        node_ids = get_area_node_ids(topo,element.id)
        new_node_ids = node_id_map[node_ids] .|> Int64
        add_area!(new_node_ids,topo_new)
    end

    topo_new

end