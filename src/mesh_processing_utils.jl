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
    return Mesh(mesh.topo, ET)
end



# function ref_mesh_at_sets(mesh::Mesh{3}, set_member_funs::T, max_ref_level::Int) where T<:Tuple 
#     topo = mesh.topo

#     for rl in 1:max_ref_level
#         for element in RootIterator{3}(mesh.topo)
#             node_ids = get_volume_node_ids(mesh.topo, element.id)
#             # only refine if at least one node is a member of any of the sets
#             !any(set_member_funs[i].(topo.nodes[node_ids]) for i in eachindex(set_member_funs)) && continue
#             element.refinement_level >= rl && continue
#             _refine!(element, mesh.topo)
#         end
#     end
# end


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


function create_constraint_handler(cv::CellValues{3}, b_case::Symbol=:MBB_sym)
    mesh = cv.mesh
    ch = ConstraintHandler{U}(mesh)

    if b_case == :MBB_sym

        add_face_set!(mesh, "symmetry_bc", x -> x[1] ≈ 0.0)
        add_face_set!(mesh, "symmetry_bc_2", x -> x[2] ≈ 0.5)
        add_node_set!(mesh, "roller_bearing", x -> x[1] ≈ 3.0 && x[3] ≈ 0.0)
        add_face_set!(mesh, "middle_traction", x -> (0 ≤ x[1] ≤ 0.251) && x[3] ≈ 1.0)

        add_dirichlet_bc!(ch, cv.dh, cv.facedata_col, "symmetry_bc", x -> SA[0.0], c_dofs=SA[1])
        add_dirichlet_bc!(ch, cv.dh, cv.facedata_col, "symmetry_bc_2", x -> SA[0.0], c_dofs=SA[2])
        add_dirichlet_bc!(ch, cv.dh, "roller_bearing", x -> SA[0.0, 0.0], c_dofs=SA[2, 3])
        add_neumann_bc!(ch, cv.dh, cv.facedata_col, "middle_traction", x -> SA[0.0, 0.0, -1.0])
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
        # add_neumann_bc!(ch, cv.dh, cv.facedata_col, "right_traction", x -> SA[0.0, 0.0, -100.0])
    else
        error("Invalid boundary value problem: $b_case")
    end
    ch
end


using Ju3VEM.VEMGeo: CustStack, pop_last!
function clear_up_mesh(topo::Topology{3},
    face_to_vols::Dict{Int,Vector{Int}},
    edge_to_vols::Dict{Int,Vector{Int}})

    areas = get_areas(topo)
    edges = get_edges(topo)

    for face in RootIterator{3}(topo)

        parent_id = face.parent_id
        parent_id == 0 && continue

        parent_face = areas[parent_id]
        face_neighs = Int[]
        all_equal = true

        @no_escape begin
            aq = CustStack(stack=@alloc(Int, 100))

    
            push!(aq, parent_face.id |> abs)
            while !isempty(aq)
                root_area_id = pop_last!(aq)
                root_area = areas[root_area_id]

                if is_root(root_area)
                    if isempty(face_neighs)
                        new_neighs = face_to_vols[root_area.id] 
                        resize!(face_neighs, length(new_neighs))
                        copyto!(face_neighs, new_neighs)
                    else
                        if !all(in(face_neighs), face_to_vols[root_area.id])
                            all_equal = false
                            break
                        end
                    end
                else
                    for child_id in root_area.childs
                        push!(aq, child_id)
                    end
                end

            end
        end

        all_equal && _coarsen!(face, topo)
    end


    for edge in RootIterator{2}(topo)

        parent_id = edge.parent_id
        parent_id == 0 && continue

        parent_edge = edges[parent_id]
        edge_neighs = Int[]
        all_equal = true
        @no_escape begin
            aq = CustStack(stack=@alloc(Int, 100))
            push!(aq, parent_edge.id |> abs)
            while !isempty(aq)
                root_edge_id = pop_last!(aq)
                root_edge = edges[root_edge_id]
                if is_root(root_edge)
                    root_edge.id <= 0 && @show root_edge 
                    if isempty(edge_neighs)
                        
                        new_neighs = edge_to_vols[root_edge.id]
                        resize!(edge_neighs, length(new_neighs))
                        copyto!(edge_neighs, new_neighs)
                    else
                        if !all(in(edge_neighs), edge_to_vols[root_edge.id])
                            all_equal = false
                            break
                        end
                    end
                else
                    for child_id in root_edge.childs
                        push!(aq, child_id)
                    end
                end
            end
        end
        all_equal && _coarsen!(edge, topo)
    end
end