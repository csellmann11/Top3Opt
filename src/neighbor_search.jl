using Ju3VEM
using Ju3VEM.VEMGeo: get_first,get_second,get_third,get_fourth, get_edge




function get_adjacent_elements(adjacency_matrix::SparseMatrixCSC, state_id::Integer)
    # Get the range of stored elements for this row using colptr
    start_idx = adjacency_matrix.colptr[state_id]
    end_idx   = adjacency_matrix.colptr[state_id + 1] - 1
     
    # Return the row indices (neighbors) directly from rowval
    # return view(adjacency_matrix.rowval, start_idx:end_idx)
    return (adjacency_matrix.rowval[i] for i in start_idx:end_idx)
end


function check_if_child_boundary_face(
    parent_id::Integer, 
    is_boundary_face::Vector{Bool},
    topo::Topology{3})

    parent_element = get_areas(topo)[parent_id]
    if is_root(parent_element,topo)
        return is_boundary_face[parent_id]
    end

    child1_id = parent_element.childs |> first
    return check_if_child_boundary_face(child1_id,is_boundary_face,topo)
end


function create_neigh_list(
    states::DesignVarInfo{D},
    cv::CellValues{D,U}
    ) where {D,U}


    e2s = states.el_id_to_state_id 
    topo = cv.mesh.topo
    fdcol = cv.facedata_col

    n_states = length(e2s)

    is_boundary_face = zeros(Bool,length(get_areas(topo)))
    # non boundary faces are just overwritten multiple times
    b_face_id_to_state_id = Dict{Int32,Int32}()

    ste_rows = Int32[]
    ste_cols = Int32[]

    # loop collects boundary faces and fills the state_id to edge_id sparse matrix
    for (el_id,state_id) in e2s

        Ju3VEM.VEMGeo.iterate_volume_areas(
            fdcol,topo,el_id) do face,fd,_ 

            face_id = face.id 
            vertex_ids = fd.face_node_ids |> get_first
            

            for (counter,v_id) in enumerate(vertex_ids) 
                counter_p1  = get_next_idx(vertex_ids,counter)
                v_id_p1     = vertex_ids[counter_p1]
                edge_id     = get_edge(v_id,v_id_p1,topo) |> get_id

                push!(ste_cols,edge_id)
                push!(ste_rows,state_id)
            end

            
            is_boundary_face[face_id] = !is_boundary_face[face_id]
        end
    end

    boundary_counter = 1
    boundary_state_map = Dict{Int32,Int32}()


    for (el_id,state_id) in e2s

        face_ids = get_volume_area_ids(topo,el_id)
        #INFO: since we mirror the element at the boundary, one neigh per parent face is enough
        for face_id in face_ids
            check_if_child_boundary_face(face_id,is_boundary_face,topo) || continue
            b_face_id_to_state_id[-face_id] = state_id

            Ju3VEM.VEMGeo.iterate_element_edges(
                topo,face_id) do _,edge_id,_ 
                    push!(ste_cols,edge_id)
                    push!(ste_rows,n_states + boundary_counter)
                end
            boundary_state_map[n_states + boundary_counter] = face_id
            boundary_counter += 1
        end
    end

 
    ste_mat = SparseArrays.spzeros!(Bool,ste_rows,ste_cols)
    ste_adj_mat = ste_mat * ste_mat'

    state_neights_col  = FixedSizeVector{FixedSizeVector{Int32}}(undef,n_states)

    for state_id in 1:n_states

        adj_states = get_adjacent_elements(ste_adj_mat,state_id)

        neighs = FixedSizeVector{Int32}(undef,length(adj_states)-1)
        counter = 1
        for adj_state in adj_states
            adj_state == state_id && continue
            if adj_state > n_states 
                adj_state = -boundary_state_map[adj_state]
            end
            neighs[counter] = adj_state
            counter += 1
        end       
        state_neights_col[state_id] = neighs
    end

    return state_neights_col,b_face_id_to_state_id
end









# function create_neigh_list(
#     states::DesignVarInfo{D},
#     cv::CellValues{D,U}
#     ) where {D,U}


#     e2s = states.el_id_to_state_id 
#     topo = cv.mesh.topo
#     fdcol = cv.facedata_col

#     n_states = length(e2s)

#     is_boundary_face = zeros(Bool,length(get_areas(topo)))
#     edge_to_state_ids= Dict{Int32,Vector{Int32}}()
#     # non boundary faces are just overwritten multiple times
    

#     # ste_rows = Int32[]
#     # ste_cols = Int32[]

#     # loop collects boundary faces and fills the state_id to edge_id sparse matrix
#     for (el_id,state_id) in e2s

#         Ju3VEM.VEMGeo.iterate_volume_areas(
#             fdcol,topo,el_id) do face,fd,_ 

#             face_id = face.id 
#             vertex_ids = fd.face_node_ids |> get_first
            

#             for (counter,v_id) in enumerate(vertex_ids) 
#                 counter_p1  = get_next_idx(vertex_ids,counter)
#                 v_id_p1     = vertex_ids[counter_p1]
#                 edge_id     = get_edge(v_id,v_id_p1,topo) |> get_id

#                 push!(get!(edge_to_state_ids,edge_id) do 
#                     Int32[]
#                 end,state_id)
#             end

            
#             is_boundary_face[face_id] = !is_boundary_face[face_id]
#         end
#     end

#     boundary_counter = 1
#     boundary_state_map = Dict{Int32,Int32}()
#     b_face_id_to_state_id = Dict{Int32,Int32}()


#     for (el_id,state_id) in e2s

#         face_ids = get_volume_area_ids(topo,el_id)
#         #INFO: since we mirror the element at the boundary, one neigh per parent face is enough
#         for face_id in face_ids
#             check_if_child_boundary_face(face_id,is_boundary_face,topo) || continue
#             b_face_id_to_state_id[-face_id] = state_id

#             Ju3VEM.VEMGeo.iterate_element_edges(
#                 topo,face_id) do _,edge_id,_ 
#                     push!(ste_cols,edge_id)
#                     push!(ste_rows,n_states + boundary_counter)
#                 end
#             boundary_state_map[n_states + boundary_counter] = face_id
#             boundary_counter += 1
#         end
#     end

 
#     ste_mat = SparseArrays.spzeros!(Bool,ste_rows,ste_cols)
#     ste_adj_mat = ste_mat * ste_mat'

#     state_neights_col  = FixedSizeVector{FixedSizeVector{Int32}}(undef,n_states)

#     for state_id in 1:n_states

#         adj_states = get_adjacent_elements(ste_adj_mat,state_id)

#         neighs = FixedSizeVector{Int32}(undef,length(adj_states)-1)
#         counter = 1
#         for adj_state in adj_states
#             adj_state == state_id && continue
#             if adj_state > n_states 
#                 adj_state = -boundary_state_map[adj_state]
#             end
#             neighs[counter] = adj_state
#             counter += 1
#         end       
#         state_neights_col[state_id] = neighs
#     end

#     return state_neights_col,b_face_id_to_state_id
# end

