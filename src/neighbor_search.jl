using Ju3VEM
using Ju3VEM.VEMGeo: get_first,get_second,get_third,get_fourth, get_edge




function get_adjacent_elements(adjacency_matrix::SparseMatrixCSC, state_id::Int)
    # Get the range of stored elements for this row using colptr
    start_idx = adjacency_matrix.colptr[state_id]
    end_idx   = adjacency_matrix.colptr[state_id + 1] - 1
     
    # Return the row indices (neighbors) directly from rowval
    # return view(adjacency_matrix.rowval, start_idx:end_idx)
    return (adjacency_matrix.rowval[i] for i in start_idx:end_idx)
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
            b_face_id_to_state_id[-face_id] = state_id

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

    # _ste_mat = spzeros(Bool,ste_rows,ste_cols)
    # # add second row neighs 
    # for node in cv.mesh.nodes
    #     is_active(node) || continue
        
    #     adj_states = get_adjacent_elements(_ste_mat,node.id)
    #     for adj_state in adj_states
    #         Ju3VEM.VEMGeo.iterate_volume_areas(
    #             fdcol,topo,Int(adj_state)) do _,fd,_ 

    #             vertex_ids = fd.face_node_ids |> get_first
    #             for v_id in vertex_ids
    #                 for adj_state_inner in adj_states
    #                     adj_state_inner == adj_state && continue
    #                     push!(ste_cols,v_id)
    #                     push!(ste_rows,adj_state_inner)
    #                 end
    #             end
    #         end
    #     end
    # end

    boundary_counter = 1
    boundary_state_map = Dict{Int32,Int32}()
    
    # loops all boundary faces, adds to every edge of the 
    # boundary faces an imaginary state_id
    for face in RootIterator{3}(topo)
        face_id = face.id
        is_boundary_face[face_id] || continue 
        # we have a boundary face

        fd = fdcol[face_id]
        vertex_ids = fd.face_node_ids |> get_first
        for (counter,v_id) in enumerate(vertex_ids) 
            counter_p1  = get_next_idx(vertex_ids,counter)
            v_id_p1     = vertex_ids[counter_p1]
            edge_id     = get_edge(v_id,v_id_p1,topo) |> get_id
            push!(ste_cols,edge_id)
            push!(ste_rows,n_states + boundary_counter)
            boundary_state_map[n_states + boundary_counter] = face_id
        end
        boundary_counter += 1
    end

    
    ste_mat = spzeros(Bool,ste_rows,ste_cols)
    # ste_mat = sparse(ste_rows,ste_cols,true)
    ste_adj_mat = ste_mat * ste_mat'

    state_neights_col  = FixedSizeVector{Vector{Int32}}(undef,n_states)

    for state_id in 1:n_states

        adj_states = get_adjacent_elements(ste_adj_mat,state_id)
        len = length(adj_states) #-1 

        n_count = 1
        neighs = Vector{Int32}(undef,len)
        for adj_state in adj_states
            # adj_state == state_id && continue
            if adj_state > n_states 
                adj_state = -boundary_state_map[adj_state]
                # b_face_id_to_state_id[adj_state] = state_id
            end
            neighs[n_count] = adj_state
            n_count += 1
        end
        state_neights_col[state_id] = neighs
    end

    return state_neights_col,b_face_id_to_state_id
end


function create_neigh2_list(
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
            b_face_id_to_state_id[-face_id] = state_id

            push!(ste_cols,face_id)
            push!(ste_rows,state_id)
            is_boundary_face[face_id] = !is_boundary_face[face_id]
        end
    end

    boundary_counter = 1
    boundary_state_map = Dict{Int32,Int32}()
    
    # loops all boundary faces, adds to every edge of the 
    # boundary faces an imaginary state_id
    for face in RootIterator{3}(topo)
        face_id = face.id
        is_boundary_face[face_id] || continue 
        # we have a boundary face
        push!(ste_cols,face_id)
        push!(ste_rows,n_states + boundary_counter)
        boundary_state_map[n_states + boundary_counter] = face_id
        boundary_counter += 1
    end

    _ste_mat = spzeros(Bool,ste_rows,ste_cols)
    # add second row neighs 
    for face in RootIterator{3}(topo)

        adj_states = get_adjacent_elements(_ste_mat,face.id)
        for adj_state in adj_states
            adj_state > n_states && continue
            adj_element_id = get_el_id(states,adj_state)
            Ju3VEM.VEMGeo.iterate_volume_areas(
                fdcol,topo,Int(adj_element_id)) do face_inner,fd,_ 

                for adj_state_inner in adj_states
                    adj_state_inner == adj_state && continue
                    push!(ste_cols,face_inner.id)
                    push!(ste_rows,adj_state_inner)
                end
            end
        end
    end


    
    ste_mat = SparseArrays.spzeros!(Bool,ste_rows,ste_cols)
    ste_adj_mat = ste_mat * ste_mat'

    state_neights_col  = FixedSizeVector{Vector{Int32}}(undef,n_states)

    for state_id in 1:n_states

        adj_states = get_adjacent_elements(ste_adj_mat,state_id)
        len = length(adj_states) -1 

        n_count = 1
        neighs = Vector{Int32}(undef,len)
        for adj_state in adj_states
            adj_state == state_id && continue
            if adj_state > n_states 
                adj_state = -boundary_state_map[adj_state]
            end
            neighs[n_count] = adj_state
            n_count += 1
        end
        state_neights_col[state_id] = neighs
    end

    return state_neights_col,b_face_id_to_state_id
end