using Ju3VEM
using Ju3VEM.VEMGeo: get_first,get_second,get_third,get_fourth, get_edge


function create_neighbor_list(
    cv::CellValues{D,U}
) where {D,U}

    # Unpack inputs 
    topo    = cv.mesh.topo
    fdcol   = cv.facedata_col

    # Intitialization
    face_to_vols = Dict{Int,Vector{Int}}()
    edge_to_vols = Dict{Int,Vector{Int}}() 
    elneighs_col = Dict{Int,Vector{Int}}()

    for element in RootIterator{4}(topo)
        el_id = element.id
        Ju3VEM.VEMGeo.iterate_volume_areas(
            fdcol,topo,el_id) do face,fd,_ 

            face_id = face.id 
            vertex_ids = fd.face_node_ids |> get_first

            for (counter,v_id) in enumerate(vertex_ids) 
                counter_p1  = get_next_idx(vertex_ids,counter)
                v_id_p1     = vertex_ids[counter_p1]
                edge_id     = get_edge(v_id,v_id_p1,topo) |> get_id

                push!(
                    get!(edge_to_vols,edge_id,Int[]),
                    el_id
                )
            end
            push!(
                get!(face_to_vols,face_id,Int[]),
                el_id
            )
        end
    end


    # Info: create adj informations - Loop Faces
    for (face_id,adj_elements) in face_to_vols

        # handling of a normal face
        if length(adj_elements) == 2
            adj_el1 = adj_elements[1]; adj_el2 = adj_elements[2]
            push!(
                get!(elneighs_col,adj_el1,Int[]),
                adj_el2
            )
            push!(
                get!(elneighs_col,adj_el2,Int[]),
                adj_el1
            )
            continue 
        end

        # if element does not have 2 neighs, it has to be a boundary face
        adj_el = only(adj_elements)

        push!(
            get!(elneighs_col,adj_el,Int[]),
            -face_id
        )

        fd         = fdcol[face_id]
        vertex_ids = fd.face_node_ids |> get_first
        for (counter,v_id) in enumerate(vertex_ids) 
            counter_p1  = get_next_idx(vertex_ids,counter)
            v_id_p1     = vertex_ids[counter_p1]
            edge_id     = get_edge(v_id,v_id_p1,topo) |> get_id
            push!(edge_to_vols[edge_id],-face_id)
        end
    end

    # Info: Loop Edges
    # collects edge neighbors, also boundary edge neighs are handled here
    for (_,adj_elements) in edge_to_vols
        
        for adj_element in adj_elements 
            adj_element < 0 && continue 
            neighs = elneighs_col[adj_element]
            for adj_element_inner in adj_elements
                adj_element == adj_element_inner && continue 
                adj_element_inner ∈ neighs       && continue
                push!(neighs,adj_element_inner)
            end
        end
    end

    return elneighs_col, face_to_vols, edge_to_vols
end




function get_adjacent_elements(adjacency_matrix::SparseMatrixCSC, state_id::Int)
    # Get the range of stored elements for this row using colptr
    start_idx = adjacency_matrix.colptr[state_id]
    end_idx   = adjacency_matrix.colptr[state_id + 1] - 1
     
    # Return the row indices (neighbors) directly from rowval
    # return view(adjacency_matrix.rowval, start_idx:end_idx)
    return (adjacency_matrix.rowval[i] for i in start_idx:end_idx)
end


function create_neigh_list(
    states::TopStates{D},
    cv::CellValues{D,U}
) where {D,U}


    e2s = states.el_id_to_state_id 
    topo = cv.mesh.topo
    fdcol = cv.facedata_col

    n_states = length(e2s)

    face_to_state_ids = Dict{Int,SVector{2,Bool}}()

    ste_rows = Int32[]
    ste_cols = Int32[]

    for (el_id,state_id) in e2s

        Ju3VEM.VEMGeo.iterate_volume_areas(
            fdcol,topo,el_id) do face,fd,_ 

            face_id = face.id 
            vertex_ids = fd.face_node_ids |> get_first

            for (counter,v_id) in enumerate(vertex_ids) 
                counter_p1  = get_next_idx(vertex_ids,counter)
                v_id_p1     = vertex_ids[counter_p1]
                edge_id     = get_edge(v_id,v_id_p1,topo) |> get_id

                push!(ste_rows,state_id)
                push!(ste_cols,edge_id)
            end
            
            face_states = get(face_to_state_ids,face_id,SA[false,false])
            vec = if !face_states[1] 
                SA[true,false]
            else
                SA[true,true]
            end
            face_to_state_ids[face_id] = vec
        end
    end

    boundary_counter = 1
    boundary_state_map = Dict{Int32,Int32}()
    

    for (face_id,adj_states) in face_to_state_ids
        adj_states[2] && continue 
        # we have a boundary face

        fd = fdcol[face_id]
        vertex_ids = fd.face_node_ids |> get_first
        for (counter,v_id) in enumerate(vertex_ids) 
            counter_p1  = get_next_idx(vertex_ids,counter)
            v_id_p1     = vertex_ids[counter_p1]
            edge_id     = get_edge(v_id,v_id_p1,topo) |> get_id
            push!(ste_rows,n_states + boundary_counter)
            push!(ste_cols,edge_id)
            boundary_state_map[n_states + boundary_counter] = face_id
            boundary_counter += 1
        end

    end

    

    ste_mat = sparse(ste_rows,ste_cols,true)
    ste_adj_mat = ste_mat * ste_mat'

    state_neights_col  = FixedSizeVector{Vector{Int32}}(undef,n_states)

    for (el_id,state_id) in e2s

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

end





# @btime unique!(data) setup = (data = rand(1:10,100))
# @btime 1 in data setup = (data = rand(1:10,100))