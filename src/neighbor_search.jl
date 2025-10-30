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



# @btime unique!(data) setup = (data = rand(1:10,100))
# @btime 1 in data setup = (data = rand(1:10,100))