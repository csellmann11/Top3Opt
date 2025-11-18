using Ju3VEM
using Ju3VEM.VEMGeo: get_edge_node_ids
using LinearAlgebra
function face_area(node_ids::AbstractVector{Int}, topo::Topology{3})
    nodes = topo.nodes
    n = length(node_ids)
    area_vec = sum(i -> cross(nodes[node_ids[i]], nodes[node_ids[mod1(i + 1, n)]]), 1:n)
    return 0.5 * norm(area_vec)
end


"""
    function improve_refined_element(
        element_id::Int, 
        topo::Topology{3},
    )

Takes in as input a parent element, which has been refined exactly once.
Function finds short edges and merges childs, which are adjacent to the short edges.
"""
function improve_refined_element!(
    topo::Topology{3},
    element_id::Int  
    ) 


    parent_element = get_volumes(topo)[element_id] 
    child_ids = parent_element.childs

    bad_edges = Int32[]
    face_ids = get_volume_area_ids(topo,element_id)

    el_volume = volume_of_topo_volume(topo,element_id)

    edges = get_edges(topo)

    for face_id in face_ids 
        parent_face = get_areas(topo)[face_id]
        face_child_ids = parent_face.childs

        face_node_ids = get_area_node_ids(topo, face_id) |> get_first
        face_area     = face_area(face_node_ids, topo)

        mark_all_face_edges = false
        if face_area^2 < el_volume * 5e-03
            mark_all_face_edges = true
        end

        face_edge_ids = get_area_edge_ids(topo, face_id)
        for edge_id in face_edge_ids
            n1_id,n2_id = get_edge_node_ids(topo, edge_id) 
            n1 = topo.nodes[n1_id]
            n2 = topo.nodes[n2_id]

            edge_len = norm(n2-n1)

            (edge_len^2 < 0.2 * face_area || mark_all_face_edges) || continue

            Ju3VEM.VEMGeo.apply_f_on_unordered_roots(
                edges[edge_id],edges) do root_edge 
                    push!(bad_edges,root_edge.id)
                end
        end  
    end 

    face_to_vols = Dict{Int,SVector{2,Int}}()
    
    for child_id in child_ids 
        Ju3VEM.VEMGeo.iterate_volume_areas(
            topo,child_id) do face,face_parent_id 

                parent_face   = get_areas(topo)[face_parent_id]
                is_outer_face = parent_face.parent_id != 0
                
                face_vols = get(face_to_vols,face.id,SA[0,0])
                new_face_vols = if face_vols[1] == 0
                    SA[face.id,0]
                else
                    SA[face_vols[1],face.id]
                end

                if is_outer_face 
                    
                end
            end

    end


    

end