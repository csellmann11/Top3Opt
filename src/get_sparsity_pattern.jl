using SparseArrays
using Ju3VEM 


function create_dense_node_id_map(mesh::Mesh{D}) where {D}
    node_id_map = FixedSizeVector{Int32}(undef,length(mesh.nodes))
    dense_node_id = 1
    for node in mesh.nodes
        is_active(node) || continue
        node_id_map[node.id] = dense_node_id
        dense_node_id += 1
    end
    return node_id_map
end

function get_sparsity_pattern(
    cv::CellValues{D,U}
    ) where {D,U}


    node_id_map = create_dense_node_id_map(cv.mesh)

 
    rowvals = Int32[]
    colptr  = Int32[]

    el_node_ids = Set{Int32}()

    colptr_counter = Ref(1)
    for element in RootIterator{4}(cv.mesh.topo)
        push!(colptr,colptr_counter[])
        
        empty!(el_node_ids)
        Ju3VEM.VEMGeo.iterate_volume_areas(
            cv.facedata_col,cv.mesh.topo,element.id) do _,fd,_ 
                node_ids = fd.face_node_ids 
                for node_id in node_ids
                    node_id in el_node_ids && continue
                    push!(el_node_ids,node_id)
                    dense_node_id = node_id_map[node_id]
                    push!(rowvals,dense_node_id)
                    colptr_counter[] += 1
  
                end
            end

        moment_ids = cv.mesh.int_coords_connect[4][element.id]
        for moment_id in moment_ids
            dense_moment_id = node_id_map[moment_id]

            push!(rowvals,dense_moment_id)
            colptr_counter[] += 1
        end
    end
    push!(colptr,colptr_counter[])

    n_active_nodes   = length(keys(cv.dh.dof_mapping))

    node2el_id = SparseMatrixCSC(n_active_nodes,length(colptr)-1,colptr,rowvals,ones(Bool,length(rowvals)))

    small_sparse_pattern = node2el_id * node2el_id' 

    
    b = ones(Bool,U,U) |> sparse

    sp = kron(small_sparse_pattern,b)
    k_global = SparseMatrixCSC(
        sp.m,sp.n,sp.colptr,sp.rowval,zeros(Float64,length(sp.nzval))
    )
    return k_global
end



# function get_sparsity_pattern(
#     cv::CellValues{D,U}
#     ) where {D,U}


#     # node_id_map = create_dense_node_id_map(cv.mesh)

 
#     rowvals = Int32[]
#     colptr  = Int32[]

#     el_node_ids = Set{Int32}()

#     colptr_counter = Ref(1)
#     for element in RootIterator{4}(cv.mesh.topo)
#         push!(colptr,colptr_counter[])
        
#         empty!(el_node_ids)
#         Ju3VEM.VEMGeo.iterate_volume_areas(
#             cv.facedata_col,cv.mesh.topo,element.id) do _,fd,_ 
#                 node_ids = fd.face_node_ids 
#                 for node_id in node_ids
#                     node_id in el_node_ids && continue
#                     push!(el_node_ids,node_id)

#                     dofs = get_dofs(cv.dh,node_id)
#                     for u in 1:U
#                         # dense_node_id = node_id_map[node_id]
#                         dof = dofs[u]
#                         push!(rowvals,dof)
#                         colptr_counter[] += 1
#                     end
#                 end
#             end

#         moment_ids = cv.mesh.int_coords_connect[4][element.id]
#         for moment_id in moment_ids
#             # dense_moment_id = node_id_map[moment_id]

#             dofs = get_dofs(cv.dh,moment_id)
#             for u in 1:U
#                 dof = dofs[u]
#                 push!(rowvals,dof)
#                 colptr_counter[] += 1
#             end
#         end
#     end
#     push!(colptr,colptr_counter[])

#     n_active_nodes = length(keys(cv.dh.dof_mapping))

#     dof2_el_id = SparseMatrixCSC(n_active_nodes*U,length(colptr)-1,colptr,rowvals,ones(Bool,length(rowvals)))

#     sp         = dof2_el_id * dof2_el_id'
#     # small_sparse_pattern = node2el_id * node2el_id' 

    
#     # b = ones(Bool,U,U) |> sparse

#     # sp = kron(small_sparse_pattern,b)
#     k_global = SparseMatrixCSC(
#         sp.m,sp.n,sp.colptr,sp.rowval,zeros(Float64,length(sp.nzval))
#     )
#     return k_global
# end
