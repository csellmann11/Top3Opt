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

    rows = Int32[]
    cols = Int32[] 

    for element in RootIterator{4}(cv.mesh.topo)

        Ju3VEM.VEMGeo.iterate_volume_areas(
            cv.facedata_col,cv.mesh.topo,element.id) do _,fd,_ 
                node_ids = fd.face_node_ids 
                for node_id in node_ids
                    dense_node_id = node_id_map[node_id]
                    push!(rows,dense_node_id)
                    push!(cols,element.id)
                end
            end
    end

    # node2el_id = sparse(rows,cols,true) 
    node2el_id = SparseArrays.spzeros!(Bool,rows,cols)
    small_sparse_pattern = node2el_id * node2el_id' 
    
    b = ones(Bool,U,U) |> sparse

    sp = kron(small_sparse_pattern,b)
    k_global = SparseMatrixCSC(
        sp.m,sp.n,sp.colptr,sp.rowval,zeros(Float64,length(sp.nzval))
    )
    return k_global
end



