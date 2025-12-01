using Ju3VEM

function create_L_mesh(left::Tuple{Float64,Float64},
    right::Tuple{Float64,Float64},
    nx::Int64,
    ny::Int64,
    ::ElT,
    rel_x = 0.5, rel_y = 0.5) where {K,ElT<:ElType{K}}

    @assert left[1] <= right[1] && left[2] <= right[2] "left must be smaller than right"

    if !iseven(nx) || !iseven(ny)
        @warn "making nx and ny even"
        nx = div(nx,2)*2
        ny = div(ny,2)*2
    end

    x_coords = range(left[1],stop=right[1],length=nx+1)
    y_coords = range(left[2],stop=right[2],length=ny+1)
    
    # create the mn_coords vector
    mn_coords = Vector([SVector(x,y) for y in y_coords for x in x_coords])
    topo = Topology{2}()


    idxs = LinearIndices((nx+1,ny+1))

    half_nx = ceil(Int,nx * rel_x)
    half_ny = ceil(Int,ny * rel_y)

    node_ids = Dict{Int,Int}()

    # for j in 1:ny, i in 1:nx
    for i in 1:half_nx, j in 1:ny

        quad_nodes = Vector([
            idxs[i,j], 
            idxs[i+1,j],
            idxs[i+1,j+1],
            idxs[i,j+1]])  
            
        for (j,node_id) in enumerate(quad_nodes)
            if haskey(node_ids,node_id)
                quad_nodes[j] = node_ids[node_id]
            else
                id = add_node!(mn_coords[node_id],topo)
                node_ids[node_id] = id
                quad_nodes[j] = id
            end
        end

        add_area!(quad_nodes,topo)
    end

    for i in half_nx+1:nx, j in 1:half_ny

        quad_nodes = Vector([
            idxs[i,j], 
            idxs[i+1,j],
            idxs[i+1,j+1],
            idxs[i,j+1]])  
            
        for (j,node_id) in enumerate(quad_nodes)
            if haskey(node_ids,node_id)
                quad_nodes[j] = node_ids[node_id]
            else
                id = add_node!(mn_coords[node_id],topo)
                node_ids[node_id] = id
                quad_nodes[j] = id
            end
        end

        add_area!(quad_nodes,topo)
    end

    return Mesh(topo,ElT())
end