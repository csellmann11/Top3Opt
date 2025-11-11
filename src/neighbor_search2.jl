using NearestNeighbors

struct NeighInformation{T1}
    neighborhood::T1
end


using JuVEM.AdaptivMeshes: iterate_element_edges 

function get_adjacent_elements(adjacency_matrix::SparseMatrixCSC, state_id::Int)
    # Get the range of stored elements for this row using colptr
    start_idx = adjacency_matrix.colptr[state_id]
    end_idx   = adjacency_matrix.colptr[state_id + 1] - 1
     
    # Return the row indices (neighbors) directly from rowval
    # return view(adjacency_matrix.rowval, start_idx:end_idx)
    return (adjacency_matrix.rowval[i] for i in start_idx:end_idx)
end


"""
    find_nearest_neighbour(states, topo, eldata_col)

Find the nearest neighbors for each state in a topology.

This function identifies the neighboring states for each element in the topology,
including both edge-connected neighbors and node-connected neighbors.

# Arguments
- `states`: A structure containing state information, including a mapping from element IDs to state IDs.
- `topo`: The topology object representing the mesh structure.
- `eldata_col`: A collection of element data, typically used to store additional information about the elements.

# Returns
- `neighborhood`: A vector of vectors, where each inner vector contains the state IDs of the neighbors for the corresponding state.

# Assumptions
- The number of neighs found is >= 5 for each state
"""
function find_nearest_neighbour(states::TopStates,topo::Topology,eldata_col::Dict)
 
    num_elements = length(states.χ_vec)

  
    el_ids = Vector{Int}(undef,num_elements)
    e2s = states.el_id_to_state_id

    # Count total node references for pre-allocation
    total_refs = 0 
    max_node_id = -1
    counter = 1


    for el_id in keys(e2s)
        node_ids = eldata_col[el_id].node_ids
        max_node_id = max(maximum(node_ids),max_node_id)
        total_refs += eldata_col[el_id].node_ids |> length
        el_ids[counter] = el_id
        counter += 1
    end
    
    # Pre-allocate arrays for sparse matrix construction
    I = Vector{Int}(undef, total_refs)
    J = Vector{Int}(undef, total_refs)
    
    # Fill arrays with element-node relationships
    idx = 1

    for (el_id,state_id) in e2s
        for node_id in eldata_col[el_id].node_ids
            I[idx] = state_id
            J[idx] = node_id
            idx += 1
        end
    end
    
    # Create element-node incidence matrix
    n_rows = num_elements #+ n_boundary_edges
    n_cols = max_node_id #+ n_boundary_edges
    A = sparse(I, J, true, n_rows, n_cols)
    
    adjacency_matrix = (A * A')

    
    neighbors = Vector{Vector{Int}}(undef,num_elements)
    for (el_id,state_id) in e2s 

        adj_els_it = get_adjacent_elements(adjacency_matrix,state_id)
        len = length(adj_els_it) -1 

        n_count = 1
        neighs = Vector{Int}(undef,len)
        for adj_state in adj_els_it
            adj_state == state_id && continue
            neighs[n_count] = adj_state
            n_count += 1 
        end


        for neigh in topo.el_neighs[el_id]
            neigh > 0 && continue 
            push!(neighs,neigh)
        end

        neighbors[state_id] = neighs
    end

    return NeighInformation(neighbors)
end