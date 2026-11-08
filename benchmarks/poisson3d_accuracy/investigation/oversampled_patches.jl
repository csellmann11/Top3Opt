"""Face adjacency of an arbitrary conforming tetrahedral mesh."""
function investigation_tet_neighbors(fan)
    neighbors = [Int[] for _ in fan.volumes]
    owners = Dict{NTuple{3,Int},Int}()
    for tet in eachindex(fan.volumes), omitted in 1:4
        face = sort([fan.tet_nodes[j,tet] for j in 1:4 if j != omitted])
        key = (face[1],face[2],face[3])
        if haskey(owners,key)
            other = owners[key]
            push!(neighbors[tet],other)
            push!(neighbors[other],tet)
        else
            owners[key] = tet
        end
    end
    return neighbors
end

function investigation_buffer_patch(inner, neighbors, layers)
    selected = Set(inner)
    frontier = collect(inner)
    for _ in 1:layers
        next = Int[]
        for tet in frontier, adjacent in neighbors[tet]
            adjacent in selected && continue
            push!(selected,adjacent)
            push!(next,adjacent)
        end
        frontier = next
    end
    return sort!(collect(selected))
end

function investigation_dof_incidence(system)
    counts = zeros(Int,system.total_dofs)
    for dof in system.tet_dofs
        counts[dof] += 1
    end
    return counts
end

"""A free patch DOF must have its entire global support inside the buffer."""
function investigation_patch_interior(tets, system, incidence)
    counts = Dict{Int,Int}()
    for tet in tets, dof in view(system.tet_dofs,:,tet)
        counts[dof] = get(counts,dof,0) + 1
    end
    return sort!([dof for (dof,count) in counts
        if count == incidence[dof] && !system.boundary_mask[dof]])
end

"""Read the residual on the original PU star, not on its larger solve patch."""
function investigation_local_readout(fan, data, node, tet_dofs, rows, delta)
    value = 0.0
    for tet in fan.stars[node]
        hat = findfirst(==(node),view(fan.tet_nodes,:,tet))
        for shape in 1:10
            row = get(rows,tet_dofs[shape,tet],0)
            row == 0 && continue
            value += data.weighted_residual[shape,hat,tet]*delta[row]
        end
    end
    return value
end

"""Buffer the existing Dirichlet patch solve while keeping its PU readout fixed.

The global P2 matrix is reused only as an assembly convenience. Complete-support
interior DOFs have no stiffness/load contribution outside the selected patch,
so their global principal submatrix is exactly the local Dirichlet matrix.
The system rhs may be a primal or dual residual; readout_data supplies the
residual used in the DWR evaluation. There is no point fitting.
"""
function oversampled_patch_estimate(mesh,fan,readout_data,system;
        layers=1,factor=0.5,include_corrections=true,
        neighbors=investigation_tet_neighbors(fan),
        incidence=investigation_dof_incidence(system))
    layers >= 0 || throw(ArgumentError("Buffer layer count must be nonnegative"))
    nn = size(fan.points,2)
    nodal = include_corrections ? copy(readout_data.conversion) : zeros(nn)
    correction = include_corrections ? readout_data.correction : zeros(length(mesh.cells))
    counts = zeros(Int,nn)
    tetra_counts = zeros(Int,nn)
    for node in 1:nn
        tets = investigation_buffer_patch(fan.stars[node],neighbors,layers)
        dofs = investigation_patch_interior(tets,system,incidence)
        counts[node],tetra_counts[node] = length(dofs),length(tets)
        isempty(dofs) && continue
        delta = cholesky(Symmetric(Matrix(system.K[dofs,dofs]))) \ system.rhs[dofs]
        rows = Dict(dof=>row for (row,dof) in enumerate(dofs))
        nodal[node] += factor*investigation_local_readout(
            fan,readout_data,node,system.tet_dofs,rows,delta)
    end
    statistics = (;patch_count=nn,total_dofs=sum(counts),max_dofs=maximum(counts),
        buffer_layers=layers,total_patch_tetrahedra=sum(tetra_counts),
        largest_patch_tetrahedra=maximum(tetra_counts),
        convention="buffered Dirichlet solves, original PU residual readout")
    return poisson_collect_estimate(mesh,fan,nodal,correction,counts,copy(counts),statistics)
end
