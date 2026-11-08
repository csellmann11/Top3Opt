"""P1-consistent VEM-style extension of the original Cartesian benchmark.

On split polygonal faces, the explicit shared P1 face fan defines the boundary
integrals used for the gradient projection. This is a laboratory extension,
not an implementation of the exact harmonic-face 3D VEM projector. It agrees
with the original operator on uniform cubes. Stabilization remains alpha=h.
"""
function adaptive_cell_operator(mesh,fan,cell)
    ids = mesh.cells[cell]
    count = length(ids)
    lookup = Dict(node=>i for (i,node) in enumerate(ids))
    G = zeros(3,count)
    volume = mesh.volumes[cell]
    for tet in fan.cell_tets[cell]
        nodes = fan.tet_nodes[:,tet]
        gradient = fan.gradients[tet]
        weight = fan.volumes[tet]/volume
        G[:,lookup[nodes[1]]] .+= weight .* gradient[:,1]
        G[:,lookup[nodes[2]]] .+= weight .* gradient[:,2]
        face_nodes = mesh.faces[fan.tet_faces[tet]]
        for node in face_nodes
            G[:,lookup[node]] .+= (weight/length(face_nodes)) .* gradient[:,3]
        end
        G .+= (weight/count) .* gradient[:,4]
    end
    center = mesh.centers[:,cell]
    centered = mesh.points[:,ids] .- center
    P = fill(1/count,count,count) + transpose(centered)*G
    defect = Matrix{Float64}(I,count,count)-P
    h,alpha = mesh.sizes[cell],mesh.sizes[cell]
    K = volume*transpose(G)*G + alpha*transpose(defect)*defect
    return (;P,G,alpha,K,center,volume,h)
end

function adaptive_cell_load(mesh,fan,cell,data,problem,rule)
    zeroth,first = 0.0,zeros(3)
    for tet in fan.cell_tets[cell]
        quadrature = tetra_quadrature(fan.points[:,fan.tet_nodes[:,tet]],rule)
        for q in eachindex(quadrature.weights)
            x = view(quadrature.points,:,q)
            value = quadrature.weights[q]*problem.rhs(x)
            zeroth += value
            first .+= value .* (x-data.center)
        end
    end
    return fill(zeroth/length(mesh.cells[cell]),length(mesh.cells[cell]))+transpose(data.G)*first
end

function adaptive_assemble(mesh,fan,problem;quadrature_order=6)
    nv,nc = size(mesh.points,2),length(mesh.cells)
    entries_count = sum(length(ids)^2 for ids in mesh.cells)
    rows,columns = Vector{Int}(undef,entries_count),Vector{Int}(undef,entries_count)
    entries = Vector{Float64}(undef,entries_count)
    rhs = zeros(nv)
    cells = NamedTuple[]
    rule = tetra_rule(quadrature_order)
    index = 0
    for cell in 1:nc
        data = adaptive_cell_operator(mesh,fan,cell)
        load = adaptive_cell_load(mesh,fan,cell,data,problem,rule)
        push!(cells,(;data...,load))
        ids = mesh.cells[cell]
        rhs[ids] .+= load
        for j in eachindex(ids),i in eachindex(ids)
            index += 1
            rows[index],columns[index],entries[index] = ids[i],ids[j],data.K[i,j]
        end
    end
    return (;K=sparse(rows,columns,entries,nv,nv),rhs,cells)
end

"""Homogeneous-Dirichlet solve; retain factorization for another RHS."""
function adaptive_solve(assembled,mesh;factor=nothing)
    K,rhs = assembled.K,assembled.rhs
    free = findall(!,mesh.boundary_vertices)
    u = zeros(length(rhs))
    if !isempty(free)
        factor === nothing && (factor=cholesky(Symmetric(K[free,free])))
        u[free] = factor\rhs[free]
    end
    residual = K*u-rhs
    algebraic_residual = norm(residual[free])/max(norm(rhs[free]),eps(Float64))
    return (;u,K,rhs,cells=assembled.cells,energy=dot(u,K*u)/2,
        algebraic_residual,free_dofs=free,factor)
end

adaptive_reconstruct(mesh,fan,u) = reconstruct_fan(mesh,fan,u)
