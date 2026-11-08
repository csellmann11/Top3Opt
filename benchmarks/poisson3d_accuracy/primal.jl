using LinearAlgebra, SparseArrays

"""P1-consistent scalar hexahedral VEM operator with stabilization alpha=h."""
function poisson_cell_operator(mesh,cell)
    ids = mesh.cells[cell]
    h,volume = mesh.sizes[cell],mesh.volumes[cell]
    centered = mesh.points[:,ids] .- mesh.centers[:,cell]
    # Each component is sign/(4h) on the corresponding four positive/negative vertices.
    G = Matrix{Float64}(undef,3,8)
    for node in 1:8, component in 1:3
        G[component,node] = (2POISSON_HEX_OFFSETS[node][component]-1)/(4h)
    end
    P = fill(1/8,8,8)+transpose(centered)*G
    defect = Matrix{Float64}(I,8,8)-P
    alpha = h
    K = volume*transpose(G)*G+alpha*transpose(defect)*defect
    return (;P,G,alpha,K,center=mesh.centers[:,cell],volume,h)
end

"""Projected body load: integral(f*Pi(v)), with the shared tetra quadrature."""
function poisson_cell_load(mesh,fan,cell,data,problem,rule)
    # For affine projection Pi(v)(x)=mean(v)+(x-center)'Gv, only four moments are needed.
    zeroth = 0.0
    first = zeros(3)
    for tet in 24(cell-1)+1:24cell
        quadrature = tetra_quadrature(fan.points[:,fan.tet_nodes[:,tet]],rule)
        points,weights = quadrature.points,quadrature.weights
        for q in eachindex(weights)
            x = points[:,q]
            value = weights[q]*problem.rhs(x)
            zeroth += value
            first .+= value .* (x-data.center)
        end
    end
    return fill(zeroth/8,8)+transpose(data.G)*first
end

"""Assemble the unconstrained scalar operator and projected right-hand side."""
function assemble_primal(mesh,fan,problem;quadrature_order=5)
    nv,nc = size(mesh.points,2),length(mesh.cells)
    rows = Vector{Int}(undef,64nc)
    columns = similar(rows)
    entries = Vector{Float64}(undef,64nc)
    rhs = zeros(nv)
    rule = tetra_rule(quadrature_order)
    cells = Vector{NamedTuple{(:P,:G,:alpha,:K,:center,:volume,:h,:load),
        Tuple{Matrix{Float64},Matrix{Float64},Float64,Matrix{Float64},Vector{Float64},
              Float64,Float64,Vector{Float64}}}}(undef,nc)
    index = 0
    for cell in 1:nc
        data = poisson_cell_operator(mesh,cell)
        load = poisson_cell_load(mesh,fan,cell,data,problem,rule)
        cells[cell] = (;data...,load)
        ids = mesh.cells[cell]
        rhs[ids] .+= load
        for j in 1:8,i in 1:8
            index += 1
            rows[index],columns[index],entries[index] = ids[i],ids[j],data.K[i,j]
        end
    end
    return (;K=sparse(rows,columns,entries,nv,nv),rhs,cells)
end

"""Solve with homogeneous boundary values while retaining the original K and rhs."""
function solve_primal(mesh,fan,problem;quadrature_order=5)
    assembled = assemble_primal(mesh,fan,problem;quadrature_order)
    K,rhs = assembled.K,assembled.rhs
    free = findall(!,mesh.boundary_vertices)
    u = zeros(length(rhs))
    if !isempty(free)
        u[free] = cholesky(Symmetric(K[free,free]))\rhs[free]
    end
    residual = K*u-rhs
    algebraic_residual = norm(residual[free])/max(norm(rhs[free]),eps(Float64))
    return (;u,K,rhs,cells=assembled.cells,energy=dot(u,K*u)/2,
        algebraic_residual,free_dofs=free)
end

"""Continuous P1 fan reconstruction using shared face and cell averages."""
function reconstruct_fan(mesh,fan,u)
    length(u) == fan.vertex_count || throw(DimensionMismatch("Original vertex solution"))
    values = Vector{Float64}(undef,size(fan.points,2))
    values[1:fan.vertex_count] .= u
    for (face,ids) in enumerate(mesh.faces)
        values[fan.vertex_count+face] = sum(u[ids])/length(ids)
    end
    for (cell,ids) in enumerate(mesh.cells)
        values[fan.vertex_count+fan.face_count+cell] = sum(u[ids])/length(ids)
    end
    return values
end
