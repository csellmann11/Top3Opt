"""Cartesian octree leaves; n denotes initial cubes per unit length."""
function adaptive_initial_leaves(domain::Symbol,n::Integer)
    n > 0 || throw(ArgumentError("n must be positive"))
    domain in (:cube,:lshape) || throw(ArgumentError("Unknown domain"))
    horizontal = domain == :cube ? (0:n-1) : (-n:n-1)
    leaves = NamedTuple{(:origin,:h,:level),Tuple{NTuple{3,Float64},Float64,Int}}[]
    for k in 0:n-1,j in horizontal,i in horizontal
        domain == :lshape && i >= 0 && j < 0 && continue
        push!(leaves,(origin=(i/n,j/n,k/n),h=1.0/n,level=0))
    end
    return leaves
end

"""Refine marked leaves into eight children; face conformity is built afterwards."""
function adaptive_refine(leaves,marked)
    selected = falses(length(leaves))
    selected[collect(marked)] .= true
    refined = similar(leaves,0)
    for (cell,leaf) in enumerate(leaves)
        if !selected[cell]
            push!(refined,leaf)
            continue
        end
        h = leaf.h/2
        for k in 0:1,j in 0:1,i in 0:1
            push!(refined,(origin=(leaf.origin[1]+i*h,leaf.origin[2]+j*h,
                leaf.origin[3]+k*h),h,level=leaf.level+1))
        end
    end
    return refined
end

adaptive_plane_axes(axis) = axis == 1 ? (2,3) : axis == 2 ? (1,3) : (1,2)

function adaptive_xyz(axis,plane,a,b)
    axis == 1 && return (plane,a,b)
    axis == 2 && return (a,plane,b)
    return (a,b,plane)
end

"""Build shared polygonal faces, including all coarse/fine edge vertices.

Coordinates are indexed on the finest dyadic lattice, without constructing a
uniform mesh at that level. Opposite leaf faces are intersected on each plane.
Their rectangles are then given every collinear mesh vertex on their edges.
This makes the tetrahedral face fans conforming without hanging constraints.
"""
function adaptive_build_mesh(leaves;domain::Symbol,n::Integer)
    isempty(leaves) && throw(ArgumentError("Empty mesh"))
    scale = n * 2^maximum(leaf.level for leaf in leaves)
    planes = Dict{Tuple{Int,Int},Tuple{Vector{NTuple{5,Int}},Vector{NTuple{5,Int}}}}()
    for (cell,leaf) in enumerate(leaves)
        lower = ntuple(d->round(Int,leaf.origin[d]*scale),3)
        width = round(Int,leaf.h*scale)
        for axis in 1:3
            a,b = adaptive_plane_axes(axis)
            side = (lower[a],lower[a]+width,lower[b],lower[b]+width,cell)
            for (direction,coordinate) in ((1,lower[axis]),(2,lower[axis]+width))
                pair = get!(planes,(axis,coordinate)) do
                    (NTuple{5,Int}[],NTuple{5,Int}[])
                end
                push!(pair[direction],side)
            end
        end
    end
    rectangles = NTuple{6,Int}[]
    owners = Tuple{Int,Int}[]
    for ((axis,plane),(minus,plus)) in sort!(collect(planes);by=first)
        covered_minus,covered_plus = zeros(Int,length(minus)),zeros(Int,length(plus))
        for (i,left) in enumerate(minus),(j,right) in enumerate(plus)
            a0,a1 = max(left[1],right[1]),min(left[2],right[2])
            b0,b1 = max(left[3],right[3]),min(left[4],right[4])
            a0 < a1 && b0 < b1 || continue
            area = (a1-a0)*(b1-b0)
            covered_minus[i] += area
            covered_plus[j] += area
            push!(rectangles,(axis,plane,a0,a1,b0,b1))
            push!(owners,(left[5],right[5]))
        end
        for (sides,coverage) in ((minus,covered_minus),(plus,covered_plus))
            for (side,area) in zip(sides,coverage)
                fullarea = (side[2]-side[1])*(side[4]-side[3])
                if area == 0
                    push!(rectangles,(axis,plane,side[1],side[2],side[3],side[4]))
                    push!(owners,(side[5],0))
                elseif area != fullarea
                    error("A leaf face is only partly covered by neighbors")
                end
            end
        end
    end
    point_keys = Set{NTuple{3,Int}}()
    for (axis,plane,a0,a1,b0,b1) in rectangles
        for (a,b) in ((a0,b0),(a1,b0),(a1,b1),(a0,b1))
            push!(point_keys,adaptive_xyz(axis,plane,a,b))
        end
    end
    ordered_points = sort!(collect(point_keys))
    point_ids = Dict(point=>i for (i,point) in enumerate(ordered_points))
    # A line index makes adding hanging edge vertices independent of finest-grid size.
    lines = Dict{NTuple{3,Int},Vector{Int}}()
    for point in ordered_points,axis in 1:3
        a,b = adaptive_plane_axes(axis)
        push!(get!(lines,(axis,point[a],point[b]),Int[]),point[axis])
    end
    for positions in values(lines)
        sort!(positions)
    end
    function edge_points(p,q)
        axis = findfirst(d->p[d]!=q[d],1:3)
        a,b = adaptive_plane_axes(axis)
        positions = lines[(axis,p[a],p[b])]
        lo,hi = minmax(p[axis],q[axis])
        values = positions[searchsortedfirst(positions,lo):searchsortedlast(positions,hi)]
        p[axis] > q[axis] && reverse!(values)
        return [ntuple(d->d==axis ? value : p[d],3) for value in values[1:end-1]]
    end
    faces = Vector{Int}[]
    cell_faces = [Int[] for _ in leaves]
    for (face,rectangle) in enumerate(rectangles)
        axis,plane,a0,a1,b0,b1 = rectangle
        corners = [adaptive_xyz(axis,plane,a,b) for (a,b) in
            ((a0,b0),(a1,b0),(a1,b1),(a0,b1))]
        nodes = Int[]
        for edge in 1:4
            append!(nodes,[point_ids[p] for p in edge_points(corners[edge],corners[mod1(edge+1,4)])])
        end
        push!(faces,nodes)
        for owner in owners[face]
            owner == 0 || push!(cell_faces[owner],face)
        end
    end
    points = poisson_point_matrix([ntuple(d->point[d]/scale,3) for point in ordered_points])
    cells = [sort!(unique(vcat((faces[face] for face in cell_faces[cell])...)))
        for cell in eachindex(leaves)]
    centers = hcat([vec(sum(points[:,ids];dims=2))/length(ids) for ids in cells]...)
    boundary_vertices = falses(length(ordered_points))
    for (face,owner) in enumerate(owners)
        owner[2] == 0 && (boundary_vertices[faces[face]] .= true)
    end
    return (;points,cells,faces,cell_faces,face_cells=owners,centers,
        sizes=[leaf.h for leaf in leaves],volumes=[leaf.h^3 for leaf in leaves],
        boundary_vertices,domain,n=Int(n),leaves=copy(leaves))
end

"""Shared face and cell fans on the adaptive polygonal mesh."""
function adaptive_build_fan(mesh)
    nv,nf,nc = size(mesh.points,2),length(mesh.faces),length(mesh.cells)
    points = Matrix{Float64}(undef,3,nv+nf+nc)
    points[:,1:nv] .= mesh.points
    for (face,nodes) in enumerate(mesh.faces)
        points[:,nv+face] .= vec(sum(mesh.points[:,nodes];dims=2))/length(nodes)
    end
    points[:,nv+nf+1:end] .= mesh.centers
    nt = sum(length(mesh.faces[face]) for cell in 1:nc for face in mesh.cell_faces[cell])
    tet_nodes = Matrix{Int}(undef,4,nt)
    tet_cells,tet_faces = Vector{Int}(undef,nt),Vector{Int}(undef,nt)
    gradients = Vector{Matrix{Float64}}(undef,nt)
    volumes = Vector{Float64}(undef,nt)
    stars = [Int[] for _ in axes(points,2)]
    node_cells = [Int[] for _ in axes(points,2)]
    boundary_nodes = falses(size(points,2))
    boundary_nodes[1:nv] .= mesh.boundary_vertices
    boundary_edges = Set{Tuple{Int,Int}}()
    cell_nodes,cell_tets = Vector{Vector{Int}}(undef,nc),[Int[] for _ in 1:nc]
    index = 0
    for cell in 1:nc
        center = nv+nf+cell
        cell_nodes[cell] = vcat(mesh.cells[cell],nv .+ mesh.cell_faces[cell],[center])
        for face in mesh.cell_faces[cell]
            ids,facecenter = mesh.faces[face],nv+face
            physical = mesh.face_cells[face][2] == 0
            physical && (boundary_nodes[facecenter] = true)
            for j in eachindex(ids)
                a,b = ids[j],ids[mod1(j+1,length(ids))]
                index += 1
                tet_nodes[:,index] .= (a,b,facecenter,center)
                tet_cells[index],tet_faces[index] = cell,face
                gradients[index],volumes[index] = poisson_tetra_geometry(points,tet_nodes[:,index])
                push!(cell_tets[cell],index)
                for node in tet_nodes[:,index]
                    push!(stars[node],index)
                    adjacent = node_cells[node]
                    (isempty(adjacent) || last(adjacent)!=cell) && push!(adjacent,cell)
                end
                if physical
                    push!(boundary_edges,minmax(a,b))
                    push!(boundary_edges,minmax(a,facecenter))
                    push!(boundary_edges,minmax(b,facecenter))
                end
            end
        end
        isapprox(sum(volumes[cell_tets[cell]]),mesh.volumes[cell];rtol=2e-12,atol=0.0) ||
            error("Adaptive fan volume mismatch in cell $cell")
    end
    return (;points,tet_nodes,tet_cells,tet_faces,gradients,volumes,stars,node_cells,
        boundary_nodes,boundary_edges,vertex_count=nv,face_count=nf,cell_nodes,cell_tets)
end
