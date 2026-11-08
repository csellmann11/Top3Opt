using LinearAlgebra

const POISSON_HEX_OFFSETS = ((0,0,0),(1,0,0),(1,1,0),(0,1,0),
                           (0,0,1),(1,0,1),(1,1,1),(0,1,1))
const POISSON_HEX_FACES = ((1,4,3,2),(5,6,7,8),(1,5,8,4),
                         (2,3,7,6),(1,2,6,5),(4,8,7,3))

function poisson_point_matrix(points)
    result = Matrix{Float64}(undef,3,length(points))
    for (i,point) in enumerate(points), component in 1:3
        result[component,i] = point[component]
    end
    return result
end

"""Uniform Cartesian hexahedra on the unit cube or the reentrant L-beam.

The L-beam is ((-1,1)^2 minus {x>0,y<0}) times (0,1). `n` denotes
cells per unit length, so both domains use side length h=1/n.
"""
function build_mesh(domain::Symbol,n::Integer)
    n > 0 || throw(ArgumentError("n must be positive"))
    domain == :lbeam && (domain = :lshape)
    domain in (:cube,:lshape) || throw(ArgumentError("domain must be :cube or :lshape"))
    h = 1.0/n
    horizontal = domain == :cube ? (0:n-1) : (-n:n-1)
    vertex_ids = Dict{NTuple{3,Int},Int}()
    coordinates = NTuple{3,Float64}[]
    centers = NTuple{3,Float64}[]
    cells = Vector{Int}[]
    faces = Vector{Int}[]
    cell_faces = Vector{Int}[]
    face_cells = Tuple{Int,Int}[]
    face_ids = Dict{NTuple{4,Int},Int}()
    for k in 0:n-1, j in horizontal, i in horizontal
        domain == :lshape && i >= 0 && j < 0 && continue
        nodes = Vector{Int}(undef,8)
        for (local_node,offset) in enumerate(POISSON_HEX_OFFSETS)
            key = (i+offset[1],j+offset[2],k+offset[3])
            id = get(vertex_ids,key,0)
            if id == 0
                push!(coordinates,(key[1]*h,key[2]*h,key[3]*h))
                id = length(coordinates)
                vertex_ids[key] = id
            end
            nodes[local_node] = id
        end
        push!(cells,nodes)
        push!(centers,((i+0.5)*h,(j+0.5)*h,(k+0.5)*h))
        cell = length(cells)
        local_faces = Int[]
        for face_nodes in POISSON_HEX_FACES
            ids = [nodes[a] for a in face_nodes]
            key = Tuple(sort(ids))
            id = get(face_ids,key,0)
            if id == 0
                push!(faces,ids)
                push!(face_cells,(cell,0))
                id = length(faces)
                face_ids[key] = id
            else
                first_cell,second_cell = face_cells[id]
                second_cell == 0 || error("Nonmanifold Cartesian face")
                face_cells[id] = (first_cell,cell)
            end
            push!(local_faces,id)
        end
        push!(cell_faces,local_faces)
    end
    boundary_vertices = falses(length(coordinates))
    for (face,owners) in enumerate(face_cells)
        owners[2] == 0 || continue
        boundary_vertices[faces[face]] .= true
    end
    return (;points=poisson_point_matrix(coordinates),cells,faces,cell_faces,
        face_cells,centers=poisson_point_matrix(centers),sizes=fill(h,length(cells)),
        volumes=fill(h^3,length(cells)),boundary_vertices,domain,n=Int(n))
end

function poisson_tetra_geometry(points,nodes)
    a = points[:,nodes[1]]
    J = hcat(points[:,nodes[2]]-a,points[:,nodes[3]]-a,points[:,nodes[4]]-a)
    determinant = det(J)
    isfinite(determinant) && determinant != 0 || error("Degenerate fan tetrahedron")
    gradients = Matrix{Float64}(undef,3,4)
    gradients[:,2:4] .= transpose(inv(J))
    gradients[:,1] .= -vec(sum(gradients[:,2:4];dims=2))
    return gradients,abs(determinant)/6
end

"""Shared face-center/cell-center tetrahedral fan with 24 tetrahedra per hex."""
function build_fan(mesh)
    nv,nf,nc = size(mesh.points,2),length(mesh.faces),length(mesh.cells)
    points = Matrix{Float64}(undef,3,nv+nf+nc)
    points[:,1:nv] .= mesh.points
    for (face,nodes) in enumerate(mesh.faces)
        points[:,nv+face] .= vec(sum(mesh.points[:,nodes];dims=2))/length(nodes)
    end
    points[:,nv+nf+1:end] .= mesh.centers
    nt = 24nc
    tet_nodes = Matrix{Int}(undef,4,nt)
    tet_cells = Vector{Int}(undef,nt)
    tet_faces = Vector{Int}(undef,nt)
    gradients = Vector{Matrix{Float64}}(undef,nt)
    volumes = Vector{Float64}(undef,nt)
    stars = [Int[] for _ in axes(points,2)]
    node_cells = [Int[] for _ in axes(points,2)]
    boundary_nodes = falses(size(points,2))
    boundary_nodes[1:nv] .= mesh.boundary_vertices
    boundary_edges = Set{Tuple{Int,Int}}()
    cell_nodes = Vector{Vector{Int}}(undef,nc)
    index = 0
    for cell in 1:nc
        center = nv+nf+cell
        cell_nodes[cell] = vcat(mesh.cells[cell],nv .+ mesh.cell_faces[cell],[center])
        total_volume = 0.0
        for face in mesh.cell_faces[cell]
            ids = mesh.faces[face]
            facecenter = nv+face
            physical = mesh.face_cells[face][2] == 0
            physical && (boundary_nodes[facecenter] = true)
            for j in eachindex(ids)
                a,b = ids[j],ids[mod1(j+1,length(ids))]
                index += 1
                tet_nodes[:,index] .= (a,b,facecenter,center)
                tet_cells[index],tet_faces[index] = cell,face
                gradients[index],volumes[index] = poisson_tetra_geometry(points,tet_nodes[:,index])
                total_volume += volumes[index]
                for node in tet_nodes[:,index]
                    push!(stars[node],index)
                    adjacent = node_cells[node]
                    (isempty(adjacent) || last(adjacent) != cell) && push!(adjacent,cell)
                end
                if physical
                    push!(boundary_edges,minmax(a,b))
                    push!(boundary_edges,minmax(a,facecenter))
                    push!(boundary_edges,minmax(b,facecenter))
                end
            end
        end
        isapprox(total_volume,mesh.volumes[cell];rtol=1e-12,atol=0.0) ||
            error("Fan volume mismatch in cell $cell")
    end
    index == nt || error("Unexpected hexahedral fan size")
    return (;points,tet_nodes,tet_cells,tet_faces,gradients,volumes,stars,node_cells,
        boundary_nodes,boundary_edges,vertex_count=nv,face_count=nf,cell_nodes)
end
