using LinearAlgebra, StaticArrays

"""A tetrahedron in the conforming vertex/face-center/cell-center PU mesh."""
struct PU3DTetra
    nodes::SVector{4,Int}
    gradients::SVector{4,SVector{3,Float64}}
    volume::Float64
    cell::Int
    face::Int
end

"""Geometry rebuilt only when the polyhedral mesh changes.

PU indices contain original active vertices, shared leaf-face centers, then
cell centers. Cells follow the design-state order; `cell_nodes` follows each
cached VEM projector's node order. Tetrahedra are contiguous by cell.
"""
struct PU3DGeometry
    node_points::Vector{SVector{3,Float64}}
    vertex_ids::Vector{Int}
    face_ids::Vector{Int}
    element_ids::Vector{Int}
    tets::Vector{PU3DTetra}
    cell_offsets::Vector{Int}
    face_nodes::Vector{Vector{Int}}
    cell_nodes::Vector{Vector{Int}}
    face_cells::Vector{SVector{2,Int}}
end

"""Evaluate the four affine PU hats of `tet` at the physical point `x`."""
@inline function pu3d_values(tet::PU3DTetra, geometry::PU3DGeometry, x)
    offset = x - geometry.node_points[tet.nodes[1]]
    return SVector(ntuple(i -> (i == 1 ? 1.0 : 0.0) +
        dot(tet.gradients[i],offset),Val(4)))
end

function pu3d_tetra(nodes, points, cell, face)
    a,b,c,d = (points[i] for i in nodes)
    ba,ca,da = b-a,c-a,d-a
    determinant = dot(ba,cross(ca,da))
    scale = norm(ba)*norm(ca)*norm(da)
    isfinite(determinant) && abs(determinant) > 128eps(Float64)*scale ||
        throw(ArgumentError("Degenerate PU tetrahedron in cell $cell, face $face"))
    g2 = cross(ca,da)/determinant
    g3 = cross(da,ba)/determinant
    g4 = cross(ba,ca)/determinant
    return PU3DTetra(nodes,SA[-g2-g3-g4,g2,g3,g4],abs(determinant)/6,cell,face)
end

"""Build a conforming face-center fan, including all hanging subfaces.

The face and volume barycenters must see their complete boundary (the same
star-shaped geometry required by the mechanics implementation). Invalid fans
are rejected instead of silently integrating overlapping tetrahedra.
"""
function build_pu_geometry(cv,states,eldata)
    element_ids = collect(keys(states.el_id_to_state_id))
    isempty(element_ids) && throw(ArgumentError("PU geometry needs active cells"))
    vertex_ids = Int[]
    vertex_index = Dict{Int,Int}()
    cell_nodes = Vector{Int}[]
    for eid in element_ids
        local_nodes = Int[]
        sizehint!(local_nodes,length(eldata[eid].node_ids))
        for id in eldata[eid].node_ids
            index = get(vertex_index,id,0)
            if index == 0
                push!(vertex_ids,id)
                index = length(vertex_ids)
                vertex_index[id] = index
            end
            push!(local_nodes,index)
        end
        push!(cell_nodes,local_nodes)
    end
    node_points = SVector{3,Float64}[cv.mesh.nodes[id].coords for id in vertex_ids]
    owner_map = Dict{Int,SVector{2,Int}}()
    face_ids = Int[]
    cell_faces = [Int[] for _ in element_ids]
    for (cell,eid) in enumerate(element_ids)
        Ju3VEM.VEMGeo.iterate_volume_areas(cv.facedata_col,cv.mesh.topo,eid) do face,_,_
            owners = get(owner_map,face.id,SA[0,0])
            owners[2] == 0 && owners[1] != cell ||
                throw(ArgumentError("Repeated or nonmanifold PU face $(face.id)"))
            owners[1] == 0 && push!(face_ids,face.id)
            owner_map[face.id] = owners[1] == 0 ? SA[cell,0] : SA[owners[1],cell]
            push!(cell_faces[cell],face.id)
        end
    end
    face_index = Dict(id => i for (i,id) in enumerate(face_ids))
    face_nodes = Vector{Int}[]
    face_cells = SVector{2,Int}[]
    nv,nf = length(vertex_ids),length(face_ids)
    sizehint!(node_points,nv+nf+length(element_ids))
    for fid in face_ids
        fd = cv.facedata_col[fid]
        center = Ju3VEM.VEMGeo.project_to_3d(fd.dΩ.bc,fd.dΩ.plane)
        push!(node_points,center)
        vertices = [vertex_index[id] for id in fd.face_node_ids.v.args[1]]
        length(vertices) >= 3 || throw(ArgumentError("PU face $fid has fewer than three vertices"))
        push!(face_nodes,vertices)
        push!(face_cells,owner_map[fid])
        area = 0.0
        orientation = 0
        for j in eachindex(vertices)
            a = node_points[vertices[j]]
            b = node_points[vertices[mod1(j+1,length(vertices))]]
            fan_normal = cross(b-a,center-a)
            signed_area2 = dot(fan_normal,fd.dΩ.plane.n)
            scale = norm(b-a)*norm(center-a)
            isfinite(signed_area2) && abs(signed_area2) > 128eps(Float64)*scale ||
                throw(ArgumentError("Degenerate PU face-center fan at face $fid"))
            sign_now = signbit(signed_area2) ? -1 : 1
            orientation == 0 && (orientation = sign_now)
            sign_now == orientation || throw(ArgumentError("PU face $fid is not star-shaped about its center"))
            area += norm(fan_normal)/2
        end
        isapprox(area,Ju3VEM.VEMGeo.get_area(fd);rtol=1e-9,atol=0.0) ||
            throw(ArgumentError("PU fan does not cover face $fid exactly"))
    end
    for eid in element_ids
        push!(node_points,eldata[eid].bc)
    end
    tets = PU3DTetra[]
    sizehint!(tets,sum(length(face_nodes[face_index[fid]]) for faces in cell_faces for fid in faces))
    cell_offsets = Int[1]
    for (cell,eid) in enumerate(element_ids)
        center_id = nv+nf+cell
        center = node_points[center_id]
        volume = 0.0
        for fid in cell_faces[cell]
            face = face_index[fid]
            vertices = face_nodes[face]
            fc = node_points[nv+face]
            fd = cv.facedata_col[fid]
            # A center on a face produces a singular tetrahedron. Adjacent
            # cells must use opposite sides of their shared face; the volume
            # check below rejects overlapping or incomplete center fans.
            normal = fd.dΩ.plane.n
            center_side = dot(center-fc,normal)
            isfinite(center_side) && abs(center_side) >
                128eps(Float64)*max(eldata[eid].hvol,eps(Float64)) ||
                throw(ArgumentError("PU cell $eid center lies on face $fid"))
            owners = face_cells[face]
            if owners[2] != 0
                other = owners[1] == cell ? owners[2] : owners[1]
                other_side = dot(node_points[nv+nf+other]-fc,normal)
                center_side*other_side < 0 ||
                    throw(ArgumentError("PU face $fid has cell centers on the same side"))
            end
            for j in eachindex(vertices)
                nodes = SA[vertices[j],vertices[mod1(j+1,length(vertices))],nv+face,center_id]
                tet = pu3d_tetra(nodes,node_points,cell,face)
                push!(tets,tet)
                volume += tet.volume
            end
        end
        isapprox(volume,eldata[eid].volume;rtol=2e-9,atol=0.0) ||
            throw(ArgumentError("PU tetrahedra do not cover cell $eid exactly"))
        push!(cell_offsets,length(tets)+1)
    end
    return PU3DGeometry(node_points,vertex_ids,face_ids,element_ids,tets,
        cell_offsets,face_nodes,cell_nodes,face_cells)
end
