# Opt-in dependency path for the production Voronoi/extrusion mesh family.
# Run with --project=<ToOpt3 root>. No topology-optimization solve is included.
using Ju3VEM, Ju3VEM.FixedSizeArrays, StaticArrays
isdefined(@__MODULE__,:remove_short_edges) ||
    include(joinpath(@__DIR__,"..","..","..","src","utils","mesh_processing_utils.jl"))
isdefined(@__MODULE__,:create_L_voronoi_mesh) ||
    include(joinpath(@__DIR__,"..","..","..","src","utils","benchmark_meshes.jl"))

"""Production planar Voronoi generator, production cleanup, production extrusion.

The package generator uses its own fixed MersenneTwister(123). The cube uses
n by n seed-resolution parameters; the L cross-section has side length two
and therefore uses 2n by 2n. The L generator removes the upper-right quadrant;
the adapter reflects y to match the manufactured lower-right-notch domain.
"""
function sdwr_voronoi_prism_source(domain::Symbol,n::Integer;layers=n,smooth=true)
    n > 0 && layers > 0 || throw(ArgumentError("Positive resolutions required"))
    planar = if domain == :cube
        generated = create_voronoi_mesh((0.0,0.0),(1.0,1.0),Int(n),Int(n),StandardEl{1},smooth)
        topo = generated.topo
        for _ in 1:3
            topo = remove_short_edges(topo)
        end
        Mesh(topo,StandardEl{1}())
    elseif domain == :lshape
        create_L_voronoi_mesh((-1.0,-1.0),(1.0,1.0),2Int(n),2Int(n),StandardEl{1},smooth)
    else
        throw(ArgumentError("Unknown domain $domain"))
    end
    return extrude_to_3d(Int(layers),planar,1.0)
end

"""Adapt actual production topology into the existing scalar Poisson laboratory.

Shared face topology and every iterative edge vertex are retained. Face and
cell centers use vertex means to preserve the benchmark reconstruction.
The existing scalar VEM-style operator is unchanged; its stabilization scale
is h=volume^(1/3), agreeing with the old cube scale on Cartesian cubes.
"""
function sdwr_prism_adapter(source;domain::Symbol,n::Integer)
    topo = source.topo
    original_cells = [cell.id for cell in RootIterator{4}(topo)]
    original_faces,face_index = Int[],Dict{Int,Int}()
    cell_faces = [Int[] for _ in original_cells]
    face_cells = Tuple{Int,Int}[]
    for (cell,eid) in enumerate(original_cells)
        Ju3VEM.VEMGeo.iterate_volume_areas(topo,eid) do face,_
            index = get(face_index,face.id,0)
            if index == 0
                push!(original_faces,face.id)
                index = length(original_faces)
                face_index[face.id] = index
                push!(face_cells,(cell,0))
            else
                left,right = face_cells[index]
                right == 0 && left != cell || error("Nonmanifold production face")
                face_cells[index] = (left,cell)
            end
            push!(cell_faces[cell],index)
        end
    end
    original_face_nodes = [collect(Ju3VEM.VEMGeo.get_iterative_area_vertex_ids(
        Ju3VEM.VEMGeo.get_areas(topo)[fid],topo)) for fid in original_faces]
    original_vertices = sort!(unique(vcat(original_face_nodes...)))
    vertex_index = Dict(node=>i for (i,node) in enumerate(original_vertices))
    points = Matrix{Float64}(undef,3,length(original_vertices))
    for (i,node) in enumerate(original_vertices)
        points[:,i] .= topo.nodes[node].coords
    end
    domain == :lshape && (points[2,:] .*= -1)
    faces = [[vertex_index[node] for node in ids] for ids in original_face_nodes]
    cells = [sort!(unique(vcat((faces[face] for face in fids)...))) for fids in cell_faces]
    centers = hcat([vec(sum(points[:,ids];dims=2))/length(ids) for ids in cells]...)
    volumes = zeros(length(cells))
    for cell in eachindex(cells),face in cell_faces[cell]
        ids = faces[face]
        facecenter = vec(sum(points[:,ids];dims=2))/length(ids)
        area_vector = sum(cross(points[:,ids[i]]-facecenter,
            points[:,ids[mod1(i+1,length(ids))]]-facecenter) for i in eachindex(ids))/2
        normal_sign = sign(dot(area_vector,facecenter-centers[:,cell]))
        normal_sign != 0 || error("Zero-area production face")
        for i in eachindex(ids)
            a,b = points[:,ids[i]],points[:,ids[mod1(i+1,length(ids))]]
            normal = normal_sign .* cross(b-a,facecenter-a)
            signed_six_volume = dot(normal,facecenter-centers[:,cell])
            signed_six_volume > 0 || error("Production face fan is not star-shaped")
            volumes[cell] += signed_six_volume/6
        end
    end
    isapprox(sum(volumes),domain==:cube ? 1.0 : 3.0;atol=1e-9,rtol=1e-10) ||
        error("Production prism domain volume mismatch")
    boundary_vertices = falses(size(points,2))
    for (face,owners) in enumerate(face_cells)
        owners[2] == 0 && (boundary_vertices[faces[face]] .= true)
    end
    sizes = cbrt.(volumes)
    leaves = [(origin=Tuple(vec(minimum(points[:,ids];dims=2))),h=sizes[cell],
        level=Int(Ju3VEM.VEMGeo.get_volumes(topo)[original_cells[cell]].refinement_level))
        for (cell,ids) in enumerate(cells)]
    return (;points,cells,faces,cell_faces,face_cells,centers,sizes,volumes,
        boundary_vertices,domain,n=Int(n),leaves,original_cells,original_faces,
        original_vertices,production_mesh=source,mesh_family=:voronoi_prisms)
end

function sdwr_voronoi_prism_mesh(domain::Symbol,n::Integer;layers=n,smooth=true)
    source = sdwr_voronoi_prism_source(domain,n;layers,smooth)
    return sdwr_prism_adapter(source;domain,n)
end

"""Refine selected cells using the production 3D corner-polyhedron algorithm.

This mutates only this experiment's source topology. The original prism is
split into one child per original vertex, using edge, face, and volume means.
The adapter then traverses leaf faces and hanging edge vertices afresh; no
independent benchmark refinement rule or forced balance is substituted.
"""
function sdwr_refine_prism_mesh(mesh,marked)
    topo = mesh.production_mesh.topo
    targets = [Ju3VEM.VEMGeo.get_volumes(topo)[mesh.original_cells[cell]]
        for cell in unique(collect(marked))]
    for target in targets
        Ju3VEM.VEMGeo._refine!(target,topo)
    end
    source = Mesh(topo,StandardEl{1}())
    return sdwr_prism_adapter(source;domain=mesh.domain,n=mesh.n)
end
