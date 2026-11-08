# julia --project=. --startup-file=no --threads=1 Tests/short_edge_cleanup.jl
using Test, LinearAlgebra, Serialization
using Ju3VEM, Ju3VEM.FixedSizeArrays, StaticArrays
include("../src/utils/mesh_processing_utils.jl")
include("../src/utils/benchmark_meshes.jl")

const CLEANUP_GEO = Ju3VEM.VEMGeo
const CLEANUP_SQUARE = [SA[-1.0,-1.0],SA[1.0,-1.0],SA[1.0,1.0],SA[-1.0,1.0]]
const CLEANUP_L = [SA[-1.0,-1.0],SA[1.0,-1.0],SA[1.0,0.0],
    SA[0.0,0.0],SA[0.0,1.0],SA[-1.0,1.0]]

cleanup_cross(a,b,c) = (b[1]-a[1])*(c[2]-a[2])-(b[2]-a[2])*(c[1]-a[1])
cleanup_area(points) = sum(cleanup_cross(first(points),points[i],points[mod1(i+1,length(points))])
    for i in eachindex(points))/2
cleanup_perimeter(points) = sum(norm(points[i]-points[mod1(i+1,length(points))]) for i in eachindex(points))
function cleanup_on_segment(p,a,b;tol=1e-10)
    length2 = sum(abs2,b-a)
    return abs(cleanup_cross(a,b,p)) <= tol*max(1.0,sqrt(length2)) &&
        -tol <= dot(p-a,b-a) <= length2+tol
end
cleanup_physical_edge(a,b,boundary) = any(eachindex(boundary)) do i
    c,d = boundary[i],boundary[mod1(i+1,length(boundary))]
    cleanup_on_segment(a,c,d) && cleanup_on_segment(b,c,d)
end
function cleanup_segments_intersect(a,b,c,d;tol=1e-12)
    p,q,r,s = cleanup_cross(a,b,c),cleanup_cross(a,b,d),cleanup_cross(c,d,a),cleanup_cross(c,d,b)
    return (p*q < -tol^2 && r*s < -tol^2) ||
        (abs(p)<=tol && cleanup_on_segment(c,a,b;tol)) ||
        (abs(q)<=tol && cleanup_on_segment(d,a,b;tol)) ||
        (abs(r)<=tol && cleanup_on_segment(a,c,d;tol)) ||
        (abs(s)<=tol && cleanup_on_segment(b,c,d;tol))
end
function cleanup_snapshot(topo)
    io = IOBuffer()
    serialize(io,topo)
    return take!(io)
end
function cleanup_polygon(topo,face)
    nodes,edges = Int[],Int[]
    CLEANUP_GEO.iterate_element_edges(topo,face.id) do node,edge,_
        push!(nodes,node)
        push!(edges,edge)
    end
    return nodes,edges
end
function cleanup_signature(topo)
    return (;nodes=[(node.id,Tuple(node.coords)) for node in topo.nodes],
        polygons=[Tuple(first(cleanup_polygon(topo,face))) for face in RootIterator{3}(topo)])
end

"""Independent geometric and directed-incidence audit, including hanging segments."""
function cleanup_check(topo,boundary;faces=nothing,dense=true)
    leaves = collect(RootIterator{3}(topo))
    faces === nothing || @test length(leaves) == faces
    owners = Dict{Tuple{Int,Int},Vector{Tuple{Int,Int,Int}}}()
    used = Set{Int}()
    area = 0.0
    for face in leaves
        ids,eids = cleanup_polygon(topo,face)
        @test length(ids) >= 3
        @test length(unique(ids)) == length(ids)
        valid = all(id->1<=id<=length(topo.nodes) && is_active(topo.nodes[id]),ids)
        @test valid
        valid || error("Invalid referenced node in cleanup output")
        union!(used,ids)
        points = [topo.nodes[id].coords for id in ids]
        signed_area = cleanup_area(points)
        @test isfinite(signed_area) && signed_area > 1e-13
        area += signed_area
        for i in eachindex(ids)
            next = mod1(i+1,length(ids))
            @test Set(get_edge_node_ids(topo,eids[i])) == Set((ids[i],ids[next]))
            @test is_active(CLEANUP_GEO.get_edges(topo)[eids[i]])
            @test norm(points[i]-points[next]) > 1e-12
            @test cleanup_cross(points[i],points[next],points[mod1(i+2,length(ids))]) >= -1e-11
            push!(get!(owners,minmax(ids[i],ids[next]),Tuple{Int,Int,Int}[]),(face.id,ids[i],ids[next]))
            for j in i+1:length(ids)
                (j == next || mod1(j+1,length(ids)) == i) && continue
                @test !cleanup_segments_intersect(points[i],points[next],points[j],points[mod1(j+1,length(ids))])
            end
        end
    end
    boundary_degree = Dict{Int,Int}()
    adjacency = Dict(face.id=>Int[] for face in leaves)
    perimeter = 0.0
    for ((a,b),incidence) in owners
        physical = cleanup_physical_edge(topo.nodes[a].coords,topo.nodes[b].coords,boundary)
        @test length(incidence) == (physical ? 1 : 2)
        if length(incidence) == 2
            entry1,entry2 = incidence
            @test entry1[2] == entry2[3] && entry1[3] == entry2[2]
            @test entry1[1] != entry2[1]
            push!(adjacency[entry1[1]],entry2[1])
            push!(adjacency[entry2[1]],entry1[1])
        elseif physical
            perimeter += norm(topo.nodes[a]-topo.nodes[b])
            boundary_degree[a] = get(boundary_degree,a,0)+1
            boundary_degree[b] = get(boundary_degree,b,0)+1
        end
    end
    @test all(==(2),values(boundary_degree))
    @test area ≈ cleanup_area(boundary) atol=2e-10 rtol=2e-11
    @test perimeter ≈ cleanup_perimeter(boundary) atol=2e-10 rtol=2e-11
    @test length(used)-length(owners)+length(leaves) == 1
    visited,pending = Set{Int}(),[first(leaves).id]
    while !isempty(pending)
        face = pop!(pending)
        face in visited && continue
        push!(visited,face)
        append!(pending,adjacency[face])
    end
    @test length(visited) == length(leaves)
    for corner in boundary
        @test count(id->norm(topo.nodes[id].coords-corner)<1e-10,used) == 1
    end
    @test length(unique(Tuple(topo.nodes[id].coords) for id in used)) == length(used)
    if dense
        @test used == Set(eachindex(topo.nodes))
        @test all(i->topo.nodes[i].id==i,eachindex(topo.nodes))
        @test length(owners) == length(CLEANUP_GEO.get_edges(topo))
        @test length(leaves) == length(CLEANUP_GEO.get_areas(topo))
    end
    return (;area,perimeter,nodes=length(used),faces=length(leaves))
end

function cleanup_fixture(points,polygons)
    topo = Topology{2}()
    for point in points
        add_node!(SVector{2,Float64}(point),topo)
    end
    for polygon in polygons
        add_area!(collect(polygon),topo)
    end
    return topo
end

"""Two short edges meet at C; contracting B-C acquires faces absent at C."""
function cleanup_chain_fixture()
    points = [SA[x,y] for (y,xs) in ((-1.0,(-1.0,-0.5,0.0,0.5,1.0)),
        (0.0,(-1.0,-0.01,0.0,0.01,1.0)),(1.0,(-1.0,-0.5,0.0,0.5,1.0))) for x in xs]
    polygons = [(5row+col,5row+col+1,5(row+1)+col+1,5(row+1)+col) for row in 0:1 for col in 1:4]
    return cleanup_fixture(points,polygons)
end

function cleanup_three_passes(topo)
    for _ in 1:3
        topo = remove_short_edges(topo)
    end
    return topo
end

"""Extrusion/refinement smoke audit independent of the scalar Poisson adapter."""
function cleanup_check_extrusion(mesh)
    topo = mesh.topo
    owners = Dict{Int,Int}()
    total_volume,boundary_area = 0.0,0.0
    for cell in RootIterator{4}(topo)
        faces = Int[]
        CLEANUP_GEO.iterate_volume_areas(topo,cell.id) do face,_
            push!(faces,face.id)
        end
        polygons = [collect(CLEANUP_GEO.get_iterative_area_vertex_ids(face,topo)) for face in faces]
        ids = unique(vcat(polygons...))
        @test all(id->1<=id<=length(topo.nodes) && is_active(topo.nodes[id]),ids)
        center = sum(topo.nodes[id].coords for id in ids)/length(ids)
        volume = 0.0
        for (face,vertices) in zip(faces,polygons)
            points = [topo.nodes[id].coords for id in vertices]
            anchor = sum(points)/length(points)
            area_vector = sum(cross(points[i]-anchor,points[mod1(i+1,length(points))]-anchor) for i in eachindex(points))/2
            orientation = sign(dot(area_vector,anchor-center))
            @test norm(area_vector)>1e-13 && orientation!=0
            for i in eachindex(points)
                signed_six_volume = orientation*dot(cross(points[mod1(i+1,length(points))]-points[i],anchor-points[i]),anchor-center)
                @test signed_six_volume > 1e-14
                volume += signed_six_volume/6
            end
            physical = all(p->abs(p[3])<1e-10,points) || all(p->abs(p[3]-1)<1e-10,points) ||
                all(i->cleanup_physical_edge(points[i][SA[1,2]],points[mod1(i+1,length(points))][SA[1,2]],CLEANUP_L),eachindex(points))
            count = get(owners,face,0)+1
            owners[face] = count
            physical && count==1 && (boundary_area += norm(area_vector))
        end
        @test volume > 0
        total_volume += volume
    end
    @test all(n->n in (1,2),values(owners))
    @test total_volume ≈ 3.0 atol=5e-9
    @test boundary_area ≈ 14.0 atol=5e-9
end

function cleanup_orientation_scale_tests()
    @testset "Clockwise orientation and translated/scaled coordinates" begin
        square = [SA[0.0,0.0],SA[1.0,0.0],SA[1.0,1.0],SA[0.0,1.0]]
        for scale in (1e-5,1.0,1e3),clockwise in (false,true)
            shift = SA[1234.5,-678.25]
            points = [shift+scale*p for p in
                [square[1],SA[0.001,0.0],square[2],square[3],square[4]]]
            ids = clockwise ? [5,4,3,2,1] : [1,2,3,4,5]
            source = cleanup_fixture(points,[ids])
            before = cleanup_snapshot(source)
            expected = cleanup_area(points[ids])
            transformed = remove_short_edges(source)
            @test cleanup_snapshot(source) == before
            @test length(transformed.nodes) == 4
            polygon = only(collect(RootIterator{3}(transformed)))
            output = [transformed.nodes[id].coords for id in first(cleanup_polygon(transformed,polygon))]
            @test sign(cleanup_area(output)) == (clockwise ? -1 : 1)
            @test cleanup_area(output) ≈ expected rtol=2e-8
            @test length(collect(RootIterator{3}(transformed))) == 1
        end
    end
end

if "--orientation-only" in ARGS
    cleanup_orientation_scale_tests()
else
@testset "Short-edge cleanup: topology-preserving contractions" begin
    BLAS.set_num_threads(1)
    @testset "Dense maps and rejected malformed connectivity" begin
        topo = cleanup_fixture([(0.0,0.0),(0.001,0.0),(0.0,1.0)],[(1,2,3)])
        add_node!(SA[99.0,99.0],topo)
        before = cleanup_snapshot(topo)
        result = remove_short_edges(topo)
        @test cleanup_snapshot(topo) == before
        @test result !== topo && result.nodes[1] !== topo.nodes[1]
        cleanup_check(result,[SA[0.0,0.0],SA[0.001,0.0],SA[0.0,1.0]];faces=1)
        @test length(result.nodes) == 3
        invalid = deepcopy(topo)
        CLEANUP_GEO.deactivate!(invalid.nodes[2])
        @test collect(create_dense_node_id_map(invalid)) == [1,0,2,3]
        before = cleanup_snapshot(invalid)
        @test_throws ArgumentError remove_short_edges(invalid)
        @test cleanup_snapshot(invalid) == before
    end
    @testset "Shared-vertex contraction chain and boundary corners" begin
        topo = cleanup_chain_fixture()
        cleanup_check(topo,CLEANUP_SQUARE;faces=8)
        before = cleanup_snapshot(topo)
        cleaned_once = remove_short_edges(topo)
        cleanup_check(cleaned_once,CLEANUP_SQUARE;faces=8)
        result = cleanup_three_passes(topo)
        cleanup_check(result,CLEANUP_SQUARE;faces=8)
        @test length(result.nodes) <= length(topo.nodes)-2
        @test cleanup_snapshot(topo) == before
        @test cleanup_signature(result) == cleanup_signature(cleanup_three_passes(topo))
        square = [SA[0.0,0.0],SA[1.0,0.0],SA[1.0,1.0],SA[0.0,1.0]]
        boundary_extra = cleanup_fixture([square[1],SA[0.001,0.0],square[2],square[3],square[4]],[(1,2,3,4,5)])
        compact = remove_short_edges(boundary_extra)
        cleanup_check(compact,square;faces=1)
        @test length(compact.nodes) == 4
        corner_points = [SA[0.0,0.0],SA[1.0,0.0],SA[1.0,0.01],SA[0.0,1.0]]
        corners = cleanup_fixture(corner_points,[(1,2,3,4)])
        preserved = remove_short_edges(corners)
        cleanup_check(preserved,corner_points;faces=1)
        @test length(preserved.nodes) == 4
    end
    @testset "Refined input keeps hanging segments and leaves input hierarchy unchanged" begin
        topo = cleanup_fixture([SA[x,y] for y in (-1.0,1.0) for x in (-1.0,0.0,1.0)],[(1,2,5,4),(2,3,6,5)])
        CLEANUP_GEO._refine!(first(RootIterator{3}(topo)),topo)
        count = length(collect(RootIterator{3}(topo)))
        @test any(length(first(cleanup_polygon(topo,face)))>length(get_area_node_ids(topo,face.id))
            for face in RootIterator{3}(topo))
        before = cleanup_snapshot(topo)
        result = remove_short_edges(topo)
        @test cleanup_snapshot(topo) == before
        cleanup_check(result,CLEANUP_SQUARE;faces=count)
    end
    cleanup_orientation_scale_tests()
    @testset "Actual failing Voronoi resolutions and production clipping" begin
        for resolution in (20,24)
            raw = create_voronoi_mesh((-1.0,-1.0),(1.0,1.0),resolution,resolution,StandardEl{1},true).topo
            faces = length(collect(RootIterator{3}(raw)))
            before = cleanup_snapshot(raw)
            cleanup_check(raw,CLEANUP_SQUARE;faces,dense=false)
            current = raw
            node_counts = [length(raw.nodes)]
            for _ in 1:3
                snapshot = cleanup_snapshot(current)
                following = remove_short_edges(current)
                @test cleanup_snapshot(current) == snapshot
                cleanup_check(following,CLEANUP_SQUARE;faces)
                push!(node_counts,length(following.nodes))
                current = following
            end
            @test cleanup_snapshot(raw) == before
            @test last(node_counts) < first(node_counts)
            repeated = nothing
            seconds = @elapsed repeated = cleanup_three_passes(raw)
            @test cleanup_signature(current) == cleanup_signature(repeated)
            println("SHORT_EDGE_CLEANUP_TIMING ",(;resolution,node_counts,three_pass_seconds=seconds))
            clipped = create_L_voronoi_mesh((-1.0,-1.0),(1.0,1.0),resolution,resolution,StandardEl{1},true)
            cleanup_check(clipped.topo,CLEANUP_L)
            if resolution == 20
                extruded = extrude_to_3d(2,clipped,1.0)
                cleanup_check_extrusion(extruded)
                target = first(RootIterator{4}(extruded.topo))
                CLEANUP_GEO._refine!(target,extruded.topo)
                cleanup_check_extrusion(Mesh(extruded.topo,StandardEl{1}()))
            end
        end
    end
end
end
