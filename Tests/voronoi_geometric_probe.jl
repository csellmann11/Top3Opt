# Independent coordinate checks for the mesh saved by voronoi_mesh_audit.jl.
# julia --project=. --startup-file=no Tests/voronoi_geometric_probe.jl
using Ju3VEM, LinearAlgebra, Statistics, Serialization, Printf
import Ju3VEM.VEMGeo as Geo

function geometric_probe()
    out = joinpath(@__DIR__, "..", "Results", "voronoi_mesh_audit")
    mesh = deserialize(joinpath(out, "level4_mesh.bin"))
    topo = mesh.topo
    edges = collect(RootIterator{2}(topo))
    lengths = [norm(topo.nodes[a]-topo.nodes[b]) for e in edges
               for (a,b) in (get_edge_node_ids(topo,e.id),)]
    h = median(lengths)
    tol = 1e-10
    bucket(x) = Tuple(floor.(Int, x ./ h))
    bins = Dict{NTuple{3,Int},Vector{Int}}()
    for node in topo.nodes
        is_active(node) || continue
        push!(get!(bins,bucket(node.coords),Int[]),node.id)
    end
    # Spatial search uses coordinates, independently of face/cell incidence.
    tjunctions = Tuple{Int,Int}[]
    for e in edges
        a,b = get_edge_node_ids(topo,e.id)
        p,q = topo.nodes[a].coords,topo.nodes[b].coords
        v = q-p; vv = dot(v,v)
        lo = floor.(Int,(min.(p,q) .- tol)./h)
        hi = floor.(Int,(max.(p,q) .+ tol)./h)
        for i in lo[1]:hi[1], j in lo[2]:hi[2], k in lo[3]:hi[3]
            for nid in get(bins,(i,j,k),Int[])
                nid in (a,b) && continue
                x = topo.nodes[nid].coords
                t = dot(x-p,v)/vv
                (tol < t < 1-tol) || continue
                norm(x-p-t*v) <= tol && push!(tjunctions,(e.id,nid))
            end
        end
    end
    faces = NamedTuple[]
    surface_edges = Set{Tuple{Int,Int}}()
    all_min_angles = Float64[]
    all_max_angles = Float64[]
    nonconvex = Int[]
    max_planarity_error = 0.0
    for face in RootIterator{3}(topo)
        ids = Int[]
        Geo.iterate_element_edges(topo,face.id) do nid,_,_
            push!(ids,nid)
        end
        pts = [topo.nodes[nid].coords for nid in ids]
        av = sum(cross(pts[i]-pts[1],pts[mod1(i+1,length(pts))]-pts[1])/2
                 for i in eachindex(pts))
        n = av/norm(av)
        max_planarity_error = max(max_planarity_error,
            maximum(abs(dot(p-pts[1],n)) for p in pts))
        lens = [norm(pts[mod1(i+1,length(pts))]-pts[i]) for i in eachindex(pts)]
        angles = Float64[]
        for i in eachindex(pts)
            before = pts[mod1(i-1,length(pts))]-pts[i]
            after = pts[mod1(i+1,length(pts))]-pts[i]
            angle = acosd(clamp(dot(before,after)/(norm(before)*norm(after)),-1.,1.))
            dot(cross(after,before),n) < -tol*norm(before)*norm(after) &&
                (angle = 360-angle)
            push!(angles,angle)
        end
        minimum(angles) > 0 && maximum(angles) < 180+1e-8 || push!(nonconvex,face.id)
        push!(all_min_angles,minimum(angles)); push!(all_max_angles,maximum(angles))
        if all(p -> abs(p[2]) < tol,pts)
            for i in eachindex(ids)
                push!(surface_edges,minmax(ids[i],ids[mod1(i+1,length(ids))]))
            end
            center = mean(pts)
            push!(faces,(id=face.id,x=center[1],z=center[3],area=norm(av),
                min_angle=minimum(angles),max_angle=maximum(angles),
                edge_ratio=maximum(lens)/minimum(lens),
                compactness=4norm(av)/sum(abs2,lens),points=pts))
        end
    end
    # Proper intersections among distinct edges of the front surface would
    # expose overlapping 2D cell boundaries, even if IDs were internally valid.
    sedges = collect(surface_edges)
    sbins = Dict{Tuple{Int,Int},Vector{Int}}()
    function boxes(p,q)
        lo = floor.(Int,min.(p,q)./h); hi = floor.(Int,max.(p,q)./h)
        return ((i,k) for i in lo[1]:hi[1] for k in lo[3]:hi[3])
    end
    cross2(a,b) = a[1]*b[3]-a[3]*b[1]
    crossings = Set{Tuple{Int,Int}}()
    for (i,(a,b)) in enumerate(sedges)
        p,q = topo.nodes[a].coords,topo.nodes[b].coords
        candidates = Set{Int}()
        for key in boxes(p,q)
            union!(candidates,get(sbins,key,Int[]))
        end
        for j in candidates
            c,d = sedges[j]
            any(n -> n in (a,b),(c,d)) && continue
            r,s = topo.nodes[c].coords,topo.nodes[d].coords
            den = cross2(q-p,s-r)
            abs(den) > tol^2 || continue
            t = cross2(r-p,s-r)/den; u = cross2(r-p,q-p)/den
            tol < t < 1-tol && tol < u < 1-tol && push!(crossings,(j,i))
        end
        for key in boxes(p,q)
            push!(get!(sbins,key,Int[]),i)
        end
    end
    open(joinpath(out,"front_face_quality.csv"),"w") do io
        println(io,"face_id,x,z,area,min_angle_deg,max_angle_deg,edge_ratio,compactness,vertices_xz")
        for f in sort(faces;by=f->f.min_angle)
            coords = join((string(p[1]," ",p[3]) for p in f.points),';')
            println(io,join((f.id,f.x,f.z,f.area,f.min_angle,f.max_angle,f.edge_ratio,f.compactness,coords),','))
        end
    end
    open(joinpath(out,"geometric_probe.txt"),"w") do io
        for dest in (stdout,io)
            println(dest,"Independent geometry probe: ",length(edges)," active edges; ",length(all_min_angles)," active faces")
            println(dest,"Nodes lying strictly inside an unsplit edge: ",length(tjunctions))
            println(dest,"Crossing front-surface edges: ",length(crossings))
            println(dest,"Nonconvex/degenerate faces: ",length(nonconvex))
            println(dest,"Maximum face planarity error: ",max_planarity_error)
            println(dest,"Minimum face angle [deg]: ",minimum(all_min_angles))
            println(dest,"Maximum face angle [deg]: ",maximum(all_max_angles))
            println(dest,"Edge lengths min/median/max: ",extrema(lengths)," / ",median(lengths))
            println(dest,"Front boundary faces: ",length(faces),"; total area: ",sum(f.area for f in faces))
            println(dest,"Front faces with angle below 20 degrees: ",count(f->f.min_angle<20,faces))
            println(dest,"Front faces with angle above 150 degrees: ",count(f->f.max_angle>150,faces))
            println(dest,"Maximum front-face edge ratio: ",maximum(f.edge_ratio for f in faces))
            println(dest,"Front-face area max/min: ",maximum(f.area for f in faces)/minimum(f.area for f in faces))
            println(dest,"Worst 8 front faces by minimum angle:")
            foreach(f->println(dest,(id=f.id,x=f.x,z=f.z,min_angle=f.min_angle,max_angle=f.max_angle,edge_ratio=f.edge_ratio)),sort(faces;by=f->f.min_angle)[1:8])
            isempty(tjunctions) || println(dest,"T-junction examples: ",first(tjunctions,min(10,length(tjunctions))))
            isempty(crossings) || println(dest,"Crossing examples: ",first(collect(crossings),min(10,length(crossings))))
        end
    end
    @assert isempty(tjunctions) "Unsplit edges contain mesh nodes"
    @assert isempty(crossings) "Surface edge crossings"
    @assert isempty(nonconvex) "Nonconvex faces"
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    geometric_probe()
end
