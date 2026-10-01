# julia --project=. --startup-file=no Tests/voronoi_mesh_audit.jl
# Geometry/connectivity audit of the deterministic production MBB Voronoi mesh.
# Does not solve mechanics or update densities.
using Ju3VEM, Ju3VEM.FixedSizeArrays, StaticArrays, LinearAlgebra, Statistics
using Serialization, Printf
using Ju3VEM.VEMGeo: _refine!
include("../src/utils/mesh_processing_utils.jl")

const GEO = Ju3VEM.VEMGeo
const AUDIT_DIR = joinpath(@__DIR__, "..", "Results", "voronoi_mesh_audit")
const REPORT = IOBuffer()
function note(args...)
    println(args...)
    println(REPORT, args...)
    flush(stdout)
end
function save_report()
    mkpath(AUDIT_DIR)
    bytes=take!(REPORT)
    write(joinpath(AUDIT_DIR,"topology_report.txt"), bytes)
    write(REPORT,bytes)
end
function record!(errors, key, id)
    push!(get!(errors,key,Int[]),id)
end
function print_errors(errors)
    for key in sort!(collect(keys(errors)))
        vals = errors[key]
        note("  FAIL ", key, ": ", length(vals), "; first IDs ", first(vals,min(10,length(vals))))
    end
    isempty(errors) && note("  PASS: all listed invariants")
end

function coordinate_duplicates(topo, used)
    seen = Dict{Tuple,Int}()
    duplicates = Tuple{Int,Int}[]
    for nid in used
        key = Tuple(round.(topo.nodes[nid].coords; digits=11))
        if haskey(seen,key)
            push!(duplicates,(seen[key],nid))
        else
            seen[key] = nid
        end
    end
    return duplicates
end

function audit_2d(topo, name)
    errors = Dict{String,Vector{Int}}()
    owners = Dict{Int,Vector{Int}}()
    used = Set{Int}()
    areas = Float64[]
    for face in RootIterator{3}(topo)
        nids = collect(get_area_node_ids(topo,face.id))
        union!(used,nids)
        length(unique(nids)) == length(nids) || record!(errors,"repeated polygon node",face.id)
        all(n->is_active(topo.nodes[n]),nids) || record!(errors,"inactive referenced node",face.id)
        pts = [topo.nodes[n].coords for n in nids]
        signed_area = sum(pts[i][1]*pts[mod1(i+1,length(pts))][2]-pts[i][2]*pts[mod1(i+1,length(pts))][1] for i in eachindex(pts))/2
        push!(areas,abs(signed_area))
        abs(signed_area)>1e-13 || record!(errors,"nonpositive polygon area",face.id)
        starts = Int[]; edges = Int[]
        GEO.iterate_element_edges(topo,face.id) do nid,eid,_
            push!(starts,nid);push!(edges,eid)
            push!(get!(owners,eid,Int[]),face.id)
        end
        for i in eachindex(starts)
            actual = sort([starts[i],starts[mod1(i+1,length(starts))]])
            actual == sort(collect(get_edge_node_ids(topo,edges[i]))) || record!(errors,"polygon edge chain broken",face.id)
        end
        for i in eachindex(pts)
            a,b,c=pts[i],pts[mod1(i+1,length(pts))],pts[mod1(i+2,length(pts))]
            orient=(b[1]-a[1])*(c[2]-b[2])-(b[2]-a[2])*(c[1]-b[1])
            orient*sign(signed_area) >= -1e-12 || record!(errors,"nonconvex polygon",face.id)
        end
    end
    perimeter=0.0
    for (eid,own) in owners
        a,b=topo.nodes[get_edge_node_ids(topo,eid)]
        boundary=any(d->(abs(a[d])<1e-10 && abs(b[d])<1e-10)||(abs(a[d]-(d==1 ? 3. : 1.))<1e-10 && abs(b[d]-(d==1 ? 3. : 1.))<1e-10),1:2)
        length(own)==(boundary ? 1 : 2) || record!(errors,"incorrect edge ownership",eid)
        boundary && (perimeter+=norm(a-b))
    end
    duplicates=coordinate_duplicates(topo,used)
    for (_,nid) in duplicates;record!(errors,"duplicate coordinate node",nid);end
    unused=setdiff(Set(n.id for n in topo.nodes if is_active(n)),used)
    for nid in unused;record!(errors,"unused active node",nid);end
    abs(sum(areas)-3.)<1e-10 || record!(errors,"domain area mismatch",0)
    abs(perimeter-8.)<1e-10 || record!(errors,"domain perimeter mismatch",0)
    note(name,": cells=",length(areas),", nodes=",length(used),", edges=",length(owners),", area=",sum(areas),", boundary length=",perimeter)
    note("  Cell area min/max: ",extrema(areas),"; coordinate duplicates: ",length(duplicates))
    print_errors(errors)
    return isempty(errors)
end

function face_geometry(topo,fid)
    nids=Int[]; eids=Int[]
    GEO.iterate_element_edges(topo,fid) do nid,eid,_
        push!(nids,nid);push!(eids,eid)
    end
    pts=[topo.nodes[n].coords for n in nids]
    anchor=mean(pts)
    area_vector=sum(cross(pts[i]-anchor,pts[mod1(i+1,length(pts))]-anchor) for i in eachindex(pts))/2
    return (;nids,eids,pts,anchor,area_vector,area=norm(area_vector))
end

function audit_3d(mesh,name)
    topo=mesh.topo
    errors=Dict{String,Vector{Int}}()
    faceowners=Dict{Int,Vector{Int}}()
    all_faces=Dict(f.id=>face_geometry(topo,f.id) for f in RootIterator{3}(topo))
    used=Set{Int}()
    volumes=Float64[]
    ids=Int[]
    diameters=Float64[]
    scaled=Float64[]
    max_closure=0.0
    max_planarity=0.0
    node_sets=Dict{Int,Set{Int}}()
    centers=Dict{Int,SVector{3,Float64}}()
    cells_with_extra_boundary_nodes=0
    for cell in RootIterator{4}(topo)
        nids=Set{Int}()
        fids=Int[]
        GEO.iterate_volume_areas(topo,cell.id) do face,_
            push!(fids,face.id)
            push!(get!(faceowners,face.id,Int[]),cell.id)
            union!(nids,all_faces[face.id].nids)
        end
        length(unique(fids))==length(fids) || record!(errors,"repeated cell face",cell.id)
        union!(used,nids)
        node_sets[cell.id]=nids
        raw=Set(get_volume_node_ids(topo,cell.id))
        issubset(raw,nids) || record!(errors,"raw cell node absent from boundary",cell.id)
        raw!=nids && (cells_with_extra_boundary_nodes+=1)
        all(is_active(topo.nodes[n]) for n in nids) || record!(errors,"inactive referenced node",cell.id)
        pts=[topo.nodes[n].coords for n in nids]
        interior=mean(pts)
        centers[cell.id]=interior
        counts=Dict{Int,Int}()
        closure=SA[0.,0.,0.]
        volume=0.0
        total_area=0.0
        for fid in fids
            fg=all_faces[fid]
            for eid in fg.eids;counts[eid]=get(counts,eid,0)+1;end
            outward=dot(fg.area_vector,fg.anchor-interior)>=0 ? fg.area_vector : -fg.area_vector
            closure+=outward
            total_area+=fg.area
            volume+=dot(outward,fg.anchor-interior)/3
            maximum(dot(p-fg.anchor,outward) for p in pts)<=1e-10*fg.area || record!(errors,"nonconvex cell",cell.id)
        end
        all(==(2),values(counts)) || record!(errors,"cell boundary edge incidence != 2",cell.id)
        length(nids)-length(counts)+length(fids)==2 || record!(errors,"cell Euler characteristic != 2",cell.id)
        isfinite(volume)&&volume>1e-12 || record!(errors,"nonpositive volume",cell.id)
        closure_error=norm(closure)/total_area
        max_closure=max(max_closure,closure_error)
        closure_error<1e-10 || record!(errors,"cell oriented boundary not closed",cell.id)
        diameter=maximum(norm(a-b) for a in pts for b in pts)
        push!(volumes,volume);push!(ids,cell.id);push!(diameters,diameter);push!(scaled,volume/diameter^3)
    end
    boundary_areas=zeros(3,2)
    geometric_faces=Dict{Tuple,Int}()
    for (fid,fg) in all_faces
        fg.area>1e-13 && isfinite(fg.area) || record!(errors,"nonpositive face area",fid)
        planarity=maximum(abs(dot(p-fg.anchor,fg.area_vector)) for p in fg.pts)/fg.area
        max_planarity=max(max_planarity,planarity)
        planarity<1e-10 || record!(errors,"nonplanar face",fid)
        for i in eachindex(fg.nids)
            actual=sort([fg.nids[i],fg.nids[mod1(i+1,length(fg.nids))]])
            actual==sort(collect(get_edge_node_ids(topo,fg.eids[i]))) || record!(errors,"face edge chain broken",fid)
        end
        key=Tuple(sort(fg.nids))
        haskey(geometric_faces,key) && record!(errors,"duplicate face with same nodes",fid)
        geometric_faces[key]=fid
        sides=Tuple{Int,Int}[]
        for d in 1:3, side in 1:2
            target=side==1 ? 0. : (3.,.5,1.)[d]
            all(abs(p[d]-target)<1e-10 for p in fg.pts) && push!(sides,(d,side))
        end
        boundary=!isempty(sides)
        own=get(faceowners,fid,Int[])
        length(own)==(boundary ? 1 : 2) || record!(errors,"incorrect face ownership",fid)
        if length(own)==2
            s1=dot(centers[own[1]]-fg.anchor,fg.area_vector)
            s2=dot(centers[own[2]]-fg.anchor,fg.area_vector)
            s1*s2<0 || record!(errors,"adjacent cells on same face side",fid)
        end
        for (d,side) in sides;boundary_areas[d,side]+=fg.area;end
    end
    duplicates=coordinate_duplicates(topo,used)
    for (_,nid) in duplicates;record!(errors,"duplicate coordinate node",nid);end
    unused=setdiff(Set(n.id for n in topo.nodes if is_active(n)),used)
    for nid in unused;record!(errors,"unused active node",nid);end
    dh=Ju3VEM.VEMUtils.DofHandler{3}(mesh)
    Set(keys(dh.dof_mapping))==used || record!(errors,"DOF node set != used node set",0)
    globaldofs=[d for v in values(dh.dof_mapping) for d in v]
    sort(globaldofs)==collect(1:3length(used)) || record!(errors,"DOF numbering not contiguous unique",0)
    for (id,ns) in node_sets
        all(haskey(dh.dof_mapping,n) for n in ns) || record!(errors,"cell node lacks DOFs",id)
        if haskey(topo.nids_col,id)
            Set(topo.nids_col[id])==ns || record!(errors,"nids_col differs from boundary nodes",id)
        end
    end
    abs(sum(volumes)-1.5)<1e-9 || record!(errors,"domain volume mismatch",0)
    expected=[.5 .5;3. 3.;1.5 1.5]
    maximum(abs.(boundary_areas-expected))<1e-9 || record!(errors,"box boundary coverage mismatch",0)
    note(name,": cells=",length(volumes),", nodes=",length(used),", faces=",length(all_faces),", volume=",sum(volumes),", boundary area=",sum(boundary_areas))
    note("  Boundary side areas [x-/x+;y-/y+;z-/z+] = ",boundary_areas)
    note("  Volume quantiles [min,1%,median,99%,max] = ",quantile(volumes,[0.,.01,.5,.99,1.]))
    note("  V/diameter^3 quantiles = ",quantile(scaled,[0.,.01,.5,.99,1.]))
    note("  Maximum relative face-vector closure residual = ",max_closure,"; max face planarity residual = ",max_planarity)
    note("  Node coordinate duplicates = ",length(duplicates),"; DOFs = ",length(globaldofs),"; nids_col entries = ",length(topo.nids_col))
    note("  Cells with leaf-boundary nodes beyond raw corner connectivity = ",cells_with_extra_boundary_nodes)
    note("  Smallest scaled-volume cells [id,centroid,V,V/d^3]:")
    for idx in sortperm(scaled)[1:min(5,length(scaled))]
        center=mean(topo.nodes[n].coords for n in node_sets[ids[idx]])
        note("    ",ids[idx],", ",center,", ",volumes[idx],", ",scaled[idx])
    end
    print_errors(errors)
    return isempty(errors)
end

function main()
    mkpath(AUDIT_DIR)
    note("Production MBB Voronoi topology audit; Julia ",VERSION)
    note("Settings: n=4; nx=12,nz=4; three short-edge removal passes; ny=2; box=(3,.5,1); uniform levels 1..4")
    mesh2d=create_voronoi_mesh((0.,0.),(3.,1.),12,4,StandardEl{1})
    good=audit_2d(mesh2d.topo,"2D original")
    topo=mesh2d.topo
    for pass in 1:3
        topo=remove_short_edges(topo)
        good=audit_2d(topo,"2D after collapse pass $pass") && good
    end
    mesh=permute_coord_dimensions(extrude_to_3d(2,Mesh(topo,StandardEl{1}()),.5),SA[1,3,2])
    good=audit_3d(mesh,"3D uniform L1") && good
    save_report()
    for level in 2:4
        for element in RootIterator{4}(mesh.topo);_refine!(element,mesh.topo);end
        mesh=Mesh(mesh.topo,StandardEl{1}())
        if level==4
            serialize(joinpath(AUDIT_DIR,"level4_mesh.bin"),mesh)
            note("Serialized L4 mesh ready: ",joinpath(AUDIT_DIR,"level4_mesh.bin"))
        end
        good=audit_3d(mesh,"3D uniform L$level") && good
        save_report()
        GC.gc()
    end
    note("FINAL: ",good ? "PASS" : "FAIL")
    note("nids_col is a legacy cache; current CellValues uses its node mapping from leaf-face node lists. This audit checks the same geometric node coverage plus DofHandler numbering.")
    save_report()
    good || error("Mesh audit found failures; see topology_report.txt")
end
if abspath(PROGRAM_FILE)==@__FILE__
    try
        main()
    catch err
        note("ERROR: ",sprint(showerror,err))
        save_report()
        rethrow()
    end
end
