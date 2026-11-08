# Read-only audit of the production cleanup; mutates only fresh generated meshes.
include("load.jl")
include("voronoi_prisms.jl")

function sdwr_cleanup_bad_references(topo)
    invalid,inactive = Tuple{Int,Int}[],Tuple{Int,Int}[]
    for face in RootIterator{3}(topo),node in get_area_node_ids(topo,face.id)
        if !(1 <= node <= length(topo.nodes))
            push!(invalid,(face.id,node))
        elseif !Ju3VEM.VEMGeo.is_active(topo.nodes[node])
            push!(inactive,(face.id,node))
        end
    end
    return (;invalid,inactive)
end

function sdwr_diagnose_cleanup(n)
    topo = create_voronoi_mesh((-1.0,-1.0),(1.0,1.0),2n,2n,StandardEl{1}).topo
    println("CLEANUP_DIAGNOSTIC ",(;n,stage="raw",sdwr_cleanup_bad_references(topo)...)); flush(stdout)
    for pass in 1:3
        before = topo
        try
            topo = remove_short_edges(before)
        catch exception
            println("CLEANUP_FAILURE ",(;n,pass,type=string(typeof(exception)),
                error=sprint(showerror,exception),old_references=sdwr_cleanup_bad_references(before)))
            flush(stdout)
            return false
        end
        old = sdwr_cleanup_bad_references(before)
        current = sdwr_cleanup_bad_references(topo)
        println("CLEANUP_DIAGNOSTIC ",(;n,pass,old_inactive=old.inactive,
            new_invalid=current.invalid,new_inactive=current.inactive)); flush(stdout)
        for (face,node) in old.inactive
            println("STALE_NODE ",(;n,pass,face,node,coordinates=Tuple(before.nodes[node].coords),
                face_nodes=collect(get_area_node_ids(before,face)))); flush(stdout)
        end
    end
    return true
end

if abspath(PROGRAM_FILE) == @__FILE__
    BLAS.set_num_threads(1)
    for n in (10,12)
        println("CLEANUP_BEGIN ",n); flush(stdout)
        good = sdwr_diagnose_cleanup(n)
        if good
            mesh = sdwr_voronoi_prism_mesh(:lshape,n)
            fan = adaptive_build_fan(mesh)
            println("PRISM_VIABILITY ",(;n,cells=length(mesh.cells),vertices=size(mesh.points,2),
                volume=sum(mesh.volumes),tetrahedra=length(fan.volumes)))
            flush(stdout)
        end
    end
end
