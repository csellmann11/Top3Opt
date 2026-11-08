using LinearAlgebra,StaticArrays,Statistics,Test,InteractiveUtils
include("../src/refinement_utils/pu_tetra.jl")
include("../src/refinement_utils/pu_geometry.jl")
include("../src/body_loads.jl")
include("../src/refinement_utils/pu_workspace.jl")
include("pu_numbering_probe.jl")

function minimal_vertex_star()
    points = [SA[0.,0.,0.]]
    ids=Dict(Tuple(first(points))=>1)
    add(x)=get!(ids,Tuple(x)) do
        push!(points,x);length(points)
    end
    for axis in 1:3,sgn in (-1.,1.)
        add(SVector{3}(ntuple(k -> k == axis ? sgn : 0.,3)))
    end
    for axis in 1:3,a in (-.5,.5),b in (-.5,.5)
        other=filter(!=(axis),1:3)
        add(SVector{3}(ntuple(k -> k == other[1] ? a : k == other[2] ? b : 0.,3)))
    end
    for x in (-.5,.5),y in (-.5,.5),z in (-.5,.5)
        add(SA[x,y,z])
    end
    tets=PU3DTetra[]
    for (cell,(x,y,z)) in enumerate(Iterators.product((-.5,.5),(-.5,.5),(-.5,.5)))
        center=ids[(x,y,z)]
        for axis in 1:3
            f=SA[x,y,z];f=setindex(f,0.,axis)
            facecenter=ids[Tuple(f)]
            other=filter(!=(axis),1:3)
            for (side,k) in enumerate(other)
                p=zero(f);p=setindex(p,2f[k],k)
                vertex=ids[Tuple(p)]
                nodes=side==1 ? SA[1,vertex,facecenter,center] : SA[vertex,1,facecenter,center]
                push!(tets,PU3DTetra(nodes,zero(SVector{4,SVector{3,Float64}}),1.,cell,facecenter-7))
            end
        end
    end
    geometry=PU3DGeometry(points,collect(1:7),collect(1:12),collect(1:8),tets,
        collect(1:6:49),[Int[] for _ in 1:12],[Int[] for _ in 1:8],fill(SA[1,2],12))
    offsets=fill(49,28);offsets[1]=1
    incidence=[(i,t.nodes[1]==1 ? 1 : 2) for (i,t) in enumerate(tets)]
    w=PU3DWorkspace();resize!(w.radial_slots,27);fill!(w.radial_slots,0)
    c=PU3DCache(geometry,nothing,offsets,incidence,fill(PU3D_FREE,27),fill(PU3D_FREE,12),
        Dict{Tuple{Int,Int},SVector{3,Bool}}(),SVector{3,Int}[],Vector{Float64}[],
        fill(zero(SVector{3,Float64}),27),fill(zero(SMatrix{3,3,Float64,9}),48),
        zeros(48),zeros(8),ones(8),ones(8),zeros(27),zeros(8),zeros(8),[w],1.,1.)
    return w,c
end

function minimal_numbering_probe()
    w,c=minimal_vertex_star()
    for f in (reference_pu3d_patch_number!,pu3d_patch_number!,prototype_pu_number!)
        numbering_batch!(f,w,c,[1],100)
        trials=[@elapsed numbering_batch!(f,w,c,[1],10000) for _ in 1:5]
        allocation=@allocated numbering_batch!(f,w,c,[1],10000)
        println((;method=string(f),ns=1e5median(trials),allocation,ndofs=w.ndofs))
        flush(stdout)
    end
    open(joinpath(@__DIR__,"..","Results","regularization_validation","pu_numbering_warntype.log"),"w") do io
        code_warntype(io,pu3d_patch_number!,Tuple{typeof(w),typeof(c),Int};optimize=true)
    end
end
minimal_numbering_probe()
