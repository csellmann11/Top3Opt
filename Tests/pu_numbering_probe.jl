isdefined(@__MODULE__,:PU3DWorkspace) || include("pu_dwr.jl")
include("pu_numbering_reference.jl")
using InteractiveUtils

function prototype_pu_number!(w,c,node)
    first,last = c.offsets[node],c.offsets[node+1]-1
    incidence = first:last
    pu3d_radial_star(c,node,incidence) || return pu3d_patch_number_generic!(w,c,node)
    @inbounds for other in w.radial_nodes
        w.radial_slots[other] = 0
    end
    empty!(w.radial_nodes); empty!(w.masks); empty!(w.dofs)
    w.ndofs = 0
    push!(w.dofs,SA[0,0,0]);push!(w.dofs,SA[0,0,0])
    resize!(w.maps,length(incidence))
    @inbounds for (localtet,index) in enumerate(incidence)
        ti,hat = c.incidence[index]
        t = c.geometry.tets[ti]
        if hat == 1
            a = pu3d_radial_slot!(w,c,t,5,1)
            b = pu3d_radial_slot!(w,c,t,6,1)
            d = pu3d_radial_slot!(w,c,t,7,1)
            w.maps[localtet] = SA[2,1,1,1,a,b,d,1,1,1]
        elseif hat == 2
            a = pu3d_radial_slot!(w,c,t,5,2)
            b = pu3d_radial_slot!(w,c,t,8,2)
            d = pu3d_radial_slot!(w,c,t,9,2)
            w.maps[localtet] = SA[1,2,1,1,a,1,1,b,d,1]
        elseif hat == 3
            a = pu3d_radial_slot!(w,c,t,6,3)
            b = pu3d_radial_slot!(w,c,t,8,3)
            d = pu3d_radial_slot!(w,c,t,10,3)
            w.maps[localtet] = SA[1,1,2,1,1,a,1,b,1,d]
        else
            a = pu3d_radial_slot!(w,c,t,7,4)
            b = pu3d_radial_slot!(w,c,t,9,4)
            d = pu3d_radial_slot!(w,c,t,10,4)
            w.maps[localtet] = SA[1,1,1,2,1,1,a,1,b,d]
        end
    end
    n = w.ndofs
    mask = c.fixed[node]
    a = mask[1] ? 0 : (n += 1)
    b = mask[2] ? 0 : (n += 1)
    d = mask[3] ? 0 : (n += 1)
    w.dofs[2] = SA[a,b,d]
    w.ndofs = n
    w.max_used = max(w.max_used,n)
    if size(w.matrix,1) < n
        capacity = max(n,ceil(Int,1.25size(w.matrix,1)))
        w.matrix = zeros(capacity,capacity)
        resize!(w.rhs,capacity);resize!(w.weighted,capacity)
    end
    @inbounds for j in 1:n
        w.rhs[j] = w.weighted[j] = 0.0
        for i in j:n
            w.matrix[i,j] = 0.0
        end
    end
    return incidence
end

function numbering_batch!(f,w,c,nodes,repeats)
    for _ in 1:repeats,node in nodes
        f(w,c,node)
    end
    return w.ndofs
end

function probe_numbering(n=8)
    cv,states,pars,ch,loads,u,K,ed,c = pu_clamped_problem(n)
    w = first(c.workspaces)
    nv,nf = length(c.geometry.vertex_ids),length(c.geometry.face_ids)
    groups = [Int[] for _ in 1:4]
    for node in eachindex(c.nodal)
        incidence = c.offsets[node]:c.offsets[node+1]-1
        group = node <= nv ? 1 : node <= nv+nf ? 2 : pu3d_radial_star(c,node,incidence) ? 3 : 4
        push!(groups[group],node)
    end
    for node in eachindex(c.nodal)
        pu3d_patch_number!(w,c,node)
        maps,dofs = copy(w.maps),copy(w.dofs)
        prototype_pu_number!(w,c,node)
        @test w.maps == maps
        @test w.dofs == dofs
    end
    open(joinpath(@__DIR__,"..","Results","regularization_validation","pu_numbering_warntype.log"),"w") do io
        code_warntype(io,pu3d_patch_number!,Tuple{typeof(w),typeof(c),Int};optimize=true)
    end
    for (group,nodes) in zip((:vertices,:faces,:radial_cells,:boundary_cells),groups)
        for f in (reference_pu3d_patch_number!,pu3d_patch_number!,prototype_pu_number!)
            numbering_batch!(f,w,c,nodes,2)
            times = Float64[]
            for _ in 1:5
                push!(times,@elapsed numbering_batch!(f,w,c,nodes,10))
            end
            alloc = @allocated numbering_batch!(f,w,c,nodes,1)
            println((;group,count=length(nodes),method=string(f),ms=100median(times),alloc))
            flush(stdout)
        end
    end
end

abspath(PROGRAM_FILE) == (@__FILE__) && probe_numbering()
