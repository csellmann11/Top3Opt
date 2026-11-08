# Include after pu_dwr.jl; isolates the old and current patch stages on one mesh.
include("pu_numbering_reference.jl")

function pu_stage_times(c,u,states,ed,pars;reference=false)
    pu3d_prepare!(c,u,states,ed,pars,states.χ_vec,1.)
    w=first(c.workspaces)
    numbering,assembly,solving=0,0,0
    ndofs_total=0
    for node in eachindex(c.nodal)
        start=time_ns()
        incidence=reference ? reference_pu3d_patch_number!(w,c,node) : pu3d_patch_number!(w,c,node)
        numbering+=time_ns()-start
        start=time_ns()
        for (localtet,index) in enumerate(incidence)
            ti,hat=c.incidence[index]
            if reference
                reference_pu3d_add_tetra!(w,c,ti,hat,w.maps[localtet])
            else
                pu3d_add_tetra!(w,c,ti,hat,w.maps[localtet])
            end
        end
        assembly+=time_ns()-start
        start=time_ns()
        if reference
            reference_pu3d_solve!(w.matrix,w.rhs,w.ndofs)
        else
            pu3d_solve!(w.matrix,w.rhs,w.ndofs)
        end
        solving+=time_ns()-start
        ndofs_total+=w.ndofs
    end
    return (;reference,nodes=length(c.nodal),numbering_ms=numbering/1e6,
        assembly_ms=assembly/1e6,solve_ms=solving/1e6,ndofs_total)
end

function reference_estimate_pu!(c,u,states,ed,pars;parallel=false)
    pu3d_prepare!(c,u,states,ed,pars,states.χ_vec,1.)
    workers=length(c.workspaces)
    if parallel && workers > 1
        Threads.@threads :static for worker in 1:workers
            for node in worker:workers:length(c.nodal)
                reference_pu3d_patch!(c.workspaces[worker],c,node,nothing)
            end
        end
    else
        for node in eachindex(c.nodal)
            reference_pu3d_patch!(first(c.workspaces),c,node,nothing)
        end
    end
    return pu3d_collect!(c,states)
end

function benchmark_pu_optimization(n=8;repeats=5)
    cv,states,pars,ch,loads,u,K,ed,c=pu_clamped_problem(n)
    old()=reference_estimate_pu!(c,u,states,ed,pars;parallel=true).indicators
    new()=estimate_pu_dwr!(c,u,states,ed,pars).indicators
    sdwr()=el_dict_to_state_vec(estimate_element_error(u,states,cv,ed),states)
    for f in (old,new,sdwr)
        f();f()
    end
    expected=copy(old())
    @assert isapprox(new(),expected;rtol=2e-8,atol=1e-17)
    rows=NamedTuple[]
    for repeat in 1:repeats
        a=@timed sum(old())
        b=@timed sum(new())
        d=@timed sum(sdwr())
        setup=@timed build_pu_dwr_cache(cv,states,ed,ch,pars;boundary_data=loads,previous=c)
        c=setup.value
        rebuilt=@timed sum(new())
        push!(rows,(;old_ms=1000a.time,new_ms=1000b.time,sdwr_ms=1000d.time,
            setup_ms=1000setup.time,rebuilt_ms=1000(setup.time+rebuilt.time),
            old_gc_ms=1000a.gctime,new_gc_ms=1000b.gctime,setup_gc_ms=1000setup.gctime,
            new_bytes=b.bytes,setup_bytes=setup.bytes))
    end
    med(field)=median(getproperty(row,field) for row in rows)
    result=(;cells=length(states.χ_vec),threads=Threads.nthreads(),
        old_ms=med(:old_ms),new_ms=med(:new_ms),sdwr_ms=med(:sdwr_ms),setup_ms=med(:setup_ms),
        rebuilt_ms=med(:rebuilt_ms),speedup=med(:old_ms)/med(:new_ms),
        sdwr_ratio=med(:rebuilt_ms)/med(:sdwr_ms))
    println("OPTIMIZED_PU ",result)
    println("OPTIMIZED_SAMPLES ",rows)
    flush(stdout)
    return result,rows
end
