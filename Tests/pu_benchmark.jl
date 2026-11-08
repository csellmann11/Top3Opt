# Includes setup cost because adaptive meshes may change after every estimate.
include("pu_dwr.jl")

function benchmark_pu(n;repeats=3)
    cv,states,pars,ch,loads,u,K,ed,c = pu_clamped_problem(n)
    estimate_pu_dwr!(c,u,states,ed,pars)
    samples = NamedTuple[]
    for _ in 1:repeats
        setup = @timed build_pu_dwr_cache(cv,states,ed,ch,pars;boundary_data=loads,previous=c)
        c = setup.value
        # One warm evaluation sizes scratch, which is retained on later rebuilds.
        trial = @timed estimate_pu_dwr!(c,u,states,ed,pars)
        push!(samples,(setup_seconds=setup.time,estimate_seconds=trial.time,
            total_seconds=setup.time+trial.time,setup_gc_seconds=setup.gctime,
            estimate_gc_seconds=trial.gctime,setup_bytes=setup.bytes,estimate_bytes=trial.bytes))
    end
    best = samples[argmin(getproperty.(samples,:total_seconds))]
    result = (;cells=length(states.χ_vec),threads=Threads.nthreads(),best...,
        setup_fraction=best.setup_seconds/best.total_seconds,
        retained_cache_bytes=Base.summarysize(c),retained_scratch_bytes=Base.summarysize(c.workspaces),
        statistics=pu3d_cache_statistics(c))
    println(result)
    return result
end

function pu_benchmark_boundary(cv)
    ch,loads = ConstraintHandler{3}(cv.mesh),PU3DBoundaryData()
    add_face_set!(cv.mesh,"clamp",x -> x[1] ≈ 0)
    add_face_set!(cv.mesh,"traction",x -> x[1] ≈ 1)
    pu3d_add_dirichlet!(loads,ch,cv.dh,cv.facedata_col,"clamp",x -> SA[0.,0.,0.])
    pu3d_add_neumann!(loads,ch,cv.dh,cv.facedata_col,"traction",x -> SA[0.,0.,-1.])
    return ch,loads
end

"""Benchmark actual alternating refinement/coarsening, preserving worker scratch.

`mesh_seconds` includes topology cleanup/adaptation, CellValues rebuild and
conservative design transfer. `mechanics_seconds` includes new boundary data,
assembly and the CPU direct equilibrium solve. PU setup and evaluation are
reported separately, and `total_seconds` includes all four stages. The first
`warmup` cycles exercise both mutation paths but are excluded from summaries.
"""
function benchmark_pu_remeshing(n=4;cycles=6,warmup=2)
    cycles > warmup >= 2 || throw(ArgumentError("Need at least two warmup cycles and one measured cycle"))
    cv,states,pars,ch,loads,u,K,ed,c = pu_clamped_problem(n)
    estimate_pu_dwr!(c,u,states,ed,pars)
    target = argmin([norm(x-SA[0.5,0.5,0.5]) for x in states.x_vec])
    parent_id = get_el_id(states,target)
    transfer = DensityTransferWorkspace()
    samples = NamedTuple[]
    for cycle in 1:cycles
        refine = isodd(cycle)
        ref_marker = falses(length(get_volumes(cv.mesh.topo)))
        coarse_marker = similar(ref_marker); fill!(coarse_marker,false)
        if refine
            ref_marker[parent_id] = true
        else
            coarse_marker[get_volumes(cv.mesh.topo)[parent_id].childs] .= true
        end
        mesh_trial = @timed begin
            clear_up_topo!(cv.mesh.topo)
            nextcv = adapt_mesh(cv,coarse_marker,ref_marker)
            update_states_after_mesh_adaption!(states,nextcv,ref_marker,coarse_marker;workspace=transfer)
            nextcv
        end
        cv = mesh_trial.value
        mechanics_trial = @timed begin
            nextch,nextloads = pu_benchmark_boundary(cv)
            nextu,nextK,nexted = compute_displacement(cv,nextch,states,ZeroBodyForce(),pars)
            (nextch,nextloads,nextu,nextK,nexted)
        end
        ch,loads,u,K,ed = mechanics_trial.value
        setup = @timed build_pu_dwr_cache(cv,states,ed,ch,pars;boundary_data=loads,previous=c)
        c = setup.value
        trial = @timed estimate_pu_dwr!(c,u,states,ed,pars)
        row = (;cycle,operation=refine ? :refine : :coarsen,cells=length(states.χ_vec),
            mesh_seconds=mesh_trial.time,mechanics_seconds=mechanics_trial.time,
            setup_seconds=setup.time,estimate_seconds=trial.time,
            mesh_gc_seconds=mesh_trial.gctime,mechanics_gc_seconds=mechanics_trial.gctime,
            setup_gc_seconds=setup.gctime,estimate_gc_seconds=trial.gctime,
            pu_seconds=setup.time+trial.time,
            total_seconds=mesh_trial.time+mechanics_trial.time+setup.time+trial.time,
            setup_fraction=setup.time/(setup.time+trial.time),
            mesh_bytes=mesh_trial.bytes,mechanics_bytes=mechanics_trial.bytes,
            setup_bytes=setup.bytes,estimate_bytes=trial.bytes,
            retained_cache_bytes=Base.summarysize(c),retained_scratch_bytes=Base.summarysize(c.workspaces))
        cycle > warmup && push!(samples,row)
        println((warmup=cycle<=warmup,row...))
    end
    summary = (;initial_cells=n^3,threads=Threads.nthreads(),measured_cycles=length(samples),
        mean_mesh_seconds=mean(x.mesh_seconds for x in samples),
        mean_mechanics_seconds=mean(x.mechanics_seconds for x in samples),
        mean_setup_seconds=mean(x.setup_seconds for x in samples),
        mean_estimate_seconds=mean(x.estimate_seconds for x in samples),
        mean_setup_gc_seconds=mean(x.setup_gc_seconds for x in samples),
        mean_estimate_gc_seconds=mean(x.estimate_gc_seconds for x in samples),
        mean_total_seconds=mean(x.total_seconds for x in samples),
        setup_fraction=sum(x.setup_seconds for x in samples)/sum(x.pu_seconds for x in samples),
        peak_retained_cache_bytes=maximum(x.retained_cache_bytes for x in samples),
        peak_retained_scratch_bytes=maximum(x.retained_scratch_bytes for x in samples))
    println(summary)
    return summary,samples
end

if abspath(PROGRAM_FILE) == @__FILE__
    for n in (2,3,4)
        benchmark_pu(n)
    end
end
