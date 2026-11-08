# Compare the production state-ordered outputs on identical solved meshes.
# julia --project=. --startup-file=no Tests/dwr_overhead_benchmark.jl
include("pu_benchmark.jl")

sdwr_indicators(u,states,cv,ed) =
    el_dict_to_state_vec(estimate_element_error(u,states,cv,ed),states)

function timed_estimator_batch(f::F,repetitions) where {F}
    trial = @timed begin
        checksum = 0.0
        for _ in 1:repetitions
            checksum += sum(f())
        end
        checksum
    end
    @assert isfinite(trial.value)
    return (;seconds=trial.time/repetitions,bytes=trial.bytes/repetitions,
        gc_seconds=trial.gctime/repetitions)
end

"""Time the same assembly/apply/solve sequence as `compute_displacement`.

The caller measures this whole helper with an outer `@timed`, so the total
retains allocation/GC and instrumentation overhead instead of adding medians.
Boundary-handler construction and mesh changes are outside this fixed-mesh test.
"""
function benchmark_primal_stages(cv,ch,states,rhs_fun,pars)
    assembled = @timed assembly(cv,states,rhs_fun,pars)
    K,rhs,eldata = assembled.value
    boundary = @timed apply!(K,rhs,ch)
    solved = @timed solve_lse(K,rhs,cv,ch)
    return (;u=solved.value,K,eldata,
        assembly_seconds=assembled.time,boundary_seconds=boundary.time,solve_seconds=solved.time,
        assembly_gc_seconds=assembled.gctime,boundary_gc_seconds=boundary.gctime,
        solve_gc_seconds=solved.gctime)
end

function benchmark_dwr_pair(n;repeats=7)
    cv,states,pars,ch,loads,u,K,ed,c = pu_clamped_problem(n)
    sdwr() = sdwr_indicators(u,states,cv,ed)
    pu() = estimate_pu_dwr!(c,u,states,ed,pars).indicators
    # Warm compilation, worker storage and both estimator wrappers.
    timed_estimator_batch(sdwr,2)
    timed_estimator_batch(pu,2)
    build_pu_dwr_cache(cv,states,ed,ch,pars;boundary_data=loads,previous=c)
    warm_primal = @timed benchmark_primal_stages(cv,ch,states,ZeroBodyForce(),pars)
    @assert isapprox(warm_primal.value.u,u;rtol=1e-10,atol=1e-13)
    u,K,ed = warm_primal.value.u,warm_primal.value.K,warm_primal.value.eldata
    samples = NamedTuple[]
    for repeat in 1:repeats
        # Alternate order so the same estimator does not always run first.
        if isodd(repeat)
            s = timed_estimator_batch(sdwr,20)
            p = timed_estimator_batch(pu,1)
        else
            p = timed_estimator_batch(pu,1)
            s = timed_estimator_batch(sdwr,20)
        end
        setup = @timed build_pu_dwr_cache(cv,states,ed,ch,pars;boundary_data=loads,previous=c)
        c = setup.value
        fresh = timed_estimator_batch(pu,1)
        mechanics = @timed benchmark_primal_stages(cv,ch,states,ZeroBodyForce(),pars)
        primal = mechanics.value
        u,K,ed = primal.u,primal.K,primal.eldata
        push!(samples,(;sdwr_ms=1e3s.seconds,pu_cached_ms=1e3p.seconds,
            pu_setup_ms=1e3setup.time,pu_fresh_ms=1e3fresh.seconds,
            pu_total_ms=1e3(setup.time+fresh.seconds),mechanics_ms=1e3mechanics.time,
            assembly_ms=1e3primal.assembly_seconds,boundary_ms=1e3primal.boundary_seconds,
            solve_ms=1e3primal.solve_seconds,
            sdwr_bytes=s.bytes,pu_bytes=p.bytes,pu_setup_bytes=setup.bytes,
            sdwr_gc_ms=1e3s.gc_seconds,pu_gc_ms=1e3p.gc_seconds,
            setup_gc_ms=1e3setup.gctime,mechanics_gc_ms=1e3mechanics.gctime,
            assembly_gc_ms=1e3primal.assembly_gc_seconds,
            boundary_gc_ms=1e3primal.boundary_gc_seconds,solve_gc_ms=1e3primal.solve_gc_seconds))
    end
    med(key) = median(getproperty(row,key) for row in samples)
    summary = (;kind=:fixed_mesh,cells=length(states.χ_vec),threads=Threads.nthreads(),repeats,
        sdwr_ms=med(:sdwr_ms),pu_cached_ms=med(:pu_cached_ms),pu_setup_ms=med(:pu_setup_ms),
        pu_total_ms=med(:pu_total_ms),mechanics_ms=med(:mechanics_ms),
        assembly_ms=med(:assembly_ms),boundary_ms=med(:boundary_ms),solve_ms=med(:solve_ms),
        cached_ratio=med(:pu_cached_ms)/med(:sdwr_ms),
        rebuilt_ratio=med(:pu_total_ms)/med(:sdwr_ms),
        sdwr_vs_mechanics=med(:sdwr_ms)/med(:mechanics_ms),
        pu_vs_mechanics=med(:pu_total_ms)/med(:mechanics_ms),
        sdwr_bytes=med(:sdwr_bytes),pu_bytes=med(:pu_bytes),
        pu_cache_bytes=Base.summarysize(c))
    println("DWR_COMPARISON ",summary)
    println("DWR_SAMPLES ",(;cells=length(states.χ_vec),samples))
    flush(stdout)
    return summary,samples
end

function benchmark_dwr_remeshing(n=8;cycles=8,warmup=2)
    cv,states,pars,ch,loads,u,K,ed,c = pu_clamped_problem(n)
    estimate_pu_dwr!(c,u,states,ed,pars)
    sdwr_indicators(u,states,cv,ed)
    parent = get_el_id(states,argmin([norm(x-SA[0.5,0.5,0.5]) for x in states.x_vec]))
    transfer = DensityTransferWorkspace()
    samples = NamedTuple[]
    for cycle in 1:cycles
        refine = isodd(cycle)
        ref = falses(length(get_volumes(cv.mesh.topo)))
        coarse = copy(ref)
        refine ? (ref[parent]=true) : (coarse[collect(get_volumes(cv.mesh.topo)[parent].childs)] .= true)
        mesh_trial = @timed begin
            clear_up_topo!(cv.mesh.topo)
            nextcv = adapt_mesh(cv,coarse,ref)
            update_states_after_mesh_adaption!(states,nextcv,ref,coarse;workspace=transfer)
            nextcv
        end
        cv = mesh_trial.value
        mechanics = @timed begin
            nextch,nextloads = pu_benchmark_boundary(cv)
            nextu,nextK,nexted = compute_displacement(cv,nextch,states,ZeroBodyForce(),pars)
            (nextch,nextloads,nextu,nextK,nexted)
        end
        ch,loads,u,K,ed = mechanics.value
        s = timed_estimator_batch(() -> sdwr_indicators(u,states,cv,ed),1)
        setup = @timed build_pu_dwr_cache(cv,states,ed,ch,pars;boundary_data=loads,previous=c)
        c = setup.value
        p = timed_estimator_batch(() -> estimate_pu_dwr!(c,u,states,ed,pars).indicators,1)
        row = (;cycle,cells=length(states.χ_vec),mesh_ms=1e3mesh_trial.time,
            mechanics_ms=1e3mechanics.time,sdwr_ms=1e3s.seconds,
            pu_setup_ms=1e3setup.time,pu_evaluate_ms=1e3p.seconds,
            pu_total_ms=1e3(setup.time+p.seconds),
            sdwr_cycle_ms=1e3(mesh_trial.time+mechanics.time+s.seconds),
            pu_cycle_ms=1e3(mesh_trial.time+mechanics.time+setup.time+p.seconds),
            sdwr_gc_ms=1e3s.gc_seconds,pu_setup_gc_ms=1e3setup.gctime,
            pu_evaluate_gc_ms=1e3p.gc_seconds,mechanics_gc_ms=1e3mechanics.gctime)
        cycle > warmup && push!(samples,row)
    end
    med(key) = median(getproperty(row,key) for row in samples)
    summary = (;kind=:remeshing,initial_cells=n^3,threads=Threads.nthreads(),
        measured_cycles=length(samples),sdwr_ms=med(:sdwr_ms),pu_setup_ms=med(:pu_setup_ms),
        pu_evaluate_ms=med(:pu_evaluate_ms),pu_total_ms=med(:pu_total_ms),
        estimator_ratio=med(:pu_total_ms)/med(:sdwr_ms),
        mechanics_ms=med(:mechanics_ms),sdwr_cycle_ms=med(:sdwr_cycle_ms),
        pu_cycle_ms=med(:pu_cycle_ms),cycle_ratio=med(:pu_cycle_ms)/med(:sdwr_cycle_ms))
    println("DWR_COMPARISON ",summary)
    println("DWR_REMESH_SAMPLES ",samples)
    flush(stdout)
    return summary,samples
end

if abspath(PROGRAM_FILE) == @__FILE__
    for n in (4,8,12)
        benchmark_dwr_pair(n)
    end
    benchmark_dwr_remeshing()
end
