# Controlled diagnostics; does not modify the production regularization scheme.
# julia --project=. --startup-file=no Tests/taylor_mod_diagnostics.jl operators
# julia --project=. --startup-file=no Tests/taylor_mod_diagnostics.jl frozen
# julia --project=. --startup-file=no Tests/taylor_mod_diagnostics.jl optimize [steps=100] [level=3]
include("regularization_benchmark.jl")

const DIAGNOSTIC_OUT = joinpath(@__DIR__, "..", "Results", "taylor_mod_diagnostics")
mkpath(DIAGNOSTIC_OUT)

function write_rows(name, rows)
    open(joinpath(DIAGNOSTIC_OUT, name), "w") do io
        println(io, join(string.(keys(first(rows))), ','))
        for row in rows
            println(io, join(values(row), ','))
        end
    end
end

function diagnostic_operator(cv, states, pars, scheme)
    beta = 2pars.β0 .* states.h_vec.^2
    if scheme == :taylor
        return compute_taylor_operator_mat(cv, states, pars)
    elseif scheme == :taylor_small_ridge
        return compute_taylor_operator_mat(cv, states, pars; ridge=1e-8)
    elseif scheme == :taylor_fixed_beta
        return compute_taylor_operator_mat(cv, states, pars; beta=fill(minimum(beta),length(beta)))
    elseif scheme in (:strong_physical, :strong_rescaled)
        ns, ghosts = create_neigh_list(states, cv)
        L = compute_laplace_operator_mat(cv.mesh.topo, ns, ghosts, states, scheme == :strong_rescaled)
        return spdiagm(0=>beta) * L
    elseif scheme == :diamond
        return compute_flux_operator_mat(cv, states, pars)
    elseif scheme == :none
        return spzeros(length(beta), length(beta))
    end
    error("Unknown diagnostic scheme: $scheme")
end

mod_value(chi, volumes, lower) = 4dot(volumes, (chi .- lower).*(1 .- chi))/sum(volumes)

function operator_metrics(label, scheme, R, states)
    v = states.area_vec
    n = length(v)
    i,j,z = findnz(R)
    off = z[i .!= j]
    return (fixture=label, scheme=scheme, cells=n,
        constant_residual=norm(R*ones(n),Inf),
        conservation_residual=norm(v'*R,Inf),
        min_offdiagonal=isempty(off) ? 0. : minimum(off),
        negative_offdiagonals=count(<(-1e-10),off),
        max_abs_entry=norm(R,Inf))
end

function operator_diagnostics()
    metrics = NamedTuple[]
    profiles = NamedTuple[]
    for fixture in (:uniform_fine, :graded_band)
        # Identical finest cells around x=1.5; graded mesh coarsens the outer bulk.
        mesh = create_rectangular_mesh(12,4,4,3.,1.,1.,StandardEl{1})
        cv = CellValues{3}(mesh)
        states = DesignVarInfo{3}(cv,0.3)
        for el in collect(RootIterator{4}(mesh.topo))
            x = states.x_vec[get_state_id(states,el.id)][1]
            if fixture == :uniform_fine || 1.0 < x < 2.0
                _refine!(el,mesh.topo)
            end
        end
        mesh = Mesh(mesh.topo,StandardEl{1}())
        cv = CellValues{3}(mesh)
        states = DesignVarInfo{3}(cv,0.3)
        for scheme in (:taylor,:strong_physical,:strong_rescaled,:taylor_small_ridge,:taylor_fixed_beta,:diamond)
            R = diagnostic_operator(cv,states,(β0=1.,),scheme)
            push!(metrics,operator_metrics(fixture,scheme,R,states))
            for center in (1.0,1.25,1.5), width in (0.08,0.2)
                v = states.area_vec
                lower = 0.001
                original = [lower+(1-lower)*(1+tanh((x[1]-center)/width))/2 for x in states.x_vec]
                chi = copy(original)
                target = dot(v,chi)/sum(v)
                before = mod_value(chi,v,lower)
                force = R*chi
                projected_force = force .- dot(v,force)/sum(v)
                instantaneous_mod_rate = 4dot(v,(1+lower .- 2chi).*projected_force)/sum(v)
                for _ in 1:8
                    trial = chi + R*chi/(8*15)
                    project_density_volume!(chi,trial,v,target,lower)
                end
                push!(profiles,(fixture=fixture,scheme=scheme,center=center,width=width,
                    mod_before=before,mod_after=mod_value(chi,v,lower),
                    delta_mod=mod_value(chi,v,lower)-before,
                    projected_mod_rate=instantaneous_mod_rate,
                    raw_mean_force=dot(v,force)/sum(v)))
            end
        end
    end
    write_rows("operator_metrics.csv",metrics)
    write_rows("planar_smoothing.csv",profiles)
    foreach(println,metrics)
    println("Planar smoothing diagnostics saved to ",DIAGNOSTIC_OUT)
end

function frozen_diagnostics()
    # Representative MBB adaptive state retained by the earlier integration test;
    # this is not a snapshot from the 20260930 cluster batch.
    cv,states,pars = deserialize(joinpath(@__DIR__,"..","Results","bad_diamond_stencil.bin"))
    println("Loaded representative MBB state with ",length(states.χ_vec)," cells")
    flush(stdout)
    # The serialized fixture already contains boundary sets from its old handler.
    for sets in (cv.mesh.node_sets,cv.mesh.edge_sets,cv.mesh.face_sets,cv.mesh.volume_sets)
        empty!(sets)
    end
    ch = create_constraint_handler(cv,:MBB_sym)
    u,K,ed = compute_displacement(cv,ch,states,x->SA[0.,0.,0.],pars)
    initial = copy(states.χ_vec)
    v = states.area_vec
    rows = NamedTuple[]
    metrics = NamedTuple[]
    for scheme in (:none,:taylor,:strong_physical,:strong_rescaled,:taylor_small_ridge,:taylor_fixed_beta,:diamond)
        R = diagnostic_operator(cv,states,pars,scheme)
        push!(metrics,operator_metrics(:saved_mbb,scheme,R,states))
        copyto!(states.χ_vec,initial)
        before = measure_of_nondiscreteness(states,pars)
        force = R*initial
        changed = state_update!(states,cv,pars,R,u,ed;beta_in_operator=true)
        push!(rows,(scheme=scheme,cells=length(v),mod_before=before,
            mod_after=measure_of_nondiscreteness(states,pars),
            delta_mod=measure_of_nondiscreteness(states,pars)-before,
            mean_density=dot(v,states.χ_vec)/sum(v),
            raw_regularization_mean=dot(v,force)/sum(v),
            change_norm=norm(changed),
            clipped_cells=count(x->x==pars.χmin || x==1.,states.χ_vec)))
        println(last(rows)); flush(stdout)
    end
    write_rows("frozen_state.csv",rows)
    write_rows("frozen_operator_metrics.csv",metrics)
end

function optimization_diagnostics(steps,level)
    rows = NamedTuple[]
    for scheme in (:taylor,:strong_physical,:strong_rescaled), adaptive in (false,true)
        # On the initial uniform hex mesh these operators coincide. One fixed
        # baseline suffices; verify that equivalence below rather than repeating it.
        !adaptive && scheme != :taylor && continue
        # Match the MBB hex geometry and parameters in the production driver.
        mesh = create_rectangular_mesh(12,2,4,3.,0.5,1.,StandardEl{1})
        for _ in 2:level, el in RootIterator{4}(mesh.topo)
            _refine!(el,mesh.topo)
        end
        mesh = Mesh(mesh.topo,StandardEl{1}())
        mesh,protected = refine_sets(mesh,get_sets_to_refine(:MBB_sym),level)
        cv = CellValues{3}(mesh)
        states = DesignVarInfo{3}(cv,0.3)
        lam,mu = E_ν_to_lame(210.e3,0.33)
        pars = SimPars(Helmholtz{3,3}(Ψlin_totopt,(lam,mu,1.)),lam,mu,1e-3,15.,1.,0.3)
        if !adaptive
            baseline = diagnostic_operator(cv,states,pars,:taylor)
            for variant in (:strong_physical,:strong_rescaled)
                difference = norm(baseline-diagnostic_operator(cv,states,pars,variant),Inf)
                println((uniform_operator=variant,difference_norm=difference))
                @assert difference < 1e-8
            end
        end
        ch = nothing
        for step in 1:steps
            if step == 1 || adaptive
                ch = create_constraint_handler(cv,:MBB_sym)
            end
            R = diagnostic_operator(cv,states,pars,scheme)
            u,K,ed = compute_displacement(cv,ch,states,x->SA[0.,0.,0.],pars)
            changed = state_update!(states,cv,pars,R,u,ed;beta_in_operator=true)
            mod_before_transfer = measure_of_nondiscreteness(states,pars)
            mass_before_transfer = dot(states.area_vec,states.χ_vec)/sum(states.area_vec)
            cells_before = length(states.χ_vec)
            energy = dot(u,K*u)/2
            if adaptive && step < steps
                err = estimate_element_error(u,states,cv,ed)
                ref,coarse = mark_elements_for_adaption(cv,err,states,changed,level,protected,true)
                clear_up_topo!(cv.mesh.topo)
                cv = adapt_mesh(cv,coarse,ref)
                update_states_after_mesh_adaption!(states,cv,ed,ref,coarse)
            end
            push!(rows,(scheme=scheme,adaptive=adaptive,step=step,cells=cells_before,
                mod=mod_before_transfer,strain_energy=energy,
                transfer_delta_mod=measure_of_nondiscreteness(states,pars)-mod_before_transfer,
                transfer_delta_mass=dot(states.area_vec,states.χ_vec)/sum(states.area_vec)-mass_before_transfer))
            @assert abs(dot(states.area_vec,states.χ_vec)/sum(states.area_vec)-0.3) < 1e-7
            if step % 10 == 0 || step == 1
                println(last(rows)); flush(stdout)
            end
        end
        # Preserve completed comparisons if a later run is interrupted.
        write_rows("optimization_r$(level)_s$(steps).csv",rows)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    mode = isempty(ARGS) ? "operators" : ARGS[1]
    if mode == "operators"
        operator_diagnostics()
    elseif mode == "frozen"
        frozen_diagnostics()
    elseif mode == "optimize"
        optimization_diagnostics(length(ARGS)>1 ? parse(Int,ARGS[2]) : 100,
            length(ARGS)>2 ? parse(Int,ARGS[3]) : 3)
    else
        error("Expected operators, frozen or optimize")
    end
end
