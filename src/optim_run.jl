function el_dict_to_state_vec(d::Dict{Int},states::DesignVarInfo{D}) where D 
    e2s = states.el_id_to_state_id
    vec = zeros(length(states.χ_vec))
    for (el_id,el_data) in d
        vec[e2s[el_id]] = el_data
    end
    vec
end



function run_optimization(
    mesh::Mesh{D},
    rhs_fun::F,
    sim_pars::SimPars{H};
    vtk_folder_name::String,
    MAX_OPT_STEPS::Int = 200,
    MAX_REF_LEVEL::Int = 3,
    density_marking::Bool = true,
    error_estimator::Symbol = :sdwr,
    pu_marking_exponent::Float64 = 1.0,
    flux_scheme::Symbol = :tpfa,
    regularization::Symbol = :mean,
    projection::Bool = false,
    projection_options::DensityProjectionOptions = DensityProjectionOptions(enabled=projection),
    filter_radius::Float64 = 0.0,
    density_substeps::Int = 0,
    filter_tolerance::Float64 = 1e-12,
    density_activity_tolerance::Float64 = 1e-3,
    update_mode::Symbol = :explicit,
    tolerance::Float64 = 1e-5,
    n_conv_until_stop::Int = 2,
    take_snapshots_at::AbstractVector{Int} = 1:30:MAX_OPT_STEPS,
    do_adaptivity::Bool = true,
    b_case::Symbol = :MBB_sym,
 ) where {D,H<:Helmholtz,F<:Function}

    error_estimator in (:sdwr,:pu_dwr) || throw(ArgumentError("error_estimator must be :sdwr or :pu_dwr"))
    isfinite(pu_marking_exponent) && pu_marking_exponent >= 0 ||
        throw(ArgumentError("pu_marking_exponent must be finite and nonnegative"))
    flux_scheme == :tpfa || throw(ArgumentError("Only flux_scheme=:tpfa is supported"))
    update_mode in (:explicit, :implicit) ||
        throw(ArgumentError("update_mode must be :explicit or :implicit"))
    options = DensityOptions(regularization=regularization,
        integrator=update_mode == :implicit ? :semi_implicit : :explicit,
        substeps=density_substeps,filter_tolerance=filter_tolerance,
        activity_tolerance=density_activity_tolerance)
    validate_density_options(options,sim_pars.η0,1)
    validate_density_projection_options(projection_options)
    projection == projection_options.enabled || throw(ArgumentError(
        "projection and projection_options.enabled must agree"))
    isfinite(sim_pars.β0) && sim_pars.β0 >= 0 || throw(ArgumentError("Invalid beta0"))
    isfinite(filter_radius) && filter_radius >= 0 || throw(ArgumentError("Invalid filter radius"))
    0 <= sim_pars.χmin <= sim_pars.ρ_init <= 1 && sim_pars.χmin < 1 ||
        throw(ArgumentError("Infeasible initial volume or lower density bound"))
    MAX_OPT_STEPS > 0 || throw(ArgumentError("MAX_OPT_STEPS must be positive"))
    n_conv_count = 0
    sim_results  = SimulationResults(MAX_REF_LEVEL,
              MAX_OPT_STEPS,sim_pars,Val{D}())
    eldata_col = Dict{Int,ElData{D}}()
    if isdir(vtk_folder_name)
        println("Removing existing vtk folder: $vtk_folder_name")
        rm(vtk_folder_name,recursive=true)
    end
    mkpath(vtk_folder_name)

    optimization_finished = false


    mesh, no_coarsening_marker = refine_sets(mesh, 
                  get_sets_to_refine(b_case), MAX_REF_LEVEL)


    cv = CellValues{U}(mesh)
    validate_vem_projectors(cv)

    
    states = DesignVarInfo{U}(cv, sim_pars.ρ_init)




    Psi0 = 0.0; Psi_step0 = 0.0; u = Float64[]; state_changed = falses(0)

    t_now = time()

    # Radius zero gives Helmholtz the penalty coefficient r²_i = 2*beta0*h_i^2;
    # a positive value overrides it with a uniform physical radius.
    radius = filter_radius
    density_projection = DensityProjection(projection_options,states.area_vec;rho_min=sim_pars.χmin)
    transfer = DensityTransferWorkspace()
    workspace = DensityMechanicalWorkspace(length(states.χ_vec))
    ch = nothing
    density_cache = nothing
    error_cache = nothing
    error_rebuild = true
    boundary_data = nothing
    rebuild_operator = true
    sim_results.error_estimator = error_estimator
    sim_results.pu_marking_exponent = pu_marking_exponent
    sim_results.regularization = regularization
    sim_results.update_mode = update_mode
    sim_results.filter_radius = regularization == :helmholtz ? radius : 0.0
    for optimization_step in 1:MAX_OPT_STEPS
        flush(stdout)
        if rebuild_operator
            boundary_data = error_estimator == :pu_dwr ? PU3DBoundaryData() : nothing
            @timeit to "adaptivity" ch = create_constraint_handler(cv,b_case;boundary_data)
            @timeit to "state_update" begin
                @timeit to "prepare_density" density_cache = prepare_density_cache(
                    density_cache,cv,states,sim_pars,options,radius)
                projection_rebuild!(density_projection,states.area_vec)
                map_design!(states,density_cache,density_projection,options)
            end
            rebuild_operator = false
        end

        @timeit to "compute_displacement" u,k_global,eldata_col = compute_displacement(cv,ch,states,rhs_fun,sim_pars)
        element_error = nothing
        if optimization_step < MAX_OPT_STEPS && (do_adaptivity || optimization_step in take_snapshots_at)
            @timeit to "adaptivity" begin
                if error_estimator == :pu_dwr
                    if error_rebuild
                        @timeit to "prepare_pu_dwr" error_cache = build_pu_dwr_cache(
                            cv,states,eldata_col,ch,sim_pars;boundary_data,rhs_fun,previous=error_cache)
                        error_rebuild = false
                    end
                    @timeit to "estimate_element_error" estimate = estimate_pu_dwr!(
                        error_cache,u,states,eldata_col,sim_pars;q=pu_marking_exponent)
                    element_error = estimate.indicators
                    push!(sim_results.error_history,(step=optimization_step,signed=estimate.signed,magnitude=estimate.magnitude))
                else
                    @timeit to "estimate_element_error" element_error = el_dict_to_state_vec(
                        estimate_element_error(u,states,cv,eldata_col),states)
                end
            end
        end
        @timeit to "state_update" state_changed = state_update!(
            states,cv,sim_pars,density_cache,u,eldata_col;options,density_projection,workspace)

        Psi = 1/2 * u' * k_global * u
        if optimization_step == 1
            Psi_step0 = Psi
            Psi0 = Psi
        end
        used_beta = density_projection === nothing ? 0.0 : density_projection.beta
        used_threshold = projection_threshold(density_projection)
        candidate_grayness = density_projection === nothing ? measure_of_nondiscreteness(states,sim_pars) : density_projection.grayness
        @timeit to "state_update" advance_projection!(states,density_cache,density_projection,options,Psi;
            allow_advance=optimization_step < MAX_OPT_STEPS)
        push!(sim_results.projection_history,(beta=used_beta,
            next_beta=density_projection === nothing ? 0.0 : density_projection.beta,
            threshold=used_threshold,grayness=candidate_grayness))
        push!(sim_results.density_history,(
            design_volume=dot(states.area_vec,states.design_vec)/sum(states.area_vec),
            physical_volume=dot(states.area_vec,states.χ_vec)/sum(states.area_vec),
            minimum_density=minimum(states.χ_vec),maximum_density=maximum(states.χ_vec)))
        ΔPsi_rel = (Psi - Psi0) / max(abs(Psi0),eps(Float64))
        mod      = measure_of_nondiscreteness(states,sim_pars)
        n_states = length(states.χ_vec) 
        n_dofs   = length(u)  
        update_sim_data!(sim_results,mod,Psi,n_states,n_dofs)
        improvement = round(Psi_step0/Psi * 100,sigdigits=2)

        run_time = round((time() - t_now)/60.0,sigdigits=2)

        println("Optimization step: $optimization_step, Relative strain energy change: $ΔPsi_rel")
        println("Improvement: $improvement %")
        println("number of states: $n_states, number of dofs: $n_dofs")
        println("Run time: $run_time minutes")
        println("Measure of nondiscreteness: $mod")
        println("Peak RSS: $(round(Sys.maxrss() / 2^20, digits = 1)) MiB")
    

        if abs(ΔPsi_rel) < tolerance && projection_converged(density_projection)
            n_conv_count += 1
            if n_conv_count >= n_conv_until_stop && !optimization_finished
                # break
                sim_results.conv_n_iter = optimization_step
                sim_results.sim_times_conv_iter = get_simulation_times(to)
                # break
                optimization_finished = true
            end
        else
            n_conv_count = 0
        end
        Psi0 = Psi

        optimization_step == MAX_OPT_STEPS && break

        if optimization_step in take_snapshots_at
            println("Writing vtk file for optimization step: $optimization_step")
            full_name = joinpath(vtk_folder_name, "temp_res_$(optimization_step)")
            el_error_v = element_error
            @timeit to "vtk_export" write_vtu_file(cv,eldata_col,full_name,u;cell_data_col = (density_cache.activity_reference,el_error_v,state_changed))
        end

        
        @timeit to "adaptivity" begin
            !do_adaptivity && continue
            ref_marker, coarse_marker = mark_elements_for_adaption(cv, 
                            element_error,states,state_changed,
                            MAX_REF_LEVEL,no_coarsening_marker,density_marking,
                            error_estimator == :pu_dwr ? 2.0 : 16.0,
                            error_estimator == :pu_dwr ? 0.125 : 0.25;
                            vertex_scaling=error_estimator == :sdwr)


            if any(ref_marker) || any(coarse_marker)
                @timeit to "mesh_clearing" clear_up_topo!(cv.mesh.topo)
                @timeit to "adapt_mesh" cv = adapt_mesh(cv,coarse_marker,ref_marker)
                @timeit to "update_states_after_mesh_adaption" update_states_after_mesh_adaption!(
                    states,cv,ref_marker,coarse_marker;workspace=transfer)
                rebuild_operator = true
                error_rebuild = true
            end
        end
    end

    # One final equilibrium solve aligns exported displacement with the accepted
    # physical density, including the final projection-continuation change.
    @timeit to "compute_displacement" u,k_global,eldata_col = compute_displacement(cv,ch,states,rhs_fun,sim_pars)
    sim_results.final_strain_energy = dot(u,k_global*u)/2
    full_name = joinpath(vtk_folder_name, "final_res")
    if error_estimator == :pu_dwr
        if error_rebuild
            @timeit to "prepare_pu_dwr" error_cache = build_pu_dwr_cache(
                cv,states,eldata_col,ch,sim_pars;boundary_data,rhs_fun,previous=error_cache)
        end
        @timeit to "estimate_element_error" estimate = estimate_pu_dwr!(
            error_cache,u,states,eldata_col,sim_pars;q=pu_marking_exponent)
        el_error_v = estimate.indicators
        push!(sim_results.error_history,(step=MAX_OPT_STEPS,signed=estimate.signed,magnitude=estimate.magnitude))
    else
        el_error_v = el_dict_to_state_vec(estimate_element_error(u,states,cv,eldata_col),states)
    end
    write_vtu_file(cv,eldata_col,full_name,u;cell_data_col = (states.χ_vec,el_error_v,state_changed))

    sim_results.simulation_times = get_simulation_times(to)

    sim_results.final_design = copy(states.design_vec)
    sim_results.final_density = copy(states.χ_vec)
    sim_results.final_volumes = copy(states.area_vec)
    return sim_results
end
