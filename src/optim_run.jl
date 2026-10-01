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
    laplace_rescale::Bool = true,
    flux_scheme::Symbol = :diamond,
    update_mode::Symbol = :explicit,
    tolerance::Float64 = 1e-5,
    n_conv_until_stop::Int = 2,
    take_snapshots_at::AbstractVector{Int} = 1:30:MAX_OPT_STEPS,
    do_adaptivity::Bool = true,
    b_case::Symbol = :MBB_sym,
 ) where {D,H<:Helmholtz,F<:Function}

    flux_scheme in (:strong, :tpfa, :diamond, :taylor) ||
        throw(ArgumentError("flux_scheme must be :strong, :tpfa, :diamond, or :taylor"))
    update_mode in (:explicit, :implicit) ||
        throw(ArgumentError("update_mode must be :explicit or :implicit"))
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




    Psi0 = 0.0; Psi_step0 = 0.0; u = Float64[]; state_changed = Float64[]

    t_now = time()

    ch = nothing; state_neights_col = nothing; b_face_id_to_state_id = nothing; laplace_operator = nothing
    implicit_cache = nothing
    for optimization_step in 1:MAX_OPT_STEPS

        flush(stdout)
        rebuild_operator = optimization_step == 1 || do_adaptivity
        @timeit to "adaptivity" if rebuild_operator

            @timeit to "create_constraint_handler" ch = create_constraint_handler(cv,b_case);
            if flux_scheme == :strong
                @timeit to "create_neighbor_list" state_neights_col, b_face_id_to_state_id = create_neigh_list(states,cv)
                @timeit to "compute_laplace_operator_mat" laplace_operator = compute_laplace_operator_mat(
                    cv.mesh.topo,state_neights_col,b_face_id_to_state_id,states,laplace_rescale)
            elseif flux_scheme == :taylor
                @timeit to "compute_laplace_operator_mat" laplace_operator = compute_taylor_operator_mat(
                    cv,states,sim_pars)
            else
                @timeit to "compute_laplace_operator_mat" laplace_operator = compute_flux_operator_mat(
                    cv,states,sim_pars; scheme=flux_scheme)
            end
        end

        @timeit to "compute_displacement" u,k_global,eldata_col = compute_displacement(cv,ch,states,rhs_fun,sim_pars)
        @timeit to "state_update" begin
            # Include solver setup in the exported density-update timing. Rebuild
            # with each new operator, including meshes with the same cell count.
            if update_mode == :implicit && rebuild_operator
                @timeit to "prepare_implicit_density" implicit_cache = DensityImplicitCache(
                    laplace_operator,sim_pars.η0,sim_pars.β0,states.h_vec;
                    beta_in_operator=flux_scheme != :strong)
            end
            state_changed = state_update!(
                states,cv,sim_pars,laplace_operator,u,eldata_col;
                beta_in_operator=flux_scheme != :strong,update_mode,implicit_cache)
        end



        Psi = 1/2 * u' * k_global * u
        if optimization_step == 1
            Psi_step0 = Psi
            Psi0 = Psi
        end
        ΔPsi_rel = (Psi - Psi0) / Psi0
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
    

        if abs(ΔPsi_rel) < tolerance 
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
            el_error_v = el_dict_to_state_vec(estimate_element_error(u,states,cv,eldata_col),states)
            @timeit to "vtk_export" write_vtu_file(cv,eldata_col,full_name,u;cell_data_col = (states.χ_vec,el_error_v,state_changed))
        end

        
        @timeit to "adaptivity" begin
            !do_adaptivity && continue
            @timeit to "estimate_element_error" element_error = estimate_element_error(u,states,cv,eldata_col)
            ref_marker, coarse_marker = mark_elements_for_adaption(cv, 
                            element_error,states,state_changed,
                            MAX_REF_LEVEL,no_coarsening_marker,density_marking)


            @timeit to "mesh_clearing" clear_up_topo!(cv.mesh.topo)
            @timeit to "adapt_mesh" cv = adapt_mesh(cv,coarse_marker,ref_marker)
            @timeit to "update_states_after_mesh_adaption" states = update_states_after_mesh_adaption!(states,cv,eldata_col,ref_marker,coarse_marker)
        end
    end

    full_name = joinpath(vtk_folder_name, "final_res")
    el_error_v = el_dict_to_state_vec(estimate_element_error(u,states,cv,eldata_col),states)
    write_vtu_file(cv,eldata_col,full_name,u;cell_data_col = (states.χ_vec,el_error_v,state_changed))

    # sim_results.simulation_times.solve_time = TimerOutputs.time(to["compute_displacement"]["solver"])/(1e09)
    # sim_results.simulation_times.assembly_time = TimerOutputs.time(to["compute_displacement"]["assembly"])/(1e09)
    # sim_results.simulation_times.state_update_time = TimerOutputs.time(to["state_update"])/(1e09)
    # try
    #     sim_results.simulation_times.adaptivity_time = TimerOutputs.time(to["adaptivity"])/(1e09)
    # catch
    #     sim_results.simulation_times.adaptivity_time = 0.0
    # end
    sim_results.simulation_times = get_simulation_times(to)

    return sim_results
end
