function el_dict_to_state_vec(d::Dict{Int},states::TopStates{D}) where D 
    e2s = states.el_id_to_state_id
    vec = zeros(length(states.χ_vec))
    for (el_id,el_data) in d
        vec[e2s[el_id]] = el_data
    end
    vec
end


function run_optimization(
    cv::CellValues{D,U},
    rhs_fun::F,
    states::TopStates{U},
    ch::ConstraintHandler{U},
    no_coarsening_marker::Vector{Bool},
    sim_pars::SimParameter{H};
    vtk_folder_name::String,
    MAX_OPT_STEPS::Int = 200,
    MAX_REF_LEVEL::Int = 3,
    tolerance::Float64 = 1e-5,
    n_conv_until_stop::Int = 2,
    take_snapshots_at::AbstractVector{Int} = 1:30:MAX_OPT_STEPS,
    do_adaptivity::Bool = true,
    b_case::Symbol = :MBB_sym,
 ) where {D,U,H<:Helmholtz,F<:Function}

    n_conv_count = 0
    sim_results  = SimulationResults(MAX_REF_LEVEL,
              MAX_OPT_STEPS,sim_pars,Val{D}())
    eldata_col = Dict{Int,ElData{D}}()
    if isdir(vtk_folder_name)
        println("Removing existing vtk folder: $vtk_folder_name")
        rm(vtk_folder_name,recursive=true)
    end
    mkdir(vtk_folder_name)

    @timeit to "create_neighbor_list" el_neighs, face_to_vols, edge_to_vols = create_neighbor_list(cv);
    @timeit to "compute_laplace_operator_mat" laplace_operator = compute_laplace_operator_mat(
                  cv.mesh.topo,el_neighs,face_to_vols,states,sim_pars)

    Psi0 = 0.0; Psi_step0 = 0.0; u = Float64[]

    # forbid_coarsening = mark_elements_which_are_part_of_sets(cv,ch)
    t_now = 0.0

    for optimization_step in 1:MAX_OPT_STEPS
        @timeit to "compute_displacement" u,k_global,eldata_col = compute_displacement(cv,ch,states,rhs_fun,sim_pars)
        @timeit to "state_update" state_changed = state_update!(
            states,sim_pars,laplace_operator,u,eldata_col)

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

        if time() - t_now > 1e-10
            println("Optimization step: $optimization_step, Relative strain energy change: $ΔPsi_rel")
            improvement = round(Psi_step0/Psi * 100,sigdigits=2)
            println("Improvement: $improvement %")
            
            println("number of states: $n_states, number of dofs: $n_dofs")
            t_now = time()
        end
    

        if abs(ΔPsi_rel) < tolerance 
            n_conv_count += 1
            if n_conv_count >= n_conv_until_stop 
                break
            end
        else
            n_conv_count = 0
        end
        Psi0 = Psi

        optimization_step == MAX_OPT_STEPS && break

        if optimization_step in take_snapshots_at
            println("Writing vtk file for optimization step: $optimization_step")
            full_name = joinpath(vtk_folder_name, "temp_res_$(optimization_step)")
            el_error_v = el_dict_to_state_vec(estimate_element_error(u,eldata_col),states)
            write_vtk(cv.mesh.topo,full_name,cv.dh,u;cell_data_col = (states.χ_vec,el_error_v,state_changed))
            # push!(sim_results.el_error_at_snapshots,estimate_element_error(u,eldata_col))
            # push!(sim_results.topology_nodes_at_snapshots,copy(cv.mesh.topo.nodes))
            # push!(sim_results.topology_connectivity_at_snapshots,deepcopy(cv.mesh.topo.connectivity))
            # push!(sim_results.states_at_snapshots,deepcopy(states))
        end

        
        @timeit to "adaptivity" begin
            !do_adaptivity && continue
            # @timeit to "estimate_element_error" element_error = estimate_element_error(u,eldata_col)
            @timeit to "estimate_element_error" element_error = estimate_element_error(u,states,cv,eldata_col)
            ref_marker, coarse_marker = mark_elements_for_adaption(cv, 
                            element_error,states,state_changed,MAX_REF_LEVEL,no_coarsening_marker)

            @timeit to "mesh_clearing" clear_up_mesh(cv.mesh.topo,face_to_vols,edge_to_vols)
            @timeit to "adapt_mesh" cv = adapt_mesh(cv,coarse_marker,ref_marker)
            @timeit to "create_constraint_handler" ch = create_constraint_handler(cv,b_case);
            @timeit to "update_states_after_mesh_adaption" states = update_states_after_mesh_adaption!(states,cv,eldata_col,ref_marker,coarse_marker)

            @timeit to "create_neighbor_list" el_neighs, face_to_vols, edge_to_vols = create_neighbor_list(cv);

            @timeit to "compute_laplace_operator_mat" laplace_operator = compute_laplace_operator_mat(
                  cv.mesh.topo,el_neighs,face_to_vols,states,sim_pars)
        end
    end

    full_name = joinpath(vtk_folder_name, "final_res")
    el_error_v = el_dict_to_state_vec(estimate_element_error(u,eldata_col),states)
    write_vtk(cv.mesh.topo,full_name,cv.dh,u;cell_data_col = (states.χ_vec,el_error_v))
    # push!(sim_results.el_error_at_snapshots,estimate_element_error(u,eldata_col))
    # push!(sim_results.states_at_snapshots,deepcopy(states))
    # push!(sim_results.topology_nodes_at_snapshots,copy(cv.mesh.topo.nodes))
    # push!(sim_results.topology_connectivity_at_snapshots,deepcopy(cv.mesh.topo.connectivity))

    sim_results.simulation_times.solve_time = TimerOutputs.time(to["compute_displacement"]["solver"])/(1e09)
    sim_results.simulation_times.assembly_time = TimerOutputs.time(to["compute_displacement"]["assembly"])/(1e09)
    sim_results.simulation_times.state_update_time = TimerOutputs.time(to["state_update"])/(1e09)
    try
        sim_results.simulation_times.adaptivity_time = TimerOutputs.time(to["adaptivity"])/(1e09)
    catch
        sim_results.simulation_times.adaptivity_time = 0.0
    end

    return sim_results
end