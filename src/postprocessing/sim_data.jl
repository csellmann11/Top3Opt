using Dates

@kwdef mutable struct SimulationTimes 
    solve_time       ::Float64 = 0.0
    assembly_time    ::Float64 = 0.0
    state_update_time::Float64 = 0.0
    adaptivity_time  ::Float64 = 0.0
    total_time       ::Float64 = 0.0
end


struct SimulationResults{H<:Helmholtz,D}

    sim_data         ::DateTime
    max_ref_level    ::Int
    max_opt_steps    ::Int
    sim_pars         ::SimParameter{H}

    simulation_times ::SimulationTimes

    mod              ::Vector{Float64} 
    strain_energy    ::Vector{Float64}
    number_of_states ::Vector{Int}
    number_of_dofs   ::Vector{Int}  

    el_error_at_snapshots::Vector{Dict{Int,Float64}}
    states_at_snapshots::Vector{TopStates{D}}
    topology_at_snapshots::Vector{Topology{D}}

end


function SimulationResults(
    max_ref_level::Int,
    max_opt_steps::Int,
    sim_pars::SimParameter{H},
    ::Val{D}
) where {H<:Helmholtz,D}

    sim_data            = now()
    mod                 = Float64[]
    strain_energy       = Float64[]
    number_of_states    = Int[]
    number_of_dofs      = Int[]
    el_error_at_snapshots = Vector{Dict{Int,Float64}}()
    states_at_snapshots   = Vector{TopStates{D}}()
    topology_at_snapshots = Vector{Topology{D}}()

    simulation_times = SimulationTimes()

    SimulationResults(sim_data,max_ref_level,
        max_opt_steps,sim_pars,simulation_times,mod,
        strain_energy,number_of_states,number_of_dofs,
        el_error_at_snapshots,states_at_snapshots,topology_at_snapshots)
end

function update_sim_data!(
    sim_data::SimulationResults,
    mod::Float64,
    strain_energy::Float64,
    number_of_states::Int,
    number_of_dofs::Int)

    push!(sim_data.mod,mod)
    push!(sim_data.strain_energy,strain_energy)
    push!(sim_data.number_of_states,number_of_states)
    push!(sim_data.number_of_dofs,number_of_dofs)
end

"""
    export_sim_data_for_latex(sim_data::SimulationResults, filepath::String)

Export simulation results to a CSV file for use with LaTeX/pgfplots.
The CSV file contains columns: step, mod, relative_strain_energy, number_of_states, number_of_dofs
Strain energy is normalized by dividing all values by the first value.
"""
function export_sim_data_for_latex(sim_data::SimulationResults, filepath::String)
    n_steps = length(sim_data.mod)
    
    # Get the first strain energy value for normalization
    first_strain_energy = sim_data.strain_energy[1]
    
    open(filepath, "w") do io
        # Write header
        println(io, "step,mod,relative_strain_energy,number_of_states,number_of_dofs")
        
        # Write data rows
        for i in 1:n_steps
            rel_strain_energy = sim_data.strain_energy[i] / first_strain_energy
            println(io, "$i,$(sim_data.mod[i]),$rel_strain_energy,$(sim_data.number_of_states[i]),$(sim_data.number_of_dofs[i])")
        end
    end
    
    @info "Simulation data exported to $filepath"
    return filepath
end