using Ju3VEM
using Ju3VEM.FixedSizeArrays
using StaticArrays
using LinearAlgebra
using SparseArrays
using Chairmarks
import Ju3VEM.FR as FR
using Statistics
using Random
using Bumper
using TimerOutputs
using Dates

const to = TimerOutput()

struct SimParameter{H<:Helmholtz}
    mat_law::H
    λ::Float64
    μ::Float64 
    
    χmin::Float64
    η0::Float64 
    β0::Float64
    ρ_init::Float64
    h_min::Float64
end

@inline function Ψlin_totopt(∇u::M,λ,μ,χ) where {M<:AbstractMatrix}
    ε = 1/2*(∇u + ∇u')
    W = λ/2 * tr(ε)^2 + μ*tr(ε*ε)
    return W*χ[1]^3
end


include("mat_states.jl")
include("laplace_operator.jl")
include("compute_displacement.jl")
include("bisection.jl")
include("mesh_processing_utils.jl")
include("neighbor_search.jl")
include("refinement_utils/estiamte_element_error.jl")
include("postprocessing/sim_data.jl")
const K = 1
const U = 3

MAX_OPT_STEPS = 1
MAX_REF_LEVEL = 2
MeshType = :Hexahedra

function MBB_rhs(x)
    SA[0.0,0.0,0.0]
end



function run_optimization(
    cv::CellValues{U},
    states::TopStates{U},
    ch::ConstraintHandler{U},
    sim_pars::SimParameter{H};
    vtk_folder_name::String,
    MAX_OPT_STEPS::Int = 200,
    MAX_REF_LEVEL::Int = 3,
    tolerance::Float64 = 1e-5,
    n_conv_until_stop::Int = 2,
    write_vtk_every_n_steps::Int = 30,
    do_adaptivity::Bool = true,
 ) where {U,H<:Helmholtz}

    n_conv_count = 0
    sim_results  = SimulationResults(MAX_REF_LEVEL,MAX_OPT_STEPS,sim_pars)
    if isdir(vtk_folder_name)
        println("Removing existing vtk folder: $vtk_folder_name")
        rm(vtk_folder_name,recursive=true)
    end
    mkdir(vtk_folder_name)

    @timeit to "create_neighbor_list" el_neighs, face_to_vols, edge_to_vols = create_neighbor_list(cv);
    @timeit to "compute_laplace_operator_mat" laplace_operator = compute_laplace_operator_mat(
                  cv.mesh.topo,el_neighs,face_to_vols,states,sim_pars)

    Psi0 = 0.0; Psi_step0 = 0.0; u = Float64[]

    for optimization_step in 1:MAX_OPT_STEPS
        @timeit to "compute_displacement" u,k_global,eldata_col = compute_displacement(cv,ch,states,MBB_rhs,sim_pars)
        @timeit to "state_update" state_changed = state_update!(
            states,sim_pars,laplace_operator,u,eldata_col)

        Psi = 1/2 * u' * k_global * u
        if optimization_step == 1
            Psi_step0 = Psi
            Psi0 = Psi
        end
        ΔPsi_rel = (Psi - Psi0) / Psi0
        println("Optimization step: $optimization_step, Relative strain energy change: $ΔPsi_rel")
        improvement = round(Psi_step0/Psi * 100,sigdigits=2)
        println("Improvement: $improvement %")
        n_states = length(states.χ_vec) 
        n_dofs   = length(u)  
        mod      = measure_of_nondiscreteness(states,sim_pars)
        update_sim_data!(sim_results,mod,Psi,n_states,n_dofs)

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

        if optimization_step % write_vtk_every_n_steps == 0
            println("Writing vtk file for optimization step: $optimization_step")
            full_name = joinpath(vtk_folder_name, "temp_res_$(optimization_step)")
            write_vtk(cv.mesh.topo,full_name,cv.dh,u;cell_data = states.χ_vec)
        end

        
        @timeit to "adaptivity" begin
            !do_adaptivity && continue
            @timeit to "estimate_element_error" element_error = estimate_element_error(u,eldata_col)
            ref_marker, coarse_marker = mark_elements_for_adaption(cv,
                            element_error,states,state_changed,MAX_REF_LEVEL)

            @timeit to "mesh_clearing" clear_up_mesh(cv.mesh.topo,face_to_vols,edge_to_vols)
            @timeit to "adapt_mesh" cv = adapt_mesh(cv,coarse_marker,ref_marker)
            @timeit to "create_constraint_handler" ch = create_constraint_handler(cv,:MBB_sym);
            @timeit to "update_states_after_mesh_adaption" states = update_states_after_mesh_adaption!(states,cv,ref_marker,coarse_marker)

            @timeit to "create_neighbor_list" el_neighs, face_to_vols, edge_to_vols = create_neighbor_list(cv);

            @timeit to "compute_laplace_operator_mat" laplace_operator = compute_laplace_operator_mat(
                  cv.mesh.topo,el_neighs,face_to_vols,states,sim_pars)
        end
    end

    full_name = joinpath(vtk_folder_name, "final_res")
    write_vtk(cv.mesh.topo,full_name,cv.dh,u;cell_data = states.χ_vec)

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

n = div(4,2)*2
l_beam = 3.0

if MeshType == :Hexahedra
    mesh = create_rectangular_mesh(
        3n,div(n,2),n,
        l_beam,0.5,1.0,StandardEl{K}
    );
elseif MeshType == :Voronoi
    mesh2d = create_voronoi_mesh(
        (0.0,0.0),
        (l_beam,0.5),
        3n,div(n,2),StandardEl{K}
        )
    mesh = extrude_to_3d(n,mesh2d,1.0);
end

h_cell = find_maximal_cell_diameter(mesh.topo)
h_cell_min = h_cell * (1/2)^(MAX_REF_LEVEL-1)
for level in 1:(MAX_REF_LEVEL-1)
    for element in RootIterator{4}(mesh.topo)
        _refine!(element,mesh.topo)
    end
end



mesh = Mesh(mesh.topo,StandardEl{1}())
n_els = count(is_active_root,get_volumes(mesh.topo))
println("Number of elements: $n_els")
n_nodes = count(is_active,get_nodes(mesh.topo))
println("Number of nodes: $n_nodes")



E = 210.e03; ν = 0.33
λ,μ = E_ν_to_lame(E,ν)
χ = 0.3
mat_law  = Helmholtz{3,3}(Ψlin_totopt,(λ,μ,χ))
mat_pars = (λ,μ)

χmin = 1e-03
η0   = 15.0 
β0   = 2*h_cell_min^2 
ρ_init = 0.3 
sim_pars = SimParameter(mat_law,λ,μ,χmin,η0,1.0,ρ_init,h_cell_min)

@time cv = CellValues{U}(mesh);

states = TopStates{U}(cv,ρ_init)

ch = create_constraint_handler(cv,:MBB_sym);

println("="^100)
println("Starting optimization")
println("="^100)
optimization_time = @elapsed sim_results = run_optimization(
    cv,
    states,
    ch,
    sim_pars,
    vtk_folder_name = "Results/vtk/Adaptive_Runs/MBB_$(n)_$(MAX_REF_LEVEL)_$(MeshType)",
    MAX_OPT_STEPS = MAX_OPT_STEPS,
    MAX_REF_LEVEL = MAX_REF_LEVEL,
    write_vtk_every_n_steps = 50,
    do_adaptivity = true
)

println("Optimization time: $optimization_time")
show(to)


export_sim_data_for_latex(sim_results, "Results/SimData/MBB_$(n)_$(MAX_REF_LEVEL)_$(MeshType).csv")

