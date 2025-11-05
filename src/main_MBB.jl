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
using Bumper
using JLD2
using Infiltrator

const to = TimerOutput()

include("general_utils.jl")
args = parse_commandline()
include("mat_states.jl")
include("laplace_operator.jl")
include("compute_displacement.jl")
include("bisection.jl")
include("mesh_processing_utils.jl")
include("neighbor_search.jl")
include("refinement_utils/estiamte_element_error.jl")
include("postprocessing/sim_data.jl")
include("optim_run.jl")
const K = 1
const U = 3

MAX_OPT_STEPS = args["max_opt_steps"]
MAX_REF_LEVEL = args["max_ref_level"]
MeshType = args["mesh_type"]
do_adaptivity = args["do_adaptivity"]
do_adaptivity_at_the_start = args["do_adaptivity_at_the_start"]

println("Optimization level: ", Base.JLOptions().opt_level)
println("Inline: ", Base.JLOptions().can_inline)
println("Check bounds: ", Base.JLOptions().check_bounds)
println("Julia version: ", VERSION)
println("Julia executable: ", Base.julia_cmd())
println("BLAS vendor: ", BLAS.vendor())
println("BLAS config: ", BLAS.get_config())


println("Running simulation with:")
println("MAX_OPT_STEPS: $MAX_OPT_STEPS")
println("MAX_REF_LEVEL: $MAX_REF_LEVEL")
println("MeshType: $MeshType")
println("do_adaptivity: $do_adaptivity")

println("Number of threads: $(Threads.nthreads())")
println("Number of BLAS threads: $(BLAS.get_num_threads())")
println("Sysimage: ", unsafe_string(Base.JLOptions().image_file))
println("Compile mode: ", Base.JLOptions().compile_enabled)
println("Active project: ", Base.active_project())

function MBB_rhs(x)
    SA[0.0,0.0,0.0]
end


function main_MBB(
    MAX_OPT_STEPS::Int,
    MAX_REF_LEVEL::Int,
    MeshType::Symbol,
    do_adaptivity::Bool,
    do_adaptivity_at_the_start::Bool
)

    n = div(4,2)*2
    l_beam = 3.0

    mesh =if MeshType == :Hexahedra
        create_rectangular_mesh(
            3n,div(n,2),n,
            l_beam,0.5,1.0,StandardEl{K}
        );
    elseif MeshType == :Voronoi
        # mesh2d = create_voronoi_mesh(
        #     (0.0,0.0),
        #     (l_beam,0.5),
        #     3n,div(n,2),StandardEl{K}
        #     )
        # extrude_to_3d(n,mesh2d,1.0);

        mesh2d = create_voronoi_mesh(
            (0.0,0.0),
            (l_beam,1.0),
            3n,n,StandardEl{K}
            )
        _mesh = extrude_to_3d(n,mesh2d,0.5);
        permute_coord_dimensions(_mesh,SA[1,3,2])
    end

    h_cell = find_maximal_cell_diameter(mesh.topo)
    h_cell_min = h_cell * (1/2)^(MAX_REF_LEVEL-1)
    if do_adaptivity_at_the_start || !do_adaptivity
        for level in 1:(MAX_REF_LEVEL-1)
            for element in RootIterator{4}(mesh.topo)
                _refine!(element,mesh.topo)
            end
        end
        mesh = Mesh(mesh.topo,StandardEl{1}())
    end




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

    sets_to_refine = get_sets_to_refine(:MBB_sym)
    cv,no_coarsening_marker = refine_sets(cv,sets_to_refine,MAX_REF_LEVEL);

    ch = create_constraint_handler(cv,:MBB_sym);
    write_vtk(cv.mesh.topo,"test")

    states = TopStates{U}(cv,ρ_init)

    
    # Get project root directory (robust to where script is called from)
    project_root = dirname(@__DIR__)

    println("="^100)
    println("Starting optimization")
    println("="^100)
    sim_results = run_optimization(
        cv,
        MBB_rhs, 
        states,
        ch,
        no_coarsening_marker,
        sim_pars,
        vtk_folder_name = joinpath(project_root, "Results", "vtk", "Adaptive_Runs", "MBB_$(n)_$(MAX_REF_LEVEL)_$(MeshType)"),
        MAX_OPT_STEPS = MAX_OPT_STEPS,
        MAX_REF_LEVEL = MAX_REF_LEVEL,
        take_snapshots_at = [1,10,20,30,50,100,200],
        do_adaptivity = do_adaptivity,
        b_case = :MBB_sym
    )

  

    show(to)


    jld2_path = joinpath(project_root, "Results", "SimData", "MBB_$(n)_$(MAX_REF_LEVEL)_$(MeshType).jld2")

    @save jld2_path sim_results
    export_sim_data_for_latex(sim_results, joinpath(project_root, "Results", "SimData", "MBB_$(n)_$(MAX_REF_LEVEL)_$(MeshType).csv"))
end

main_MBB(
    MAX_OPT_STEPS,MAX_REF_LEVEL, 
    MeshType,do_adaptivity,do_adaptivity_at_the_start)
