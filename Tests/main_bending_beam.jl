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


const to = TimerOutput()


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
include("optim_run.jl")
const K = 1
const U = 3

if length(ARGS) == 0
    MAX_OPT_STEPS = 500
    MAX_REF_LEVEL = 3
    MeshType = :Hexahedra
    do_adaptivity = false
elseif length(ARGS) == 1
    MAX_OPT_STEPS = parse(Int, ARGS[1])
    MAX_REF_LEVEL = 3
    MeshType = :Hexahedra
    do_adaptivity = true
elseif length(ARGS) == 2
    MAX_OPT_STEPS = parse(Int, ARGS[1])
    MAX_REF_LEVEL = parse(Int, ARGS[2])
    MeshType = :Hexahedra
    do_adaptivity = true
elseif length(ARGS) == 3
    MAX_OPT_STEPS = parse(Int, ARGS[1])
    MAX_REF_LEVEL = parse(Int, ARGS[2])
    MeshType = ARGS[3] |> Symbol
    do_adaptivity = true
elseif length(ARGS) == 4
    MAX_OPT_STEPS = parse(Int, ARGS[1])
    MAX_REF_LEVEL = parse(Int, ARGS[2])
    MeshType = ARGS[3] |> Symbol
    do_adaptivity = parse(Bool, ARGS[4])
else
    error("Invalid number of arguments: $(length(ARGS))")
end


println("Running simulation with:")
println("MAX_OPT_STEPS: $MAX_OPT_STEPS")
println("MAX_REF_LEVEL: $MAX_REF_LEVEL")
println("MeshType: $MeshType")
println("do_adaptivity: $do_adaptivity")

#print number of threads
println("Number of threads: $(Threads.nthreads())")


function Bending_Beam_rhs(x)
    SA[0.0,0.0,0.0]
end




n = div(4,2)*2
l_beam = 3.0

mesh = if MeshType == :Hexahedra
    create_rectangular_mesh(
        3n,div(n,2),n,
        l_beam,0.5,1.0,StandardEl{K}
    )
elseif MeshType == :Voronoi
    mesh2d = create_voronoi_mesh(
        (0.0,0.0),
        (l_beam,0.5),
        3n,div(n,2),StandardEl{K}
        )
    extrude_to_3d(n,mesh2d,1.0)
else
    error("Invalid MeshType: $MeshType")
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



E = 2.e05; ν = 0.30
λ,μ = E_ν_to_lame(E,ν)
χ = 0.1
mat_law  = Helmholtz{3,3}(Ψlin_totopt,(λ,μ,χ))
mat_pars = (λ,μ)

χmin = 1e-03
η0   = 15.0 
β0   = 2*h_cell_min^2 
ρ_init = 0.15 
sim_pars = SimParameter(mat_law,λ,μ,χmin,η0,1.0,ρ_init,h_cell_min)

@time cv = CellValues{U}(mesh);

states = TopStates{U}(cv,ρ_init)

ch = create_constraint_handler(cv,:Bending_Beam_sym);

# Get project root directory (robust to where script is called from)
project_root = dirname(@__DIR__)

println("="^100)
println("Starting optimization")
println("="^100)
optimization_time = @elapsed sim_results = run_optimization(
    cv,
    Bending_Beam_rhs,
    states,
    ch,
    sim_pars,
    vtk_folder_name = joinpath(project_root, "Results", "vtk", "Adaptive_Runs", "Bending_Beam_$(n)_$(MAX_REF_LEVEL)_$(MeshType)"),
    MAX_OPT_STEPS = MAX_OPT_STEPS,
    MAX_REF_LEVEL = MAX_REF_LEVEL,
    take_snapshots_at = [1,10,20,30,50,100,200],
    do_adaptivity = do_adaptivity,
    b_case = :Bending_Beam_sym
)

println("Optimization time: $optimization_time")
show(to)


jld2_path = joinpath(project_root, "Results", "SimData", "Bending_Beam_$(n)_$(MAX_REF_LEVEL)_$(MeshType).jld2")

@time "jld_save" @save jld2_path sim_results
@time "export_data" export_sim_data_for_latex(sim_results, joinpath(project_root, "Results", "SimData", "Bending_Beam_$(n)_$(MAX_REF_LEVEL)_$(MeshType).csv"))

