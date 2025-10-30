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
include("optim_run.jl")
const K = 1
const U = 3

MAX_OPT_STEPS = 1
MAX_REF_LEVEL = 3
MeshType = :Hexahedra

function Cantilever_rhs(x)
    SA[0.0,0.0,0.0]
end




n = div(4,2)*2
l_beam = 2.0

if MeshType == :Hexahedra
    mesh = create_rectangular_mesh(
        2n,n,n,
        l_beam,1.0,1.0,StandardEl{K}
    );
elseif MeshType == :Voronoi
    mesh2d = create_voronoi_mesh(
        (0.0,0.0),
        (l_beam,0.5),
        2n,div(n,2),StandardEl{K}
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

ch = create_constraint_handler(cv,:Cantilever_sym);

# Get project root directory (robust to where script is called from)
project_root = dirname(@__DIR__)

println("="^100)
println("Starting optimization")
println("="^100)
optimization_time = @elapsed sim_results = run_optimization(
    cv,
    Cantilever_rhs,
    states,
    ch,
    sim_pars,
    vtk_folder_name = joinpath(project_root, "Results", "vtk", "Adaptive_Runs", "Cant_$(n)_$(MAX_REF_LEVEL)_$(MeshType)"),
    MAX_OPT_STEPS = MAX_OPT_STEPS,
    MAX_REF_LEVEL = MAX_REF_LEVEL,
    write_vtk_every_n_steps = 50,
    do_adaptivity = false,
    b_case = :Cantilever_sym
)

println("Optimization time: $optimization_time")
show(to)


export_sim_data_for_latex(sim_results, joinpath(project_root, "Results", "SimData", "Cant_$(n)_$(MAX_REF_LEVEL)_$(MeshType).csv"))

