# CPU integration loader and small benchmark for the production regularization paths.
using Ju3VEM, Ju3VEM.FixedSizeArrays, StaticArrays, LinearAlgebra, SparseArrays
using OrderedCollections, Statistics, Bumper, TimerOutputs, Test
import Ju3VEM.FR as FR
const U = 3
const to = TimerOutput()
BLAS.set_num_threads(1)
include("../src/mat_states.jl")
include("../src/laplace_operator.jl")
include("../src/compute_displacement.jl")
include("../src/bisection.jl")
include("../src/utils/mesh_processing_utils.jl")
include("../src/refinement_utils/estiamte_element_error.jl")
include("../src/get_sparsity_pattern.jl")
include("../src/utils/temp_utils.jl")
include("../src/utils/general_utils.jl")
include("../src/postprocessing/sim_data.jl")
include("../src/postprocessing/topopt_vtk_export.jl")
include("../src/optim_run.jl")
# Isolate density behavior from cluster solver availability.
solve_lse(K, f, cv, ch) = cholesky(Symmetric(K)) \ f

function benchmark_regularization(regularization=:mean;steps=10,level=2,projection=false,update_mode=:explicit)
    mesh = create_rectangular_mesh(12,2,4,3.,0.5,1.,StandardEl{1})
    lam,mu = E_ν_to_lame(210.e3,0.33)
    pars = SimPars(Helmholtz{3,3}(Ψlin_totopt,(lam,mu,1.)),lam,mu,1e-3,15.,1.,0.3)
    destination = joinpath(mktempdir(;cleanup=false),"density_benchmark")
    result = run_optimization(mesh,x -> SA[0.,0.,0.],pars;vtk_folder_name=destination,
        MAX_OPT_STEPS=steps,MAX_REF_LEVEL=level,regularization,projection,update_mode,
        filter_radius=0.1,take_snapshots_at=Int[])
    show(to)
    return result
end
if abspath(PROGRAM_FILE) == @__FILE__
    benchmark_regularization(isempty(ARGS) ? :mean : Symbol(ARGS[1]);
        steps=length(ARGS)>1 ? parse(Int,ARGS[2]) : 10,
        projection=length(ARGS)>2 ? parse(Bool,ARGS[3]) : false)
end
