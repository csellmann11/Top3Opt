# CPU-only integration benchmark: same assembly/adaptivity as the cluster driver,
# with a sparse direct solve to isolate regularization from PETSc configuration.
using Ju3VEM, Ju3VEM.FixedSizeArrays, StaticArrays, LinearAlgebra, SparseArrays
using OrderedCollections, Statistics, Bumper, TimerOutputs, Serialization, Test
import Ju3VEM.FR as FR
const U = 3
const to = TimerOutput()
BLAS.set_num_threads(1)
include("../src/mat_states.jl")
include("../src/laplace_operator.jl")
include("../src/compute_displacement.jl")
include("../src/bisection.jl")
include("../src/utils/mesh_processing_utils.jl")
include("../src/neighbor_search.jl")
include("../src/refinement_utils/estiamte_element_error.jl")
include("../src/get_sparsity_pattern.jl")
include("taylor_reference.jl")
solve_lse(K, f, cv, ch) = cholesky(Symmetric(K)) \ f

function benchmark_regularization(scheme=:diamond; steps=30, level=3, base_n=4)
    mesh = create_rectangular_mesh(3base_n,div(base_n,2),base_n,3.,0.5,1.,StandardEl{1})
    for _ in 2:level, el in RootIterator{4}(mesh.topo)
        _refine!(el,mesh.topo)
    end
    mesh = Mesh(mesh.topo,StandardEl{1}())
    mesh, protected = refine_sets(mesh,get_sets_to_refine(:MBB_sym),level)
    cv = CellValues{3}(mesh)
    states = DesignVarInfo{3}(cv,0.3)
    lam,mu = E_ν_to_lame(210.e3,0.33)
    pars = SimPars(Helmholtz{3,3}(Ψlin_totopt,(lam,mu,1.)),lam,mu,1e-3,15.,1.,0.3)
    for step in 1:steps
        ch = create_constraint_handler(cv,:MBB_sym)
        top = @elapsed R = if scheme in (:diamond,:tpfa)
            compute_flux_operator_mat(cv,states,pars;scheme)
        elseif scheme == :taylor
            compute_reference_taylor_mat(cv,states,pars)
        elseif scheme == :strong
            ns, ghosts = create_neigh_list(states,cv)
            compute_laplace_operator_mat(cv.mesh.topo,ns,ghosts,states,true)
        else
            error("Unknown comparison scheme: $scheme")
        end
        row = maximum(vec(sum(abs,R;dims=2)))
        println((step=step,scheme=scheme,cells=length(states.χ_vec),row_bound=row,operator_seconds=top))
        flush(stdout)
        if row > 1e5 && scheme == :diamond
            dest = joinpath(@__DIR__,"..","Results","bad_diamond_stencil.bin")
            mkpath(dirname(dest)); serialize(dest,(cv,states,pars))
            error("Oversized diamond coefficients; saved mesh to $dest")
        end
        u,K,ed = compute_displacement(cv,ch,states,x->SA[0.,0.,0.],pars)
        tu = @elapsed changed = state_update!(states,cv,pars,R,u,ed;beta_in_operator=scheme != :strong)
        @test all(isfinite,states.χ_vec)
        @test all(x->pars.χmin <= x <= 1.,states.χ_vec)
        @test abs(dot(states.area_vec,states.χ_vec)/sum(states.area_vec)-pars.ρ_init) < 1e-8
        println((energy=dot(u,K*u)/2,density_seconds=tu,peak_MiB=Sys.maxrss()/2.0^20))
        flush(stdout)
        step == steps && break
        err = estimate_element_error(u,states,cv,ed)
        ref,coarse = mark_elements_for_adaption(cv,err,states,changed,level,protected,true)
        clear_up_topo!(cv.mesh.topo)
        cv = adapt_mesh(cv,coarse,ref)
        update_states_after_mesh_adaption!(states,cv,ed,ref,coarse)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    benchmark_regularization(isempty(ARGS) ? :diamond : Symbol(ARGS[1]);
        steps=length(ARGS)>1 ? parse(Int,ARGS[2]) : 30,
        level=length(ARGS)>2 ? parse(Int,ARGS[3]) : 3)
end
