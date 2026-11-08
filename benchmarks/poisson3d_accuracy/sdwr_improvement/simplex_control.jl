include("hierarchical_run.jl")
include("../investigation/irregular_tetrahedra.jl")
using Test

"""True P1 FEM unknowns on every vertex of a perturbed tetrahedral mesh."""
function simplex_prepare(problem,n;order=8,amplitude=0.08)
    generated = irregular_tetrahedral_mesh(problem.domain,n;amplitude)
    mesh,fan = generated.mesh,generated.fan
    assembled = assemble_tetrahedral_p1(fan,problem;quadrature_order=order)
    free = findall(!,fan.boundary_nodes)
    factor = cholesky(Symmetric(assembled.K[free,free]))
    u = zeros(length(assembled.rhs))
    u[free] = factor\assembled.rhs[free]
    residual = norm((assembled.K*u-assembled.rhs)[free])/norm(assembled.rhs[free])
    primal = (;u,algebraic_residual=residual)
    return (;mesh,fan,assembled,free,factor,primal,U=u,geometry_stats=generated.geometry_stats)
end

"""Exact moments and derivative of the continuous tetrahedral P1 field."""
function simplex_goal_data(prepared,name)
    fan,U = prepared.fan,prepared.U
    rhs = zeros(length(U))
    value = 0.0
    if name == :energy
        rhs = prepared.assembled.K*U
        return (;rhs,value=dot(U,rhs)/2)
    end
    for tet in eachindex(fan.volumes)
        ids,volume = fan.tet_nodes[:,tet],fan.volumes[tet]
        local_u = U[ids]
        if name == :integral_u
            value += volume*sum(local_u)/4
            rhs[ids] .+= volume/4
        elseif name == :integral_u2
            value += volume*(sum(local_u)^2+sum(abs2,local_u))/20
            rhs[ids] .+= (volume/10).*(sum(local_u).+local_u)
        else
            throw(ArgumentError("unsupported simplex goal $name"))
        end
    end
    return (;rhs,value)
end

function simplex_evaluate(prepared,problem,name)
    goal = sine_goal(name)
    data = simplex_goal_data(prepared,name)
    z = zeros(length(data.rhs))
    z[prepared.free] = prepared.factor\data.rhs[prepared.free]
    residual = norm((prepared.assembled.K*z-data.rhs)[prepared.free])/
        max(norm(data.rhs[prepared.free]),eps())
    exact = hasproperty(problem,:modes) ? sine_reference(problem,name) :
        name == :energy ? problem.reference_energy : sdwr_goal_reference(problem,goal).value
    return (;goal,dual=(;u=z),exact,reconstructed=data.value,error=exact-data.value,
        dual_residual=residual)
end

"""Simplex control: the affine proxy is the original P1 field itself.

Hence p=U and q=Z on each actual tetrahedron; the original signed sDWR equals
R_U(Z), which vanishes by the actual P1 Galerkin equations. Parent cube labels
are used only for reporting; no polygonal VEM projection is applied here.
The new estimator uses the same generic hierarchical residual implementation
as the VEM tests, with an explicit P1 embedding and the reused P1 factorization.
"""
function run_simplex_control(;jobs=((:cube_smooth,2),(:cube_smooth,4),(:sine_511,4)),
        order=8,amplitude=0.08,checkpoints=(8,16),
        output=joinpath(@__DIR__,"..","results","sdwr_improvement","simplex_control.csv"))
    BLAS.set_num_threads(1)
    mkpath(dirname(output))
    rows = NamedTuple[]
    @testset "Perturbed simplex residual control" begin
        for (name,n) in jobs
            problem = name == :sine_511 ? sine_problem((5,1,1)) : manufactured_problem(name)
            println("SIMPLEX_PREPARING ",(;name,n));flush(stdout)
            prepared = simplex_prepare(problem,n;order,amplitude)
            fan = prepared.fan
            geometry = hierarchical_geometry(fan)
            nv,nd = length(prepared.U),size(geometry.K,1)
            prolongation = sparse(collect(1:nv),collect(1:nv),ones(nv),nd,nv)
            @test geometry.K[1:nv,1:nv] ≈ prepared.assembled.K rtol=2e-13 atol=1e-14
            @test prepared.geometry_stats.minimum_orientation_ratio > 0
            @test prepared.primal.algebraic_residual < 1e-11
            for goal_name in (:integral_u,:integral_u2,:energy)
                evaluated = simplex_evaluate(prepared,problem,goal_name)
                residuals = hierarchical_residuals(prepared.mesh,fan,geometry,prepared.primal,
                    evaluated.dual,problem,evaluated.goal;order,U=prepared.U,Z=evaluated.dual.u)
                original = residuals.consistency
                algebraic_consistency = dot(prepared.assembled.rhs,evaluated.dual.u)-
                    dot(prepared.U,prepared.assembled.K*evaluated.dual.u)
                @test abs(original) < 1e-10*max(abs(evaluated.exact),eps())
                @test original ≈ algebraic_consistency atol=1e-11*max(abs(evaluated.exact),eps())
                @test evaluated.dual_residual < 1e-11
                remainder = goal_name == :integral_u ? (delta->0.0) :
                    goal_name == :energy ? (delta->dot(delta,geometry.K*delta)/2) :
                    (delta->hierarchical_goal_remainder(fan,geometry,delta,evaluated.goal))
                estimates = hierarchical_pcg_estimates(prepared.mesh,fan,geometry,residuals,
                    prepared;checkpoints,remainder,energy_goal=goal_name==:energy,prolongation)
                for result in estimates
                    row = (;problem=String(problem.name),n,goal=String(goal_name),
                        perturbation=amplitude,parent_cells=length(prepared.mesh.cells),
                        tetrahedra=length(fan.volumes),p1_free_dofs=length(prepared.free),
                        enriched_free_dofs=length(geometry.free),exact_goal=evaluated.exact,
                        reconstructed_goal=evaluated.reconstructed,true_goal_error=evaluated.error,
                        relative_goal_error=abs(evaluated.error/evaluated.exact),
                        original_estimate=original,original_effectivity=original/evaluated.error,
                        method=String(result.method),steps=result.steps,estimate=result.estimate,
                        effectivity=result.estimate/evaluated.error,
                        primal_correction_residual=result.primal_residual,
                        dual_correction_residual=result.dual_residual,
                        primal_solve_residual=prepared.primal.algebraic_residual,
                        dual_solve_residual=evaluated.dual_residual)
                    push!(rows,row)
                end
                write_csv(output,rows)
                println("SIMPLEX_RESULT ",(;name,n,goal=goal_name,
                    old=original/evaluated.error,
                    pcg8=estimates[2].estimate/evaluated.error,
                    pcg16=estimates[end].estimate/evaluated.error));flush(stdout)
            end
            GC.gc()
        end
    end
    return rows
end

if abspath(PROGRAM_FILE)==@__FILE__
    run_simplex_control()
end
