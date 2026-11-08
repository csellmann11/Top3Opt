include("hierarchical_run.jl")
include("compact_local.jl")
using Test

function compact_local_polynomial_check(prepared,geometry)
    geometric = compact_local_defects(prepared.mesh,prepared.fan,geometry)
    reconstructed = zeros(size(geometric.defects))
    for node in 1:prepared.fan.vertex_count
        ids,B = compact_local_basis(geometry,geometric,node;mask_boundary=false)
        reconstructed[ids,:] .+= B
    end
    target = copy(geometric.defects)
    for (edge,(a,b)) in enumerate(geometry.edges)
        target[geometry.vertex_count+edge,:] .-= (target[a,:]+target[b,:])/2
    end
    defect = norm(reconstructed-target,Inf)/max(norm(target,Inf),eps())
    @test defect < 2e-12
    return defect
end

function run_compact_local_tests(;jobs=((:cube_smooth,4),(:sine_511,4),
        (:cube_smooth,8),(:sine_511,8)),goals=(:integral_u,:energy),order=6,
        output=joinpath(@__DIR__,"..","results","sdwr_improvement","compact_local.csv"))
    BLAS.set_num_threads(1)
    rows = NamedTuple[]
    for (name,n) in jobs
        println("COMPACT_PREPARING ",(;name,n)); flush(stdout)
        problem = name == :sine_511 ? sine_problem((5,1,1)) : manufactured_problem(name)
        prepared = hierarchical_prepare(problem,n;order)
        geometry = hierarchical_geometry(prepared.fan)
        polynomial_defect = compact_local_polynomial_check(prepared,geometry)
        collections = [compact_local_blocks(prepared.mesh,prepared.fan,geometry;include_hat)
            for include_hat in (false,true)]
        println("COMPACT_BLOCKS ",(;name,n,polynomial_defect,
            six=collections[1].statistics,seven=collections[2].statistics)); flush(stdout)
        for goal_name in goals
            evaluated = hierarchical_evaluate_goal(prepared,problem,goal_name)
            residuals = hierarchical_residuals(prepared.mesh,prepared.fan,geometry,
                prepared.primal,evaluated.dual,problem,evaluated.goal;order)
            for collection in collections
                @test collection.statistics.max_energy_defect < 2e-10
                results = compact_local_estimates(geometry,residuals,collection;
                    energy_goal=goal_name==:energy)
                for result in results
                    @test result.residual_drift < 1e-10
                    push!(rows,(;problem=String(problem.name),n,goal=String(goal_name),
                        include_hat=collection.statistics.include_hat,steps=result.steps,
                        primal_dofs=length(prepared.free),fine_dofs=length(geometry.free),
                        patches=collection.statistics.active_patches,
                        local_coefficients=collection.statistics.local_coefficients,
                        true_goal_error=evaluated.error,estimate=result.estimate,
                        effectivity=result.estimate/evaluated.error,
                        primal_residual=result.primal_residual,dual_residual=result.dual_residual,
                        residual_drift=result.residual_drift,polynomial_defect))
                end
                println("COMPACT_RESULT ",(;name,n,goal=goal_name,
                    include_hat=collection.statistics.include_hat,
                    effectivities=[r.estimate/evaluated.error for r in results])); flush(stdout)
            end
            mkpath(dirname(output)); write_csv(output,rows)
        end
        GC.gc()
    end
    return rows
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_compact_local_tests()
end
