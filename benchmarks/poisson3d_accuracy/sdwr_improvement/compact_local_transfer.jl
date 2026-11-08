include("run_mesh_transfer.jl")
include("compact_local.jl")

"""Irregular-prism transfer control for small local curvature blocks.

Both variants use only local residual corrections. No global/coarse correction
factorization is called here. The assembled fine K is a validation harness for
the local products and residual updates. Localized goals retain the existing
exact tetrahedron/box intersection integration.
"""
function run_compact_local_transfer(;levels=(4,8),goals=(:integral_u,:localized_u),
        order=6,checkpoints=(1,2,4),
        output=joinpath(@__DIR__,"..","results","sdwr_improvement","compact_local_transfer"))
    BLAS.set_num_threads(1)
    mkpath(output)
    records = NamedTuple[]
    setup_records = NamedTuple[]
    problem = manufactured_problem(:cube_smooth)
    for n in levels
        println("COMPACT_TRANSFER_PREPARING ",(;n));flush(stdout)
        primal_time = @elapsed prepared = transfer_prepare(problem,:prism,n;order)
        mesh,fan = prepared.mesh,prepared.fan
        geometry_time = @elapsed geometry = hierarchical_geometry(fan)
        collections = NamedTuple[]
        for include_hat in (false,true)
            build_time = @elapsed collection = compact_local_blocks(mesh,fan,geometry;include_hat)
            push!(collections,(;include_hat,collection,build_time))
            setup_row = (;n,include_hat,cells=length(mesh.cells),fine_dofs=length(geometry.free),
                blocks=length(collection.blocks),block_setup_seconds=build_time,
                statistics=string(collection.statistics))
            push!(setup_records,setup_row)
            write_csv(joinpath(output,"setup.csv"),setup_records)
            println("COMPACT_TRANSFER_BLOCKS ",setup_row);flush(stdout)
        end
        for goal_name in goals
            goal = sdwr_goal(goal_name,problem.domain)
            evaluated = sdwr_evaluate_prism(prepared,problem,goal;quadrature_order=order)
            residual_time = @elapsed residuals = transfer_residuals(prepared,evaluated,problem,goal,geometry;order)
            baseline = (;method=:sdwr,steps=0,estimate=evaluated.estimate.signed,
                primal_residual=prepared.primal.algebraic_residual,
                dual_residual=evaluated.dual.algebraic_residual)
            candidates = NamedTuple[(;basis="sdwr",candidate=baseline,build_time=0.0,correction_time=0.0)]
            for item in collections
                correction_time = @elapsed results = compact_local_estimates(geometry,residuals,item.collection;
                    checkpoints,remainder=delta->0.0)
                basis = item.include_hat ? "curvature_plus_hat" : "curvature_only"
                for result in results
                    push!(candidates,(;basis,candidate=result,build_time=item.build_time,correction_time))
                end
            end
            for item in candidates
                candidate = item.candidate
                row = (;family="prism",problem=String(problem.name),n,goal=String(goal_name),
                    cells=length(mesh.cells),coarse_dofs=length(prepared.free),fine_dofs=length(geometry.free),
                    basis=item.basis,method=String(candidate.method),steps=candidate.steps,
                    exact_goal=evaluated.reference.value,reconstructed_goal=evaluated.reconstructed.value,
                    true_goal_error=evaluated.error,relative_goal_error=abs(evaluated.error/evaluated.reference.value),
                    estimate=candidate.estimate,effectivity=candidate.estimate/evaluated.error,
                    primal_correction_residual=candidate.primal_residual,
                    dual_correction_residual=candidate.dual_residual,
                    primal_preparation_seconds=primal_time,fine_geometry_seconds=geometry_time,
                    block_setup_seconds=item.build_time,residual_assembly_seconds=residual_time,
                    all_checkpoint_corrections_seconds=item.correction_time)
                push!(records,row)
                println("COMPACT_TRANSFER ",(;n,goal=goal_name,basis=item.basis,
                    steps=candidate.steps,effectivity=row.effectivity));flush(stdout)
            end
            write_csv(joinpath(output,"accuracy.csv"),records)
        end
        prepared,geometry,collections = nothing,nothing,nothing
        GC.gc()
    end
    return records
end

if abspath(PROGRAM_FILE)==@__FILE__
    run_compact_local_transfer()
end
