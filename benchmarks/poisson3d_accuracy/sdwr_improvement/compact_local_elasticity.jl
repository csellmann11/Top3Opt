include("elasticity_control.jl")
isdefined(@__MODULE__,:compact_local_defects) || include("compact_local.jl")

"""Lift geometric scalar modes componentwise and use the actual elastic energy.

The scalar Poisson Gram matrix is never used. Each retained vector patch has
at most 18/21 coefficients. The enriched sparse matrix is only an integration
and residual-update harness; no enriched or coarse correction solve is made.
"""
function compact_elasticity_blocks(mesh,fan,scalar_geometry,enriched;
        include_hat=false,rank_tolerance=1e-11)
    geometric = compact_local_defects(mesh,fan,scalar_geometry)
    blocks = NamedTuple[]
    ranks = zeros(Int,fan.vertex_count)
    max_energy_defect = 0.0
    for node in 1:fan.vertex_count
        ids,B = compact_local_basis(scalar_geometry,geometric,node;include_hat)
        vector_ids = elasticity_dofs(ids)
        vector_B = kron(B,Matrix{Float64}(I,3,3))
        Q = compact_local_energy_basis(enriched.K,vector_ids,vector_B;rank_tolerance)
        rank = size(Q,2)
        ranks[node] = rank
        rank == 0 && continue
        image_ids = sort!(unique!([enriched.K.rowval[p] for id in vector_ids
            for p in nzrange(enriched.K,id)]))
        image = enriched.K[image_ids,vector_ids]*Q
        defect = norm(transpose(Q)*enriched.K[vector_ids,vector_ids]*Q-I,Inf)
        max_energy_defect = max(max_energy_defect,defect)
        push!(blocks,(;node,ids=vector_ids,basis=Q,image_ids,image))
    end
    max_energy_defect < 1e-8 || error("Elastic patch energy basis is not orthonormal")
    statistics = (;patch_count=fan.vertex_count,active_patches=length(blocks),
        local_coefficients=sum(ranks),minimum_rank=minimum(ranks),maximum_rank=maximum(ranks),
        rank_histogram=[count(==(r),ranks) for r in 0:3(6+include_hat)],
        include_hat,max_energy_defect,rank_tolerance)
    return (;blocks,statistics)
end

function run_compact_elasticity(;levels=(4,),mode=3,order=6,
        goals=(:integral_u,:energy),checkpoints=(1,2,4),include_hats=(false,true),
        output=joinpath(@__DIR__,"..","results","compact_local_elasticity"))
    BLAS.set_num_threads(1)
    mkpath(output)
    reference = sine_uniform_reference(;order=4)
    rows = NamedTuple[]
    for n in levels
        println("COMPACT_ELASTICITY_PREPARE ",(;n)); flush(stdout)
        mesh = build_mesh(:cube,n)
        fan = build_fan(mesh)
        geometry = hierarchical_geometry(fan)
        coarse = elasticity_coarse_operator(mesh,1.0,1.0)
        enriched = elasticity_enriched_operator(fan,geometry,1.0,1.0)
        P = kron(hierarchical_prolongation(mesh,fan,geometry),sparse(Matrix{Float64}(I,3,3)))
        problem = elasticity_problem(mode)
        source = elasticity_loads(mesh,fan,geometry,coarse,problem;order)
        uh = elasticity_solve(coarse,source.projected)
        U = P*uh
        bu = source.enriched-enriched.K*U
        for include_hat in include_hats
            collection = nothing
            preparation_seconds = @elapsed collection = compact_elasticity_blocks(mesh,fan,geometry,enriched;include_hat)
            println("COMPACT_ELASTICITY_BLOCKS ",(;n,include_hat,
                maximum_rank=collection.statistics.maximum_rank,preparation_seconds)); flush(stdout)
            for name in goals
                goal = elasticity_goal(enriched,U,problem,name)
                zh = elasticity_solve(coarse,transpose(P)*goal.rhs)
                Z = P*zh
                bz = goal.rhs-enriched.K*Z
                error = goal.exact-goal.value
                residuals = (;bu,bz,consistency=dot(source.enriched,Z)-dot(U,enriched.K*Z),
                    primal_work=dot(source.enriched,U)-dot(U,enriched.K*U))
                baseline = elasticity_sdwr(mesh,coarse,enriched,source,uh,zh,U,Z,name,reference)
                results = nothing
                solve_seconds = @elapsed results = compact_local_estimates(enriched,residuals,collection;
                    checkpoints,remainder=goal.remainder,energy_goal=name==:energy)
                for result in results
                    result.residual_drift < 1e-10 || error("Elastic local residual update drift")
                    row = (;n,mode,goal=String(name),include_hat,steps=result.steps,
                        local_maximum_rank=collection.statistics.maximum_rank,
                        local_minimum_rank=collection.statistics.minimum_rank,
                        patch_count=collection.statistics.patch_count,
                        coarse_unknowns=length(coarse.free),fine_harness_unknowns=length(enriched.free),
                        relative_goal_error=abs(error/goal.exact),signed_goal_error=error,
                        original_effectivity=baseline.estimate/error,signed_estimate=result.estimate,
                        signed_effectivity=result.estimate/error,primal_residual=result.primal_residual,
                        dual_residual=result.dual_residual,residual_drift=result.residual_drift,
                        max_energy_defect=collection.statistics.max_energy_defect,
                        preparation_seconds,all_checkpoint_sweeps_seconds=solve_seconds)
                    push!(rows,row)
                    write_csv(joinpath(output,"accuracy.csv"),rows)
                    println("COMPACT_ELASTICITY ",(;n,mode,goal=name,include_hat,steps=result.steps,
                        original=baseline.estimate/error,effectivity=row.signed_effectivity)); flush(stdout)
                end
            end
        end
        GC.gc()
    end
    return rows
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_compact_elasticity()
end
