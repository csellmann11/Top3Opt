include("run_voronoi.jl")
using Test

function sdwr_validate_voronoi(;cases=((:cube_smooth,4),(:lshape_singular,8)),
        output=joinpath(@__DIR__,"..","results","sdwr_voronoi","quadrature_checks.csv"))
    BLAS.set_num_threads(1)
    records = NamedTuple[]
    @testset "Prism goals and independent source quadrature" begin
        for (name,n) in cases
            problem = manufactured_problem(name)
            coarse = sdwr_prepare_prism(problem,n;quadrature_order=6)
            fine = sdwr_prepare_prism(problem,n;quadrature_order=8)
            @test coarse.mesh.points == fine.mesh.points
            @test coarse.mesh.cells == fine.mesh.cells
            for goal_name in (:integral_u,:integral_u2,:localized_u)
                goal = sdwr_goal(goal_name,problem.domain)
                rules = sdwr_geometric_goal_rules(coarse.fan,goal)
                old = sdwr_evaluate_prism(coarse,problem,goal;quadrature_order=6,goal_rules=rules)
                checked = sdwr_evaluate_prism(fine,problem,goal;quadrature_order=8,goal_rules=rules)
                goal_change = abs(checked.error-old.error)/abs(checked.error)
                estimator_change = abs(checked.estimate.signed-old.estimate.signed)/abs(checked.error)
                @test goal_change < 1e-3
                @test estimator_change < 1e-3
                direction = [sin(0.17i) for i in eachindex(fine.primal.u)]
                direction[fine.mesh.boundary_vertices] .= 0.0
                epsilon = 1e-5
                plus = reconstruct_fan(fine.mesh,fine.fan,fine.primal.u+epsilon*direction)
                minus = reconstruct_fan(fine.mesh,fine.fan,fine.primal.u-epsilon*direction)
                finite_difference = (sdwr_geometric_goal(fine.mesh,fine.fan,plus,goal,rules).value-
                    sdwr_geometric_goal(fine.mesh,fine.fan,minus,goal,rules).value)/(2epsilon)
                @test dot(checked.dual.rhs,direction) ≈ finite_difference rtol=1e-8 atol=1e-11
                if goal_name == :localized_u
                    @test sum(rule.volume for rule in rules) ≈
                        prod(bounds[2]-bounds[1] for bounds in goal.box) rtol=2e-10
                end
                record = (;problem=String(name),goal=String(goal_name),n,
                    cells=length(fine.mesh.cells),free_dofs=length(fine.free),
                    goal_change,estimator_change,
                    checked_effectivity=checked.estimate.signed/checked.error)
                push!(records,record)
                println("PRISM_QUADRATURE ",record); flush(stdout)
                mkpath(dirname(output)); write_csv(output,records)
            end
        end
    end
    return records
end

if abspath(PROGRAM_FILE) == @__FILE__
    sdwr_validate_voronoi()
end
