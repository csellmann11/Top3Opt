include("run.jl")
using Test

function sdwr_read_csv(path)
    lines = readlines(path)
    names = Symbol.(split(first(lines),','))
    return [Dict(names .=> split(line,',')) for line in lines[2:end] if !isempty(line)]
end

"""Restore exact dyadic leaf boxes from their interior centers and side lengths."""
function sdwr_restore_leaves(path)
    rows = sdwr_read_csv(path)
    return [begin
        h = parse(Float64,row[:h])
        x = ntuple(d->parse(Float64,row[(:x,:y,:z)[d]]),3)
        origin = ntuple(d->floor(x[d]/h)*h,3)
        (;origin,h,level=parse(Int,row[:level]))
    end for row in rows]
end

function sdwr_validate_runs(;directory=joinpath(@__DIR__,"..","results","sdwr_adaptive"),
        cases=((:cube_smooth,:integral_u2),(:cube_smooth,:localized_u),
            (:lshape_singular,:integral_u2),(:lshape_singular,:localized_u)),
        check_order=8)
    accuracy = sdwr_read_csv(joinpath(directory,"accuracy.csv"))
    records = NamedTuple[]
    @testset "Adaptive histories and independent final-mesh quadrature" begin
        @test all(row->isfinite(parse(Float64,row[:signed_effectivity])),accuracy)
        @test all(row->parse(Float64,row[:primal_residual])<1e-10,accuracy)
        @test all(row->parse(Float64,row[:dual_residual])<1e-10,accuracy)
        @test all(row->parse(Float64,row[:reference_uncertainty]) <
            1e-8*abs(parse(Float64,row[:signed_goal_error])),accuracy)
        for (name,goal_name) in cases
            candidates = filter(row->row[:problem]==String(name) && row[:goal]==String(goal_name),accuracy)
            isempty(candidates) && continue
            saved = last(candidates)
            cycle = parse(Int,saved[:cycle])
            strategy = saved[:strategy]
            cellsfile = joinpath(directory,"$(name)_$(goal_name)_$(strategy)_$(cycle)_cells.csv")
            leaves = sdwr_restore_leaves(cellsfile)
            problem = manufactured_problem(name)
            goal = sdwr_goal(goal_name,problem.domain)
            prepared = sdwr_prepare_mesh(leaves,problem,parse(Int,saved[:base_n]);
                quadrature_order=check_order)
            evaluated = sdwr_evaluate(prepared,problem,goal;quadrature_order=check_order,
                localization=Symbol(saved[:localization]))
            @test length(prepared.mesh.cells) == parse(Int,saved[:cells])
            @test length(prepared.free) == parse(Int,saved[:free_dofs])
            @test all(pair->pair[2]==0 || abs(leaves[pair[1]].level-leaves[pair[2]].level)<=1,
                prepared.mesh.face_cells)
            old_error = parse(Float64,saved[:signed_goal_error])
            old_eta = parse(Float64,saved[:signed_estimate])
            goal_change = abs(evaluated.error-old_error)/abs(old_error)
            estimator_change = abs(evaluated.estimate.signed-old_eta)/abs(old_error)
            @test goal_change < 1e-3
            @test estimator_change < 1e-3
            # A finite-difference oracle for the nonlinear/linear T-load on this
            # actual adaptive mesh, independent of the dual solve.
            direction = [sin(0.19node) for node in eachindex(prepared.primal.u)]
            direction[prepared.mesh.boundary_vertices] .= 0.0
            epsilon = 1e-5
            plus = reconstruct_fan(prepared.mesh,prepared.fan,prepared.primal.u+epsilon*direction)
            minus = reconstruct_fan(prepared.mesh,prepared.fan,prepared.primal.u-epsilon*direction)
            difference = (sdwr_reconstructed_goal(prepared.mesh,prepared.fan,plus,goal).value-
                sdwr_reconstructed_goal(prepared.mesh,prepared.fan,minus,goal).value)/(2epsilon)
            @test dot(evaluated.dual.rhs,direction) ≈ difference rtol=1e-8 atol=1e-11
            record = (;problem=String(name),goal=String(goal_name),cycle,
                cells=length(leaves),check_order,goal_change,estimator_change,
                checked_effectivity=evaluated.estimate.signed/evaluated.error)
            push!(records,record)
            println("SDWR_QUADRATURE ",record); flush(stdout)
            write_csv(joinpath(directory,"quadrature_checks.csv"),records)
        end
    end
    return records
end

if abspath(PROGRAM_FILE) == @__FILE__
    sdwr_validate_runs(directory=isempty(ARGS) ?
        joinpath(@__DIR__,"..","results","sdwr_adaptive") : first(ARGS))
end
