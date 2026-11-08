include("run.jl")
using Test

function sine_direct_fan_load(mesh,fan,problem;order=8)
    rule = tetra_rule(order)
    load = zeros(size(fan.points,2))
    for tet in eachindex(fan.volumes)
        ids = fan.tet_nodes[:,tet]
        points = fan.points[:,ids]*rule.barycentric
        for q in eachindex(rule.weights)
            value = fan.volumes[tet]*rule.weights[q]*problem.rhs(view(points,:,q))
            load[ids] .+= value.*rule.barycentric[:,q]
        end
    end
    return sdwr_reconstruction_transpose(mesh,fan,load)
end

"""Compare translated tensors and reduced estimator with the original full fan.

This deliberately assembles, solves, reconstructs and integrates through the
pre-existing benchmark functions, rather than rechecking the reduced formula.
"""
function validate_sine_original(;cases=((4,(1,1,1)),(4,(3,1,1)),(8,(5,1,1))),order=8)
    BLAS.set_num_threads(1)
    reference = sine_uniform_reference(;order)
    records = NamedTuple[]
    @testset "Sine sweep agrees with original explicit fan and sDWR" begin
        for (n,mode) in cases
            problem = sine_problem(mode)
            system = sine_uniform_system(n,reference)
            source = sine_uniform_source(system,reference,mode)
            fast_u = sine_uniform_solve(system,source.projected).u
            mesh = build_mesh(:cube,n)
            fan = build_fan(mesh)
            assembled = assemble_primal(mesh,fan,problem;quadrature_order=order)
            free = findall(!,mesh.boundary_vertices)
            factor = cholesky(Symmetric(assembled.K[free,free]))
            primal = sdwr_solve_rhs(assembled,mesh,assembled.rhs,factor,free)
            mapping = [begin
                i,j,k = round.(Int,n.*mesh.points[:,a])
                1+i+(n+1)*(j+(n+1)*k)
            end for a in axes(mesh.points,2)]
            @test assembled.K ≈ system.K[mapping,mapping] rtol=2e-13 atol=1e-14
            @test assembled.rhs ≈ source.projected[mapping] rtol=2e-9 atol=1e-12
            @test sine_direct_fan_load(mesh,fan,problem;order) ≈
                source.reconstructed[mapping] rtol=2e-9 atol=1e-12
            @test primal.u ≈ fast_u[mapping] rtol=2e-9 atol=1e-12
            U = reconstruct_fan(mesh,fan,primal.u)
            for name in (:integral_u,:integral_u2,:energy)
                goal = sine_goal(name)
                if name == :energy
                    full_goal = sine_full_energy(mesh,fan,U)
                    rhs,value = full_goal.load,full_goal.value
                else
                    rhs = sdwr_goal_load(mesh,fan,U,goal)
                    value = sdwr_reconstructed_goal(mesh,fan,U,goal).value
                end
                dual = sdwr_solve_rhs(assembled,mesh,rhs,factor,free)
                full = sdwr_estimate(mesh,fan,primal,dual,problem,goal;quadrature_order=order)
                fast = sine_uniform_evaluate(system,reference,problem,source,fast_u,name)
                @test rhs ≈ fast.rhs[mapping] rtol=3e-9 atol=2e-12
                @test dual.u ≈ fast.z[mapping] rtol=3e-9 atol=2e-12
                @test value ≈ fast.reconstructed rtol=3e-9 atol=2e-12
                @test full.signed ≈ fast.estimate rtol=3e-9 atol=2e-12
                record = (;n,mode=join(mode,"_"),goal=String(name),
                    original_estimate=full.signed,reduced_estimate=fast.estimate,
                    relative_difference=abs(full.signed-fast.estimate)/max(abs(full.signed),eps()))
                push!(records,record)
                println("SINE_ORIGINAL_CHECK ",record); flush(stdout)
            end
        end
    end
    return records
end

function validate_sine_quadrature(;n=16,mode=(5,1,1))
    BLAS.set_num_threads(1)
    problem = sine_problem(mode)
    low,high = sine_uniform_reference(;order=6),sine_uniform_reference(;order=10)
    system = sine_uniform_system(n,high)
    a,b = sine_uniform_source(system,low,mode),sine_uniform_source(system,high,mode)
    ua,ub = sine_uniform_solve(system,a.projected).u,sine_uniform_solve(system,b.projected).u
    records = NamedTuple[]
    @testset "Sine quadrature and analytic references" begin
        @test a.projected ≈ b.projected rtol=1e-10 atol=1e-13
        @test a.reconstructed ≈ b.reconstructed rtol=1e-10 atol=1e-13
        box = ((0.0,1.0),(0.0,1.0),(0.0,1.0))
        @test sdwr_box_integral(problem.exact,box;order=32) ≈ sine_reference(problem,:integral_u) rtol=1e-12
        @test sdwr_box_integral(x->problem.exact(x)^2,box;order=32) ≈ 1/8 rtol=1e-12
        @test sdwr_box_integral(x->sum(abs2,problem.gradient(x))/2,box;order=32) ≈
            sine_reference(problem,:energy) rtol=1e-12
        for name in (:integral_u,:integral_u2,:energy)
            ea = sine_uniform_evaluate(system,low,problem,a,ua,name)
            eb = sine_uniform_evaluate(system,high,problem,b,ub,name)
            goal_change = abs(ea.error-eb.error)/abs(eb.error)
            estimate_change = abs(ea.estimate-eb.estimate)/abs(eb.error)
            @test goal_change < 1e-9
            @test estimate_change < 1e-9
            push!(records,(;n,mode=join(mode,"_"),goal=String(name),goal_change,estimate_change))
        end
    end
    return records
end

"""Check saved sparse solves against independently derived sine symbols."""
function validate_sine_saved(paths)
    records = NamedTuple[]
    @testset "Sparse results versus independent sine symbols" begin
        for path in paths
            lines = readlines(path)
            names = Symbol.(split(first(lines),','))
            for line in lines[2:end]
                row = Dict(zip(names,split(line,',')))
                n = parse(Int,row[:n])
                mode = Tuple(parse(Int,row[key]) for key in (:m,:k,:p))
                goal = Symbol(row[:goal])
                symbols = sine_discrete_symbols(mode,n)
                amplitude = parse(Float64,row[:nodal_amplitude])
                @test amplitude ≈ symbols.amplitude rtol=3e-11 atol=1e-13
                @test parse(Float64,row[:relative_nonsine_component]) < 1e-10
                exact_discrete = if goal == :integral_u
                    symbols.amplitude*prod(cot(pi*mode[d]/(2n)) for d in 1:3)/n^3
                elseif goal == :integral_u2
                    symbols.amplitude^2*(n^3/8)*symbols.M
                else
                    symbols.amplitude^2*(n^3/16)*symbols.H
                end
                value = parse(Float64,row[:reconstructed_goal])
                @test value ≈ exact_discrete rtol=3e-11 atol=1e-13
                push!(records,(;n,mode=join(mode,"_"),goal=String(goal),
                    amplitude_relative_error=abs(amplitude-symbols.amplitude)/abs(symbols.amplitude),
                    goal_relative_error=abs(value-exact_discrete)/abs(exact_discrete),
                    effectivity=parse(Float64,row[:signed_effectivity]),
                    predicted_limit=sine_effectivity_limit(mode,goal)))
            end
        end
    end
    return records
end

if abspath(PROGRAM_FILE) == @__FILE__
    output = joinpath(@__DIR__,"..","results","sdwr_sine")
    mkpath(output)
    write_csv(joinpath(output,"original_checks.csv"),validate_sine_original())
    write_csv(joinpath(output,"quadrature_checks.csv"),validate_sine_quadrature())
end
