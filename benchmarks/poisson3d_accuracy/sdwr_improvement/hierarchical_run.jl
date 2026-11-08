include("../sdwr_adaptive/run.jl")
include("../sdwr_sine/cases.jl")
include("hierarchical_diagonal.jl")

function hierarchical_prepare(problem,n;order=6)
    mesh = build_mesh(problem.domain,n)
    fan = build_fan(mesh)
    assembled = assemble_primal(mesh,fan,problem;quadrature_order=order)
    free = findall(!,mesh.boundary_vertices)
    factor = cholesky(Symmetric(assembled.K[free,free]))
    primal = sdwr_solve_rhs(assembled,mesh,assembled.rhs,factor,free)
    U = reconstruct_fan(mesh,fan,primal.u)
    return (;mesh,fan,assembled,free,factor,primal,U)
end

function hierarchical_evaluate_goal(prepared,problem,name)
    mesh,fan,U = prepared.mesh,prepared.fan,prepared.U
    goal = sine_goal(name)
    if name == :energy
        value = sine_full_energy(mesh,fan,U)
        rhs,reconstructed = value.load,value.value
    else
        rhs = sdwr_goal_load(mesh,fan,U,goal)
        reconstructed = sdwr_reconstructed_goal(mesh,fan,U,goal).value
    end
    dual = sdwr_solve_rhs(prepared.assembled,mesh,rhs,prepared.factor,prepared.free)
    exact = hasproperty(problem,:modes) ? sine_reference(problem,name) :
        (name == :energy ? problem.reference_energy : sdwr_goal_reference(problem,goal).value)
    return (;goal,dual,reconstructed,exact,error=exact-reconstructed)
end

function run_hierarchical_comparison(;jobs=((:sine_111,4),(:sine_511,4),(:sine_111,8),
        (:sine_511,8),(:cube_smooth,4),(:cube_smooth,8)),order=6,
        goals=(:integral_u,:integral_u2,:energy),
        output=joinpath(@__DIR__,"..","results","sdwr_improvement","hierarchical.csv"))
    BLAS.set_num_threads(1)
    rows = NamedTuple[]
    mkpath(dirname(output))
    for (name,n) in jobs
        problem = name == :sine_111 ? sine_problem((1,1,1)) :
            name == :sine_511 ? sine_problem((5,1,1)) : manufactured_problem(name)
        println("HIERARCHICAL preparing ",(;name,n)); flush(stdout)
        prepared = hierarchical_prepare(problem,n;order)
        geometry = hierarchical_geometry(prepared.fan)
        for goal_name in goals
            evaluated = hierarchical_evaluate_goal(prepared,problem,goal_name)
            residuals = hierarchical_residuals(prepared.mesh,prepared.fan,geometry,
                prepared.primal,evaluated.dual,problem,evaluated.goal;order)
            remainder = goal_name == :integral_u ? (delta->0.0) :
                goal_name == :energy ? (delta->dot(delta,geometry.K*delta)/2) :
                (delta->hierarchical_goal_remainder(prepared.fan,geometry,delta,evaluated.goal))
            local_results = hierarchical_estimates(geometry,residuals;sweeps=(1,2,4),
                coarse=false,direct=n<=4,remainder)
            coupled = hierarchical_pcg_estimates(prepared.mesh,prepared.fan,geometry,residuals,
                prepared;checkpoints=(4,8,16),remainder,energy_goal=goal_name==:energy)
            for result in vcat(local_results,coupled)
                push!(rows,(;problem=String(problem.name),n,goal=String(goal_name),
                    method=String(result.method),steps=result.steps,cells=length(prepared.mesh.cells),
                    primal_dofs=length(prepared.free),fine_dofs=length(geometry.free),
                    true_goal_error=evaluated.error,relative_goal_error=abs(evaluated.error/evaluated.exact),
                    estimate=result.estimate,effectivity=result.estimate/evaluated.error,
                    raw_effectivity=result.raw_estimate/evaluated.error,
                    quadratic_correction=result.quadratic_correction,
                    primal_residual=get(result,:primal_residual,NaN),
                    dual_residual=get(result,:dual_residual,NaN)))
            end
            write_csv(output,rows)
            println("HIERARCHICAL_RESULT ",(;name,n,goal=goal_name,
                edge=local_results[1].estimate/evaluated.error,
                all=local_results[2].estimate/evaluated.error,
                pcg4=coupled[2].estimate/evaluated.error,
                pcg8=coupled[4].estimate/evaluated.error,
                pcg16=coupled[6].estimate/evaluated.error)); flush(stdout)
        end
        GC.gc()
    end
    return rows
end

"""Warmed isolated phase measurements; mesh-dependent setup is included."""
function hierarchical_timing(;n=8,mode=(5,1,1),order=6)
    BLAS.set_num_threads(1)
    problem = sine_problem(mode)
    mesh_time = @elapsed mesh = build_mesh(:cube,n)
    fan_time = @elapsed fan = build_fan(mesh)
    assembly_time = @elapsed assembled = assemble_primal(mesh,fan,problem;quadrature_order=order)
    free = findall(!,mesh.boundary_vertices)
    primal_solve_time = @elapsed begin
        factor = cholesky(Symmetric(assembled.K[free,free]))
        primal = sdwr_solve_rhs(assembled,mesh,assembled.rhs,factor,free)
    end
    U = reconstruct_fan(mesh,fan,primal.u)
    prepared = (;mesh,fan,assembled,free,factor,primal,U)
    dual_time = @elapsed evaluated = hierarchical_evaluate_goal(prepared,problem,:integral_u)
    sdwr_time = @elapsed sdwr_estimate(mesh,fan,primal,evaluated.dual,problem,evaluated.goal;
        quadrature_order=order)
    setup_time = @elapsed geometry = hierarchical_geometry(fan)
    residual_time = @elapsed residuals = hierarchical_residuals(mesh,fan,geometry,primal,
        evaluated.dual,problem,evaluated.goal;order)
    correction_time = @elapsed result = hierarchical_pcg_estimates(mesh,fan,geometry,residuals,
        prepared;checkpoints=(4,8,16),remainder=delta->0.0)
    return (;n,cells=n^3,primal_dofs=length(free),fine_dofs=length(geometry.free),
        mesh_time,fan_time,assembly_time,primal_solve_time,dual_time,sdwr_time,
        setup_time,residual_time,correction_time,effectivity=result[end].estimate/evaluated.error)
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_hierarchical_comparison()
end
