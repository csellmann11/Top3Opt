include("../sdwr_adaptive/run_voronoi_adaptive.jl")
include("hierarchical_diagonal.jl")

"""Integrate localized linear goals on exact tetrahedron/box intersections."""
function transfer_residuals(prepared,evaluated,problem,goal,geometry;order=6)
    mesh,fan = prepared.mesh,prepared.fan
    if goal.name != :localized_u
        return hierarchical_residuals(mesh,fan,geometry,prepared.primal,evaluated.dual,
            problem,goal;order)
    end
    zero_goal = (;derivative_value=(x,u)->0.0,
        derivative_gradient=(x,gradient)->(0.0,0.0,0.0))
    residuals = hierarchical_residuals(mesh,fan,geometry,prepared.primal,evaluated.dual,
        problem,zero_goal;order)
    for tet in eachindex(fan.volumes)
        rule = evaluated.goal_rules[tet]
        local_load = zeros(10)
        for q in eachindex(rule.weights)
            N = view(rule.barycentric,:,q)
            local_load[1:4] .+= rule.weights[q].*N
            for (edge,(a,b)) in enumerate(HIERARCHICAL_TET_EDGES)
                local_load[4+edge] += 4rule.weights[q]*N[a]*N[b]
            end
        end
        residuals.bz[geometry.dofs[:,tet]] .+= local_load
    end
    return residuals
end

function transfer_prepare(problem,family,n;order=6)
    if family == :prism
        return sdwr_prepare_prism(problem,n;quadrature_order=order)
    elseif family == :hex
        return sdwr_prepare_mesh(adaptive_initial_leaves(problem.domain,n),problem,n;
            quadrature_order=order)
    elseif family == :adaptive_hex
        # n counts refinement cycles here, not nominal cells per dimension.
        leaves = adaptive_initial_leaves(problem.domain,2)
        goal = sdwr_goal(:localized_u,problem.domain)
        for cycle in 1:n
            prepared = sdwr_prepare_mesh(leaves,problem,2;quadrature_order=order)
            cycle == n && return prepared
            evaluated = sdwr_evaluate(prepared,problem,goal;quadrature_order=order)
            marked = sdwr_mark(evaluated.estimate.indicators;theta=0.5)
            balanced = sdwr_balance_marked(prepared.mesh,marked)
            leaves = adaptive_refine(leaves,balanced)
        end
    end
    throw(ArgumentError("unknown mesh family $family"))
end

function run_mesh_transfer(;families=(:hex,:prism),levels=(4,8),
        problems=(:cube_smooth,:lshape_smooth,:lshape_singular),
        goals=(:integral_u,:integral_u2,:localized_u),order=6,
        output=joinpath(@__DIR__,"..","results","sdwr_improvement","mesh_transfer"))
    BLAS.set_num_threads(1)
    mkpath(output)
    records = NamedTuple[]
    for family in families,name in problems,n in levels
        problem = manufactured_problem(name)
        println("TRANSFER_PREPARING ",(;family,name,n)); flush(stdout)
        primal_time = @elapsed prepared = transfer_prepare(problem,family,n;order)
        mesh,fan = prepared.mesh,prepared.fan
        setup_time = @elapsed geometry = hierarchical_geometry(fan)
        for goal_name in goals
            goal = sdwr_goal(goal_name,problem.domain)
            evaluated = sdwr_evaluate_prism(prepared,problem,goal;quadrature_order=order)
            remainder = goal_name == :integral_u2 ?
                (delta->hierarchical_goal_remainder(fan,geometry,delta,goal;order=4)) : (delta->0.0)
            residual_time = @elapsed residuals = transfer_residuals(prepared,evaluated,problem,goal,geometry;order)
            correction_time = @elapsed candidates = hierarchical_pcg_estimates(mesh,fan,geometry,residuals,prepared;
                checkpoints=(4,8,16),remainder)
            selected = filter(row->row.method==:pcg_variational,candidates)
            baseline = (;method=:sdwr,steps=0,estimate=evaluated.estimate.signed,
                primal_residual=prepared.primal.algebraic_residual,
                dual_residual=evaluated.dual.algebraic_residual)
            for candidate in (baseline,selected...)
                record = (;family=String(family),problem=String(name),resolution=n,
                    goal=String(goal_name),cells=length(mesh.cells),coarse_dofs=length(prepared.free),
                    enriched_dofs=length(geometry.free),method=String(candidate.method),steps=candidate.steps,
                    exact_goal=evaluated.reference.value,reconstructed_goal=evaluated.reconstructed.value,
                    true_error=evaluated.error,relative_goal_error=abs(evaluated.error/evaluated.reference.value),
                    estimate=candidate.estimate,effectivity=candidate.estimate/evaluated.error,
                    primal_residual=candidate.primal_residual,dual_residual=candidate.dual_residual,
                    primal_preparation_seconds=primal_time,enrichment_setup_seconds=setup_time,
                    residual_assembly_seconds=residual_time,all_corrections_seconds=correction_time)
                push!(records,record)
                println("TRANSFER ",(;family,name,n,goal=goal_name,method=candidate.method,
                    steps=candidate.steps,effectivity=record.effectivity)); flush(stdout)
            end
            write_csv(joinpath(output,"accuracy.csv"),records)
        end
        geometry,prepared = nothing,nothing
        GC.gc()
    end
    return records
end

if abspath(PROGRAM_FILE) == @__FILE__
    options = Dict(split(argument[3:end],'=';limit=2) for argument in ARGS)
    symbols(key,default) = Tuple(Symbol.(split(get(options,key,default),',')))
    run_mesh_transfer(;families=symbols("families","hex,prism"),
        levels=Tuple(parse.(Int,split(get(options,"levels","4,8"),','))),
        problems=symbols("problems","cube_smooth,lshape_smooth,lshape_singular"),
        goals=symbols("goals","integral_u,integral_u2,localized_u"),
        output=get(options,"output",joinpath(@__DIR__,"..","results","sdwr_improvement","mesh_transfer")))
end
