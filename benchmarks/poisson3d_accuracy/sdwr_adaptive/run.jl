include("load.jl")

"""Close a marked set under face-neighbor 2:1 balance before octree splitting."""
function sdwr_balance_marked(mesh,marked)
    selected = falses(length(mesh.cells))
    selected[marked] .= true
    changed = true
    while changed
        changed = false
        for (left,right) in mesh.face_cells
            right == 0 && continue
            a = mesh.leaves[left].level+selected[left]
            b = mesh.leaves[right].level+selected[right]
            if a > b+1 && !selected[right]
                selected[right] = true
                changed = true
            elseif b > a+1 && !selected[left]
                selected[left] = true
                changed = true
            end
        end
    end
    return findall(selected)
end

function sdwr_solve_rhs(assembled,mesh,rhs,factor,free)
    u = zeros(length(rhs))
    isempty(free) || (u[free] = factor\rhs[free])
    residual = assembled.K*u-rhs
    relative_residual = norm(residual[free])/max(norm(rhs[free]),eps(Float64))
    return (;u,K=assembled.K,rhs,cells=assembled.cells,free_dofs=free,
        energy=dot(u,assembled.K*u)/2,algebraic_residual=relative_residual)
end

function sdwr_prepare_mesh(leaves,problem,n;quadrature_order=6)
    mesh = adaptive_build_mesh(leaves;domain=problem.domain,n)
    fan = adaptive_build_fan(mesh)
    assembled = adaptive_assemble(mesh,fan,problem;quadrature_order)
    free = findall(!,mesh.boundary_vertices)
    factor = isempty(free) ? nothing : cholesky(Symmetric(assembled.K[free,free]))
    primal = sdwr_solve_rhs(assembled,mesh,assembled.rhs,factor,free)
    U = reconstruct_fan(mesh,fan,primal.u)
    return (;mesh,fan,assembled,free,factor,primal,U)
end

function sdwr_evaluate(prepared,problem,goal;quadrature_order=6,localization=:separated)
    mesh,fan = prepared.mesh,prepared.fan
    dual_rhs = sdwr_goal_load(mesh,fan,prepared.U,goal)
    dual = sdwr_solve_rhs(prepared.assembled,mesh,dual_rhs,prepared.factor,prepared.free)
    estimate = sdwr_estimate(mesh,fan,prepared.primal,dual,problem,goal;
        quadrature_order,localization)
    reference = sdwr_goal_reference(problem,goal)
    reconstructed = sdwr_reconstructed_goal(mesh,fan,prepared.U,goal)
    error = reference.value-reconstructed.value
    denominator = norm(prepared.primal.u)*norm(dual.u)
    correlation = denominator > 0 ? dot(prepared.primal.u,dual.u)/denominator : NaN
    return (;dual,estimate,reference,reconstructed,error,correlation)
end

function sdwr_result_row(prepared,evaluated,problem,goal;
        strategy,cycle,n,theta,marked,balance_marked,quadrature_order,localization)
    mesh,fan = prepared.mesh,prepared.fan
    e = evaluated.estimate
    return (;problem=String(problem.name),goal=String(goal.name),strategy=String(strategy),
        cycle,base_n=n,cells=length(mesh.cells),vertices=size(mesh.points,2),
        free_dofs=length(prepared.free),tetrahedra=length(fan.volumes),
        minimum_level=minimum(leaf.level for leaf in mesh.leaves),
        maximum_level=maximum(leaf.level for leaf in mesh.leaves),
        exact_goal=evaluated.reference.value,reconstructed_goal=evaluated.reconstructed.value,
        signed_goal_error=evaluated.error,
        relative_goal_error=abs(evaluated.error/evaluated.reference.value),
        signed_estimate=e.signed,
        signed_effectivity=safe_ratio(e.signed,evaluated.error;scale=evaluated.reference.value),
        indicator_sum=e.magnitude,consistency=sum(e.cell_consistency),
        primal_term=sum(e.cell_primal),dual_term=sum(e.cell_dual),
        reference_uncertainty=evaluated.reference.uncertainty,
        primal_residual=prepared.primal.algebraic_residual,
        dual_residual=evaluated.dual.algebraic_residual,
        primal_dual_correlation=evaluated.correlation,
        theta,marked_cells=length(marked),balance_added=length(balance_marked)-length(marked),
        quadrature_order,localization=String(localization))
end

function sdwr_save_cells(path,prepared,evaluated,goal,marked)
    mesh = prepared.mesh
    marked_set = Set(marked)
    rows = [begin
        x = mesh.centers[:,cell]
        (;cell,x=x[1],y=x[2],z=x[3],h=mesh.sizes[cell],
            level=mesh.leaves[cell].level,vertices=length(mesh.cells[cell]),
            volume=mesh.volumes[cell],in_goal=goal.weight(x),
            signed_contribution=evaluated.estimate.cell_signed[cell],
            indicator=evaluated.estimate.indicators[cell],marked=cell in marked_set)
    end for cell in eachindex(mesh.cells)]
    write_csv(path,rows)
end

"""Accuracy-only run with fixed marking policy; exact truth never drives refinement.

Each adaptive goal starts from the same n=2 mesh. Uniform controls share one
primal solve per mesh, and all duals reuse its factorization. The intended
comparison axis is free VEM vertex DOFs, because adaptive faces add vertices.
No PU reconstruction problems are assembled or solved.
"""
function run_sdwr_adaptive(;problems=(:cube_smooth,:lshape_smooth,:lshape_singular),
        goals=(:integral_u,:integral_u2,:localized_u),strategies=(:uniform,:adaptive),
        uniform_levels=(2,4,6,8),base_n=2,cycles=7,max_cells=3000,
        theta=0.5,quadrature_order=6,localization=:separated,
        output=joinpath(@__DIR__,"..","results","sdwr_adaptive"))
    BLAS.set_num_threads(1)
    mkpath(output)
    rows = NamedTuple[]
    open(joinpath(output,"settings.toml"),"w") do io
        TOML.print(io,Dict("problems"=>collect(string.(problems)),"goals"=>collect(string.(goals)),
            "strategies"=>collect(string.(strategies)),"uniform_levels"=>collect(uniform_levels),
            "base_n"=>base_n,"cycles"=>cycles,"max_cells"=>max_cells,"theta"=>theta,
            "quadrature_order"=>quadrature_order,"localization"=>String(localization),
            "target"=>"J(u)-J(U), U=continuous fan reconstruction; no J_h conversion",
            "enrichment"=>"cellwise affine projections Pi_1 u_h and Pi_1 z_h only",
            "dual_rhs"=>"exact derivative of J(T u_h), positive adjoint convention",
            "refinement"=>"indicator bulk marking, octree children, face 2:1 balance",
            "production_includes"=>false,"patch_solves"=>false))
    end
    function record(prepared,evaluated,problem,goal;strategy,cycle,n,marked=Int[],balanced=Int[])
        row = sdwr_result_row(prepared,evaluated,problem,goal;strategy,cycle,n,theta,
            marked,balance_marked=balanced,quadrature_order,localization)
        push!(rows,row)
        write_csv(joinpath(output,"accuracy.csv"),rows)
        filename = "$(problem.name)_$(goal.name)_$(strategy)_$(cycle)_cells.csv"
        sdwr_save_cells(joinpath(output,filename),prepared,evaluated,goal,balanced)
        println("SDWR ",(;problem=problem.name,goal=goal.name,strategy,cycle,
            cells=row.cells,dofs=row.free_dofs,relative_error=row.relative_goal_error,
            effectivity=row.signed_effectivity,marked=row.marked_cells,balance=row.balance_added))
        flush(stdout)
    end
    for name in problems
        problem = manufactured_problem(name)
        if :uniform in strategies
            for (cycle,n) in enumerate(uniform_levels)
                prepared = sdwr_prepare_mesh(adaptive_initial_leaves(problem.domain,n),problem,n;
                    quadrature_order)
                for goal_name in goals
                    goal = sdwr_goal(goal_name,problem.domain)
                    evaluated = sdwr_evaluate(prepared,problem,goal;quadrature_order,localization)
                    record(prepared,evaluated,problem,goal;strategy=:uniform,cycle,n)
                end
            end
        end
        if :adaptive in strategies
            for goal_name in goals
                goal = sdwr_goal(goal_name,problem.domain)
                leaves = adaptive_initial_leaves(problem.domain,base_n)
                for cycle in 1:cycles
                    prepared = sdwr_prepare_mesh(leaves,problem,base_n;quadrature_order)
                    evaluated = sdwr_evaluate(prepared,problem,goal;quadrature_order,localization)
                    marked = sdwr_mark(evaluated.estimate.indicators;theta)
                    balanced = sdwr_balance_marked(prepared.mesh,marked)
                    record(prepared,evaluated,problem,goal;strategy=:adaptive,cycle,n=base_n,
                        marked,balanced)
                    cycle == cycles && break
                    length(leaves)+7length(balanced) > max_cells && break
                    leaves = adaptive_refine(leaves,balanced)
                end
            end
        end
    end
    return (;rows,output=abspath(output))
end

function sdwr_adaptive_cli(args)
    options = Dict{String,String}()
    for argument in args
        startswith(argument,"--") && occursin('=',argument) || error("Use --key=value")
        key,value = split(argument[3:end],'=';limit=2)
        key in ("problems","goals","strategies","uniform-levels","base-n","cycles",
            "max-cells","theta","quadrature","localization","output") || error("Unknown option $key")
        options[key] = value
    end
    symbols(key,default) = Symbol.(split(get(options,key,default),','))
    return run_sdwr_adaptive(
        problems=symbols("problems","cube_smooth,lshape_smooth,lshape_singular"),
        goals=symbols("goals","integral_u,integral_u2,localized_u"),
        strategies=symbols("strategies","uniform,adaptive"),
        uniform_levels=parse.(Int,split(get(options,"uniform-levels","2,4,6,8"),',')),
        base_n=parse(Int,get(options,"base-n","2")),cycles=parse(Int,get(options,"cycles","7")),
        max_cells=parse(Int,get(options,"max-cells","3000")),
        theta=parse(Float64,get(options,"theta","0.5")),
        quadrature_order=parse(Int,get(options,"quadrature","6")),
        localization=Symbol(get(options,"localization","separated")),
        output=get(options,"output",joinpath(@__DIR__,"..","results","sdwr_adaptive")))
end

if abspath(PROGRAM_FILE) == @__FILE__
    sdwr_adaptive_cli(ARGS)
end
