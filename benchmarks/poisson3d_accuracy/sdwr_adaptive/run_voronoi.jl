include("run.jl")
include("voronoi_prisms.jl")
include("cut_goal_quadrature.jl")
include("geometric_goals.jl")

function sdwr_prepare_prism(problem,n;layers=n,quadrature_order=6,smooth=true)
    mesh = sdwr_voronoi_prism_mesh(problem.domain,n;layers,smooth)
    fan = adaptive_build_fan(mesh)
    assembled = adaptive_assemble(mesh,fan,problem;quadrature_order)
    free = findall(!,mesh.boundary_vertices)
    factor = isempty(free) ? nothing : cholesky(Symmetric(assembled.K[free,free]))
    primal = sdwr_solve_rhs(assembled,mesh,assembled.rhs,factor,free)
    U = reconstruct_fan(mesh,fan,primal.u)
    return (;mesh,fan,assembled,free,factor,primal,U)
end

function sdwr_evaluate_prism(prepared,problem,goal;quadrature_order=6,
        localization=:separated,goal_rules=sdwr_geometric_goal_rules(prepared.fan,goal))
    mesh,fan = prepared.mesh,prepared.fan
    dual_rhs = sdwr_geometric_goal_load(mesh,fan,prepared.U,goal,goal_rules)
    dual = sdwr_solve_rhs(prepared.assembled,mesh,dual_rhs,prepared.factor,prepared.free)
    estimate = sdwr_estimate(mesh,fan,prepared.primal,dual,problem,goal;
        quadrature_order,localization,goal_rules)
    reference = sdwr_goal_reference(problem,goal)
    reconstructed = sdwr_geometric_goal(mesh,fan,prepared.U,goal,goal_rules)
    error = reference.value-reconstructed.value
    denominator = norm(prepared.primal.u)*norm(dual.u)
    correlation = denominator > 0 ? dot(prepared.primal.u,dual.u)/denominator : NaN
    return (;dual,estimate,reference,reconstructed,error,correlation,goal_rules)
end

"""Non-nested h-sequence from the actual production Voronoi/extrusion generator.

Only geometry is imported from production. The scalar Poisson operator,
reconstruction, stabilization scaling, and sDWR proxy remain those of the
isolated benchmark, allowing a mesh-type comparison. No PUDWR solve is used.
"""
function run_sdwr_voronoi(;problems=(:cube_smooth,:lshape_smooth,:lshape_singular),
        goals=(:integral_u,:integral_u2,:localized_u),levels=(2,4,6,8,12),
        quadrature_order=6,localization=:separated,smooth=true,
        output=joinpath(@__DIR__,"..","results","sdwr_voronoi"))
    BLAS.set_num_threads(1)
    mkpath(output)
    rows = NamedTuple[]
    open(joinpath(output,"settings.toml"),"w") do io
        TOML.print(io,Dict("problems"=>collect(string.(problems)),"goals"=>collect(string.(goals)),
            "levels"=>collect(levels),"extrusion_layers"=>collect(levels),
            "quadrature_order"=>quadrature_order,"localization"=>String(localization),
            "smooth"=>smooth,"target"=>"J(u)-J(U); no J_h conversion",
            "enrichment"=>"cellwise affine projections Pi_1 u_h and Pi_1 z_h",
            "mesh"=>"production deterministic 2D Voronoi generator and 3D prism extrusion",
            "sequence"=>"regenerated non-nested h-refinement, n layers and n seeds per unit length",
            "operator"=>"standalone P1-consistent face-fan VEM-style Poisson operator",
            "stabilization"=>"alpha=volume^(1/3), agrees with alpha=h on Cartesian cubes",
            "localized_goal"=>"exact tetrahedron/box clipping and degree-2 moment integration",
            "production_mesh_generator"=>true,"production_solver"=>false,"patch_solves"=>false))
    end
    for name in problems,n in levels
        problem = manufactured_problem(name)
        println("VORONOI_PREPARING ",(;name,n)); flush(stdout)
        prepared = sdwr_prepare_prism(problem,n;quadrature_order,smooth)
        mesh,fan = prepared.mesh,prepared.fan
        for goal_name in goals
            goal = sdwr_goal(goal_name,problem.domain)
            evaluated = sdwr_evaluate_prism(prepared,problem,goal;quadrature_order,localization)
            e = evaluated.estimate
            cut_tets = evaluated.goal_rules === nothing ? 0 : count(rule->rule.cut,evaluated.goal_rules)
            row = (;problem=String(name),goal=String(goal_name),strategy="voronoi_prism",
                n,layers=n,cells=length(mesh.cells),vertices=size(mesh.points,2),
                free_dofs=length(prepared.free),tetrahedra=length(fan.volumes),
                minimum_vertices=minimum(length,mesh.cells),maximum_vertices=maximum(length,mesh.cells),
                minimum_volume=minimum(mesh.volumes),maximum_volume=maximum(mesh.volumes),
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
                primal_dual_correlation=evaluated.correlation,cut_goal_tetrahedra=cut_tets,
                quadrature_order,localization=String(localization))
            push!(rows,row)
            write_csv(joinpath(output,"accuracy.csv"),rows)
            println("SDWR_VORONOI ",(;name,goal=goal_name,n,cells=row.cells,dofs=row.free_dofs,
                relative_error=row.relative_goal_error,effectivity=row.signed_effectivity,cut_tets))
            flush(stdout)
        end
    end
    return (;rows,output=abspath(output))
end

function sdwr_voronoi_cli(args)
    options = Dict{String,String}()
    for argument in args
        startswith(argument,"--") && occursin('=',argument) || error("Use --key=value")
        key,value = split(argument[3:end],'=';limit=2)
        key in ("problems","goals","levels","quadrature","localization","smooth","output") ||
            error("Unknown option $key")
        options[key] = value
    end
    symbols(key,default) = Symbol.(split(get(options,key,default),','))
    return run_sdwr_voronoi(
        problems=symbols("problems","cube_smooth,lshape_smooth,lshape_singular"),
        goals=symbols("goals","integral_u,integral_u2,localized_u"),
        levels=parse.(Int,split(get(options,"levels","2,4,6,8,12"),',')),
        quadrature_order=parse(Int,get(options,"quadrature","6")),
        localization=Symbol(get(options,"localization","separated")),
        smooth=parse(Bool,get(options,"smooth","true")),
        output=get(options,"output",joinpath(@__DIR__,"..","results","sdwr_voronoi")))
end

if abspath(PROGRAM_FILE) == @__FILE__
    sdwr_voronoi_cli(ARGS)
end
