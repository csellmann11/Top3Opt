include("run_voronoi.jl")

function sdwr_prepare_existing_prism(mesh,problem;quadrature_order=6)
    fan = adaptive_build_fan(mesh)
    assembled = adaptive_assemble(mesh,fan,problem;quadrature_order)
    free = findall(!,mesh.boundary_vertices)
    factor = isempty(free) ? nothing : cholesky(Symmetric(assembled.K[free,free]))
    primal = sdwr_solve_rhs(assembled,mesh,assembled.rhs,factor,free)
    U = reconstruct_fan(mesh,fan,primal.u)
    return (;mesh,fan,assembled,free,factor,primal,U)
end

"""Indicator-driven local refinement through the production prism topology.

Starts from the production Voronoi extrusion. The library creates corner
children using edge/face/volume centers; descendants need not remain Voronoi
polygons. Coarse/fine interfaces are kept conforming by shared face fans.
"""
function run_sdwr_voronoi_adaptive(;
        problems=(:cube_smooth,:lshape_smooth,:lshape_singular),
        goals=(:integral_u,:integral_u2,:localized_u),base_n=2,cycles=7,
        max_cells=3000,theta=0.5,quadrature_order=6,localization=:separated,
        output=joinpath(@__DIR__,"..","results","sdwr_voronoi_adaptive"))
    BLAS.set_num_threads(1)
    mkpath(output)
    rows = NamedTuple[]
    open(joinpath(output,"settings.toml"),"w") do io
        TOML.print(io,Dict("problems"=>collect(string.(problems)),"goals"=>collect(string.(goals)),
            "base_n"=>base_n,"cycles"=>cycles,"max_cells"=>max_cells,"theta"=>theta,
            "quadrature_order"=>quadrature_order,"localization"=>String(localization),
            "target"=>"J(u)-J(U), affine projected sDWR only",
            "initial_mesh"=>"actual production 2D Voronoi + prism extrusion",
            "refinement"=>"production VEMGeo._refine!; corner children; no oracle marking",
            "max_cells"=>max_cells,"budget_estimate"=>"conservative current vertex count per marked cell",
            "operator"=>"standalone P1-consistent face-fan VEM-style Poisson operator",
            "localized_goal"=>"exact tetrahedron-box integration",
            "production_solver"=>false,"patch_solves"=>false))
    end
    for name in problems,goal_name in goals
        problem = manufactured_problem(name)
        goal = sdwr_goal(goal_name,problem.domain)
        mesh = sdwr_voronoi_prism_mesh(problem.domain,base_n)
        for cycle in 1:cycles
            prepared = sdwr_prepare_existing_prism(mesh,problem;quadrature_order)
            evaluated = sdwr_evaluate_prism(prepared,problem,goal;quadrature_order,localization)
            marked = sdwr_mark(evaluated.estimate.indicators;theta)
            row = sdwr_result_row(prepared,evaluated,problem,goal;
                strategy=:voronoi_adaptive,cycle,n=base_n,theta,
                marked,balance_marked=marked,quadrature_order,localization)
            push!(rows,row)
            write_csv(joinpath(output,"accuracy.csv"),rows)
            sdwr_save_cells(joinpath(output,"$(name)_$(goal_name)_adaptive_$(cycle)_cells.csv"),
                prepared,evaluated,goal,marked)
            println("SDWR_VORONOI_ADAPTIVE ",(;name,goal=goal_name,cycle,cells=row.cells,
                dofs=row.free_dofs,relative_error=row.relative_goal_error,
                effectivity=row.signed_effectivity,marked=length(marked)))
            flush(stdout)
            cycle == cycles && break
            # Every corner child has one original corner; hanging vertices
            # may make this a conservative overestimate of the next size.
            predicted = length(mesh.cells)+sum(length(mesh.cells[cell])-1 for cell in marked)
            predicted > max_cells && break
            mesh = sdwr_refine_prism_mesh(mesh,marked)
        end
    end
    return (;rows,output=abspath(output))
end

function sdwr_voronoi_adaptive_cli(args)
    options = Dict{String,String}()
    for argument in args
        startswith(argument,"--") && occursin('=',argument) || error("Use --key=value")
        key,value = split(argument[3:end],'=';limit=2)
        key in ("problems","goals","base-n","cycles","max-cells","theta","quadrature",
            "localization","output") || error("Unknown option $key")
        options[key] = value
    end
    symbols(key,default) = Symbol.(split(get(options,key,default),','))
    return run_sdwr_voronoi_adaptive(
        problems=symbols("problems","cube_smooth,lshape_smooth,lshape_singular"),
        goals=symbols("goals","integral_u,integral_u2,localized_u"),
        base_n=parse(Int,get(options,"base-n","2")),cycles=parse(Int,get(options,"cycles","7")),
        max_cells=parse(Int,get(options,"max-cells","3000")),theta=parse(Float64,get(options,"theta","0.5")),
        quadrature_order=parse(Int,get(options,"quadrature","6")),
        localization=Symbol(get(options,"localization","separated")),
        output=get(options,"output",joinpath(@__DIR__,"..","results","sdwr_voronoi_adaptive")))
end

if abspath(PROGRAM_FILE) == @__FILE__
    sdwr_voronoi_adaptive_cli(ARGS)
end
