include("load.jl")

function run_accuracy(;levels=(2,4),
        cases=(:cube_smooth,:lshape_smooth,:lshape_singular),quadrature_order=6,
        truth_order=10,output=joinpath(@__DIR__,"results","latest"))
    mkpath(output)
    BLAS.set_num_threads(1)
    rows,comparisons = NamedTuple[],NamedTuple[]
    open(joinpath(output,"settings.toml"),"w") do io
        TOML.print(io,Dict("julia_version"=>string(VERSION),"levels"=>collect(levels),
            "cases"=>string.(collect(cases)),"quadrature_order"=>quadrature_order,
            "truth_order"=>truth_order,"goal"=>"J(u)=1/2 integral |gradient u|^2",
            "boundary_conditions"=>"homogeneous Dirichlet on every physical boundary",
            "runtime_comparison"=>false,"production_includes"=>false,
            "sdwr"=>"linear projected enrichment with full load/goal corrections",
            "primal"=>"scalar first-order hexahedral VEM; alpha=h; projected body load"))
    end
    for name in cases, n in levels
        problem = manufactured_problem(name)
        mesh = build_mesh(problem.domain,n)
        fan = build_fan(mesh)
        primal = solve_primal(mesh,fan,problem;quadrature_order)
        estimates = estimate_errors(mesh,fan,primal,problem;quadrature_order)
        truth = exact_cell_diagnostics(mesh,fan,primal,problem;quadrature_order=truth_order)
        for method in (:full,:reduced,:sdwr)
            push!(rows,accuracy_row(problem,mesh,primal,truth,method,getproperty(estimates,method);
                quadrature_order,truth_order))
        end
        full,reduced = estimates.full,estimates.reduced
        push!(comparisons,(;problem=String(name),n,cells=length(mesh.cells),
            relative_signed_change=safe_ratio(abs(reduced.signed-full.signed),abs(full.signed);
                scale=problem.reference_energy),
            relative_indicator_change=safe_ratio(sum(abs,reduced.indicators-full.indicators),
                sum(full.indicators);scale=problem.reference_energy)))
        stem = "$(name)_n$(n)"
        write_cell_results(joinpath(output,stem*"_cells.csv"),mesh,primal,truth,estimates)
        write_node_results(joinpath(output,stem*"_nodes.csv"),fan,estimates)
        write_cell_vtk(joinpath(output,stem*".vtk"),mesh,primal,truth,estimates)
        # Write after each case so partial progress remains inspectable.
        write_csv(joinpath(output,"summary.csv"),rows)
        write_csv(joinpath(output,"comparison.csv"),comparisons)
        write_accuracy_report(joinpath(output,"report.md"),rows,comparisons)
        recent = rows[end-2:end]
        println("ACCURACY ",(;problem=name,n,cells=length(mesh.cells),
            relative_goal_error=recent[1].relative_goal_error,
            effectivities=Tuple(row.signed_effectivity for row in recent)))
        flush(stdout)
    end
    println("Accuracy results: ",abspath(output))
    return (;rows,comparisons,output=abspath(output))
end

function accuracy_cli(args)
    options = Dict{String,String}()
    for argument in args
        startswith(argument,"--") && occursin('=',argument) ||
            error("Use --levels=2,4,6 --cases=cube_smooth,lshape_smooth,lshape_singular --quadrature=6 --truth-order=10 --output=PATH")
        key,value = split(argument[3:end],'=';limit=2)
        key in ("levels","cases","quadrature","truth-order","output") || error("Unknown option: $key")
        options[key] = value
    end
    return run_accuracy(;levels=parse.(Int,split(get(options,"levels","4,6,8,10"),',')),
        cases=Symbol.(split(get(options,"cases","cube_smooth,lshape_smooth,lshape_singular"),',')),
        quadrature_order=parse(Int,get(options,"quadrature","6")),
        truth_order=parse(Int,get(options,"truth-order","10")),
        output=get(options,"output",joinpath(@__DIR__,"results","latest")))
end

if abspath(PROGRAM_FILE) == @__FILE__
    accuracy_cli(ARGS)
end
