include("run_selected_weighted.jl")
include("regularized_weighted_patches.jl")

function patch_regularization_summary(diagnostics)
    isempty(diagnostics) && return (;patch_count=0,
        combined_original_relative_residual=NaN,combined_energy_relative_residual=NaN,
        maximum_energy_gain=NaN,maximum_chosen_alpha=NaN,
        maximum_coefficient_norm=NaN,patch_correction_energy_sum=NaN,
        maximum_absolute_nodal_readout=NaN)
    original_rhs = sum(row.rhs_norm^2 for row in diagnostics)
    normalized_rhs = sum(row.normalized_rhs_norm^2 for row in diagnostics)
    return (;patch_count=length(diagnostics),
        combined_original_relative_residual=sqrt(sum(row.residual_norm^2 for row in diagnostics)/
            max(original_rhs,eps(Float64)^2)),
        combined_energy_relative_residual=sqrt(sum(row.normalized_residual_norm^2 for row in diagnostics)/
            max(normalized_rhs,eps(Float64)^2)),
        maximum_energy_gain=maximum(row.actual_gain for row in diagnostics),
        maximum_chosen_alpha=maximum(row.chosen_alpha for row in diagnostics),
        maximum_coefficient_norm=maximum(row.coefficient_norm for row in diagnostics),
        patch_correction_energy_sum=sum(row.correction_energy for row in diagnostics),
        maximum_absolute_nodal_readout=maximum(abs(row.readout) for row in diagnostics))
end

function patch_regularization_baseline_rows(rows)
    return [merge(row,(;normalization="baseline",filter="baseline",parameter=NaN),
        patch_regularization_summary(NamedTuple[])) for row in rows]
end

function patch_regularization_record!(rows,cellrows,case,goal,spec,estimate,system,diagnostics;
        quadrature_order,truth_order,kwargs...)
    row = investigation_row(case,goal,spec.name,estimate,system;
        quadrature_order,truth_order,kwargs...)
    @assert isfinite(row.effectivity) && all(isfinite,estimate.indicators)
    push!(rows,merge(row,(;normalization=String(spec.normalization),
        filter=String(spec.filter),parameter=spec.parameter),
        patch_regularization_summary(diagnostics)))
    signed_cells = selected_signed_cells(case.mesh,case.fan,estimate)
    for cell in eachindex(signed_cells)
        push!(cellrows,(;source=String(case.source),problem=String(case.problem.name),
            n=case.mesh.n,amplitude=case.geometry.amplitude,goal=String(goal),
            method=String(spec.name),cell,signed_contribution=signed_cells[cell],
            indicator=estimate.indicators[cell],
            reconstruction_energy_error=goal==:energy ? case.truth.reconstruction_error[cell] : NaN))
    end
end

function patch_regularization_goal!(rows,cellrows,case,goal,system,rhs_data;
        specs,output,base=0.0,quadrature_order,truth_order)
    energy = goal == :energy
    result = regularized_weighted_patch_estimates(case.mesh,case.fan,rhs_data;
        readout_data=case.data,factor=energy ? 0.5 : 1.0,
        include_corrections=energy,specs)
    filename = "$(case.source)_$(case.problem.name)_n$(case.mesh.n)_$(goal)_patches.csv"
    write_csv(joinpath(output,filename),result.diagnostics)
    for spec in specs
        raw = result.estimates[spec.name]
        diagnostics = filter(row->row.method==String(spec.name),result.diagnostics)
        if energy
            patch_regularization_record!(rows,cellrows,case,goal,spec,raw,system,diagnostics;
                quadrature_order,truth_order)
        else
            estimate = investigation_linear_base(case.mesh,case.fan,raw,base)
            exact_goal = case.problem.name == :cube_smooth ? 1/216 : 1/96
            discrete_goal = investigation_integral(case.fan,case.data.U)
            patch_regularization_record!(rows,cellrows,case,goal,spec,estimate,system,diagnostics;
                quadrature_order,truth_order,exact_goal,discrete_goal,truth=nothing)
        end
    end
end

function patch_regularization_case!(rows,case;specs,output,quadrature_order,truth_order)
    baselines,cellrows = NamedTuple[],NamedTuple[]
    system = investigation_energy!(baselines,case;buffers=(),sweeps=(),cycles=(),
        quadrature_order,truth_order)
    investigation_linear!(baselines,case,system;buffers=(),sweeps=(),cycles=(),
        quadrature_order,truth_order)
    append!(rows,patch_regularization_baseline_rows(baselines))
    patch_regularization_goal!(rows,cellrows,case,:energy,system,case.data;
        specs,output,quadrature_order,truth_order)
    if case.problem.name in (:cube_smooth,:lshape_smooth)
        dual_problem = merge(case.problem,(;rhs=x->1.0))
        if case.source == :cartesian_vem
            dual_primal = solve_primal(case.mesh,case.fan,dual_problem;quadrature_order)
            dual_data = poisson_estimator_data(case.mesh,case.fan,dual_primal,dual_problem;quadrature_order)
        else
            dual_primal = solve_tetrahedral_p1(case.fan,dual_problem;quadrature_order)
            dual_data = tetrahedral_p1_estimator_data(case.mesh,case.fan,dual_primal,dual_problem;quadrature_order)
        end
        base = dot(system.rhs,investigation_p1_in_p2(system,dual_data.U))
        patch_regularization_goal!(rows,cellrows,case,:integral_u,system,dual_data;
            specs,output,base,quadrature_order,truth_order)
    end
    filename = "$(case.source)_$(case.problem.name)_n$(case.mesh.n)_cells.csv"
    write_csv(joinpath(output,filename),cellrows)
    return nothing
end

function run_patch_regularization(;levels=(4,6),cases=(:cube_smooth,:lshape_singular),
        sources=(:cartesian_vem,:irregular_tet_fem,:refined_tet_fem),
        amplitude=0.08,quadrature_order=6,truth_order=10,
        specs=regularized_weighted_default_specs(),
        output=joinpath(@__DIR__,"..","results","patch_regularization"))
    BLAS.set_num_threads(1)
    mkpath(output)
    rows = NamedTuple[]
    open(joinpath(output,"settings.toml"),"w") do io
        TOML.print(io,Dict("levels"=>collect(levels),"cases"=>collect(string.(cases)),
            "sources"=>collect(string.(sources)),"amplitude"=>amplitude,
            "quadrature_order"=>quadrature_order,"truth_order"=>truth_order,
            "numerical_rank_floor"=>1e-12,"runtime_comparison"=>false,
            "truth_used_for_regularization"=>false,"production_includes"=>false,
            "methods"=>[Dict("name"=>String(spec.name),
                "normalization"=>String(spec.normalization),"filter"=>String(spec.filter),
                "parameter"=>spec.parameter) for spec in specs]))
    end
    for source in sources,name in cases,n in levels
        println("REGULARIZATION preparing ",(;source,name,n,amplitude)); flush(stdout)
        case = investigation_case(source,name,n;amplitude,quadrature_order,truth_order)
        start = length(rows)+1
        patch_regularization_case!(rows,case;specs,output,quadrature_order,truth_order)
        write_csv(joinpath(output,"accuracy.csv"),rows)
        for goal in (:energy,:integral_u)
            recent = filter(row->row.goal==String(goal),rows[start:end])
            isempty(recent) && continue
            println("REGULARIZATION ",(;source,name,n,goal,
                methods=[row.method for row in recent],
                effectivities=[row.effectivity for row in recent]))
        end
        flush(stdout)
    end
    return (;rows,output=abspath(output))
end

function patch_regularization_cli(args)
    options = Dict{String,String}()
    allowed = ("levels","cases","sources","amplitude","quadrature","truth-order","methods","output")
    for argument in args
        startswith(argument,"--") && occursin('=',argument) ||
            error("Use --key=value arguments")
        key,value = split(argument[3:end],'=';limit=2)
        key in allowed || error("Unknown option: $key")
        options[key] = value
    end
    integers(key,default) = parse.(Int,split(get(options,key,default),','))
    symbols(key,default) = Symbol.(split(get(options,key,default),','))
    specs = regularized_weighted_default_specs()
    if haskey(options,"methods")
        requested = symbols("methods","")
        all(name->any(spec->spec.name==name,specs),requested) || error("Unknown regularization method")
        specs = filter(spec->spec.name in requested,specs)
    end
    return run_patch_regularization(levels=integers("levels","4,6"),
        cases=symbols("cases","cube_smooth,lshape_singular"),
        sources=symbols("sources","cartesian_vem,irregular_tet_fem,refined_tet_fem"),
        amplitude=parse(Float64,get(options,"amplitude","0.08")),
        quadrature_order=parse(Int,get(options,"quadrature","6")),
        truth_order=parse(Int,get(options,"truth-order","10")),specs=specs,
        output=get(options,"output",joinpath(@__DIR__,"..","results","patch_regularization")))
end

if abspath(PROGRAM_FILE)==@__FILE__
    patch_regularization_cli(ARGS)
end
