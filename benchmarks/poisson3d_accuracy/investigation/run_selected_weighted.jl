include("run_investigation.jl")
include("selected_weighted_patches.jl")

"""Algebraic diagnostics; incompatibility is reported, never called an exact solve."""
function selected_diagnostic_summary(diagnostics)
    isempty(diagnostics) && return (;selection_patches=0,singular_patches=0,
        incompatible_patches=0,combined_relative_patch_residual=NaN,
        maximum_relative_patch_residual=NaN,maximum_absolute_patch_residual=NaN,
        maximum_relative_readout_null=NaN,minimum_retained_singular_ratio=NaN)
    rhs_squared = sum(row.rhs_norm^2 for row in diagnostics)
    residual_squared = sum(row.residual_norm^2 for row in diagnostics)
    ratios = [row.smallest_retained_ratio for row in diagnostics
        if isfinite(row.smallest_retained_ratio) && row.smallest_retained_ratio > 0]
    return (;selection_patches=length(diagnostics),
        singular_patches=count(row->row.nullity>0,diagnostics),
        incompatible_patches=count(row->row.relative_residual>1e-8,diagnostics),
        combined_relative_patch_residual=sqrt(residual_squared/max(rhs_squared,eps(Float64)^2)),
        maximum_relative_patch_residual=maximum(row.relative_residual for row in diagnostics),
        maximum_absolute_patch_residual=maximum(row.residual_norm for row in diagnostics),
        maximum_relative_readout_null=maximum(row.relative_readout_null_norm for row in diagnostics),
        minimum_retained_singular_ratio=isempty(ratios) ? NaN : minimum(ratios))
end

function selected_signed_cells(mesh,fan,estimate)
    signed = copy(estimate.correction)
    for node in eachindex(estimate.nodal)
        cells = fan.node_cells[node]
        volume = sum(mesh.volumes[cell] for cell in cells)
        for cell in cells
            signed[cell] += estimate.nodal[node]*mesh.volumes[cell]/volume
        end
    end
    @assert isapprox(sum(signed),estimate.signed;atol=1e-13,rtol=1e-10)
    return signed
end

function selected_record!(rows,cellrows,case,goal,method,estimate,system;
        diagnostics=NamedTuple[],rank_tolerance=NaN,common_correction,
        quadrature_order,truth_order,kwargs...)
    row = investigation_row(case,goal,method,estimate,system;
        quadrature_order,truth_order,kwargs...)
    push!(rows,merge(row,(;rank_tolerance,common_correction),
        selected_diagnostic_summary(diagnostics)))
    signed_cells = selected_signed_cells(case.mesh,case.fan,estimate)
    for cell in eachindex(signed_cells)
        push!(cellrows,(;source=String(case.source),problem=String(case.problem.name),
            n=case.mesh.n,goal=String(goal),method=String(method),rank_tolerance,cell,
            signed_contribution=signed_cells[cell],indicator=estimate.indicators[cell],
            reconstruction_energy_error=goal==:energy ? case.truth.reconstruction_error[cell] : NaN))
    end
    return nothing
end

function selected_save_diagnostics!(patchrows,case,goal,tolerance,diagnostics)
    prefix = (;source=String(case.source),problem=String(case.problem.name),
        n=case.mesh.n,goal=String(goal),rank_tolerance=tolerance)
    append!(patchrows,[merge(prefix,row) for row in diagnostics])
end

function selected_energy!(rows,cellrows,patchrows,case;
        tolerances,quadrature_order,truth_order)
    mesh,fan,data = case.mesh,case.fan,case.data
    patches = tetrahedral_patch_estimates(mesh,fan,data)
    coupled = coupled_p2_estimates(mesh,fan,data;sweep_counts=())
    system = coupled.assembly
    common_correction = sum(data.conversion)+sum(data.correction)
    function record(method,estimate;kwargs...)
        selected_record!(rows,cellrows,case,:energy,method,estimate,system;
            common_correction,quadrature_order,truth_order,kwargs...)
    end
    record(:full_pu,patches.full)
    record(:reduced_pu,patches.reduced)
    case.source == :cartesian_vem &&
        record(:sdwr,poisson_projected_estimate(mesh,fan,case.primal,data))
    record(:direct,coupled.direct.estimate;
        algebraic_residual=coupled.direct.relative_residual,
        variational_estimate=coupled.direct.signed_variational)
    for tolerance in tolerances
        selected = selected_weighted_patch_estimates(mesh,fan,data;rank_tolerance=tolerance)
        selected_save_diagnostics!(patchrows,case,:energy,tolerance,selected.diagnostics)
        for name in propertynames(selected.estimates)
            record(Symbol("weighted_",name),getproperty(selected.estimates,name);
                diagnostics=selected.diagnostics,rank_tolerance=tolerance)
        end
    end
    return system
end

"""Separate dual forcing for integral(u); preserve R_U(Z) on VEM reconstructions."""
function selected_linear!(rows,cellrows,patchrows,case,primal_system;
        tolerances,quadrature_order,truth_order)
    name = case.problem.name
    name in (:cube_smooth,:lshape_smooth) || return
    mesh,fan,data = case.mesh,case.fan,case.data
    problem = merge(case.problem,(;rhs=x->1.0))
    if case.source == :cartesian_vem
        primal = solve_primal(mesh,fan,problem;quadrature_order)
        dual_data = poisson_estimator_data(mesh,fan,primal,problem;quadrature_order)
    else
        primal = solve_tetrahedral_p1(fan,problem;quadrature_order)
        dual_data = tetrahedral_p1_estimator_data(mesh,fan,primal,problem;quadrature_order)
    end
    system = coupled_p2_data(fan,dual_data)
    system.tet_dofs == primal_system.tet_dofs || error("Primal/dual numbering mismatch")
    base = dot(primal_system.rhs,investigation_p1_in_p2(system,dual_data.U))
    exact_goal = name == :cube_smooth ? 1/216 : 1/96
    discrete_goal = investigation_integral(fan,data.U)
    function record(method,estimate;kwargs...)
        selected_record!(rows,cellrows,case,:integral_u,method,estimate,system;
            common_correction=base,quadrature_order,truth_order,
            exact_goal,discrete_goal,truth=nothing,kwargs...)
    end
    patches = tetrahedral_patch_estimates(mesh,fan,data;
        rhs_data=dual_data,readout_data=data,factor=1.0,include_corrections=false)
    record(:full_pu,investigation_linear_base(mesh,fan,patches.full,base))
    record(:reduced_pu,investigation_linear_base(mesh,fan,patches.reduced,base))
    delta = zeros(system.total_dofs)
    delta[system.free_dofs] = cholesky(Symmetric(system.free_operator)) \ system.free_rhs
    statistics = (;total_dofs=system.free_dof_count,max_dofs=system.free_dof_count,
        patch_count=0,global_dofs=system.total_dofs,operator_nnz=system.operator_nnz)
    direct = investigation_linear_localization(mesh,fan,data,system,delta,base;statistics)
    # Reciprocity against a separately assembled integral load, including the base term.
    primal_delta = zeros(primal_system.total_dofs)
    primal_delta[primal_system.free_dofs] =
        cholesky(Symmetric(primal_system.free_operator)) \ primal_system.free_rhs
    integral_load = zeros(system.total_dofs)
    for tet in eachindex(fan.volumes),shape in 1:10
        integral_load[system.tet_dofs[shape,tet]] +=
            fan.volumes[tet]*(shape <= 4 ? -1/20 : 1/5)
    end
    @assert isapprox(direct.signed,dot(integral_load,primal_delta);
        rtol=1e-10,atol=1e-11*exact_goal)
    record(:direct,direct;algebraic_residual=
        norm(system.free_rhs-system.free_operator*delta[system.free_dofs])/
        max(norm(system.free_rhs),eps(Float64)))
    for tolerance in tolerances
        selected = selected_weighted_patch_estimates(mesh,fan,dual_data;
            readout_data=data,factor=1.0,include_corrections=false,rank_tolerance=tolerance)
        selected_save_diagnostics!(patchrows,case,:integral_u,tolerance,selected.diagnostics)
        for selection in propertynames(selected.estimates)
            raw = getproperty(selected.estimates,selection)
            estimate = investigation_linear_base(mesh,fan,raw,base)
            record(Symbol("weighted_",selection),estimate;
                diagnostics=selected.diagnostics,rank_tolerance=tolerance)
        end
    end
    return nothing
end

function run_selected_weighted(;levels=(4,6),
        cases=(:cube_smooth,:lshape_singular),
        sources=(:cartesian_vem,:irregular_tet_fem,:refined_tet_fem),
        tolerances=(1e-11,),amplitude=0.08,quadrature_order=6,truth_order=10,
        output=joinpath(@__DIR__,"..","results","selected_weighted"))
    BLAS.set_num_threads(1)
    mkpath(output)
    rows,cellrows,patchrows = NamedTuple[],NamedTuple[],NamedTuple[]
    open(joinpath(output,"settings.toml"),"w") do io
        TOML.print(io,Dict("levels"=>collect(levels),"cases"=>collect(string.(cases)),
            "sources"=>collect(string.(sources)),"rank_tolerances"=>collect(tolerances),
            "amplitude"=>amplitude,"quadrature_order"=>quadrature_order,
            "truth_order"=>truth_order,"old_p1_vertex_values"=>"fixed",
            "artificial_boundary_edge_values"=>"free",
            "physical_boundary_edge_values"=>"fixed",
            "incompatible_rhs"=>"Rank-truncated Euclidean least squares; original residual reported",
            "total_energy"=>"Dirichlet strain energy, without body-load potential",
            "runtime_comparison"=>false,"production_includes"=>false))
    end
    for source in sources, name in cases, n in levels
        println("SELECTED_WEIGHTED preparing ",(;source,name,n)); flush(stdout)
        case = investigation_case(source,name,n;amplitude,quadrature_order,truth_order)
        first_row = length(rows)+1
        system = selected_energy!(rows,cellrows,patchrows,case;
            tolerances,quadrature_order,truth_order)
        selected_linear!(rows,cellrows,patchrows,case,system;
            tolerances,quadrature_order,truth_order)
        write_csv(joinpath(output,"accuracy.csv"),rows)
        write_csv(joinpath(output,"cells.csv"),cellrows)
        write_csv(joinpath(output,"patches.csv"),patchrows)
        for goal in (:energy,:integral_u)
            recent = filter(row->row.goal==String(goal),rows[first_row:end])
            isempty(recent) && continue
            println("SELECTED_WEIGHTED ",(;source,name,n,goal,
                methods=[row.method for row in recent],
                tolerances=[row.rank_tolerance for row in recent],
                effectivities=[row.effectivity for row in recent]))
        end
        flush(stdout)
    end
    return (;rows,output=abspath(output))
end

function selected_weighted_cli(args)
    options = Dict{String,String}()
    allowed = ("levels","cases","sources","tolerances","amplitude","quadrature","truth-order","output")
    for argument in args
        startswith(argument,"--") && occursin('=',argument) ||
            error("Arguments must use --key=value")
        key,value = split(argument[3:end],'=';limit=2)
        key in allowed || error("Unknown option: $key")
        options[key] = value
    end
    integer_list(key,default) = parse.(Int,split(get(options,key,default),','))
    symbol_list(key,default) = Symbol.(split(get(options,key,default),','))
    return run_selected_weighted(levels=integer_list("levels","4,6"),
        cases=symbol_list("cases","cube_smooth,lshape_singular"),
        sources=symbol_list("sources","cartesian_vem,irregular_tet_fem,refined_tet_fem"),
        tolerances=parse.(Float64,split(get(options,"tolerances","1e-11"),',')),
        amplitude=parse(Float64,get(options,"amplitude","0.08")),
        quadrature_order=parse(Int,get(options,"quadrature","6")),
        truth_order=parse(Int,get(options,"truth-order","10")),
        output=get(options,"output",joinpath(@__DIR__,"..","results","selected_weighted")))
end

if abspath(PROGRAM_FILE) == @__FILE__
    selected_weighted_cli(ARGS)
end
