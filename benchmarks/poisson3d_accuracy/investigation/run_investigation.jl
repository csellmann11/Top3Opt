include("../load.jl")
include("coupled_enrichment.jl")
include("irregular_tetrahedra.jl")
include("oversampled_patches.jl")
include("two_level_correction.jl")
include("local_tetra_refinement.jl")

function investigation_case(source, name, n; amplitude, quadrature_order, truth_order)
    problem = manufactured_problem(name)
    if source == :cartesian_vem
        mesh = build_mesh(problem.domain,n)
        fan = build_fan(mesh)
        primal = solve_primal(mesh,fan,problem;quadrature_order)
        data = poisson_estimator_data(mesh,fan,primal,problem;quadrature_order)
        truth = exact_cell_diagnostics(mesh,fan,primal,problem;quadrature_order=truth_order)
        geometry = (;amplitude=0.0,minimum_orientation_ratio=1.0)
    elseif source in (:irregular_tet_fem,:refined_tet_fem)
        generated = irregular_tetrahedral_mesh(problem.domain,n;amplitude)
        if source == :refined_tet_fem
            generated = locally_refined_tetrahedral_mesh(generated)
        end
        mesh,fan,geometry = generated.mesh,generated.fan,generated.geometry_stats
        primal = solve_tetrahedral_p1(fan,problem;quadrature_order)
        data = tetrahedral_p1_estimator_data(mesh,fan,primal,problem;quadrature_order)
        truth = tetrahedral_p1_truth(mesh,fan,primal,problem;quadrature_order=truth_order)
    else
        throw(ArgumentError("Unknown primal source: $source"))
    end
    return (;source,problem,mesh,fan,primal,data,truth,geometry)
end

function investigation_row(case,goal,method,estimate,system;
        exact_goal=case.problem.reference_energy,discrete_goal=case.primal.energy,
        truth=case.truth,variational_estimate=NaN,algebraic_residual=NaN,
        sweeps=0,cycles=0,quadrature_order,truth_order)
    goal_error = exact_goal-discrete_goal
    spatial = truth === nothing ?
        (;distribution_l1=NaN,captured_error_fraction=NaN,
          optimal_captured_fraction=NaN,marked_cells=0) :
        marking_diagnostics(estimate.indicators,truth.reconstruction_error)
    return (;source=String(case.source),problem=String(case.problem.name),
        goal=String(goal),n=case.mesh.n,parent_cells=length(case.mesh.cells),
        tetrahedra=length(case.fan.volumes),primal_dofs=length(case.primal.free_dofs),
        perturbation=case.geometry.amplitude,
        minimum_orientation_ratio=case.geometry.minimum_orientation_ratio,
        minimum_scaled_jacobian=get(case.geometry,:minimum_scaled_jacobian,NaN),
        refined_tetrahedra=get(case.geometry,:refined_tetrahedra,0),
        method=String(method),exact_goal,discrete_goal,goal_error,
        relative_goal_error=abs(goal_error/exact_goal),
        signed_estimate=estimate.signed,
        effectivity=safe_ratio(estimate.signed,goal_error;scale=exact_goal),
        indicator_sum=estimate.magnitude,
        variational_estimate,
        variational_effectivity=safe_ratio(variational_estimate,goal_error;scale=exact_goal),
        global_enriched_dofs=system.free_dof_count,global_enriched_nnz=system.operator_nnz,
        local_dof_sum=get(estimate.statistics,:total_dofs,0),
        largest_local_solve=get(estimate.statistics,:max_dofs,0),
        buffer_layers=get(estimate.statistics,:buffer_layers,0),sweeps,cycles,
        coarse_dofs=get(estimate.statistics,:coarse_dofs,0),
        coarse_nnz=get(estimate.statistics,:coarse_nnz,0),
        algebraic_residual,
        primal_algebraic_residual=case.primal.algebraic_residual,
        exact_energy_integration_discrepancy=truth === nothing ? NaN :
            abs(truth.integrated_exact_energy-case.problem.reference_energy),
        quadrature_order,truth_order,spatial...)
end

function investigation_energy!(rows,case;buffers,sweeps,cycles,quadrature_order,truth_order)
    mesh,fan,data = case.mesh,case.fan,case.data
    patches = tetrahedral_patch_estimates(mesh,fan,data)
    coupled = coupled_p2_estimates(mesh,fan,data;sweep_counts=sweeps)
    system = coupled.assembly
    function add(method,estimate;kwargs...)
        push!(rows,investigation_row(case,:energy,method,estimate,system;
            quadrature_order,truth_order,kwargs...))
    end
    add(:full_pu,patches.full)
    add(:reduced_pu,patches.reduced)
    if case.source == :cartesian_vem
        add(:sdwr,poisson_projected_estimate(mesh,fan,case.primal,data))
    end
    neighbors,incidence = investigation_tet_neighbors(fan),investigation_dof_incidence(system)
    for layers in buffers
        estimate = oversampled_patch_estimate(mesh,fan,data,system;layers,neighbors,incidence)
        add(Symbol("buffer_",layers),estimate)
    end
    for snapshot in (coupled.direct,coupled.sweeps...)
        add(snapshot.name,snapshot.estimate;
            variational_estimate=snapshot.signed_variational,
            algebraic_residual=snapshot.relative_residual,sweeps=snapshot.sweep_count)
    end
    if !isempty(cycles)
        hierarchy = two_level_corrections(system;cycles)
        for state in hierarchy.snapshots
            name = Symbol("two_level_",state.cycle)
            snapshot = coupled_p2_snapshot(mesh,fan,data,system,state.free_delta;
                name,sweep_count=0)
            statistics = merge(snapshot.estimate.statistics,
                (;coarse_dofs=hierarchy.coarse_dofs,coarse_nnz=hierarchy.coarse_nnz))
            estimate = merge(snapshot.estimate,(;statistics))
            add(name,estimate;variational_estimate=snapshot.signed_variational,
                algebraic_residual=state.relative_residual,cycles=state.cycle)
        end
        system = merge(system,(;coarse_space=hierarchy.coarse_space))
    end
    return system
end

function investigation_integral(fan,U)
    return sum(fan.volumes[tet]*sum(U[fan.tet_nodes[:,tet]])/4
        for tet in eachindex(fan.volumes))
end

function investigation_p1_in_p2(system,U)
    values = zeros(system.total_dofs)
    values[1:system.vertex_count] .= U
    for (edge,(a,b)) in enumerate(system.edge_nodes)
        values[system.vertex_count+edge] = (U[a]+U[b])/2
    end
    return values
end

function investigation_linear_localization(mesh,fan,data,system,delta,base;
                                            statistics)
    nodal = zeros(size(fan.points,2))
    for tet in eachindex(fan.volumes)
        coefficients = delta[system.tet_dofs[:,tet]]
        for hat in 1:4
            nodal[fan.tet_nodes[hat,tet]] +=
                dot(view(data.weighted_residual,:,hat,tet),coefficients)
        end
    end
    correction = base .* mesh.volumes ./ sum(mesh.volumes)
    return poisson_collect_estimate(mesh,fan,nodal,correction,
        zeros(Int,length(nodal)),zeros(Int,length(nodal)),statistics)
end

function investigation_linear_base(mesh,fan,estimate,base)
    correction = base .* mesh.volumes ./ sum(mesh.volumes)
    return poisson_collect_estimate(mesh,fan,estimate.nodal,correction,
        estimate.patchdofs,estimate.ranks,estimate.statistics)
end

"""Independent linear goal: solve the dual -Delta(z)=1, not the primal again.

The reported discrete goal is integral(U), so no goal projection defect is
hidden. Retain R_U(Z) explicitly: VEM fan reconstructions are not automatically
Galerkin-orthogonal. Its global value is a separate consistency term distributed
by volume. Linear-goal indicators are exported, but no claim is made that the
primal energy-error distribution is their exact goal-oriented spatial truth.
"""
function investigation_linear!(rows,case,primal_system;
        buffers,sweeps,cycles,quadrature_order,truth_order)
    name = case.problem.name
    name in (:cube_smooth,:lshape_smooth) || return
    exact_goal = name == :cube_smooth ? 1/216 : 1/96
    mesh,fan,data = case.mesh,case.fan,case.data
    dual_problem = merge(case.problem,(;rhs=x->1.0))
    if case.source == :cartesian_vem
        dual_primal = solve_primal(mesh,fan,dual_problem;quadrature_order)
        dual_data = poisson_estimator_data(mesh,fan,dual_primal,dual_problem;quadrature_order)
    else
        dual_primal = solve_tetrahedral_p1(fan,dual_problem;quadrature_order)
        dual_data = tetrahedral_p1_estimator_data(mesh,fan,dual_primal,dual_problem;quadrature_order)
    end
    system = coupled_p2_data(fan,dual_data)
    system.tet_dofs == primal_system.tet_dofs || error("Primal/dual P2 maps differ")
    Z = investigation_p1_in_p2(system,dual_data.U)
    base = dot(primal_system.rhs,Z)
    discrete_goal = investigation_integral(fan,data.U)
    function add(method,estimate;kwargs...)
        push!(rows,investigation_row(case,:integral_u,method,estimate,system;
            exact_goal,discrete_goal,truth=nothing,quadrature_order,truth_order,kwargs...))
    end
    patches = tetrahedral_patch_estimates(mesh,fan,data;
        rhs_data=dual_data,readout_data=data,factor=1.0,include_corrections=false)
    add(:full_pu,investigation_linear_base(mesh,fan,patches.full,base))
    add(:reduced_pu,investigation_linear_base(mesh,fan,patches.reduced,base))
    neighbors,incidence = investigation_tet_neighbors(fan),investigation_dof_incidence(system)
    for layers in buffers
        raw = oversampled_patch_estimate(mesh,fan,data,system;
            layers,factor=1.0,include_corrections=false,neighbors,incidence)
        add(Symbol("buffer_",layers),investigation_linear_base(mesh,fan,raw,base))
    end
    direct = cholesky(Symmetric(system.free_operator)) \ system.free_rhs
    counts = sort(unique(collect(sweeps)))
    iterate,residual = zeros(length(direct)),copy(system.free_rhs)
    completed = 0
    for count in (0,counts...)
        if count == 0
            free_delta = direct
        else
            for _ in completed+1:count
                coupled_p2_symmetric_sweep!(iterate,residual,system.free_operator,system.diagonal)
            end
            completed = count
            residual .= system.free_rhs-system.free_operator*iterate
            free_delta = iterate
        end
        delta = zeros(system.total_dofs)
        delta[system.free_dofs] .= free_delta
        relative_residual = norm(system.free_rhs-system.free_operator*free_delta) /
            max(norm(system.free_rhs),eps(Float64))
        statistics = (;total_dofs=system.free_dof_count,max_dofs=system.free_dof_count,
            patch_count=0,global_dofs=system.total_dofs,operator_nnz=system.operator_nnz)
        estimate = investigation_linear_localization(mesh,fan,data,system,delta,base;statistics)
        expected = base+dot(primal_system.rhs,delta)
        @assert isapprox(estimate.signed,expected;atol=1e-12*exact_goal,rtol=1e-11)
        if count == 0
            # Independent load-side identity for the same P2 Galerkin fields.
            primal_delta = zeros(primal_system.total_dofs)
            primal_delta[primal_system.free_dofs] =
                cholesky(Symmetric(primal_system.free_operator)) \ primal_system.free_rhs
            enriched_load = zeros(system.total_dofs)
            for tet in eachindex(fan.volumes), shape in 1:10
                enriched_load[system.tet_dofs[shape,tet]] +=
                    fan.volumes[tet]*(shape <= 4 ? -1/20 : 1/5)
            end
            @assert isapprox(estimate.signed,dot(enriched_load,primal_delta);
                atol=1e-11*exact_goal,rtol=1e-10)
        end
        add(count == 0 ? :direct : Symbol("sgs_",count),estimate;
            algebraic_residual=relative_residual,sweeps=count)
    end
    if !isempty(cycles)
        hierarchy = two_level_corrections(system;cycles,
            coarse_space=get(primal_system,:coarse_space,nothing))
        for state in hierarchy.snapshots
            statistics = (;total_dofs=system.free_dof_count,max_dofs=system.free_dof_count,
                patch_count=0,global_dofs=system.total_dofs,operator_nnz=system.operator_nnz,
                coarse_dofs=hierarchy.coarse_dofs,coarse_nnz=hierarchy.coarse_nnz)
            estimate = investigation_linear_localization(mesh,fan,data,system,state.delta,base;
                statistics)
            @assert isapprox(estimate.signed,base+dot(primal_system.rhs,state.delta);
                atol=1e-12*exact_goal,rtol=1e-11)
            add(Symbol("two_level_",state.cycle),estimate;
                algebraic_residual=state.relative_residual,cycles=state.cycle)
        end
    end
    return nothing
end

function run_investigation(;levels=(2,4),
        cases=(:cube_smooth,:lshape_smooth,:lshape_singular),
        sources=(:cartesian_vem,:irregular_tet_fem),buffers=(1,),
        sweeps=(1,2,4,8),cycles=(1,2,4),amplitude=0.08,quadrature_order=6,truth_order=10,
        output=joinpath(@__DIR__,"..","results","investigation"))
    BLAS.set_num_threads(1)
    mkpath(output)
    rows = NamedTuple[]
    open(joinpath(output,"settings.toml"),"w") do io
        TOML.print(io,Dict("levels"=>collect(levels),"cases"=>collect(string.(cases)),
            "sources"=>collect(string.(sources)),"buffer_layers"=>collect(buffers),
            "sweeps"=>collect(sweeps),"cycles"=>collect(cycles),"perturbation"=>amplitude,
            "quadrature_order"=>quadrature_order,"truth_order"=>truth_order,
            "production_includes"=>false,"runtime_comparison"=>false,
            "irregular_control"=>"conforming tetrahedral P1 FEM; not polyhedral VEM",
            "refined_control"=>"every seventh tetrahedron split at its interior barycenter",
            "linear_goal"=>"integral(u); independent dual rhs=1; retain R_U(Z)"))
    end
    for source in sources, name in cases, n in levels
        println("INVESTIGATING ",(;source,name,n)); flush(stdout)
        case = investigation_case(source,name,n;amplitude,quadrature_order,truth_order)
        start = length(rows)+1
        system = investigation_energy!(rows,case;buffers,sweeps,cycles,quadrature_order,truth_order)
        investigation_linear!(rows,case,system;buffers,sweeps,cycles,quadrature_order,truth_order)
        write_csv(joinpath(output,"accuracy.csv"),rows)
        for goal in (:energy,:integral_u)
            recent = filter(row->row.goal==String(goal),rows[start:end])
            isempty(recent) && continue
            println("INVESTIGATION ",(;source,name,n,goal,
                methods=[row.method for row in recent],
                effectivities=[row.effectivity for row in recent]))
        end
        flush(stdout)
    end
    return (;rows,output=abspath(output))
end

function investigation_cli(args)
    options = Dict{String,String}()
    allowed = ("levels","cases","sources","buffers","sweeps","cycles","amplitude",
        "quadrature","truth-order","output")
    for argument in args
        startswith(argument,"--") && occursin('=',argument) ||
            error("Use --levels=2,4 --cases=cube_smooth --sources=cartesian_vem,irregular_tet_fem")
        key,value = split(argument[3:end],'=';limit=2)
        key in allowed || error("Unknown option: $key")
        options[key] = value
    end
    integers(key,default) = parse.(Int,split(get(options,key,default),','))
    symbols(key,default) = Symbol.(split(get(options,key,default),','))
    return run_investigation(
        levels=integers("levels","2,4"),
        cases=symbols("cases","cube_smooth,lshape_smooth,lshape_singular"),
        sources=symbols("sources","cartesian_vem,irregular_tet_fem"),
        buffers=integers("buffers","1"),sweeps=integers("sweeps","1,2,4,8"),
        cycles=integers("cycles","1,2,4"),
        amplitude=parse(Float64,get(options,"amplitude","0.08")),
        quadrature_order=parse(Int,get(options,"quadrature","6")),
        truth_order=parse(Int,get(options,"truth-order","10")),
        output=get(options,"output",joinpath(@__DIR__,"..","results","investigation")))
end

if abspath(PROGRAM_FILE) == @__FILE__
    investigation_cli(ARGS)
end
