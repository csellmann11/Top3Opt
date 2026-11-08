isdefined(@__MODULE__, :investigation_case) ||
    include(joinpath(@__DIR__, "run_investigation.jl"))

"""Compare full P2 Galerkin enrichment with its globally coupled edge-only subspace.

The edge-only correction fixes every existing fan-P1 vertex value, including
interior vertices, while retaining all physically admissible P2 edge DOFs.
Both systems use global coupling. This isolates the fixed-vertex restriction;
it is not an upper bound for independent PU blends, which are piecewise cubic.
"""
function fixed_vertex_control(case)
    mesh, fan, data = case.mesh, case.fan, case.data
    system = coupled_p2_data(fan, data)
    full_free = cholesky(Symmetric(system.free_operator)) \ system.free_rhs
    full = coupled_p2_snapshot(mesh, fan, data, system, full_free;
        name=:full_p2, sweep_count=0)

    edge_positions = findall(dof -> dof > system.vertex_count, system.free_dofs)
    edge_dofs = system.free_dofs[edge_positions]
    edge_operator = system.K[edge_dofs, edge_dofs]
    edge_rhs = system.rhs[edge_dofs]
    edge_values = cholesky(Symmetric(edge_operator)) \ edge_rhs
    edge_free = zeros(length(system.free_dofs))
    edge_free[edge_positions] .= edge_values
    edge = coupled_p2_snapshot(mesh, fan, data, system, edge_free;
        name=:fixed_p1_vertices, sweep_count=0)
    edge_residual = norm(edge_rhs - edge_operator * edge_values) /
        max(norm(edge_rhs), eps(Float64))

    @assert full.relative_residual < 1e-10
    @assert edge_residual < 1e-10
    @assert all(iszero, edge.delta[1:system.vertex_count])
    @assert all(iszero, edge.delta[system.boundary_dofs])
    @assert isapprox(edge.correction_dwr, edge.correction_variational;
        rtol=1e-10, atol=1e-15)
    @assert edge.correction_variational <= full.correction_variational +
        1e-10 * max(abs(full.correction_variational), eps(Float64))
    return (; system, full, edge, edge_residual,
        edge_dofs=length(edge_dofs), edge_nnz=nnz(edge_operator))
end

function fixed_vertex_control_rows(case, result; quadrature_order, truth_order)
    system = result.system
    goal_error = case.problem.reference_energy - case.primal.energy
    exact_correction = sum(case.truth.reconstruction_error)
    rows = NamedTuple[]
    for (snapshot, dofs, entries, restricted_residual) in
            ((result.full, system.free_dof_count, system.operator_nnz, result.full.relative_residual),
             (result.edge, result.edge_dofs, result.edge_nnz, result.edge_residual))
        marking = marking_diagnostics(snapshot.estimate.indicators,
            case.truth.reconstruction_error)
        push!(rows, (; source=String(case.source), problem=String(case.problem.name),
            n=case.mesh.n, parent_cells=length(case.mesh.cells),
            tetrahedra=length(case.fan.volumes), p1_vertices=system.vertex_count,
            free_p1_vertices=count(dof -> dof <= system.vertex_count, system.free_dofs),
            full_p2_free_dofs=system.free_dof_count, method=String(snapshot.name),
            solve_dofs=dofs, solve_nnz=entries,
            exact_goal=case.problem.reference_energy, discrete_goal=case.primal.energy,
            goal_error, common_correction=snapshot.common_correction,
            exact_reconstruction_energy=exact_correction,
            estimated_correction=snapshot.correction_dwr,
            correction_effectivity=snapshot.correction_dwr / exact_correction,
            signed_estimate=snapshot.signed_dwr, effectivity=snapshot.signed_dwr / goal_error,
            indicator_sum=snapshot.estimate.magnitude,
            variational_estimate=snapshot.signed_variational,
            restricted_solve_relative_residual=restricted_residual,
            full_free_equation_relative_residual=snapshot.relative_residual,
            localization_identity_error=snapshot.localization_identity_error,
            quadrature_order, truth_order, marking...))
    end
    return rows
end

function run_fixed_vertex_control(; levels=(4, 6), amplitude=0.08,
        sources=(:cartesian_vem, :irregular_tet_fem, :refined_tet_fem),
        include_singular=true, quadrature_order=6,
        output=joinpath(@__DIR__, "..", "results", "selected_weighted_fixed_vertices", "accuracy.csv"))
    BLAS.set_num_threads(1)
    jobs = [(source, :cube_smooth, n) for source in sources for n in levels]
    include_singular && push!(jobs, (:irregular_tet_fem, :lshape_singular, 4))
    rows = NamedTuple[]
    mkpath(dirname(abspath(output)))
    for (source, name, n) in jobs
        truth_order = name == :cube_smooth ? 7 : 10
        println("FIXED_VERTEX preparing ", (; source, name, n))
        flush(stdout)
        case = investigation_case(source, name, n;
            amplitude, quadrature_order, truth_order)
        result = fixed_vertex_control(case)
        append!(rows, fixed_vertex_control_rows(case, result; quadrature_order, truth_order))
        write_csv(output, rows)
        println("FIXED_VERTEX_RESULT ", (; source, name, n,
            full_effectivity=rows[end-1].effectivity, fixed_vertex_effectivity=rows[end].effectivity,
            full_correction_effectivity=rows[end-1].correction_effectivity,
            fixed_vertex_correction_effectivity=rows[end].correction_effectivity,
            full_dofs=result.system.free_dof_count, edge_only_dofs=result.edge_dofs,
            restricted_residual=result.edge_residual))
        flush(stdout)
    end
    println("Fixed-vertex control: ", abspath(output))
    return rows
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_fixed_vertex_control()
end
