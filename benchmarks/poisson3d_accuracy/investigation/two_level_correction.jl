# Include ../load.jl and coupled_enrichment.jl before this file.

"""Exact homogeneous P1-to-P2 interpolation and its Galerkin coarse operator.

P1 vertices are shared with the P2 mesh; an edge midpoint receives the average
of its endpoint values. Physical Dirichlet values are zero in correction space.
This construction uses mesh connectivity, not a polynomial fit. The coarse
operator is the fan's conforming P1 stiffness, not a Cartesian VEM operator.
"""
function two_level_space(system)
    coarse_vertices = [dof for dof in system.free_dofs if dof <= system.vertex_count]
    coarse_index = zeros(Int, system.vertex_count)
    for (column, vertex) in enumerate(coarse_vertices)
        coarse_index[vertex] = column
    end
    rows, columns, values = Int[], Int[], Float64[]
    for (row, dof) in enumerate(system.free_dofs)
        if dof <= system.vertex_count
            push!(rows, row); push!(columns, coarse_index[dof]); push!(values, 1.0)
        else
            a, b = system.edge_nodes[dof - system.vertex_count]
            for vertex in (a, b)
                column = coarse_index[vertex]
                column == 0 && continue
                push!(rows, row); push!(columns, column); push!(values, 0.5)
            end
        end
    end
    P = sparse(rows, columns, values, system.free_dof_count, length(coarse_vertices))
    K = transpose(P) * system.free_operator * P
    dropzeros!(K)
    all(value -> isfinite(value) && value > 0, diag(K)) ||
        error("The P1 coarse stiffness must have positive finite diagonal")
    factorization = isempty(coarse_vertices) ? nothing : cholesky(Symmetric(K))
    return (; prolongation=P, operator=K, factorization, coarse_vertices,
        coarse_dofs=length(coarse_vertices), coarse_nnz=nnz(K),
        prolongation_nnz=nnz(P))
end

"""A coordinate pass updates the full fine-grid residual after every change."""
function two_level_coordinate_pass!(x, residual, A, diagonal, order)
    @inbounds for column in order
        increment = residual[column] / diagonal[column]
        x[column] += increment
        for position in nzrange(A, column)
            residual[A.rowval[position]] -= A.nzval[position] * increment
        end
    end
    return x
end

"""Forward smoothing, exact P1 coarse correction, and backward smoothing.

All three stages minimize the same positive-definite quadratic energy over
their respective correction subspaces. This gives a monotone variational
energy correction, not an automatic bound for a generic linear-goal DWR readout.
"""
function two_level_cycle!(x, residual, system, coarse_space)
    A = system.free_operator
    two_level_coordinate_pass!(x, residual, A, system.diagonal, eachindex(x))
    if coarse_space.coarse_dofs > 0
        P = coarse_space.prolongation
        coarse_correction = coarse_space.factorization \ (transpose(P) * residual)
        fine_correction = P * coarse_correction
        x .+= fine_correction
        residual .-= A * fine_correction
    end
    two_level_coordinate_pass!(x, residual, A, system.diagonal, reverse(eachindex(x)))
    return x
end

"""Coupled P2 corrections with a reusable global P1 coarse solve.

`rhs` may be a primal or dual residual on the free P2 DOFs. Passing a previously
constructed `coarse_space` reuses the same interpolation and factorization for
additional right-hand sides. Checkpoints all belong to one run from zero.

This clarity-first prototype makes no universal mesh-independent convergence
claim. Coarse setup/factorization and iteration work must both be measured when
assessing frequently changing meshes.
"""
function two_level_corrections(system; cycles=(1, 2, 4), rhs=system.free_rhs,
                                 coarse_space=nothing)
    length(rhs) == system.free_dof_count || throw(DimensionMismatch("Free P2 right-hand side"))
    counts = sort!(unique!(collect(Int, cycles)))
    all(>(0), counts) || throw(ArgumentError("Cycle counts must be positive"))
    coarse = isnothing(coarse_space) ? two_level_space(system) : coarse_space
    size(coarse.prolongation, 1) == system.free_dof_count ||
        throw(DimensionMismatch("Coarse space does not match the free P2 space"))
    x, residual = zeros(length(rhs)), copy(rhs)
    snapshots = NamedTuple[]
    completed = 0
    for count in counts
        for _ in completed + 1:count
            two_level_cycle!(x, residual, system, coarse)
        end
        completed = count
        applied = system.free_operator * x
        residual .= rhs - applied
        delta = zeros(system.total_dofs)
        delta[system.free_dofs] .= x
        push!(snapshots, (; cycle=count, free_delta=copy(x), delta,
            relative_residual=norm(residual) / max(norm(rhs), eps(Float64)),
            residual_norm=norm(residual),
            variational_correction=dot(rhs, x) - dot(x, applied) / 2))
    end
    return (; coarse_dofs=coarse.coarse_dofs, coarse_nnz=coarse.coarse_nnz,
        prolongation_nnz=coarse.prolongation_nnz, coarse_space=coarse, snapshots)
end

"""Focused independent checks using an irregular tetrahedral P1 control.

Pass its fan, primal P1 solution, and a `coupled_p2_data` system. This checks the
coarse operator against independent P1 assembly, energy monotonicity, and
convergence to all quadratic terms with exact nonhomogeneous physical data.
"""
function validate_two_level_correction(fan, primal, system)
    coarse = two_level_space(system)
    p1_operator = primal.K[coarse.coarse_vertices, coarse.coarse_vertices]
    coarse_operator_relative_error = norm(coarse.operator-p1_operator) /
        max(norm(p1_operator), eps(Float64))
    @assert coarse_operator_relative_error < 2e-12 "P1/P2 Galerkin coarse operators disagree"
    trial = two_level_corrections(system; cycles=(1, 2, 4, 8, 16), coarse_space=coarse)
    direct = cholesky(Symmetric(system.free_operator)) \ system.free_rhs
    best_energy = dot(system.free_rhs, direct) / 2
    energies = [0.0; [state.variational_correction for state in trial.snapshots]]
    tolerance = 1e-11 * max(abs(best_energy), eps(Float64))
    @assert all(diff(energies) .>= -tolerance) "Two-level energy decreased"
    @assert maximum(energies) <= best_energy + tolerance "Energy exceeded exact P2 optimum"

    polynomial(x) = 0.4 + 0.7x[1] - 0.3x[2] + 0.2x[3] +
        x[1]^2 + 2x[2]^2 + 3x[3]^2 +
        0.5x[1]*x[2] - 0.4x[1]*x[3] + 0.6x[2]*x[3]
    exact_values = [polynomial(view(system.coordinates, :, dof))
        for dof in 1:system.total_dofs]
    load = zeros(system.total_dofs)
    # Independent exact moments for f=-12, rather than the existing residual.
    for tet in eachindex(fan.volumes), shape in 1:10
        load[system.tet_dofs[shape, tet]] +=
            fan.volumes[tet] * (shape <= 4 ? 0.6 : -2.4)
    end
    free, boundary = system.free_dofs, system.boundary_dofs
    quadratic_rhs = load[free] - system.K[free, boundary] * exact_values[boundary]
    quadratic_cycles = (1, 2, 4, 8, 16, 32, 64)
    quadratic = two_level_corrections(system; cycles=quadratic_cycles,
        rhs=quadratic_rhs, coarse_space=coarse)
    quadratic_errors = [maximum(abs, state.free_delta-exact_values[free])
        for state in quadratic.snapshots]
    @assert last(quadratic_errors) < 2e-10 "Quadratic correction did not converge: $quadratic_errors"
    @assert all(iszero, last(quadratic.snapshots).delta[boundary])
    return (; coarse_operator_relative_error, coarse_dofs=coarse.coarse_dofs,
        coarse_nnz=coarse.coarse_nnz, fine_dofs=system.free_dof_count,
        residuals=[state.relative_residual for state in trial.snapshots],
        quadratic_cycles, quadratic_errors, variational_corrections=energies[2:end])
end

# Explicit standalone validation; including this file performs no solves.
if abspath(PROGRAM_FILE) == @__FILE__
    include(joinpath(@__DIR__, "..", "load.jl"))
    include(joinpath(@__DIR__, "irregular_tetrahedra.jl"))
    include(joinpath(@__DIR__, "coupled_enrichment.jl"))
    BLAS.set_num_threads(1)
    problem = manufactured_problem(:cube_smooth)
    control = irregular_tetrahedral_mesh(:cube, 2; amplitude=0.08)
    primal = solve_tetrahedral_p1(control.fan, problem)
    data = tetrahedral_p1_estimator_data(control.mesh, control.fan, primal, problem)
    system = coupled_p2_data(control.fan, data)
    println("TWO_LEVEL_VALIDATION ",
        validate_two_level_correction(control.fan, primal, system))
end
