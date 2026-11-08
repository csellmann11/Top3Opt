# Include ../load.jl before this file. These helpers use only the isolated lab.

"""Assemble a globally conforming P2 residual problem on the existing fan.

All fan vertices and shared edge midpoints are global DOFs. Only physical
Dirichlet DOFs are constrained; there are no artificial patch boundaries.
The full operator/map is retained for independent boundary-lifting checks.
"""
function coupled_p2_data(fan, data)
    nv, nt = size(fan.points, 2), length(fan.volumes)
    edge_ids = Dict{Tuple{Int,Int},Int}()
    edge_nodes = Tuple{Int,Int}[]
    tet_dofs = Matrix{Int}(undef, 10, nt)
    for tet in 1:nt
        ids = view(fan.tet_nodes, :, tet)
        tet_dofs[1:4, tet] .= ids
        for (edge, (a, b)) in enumerate(POISSON_P2_EDGES)
            key = poisson_ordered_edge(ids[a], ids[b])
            id = get(edge_ids, key, 0)
            if id == 0
                push!(edge_nodes, key)
                id = nv + length(edge_nodes)
                edge_ids[key] = id
            end
            tet_dofs[4 + edge, tet] = id
        end
    end
    ndofs = nv + length(edge_nodes)
    coordinates = Matrix{Float64}(undef, 3, ndofs)
    coordinates[:, 1:nv] .= fan.points
    boundary = falses(ndofs)
    boundary[1:nv] .= poisson_boundary_mask(fan)
    for (edge, (a, b)) in enumerate(edge_nodes)
        coordinates[:, nv + edge] .= (fan.points[:, a] + fan.points[:, b]) / 2
        boundary[nv + edge] = (a, b) in fan.boundary_edges
    end

    rows, columns = Vector{Int}(undef, 100nt), Vector{Int}(undef, 100nt)
    entries = Vector{Float64}(undef, 100nt)
    rhs = zeros(ndofs)
    position = 0
    for tet in 1:nt
        ids = view(tet_dofs, :, tet)
        @inbounds for i in 1:10
            rhs[ids[i]] += data.residual[i, tet]
        end
        @inbounds for j in 1:10, i in 1:10
            position += 1
            rows[position], columns[position] = ids[i], ids[j]
            entries[position] = data.stiffness[tet][i, j]
        end
    end
    K = sparse(rows, columns, entries, ndofs, ndofs)
    dropzeros!(K)
    free_dofs = findall(!, boundary)
    boundary_dofs = findall(boundary)
    free_operator, free_rhs = K[free_dofs, free_dofs], rhs[free_dofs]
    diagonal = diag(free_operator)
    all(d -> isfinite(d) && d > 0, diagonal) ||
        error("The free P2 operator must have positive finite diagonal entries")
    return (; K, rhs, free_operator, free_rhs, diagonal, free_dofs,
        boundary_dofs, boundary_mask=boundary, coordinates, tet_dofs,
        edge_nodes, vertex_count=nv, total_dofs=ndofs,
        free_dof_count=length(free_dofs), operator_nnz=nnz(free_operator),
        full_operator_nnz=nnz(K))
end

"""One forward/backward coordinate-energy minimization sweep.

The maintained residual is b-A*x. Each coordinate update has exact energy
gain residual[j]^2/(2A[j,j]); no damping parameter or independent patch solve
is used. One symmetric Gauss-Seidel sweep contains two coordinate passes.
"""
function coupled_p2_symmetric_sweep!(x, residual, A, diagonal)
    for order in (eachindex(x), reverse(eachindex(x)))
        @inbounds for j in order
            increment = residual[j] / diagonal[j]
            x[j] += increment
            for position in nzrange(A, j)
                residual[A.rowval[position]] -= A.nzval[position] * increment
            end
        end
    end
    return x
end

"""Evaluate one global P2 correction using the existing DWR localization.

`correction_dwr = R_U(delta)/2` is the existing energy-goal DWR substitution.
`correction_variational = R_U(delta)-a(delta,delta)/2` is a separate quadratic
energy diagnostic: with exact forms, its gap to a(e,e)/2 is a(e-delta,e-delta)/2.
That lower-bound identity is not asserted for generic goals or inexact load
quadrature. Spatial indicators below localize only the DWR expression.
"""
function coupled_p2_snapshot(mesh, fan, data, assembled, free_delta;
                             name, sweep_count)
    delta = zeros(assembled.total_dofs)
    delta[assembled.free_dofs] .= free_delta
    applied = assembled.free_operator * free_delta
    residual = assembled.free_rhs - applied
    rhs_work = dot(assembled.free_rhs, free_delta)
    energy_norm2 = dot(free_delta, applied)
    correction_dwr = rhs_work / 2
    correction_variational = rhs_work - energy_norm2 / 2
    B = sum(data.conversion) + sum(data.correction)
    residual_norm = norm(residual)
    relative_residual = residual_norm / max(norm(assembled.free_rhs), eps(Float64))
    # Boundary entries are reactions; equilibrium is checked only on free DOFs.
    global_residual = assembled.rhs - assembled.K * delta

    nodal = copy(data.conversion)
    for tet in eachindex(fan.volumes)
        ids = view(fan.tet_nodes, :, tet)
        local_delta = delta[view(assembled.tet_dofs, :, tet)]
        for hat in 1:4
            nodal[ids[hat]] += dot(view(data.weighted_residual, :, hat, tet), local_delta) / 2
        end
    end
    statistics = (; total_dofs=assembled.free_dof_count,
        max_dofs=assembled.free_dof_count, patch_count=0,
        global_dofs=assembled.total_dofs, operator_nnz=assembled.operator_nnz,
        full_operator_nnz=assembled.full_operator_nnz, name, sweep_count,
        residual_norm, relative_residual,
        convention="globally conforming P2 correction with physical Dirichlet constraints")
    nn = length(nodal)
    estimate = poisson_collect_estimate(mesh, fan, nodal, data.correction,
        zeros(Int, nn), zeros(Int, nn), statistics)
    localization_identity_error = abs(estimate.signed - (B + correction_dwr))
    return (; name, sweep_count, delta, correction_dwr, correction_variational,
        signed_dwr=estimate.signed, signed_variational=B + correction_variational,
        common_correction=B, rhs_work, energy_norm2, residual_norm,
        relative_residual, global_residual, localization_identity_error, estimate)
end

"""Direct P2 reference and coupled symmetric Gauss-Seidel checkpoints.

Returns `(assembly, direct, sweeps)`. Every snapshot includes its global
correction vector, both global energy-goal estimates, and `.estimate` in the
existing lab indicator format. Checkpoints share one sequence starting at
zero. No runtime ranking is attached; DOF and matrix-entry counts are reported.
"""
function coupled_p2_estimates(mesh, fan, data; sweep_counts=(1, 2, 4, 8))
    counts = sort!(unique!(collect(Int, sweep_counts)))
    all(>(0), counts) || throw(ArgumentError("Sweep counts must be positive integers"))
    assembly = coupled_p2_data(fan, data)
    direct_delta = isempty(assembly.free_dofs) ? Float64[] :
        cholesky(Symmetric(assembly.free_operator)) \ assembly.free_rhs
    direct = coupled_p2_snapshot(mesh, fan, data, assembly, direct_delta;
        name=:direct, sweep_count=0)
    sweeps = NamedTuple[]
    x, residual = zeros(length(assembly.free_rhs)), copy(assembly.free_rhs)
    completed = 0
    for count in counts
        for _ in completed + 1:count
            coupled_p2_symmetric_sweep!(x, residual, assembly.free_operator, assembly.diagonal)
        end
        completed = count
        # Refresh at checkpoints to prevent accumulated floating-point residual drift.
        residual .= assembly.free_rhs - assembly.free_operator * x
        push!(sweeps, coupled_p2_snapshot(mesh, fan, data, assembly, x;
            name=Symbol("sgs_", count), sweep_count=count))
    end
    return (; assembly, direct, sweeps)
end
