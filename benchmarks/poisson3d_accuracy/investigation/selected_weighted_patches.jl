isdefined(@__MODULE__, :poisson_estimator_data) ||
    include(joinpath(@__DIR__, "..", "load.jl"))

@inline function selected_weighted_cubic_mean(a, b, c)
    a == b == c && return 1 / 20
    (a == b || a == c || b == c) && return 1 / 60
    return 1 / 120
end

"""Exact a(phi_trial,N_hat*phi_test) for two tetrahedral P2 edge shapes."""
@inline function selected_weighted_edge_entry(gram, volume, hat, test, trial)
    c, d = POISSON_P2_EDGES[test]
    a, b = POISSON_P2_EDGES[trial]
    moment = selected_weighted_cubic_mean
    return 16volume * (
        moment(a, hat, c) * gram[b, d] + moment(a, hat, d) * gram[b, c] +
        moment(a, c, d) * gram[b, hat] + moment(b, hat, c) * gram[a, d] +
        moment(b, hat, d) * gram[a, c] + moment(b, c, d) * gram[a, hat])
end

"""Assemble the proposed weighted patch problem with free artificial-edge traces.

All existing P1 vertex values are fixed. Corrections use every P2 edge midpoint
in the star except physical Dirichlet edges. A[test,trial]=a(phi_trial,N_i phi_test)
is integrated by exact cubic barycentric moments, without patch quadrature.
`b` is the weighted solve residual; `g` is the independently selectable DWR
readout residual. `internal[j]=a(U_rhs,phi_j)` belongs to the solved field.
"""
function selected_weighted_patch_system(fan, rhs_data, node; readout_data=rhs_data)
    tets = fan.stars[node]
    edges = sort!(unique([poisson_ordered_edge(fan.tet_nodes[a, tet], fan.tet_nodes[b, tet])
        for tet in tets for (a, b) in POISSON_P2_EDGES]))
    filter!(edge -> !(edge in fan.boundary_edges), edges)
    edge_rows = Dict(edge => row for (row, edge) in enumerate(edges))
    ndofs = length(edges)
    A, K = zeros(ndofs, ndofs), zeros(ndofs, ndofs)
    b, g, internal = zeros(ndofs), zeros(ndofs), zeros(ndofs)
    rows = zeros(Int, 6)
    for tet in tets
        ids, G = view(fan.tet_nodes, :, tet), fan.gradients[tet]
        hat = findfirst(==(node), ids)
        gram, volume = transpose(G) * G, fan.volumes[tet]
        gradient = G * rhs_data.U[ids]
        internal_projection = transpose(G) * gradient
        for (edge, (a, c)) in enumerate(POISSON_P2_EDGES)
            rows[edge] = get(edge_rows, poisson_ordered_edge(ids[a], ids[c]), 0)
        end
        for test in 1:6
            row = rows[test]
            row == 0 && continue
            b[row] += rhs_data.weighted_residual[4 + test, hat, tet]
            g[row] += readout_data.weighted_residual[4 + test, hat, tet]
            a, c = POISSON_P2_EDGES[test]
            internal[row] += volume * (internal_projection[a] + internal_projection[c])
            for trial in 1:6
                column = rows[trial]
                column == 0 && continue
                A[row, column] += selected_weighted_edge_entry(gram, volume, hat, test, trial)
                K[row, column] += rhs_data.stiffness[tet][4 + test, 4 + trial]
            end
        end
    end
    return (; A, b, g, K, internal, edges, tets, node, edge_rows)
end

function selected_weighted_state(system, coefficients)
    residual = system.A * coefficients - system.b
    correction_energy = dot(coefficients, system.K * coefficients) / 2
    return (; coefficients, residual, residual_norm=norm(residual),
        relative_residual=norm(residual) / max(norm(system.b), eps(Float64)),
        coefficient_norm=norm(coefficients), correction_energy,
        total_dirichlet_energy_change=correction_energy + dot(system.internal, coefficients),
        readout=dot(system.g, coefficients))
end

"""Choose three representatives of the same truncated-SVD solution family.

The original RHS and original DWR readout are retained, including incompatibility.
This is the original least-squares solution family if discarded modes are
genuinely null. Discarding nonzero modes instead regularizes the operator;
the original residual is still evaluated for every representative.
Euclidean least squares and coefficient norms use the standard, unscaled P2
edge basis; no row or column equilibration changes the selection convention.
Minimum total energy means Dirichlet/strain energy a(U+delta,U+delta)/2;
it is NOT Poisson potential energy, which would additionally contain -F(delta).
K is positive definite because the correction vanishes at every P1 vertex.
"""
function selected_weighted_representatives(system; rank_tolerance=1e-11)
    0 < rank_tolerance < 1 || throw(ArgumentError("rank_tolerance must lie in (0,1)"))
    ndofs = length(system.b)
    if ndofs == 0
        empty_state = selected_weighted_state(system, Float64[])
        return (; minimum_coefficient=empty_state, minimum_correction_energy=empty_state,
            minimum_total_energy=empty_state, rank=0, nullity=0,
            right_nullspace=zeros(0, 0), left_nullspace=zeros(0, 0),
            projected_rhs=Float64[], singular_values=Float64[],
            smallest_retained_ratio=NaN, condition_estimate=NaN,
            rhs_norm=0.0, residual_norm=0.0, relative_residual=0.0,
            readout_null_norm=0.0, relative_readout_null_norm=0.0)
    end
    cholesky(Symmetric(system.K)) # Verify energy positivity, rather than silently regularizing it.
    decomposition = svd(system.A; full=true)
    largest = maximum(decomposition.S)
    rank = iszero(largest) ? 0 : count(s -> s > rank_tolerance * largest, decomposition.S)
    Z = decomposition.V[:, rank+1:end]
    left_nullspace = decomposition.U[:, rank+1:end]
    if rank == 0
        particular, projected_rhs = zeros(ndofs), zeros(ndofs)
    else
        left_coefficients = transpose(decomposition.U[:, 1:rank]) * system.b
        particular = decomposition.V[:, 1:rank] * (left_coefficients ./ decomposition.S[1:rank])
        projected_rhs = decomposition.U[:, 1:rank] * left_coefficients
    end
    correction, total = copy(particular), copy(particular)
    if size(Z, 2) > 0
        null_energy = cholesky(Symmetric(transpose(Z) * system.K * Z))
        energy_gradient = system.K * particular
        correction .-= Z * (null_energy \ (transpose(Z) * energy_gradient))
        total .-= Z * (null_energy \ (transpose(Z) * (energy_gradient + system.internal)))
    end
    compatibility = norm(system.b - projected_rhs)
    readout_null_norm = norm(transpose(Z) * system.g)
    smallest_retained_ratio = rank == 0 ? 0.0 : decomposition.S[rank] / largest
    return (; minimum_coefficient=selected_weighted_state(system, particular),
        minimum_correction_energy=selected_weighted_state(system, correction),
        minimum_total_energy=selected_weighted_state(system, total),
        rank, nullity=ndofs-rank, right_nullspace=Z, left_nullspace,
        projected_rhs, singular_values=decomposition.S, smallest_retained_ratio,
        condition_estimate=rank == 0 ? Inf : inv(smallest_retained_ratio),
        rhs_norm=norm(system.b), residual_norm=compatibility,
        relative_residual=compatibility / max(norm(system.b), eps(Float64)),
        readout_null_norm,
        relative_readout_null_norm=readout_null_norm / max(norm(system.g), eps(Float64)))
end

"""Evaluate selected weighted-patch reconstructions with unchanged PU localization.

Returns `(estimates, diagnostics, coefficients)`. Diagnostics have scalar CSV
fields only. Optional retained coefficients contain the edge map per patch.
For a generic linear-goal experiment, supply dual rhs_data, primal readout_data,
factor=1, and include_corrections=false; the coarse residual remains the caller's
responsibility. No sampling fit, artificial Dirichlet edge data, or rank fallback
is introduced.
"""
function selected_weighted_patch_estimates(mesh, fan, rhs_data;
        readout_data=rhs_data, factor=0.5, include_corrections=true,
        rank_tolerance=1e-11, retain_coefficients=false)
    selectors = (:minimum_coefficient, :minimum_correction_energy, :minimum_total_energy)
    nn = length(rhs_data.U)
    base = include_corrections ? readout_data.conversion : zeros(nn)
    correction = include_corrections ? readout_data.correction : zeros(length(mesh.cells))
    nodal = [copy(base) for _ in selectors]
    counts, ranks = zeros(Int, nn), zeros(Int, nn)
    diagnostics = NamedTuple[]
    retained = retain_coefficients ? Vector{NamedTuple}(undef, nn) : nothing
    for node in 1:nn
        system = selected_weighted_patch_system(fan, rhs_data, node; readout_data)
        selected = selected_weighted_representatives(system; rank_tolerance)
        counts[node], ranks[node] = length(system.b), selected.rank
        states = map(name -> getproperty(selected, name), selectors)
        for method in eachindex(selectors)
            nodal[method][node] += factor * states[method].readout
        end
        coefficient, energy, total = states
        push!(diagnostics, (; node, dofs=counts[node], rank=selected.rank, nullity=selected.nullity,
            rhs_norm=selected.rhs_norm, residual_norm=selected.residual_norm,
            relative_residual=selected.relative_residual,
            readout_null_norm=selected.readout_null_norm,
            relative_readout_null_norm=selected.relative_readout_null_norm,
            smallest_retained_ratio=selected.smallest_retained_ratio,
            condition_estimate=selected.condition_estimate,
            coefficient_norm_minimum_coefficient=coefficient.coefficient_norm,
            coefficient_norm_minimum_correction_energy=energy.coefficient_norm,
            coefficient_norm_minimum_total_energy=total.coefficient_norm,
            correction_energy_minimum_coefficient=coefficient.correction_energy,
            correction_energy_minimum_correction_energy=energy.correction_energy,
            correction_energy_minimum_total_energy=total.correction_energy,
            readout_minimum_coefficient=coefficient.readout,
            readout_minimum_correction_energy=energy.readout,
            readout_minimum_total_energy=total.readout,
            residual_norm_minimum_coefficient=coefficient.residual_norm,
            residual_norm_minimum_correction_energy=energy.residual_norm,
            residual_norm_minimum_total_energy=total.residual_norm,
            maximum_residual_change=max(norm(energy.residual-coefficient.residual),
                norm(total.residual-coefficient.residual))))
        if retain_coefficients
            retained[node] = (; node, edges=system.edges,
                minimum_coefficient=coefficient.coefficients,
                minimum_correction_energy=energy.coefficients,
                minimum_total_energy=total.coefficients)
        end
    end
    common_stats = (; total_dofs=sum(counts), max_dofs=maximum(counts), patch_count=nn,
        total_retained_rank=sum(ranks), singular_patches=count(i -> ranks[i] < counts[i], 1:nn),
        maximum_relative_residual=maximum(row.relative_residual for row in diagnostics),
        maximum_relative_readout_null_norm=maximum(row.relative_readout_null_norm for row in diagnostics),
        rank_tolerance, convention="free artificial-edge P2 patches with weighted tests and SVD selection")
    values = map(eachindex(selectors)) do method
        statistics = merge(common_stats, (; selector=selectors[method]))
        poisson_collect_estimate(mesh, fan, nodal[method], correction,
            copy(counts), copy(ranks), statistics)
    end
    estimates = NamedTuple{selectors}(Tuple(values))
    return (; estimates, diagnostics, coefficients=retained)
end
