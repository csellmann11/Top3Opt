using LinearAlgebra, SparseArrays

"""Perturb a conforming fan mesh while preserving its physical boundary.

The returned `mesh` retains parent-cell labels only for reporting and indicator
aggregation. Its cells are unions of the perturbed tetrahedra, not Cartesian
VEM cells; do not pass it to the Cartesian VEM primal functions. All PDE
assembly below uses the actual tetrahedral coordinates and gradients.
"""
function irregular_tetrahedral_mesh(domain, n; amplitude=0.08)
    0 <= amplitude <= 0.15 || throw(ArgumentError("Use amplitude in [0,0.15]"))
    base_mesh = build_mesh(domain, n)
    base_fan = build_fan(base_mesh)
    points = copy(base_fan.points)
    h = 1.0 / n
    for node in axes(points, 2)
        base_fan.boundary_nodes[node] && continue
        for component in 1:3
            # Fixed deterministic displacements; no random state or fitted data.
            direction = sin(1.731 * node + 2.317 * component) *
                        cos(0.917 * node - 1.113 * component)
            points[component, node] += amplitude * h * direction
        end
    end
    nt = length(base_fan.volumes)
    gradients = Vector{Matrix{Float64}}(undef, nt)
    volumes = zeros(nt)
    parent_volumes = zeros(length(base_mesh.cells))
    parent_centers = zeros(3, length(base_mesh.cells))
    minimum_orientation_ratio = Inf
    minimum_scaled_jacobian = Inf
    for tet in 1:nt
        ids = base_fan.tet_nodes[:, tet]
        original = base_fan.points[:, ids]
        moved = points[:, ids]
        J0 = original[:, 2:4] .- original[:, 1]
        J = moved[:, 2:4] .- moved[:, 1]
        ratio = det(J) / det(J0)
        isfinite(ratio) && ratio > 0 ||
            error("Perturbation inverted tetrahedron $tet; reduce amplitude")
        minimum_orientation_ratio = min(minimum_orientation_ratio, ratio)
        scaled = abs(det(J)) / prod(norm(view(J, :, j)) for j in 1:3)
        minimum_scaled_jacobian = min(minimum_scaled_jacobian, scaled)
        gradients[tet], volumes[tet] = poisson_tetra_geometry(points, ids)
        cell = base_fan.tet_cells[tet]
        parent_volumes[cell] += volumes[tet]
        parent_centers[:, cell] .+= volumes[tet] .* vec(sum(moved; dims=2)) ./ 4
    end
    parent_centers ./= transpose(parent_volumes)
    expected_volume = base_mesh.domain == :cube ? 1.0 : 3.0
    isapprox(sum(volumes), expected_volume; rtol=5e-12, atol=0.0) ||
        error("Perturbation changed the physical domain volume")
    points[:, base_fan.boundary_nodes] == base_fan.points[:, base_fan.boundary_nodes] ||
        error("Physical boundary points moved")
    mesh = merge(base_mesh, (; points=copy(points[:, 1:base_fan.vertex_count]),
        centers=parent_centers, volumes=parent_volumes))
    fan = merge(base_fan, (; points, gradients, volumes))
    displacement = points - base_fan.points
    geometry_stats = (; amplitude, tetrahedra=nt,
        vertices=size(points, 2), moved_vertices=count(!, fan.boundary_nodes),
        minimum_orientation_ratio, minimum_scaled_jacobian,
        maximum_displacement_over_h=maximum(norm(view(displacement, :, j))
            for j in axes(points, 2)) / h,
        domain_volume=sum(volumes), expected_volume)
    return (; mesh, fan, geometry_stats)
end

"""Assemble conforming P1 FEM on the actual tetrahedral mesh with body loading."""
function assemble_tetrahedral_p1(fan, problem; quadrature_order=6)
    nt, nv = length(fan.volumes), size(fan.points, 2)
    rows, columns = Vector{Int}(undef, 16nt), Vector{Int}(undef, 16nt)
    entries = Vector{Float64}(undef, 16nt)
    rhs = zeros(nv)
    rule = tetra_rule(quadrature_order)
    force_weights = zeros(length(rule.weights))
    index = 0
    for tet in 1:nt
        ids = fan.tet_nodes[:, tet]
        G, volume = fan.gradients[tet], fan.volumes[tet]
        local_K = volume * transpose(G) * G
        xyz = fan.points[:, ids] * rule.barycentric
        for q in eachindex(force_weights)
            force_weights[q] = volume * rule.weights[q] * problem.rhs(view(xyz, :, q))
        end
        rhs[ids] .+= rule.barycentric * force_weights
        for j in 1:4, i in 1:4
            index += 1
            rows[index], columns[index], entries[index] = ids[i], ids[j], local_K[i, j]
        end
    end
    return (; K=sparse(rows, columns, entries, nv, nv), rhs)
end

"""P1 solve; nonzero Dirichlet data are supported for reproduction checks.

The estimator adapter below intentionally requires homogeneous physical data,
matching the manufactured accuracy cases and their energy identity.
"""
function solve_tetrahedral_p1(fan, problem; quadrature_order=6,
                               dirichlet=x -> 0.0)
    assembled = assemble_tetrahedral_p1(fan, problem; quadrature_order)
    K, rhs = assembled.K, assembled.rhs
    free = findall(!, fan.boundary_nodes)
    boundary = findall(fan.boundary_nodes)
    u = zeros(length(rhs))
    for node in boundary
        u[node] = dirichlet(view(fan.points, :, node))
    end
    reduced_rhs = rhs[free] - K[free, boundary] * u[boundary]
    isempty(free) || (u[free] = cholesky(Symmetric(K[free, free])) \ reduced_rhs)
    residual = K * u - rhs
    algebraic_residual = norm(residual[free]) / max(norm(reduced_rhs), eps(Float64))
    return (; u, K, rhs, energy=dot(u, K * u) / 2,
        algebraic_residual, free_dofs=free, boundary_dofs=boundary,
        discretization=:tetrahedral_p1)
end

"""Adapt conforming P1 FEM to the existing full/reduced PU kernels.

No VEM projection or stabilization is used. The global common energy correction
B=F(U)-a(U,U) is only the algebraic solve residual here. Its roundoff value is
distributed by parent-cell volume solely to retain the exact global identity;
large cancelling local flux-work terms are not added to the marking measure.
"""
function tetrahedral_p1_estimator_data(mesh, fan, primal, problem; quadrature_order=6)
    length(primal.u) == size(fan.points, 2) || throw(DimensionMismatch("P1 fan values"))
    maximum(abs, primal.u[fan.boundary_nodes]; init=0.0) <= 100eps(Float64) ||
        throw(ArgumentError("This estimator adapter requires homogeneous Dirichlet data"))
    rule = tetra_rule(quadrature_order)
    N, weights = rule.barycentric, rule.weights
    shapes, weighted_shapes = poisson_load_shapes(N)
    U = copy(primal.u)
    nt, nn, nc = length(fan.volumes), length(U), length(mesh.cells)
    stiffness = Vector{Matrix{Float64}}(undef, nt)
    residual = zeros(10, nt)
    weighted_residual = zeros(10, 4, nt)
    force_weights, load_weighted = zeros(length(weights)), zeros(40)
    for tet in 1:nt
        ids = fan.tet_nodes[:, tet]
        G, volume = fan.gradients[tet], fan.volumes[tet]
        gradient = G * U[ids]
        internal_projection = transpose(G) * gradient
        K = zeros(10, 10)
        for j in 1:10, i in j:10
            K[i, j] = K[j, i] = volume * poisson_p2_stiffness_mean(G, i, j)
        end
        stiffness[tet] = K
        xyz = fan.points[:, ids] * N
        for q in eachindex(weights)
            force_weights[q] = volume * weights[q] * problem.rhs(view(xyz, :, q))
        end
        mul!(view(residual, :, tet), shapes, force_weights)
        mul!(load_weighted, weighted_shapes, force_weights)
        for hat in 1:4, shape in 1:10
            ordinary, weighted = poisson_internal_means(internal_projection, shape, hat)
            hat == 1 && (residual[shape, tet] -= volume * ordinary)
            weighted_residual[shape, hat, tet] =
                load_weighted[shape + 10(hat - 1)] - volume * weighted
        end
    end
    common_correction = dot(primal.rhs, U) - dot(U, primal.K * U)
    correction = common_correction .* mesh.volumes ./ sum(mesh.volumes)
    return (; U, stiffness, residual, weighted_residual, conversion=zeros(nn),
        correction, stabilization=zeros(nc), stabilization_nodal=zeros(nn),
        common_correction, discretization=:tetrahedral_p1)
end

"""Existing local patch solves with independently selectable DWR readout.

For the quadratic energy goal, retain the defaults. For a linear goal, pass
dual residual data as `rhs_data`, primal residual data as `readout_data`,
`factor=1`, and `include_corrections=false`. This changes neither patch space.
"""
function tetrahedral_patch_estimates(mesh, fan, data; rhs_data=data,
        readout_data=data, factor=0.5, include_corrections=true, rank_tolerance=1e-11)
    nn = length(data.U)
    base = include_corrections ? readout_data.conversion : zeros(nn)
    correction = include_corrections ? readout_data.correction : zeros(length(mesh.cells))
    full_nodal, reduced_nodal = copy(base), copy(base)
    patchdofs, ranks = zeros(Int, nn), zeros(Int, nn)
    boundary_mask = poisson_boundary_mask(fan)
    for node in 1:nn
        dofs = poisson_patch_dofs(fan, node, boundary_mask)
        K, b, g = poisson_patch_system(fan, rhs_data, node, dofs)
        if readout_data !== rhs_data
            _, _, g = poisson_patch_system(fan, readout_data, node, dofs)
        end
        patchdofs[node] = length(b)
        isempty(b) && continue
        delta = cholesky(Symmetric(K)) \ b
        full_nodal[node] += factor * dot(g, delta)
        Q, rank = poisson_reduced_basis(fan, node, dofs; rank_tolerance)
        ranks[node] = rank
        if rank > 0
            reduced_K = transpose(Q) * K * Q
            reduced_delta = Q * (cholesky(Symmetric(reduced_K)) \ (transpose(Q) * b))
            reduced_nodal[node] += factor * dot(g, reduced_delta)
        end
    end
    full_stats = (; total_dofs=sum(patchdofs), max_dofs=maximum(patchdofs),
        patch_count=nn, discretization=:tetrahedral_p1)
    reduced_stats = (; total_dofs=sum(ranks), max_dofs=maximum(ranks),
        reduced_patches=count(i -> ranks[i] < patchdofs[i], 1:nn),
        patch_count=nn, discretization=:tetrahedral_p1)
    full = poisson_collect_estimate(mesh, fan, full_nodal, correction,
        patchdofs, copy(patchdofs), full_stats)
    reduced = poisson_collect_estimate(mesh, fan, reduced_nodal, correction,
        copy(ranks), ranks, reduced_stats)
    return (; full, reduced)
end

"""Exact-field diagnostics grouped by the unchanged parent-cell labels."""
function tetrahedral_p1_truth(mesh, fan, primal, problem; quadrature_order=10)
    rule = tetra_rule(quadrature_order)
    exact_energy = zeros(length(mesh.cells))
    discrete_energy = similar(exact_energy); fill!(discrete_energy, 0.0)
    reconstruction_error = similar(exact_energy); fill!(reconstruction_error, 0.0)
    for tet in eachindex(fan.volumes)
        ids = fan.tet_nodes[:, tet]
        gradient = fan.gradients[tet] * primal.u[ids]
        cell = fan.tet_cells[tet]
        discrete_energy[cell] += fan.volumes[tet] * dot(gradient, gradient) / 2
        quadrature = tetra_quadrature(fan.points[:, ids], rule)
        for q in eachindex(quadrature.weights)
            exact_gradient = collect(problem.gradient(view(quadrature.points, :, q)))
            weight = quadrature.weights[q] / 2
            exact_energy[cell] += weight * dot(exact_gradient, exact_gradient)
            difference = exact_gradient - gradient
            reconstruction_error[cell] += weight * dot(difference, difference)
        end
    end
    return (; exact_energy, discrete_energy,
        goal_difference=exact_energy-discrete_energy, reconstruction_error,
        integrated_exact_energy=sum(exact_energy))
end
