using LinearAlgebra

const POISSON_P2_EDGES = ((1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4))

function poisson_p2_values(N)
    values = [N[i] * (2N[i] - 1) for i in 1:4]
    for (a, b) in POISSON_P2_EDGES
        push!(values, 4N[a] * N[b])
    end
    return values
end

function poisson_p2_gradients(G, N)
    gradients = zeros(3, 10)
    for i in 1:4
        gradients[:, i] .= (4N[i] - 1) .* view(G, :, i)
    end
    for (edge, (a, b)) in enumerate(POISSON_P2_EDGES)
        gradients[:, 4 + edge] .= 4 .* (N[a] .* view(G, :, b) + N[b] .* view(G, :, a))
    end
    return gradients
end

"""P2 shape integrals against the load, including each of the four PU hats."""
function poisson_load_shapes(barycentric)
    shapes = zeros(10, size(barycentric, 2))
    weighted = zeros(40, size(barycentric, 2))
    for q in axes(barycentric, 2)
        for i in 1:4
            n = barycentric[i, q]
            shapes[i, q] = n * (2n - 1)
        end
        for (edge, (a, b)) in enumerate(POISSON_P2_EDGES)
            shapes[4 + edge, q] = 4barycentric[a, q] * barycentric[b, q]
        end
        for hat in 1:4, i in 1:10
            weighted[i + 10(hat - 1), q] = barycentric[hat, q] * shapes[i, q]
        end
    end
    return shapes, weighted
end

"""Exact tetrahedral mean of grad(phi_i) dot grad(phi_j)."""
function poisson_p2_stiffness_mean(G, i, j)
    if i <= 4 && j <= 4
        return ((4(i == j) - 1) / 5) * dot(view(G, :, i), view(G, :, j))
    elseif i <= 4
        a, b = POISSON_P2_EDGES[j - 4]
        return ((4(i == a) - 1) * dot(view(G, :, i), view(G, :, b)) +
                (4(i == b) - 1) * dot(view(G, :, i), view(G, :, a))) / 5
    elseif j <= 4
        return poisson_p2_stiffness_mean(G, j, i)
    end
    a, b = POISSON_P2_EDGES[i - 4]
    c, d = POISSON_P2_EDGES[j - 4]
    return 0.8 * ((1 + (a == c)) * dot(view(G, :, b), view(G, :, d)) +
                  (1 + (a == d)) * dot(view(G, :, b), view(G, :, c)) +
                  (1 + (b == c)) * dot(view(G, :, a), view(G, :, d)) +
                  (1 + (b == d)) * dot(view(G, :, a), view(G, :, c)))
end

"""Exact mean grad(U) dot grad(phi), with and without a PU hat."""
function poisson_internal_means(projected, shape, hat)
    if shape <= 4
        return 0.0, ((4(shape == hat) - 1) * projected[shape] - projected[hat]) / 20
    end
    a, b = POISSON_P2_EDGES[shape - 4]
    ordinary = projected[a] + projected[b]
    weighted = ((1 + (hat == a)) * projected[b] +
                (1 + (hat == b)) * projected[a] + projected[hat]) / 5
    return ordinary, weighted
end

"""Shared tetra data for the full and reduced scalar Poisson estimators.

Only load integrals require quadrature. Stiffness, internal residuals and the
energy conversion use exact polynomial moments. No polynomial displacement fit
is used by either reconstruction.
"""
function poisson_estimator_data(mesh, fan, primal, problem; quadrature_order=6)
    rule = tetra_rule(quadrature_order)
    N, weights = rule.barycentric, rule.weights
    shapes, weighted_shapes = poisson_load_shapes(N)
    U = reconstruct_fan(mesh, fan, primal.u)
    nt = size(fan.tet_nodes, 2)
    nn = size(fan.points, 2)
    nc = length(mesh.cells)
    stiffness = Vector{Matrix{Float64}}(undef, nt)
    residual = zeros(10, nt)
    weighted_residual = zeros(10, 4, nt)
    conversion = zeros(nn)
    correction = zeros(nc)
    stabilization = zeros(nc)
    stabilization_nodal = zeros(nn)
    projected_energy = zeros(nc)
    for cell in 1:nc
        ids = mesh.cells[cell]
        data = primal.cells[cell]
        local_u = primal.u[ids]
        defect = local_u - data.P * local_u
        projected_energy[cell] = sum(abs2, data.G * local_u)
        stabilization[cell] = data.alpha * sum(abs2, defect)
        for (local_node, node) in enumerate(ids)
            contribution = data.alpha * defect[local_node]^2
            conversion[node] -= contribution / 2
            stabilization_nodal[node] += contribution
        end
        correction[cell] = mesh.volumes[cell] * projected_energy[cell] +
            stabilization[cell] - dot(data.load, local_u)
    end

    force_weights = zeros(length(weights))
    load_weighted = zeros(40)
    for tet in 1:nt
        ids = view(fan.tet_nodes, :, tet)
        G = fan.gradients[tet]
        volume = fan.volumes[tet]
        cell = fan.tet_cells[tet]
        u_tet = U[ids]
        gradient = G * u_tet
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
        for hat in 1:4, i in 1:10
            ordinary, weighted = poisson_internal_means(internal_projection, i, hat)
            hat == 1 && (residual[i, tet] -= volume * ordinary)
            weighted_residual[i, hat, tet] = load_weighted[i + 10(hat - 1)] - volume * weighted
        end
        energy = sum(abs2, gradient)
        load_work = dot(u_tet, N * force_weights)
        correction[cell] += load_work - volume * energy
        localized_conversion = volume * (energy - projected_energy[cell]) / 8
        for node in ids
            conversion[node] += localized_conversion
        end
    end
    return (; U, stiffness, residual, weighted_residual, conversion, correction,
            stabilization, stabilization_nodal)
end

poisson_ordered_edge(a, b) = a < b ? (a, b) : (b, a)

function poisson_boundary_mask(fan)
    fan.boundary_nodes isa AbstractVector{Bool} && return fan.boundary_nodes
    mask = falses(size(fan.points, 2))
    for node in fan.boundary_nodes
        mask[node] = true
    end
    return mask
end

"""Number the central P2 vertex and radial edges, removing Dirichlet DOFs.

The entire physical boundary is homogeneous Dirichlet. Consequently every
other P2 shape has zero trace on the artificial patch boundary and is removed.
The returned radial node IDs follow the free-DOF row order; 0 denotes the center.
"""
function poisson_patch_dofs(fan, node, boundary_mask)
    neighbors = Int[]
    for tet in fan.stars[node], other in view(fan.tet_nodes, :, tet)
        other == node || push!(neighbors, other)
    end
    sort!(neighbors)
    unique!(neighbors)
    radial_rows = Dict{Int, Int}()
    row_nodes = Int[]
    center_row = 0
    if !boundary_mask[node]
        push!(row_nodes, 0)
        center_row = 1
    end
    for other in neighbors
        poisson_ordered_edge(node, other) in fan.boundary_edges && continue
        push!(row_nodes, other)
        radial_rows[other] = length(row_nodes)
    end
    return (; center_row, radial_rows, row_nodes)
end

"""Assemble the common full-P2 patch operator and both residual vectors."""
function poisson_patch_system(fan, data, node, dofs)
    n = length(dofs.row_nodes)
    K = zeros(n, n)
    b = zeros(n)
    g = zeros(n)
    local_shapes = zeros(Int, 4)
    rows = zeros(Int, 4)
    for tet in fan.stars[node]
        ids = view(fan.tet_nodes, :, tet)
        hat = findfirst(==(node), ids)
        count = 0
        if dofs.center_row != 0
            count += 1
            rows[count] = dofs.center_row
            local_shapes[count] = hat
        end
        for (edge, (a, bnode)) in enumerate(POISSON_P2_EDGES)
            (a == hat || bnode == hat) || continue
            other = ids[a == hat ? bnode : a]
            row = get(dofs.radial_rows, other, 0)
            row == 0 && continue
            count += 1
            rows[count] = row
            local_shapes[count] = 4 + edge
        end
        for i in 1:count
            row, shape = rows[i], local_shapes[i]
            b[row] += data.residual[shape, tet]
            g[row] += data.weighted_residual[shape, hat, tet]
            for j in 1:count
                K[row, rows[j]] += data.stiffness[tet][shape, local_shapes[j]]
            end
        end
    end
    return K, b, g
end

"""N_i P1 plus six radial quadratic aggregates, restricted to free P2 rows.

Columns are N_i, N_i*xi_1:3, and radial features
[dx^2,dy^2,dz^2,2dx*dy,2dx*dz,2dy*dz], xi=(x-x_i)/h.
The six curvature coefficients are free Galerkin coefficients, not fitted
Hessian entries. Dirichlet nodal masking changes the corresponding boundary
basis functions but keeps them conforming and admissible; no full-P2 fallback
is made. SVD removes dependent columns, including regular face-star identities.
"""
function poisson_reduced_basis(fan, node, dofs; rank_tolerance=1e-11)
    n = length(dofs.row_nodes)
    T = zeros(n, 10)
    center = view(fan.points, :, node)
    h = 0.0
    for other in dofs.row_nodes
        other == 0 && continue
        h = max(h, norm(view(fan.points, :, other) - center))
    end
    h = iszero(h) ? 1.0 : h
    for (row, other) in enumerate(dofs.row_nodes)
        if other == 0
            T[row, 1] = 1.0
        else
            dx = (fan.points[1, other] - center[1]) / h
            dy = (fan.points[2, other] - center[2]) / h
            dz = (fan.points[3, other] - center[3]) / h
            T[row, :] = [0.5, dx / 4, dy / 4, dz / 4,
                         dx^2, dy^2, dz^2, 2dx*dy, 2dx*dz, 2dy*dz]
        end
    end
    n == 0 && return zeros(0, 0), 0
    decomposition = svd(T; full=false)
    largest = first(decomposition.S)
    rank = iszero(largest) ? 0 : count(s -> s > rank_tolerance * largest, decomposition.S)
    return decomposition.U[:, 1:rank], rank
end

"""Take absolute values only after complete nodal localization."""
function poisson_collect_estimate(mesh, fan, nodal, correction, patchdofs, ranks, statistics)
    indicators = abs.(correction)
    for node in eachindex(nodal)
        cells = fan.node_cells[node]
        total_volume = sum(mesh.volumes[cell] for cell in cells)
        for cell in cells
            indicators[cell] += abs(nodal[node]) * mesh.volumes[cell] / total_volume
        end
    end
    return (; signed=sum(nodal) + sum(correction),
            magnitude=sum(abs, nodal) + sum(abs, correction), indicators, nodal,
            correction=copy(correction), patchdofs, ranks, statistics)
end

"""sDWR by substituting the linear VEM projection for the enriched field.

The projected enrichment p=Pi_K(u_h) is generally broken between cells. Evaluate
the same volume residual with delta=p-U, retaining the same goal conversion and
cell correction. There is no patch solve. This is the requested projected
substitution, not a continuous higher-order reconstruction or a jump estimator.

For projected loads F_h(u_h)=F(p), equilibrium and the mean-gradient-preserving
fan give signed eta_sDWR = (S + F(U)-F_h(u_h))/2. Its localization is still by
complete signed PU-node contributions and separate cell corrections.
"""
function poisson_projected_estimate(mesh,fan,primal,data)
    nodal = copy(data.conversion)
    load_difference = 0.0
    for tet in eachindex(fan.volumes)
        cell = fan.tet_cells[tet]
        ids = fan.tet_nodes[:,tet]
        local_u = primal.u[mesh.cells[cell]]
        cell_data = primal.cells[cell]
        projected_gradient = cell_data.G*local_u
        projected_values = fill(sum(local_u)/length(local_u),4) +
            transpose(fan.points[:,ids] .- cell_data.center)*projected_gradient
        difference = projected_values-data.U[ids]
        # An affine difference has these exact P2 nodal coefficients.
        coefficients = vcat(difference,
            [(difference[a]+difference[b])/2 for (a,b) in POISSON_P2_EDGES])
        for hat in 1:4
            nodal[ids[hat]] += dot(data.weighted_residual[:,hat,tet],coefficients)/2
        end
        # R_U(p-U)+a(U,p-U)=F(p-U), so this also records the load defect
        # without substituting equilibrium or using the answer's energy error.
        gradient_U = fan.gradients[tet]*data.U[ids]
        gradient_difference = fan.gradients[tet]*difference
        load_difference -= dot(data.residual[:,tet],coefficients) +
            fan.volumes[tet]*dot(gradient_U,gradient_difference)
    end
    statistics = (;patch_count=0,total_dofs=0,max_dofs=0,
        stabilization_part=sum(data.stabilization)/2,
        load_consistency_part=load_difference/2,
        convention="linear projected enrichment with complete goal/load corrections")
    return poisson_collect_estimate(mesh,fan,nodal,data.correction,
        Int[],Int[],statistics)
end

"""Compare full P2, curvature-reduced PU-DWR, and projected sDWR.

The goal is J(u)=a(u,u)/2. Each PU contribution is
  .5 R(N_i*delta_i) + .5 integral N_i*(|grad U|^2-|G u_h|^2) -.5 S_i.
Each separate cell correction is F(U)-a(U,U)-[F_h(u_h)-a_h(u_h,u_h)].
This convention estimates J(u)-J_h(u_h), not just the error of the reconstructed
continuous field U. Reduced solves use exactly the same residual and correction.

sDWR replaces the enriched field by the elementwise linear VEM projection,
including body-load consistency. All three use the same conversion, separate
cell correction, and absolute-value localization. No fitted scale is used.
"""
function estimate_errors(mesh, fan, primal, problem; quadrature_order=6,
                         rank_tolerance=1e-11)
    0 < rank_tolerance < 1 || throw(ArgumentError("rank_tolerance must lie in (0,1)"))
    data = poisson_estimator_data(mesh, fan, primal, problem; quadrature_order)
    nn = length(data.conversion)
    boundary_mask = poisson_boundary_mask(fan)
    full_nodal = copy(data.conversion)
    reduced_nodal = copy(data.conversion)
    patchdofs = zeros(Int, nn)
    ranks = zeros(Int, nn)
    for node in 1:nn
        dofs = poisson_patch_dofs(fan, node, boundary_mask)
        K, b, g = poisson_patch_system(fan, data, node, dofs)
        n = length(b)
        patchdofs[node] = n
        n == 0 && continue
        delta = cholesky(Symmetric(K)) \ b
        full_nodal[node] += dot(g, delta) / 2
        Q, rank = poisson_reduced_basis(fan, node, dofs; rank_tolerance)
        ranks[node] = rank
        if rank > 0
            reduced_K = transpose(Q) * K * Q
            reduced_b = transpose(Q) * b
            reduced_delta = Q * (cholesky(Symmetric(reduced_K)) \ reduced_b)
            reduced_nodal[node] += dot(g, reduced_delta) / 2
        end
    end
    total_full, total_reduced = sum(patchdofs), sum(ranks)
    reduction_fraction = iszero(total_full) ? 0.0 : 1 - total_reduced / total_full
    common = (; patch_count=nn, active_patch_count=count(>(0), patchdofs),
              total_full_dofs=total_full, physical_boundary_nodes=count(boundary_mask))
    full_statistics = merge(common, (; total_dofs=total_full, max_dofs=maximum(patchdofs),
        mean_dofs=total_full / nn, reduction_fraction=0.0, full_fallback_count=0))
    reduced_statistics = merge(common, (; total_dofs=total_reduced, max_dofs=maximum(ranks),
        mean_dofs=total_reduced / nn, reduction_fraction,
        reduced_patches=count(i -> ranks[i] < patchdofs[i], 1:nn),
        unchanged_patches=count(i -> ranks[i] == patchdofs[i], 1:nn),
        rank_tolerance, full_fallback_count=0, boundary_treatment="masked Dirichlet P2 rows"))
    full = poisson_collect_estimate(mesh, fan, full_nodal, data.correction,
        patchdofs, copy(patchdofs), full_statistics)
    reduced = poisson_collect_estimate(mesh, fan, reduced_nodal, data.correction,
        copy(ranks), ranks, reduced_statistics)
    sdwr = poisson_projected_estimate(mesh,fan,primal,data)
    return (; full, reduced, sdwr)
end
