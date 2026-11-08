using LinearAlgebra

"""Redistribute complete signed PU-node contributions by adjacent cell volume.

Absolute values are taken after summing the entire node support. Optional
`cell_weights` permits density times volume; the Poisson benchmark uses volume.
"""
function sdwr_distribute_nodes(mesh, fan, nodal; cell_weights=mesh.volumes)
    length(cell_weights) == length(mesh.cells) || throw(DimensionMismatch("cell weights"))
    all(w -> isfinite(w) && w > 0, cell_weights) ||
        throw(ArgumentError("cell weights must be finite and positive"))
    signed = zeros(length(mesh.cells))
    absolute = zeros(length(mesh.cells))
    for node in eachindex(nodal)
        cells = fan.node_cells[node]
        denominator = sum(cell_weights[cell] for cell in cells)
        for cell in cells
            weight = cell_weights[cell] / denominator
            signed[cell] += weight * nodal[node]
            absolute[cell] += weight * abs(nodal[node])
        end
    end
    return (; signed, absolute)
end

"""Integrate the quadratic-goal DWR identity with supplied enriched fields.

Positive adjoint convention: a(v,z) = J'(u)[v]. The target is J(u)-J(U),
where U is the continuous fan reconstruction. For linear Poisson and a goal
of degree at most two, the cubic remainder vanishes and exact enrichments give

    R_U(Z) + 1/2 R_U(z^+-Z)
             + 1/2 (J'(U)[u^+-U] - a(u^+-U,Z)).

`enrichment(tet,cell,x)` returns `(u, z, gradient_u, gradient_z)` as named
fields. Each PU hat multiplies the complete error argument, including its
product-rule gradient. Consistency is integrated as R_U(N_i Z), not assigned
uniformly from a cell scalar. No goal conversion or discrete-residual
subtraction belongs to this continuous-goal target.

The default `:separated` marking preserves the earlier convention of separate
cell consistency and complete nodal enrichment terms. `:combined` takes the
absolute value after combining all three terms at each PU node. Both are
returned explicitly; neither changes the signed estimator.
"""
function sdwr_integrate_dwr(mesh, fan, U, Z, problem, goal, enrichment;
                            quadrature_order=6, localization=:separated,
                            cell_weights=mesh.volumes, goal_rules=nothing)
    localization in (:separated, :combined) ||
        throw(ArgumentError("localization must be :separated or :combined"))
    nn, nc = size(fan.points, 2), length(mesh.cells)
    length(U) == nn && length(Z) == nn || throw(DimensionMismatch("fan fields"))
    nodal_consistency, nodal_primal, nodal_dual = zeros(nn), zeros(nn), zeros(nn)
    cell_consistency, cell_primal, cell_dual = zeros(nc), zeros(nc), zeros(nc)
    rule = tetra_rule(quadrature_order)
    goal_rules === nothing || length(goal_rules) == length(fan.volumes) ||
        throw(DimensionMismatch("one goal integration rule per tetrahedron"))
    N, weights = rule.barycentric, rule.weights
    for tet in eachindex(fan.volumes)
        ids = view(fan.tet_nodes, :, tet)
        cell, volume = fan.tet_cells[tet], fan.volumes[tet]
        G = fan.gradients[tet]
        local_u, local_z = U[ids], Z[ids]
        grad_u, grad_z = G * local_u, G * local_z
        xyz = fan.points[:, ids] * N
        for q in eachindex(weights)
            x, shape = view(xyz, :, q), view(N, :, q)
            u, z = dot(local_u, shape), dot(local_z, shape)
            plus = enrichment(tet, cell, x)
            error_u, error_z = plus.u - u, plus.z - z
            force = problem.rhs(x)
            # A cut goal is integrated separately on its exact geometric
            # intersection; the PDE residual still uses the whole tetrahedron.
            goal_value = goal_rules === nothing ? goal.derivative_value(x, u) : 0.0
            goal_gradient = goal_rules === nothing ? goal.derivative_gradient(x, grad_u) : (0.0,0.0,0.0)
            weight = volume * weights[q]
            for hat in 1:4
                node, ni = ids[hat], shape[hat]
                grad_ni = view(G, :, hat)
                # Explicit products avoid treating a weighted error as a
                # scalar weight on an already differentiated residual.
                consistency_internal = 0.0
                primal_internal = 0.0
                dual_internal = 0.0
                goal_gradient_term = 0.0
                for component in 1:3
                    gradient_z_hat = ni * grad_z[component] + z * grad_ni[component]
                    gradient_ez_hat = ni * (plus.gradient_z[component]-grad_z[component]) + error_z * grad_ni[component]
                    gradient_eu_hat = ni * (plus.gradient_u[component]-grad_u[component]) + error_u * grad_ni[component]
                    consistency_internal += grad_u[component] * gradient_z_hat
                    primal_internal += grad_u[component] * gradient_ez_hat
                    dual_internal += grad_z[component] * gradient_eu_hat
                    goal_gradient_term += goal_gradient[component] * gradient_eu_hat
                end
                consistency = weight * (force * ni * z - consistency_internal)
                primal = weight * (force * ni * error_z - primal_internal) / 2
                dual = weight * (goal_value * ni * error_u + goal_gradient_term - dual_internal) / 2
                nodal_consistency[node] += consistency
                nodal_primal[node] += primal
                nodal_dual[node] += dual
                cell_consistency[cell] += consistency
                cell_primal[cell] += primal
                cell_dual[cell] += dual
            end
        end
        if goal_rules !== nothing
            goal_rule = goal_rules[tet]
            goal_xyz = fan.points[:,ids]*goal_rule.barycentric
            for q in eachindex(goal_rule.weights)
                shape = view(goal_rule.barycentric,:,q)
                x = view(goal_xyz,:,q)
                u = dot(local_u,shape)
                plus = enrichment(tet,cell,x)
                error_u = plus.u-u
                goal_value = goal.derivative_value(x,u)
                goal_gradient = goal.derivative_gradient(x,grad_u)
                for hat in 1:4
                    gradient_term = sum(goal_gradient[d]*(shape[hat]*
                        (plus.gradient_u[d]-grad_u[d])+error_u*G[d,hat]) for d in 1:3)
                    contribution = goal_rule.weights[q]*(goal_value*shape[hat]*error_u+
                        gradient_term)/2
                    nodal_dual[ids[hat]] += contribution
                    cell_dual[cell] += contribution
                end
            end
        end
    end
    nodal_enrichment = nodal_primal + nodal_dual
    nodal_total = nodal_consistency + nodal_enrichment
    enrichment_distribution = sdwr_distribute_nodes(mesh, fan, nodal_enrichment; cell_weights)
    combined_distribution = sdwr_distribute_nodes(mesh, fan, nodal_total; cell_weights)
    indicators_separated = abs.(cell_consistency) + enrichment_distribution.absolute
    indicators_combined = combined_distribution.absolute
    cell_signed = cell_consistency + cell_primal + cell_dual
    indicators = localization == :separated ? indicators_separated : indicators_combined
    signed = sum(nodal_total)
    statistics = (; patch_count=0, total_dofs=0, localization,
        convention="positive adjoint; projected sDWR for J(u)-J(U); broken volume residual")
    return (; signed, magnitude=sum(indicators), indicators,
        indicators_separated, indicators_combined,
        cell_signed, cell_consistency, cell_primal, cell_dual,
        nodal=nodal_enrichment, nodal_total, nodal_consistency, nodal_primal, nodal_dual,
        distributed_signed_separated=cell_consistency+enrichment_distribution.signed,
        distributed_signed_combined=combined_distribution.signed,
        statistics)
end

"""Affine VEM projection coefficients for one already solved field."""
function sdwr_projection_fields(mesh, primal)
    nc = length(mesh.cells)
    gradients = Matrix{Float64}(undef, 3, nc)
    constants = Vector{Float64}(undef, nc)
    for cell in 1:nc
        local_u = primal.u[mesh.cells[cell]]
        data = primal.cells[cell]
        gradient = data.G * local_u
        gradients[:, cell] = gradient
        constants[cell] = sum(local_u) / length(local_u) - dot(gradient, data.center)
    end
    return (; gradients, constants)
end

"""Literal sDWR substitution u^+=Pi_1 u_h and z^+=Pi_1 z_h.

The affine projections are cellwise and generally discontinuous. This is the
requested broken-volume residual proxy, not a conforming enriched DWR field.
No interface-flux or jump correction is silently added, and no patch is solved.
"""
function sdwr_estimate(mesh, fan, primal, dual, problem, goal;
                       quadrature_order=6, localization=:separated,
                       cell_weights=mesh.volumes, goal_rules=nothing)
    U = reconstruct_fan(mesh, fan, primal.u)
    Z = reconstruct_fan(mesh, fan, dual.u)
    pu, pz = sdwr_projection_fields(mesh, primal), sdwr_projection_fields(mesh, dual)
    function enrichment(tet, cell, x)
        gu, gz = view(pu.gradients, :, cell), view(pz.gradients, :, cell)
        return (; u=pu.constants[cell]+dot(gu, x), z=pz.constants[cell]+dot(gz, x),
            gradient_u=gu, gradient_z=gz)
    end
    return sdwr_integrate_dwr(mesh, fan, U, Z, problem, goal, enrichment;
        quadrature_order, localization, cell_weights, goal_rules)
end
