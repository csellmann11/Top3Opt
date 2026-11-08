"""Independent quadrature oracle for the projected full-estimator substitution.

This deliberately does not use poisson_estimator_data, the cached weighted
residual, or the estimator's collector. The cellwise polynomial p=Pi_1(u_h) is
evaluated directly at quadrature points. Its possibly discontinuous trace is
not interpreted as a continuous patch reconstruction.
"""
function independent_projected_sdwr(mesh, fan, primal, problem; quadrature_order=6)
    rule = tetra_rule(quadrature_order)
    U = reconstruct_fan(mesh, fan, primal.u)
    nn, nc = length(U), length(mesh.cells)
    nodal = zeros(nn)
    correction = zeros(nc)
    stabilization = zeros(nc)
    projection_gradients = Vector{Vector{Float64}}(undef, nc)
    projection_means = zeros(nc)
    for cell in 1:nc
        ids = mesh.cells[cell]
        local_u = primal.u[ids]
        data = primal.cells[cell]
        defect = local_u - data.P * local_u
        projection_gradients[cell] = data.G * local_u
        projection_means[cell] = sum(local_u) / length(local_u)
        stabilization[cell] = data.alpha * dot(defect, defect)
        correction[cell] = mesh.volumes[cell] * sum(abs2, projection_gradients[cell]) +
            stabilization[cell] - dot(data.load, local_u)
        for (local_node, node) in enumerate(ids)
            nodal[node] -= data.alpha * defect[local_node]^2 / 2
        end
    end

    F_U, F_p, B, A, A_projection = 0.0, 0.0, 0.0, 0.0, 0.0
    for tet in axes(fan.tet_nodes, 2)
        ids = fan.tet_nodes[:, tet]
        cell = fan.tet_cells[tet]
        G = fan.gradients[tet]
        gradient_U = G * U[ids]
        gradient_p = projection_gradients[cell]
        gradient_difference = gradient_p - gradient_U
        quadrature = tetra_quadrature(fan.points[:, ids], rule)
        energy_U = dot(gradient_U, gradient_U)
        energy_p = dot(gradient_p, gradient_p)
        cross_energy = dot(gradient_U, gradient_p)
        for q in eachindex(quadrature.weights)
            N = view(rule.barycentric, :, q)
            x = view(quadrature.points, :, q)
            weight = quadrature.weights[q]
            f = problem.rhs(x)
            value_U = dot(N, U[ids])
            value_p = projection_means[cell] +
                dot(gradient_p, x - mesh.centers[:, cell])
            difference = value_p - value_U
            F_U += weight * f * value_U
            F_p += weight * f * value_p
            B += weight * cross_energy
            A += weight * energy_U
            A_projection += weight * energy_p
            correction[cell] += weight * (f * value_U - energy_U)
            for hat in 1:4
                # Direct product rule for grad(N_i*(p-U)); no P2 residual cache.
                product_gradient = N[hat] * gradient_difference + difference * G[:, hat]
                residual = f * N[hat] * difference - dot(gradient_U, product_gradient)
                conversion = N[hat] * (energy_U - energy_p)
                nodal[ids[hat]] += weight * (residual + conversion) / 2
            end
        end
    end

    indicators = abs.(correction)
    for node in eachindex(nodal)
        adjacent = fan.node_cells[node]
        total_volume = sum(mesh.volumes[cell] for cell in adjacent)
        for cell in adjacent
            indicators[cell] += abs(nodal[node]) * mesh.volumes[cell] / total_volume
        end
    end
    F_h = dot(primal.u, primal.rhs)
    H = dot(primal.u, primal.K * primal.u)
    S = sum(stabilization)
    signed = sum(nodal) + sum(correction)
    magnitude = sum(abs, nodal) + sum(abs, correction)
    expanded = (F_p + F_U - B + H) / 2 - F_h
    equilibrium_identity = (S + F_U - F_h) / 2
    return (; nodal, correction, indicators, signed, magnitude, expanded,
            equilibrium_identity, F_U, F_p, F_h, A, B, A_projection, H, S,
            load_consistency=F_U-F_h, old_scaled_proxy=S/2)
end

@testset "sDWR uses the complete projected estimator and its localization" begin
    comparisons = NamedTuple[]
    for name in (:cube_smooth, :lshape_smooth, :lshape_singular)
        problem = manufactured_problem(name)
        mesh = build_mesh(problem.domain, 2)
        fan = build_fan(mesh)
        primal = solve_primal(mesh, fan, problem; quadrature_order=6)
        estimates = estimate_errors(mesh, fan, primal, problem; quadrature_order=6)
        expected = independent_projected_sdwr(mesh, fan, primal, problem; quadrature_order=6)
        actual = estimates.sdwr

        @test expected.F_p ≈ expected.F_h rtol=5e-12 atol=2e-14
        @test expected.H ≈ expected.F_h rtol=5e-12 atol=2e-14
        @test expected.B ≈ expected.A_projection rtol=5e-12 atol=2e-14
        @test expected.H ≈ expected.A_projection + expected.S rtol=5e-12 atol=2e-14
        @test expected.signed ≈ expected.expanded rtol=5e-11 atol=2e-14
        @test expected.signed ≈ expected.equilibrium_identity rtol=5e-11 atol=2e-14
        @test sum(expected.correction) ≈
            expected.F_U - expected.A - expected.F_h + expected.H rtol=5e-11 atol=2e-14

        @test actual.nodal ≈ expected.nodal rtol=5e-11 atol=2e-14
        @test actual.correction ≈ expected.correction rtol=5e-11 atol=2e-14
        @test actual.correction == estimates.full.correction
        @test actual.correction == estimates.reduced.correction
        @test actual.signed ≈ expected.signed rtol=5e-11 atol=2e-14
        @test actual.signed ≈ expected.expanded rtol=5e-11 atol=2e-14
        @test actual.indicators ≈ expected.indicators rtol=5e-11 atol=2e-14
        @test actual.magnitude ≈ expected.magnitude rtol=5e-11 atol=2e-14
        @test sum(actual.indicators) ≈ actual.magnitude rtol=5e-12 atol=2e-14
        @test actual.signed - expected.old_scaled_proxy ≈
            expected.load_consistency / 2 rtol=5e-11 atol=2e-14

        push!(comparisons, (; problem=name, projected=actual.signed,
            old_scaled_proxy=expected.old_scaled_proxy,
            load_consistency=expected.load_consistency,
            localized_magnitude=actual.magnitude))
    end
    # A scaled stabilization proxy cannot satisfy the full formula whenever
    # load consistency is nonzero; no empirical effectivity multiplier is used.
    @test any(abs(result.load_consistency) > 1e-12 for result in comparisons)
    @test any(abs(result.projected - result.old_scaled_proxy) > 5e-13 for result in comparisons)
    println("SDWR_PROJECTION_CHECK ", comparisons)
end
