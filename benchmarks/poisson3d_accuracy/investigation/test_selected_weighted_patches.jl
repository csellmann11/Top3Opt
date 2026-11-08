using Test

isdefined(@__MODULE__, :poisson_estimator_data) ||
    include(joinpath(@__DIR__, "..", "load.jl"))
isdefined(@__MODULE__, :irregular_tetrahedral_mesh) || include("irregular_tetrahedra.jl")
isdefined(@__MODULE__, :locally_refined_tetrahedral_mesh) || include("local_tetra_refinement.jl")
isdefined(@__MODULE__, :selected_weighted_patch_system) || include("selected_weighted_patches.jl")

function selected_test_interior_node(fan)
    candidates = Int[]
    for node in fan.vertex_count + 1:fan.vertex_count + fan.face_count
        vertices = unique(vec(fan.tet_nodes[:, fan.stars[node]]))
        all(v -> !fan.boundary_nodes[v], vertices) && push!(candidates, node)
    end
    isempty(candidates) && error("No wholly interior face-center patch")
    return candidates[argmin([sum(abs2, fan.points[:, node] .- 0.5) for node in candidates])]
end

"""Direct element quadrature, independent of the assembled cubic-moment formulas."""
function selected_test_quadrature(fan, system, U, forcing;
        readout_U=U, readout_forcing=forcing, order=5)
    n = length(system.edges)
    A, K = zeros(n, n), zeros(n, n)
    b, g, internal = zeros(n), zeros(n), zeros(n)
    edge_rows = Dict(edge => row for (row, edge) in enumerate(system.edges))
    rule = tetra_rule(order)
    for tet in fan.stars[system.node]
        ids = fan.tet_nodes[:, tet]
        G, volume = fan.gradients[tet], fan.volumes[tet]
        hat = findfirst(==(system.node), ids)
        rows = [get(edge_rows, minmax(ids[a], ids[b]), 0) for (a, b) in POISSON_P2_EDGES]
        gradient_U, gradient_readout = G * U[ids], G * readout_U[ids]
        xyz = fan.points[:, ids] * rule.barycentric
        for q in eachindex(rule.weights)
            N = rule.barycentric[:, q]
            values = [4N[a]*N[b] for (a, b) in POISSON_P2_EDGES]
            gradients = [4 .* (N[a] .* G[:, b] + N[b] .* G[:, a])
                for (a, b) in POISSON_P2_EDGES]
            weight = volume * rule.weights[q]
            for test in 1:6
                row = rows[test]
                row == 0 && continue
                test_value = N[hat] * values[test]
                test_gradient = N[hat] .* gradients[test] + values[test] .* G[:, hat]
                b[row] += weight * (forcing(view(xyz, :, q)) * test_value - dot(gradient_U, test_gradient))
                g[row] += weight * (readout_forcing(view(xyz, :, q)) * test_value - dot(gradient_readout, test_gradient))
                internal[row] += weight * dot(gradient_U, gradients[test])
                for trial in 1:6
                    column = rows[trial]
                    column == 0 && continue
                    A[row, column] += weight * dot(gradients[trial], test_gradient)
                    K[row, column] += weight * dot(gradients[trial], gradients[test])
                end
            end
        end
    end
    return (; A, K, b, g, internal)
end

"""Evaluate the weighted residual of a reconstructed field directly."""
function selected_test_readout(fan, system, coefficients, U, forcing; order=5)
    rule = tetra_rule(order)
    rows = Dict(edge => row for (row, edge) in enumerate(system.edges))
    total = 0.0
    for tet in fan.stars[system.node]
        ids, G = fan.tet_nodes[:, tet], fan.gradients[tet]
        hat = findfirst(==(system.node), ids)
        gradient_U = G * U[ids]
        xyz = fan.points[:, ids] * rule.barycentric
        for q in eachindex(rule.weights)
            N = rule.barycentric[:, q]
            value, gradient = 0.0, zeros(3)
            for (a, b) in POISSON_P2_EDGES
                row = get(rows, minmax(ids[a], ids[b]), 0)
                row == 0 && continue
                coefficient = coefficients[row]
                value += coefficient * 4N[a]*N[b]
                gradient .+= 4coefficient .* (N[a] .* G[:, b] + N[b] .* G[:, a])
            end
            total += fan.volumes[tet] * rule.weights[q] *
                (forcing(view(xyz, :, q)) * N[hat] * value -
                 dot(gradient_U, N[hat] .* gradient + value .* G[:, hat]))
        end
    end
    return total
end

function selected_test_physical_trace(fan, system, coefficients, boundary_triangles)
    rows = Dict(edge => row for (row, edge) in enumerate(system.edges))
    maximum_value = 0.0
    for tet in fan.stars[system.node], omitted in 1:4
        ids = fan.tet_nodes[:, tet]
        face = Tuple(sort([ids[j] for j in 1:4 if j != omitted]))
        face in boundary_triangles || continue
        N = fill(1/3, 4)
        N[omitted] = 0.0
        value = 0.0
        for (a, b) in POISSON_P2_EDGES
            row = get(rows, minmax(ids[a], ids[b]), 0)
            row == 0 || (value += coefficients[row] * 4N[a]*N[b])
        end
        maximum_value = max(maximum_value, abs(value))
    end
    return maximum_value
end

function selected_test_representative_properties(system, representatives)
    A, b, K = system.A, system.b, system.K
    Z, L = representatives.right_nullspace, representatives.left_nullspace
    states = (representatives.minimum_coefficient,
        representatives.minimum_correction_energy, representatives.minimum_total_energy)
    scale = max(norm(b), norm(A), 1.0)
    @test isposdef(Symmetric(K))
    @test norm(A * Z) <= 5e-10 * max(norm(A), eps(Float64))
    @test representatives.projected_rhs ≈ b - L * (transpose(L) * b)
    @test norm(transpose(Z) * states[1].coefficients) <= 5e-10 * scale
    @test norm(transpose(Z) * (K * states[2].coefficients)) <= 5e-10 * scale
    @test norm(transpose(Z) * (K * states[3].coefficients + system.internal)) <= 5e-10 * scale
    for state in states
        @test norm(A * state.coefficients - representatives.projected_rhs) <=
            5e-10 * max(norm(A) * norm(state.coefficients), norm(b), eps(Float64))
        @test state.residual_norm ≈ norm(A * state.coefficients - b)
        @test state.correction_energy ≈ dot(state.coefficients, K * state.coefficients)/2
        @test state.total_dirichlet_energy_change ≈
            dot(state.coefficients, K * state.coefficients)/2 + dot(system.internal, state.coefficients)
        @test state.readout ≈ dot(system.g, state.coefficients)
    end
    perturbation = Z * [0.17sin(1.31j) for j in axes(Z, 2)]
    correction = states[2].coefficients
    total = states[3].coefficients
    E(x) = dot(x, K*x)/2
    T(x) = E(x) + dot(system.internal, x)
    @test E(correction) <= E(states[1].coefficients) + 1e-12 * scale
    @test T(total) <= T(states[1].coefficients) + 1e-12 * scale
    @test E(correction) <= E(correction+perturbation) + 1e-12 * scale
    @test E(correction) <= E(correction-perturbation) + 1e-12 * scale
    @test T(total) <= T(total+perturbation) + 1e-12 * scale
    @test T(total) <= T(total-perturbation) + 1e-12 * scale
    return states
end

function selected_test_polynomial_data(mesh, fan, primal, polynomial, forcing)
    U = [fan.boundary_nodes[node] ? 0.0 : polynomial(view(fan.points, :, node))
        for node in axes(fan.points, 2)]
    rhs = zeros(length(U))
    for tet in eachindex(fan.volumes), node in fan.tet_nodes[:, tet]
        rhs[node] += forcing * fan.volumes[tet]/4
    end
    supplied = merge(primal, (;u=U, rhs, energy=dot(U, primal.K*U)/2))
    data = tetrahedral_p1_estimator_data(mesh, fan, supplied, (;rhs=x -> forcing))
    return data
end

@testset "Selected weighted patch representatives" begin
    BLAS.set_num_threads(1)
    problem = manufactured_problem(:cube_smooth)
    regular = irregular_tetrahedral_mesh(:cube, 3; amplitude=0.0)
    perturbed = irregular_tetrahedral_mesh(:cube, 3; amplitude=0.08)
    refined = locally_refined_tetrahedral_mesh(perturbed; stride=7)
    for (label, control) in ((:regular, regular), (:perturbed, perturbed), (:refined, refined))
        @testset "$label" begin
            mesh, fan = control.mesh, control.fan
            primal = solve_tetrahedral_p1(fan, problem)
            data = tetrahedral_p1_estimator_data(mesh, fan, primal, problem)
            boundary_triangles = poisson_refinement_face_audit(fan.tet_nodes).boundary_triangles
            physical_edges = Set{Tuple{Int,Int}}()
            for (a, b, c) in boundary_triangles
                push!(physical_edges, minmax(a,b), minmax(b,c), minmax(c,a))
            end
            interior = selected_test_interior_node(fan)
            boundary = findfirst(fan.boundary_nodes)
            for node in (interior, boundary)
                system = selected_weighted_patch_system(fan, data, node)
                all_edges = Set(minmax(fan.tet_nodes[a,tet], fan.tet_nodes[b,tet])
                    for tet in fan.stars[node] for (a,b) in POISSON_P2_EDGES)
                @test Set(system.edges) == setdiff(all_edges, physical_edges)
                @test isempty(intersect(Set(system.edges), physical_edges))
                direct = selected_test_quadrature(fan, system, data.U, problem.rhs)
                @test isapprox(system.A, direct.A; rtol=2e-11, atol=2e-14)
                @test isapprox(system.K, direct.K; rtol=2e-11, atol=2e-14)
                @test isapprox(system.b, direct.b; rtol=2e-11, atol=2e-15)
                @test isapprox(system.g, direct.g; rtol=2e-11, atol=2e-15)
                @test isapprox(system.internal, direct.internal; rtol=2e-11, atol=2e-15)
                representatives = selected_weighted_representatives(system)
                states = selected_test_representative_properties(system, representatives)
                for state in states
                    @test isapprox(state.readout,
                        selected_test_readout(fan, system, state.coefficients, data.U, problem.rhs);
                        rtol=2e-10, atol=2e-15)
                    @test selected_test_physical_trace(fan, system, state.coefficients,
                        boundary_triangles) < 1e-13
                end
            end

            # This checks the entire proposed equation with exact quadratic
            # nodal data, without requiring a selector to recover that field.
            quadratic(x) = 0.4 + 0.7x[1] - 0.3x[2] + 0.2x[3] +
                x[1]^2 + 2x[2]^2 + 3x[3]^2 +
                0.5x[1]*x[2] - 0.4x[1]*x[3] + 0.6x[2]*x[3]
            qdata = selected_test_polynomial_data(mesh, fan, primal, quadratic, -12.0)
            qsystem = selected_weighted_patch_system(fan, qdata, interior)
            exact_correction = [quadratic((fan.points[:,a]+fan.points[:,b])/2) -
                (qdata.U[a]+qdata.U[b])/2 for (a,b) in qsystem.edges]
            @test isapprox(qsystem.A * exact_correction, qsystem.b;
                rtol=2e-10, atol=5e-14)
            qselected = selected_weighted_representatives(qsystem)
            @test qselected.residual_norm <= 2e-10 * max(norm(qsystem.b), eps(Float64))
            quadratic_defects = [norm(getproperty(qselected, selector).coefficients-exact_correction)
                for selector in (:minimum_coefficient, :minimum_correction_energy, :minimum_total_energy)]
            quadratic_exact_readout = dot(qsystem.g, exact_correction)
            quadratic_readout_defects = [getproperty(qselected, selector).readout-quadratic_exact_readout
                for selector in (:minimum_coefficient, :minimum_correction_energy, :minimum_total_energy)]

            affine(x) = 0.7 + 0.2x[1] - 0.3x[2] + 0.4x[3]
            adata = selected_test_polynomial_data(mesh, fan, primal, affine, 0.0)
            asystem = selected_weighted_patch_system(fan, adata, interior)
            aselected = selected_weighted_representatives(asystem)
            @test norm(asystem.b) < 5e-13
            @test norm(aselected.minimum_coefficient.coefficients) < 5e-11
            @test norm(aselected.minimum_correction_energy.coefficients) < 5e-11
            for selector in (:minimum_coefficient, :minimum_correction_energy, :minimum_total_energy)
                @test abs(getproperty(aselected, selector).readout) < 5e-13
            end
            println("SELECTED_PATCH_CONSISTENCY ", (;geometry=label, node=interior,
                quadratic_equation_defect=norm(qsystem.A*exact_correction-qsystem.b),
                quadratic_selector_defects=quadratic_defects,
                quadratic_exact_readout, quadratic_readout_defects,
                affine_total_energy_field_change=norm(aselected.minimum_total_energy.coefficients),
                affine_total_energy_readout=aselected.minimum_total_energy.readout))

            # Separate primal readout and dual solve, as required by linear-goal DWR.
            dual_problem = (;rhs=x -> 1.0)
            dual = solve_tetrahedral_p1(fan, dual_problem)
            dual_data = tetrahedral_p1_estimator_data(mesh, fan, dual, dual_problem)
            mixed = selected_weighted_patch_system(fan, dual_data, interior; readout_data=data)
            mixed_direct = selected_test_quadrature(fan, mixed, dual_data.U, dual_problem.rhs;
                readout_U=data.U, readout_forcing=problem.rhs)
            @test isapprox(mixed.b, mixed_direct.b; rtol=2e-11, atol=2e-15)
            @test isapprox(mixed.g, mixed_direct.g; rtol=2e-11, atol=2e-15)
            @test norm(mixed.b-mixed.g) > 1e-10
        end
    end

    @testset "Projection and rank threshold" begin
        A = Matrix(Diagonal([2.0, 0.5, 0.0, 0.0]))
        b, g = [1.0, 2.0, 0.3, -0.4], [0.7, -0.1, 0.2, 0.6]
        K = Matrix(Diagonal([2.0, 3.0, 4.0, 5.0])) + fill(0.1, 4, 4)
        internal = [0.3, -0.2, 0.5, 0.7]
        system = (;A, b, g, K, internal)
        selected = selected_weighted_representatives(system)
        @test selected.rank == 2
        @test selected.nullity == 2
        @test selected.projected_rhs ≈ [1.0, 2.0, 0.0, 0.0]
        @test selected.minimum_coefficient.coefficients ≈ [0.5, 4.0, 0.0, 0.0]
        @test selected.residual_norm ≈ 0.5
        selected_test_representative_properties(system, selected)
        perturbed_system = merge(system, (;A=Matrix(Diagonal([2.0, 0.5, 1e-13, 0.0]))))
        variants = [selected_weighted_representatives(perturbed_system;rank_tolerance=tolerance)
            for tolerance in (1e-10, 1e-11, 1e-12)]
        @test all(variant.rank == 2 for variant in variants)
        for variant in variants, selector in
                (:minimum_coefficient, :minimum_correction_energy, :minimum_total_energy)
            @test isapprox(getproperty(variant, selector).coefficients,
                getproperty(selected, selector).coefficients; rtol=1e-11, atol=1e-12)
        end
    end
end
