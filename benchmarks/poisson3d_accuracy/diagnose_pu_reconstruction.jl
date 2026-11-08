include("load.jl")

"""Retain the existing patch solutions as tetrahedral P2 coefficients.

Only the central vertex and radial edge shapes are populated. Consequently
each local correction has exactly the same zero artificial/physical trace as
the estimator; no fitting or new reconstruction space is introduced.
"""
function poisson_scatter_patch!(coefficients, fan, node, dofs, delta)
    for tet in fan.stars[node]
        ids = view(fan.tet_nodes, :, tet)
        hat = findfirst(==(node), ids)
        dofs.center_row == 0 ||
            (coefficients[hat, hat, tet] = delta[dofs.center_row])
        for (edge, (a, b)) in enumerate(POISSON_P2_EDGES)
            (a == hat || b == hat) || continue
            other = ids[a == hat ? b : a]
            row = get(dofs.radial_rows, other, 0)
            row == 0 || (coefficients[4 + edge, hat, tet] = delta[row])
        end
    end
end

function poisson_retained_patch_solutions(fan, data; rank_tolerance=1e-11)
    nt = length(fan.volumes)
    full, reduced = zeros(10, 4, nt), zeros(10, 4, nt)
    full_nodal, reduced_nodal = copy(data.conversion), copy(data.conversion)
    residual_corrections = zeros(2)
    boundary_mask = poisson_boundary_mask(fan)
    for node in eachindex(data.conversion)
        dofs = poisson_patch_dofs(fan, node, boundary_mask)
        K, b, g = poisson_patch_system(fan, data, node, dofs)
        isempty(b) && continue
        delta = cholesky(Symmetric(K)) \ b
        contribution = dot(g, delta) / 2
        full_nodal[node] += contribution
        residual_corrections[1] += contribution
        poisson_scatter_patch!(full, fan, node, dofs, delta)
        Q, rank = poisson_reduced_basis(fan, node, dofs; rank_tolerance)
        if rank > 0
            reduced_K = transpose(Q) * K * Q
            reduced_b = transpose(Q) * b
            reduced_delta = Q * (cholesky(Symmetric(reduced_K)) \ reduced_b)
            contribution = dot(g, reduced_delta) / 2
            reduced_nodal[node] += contribution
            residual_corrections[2] += contribution
            poisson_scatter_patch!(reduced, fan, node, dofs, reduced_delta)
        end
    end
    signed = (sum(full_nodal) + sum(data.correction),
              sum(reduced_nodal) + sum(data.correction))
    return (; coefficients=(full, reduced), residual_corrections, signed)
end

"""P2 interpolation nodes in tetrahedral barycentric coordinates."""
function poisson_diagnostic_p2_nodes()
    nodes = zeros(4, 10)
    for i in 1:4
        nodes[i, i] = 1.0
    end
    for (edge, (a, b)) in enumerate(POISSON_P2_EDGES)
        nodes[a, 4 + edge] = nodes[b, 4 + edge] = 0.5
    end
    return nodes
end

"""Quadratic nodal coefficients of A=sum(N_i grad(delta_i)) and H=sum(delta_i grad(N_i)).

Their sum is grad(e_hat), where e_hat=sum(N_i delta_i) is piecewise cubic.
Evaluating A and H at P2 interpolation nodes represents both exactly.
"""
function poisson_blended_gradient_coefficients!(A, H, barycentric_A, G, C, nodes)
    @inbounds for sample in 1:10, component in 1:4
        value = 0.0
        for hat in 1:4
            weight = nodes[hat, sample]
            iszero(weight) && continue
            derivative = (4nodes[component, sample] - 1) * C[component, hat]
            for (edge, (a, b)) in enumerate(POISSON_P2_EDGES)
                if component == a
                    derivative += 4nodes[b, sample] * C[4 + edge, hat]
                elseif component == b
                    derivative += 4nodes[a, sample] * C[4 + edge, hat]
                end
            end
            value += weight * derivative
        end
        barycentric_A[component, sample] = value
    end
    mul!(A, G, barycentric_A)
    # C[sample,hat] is delta_hat at this P2 interpolation node.
    mul!(H, G, transpose(C))
    return nothing
end

"""Independent exact-error integration of the two retained PU fields."""
function poisson_pu_gradient_diagnostics(fan, data, problem, coefficients;
                                         quadrature_order=7)
    rule = tetra_rule(quadrature_order)
    shapes, _ = poisson_load_shapes(rule.barycentric)
    nodes = poisson_diagnostic_p2_nodes()
    nq = length(rule.weights)
    xyz, error_gradient = zeros(3, nq), zeros(3, nq)
    A, H, barycentric_A = zeros(3, 10), zeros(3, 10), zeros(4, 10)
    values_A, values_H = zeros(3, nq), zeros(3, nq)
    error_norm2 = 0.0
    norm2, norm2_A, norm2_H = zeros(2), zeros(2), zeros(2)
    cross_A, cross_H = zeros(2), zeros(2)
    nt = length(fan.volumes)
    progress_step = max(1, nt ÷ 4)
    for tet in 1:nt
        ids = view(fan.tet_nodes, :, tet)
        G, volume = fan.gradients[tet], fan.volumes[tet]
        gradient_U = G * data.U[ids]
        mul!(xyz, view(fan.points, :, ids), rule.barycentric)
        @inbounds for q in 1:nq
            exact_gradient = problem.gradient(view(xyz, :, q))
            for component in 1:3
                difference = exact_gradient[component] - gradient_U[component]
                error_gradient[component, q] = difference
                error_norm2 += volume * rule.weights[q] * difference^2
            end
        end
        for method in 1:2
            C = view(coefficients[method], :, :, tet)
            poisson_blended_gradient_coefficients!(A, H, barycentric_A, G, C, nodes)
            mul!(values_A, A, shapes)
            mul!(values_H, H, shapes)
            @inbounds for q in 1:nq
                weight = volume * rule.weights[q]
                for component in 1:3
                    a, h = values_A[component, q], values_H[component, q]
                    e = error_gradient[component, q]
                    norm2[method] += weight * (a + h)^2
                    norm2_A[method] += weight * a^2
                    norm2_H[method] += weight * h^2
                    cross_A[method] += weight * e * a
                    cross_H[method] += weight * e * h
                end
            end
        end
        if tet % progress_step == 0 || tet == nt
            println("PU gradient integration: $tet/$nt tetrahedra")
            flush(stdout)
        end
    end
    return (; error_norm2, norm2, norm2_A, norm2_H, cross_A, cross_H)
end

"""Diagnose existing cube estimators, preserving results/latest unchanged.

Order 7 integrates the smooth cube's squared exact-gradient error exactly:
its total polynomial degree is 10, with at most two additional Duffy degrees.
The patch data retain the existing order-6 load quadrature.
"""
function run_pu_reconstruction_diagnostics(; levels=(4, 6, 10),
        quadrature_order=6, truth_order=7,
        output=joinpath(@__DIR__, "results", "pu_reconstruction_diagnostics.csv"))
    truth_order >= 7 || throw(ArgumentError("Use truth_order >= 7 for exact cube integration"))
    BLAS.set_num_threads(1)
    rows = NamedTuple[]
    problem = manufactured_problem(:cube_smooth)
    mkpath(dirname(abspath(output)))
    for n in levels
        println("PU reconstruction diagnostic: cube n=$n")
        flush(stdout)
        mesh = build_mesh(problem.domain, n)
        fan = build_fan(mesh)
        primal = solve_primal(mesh, fan, problem; quadrature_order)
        data = poisson_estimator_data(mesh, fan, primal, problem; quadrature_order)
        retained = poisson_retained_patch_solutions(fan, data)
        integrals = poisson_pu_gradient_diagnostics(fan, data, problem,
            retained.coefficients; quadrature_order=truth_order)
        E = integrals.error_norm2 / 2
        B = sum(data.conversion) + sum(data.correction)
        goal_error = problem.reference_energy - primal.energy
        identity_error = abs(B + E - goal_error)
        @assert identity_error <= 1e-10 * problem.reference_energy "Exact goal decomposition failed"
        for (method, label) in enumerate((:full, :reduced))
            norm_ratio = sqrt(integrals.norm2[method] / integrals.error_norm2)
            cross = integrals.cross_A[method] + integrals.cross_H[method]
            cosine = cross / sqrt(integrals.norm2[method] * integrals.error_norm2)
            Q_direct = cross / 2
            Q_residual = retained.residual_corrections[method]
            residual_identity_error = abs(Q_direct - Q_residual)
            @assert residual_identity_error <= 1e-10 * max(E, abs(Q_residual)) "Residual/gradient identity failed"
            push!(rows, (; problem=String(problem.name), n, cells=length(mesh.cells),
                method=String(label), signed_goal_error=goal_error,
                signed_estimate=retained.signed[method],
                signed_effectivity=retained.signed[method] / goal_error,
                common_correction=B, exact_reconstruction_energy=E,
                correction_from_patch_residual=Q_residual,
                correction_from_exact_gradient=Q_direct,
                correction_effectivity=Q_direct / E,
                norm_ratio, gradient_cosine=cosine,
                norm_ratio_times_cosine=norm_ratio * cosine,
                weighted_gradient_contribution=integrals.cross_A[method] / integrals.error_norm2,
                hat_gradient_contribution=integrals.cross_H[method] / integrals.error_norm2,
                weighted_gradient_norm_ratio=sqrt(integrals.norm2_A[method] / integrals.error_norm2),
                hat_gradient_norm_ratio=sqrt(integrals.norm2_H[method] / integrals.error_norm2),
                goal_identity_absolute_error=identity_error,
                residual_identity_absolute_error=residual_identity_error,
                quadrature_order, truth_order))
            println("PU_DIAGNOSTIC ", last(rows))
        end
        write_csv(output, rows)
        flush(stdout)
    end
    println("PU reconstruction diagnostics: ", abspath(output))
    return rows
end

function pu_reconstruction_diagnostics_cli(args)
    options = Dict{String, String}()
    for argument in args
        startswith(argument, "--") && occursin('=', argument) ||
            error("Use --levels=4,6,10 --quadrature=6 --truth-order=7 --output=PATH")
        key, value = split(argument[3:end], '='; limit=2)
        key in ("levels", "quadrature", "truth-order", "output") || error("Unknown option: $key")
        options[key] = value
    end
    return run_pu_reconstruction_diagnostics(;
        levels=parse.(Int, split(get(options, "levels", "4,6,10"), ',')),
        quadrature_order=parse(Int, get(options, "quadrature", "6")),
        truth_order=parse(Int, get(options, "truth-order", "7")),
        output=get(options, "output", joinpath(@__DIR__, "results", "pu_reconstruction_diagnostics.csv")))
end

if abspath(PROGRAM_FILE) == @__FILE__
    pu_reconstruction_diagnostics_cli(ARGS)
end
