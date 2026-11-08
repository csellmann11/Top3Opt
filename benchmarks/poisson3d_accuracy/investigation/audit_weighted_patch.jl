isdefined(@__MODULE__, :poisson_estimator_data) ||
    include(joinpath(@__DIR__, "..", "load.jl"))
isdefined(@__MODULE__, :irregular_tetrahedral_mesh) ||
    include(joinpath(@__DIR__, "irregular_tetrahedra.jl"))

"""Choose an interior face-center star without any physical-boundary vertices."""
function weighted_audit_node(fan)
    candidates = Int[]
    for node in fan.vertex_count + 1:fan.vertex_count + fan.face_count
        vertices = unique(vec(fan.tet_nodes[:, fan.stars[node]]))
        all(v -> !fan.boundary_nodes[v], vertices) && push!(candidates, node)
    end
    isempty(candidates) && error("Use a finer cube so the audit star is wholly interior")
    return candidates[argmin([sum(abs2, fan.points[:, node] .- 0.5) for node in candidates])]
end

weighted_audit_monomials(x) =
    [1.0, x[1], x[2], x[3], x[1]^2, x[2]^2, x[3]^2,
     x[1]*x[2], x[1]*x[3], x[2]*x[3]]

"""Audit the proposed free-edge P2 trial / N_i-times-P2 test formulation.

All existing P1 vertex values are fixed, while every edge midpoint on the star
is free, including its artificial boundary. This is an algebraic nullspace
audit, not a polynomial-fitting reconstruction. Harmonic polynomials are used
only to exhibit exact invisible modes of the proposed local operator.
"""
function weighted_patch_audit_case(; n=4, amplitude=0.08, node=nothing,
                                     quadrature_order=5, rank_tolerance=1e-11)
    println("WEIGHTED_PATCH preparing n=$n amplitude=$amplitude")
    flush(stdout)
    control = irregular_tetrahedral_mesh(:cube, n; amplitude)
    mesh, fan = control.mesh, control.fan
    node = isnothing(node) ? weighted_audit_node(fan) : node
    tets = fan.stars[node]
    vertices = sort!(unique(vec(fan.tet_nodes[:, tets])))
    all(v -> !fan.boundary_nodes[v], vertices) ||
        error("The audit must not release physical Dirichlet DOFs")
    problem = manufactured_problem(:cube_smooth)
    primal = solve_tetrahedral_p1(fan, problem; quadrature_order=6)
    data = tetrahedral_p1_estimator_data(mesh, fan, primal, problem; quadrature_order=6)

    edges = sort!(unique([poisson_ordered_edge(fan.tet_nodes[a, tet], fan.tet_nodes[b, tet])
        for tet in tets for (a, b) in POISSON_P2_EDGES]))
    edge_rows = Dict(edge => row for (row, edge) in enumerate(edges))
    ndofs = length(edges)
    A, gram, g = zeros(ndofs, ndofs), zeros(ndofs, ndofs), zeros(ndofs)
    rule = tetra_rule(quadrature_order)
    for tet in tets
        ids, G = view(fan.tet_nodes, :, tet), fan.gradients[tet]
        hat = findfirst(==(node), ids)
        rows = [edge_rows[poisson_ordered_edge(ids[a], ids[b])] for (a, b) in POISSON_P2_EDGES]
        for (test, row) in enumerate(rows)
            g[row] += data.weighted_residual[4 + test, hat, tet]
        end
        for q in eachindex(rule.weights)
            N = view(rule.barycentric, :, q)
            values, gradients = poisson_p2_values(N), poisson_p2_gradients(G, N)
            weight, Ni = fan.volumes[tet] * rule.weights[q], N[hat]
            for (test, row) in enumerate(rows), (trial, column) in enumerate(rows)
                s, r = 4 + test, 4 + trial
                A[row, column] += weight *
                    (Ni * dot(view(gradients, :, r), view(gradients, :, s)) +
                     values[s] * dot(view(gradients, :, r), view(G, :, hat)))
                gram[row, column] += weight * Ni^2 * values[r] * values[s]
            end
        end
    end

    decomposition = svd(A; full=true)
    threshold = rank_tolerance * maximum(decomposition.S)
    rank = count(>(threshold), decomposition.S)
    left_null, right_null = decomposition.U[:, rank+1:end], decomposition.V[:, rank+1:end]
    @assert rank < ndofs "This seven-vertex star must retain harmonic null modes"

    # Scaled coordinates keep this analytical polynomial nullspace calculation
    # independent of patch size. The final row imposes Delta(p)=0.
    center = view(fan.points, :, node)
    h = maximum(norm(view(fan.points, :, vertex) - center) for vertex in vertices)
    constraints = zeros(length(vertices) + 1, 10)
    for (row, vertex) in enumerate(vertices)
        constraints[row, :] .= weighted_audit_monomials((fan.points[:, vertex] - center) / h)
    end
    constraints[end, 5:7] .= 1.0
    polynomial_svd = svd(constraints; full=true)
    polynomial_rank = count(s -> s > rank_tolerance * maximum(polynomial_svd.S), polynomial_svd.S)
    harmonic_polynomials = polynomial_svd.V[:, polynomial_rank+1:end]
    edge_values = zeros(ndofs, 10)
    for (row, (a, b)) in enumerate(edges)
        midpoint = (fan.points[:, a] + fan.points[:, b]) / 2
        edge_values[row, :] .= weighted_audit_monomials((midpoint - center) / h)
    end
    modes = edge_values * harmonic_polynomials
    modes_svd = svd(modes; full=false)
    harmonic_count = size(harmonic_polynomials, 2)
    @assert minimum(modes_svd.S) > rank_tolerance * maximum(modes_svd.S)
    harmonic_edges = modes_svd.U[:, 1:harmonic_count]
    # Choose the harmonic direction with greatest readout sensitivity, rather
    # than accidentally choosing one that is also a left-null vector.
    sensitivity_svd = svd(transpose(A) * harmonic_edges; full=false)
    invisible_mode = harmonic_edges * sensitivity_svd.V[:, 1]
    compatible_delta = transpose(A) * invisible_mode
    compatible_rhs = A * compatible_delta
    changed_delta = compatible_delta + invisible_mode
    estimator_change = dot(compatible_rhs, invisible_mode) / 2
    algebra_scale = max(norm(A), eps(Float64))
    @assert norm(A * harmonic_edges) <= 1e-9 * algebra_scale
    @assert norm(A * changed_delta - compatible_rhs) <= 1e-9 * algebra_scale
    @assert isapprox(2estimator_change, sum(abs2, compatible_delta);
        rtol=1e-9, atol=1e-20 * algebra_scale^2)

    # The weighted test functions are independent. Their positive-definite L2
    # Gram matrix constructs an actual body load f=sum(alpha_j N_i phi_j) with
    # any specified moment vector (here the compatible witness RHS, with U=0).
    load_coefficients = cholesky(Symmetric(gram)) \ compatible_rhs
    load_realization_error = norm(gram * load_coefficients - compatible_rhs) /
        max(norm(compatible_rhs), eps(Float64))
    @assert load_realization_error < 1e-9
    @assert amplitude == 0 || estimator_change > 1e-12 * algebra_scale^2

    rhs_scale = max(norm(g), eps(Float64))
    compatibility = norm(transpose(left_null) * g)
    output_sensitivity = norm(transpose(right_null) * g)
    row = (; n, amplitude, node, patch_tetrahedra=length(tets),
        fixed_p1_vertices=length(vertices), free_p2_edge_dofs=ndofs,
        matrix_rank=rank, left_nullity=size(left_null, 2), right_nullity=size(right_null, 2),
        harmonic_nullity=harmonic_count, rank_relative_tolerance=rank_tolerance,
        largest_singular_value=maximum(decomposition.S),
        smallest_nonzero_singular_value=decomposition.S[rank],
        largest_null_singular_value=decomposition.S[rank+1],
        actual_rhs_norm=norm(g), actual_rhs_compatibility_defect=compatibility,
        actual_rhs_compatibility_relative=compatibility / rhs_scale,
        actual_readout_right_null_component=output_sensitivity,
        actual_readout_right_null_relative=output_sensitivity / rhs_scale,
        maximum_unit_null_estimator_change=output_sensitivity / 2,
        harmonic_right_null_residual=norm(A * harmonic_edges),
        selected_harmonic_left_residual=norm(transpose(A) * invisible_mode),
        compatible_rhs_left_null_defect=norm(transpose(left_null) * compatible_rhs),
        compatible_equation_change=norm(A * changed_delta - compatible_rhs),
        compatible_estimator_change=estimator_change,
        load_realization_relative_error=load_realization_error,
        trial_space="continuous P2, all P1 vertices fixed, every edge midpoint free",
        test_space="N_i times the free P2 edge basis", quadrature_order)
    return (; row, A, g, gram, edges, coordinates=fan.points, node, tets,
        left_null, right_null, harmonic_polynomials, harmonic_edges,
        invisible_mode, compatible_delta, compatible_rhs, load_coefficients,
        singular_values=decomposition.S)
end

function run_weighted_patch_audit(; n=4, amplitudes=(0.0, 0.08),
        output=joinpath(@__DIR__, "..", "results", "investigation", "weighted_patch_audit.csv"))
    BLAS.set_num_threads(1)
    rows, modes = NamedTuple[], NamedTuple[]
    fixed_node = nothing
    mkpath(dirname(abspath(output)))
    for amplitude in amplitudes
        audit = weighted_patch_audit_case(; n, amplitude, node=fixed_node)
        fixed_node = audit.node
        push!(rows, audit.row)
        for (kind, basis) in ((:left_null, audit.left_null), (:right_null, audit.right_null),
                              (:harmonic_right_null, audit.harmonic_edges))
            for column in axes(basis, 2), edge in axes(basis, 1)
                a, b = audit.edges[edge]
                push!(modes, (; n, amplitude, node=audit.node, space=String(kind),
                    mode=column, edge, endpoint_a=a, endpoint_b=b, coefficient=basis[edge, column]))
            end
        end
        write_csv(output, rows)
        write_csv(joinpath(dirname(output), "weighted_patch_audit_nullspaces.csv"), modes)
        println("WEIGHTED_PATCH_RESULT ", audit.row)
        flush(stdout)
    end
    return rows
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_weighted_patch_audit()
end
