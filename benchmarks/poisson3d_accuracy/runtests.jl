using Test
include("load.jl")
BLAS.set_num_threads(1)
include("test_problems_quadrature.jl")
include("test_geometry_primal.jl")

@testset "Independent P2 integration and PU-weighted residual" begin
    vertices = [0.0 1.1 0.2 0.1; 0.0 0.1 0.9 0.2; 0.0 0.2 0.1 1.2]
    G,volume = poisson_tetra_geometry(vertices,1:4)
    rule = tetra_rule(4)
    integrated_K = zeros(10,10)
    integrated_internal = zeros(10)
    integrated_weighted = zeros(10,4)
    gradient_U = [0.3,-0.7,0.2]
    for q in eachindex(rule.weights)
        N = rule.barycentric[:,q]
        values = poisson_p2_values(N)
        gradients = poisson_p2_gradients(G,N)
        @test sum(values) ≈ 1 atol=2e-14
        @test norm(vec(sum(gradients;dims=2))) < 2e-13
        integrated_K .+= rule.weights[q].*(transpose(gradients)*gradients)
        integrated_internal .+= rule.weights[q].*(transpose(gradients)*gradient_U)
        for hat in 1:4, shape in 1:10
            product_gradient = N[hat]*gradients[:,shape]+values[shape]*G[:,hat]
            integrated_weighted[shape,hat] += rule.weights[q]*dot(gradient_U,product_gradient)
        end
    end
    analytic_K = [poisson_p2_stiffness_mean(G,i,j) for i in 1:10,j in 1:10]
    @test analytic_K ≈ integrated_K rtol=2e-12 atol=2e-13
    for hat in 1:4,shape in 1:10
        ordinary,weighted = poisson_internal_means(transpose(G)*gradient_U,shape,hat)
        @test ordinary ≈ integrated_internal[shape] atol=2e-13
        @test weighted ≈ integrated_weighted[shape,hat] atol=2e-13
    end
end

@testset "Reduced Galerkin spaces and manufactured energy identity" begin
    for name in (:cube_smooth,:lshape_smooth,:lshape_singular)
        problem = manufactured_problem(name)
        mesh = build_mesh(problem.domain,2)
        fan = build_fan(mesh)
        primal = solve_primal(mesh,fan,problem;quadrature_order=6)
        data = poisson_estimator_data(mesh,fan,primal,problem;quadrature_order=6)
        estimates = estimate_errors(mesh,fan,primal,problem;quadrature_order=6)
        @test primal.algebraic_residual < 2e-12
        @test dot(primal.u,primal.rhs) ≈ 2primal.energy rtol=3e-12
        @test estimates.full.correction == estimates.reduced.correction
        @test estimates.reduced.statistics.full_fallback_count == 0
        @test estimates.reduced.statistics.reduced_patches > 0
        @test maximum(estimates.reduced.ranks) <= 10
        for estimate in (estimates.full,estimates.reduced,estimates.sdwr)
            @test all(isfinite,estimate.indicators)
            @test all(>=(0),estimate.indicators)
            @test sum(estimate.indicators) ≈ estimate.magnitude rtol=2e-13
            @test sum(estimate.nodal)+sum(estimate.correction) ≈ estimate.signed atol=2e-13
        end
        @test estimates.sdwr.correction == estimates.full.correction
        @test estimates.sdwr.signed ≈ estimates.sdwr.statistics.stabilization_part+
            estimates.sdwr.statistics.load_consistency_part rtol=2e-12
        for node in eachindex(fan.stars)
            dofs = poisson_patch_dofs(fan,node,fan.boundary_nodes)
            K,b,g = poisson_patch_system(fan,data,node,dofs)
            isempty(b) && continue
            Q,rank = poisson_reduced_basis(fan,node,dofs)
            @test 0 < rank <= min(10,length(b))
            @test transpose(Q)*Q ≈ Matrix{Float64}(I,rank,rank) atol=5e-13
            full = cholesky(Symmetric(K))\b
            reduced = Q*(cholesky(Symmetric(transpose(Q)*K*Q))\(transpose(Q)*b))
            @test norm(transpose(Q)*(b-K*reduced)) <= 1e-11*max(norm(b),1e-12)
            omitted = full-reduced
            @test dot(full,K*full)-dot(reduced,K*reduced) ≈ dot(omitted,K*omitted) atol=2e-13
            if rank == length(b)
                @test dot(g,full) ≈ dot(g,reduced) atol=2e-13
            end
        end
        if name != :lshape_singular
            truth = exact_cell_diagnostics(mesh,fan,primal,problem;quadrature_order=8)
            @test truth.integrated_exact_energy ≈ problem.reference_energy rtol=2e-12
            # Replacing the approximate reconstruction by the exact error must
            # recover J_exact-J_h, independently of either patch solve.
            oracle_estimate = sum(data.conversion)+sum(data.correction)+sum(truth.reconstruction_error)
            @test oracle_estimate ≈ problem.reference_energy-primal.energy rtol=3e-11 atol=2e-14
        end
    end
end

include("test_sdwr.jl")

@testset "Affine patch and mesh convergence" begin
    mesh = build_mesh(:cube,2)
    fan = build_fan(mesh)
    homogeneous = (;rhs=x -> 0.0)
    zero_solution = solve_primal(mesh,fan,homogeneous)
    affine = [0.1+0.3x[1]-0.2x[2]+0.4x[3] for x in eachcol(mesh.points)]
    primal = merge(zero_solution,(;u=affine,energy=dot(affine,zero_solution.K*affine)/2))
    estimates = estimate_errors(mesh,fan,primal,homogeneous)
    @test estimates.full.magnitude < 2e-13
    @test estimates.reduced.magnitude < 2e-13
    @test estimates.sdwr.magnitude < 2e-13
    for name in (:cube_smooth,:lshape_smooth,:lshape_singular)
        problem = manufactured_problem(name)
        errors = Float64[]
        for n in (2,4)
            mesh = build_mesh(problem.domain,n)
            primal = solve_primal(mesh,build_fan(mesh),problem;quadrature_order=6)
            push!(errors,abs(problem.reference_energy-primal.energy))
        end
        @test errors[2] < errors[1]
    end
end

@testset "Quadrature refinement is separate from estimator comparison" begin
    problem = manufactured_problem(:lshape_singular)
    mesh = build_mesh(problem.domain,2)
    fan = build_fan(mesh)
    samples = NamedTuple[]
    for order in (6,8)
        primal = solve_primal(mesh,fan,problem;quadrature_order=order)
        estimates = estimate_errors(mesh,fan,primal,problem;quadrature_order=order)
        push!(samples,(;energy=primal.energy,full=estimates.full.signed,reduced=estimates.reduced.signed))
    end
    for field in (:energy,:full,:reduced)
        a,b = getproperty(samples[1],field),getproperty(samples[2],field)
        @test abs(a-b) < 2e-3*max(abs(b),1e-12)
    end
    println("QUADRATURE_CHECK ",samples)
end
