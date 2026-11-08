# Included by the standalone validation runner after load.jl and using Test.

@testset "Normalized tetrahedral polynomial moments" begin
    @test_throws ArgumentError gauss_legendre01(0)
    @test_throws ArgumentError tetra_rule(1)
    for order in (1,2,4)
        nodes,weights = gauss_legendre01(order)
        @test all(x -> 0 < x < 1,nodes)
        @test all(>(0),weights)
        for degree in 0:2order-1
            @test sum(weights .* nodes.^degree) ≈ 1/(degree+1) atol=2e-14 rtol=2e-13
        end
    end
    for order in (4,5)
        rule = tetra_rule(order)
        barycentric,weights = rule.barycentric,rule.weights
        @test size(barycentric) == (4,order^3)
        @test sum(weights) ≈ 1.0 atol=2e-14
        @test all(>(0),weights)
        @test all(x -> 0 < x < 1,barycentric)
        @test vec(sum(barycentric;dims=1)) ≈ ones(length(weights)) atol=2e-14
        # Dirichlet-simplex moments are independent of the Duffy mapping:
        # normalized integral(prod(lambda_i^a_i)) = 3!*prod(a_i!)/(3+sum(a_i))!.
        for a in 0:4,b in 0:4,c in 0:4,d in 0:4
            degree = a+b+c+d
            degree <= 4 || continue
            powers = (a,b,c,d)
            expected = 6prod(factorial,powers)/factorial(3+degree)
            observed = sum(weights[q]*prod(barycentric[i,q]^powers[i] for i in 1:4)
                for q in eachindex(weights))
            @test observed ≈ expected atol=2e-14 rtol=2e-13
        end
    end
end

@testset "Physical tetrahedral affine moments" begin
    reference = [0.0 1.0 0.0 0.0;0.0 0.0 1.0 0.0;0.0 0.0 0.0 1.0]
    transform = [1.2 0.3 -0.1;0.2 0.8 0.25;0.1 -0.15 0.6]
    translated = transform*reference .+ [0.2,-0.3,0.4]
    anisotropic = [0.001 0.0 0.0;0.0 2.0 0.0;0.0 0.0 0.05]*reference
    rule = tetra_rule(4)
    for vertices in (reference,translated,anisotropic), permutation in ((1,2,3,4),(2,1,3,4))
        points = vertices[:,collect(permutation)]
        volume = abs(det(hcat(points[:,2]-points[:,1],points[:,3]-points[:,1],
            points[:,4]-points[:,1])))/6
        centroid = vec(sum(points;dims=2))/4
        quadrature = tetra_quadrature(points,rule)
        @test quadrature.barycentric === rule.barycentric
        @test sum(quadrature.weights) ≈ volume rtol=2e-13
        @test quadrature.points*quadrature.weights ≈ volume*centroid atol=2e-14 rtol=2e-13
        coefficients = [0.7,-0.9,0.2]
        affine(x) = 1.2+dot(coefficients,x)
        observed = sum(quadrature.weights[q]*affine(quadrature.points[:,q])
            for q in eachindex(quadrature.weights))
        @test observed ≈ volume*affine(centroid) atol=2e-14 rtol=2e-13
        rebuilt = tetra_quadrature(points,4)
        @test rebuilt.points ≈ quadrature.points atol=2e-14
        @test rebuilt.weights ≈ quadrature.weights atol=2e-14
    end
    @test_throws DimensionMismatch tetra_quadrature(zeros(4,4),rule)
    @test_throws ArgumentError tetra_quadrature(zeros(3,4),rule)
    @test_throws ArgumentError tetra_quadrature(fill(NaN,3,4),rule)
end

@testset "Manufactured gradients and Poisson right-hand sides" begin
    for name in (:cube_smooth,:lshape_smooth,:lshape_singular)
        problem = manufactured_problem(name)
        points = problem.domain == :cube ?
            ((0.23,0.37,0.61),(0.72,0.18,0.43)) :
            ((0.31,0.42,0.57),(-0.36,0.29,0.41),(-0.41,-0.27,0.63),(-0.08,0.06,0.37))
        for point in points
            x = collect(point)
            gradient = problem.gradient(x)
            @test length(gradient) == 3
            @test all(isfinite,gradient)
            h = 3e-4
            laplacian = 0.0
            for component in 1:3
                offset = zeros(3)
                offset[component] = h
                fm2,fm1 = problem.exact(x-2offset),problem.exact(x-offset)
                fp1,fp2 = problem.exact(x+offset),problem.exact(x+2offset)
                derivative = (fm2-8fm1+8fp1-fp2)/(12h)
                @test derivative ≈ gradient[component] atol=2e-8 rtol=2e-7
                laplacian += (-fp2+16fp1-30problem.exact(x)+16fm1-fm2)/(12h*h)
            end
            @test -laplacian ≈ problem.rhs(x) atol=3e-6 rtol=3e-6
        end
    end
    @test_throws ArgumentError manufactured_problem(:unknown)
end

@testset "Homogeneous manufactured physical boundary values" begin
    for name in (:cube_smooth,:lshape_smooth,:lshape_singular)
        problem = manufactured_problem(name)
        mesh = build_mesh(problem.domain,4)
        boundary = findall(mesh.boundary_vertices)
        @test !isempty(boundary)
        for node in boundary
            @test abs(problem.exact(mesh.points[:,node])) <= 5e-15
        end
        if problem.domain == :lshape
            edge_nodes = [node for node in boundary if iszero(mesh.points[1,node]) &&
                iszero(mesh.points[2,node]) && 0 < mesh.points[3,node] < 1]
            @test length(edge_nodes) == 3
            @test all(node -> iszero(problem.exact(mesh.points[:,node])),edge_nodes)
        end
    end
    singular = manufactured_problem(:lshape_singular)
    for z in (0.0,0.25,0.5,0.75,1.0)
        @test iszero(singular.exact((0.0,0.0,z)))
        @test iszero(singular.rhs((0.0,0.0,z)))
    end
    @test_throws DomainError singular.gradient((0.0,0.0,0.5))
    @test singular.gradient((0.0,0.0,0.0)) == (0.0,0.0,0.0)
    @test singular.gradient((0.0,0.0,1.0)) == (0.0,0.0,0.0)
end

@testset "Analytic polynomial energy and singular reference convergence" begin
    nodes,weights = gauss_legendre01(5)
    for name in (:cube_smooth,:lshape_smooth)
        problem = manufactured_problem(name)
        # Independent Cartesian integration over one cube or three unit cubes.
        origins = problem.domain == :cube ? ((0.0,0.0),) :
            ((-1.0,-1.0),(-1.0,0.0),(0.0,0.0))
        energy = 0.0
        for (x0,y0) in origins,i in eachindex(nodes),j in eachindex(nodes),k in eachindex(nodes)
            gradient = problem.gradient((x0+nodes[i],y0+nodes[j],nodes[k]))
            energy += weights[i]*weights[j]*weights[k]*sum(abs2,gradient)/2
        end
        @test energy ≈ problem.reference_energy atol=2e-14 rtol=2e-13
        @test iszero(problem.reference_uncertainty)
    end
    singular = manufactured_problem(:lshape_singular)
    references = [poisson_singular_reference_energy(order) for order in (32,48,64)]
    @test all(x -> isfinite(x) && x > 0,references)
    @test singular.reference_energy ≈ references[2] atol=2eps(references[2])
    @test maximum(abs.(references .- references[2])) <= singular.reference_uncertainty
    @test 0 < singular.reference_uncertainty/singular.reference_energy < 5e-12
    # This checks observed order convergence; the uncertainty is not a
    # mathematically certified bound on the remaining quadrature error.
end
