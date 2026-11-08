# Included after load.jl by the standalone benchmark's test runner.

@testset "Cartesian geometry and continuous fan" begin
    for domain in (:cube,:lshape), n in (1,2)
        mesh = build_mesh(domain,n)
        fan = build_fan(mesh)
        nc = length(mesh.cells)
        expected_volume = domain == :cube ? 1.0 : 3.0
        @test mesh.domain == domain
        @test nc == (domain == :cube ? n^3 : 3n^3)
        @test size(fan.tet_nodes,2) == 24nc
        @test sum(mesh.volumes) ≈ expected_volume atol=1e-13
        @test sum(fan.volumes) ≈ expected_volume atol=1e-13
        @test all(>(0),fan.volumes)
        @test all(length(ids) == 8 for ids in mesh.cells)
        @test all(length(ids) == 4 for ids in mesh.faces)
        @test all(length(ids) == 6 for ids in mesh.cell_faces)
        @test all(count(==(cell),fan.tet_cells) == 24 for cell in 1:nc)
        if domain == :lshape
            @test all(!(x > 0 && y < 0) for (x,y) in zip(mesh.centers[1,:],mesh.centers[2,:]))
        end

        # Check ownership from actual cell incidences, including boundary faces.
        owners = [Int[] for _ in mesh.faces]
        for (cell,faces) in enumerate(mesh.cell_faces), face in faces
            push!(owners[face],cell)
        end
        @test all(length(ids) in (1,2) for ids in owners)
        @test all(sort(ids) == sort(filter(!iszero,collect(mesh.face_cells[face])))
            for (face,ids) in enumerate(owners))
        boundary_vertices = Set{Int}()
        for (face,ids) in enumerate(owners)
            length(ids) == 1 && union!(boundary_vertices,mesh.faces[face])
        end
        @test Set(findall(mesh.boundary_vertices)) == boundary_vertices
        @test fan.boundary_nodes[1:fan.vertex_count] == mesh.boundary_vertices
        @test all(fan.boundary_nodes[fan.vertex_count+face] == (length(ids) == 1)
            for (face,ids) in enumerate(owners))
        @test !any(fan.boundary_nodes[fan.vertex_count+fan.face_count+1:end])
        @test all(all(fan.boundary_nodes[node] for node in edge) for edge in fan.boundary_edges)

        affine_gradient = [0.2,-0.3,0.7]
        affine(x) = 1.0+dot(affine_gradient,x)
        u = [affine(mesh.points[:,i]) for i in axes(mesh.points,2)]
        U = reconstruct_fan(mesh,fan,u)
        @test U ≈ [affine(fan.points[:,i]) for i in axes(fan.points,2)] atol=2e-14

        nonlinear(x) = sin(x[1])+x[2]*x[3]+x[1]^2
        trial = [nonlinear(mesh.points[:,i]) for i in axes(mesh.points,2)]
        reconstructed = reconstruct_fan(mesh,fan,trial)
        for cell in 1:nc
            data = poisson_cell_operator(mesh,cell)
            ids = mesh.cells[cell]
            X = hcat(ones(8),transpose(mesh.points[:,ids]))
            @test data.P*X ≈ X atol=2e-14
            @test data.P*ones(8) ≈ ones(8) atol=2e-14
            @test data.P*data.P ≈ data.P atol=2e-14
            @test data.G*ones(8) ≈ zeros(3) atol=2e-14
            @test data.G*transpose(mesh.points[:,ids]) ≈ Matrix{Float64}(I,3,3) atol=2e-14
            @test data.K ≈ transpose(data.K) atol=2e-14
            @test norm(data.K*ones(8)) <= 2e-13*norm(data.K)
            spectrum = eigvals(Symmetric(data.K))
            @test abs(first(spectrum)) <= 2e-13*last(spectrum)
            @test spectrum[2] > 1e-10*last(spectrum)
            @test dot(u[ids],data.K*u[ids]) ≈ mesh.volumes[cell]*sum(abs2,affine_gradient) atol=2e-14

            # P1 tetra integration is exactly volume times the four-node mean.
            integral = 0.0
            volume = 0.0
            for tet in findall(==(cell),fan.tet_cells)
                integral += fan.volumes[tet]*sum(reconstructed[fan.tet_nodes[:,tet]])/4
                volume += fan.volumes[tet]
            end
            @test volume ≈ mesh.volumes[cell] atol=2e-14
            @test integral ≈ mesh.volumes[cell]*sum(trial[ids])/8 atol=2e-14
        end
    end
end

@testset "Scalar assembly and homogeneous Dirichlet solve" begin
    for domain in (:cube,:lshape)
        mesh = build_mesh(domain,2)
        fan = build_fan(mesh)
        solution = solve_primal(mesh,fan,(rhs=x -> 1.0,);quadrature_order=3)
        @test sum(solution.rhs) ≈ sum(mesh.volumes) atol=2e-13
        @test solution.algebraic_residual < 1e-12
        @test all(iszero,solution.u[mesh.boundary_vertices])
        @test solution.energy ≈ dot(solution.u,solution.rhs)/2 rtol=2e-13
        @test solution.energy > 0
        @test !isempty(solution.free_dofs)
        @test minimum(eigvals(Symmetric(Matrix(solution.K[solution.free_dofs,solution.free_dofs])))) > 0
        # The returned K/rhs retain the unconstrained system for energy and DWR.
        @test norm(solution.K*ones(length(solution.u))) < 2e-13*norm(solution.K)
        @test any(!iszero,solution.rhs[mesh.boundary_vertices])
        zero_solution = solve_primal(mesh,fan,(rhs=x -> 0.0,);quadrature_order=2)
        @test iszero(zero_solution.u)
        @test iszero(zero_solution.energy)
        @test iszero(zero_solution.algebraic_residual)
    end
end
