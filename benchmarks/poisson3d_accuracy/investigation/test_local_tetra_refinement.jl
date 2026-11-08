include(joinpath(@__DIR__, "..", "load.jl"))
include("irregular_tetrahedra.jl")
include("local_tetra_refinement.jl")
using Test

@testset "Conforming local tetrahedral refinement" begin
    for domain in (:cube, :lshape)
        generated = irregular_tetrahedral_mesh(domain, 2; amplitude=0.08)
        original_points = copy(generated.fan.points)
        original_tetrahedra = copy(generated.fan.tet_nodes)
        refined = locally_refined_tetrahedral_mesh(generated; stride=7)
        mesh, fan, stats = refined.mesh, refined.fan, refined.geometry_stats
        source = generated.fan
        nt, nn = length(source.volumes), size(source.points, 2)
        added = nt ÷ 7
        @test length(fan.volumes) == nt + 3added
        @test size(fan.points, 2) == nn + added
        @test generated.fan.points == original_points
        @test generated.fan.tet_nodes == original_tetrahedra
        @test fan.points[:, 1:nn] == original_points
        @test fan.boundary_nodes[1:nn] == source.boundary_nodes
        @test !any(fan.boundary_nodes[nn+1:end])
        @test fan.boundary_edges == source.boundary_edges
        @test fan.vertex_count == source.vertex_count
        @test fan.face_count == source.face_count
        @test fan.cell_nodes == source.cell_nodes
        @test fan.refined_nodes == collect(nn+1:nn+added)
        @test stats.minimum_child_orientation_ratio ≈ 0.25
        @test stats.minimum_scaled_jacobian > 0
        @test stats.face_connected
        @test sum(fan.volumes) ≈ (domain == :cube ? 1.0 : 3.0)
        @test mesh.volumes == generated.mesh.volumes
        @test mesh.centers == generated.mesh.centers
        @test stats.maximum_parent_volume_relative_error < 1e-12
        @test stats.maximum_parent_center_error < 1e-12

        old_faces = poisson_refinement_face_audit(source.tet_nodes)
        new_faces = poisson_refinement_face_audit(fan.tet_nodes)
        @test new_faces.boundary_triangles == old_faces.boundary_triangles
        @test new_faces.interior_faces == old_faces.interior_faces + 6added
        for node in fan.refined_nodes
            @test length(fan.stars[node]) == 4
            @test length(fan.node_cells[node]) == 1
            parent = fan.refinement_parent_tetrahedra[first(fan.stars[node])]
            @test parent % 7 == 0
            @test fan.points[:, node] ≈ vec(sum(source.points[:, source.tet_nodes[:, parent]]; dims=2)) / 4
            @test all(tet -> fan.volumes[tet] ≈ source.volumes[parent]/4, fan.stars[node])
        end
        for node in 1:nn
            split_incident = count(tet -> tet % 7 == 0, source.stars[node])
            @test length(fan.stars[node]) == length(source.stars[node]) + 2split_incident
            @test fan.node_cells[node] == sort(unique(source.node_cells[node]))
        end

        affine(x) = 0.7 + x[1] - 0.3x[2] + 0.2x[3]
        affine_primal = solve_tetrahedral_p1(fan, (; rhs=x -> 0.0);
            quadrature_order=2, dirichlet=affine)
        affine_values = [affine(view(fan.points, :, node)) for node in axes(fan.points, 2)]
        @test maximum(abs, affine_primal.u-affine_values) < 5e-12
        @test affine_primal.algebraic_residual < 5e-12

        case = domain == :cube ? :cube_smooth : :lshape_smooth
        problem = manufactured_problem(case)
        primal = solve_tetrahedral_p1(fan, problem)
        data = tetrahedral_p1_estimator_data(mesh, fan, primal, problem)
        truth = tetrahedral_p1_truth(mesh, fan, primal, problem; quadrature_order=10)
        @test primal.algebraic_residual < 5e-12
        @test abs(data.common_correction) < 1e-11 * primal.energy
        @test sum(truth.discrete_energy) ≈ primal.energy
        @test data.common_correction + sum(truth.reconstruction_error) ≈
            problem.reference_energy-primal.energy

        # Independently express each P1 hat in the P2 basis to test compatibility
        # of the existing estimator residuals on the newly connected mesh.
        hat_residual = zeros(length(primal.u))
        for tet in eachindex(fan.volumes), local_node in 1:4
            value = data.residual[local_node, tet]
            for (edge, pair) in enumerate(POISSON_P2_EDGES)
                local_node in pair && (value += data.residual[4+edge, tet] / 2)
            end
            hat_residual[fan.tet_nodes[local_node, tet]] += value
        end
        @test norm(hat_residual[primal.free_dofs]) < 5e-12 * norm(primal.rhs)
        @test hat_residual ≈ primal.rhs-primal.K*primal.u
        @test dropdims(sum(data.weighted_residual; dims=2); dims=2) ≈ data.residual
        estimates = tetrahedral_patch_estimates(mesh, fan, data)
        @test all(isfinite, estimates.full.nodal)
        @test all(isfinite, estimates.reduced.nodal)
        @test length(estimates.full.nodal) == size(fan.points, 2)
        @test estimates.full.signed ≈ sum(estimates.full.nodal) + sum(data.correction)
        println("LOCAL_REFINEMENT_CONTROL ", (; domain,
            input_tetrahedra=nt, tetrahedra=length(fan.volumes), added_vertices=added,
            primal_dofs=length(primal.u), minimum_scaled_jacobian=stats.minimum_scaled_jacobian,
            effectivities=(estimates.full.signed, estimates.reduced.signed) ./
                (problem.reference_energy-primal.energy)))
    end

    generated = irregular_tetrahedral_mesh(:cube, 2; amplitude=0.0)
    @test_throws ArgumentError locally_refined_tetrahedral_mesh(generated; stride=0)
    unchanged = locally_refined_tetrahedral_mesh(generated; stride=length(generated.fan.volumes)+1)
    @test unchanged.fan.tet_nodes == generated.fan.tet_nodes
    @test unchanged.fan.points == generated.fan.points
    @test isempty(unchanged.fan.refined_nodes)
    @test unchanged.geometry_stats.minimum_child_orientation_ratio == 1.0
end
