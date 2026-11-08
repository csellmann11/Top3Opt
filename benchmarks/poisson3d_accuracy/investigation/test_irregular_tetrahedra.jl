include(joinpath(@__DIR__, "..", "load.jl"))
include("irregular_tetrahedra.jl")
using Test

@testset "Irregular tetrahedral FEM control" begin
    for domain in (:cube, :lshape)
        control = irregular_tetrahedral_mesh(domain, 2; amplitude=0.08)
        mesh, fan, stats = control.mesh, control.fan, control.geometry_stats
        @test stats.minimum_orientation_ratio > 0
        @test stats.minimum_scaled_jacobian > 0
        @test stats.maximum_displacement_over_h > 0
        @test sum(mesh.volumes) ≈ sum(fan.volumes)
        @test sum(fan.volumes) ≈ (domain == :cube ? 1.0 : 3.0)
        affine(x) = 0.7 + x[1] - 0.3x[2] + 0.2x[3]
        affine_problem = (; rhs=x -> 0.0)
        primal = solve_tetrahedral_p1(fan, affine_problem; quadrature_order=2,
            dirichlet=affine)
        exact_values = [affine(view(fan.points, :, node)) for node in axes(fan.points, 2)]
        @test maximum(abs, primal.u-exact_values) < 2e-12
        @test primal.algebraic_residual < 2e-12
        for tet in eachindex(fan.volumes)
            G = fan.gradients[tet]
            ids = fan.tet_nodes[:, tet]
            @test norm(G * ones(4)) < 1e-12
            @test G * transpose(fan.points[:, ids]) ≈ Matrix{Float64}(I, 3, 3)
        end
    end
    problem = manufactured_problem(:cube_smooth)
    control = irregular_tetrahedral_mesh(:cube, 2; amplitude=0.08)
    mesh, fan = control.mesh, control.fan
    primal = solve_tetrahedral_p1(fan, problem)
    data = tetrahedral_p1_estimator_data(mesh, fan, primal, problem)
    truth = tetrahedral_p1_truth(mesh, fan, primal, problem; quadrature_order=7)
    @test primal.algebraic_residual < 1e-12
    @test abs(data.common_correction) < 1e-12 * primal.energy
    @test sum(truth.discrete_energy) ≈ primal.energy
    @test data.common_correction + sum(truth.reconstruction_error) ≈
        problem.reference_energy-primal.energy
    estimates = tetrahedral_patch_estimates(mesh, fan, data)
    @test all(isfinite, estimates.full.nodal)
    @test all(isfinite, estimates.reduced.nodal)
    @test estimates.full.signed ≈ sum(estimates.full.nodal) + sum(data.correction)
    double_estimates = tetrahedral_patch_estimates(mesh, fan, data;
        factor=1, include_corrections=false)
    @test double_estimates.full.signed ≈
        2 * (estimates.full.signed-sum(data.correction))
    @test double_estimates.reduced.signed ≈
        2 * (estimates.reduced.signed-sum(data.correction))
    println("IRREGULAR_CONTROL ", (; primal_dofs=length(primal.u),
        minimum_orientation_ratio=control.geometry_stats.minimum_orientation_ratio,
        relative_goal_error=(problem.reference_energy-primal.energy)/problem.reference_energy,
        effectivities=(estimates.full.signed, estimates.reduced.signed) ./
            (problem.reference_energy-primal.energy)))
end
