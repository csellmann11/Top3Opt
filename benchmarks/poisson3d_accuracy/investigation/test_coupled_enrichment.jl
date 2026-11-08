using Test

isdefined(@__MODULE__, :poisson_estimator_data) ||
    include(joinpath(@__DIR__, "..", "load.jl"))
isdefined(@__MODULE__, :coupled_p2_estimates) ||
    include(joinpath(@__DIR__, "coupled_enrichment.jl"))
isdefined(@__MODULE__, :oversampled_patch_estimate) ||
    include(joinpath(@__DIR__, "oversampled_patches.jl"))
isdefined(@__MODULE__, :irregular_tetrahedral_mesh) ||
    include(joinpath(@__DIR__, "irregular_tetrahedra.jl"))

"""Check all quadratic terms using an independent constant body load and exact trace."""
function test_coupled_quadratic_lifting(fan, system)
    polynomial(x) = 0.4 + 0.7x[1] - 0.3x[2] + 0.2x[3] +
        x[1]^2 + 2x[2]^2 + 3x[3]^2 +
        0.5x[1]*x[2] - 0.4x[1]*x[3] + 0.6x[2]*x[3]
    values = [polynomial(view(system.coordinates, :, dof))
        for dof in 1:system.total_dofs]
    load = zeros(system.total_dofs)
    # -Delta(polynomial)=-12. Exact P2 shape integrals are -V/20 at a
    # vertex and V/5 at an edge midpoint, independently of tetra geometry.
    for tet in eachindex(fan.volumes), shape in 1:10
        load[system.tet_dofs[shape, tet]] +=
            fan.volumes[tet] * (shape <= 4 ? 0.6 : -2.4)
    end
    boundary, free = system.boundary_dofs, system.free_dofs
    recovered = zeros(system.total_dofs)
    recovered[boundary] .= values[boundary]
    recovered[free] .= cholesky(Symmetric(system.free_operator)) \
        (load[free] - system.K[free, boundary] * values[boundary])
    @test isapprox(recovered, values; rtol=2e-12, atol=2e-12)
    return maximum(abs, recovered - values)
end

function test_coupled_energy_and_localization(mesh, fan, data, existing_full)
    result = coupled_p2_estimates(mesh, fan, data; sweep_counts=(1, 2, 4, 8))
    system, direct = result.assembly, result.direct
    @test direct.relative_residual < 1e-11
    @test isapprox(direct.rhs_work, direct.energy_norm2; rtol=1e-11, atol=1e-16)
    @test isapprox(direct.correction_dwr, direct.correction_variational;
        rtol=1e-11, atol=1e-16)

    variational_values = [0.0; [state.correction_variational for state in result.sweeps]]
    tolerance = 100eps(Float64) * max(abs(direct.correction_variational), eps(Float64))
    @test all(diff(variational_values) .>= -tolerance)
    @test all(state.correction_variational <= direct.correction_variational + tolerance
        for state in result.sweeps)
    for state in [direct; result.sweeps]
        @test all(iszero, state.delta[system.boundary_dofs])
        @test isapprox(norm(state.global_residual[system.free_dofs]), state.residual_norm;
            rtol=1e-10, atol=1e-15)
        @test isapprox(state.estimate.signed,
            state.common_correction + dot(system.rhs, state.delta) / 2;
            rtol=1e-12, atol=1e-16)
        @test isapprox(sum(state.estimate.indicators), state.estimate.magnitude;
            rtol=1e-12, atol=1e-16)
    end

    # This compares two independent assembly paths. It validates the support
    # criterion and the unchanged original-star readout, including node-to-cell
    # localization; matching only their global sums would be insufficient.
    zero_buffer = oversampled_patch_estimate(mesh, fan, data, system; layers=0)
    @test isapprox(zero_buffer.nodal, existing_full.nodal; rtol=1e-11, atol=1e-16)
    @test isapprox(zero_buffer.indicators, existing_full.indicators; rtol=1e-11, atol=1e-16)
    @test isapprox(zero_buffer.signed, existing_full.signed; rtol=1e-11, atol=1e-16)
    quadratic_error = test_coupled_quadratic_lifting(fan, system)
    return (; total_dofs=system.total_dofs, free_dofs=system.free_dof_count,
        nnz=system.operator_nnz, relative_residual=direct.relative_residual,
        quadratic_error,
        zero_buffer_node_error=maximum(abs, zero_buffer.nodal-existing_full.nodal),
        zero_buffer_indicator_error=maximum(abs, zero_buffer.indicators-existing_full.indicators))
end

@testset "Coupled P2 enrichment and zero-buffer equivalence" begin
    BLAS.set_num_threads(1)
    problem = manufactured_problem(:cube_smooth)
    @testset "Cartesian VEM primal" begin
        mesh = build_mesh(:cube, 2)
        fan = build_fan(mesh)
        primal = solve_primal(mesh, fan, problem; quadrature_order=6)
        data = poisson_estimator_data(mesh, fan, primal, problem; quadrature_order=6)
        full = estimate_errors(mesh, fan, primal, problem; quadrature_order=6).full
        stats = test_coupled_energy_and_localization(mesh, fan, data, full)
        println("COUPLED_TEST ", (; geometry=:cartesian_vem, stats...))
    end
    @testset "Perturbed tetrahedral P1 primal" begin
        control = irregular_tetrahedral_mesh(:cube, 2; amplitude=0.08)
        mesh, fan = control.mesh, control.fan
        primal = solve_tetrahedral_p1(fan, problem; quadrature_order=6)
        data = tetrahedral_p1_estimator_data(mesh, fan, primal, problem; quadrature_order=6)
        full = tetrahedral_patch_estimates(mesh, fan, data).full
        stats = test_coupled_energy_and_localization(mesh, fan, data, full)
        println("COUPLED_TEST ", (; geometry=:perturbed_tetrahedral, stats...))
    end
end
