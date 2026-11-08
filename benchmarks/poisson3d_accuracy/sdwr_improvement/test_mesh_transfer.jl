include("run_mesh_transfer.jl")
using Test

function transfer_cut_functional(fan,geometry,coefficients,rules)
    value = 0.0
    for tet in eachindex(fan.volumes)
        local_values = coefficients[geometry.dofs[:,tet]]
        rule = rules[tet]
        for q in eachindex(rule.weights)
            N = view(rule.barycentric,:,q)
            field = dot(local_values[1:4],N)+sum(4N[a]*N[b]*local_values[4+edge]
                for (edge,(a,b)) in enumerate(HIERARCHICAL_TET_EDGES))
            value += rule.weights[q]*field
        end
    end
    return value
end

@testset "Localized goal on irregular prism hierarchy" begin
    BLAS.set_num_threads(1)
    problem = manufactured_problem(:cube_smooth)
    prepared = sdwr_prepare_prism(problem,2;quadrature_order=6)
    mesh,fan = prepared.mesh,prepared.fan
    goal = sdwr_goal(:localized_u,:cube)
    evaluated = sdwr_evaluate_prism(prepared,problem,goal;quadrature_order=6)
    geometry = hierarchical_geometry(fan)
    residuals = transfer_residuals(prepared,evaluated,problem,goal,geometry)
    P = hierarchical_prolongation(mesh,fan,geometry)
    Z = P*evaluated.dual.u
    load = residuals.bz+geometry.K*Z
    constant = vcat(ones(geometry.vertex_count),zeros(length(geometry.edges)))
    @test dot(load,constant) ≈ 1/8 rtol=2e-12 atol=1e-14
    trial = sin.(0.37.*collect(1:length(load)))
    trial[geometry.boundary] .= 0.0
    @test dot(load,trial) ≈ transfer_cut_functional(fan,geometry,trial,evaluated.goal_rules) rtol=3e-12 atol=1e-14
    @test residuals.consistency ≈ dot(residuals.bu,Z) rtol=3e-12 atol=1e-14
    @test all(iszero,Z[geometry.boundary])
    free = geometry.free
    factor = cholesky(Symmetric(geometry.K[free,free]))
    du,dz = zeros(length(load)),zeros(length(load))
    du[free] = factor\residuals.bu[free]
    dz[free] = factor\residuals.bz[free]
    estimate = hierarchical_readout(residuals,du,dz)
    true_fine_difference = transfer_cut_functional(fan,geometry,du,evaluated.goal_rules)
    @test estimate ≈ true_fine_difference rtol=4e-11 atol=1e-14
    pcg = hierarchical_pcg_estimates(mesh,fan,geometry,residuals,prepared;checkpoints=(8,32,64))
    @test last(pcg).estimate ≈ estimate rtol=3e-10 atol=1e-14
end
