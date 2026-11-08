using Test, LinearAlgebra
isdefined(@__MODULE__, :build_mesh) || include(joinpath(@__DIR__, "..", "load.jl"))
isdefined(@__MODULE__, :sdwr_estimate) || include("estimators.jl")

sdwr_test_linear_goal(g) = (; value=(x,u)->g(x)*u,
    derivative_value=(x,u)->g(x), derivative_gradient=(x,gradient)->(0.0,0.0,0.0))
sdwr_test_quadratic_goal() = (; value=(x,u)->u*u,
    derivative_value=(x,u)->2u, derivative_gradient=(x,gradient)->(0.0,0.0,0.0))

"""Direct unlocalized volume integral; independent of all PU accumulation."""
function sdwr_test_direct_integral(mesh, fan, primal, dual, problem, goal; order=7)
    U, Z = reconstruct_fan(mesh,fan,primal.u), reconstruct_fan(mesh,fan,dual.u)
    result = zeros(length(mesh.cells))
    rule = tetra_rule(order)
    for tet in eachindex(fan.volumes)
        ids, cell = fan.tet_nodes[:,tet], fan.tet_cells[tet]
        gu, gz = fan.gradients[tet]*U[ids], fan.gradients[tet]*Z[ids]
        cell_u, cell_z = primal.u[mesh.cells[cell]], dual.u[mesh.cells[cell]]
        gpu, gpz = primal.cells[cell].G*cell_u, dual.cells[cell].G*cell_z
        physical = tetra_quadrature(fan.points[:,ids],rule)
        for q in eachindex(physical.weights)
            x, N = physical.points[:,q], rule.barycentric[:,q]
            u, z = dot(U[ids],N), dot(Z[ids],N)
            pu = sum(cell_u)/length(cell_u)+dot(gpu,x-primal.cells[cell].center)
            pz = sum(cell_z)/length(cell_z)+dot(gpz,x-dual.cells[cell].center)
            residual = problem.rhs(x)*(z+(pz-z)/2)-dot(gu,gz+(gpz-gz)/2)
            dual_residual = goal.derivative_value(x,u)*(pu-u)+
                dot(goal.derivative_gradient(x,gu),gpu-gu)-dot(gpu-gu,gz)
            result[cell] += physical.weights[q]*(residual+dual_residual/2)
        end
    end
    return result
end

function sdwr_test_goal_value(fan, field, goal; order=10)
    rule = tetra_rule(order)
    result = 0.0
    for tet in eachindex(fan.volumes)
        ids = fan.tet_nodes[:,tet]
        physical = tetra_quadrature(fan.points[:,ids],rule)
        for q in eachindex(physical.weights)
            u = dot(field[ids],rule.barycentric[:,q])
            result += physical.weights[q]*goal.value(physical.points[:,q],u)
        end
    end
    return result
end

@testset "sDWR distinct adjoints and genuine PU localization" begin
    mesh = build_mesh(:cube,2)
    fan = build_fan(mesh)
    problem = manufactured_problem(:cube_smooth)
    primal = solve_primal(mesh,fan,problem;quadrature_order=6)
    dual = solve_primal(mesh,fan,(;rhs=x->1+x[1]+2x[2]);quadrature_order=6)
    data = poisson_estimator_data(mesh,fan,primal,problem;quadrature_order=6)
    Z = reconstruct_fan(mesh,fan,dual.u)
    for goal in (sdwr_test_linear_goal(x->1.0), sdwr_test_quadratic_goal(),
                 sdwr_test_linear_goal(x->1+x[1]^2+3x[2]*x[3]))
        estimate = sdwr_estimate(mesh,fan,primal,dual,problem,goal;quadrature_order=6)
        direct = sdwr_test_direct_integral(mesh,fan,primal,dual,problem,goal)
        @test estimate.cell_signed ≈ direct rtol=2e-12 atol=1e-14
        @test estimate.signed ≈ sum(direct) rtol=2e-12 atol=1e-14
        @test sum(estimate.distributed_signed_separated) ≈ estimate.signed atol=1e-14
        @test sum(estimate.distributed_signed_combined) ≈ estimate.signed atol=1e-14
        @test sum(estimate.indicators_combined) ≈ sum(abs,estimate.nodal_total) atol=1e-14
        @test sum(estimate.indicators_separated) ≈
            sum(abs,estimate.cell_consistency)+sum(abs,estimate.nodal) atol=1e-14
        @test estimate.indicators == estimate.indicators_separated
        @test estimate.statistics.patch_count == 0
        # Independent P2 coefficient representation of N_i*Z, evaluated
        # against the older unweighted P2 residual cache.
        expected = zeros(length(Z))
        for tet in eachindex(fan.volumes)
            ids = fan.tet_nodes[:,tet]
            for hat in 1:4
                coefficients = zeros(10)
                coefficients[hat] = Z[ids[hat]]
                for (edge,(a,b)) in enumerate(POISSON_P2_EDGES)
                    if a == hat || b == hat
                        coefficients[4+edge] = (Z[ids[a]]+Z[ids[b]])/4
                    end
                end
                expected[ids[hat]] += dot(data.residual[:,tet],coefficients)
            end
        end
        @test estimate.nodal_consistency ≈ expected rtol=3e-12 atol=2e-15
    end
    combined = sdwr_estimate(mesh,fan,primal,dual,problem,sdwr_test_quadratic_goal();
        localization=:combined)
    @test combined.indicators == combined.indicators_combined
    @test_throws ArgumentError sdwr_estimate(mesh,fan,primal,dual,problem,
        sdwr_test_quadratic_goal();localization=:unknown)
    @test_throws ArgumentError sdwr_distribute_nodes(mesh,fan,ones(size(fan.points,2));
        cell_weights=zeros(length(mesh.cells)))
end

@testset "sDWR energy reduction and reciprocity" begin
    energy = (;derivative_value=(x,u)->0.0,derivative_gradient=(x,gradient)->gradient)
    for n in (2,3)
        mesh = build_mesh(:cube,n)
        fan = build_fan(mesh)
        problem = manufactured_problem(:cube_smooth)
        primal = solve_primal(mesh,fan,problem;quadrature_order=6)
        data = poisson_estimator_data(mesh,fan,primal,problem;quadrature_order=6)
        old = poisson_projected_estimate(mesh,fan,primal,data)
        new = sdwr_estimate(mesh,fan,primal,primal,problem,energy;quadrature_order=6)
        @test new.signed ≈ old.signed-sum(data.conversion) rtol=3e-12 atol=2e-15
        @test maximum(abs,new.nodal_dual) < 1e-17
        @test new.nodal_primal ≈ old.nodal-data.conversion rtol=3e-12 atol=2e-15
        g = x->1+x[1]^2
        other = (;rhs=g)
        dual = solve_primal(mesh,fan,other;quadrature_order=6)
        forward = sdwr_estimate(mesh,fan,primal,dual,problem,sdwr_test_linear_goal(g))
        reverse = sdwr_estimate(mesh,fan,dual,primal,other,sdwr_test_linear_goal(problem.rhs))
        U,Z = reconstruct_fan(mesh,fan,primal.u),reconstruct_fan(mesh,fan,dual.u)
        G_U = sdwr_test_goal_value(fan,U,sdwr_test_linear_goal(g))
        F_Z = sdwr_test_goal_value(fan,Z,sdwr_test_linear_goal(problem.rhs))
        @test forward.signed+G_U ≈ reverse.signed+F_Z rtol=3e-12 atol=2e-15
        @test dot(primal.rhs,dual.u) ≈ dot(dual.rhs,primal.u) rtol=3e-13
    end
end

@testset "Exact endpoint enrichments: linear and quadratic goal identity" begin
    mesh = build_mesh(:cube,2)
    fan = build_fan(mesh)
    eigenvalue = 3pi^2
    exact = x->sinpi(x[1])*sinpi(x[2])*sinpi(x[3])
    gradient = x->(pi*cospi(x[1])*sinpi(x[2])*sinpi(x[3]),
                   pi*sinpi(x[1])*cospi(x[2])*sinpi(x[3]),
                   pi*sinpi(x[1])*sinpi(x[2])*cospi(x[3]))
    problem = (;rhs=x->eigenvalue*exact(x))
    U = [0.6exact(fan.points[:,node]) for node in axes(fan.points,2)]
    Z = [0.4exact(fan.points[:,node])/eigenvalue for node in axes(fan.points,2)]
    for (goal,dual_scale,exact_goal) in (
        (sdwr_test_linear_goal(x->eigenvalue*exact(x)),1.0,eigenvalue/8),
        (sdwr_test_quadratic_goal(),2/eigenvalue,1/8))
        enrichment = (tet,cell,x)->(;u=exact(x),z=dual_scale*exact(x),
            gradient_u=gradient(x),gradient_z=dual_scale.*gradient(x))
        estimate = sdwr_integrate_dwr(mesh,fan,U,Z,problem,goal,enrichment;
            quadrature_order=10)
        goal_U = sdwr_test_goal_value(fan,U,goal;order=10)
        @test estimate.signed ≈ exact_goal-goal_U rtol=2e-11 atol=2e-12
        @test sum(estimate.cell_signed) ≈ estimate.signed atol=2e-12
    end
    # For this quadratic goal, using the frozen-state adjoint instead of
    # the exact endpoint adjoint misses half the quadratic error term.
    goal = sdwr_test_quadratic_goal()
    wrong = (tet,cell,x)->(;u=exact(x),z=0.0,gradient_u=gradient(x),
        gradient_z=(0.0,0.0,0.0))
    erroneous = sdwr_integrate_dwr(mesh,fan,zeros(length(U)),zeros(length(Z)),
        problem,goal,wrong;quadrature_order=8)
    @test abs(erroneous.signed) < 1e-14
    @test abs(erroneous.signed-1/8) > 0.12
end
