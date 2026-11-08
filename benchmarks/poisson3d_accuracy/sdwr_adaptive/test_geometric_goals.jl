using Test
include("load.jl")
include("cut_goal_quadrature.jl")
include("geometric_goals.jl")

@testset "Unchanged aligned goals and exact cut-goal coupling" begin
    for name in (:cube_smooth,:lshape_smooth)
        problem = manufactured_problem(name)
        mesh = build_mesh(problem.domain,2)
        fan = build_fan(mesh)
        primal = solve_primal(mesh,fan,problem;quadrature_order=6)
        U = reconstruct_fan(mesh,fan,primal.u)
        goal = sdwr_goal(:localized_u,problem.domain)
        rules = sdwr_geometric_goal_rules(fan,goal)
        old_load = sdwr_goal_load(mesh,fan,U,goal)
        new_load = sdwr_geometric_goal_load(mesh,fan,U,goal,rules)
        @test new_load ≈ old_load rtol=2e-12 atol=1e-14
        @test sdwr_geometric_goal(mesh,fan,U,goal,rules).value ≈
            sdwr_reconstructed_goal(mesh,fan,U,goal).value rtol=2e-12
        z = zeros(length(U[1:fan.vertex_count]))
        free = primal.free_dofs
        z[free] = cholesky(Symmetric(primal.K[free,free]))\new_load[free]
        dual = merge(primal,(;u=z,rhs=new_load))
        original = sdwr_estimate(mesh,fan,primal,dual,problem,goal)
        clipped = sdwr_estimate(mesh,fan,primal,dual,problem,goal;goal_rules=rules)
        @test clipped.signed ≈ original.signed rtol=2e-12 atol=1e-14
        @test clipped.nodal_total ≈ original.nodal_total rtol=2e-12 atol=1e-14
        @test clipped.indicators ≈ original.indicators rtol=2e-12 atol=1e-14
    end

    mesh = build_mesh(:cube,2)
    fan = build_fan(mesh)
    box = ((0.13,0.81),(0.24,0.92),(0.17,0.73))
    weight(x) = all(box[d][1] <= x[d] <= box[d][2] for d in 1:3) ? 1.0 : 0.0
    goal = merge(sdwr_goal(:localized_u,:cube),(;box,weight,
        value=(x,u)->weight(x)*u,derivative_value=(x,u)->weight(x)))
    rules = sdwr_geometric_goal_rules(fan,goal)
    volume = prod(bounds[2]-bounds[1] for bounds in box)
    @test sum(sum(rule.weights) for rule in rules) ≈ volume rtol=2e-12
    values = [1+dot((0.2,-0.3,0.4),x) for x in eachcol(mesh.points)]
    U = reconstruct_fan(mesh,fan,values)
    exact = volume*(1+sum((0.2,-0.3,0.4)[d]*(box[d][1]+box[d][2])/2 for d in 1:3))
    @test sdwr_geometric_goal(mesh,fan,U,goal,rules).value ≈ exact rtol=2e-12
    direction = [sin(0.13i) for i in eachindex(values)]
    load = sdwr_geometric_goal_load(mesh,fan,U,goal,rules)
    step = 1e-4
    plus = reconstruct_fan(mesh,fan,values+step*direction)
    minus = reconstruct_fan(mesh,fan,values-step*direction)
    difference = (sdwr_geometric_goal(mesh,fan,plus,goal,rules).value-
        sdwr_geometric_goal(mesh,fan,minus,goal,rules).value)/(2step)
    @test dot(load,direction) ≈ difference rtol=2e-9 atol=1e-12

    problem = manufactured_problem(:cube_smooth)
    primal = solve_primal(mesh,fan,problem;quadrature_order=6)
    U = reconstruct_fan(mesh,fan,primal.u)
    free = primal.free_dofs
    z = zeros(length(primal.u))
    z[free] = cholesky(Symmetric(primal.K[free,free]))\load[free]
    dual = merge(primal,(;u=z,rhs=load))
    estimate = sdwr_estimate(mesh,fan,primal,dual,problem,goal;goal_rules=rules)
    zero_goal = merge(goal,(;derivative_value=(x,u)->0.0))
    base = sdwr_estimate(mesh,fan,primal,dual,problem,zero_goal)
    projection = sdwr_projection_fields(mesh,primal)
    correction = 0.0
    for tet in eachindex(fan.volumes)
        ids,cell = fan.tet_nodes[:,tet],fan.tet_cells[tet]
        p = projection.constants[cell] .+ vec(transpose(projection.gradients[:,cell])*fan.points[:,ids])
        # Independent first moments of the affine error, without PU product loops.
        correction += dot(p-U[ids],rules[tet].barycentric*rules[tet].weights)/2
    end
    @test estimate.signed ≈ base.signed+correction rtol=2e-12 atol=1e-14
    @test sum(estimate.cell_signed) ≈ estimate.signed rtol=2e-12
end
