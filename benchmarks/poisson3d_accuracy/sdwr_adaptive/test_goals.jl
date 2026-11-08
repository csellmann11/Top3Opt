using Test, LinearAlgebra
isdefined(@__MODULE__,:build_mesh) || include("../load.jl")
isdefined(@__MODULE__,:sdwr_goal) || include("goals.jl")

@testset "Continuous goals, independent references, and adjoint loads" begin
    @test sdwr_goal_reference(manufactured_problem(:cube_smooth),
        sdwr_goal(:localized_u,:cube)).value ≈ 1/1728 rtol=1e-13
    @test sdwr_goal_reference(manufactured_problem(:lshape_smooth),
        sdwr_goal(:localized_u,:lshape)).value ≈ -27/16384 rtol=1e-13
    for name in (:cube_smooth,:lshape_smooth,:lshape_singular),
            goal_name in (:integral_u,:integral_u2,:localized_u)
        problem = manufactured_problem(name)
        goal = sdwr_goal(goal_name,problem.domain)
        reference = sdwr_goal_reference(problem,goal)
        @test reference.uncertainty < 1e-11*abs(reference.value)
        mesh = build_mesh(problem.domain,2)
        fan = build_fan(mesh)
        values = [sin(0.73i) for i in axes(mesh.points,2)]
        direction = [cos(0.41i) for i in eachindex(values)]
        U = reconstruct_fan(mesh,fan,values)
        derivative = sdwr_goal_load(mesh,fan,U,goal)
        step = 1e-4
        plus = sdwr_reconstructed_goal(mesh,fan,reconstruct_fan(mesh,fan,values+step*direction),goal).value
        minus = sdwr_reconstructed_goal(mesh,fan,reconstruct_fan(mesh,fan,values-step*direction),goal).value
        @test dot(derivative,direction) ≈ (plus-minus)/(2step) rtol=2e-10 atol=2e-12
        # Separate quadrature of both J(U) and its derivative against T(direction).
        rule = tetra_rule(6)
        testfield = reconstruct_fan(mesh,fan,direction)
        integral,variation = 0.0,0.0
        for tet in eachindex(fan.volumes)
            ids = fan.tet_nodes[:,tet]
            quadrature = tetra_quadrature(fan.points[:,ids],rule)
            for q in eachindex(rule.weights)
                x = view(quadrature.points,:,q)
                shape = view(rule.barycentric,:,q)
                u,v = dot(U[ids],shape),dot(testfield[ids],shape)
                integral += quadrature.weights[q]*goal.value(x,u)
                variation += quadrature.weights[q]*goal.derivative_value(x,u)*v
            end
        end
        @test sdwr_reconstructed_goal(mesh,fan,U,goal).value ≈ integral rtol=2e-12 atol=1e-14
        @test dot(derivative,direction) ≈ variation rtol=2e-12 atol=1e-14
    end
    @test sdwr_mark([0.1,0.5,0.2,0.2];theta=0.6) == [2,3]
    @test sdwr_mark([0.0,0.0];theta=0.5) == [1]
end
