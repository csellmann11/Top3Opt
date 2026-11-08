include("hierarchical_run.jl")
using Test

function hierarchical_quadratic_coefficients(fan,geometry,q)
    values = [q(view(fan.points,:,node)) for node in axes(fan.points,2)]
    return vcat(values,[q((fan.points[:,a]+fan.points[:,b])/2)-(values[a]+values[b])/2
        for (a,b) in geometry.edges])
end

function hierarchical_goal_value(fan,geometry,base,delta,name;order=5)
    rule = tetra_rule(order)
    result = 0.0
    for tet in eachindex(fan.volumes)
        ids = fan.tet_nodes[:,tet]
        local_delta = delta[geometry.dofs[:,tet]]
        G = fan.gradients[tet]
        for q in eachindex(rule.weights)
            N = rule.barycentric[:,q]
            value = dot(base[ids]+local_delta[1:4],N)
            gradient = G*(base[ids]+local_delta[1:4])
            for (edge,(a,b)) in enumerate(HIERARCHICAL_TET_EDGES)
                value += 4N[a]*N[b]*local_delta[4+edge]
                gradient .+= 4local_delta[4+edge].*(N[a].*G[:,b]+N[b].*G[:,a])
            end
            integrand = name == :energy ? dot(gradient,gradient)/2 :
                name == :integral_u2 ? value^2 : value
            result += fan.volumes[tet]*rule.weights[q]*integrand
        end
    end
    return result
end

@testset "Hierarchical basis and generic quadratic-goal identity" begin
    BLAS.set_num_threads(1)
    problem = manufactured_problem(:cube_smooth)
    prepared = hierarchical_prepare(problem,2;order=6)
    fan,mesh = prepared.fan,prepared.mesh
    geometry = hierarchical_geometry(fan)
    constant = vcat(ones(geometry.vertex_count),zeros(length(geometry.edges)))
    @test norm(geometry.K*constant) < 1e-12
    @test all(>(0),geometry.diagonal)
    P = hierarchical_prolongation(mesh,fan,geometry)
    @test (P*prepared.primal.u)[1:geometry.vertex_count] ≈ prepared.U atol=1e-14
    @test norm((P*prepared.primal.u)[geometry.boundary]) == 0
    for q in (x->x[1]^2,x->x[1]*x[2],x->sum(abs2,x))
        coefficients = hierarchical_quadratic_coefficients(fan,geometry,q)
        for tet in (1,10,length(fan.volumes))
            N = [0.1,0.2,0.3,0.4]
            x = fan.points[:,fan.tet_nodes[:,tet]]*N
            local_c = coefficients[geometry.dofs[:,tet]]
            value = dot(local_c[1:4],N)+sum(4N[a]*N[b]*local_c[4+edge]
                for (edge,(a,b)) in enumerate(HIERARCHICAL_TET_EDGES))
            @test value ≈ q(x) atol=1e-14
        end
    end
    for name in (:integral_u,:integral_u2,:energy)
        evaluated = hierarchical_evaluate_goal(prepared,problem,name)
        residuals = hierarchical_residuals(mesh,fan,geometry,prepared.primal,evaluated.dual,
            problem,evaluated.goal;order=6)
        free = geometry.free
        factor = cholesky(Symmetric(geometry.K[free,free]))
        du,dz = zeros(length(geometry.diagonal)),zeros(length(geometry.diagonal))
        du[free],dz[free] = factor\residuals.bu[free],factor\residuals.bz[free]
        remainder = hierarchical_goal_remainder(fan,geometry,du,evaluated.goal)
        computed = hierarchical_readout(residuals,du,dz)+remainder
        difference = hierarchical_goal_value(fan,geometry,prepared.U,du,name)-evaluated.reconstructed
        @test computed ≈ difference rtol=3e-11 atol=1e-14
        if name == :energy
            alternate = residuals.primal_work+dot(residuals.bu,du)-dot(du,geometry.K*du)/2
            @test alternate ≈ difference rtol=3e-11 atol=1e-14
        end
        results = hierarchical_pcg_estimates(mesh,fan,geometry,residuals,prepared;
            checkpoints=(4,16,64),remainder=d->hierarchical_goal_remainder(fan,geometry,d,evaluated.goal),
            energy_goal=name==:energy)
        @test results[end].estimate ≈ computed rtol=1e-10 atol=1e-14
        @test results[end].primal_residual < 1e-10
    end
end
