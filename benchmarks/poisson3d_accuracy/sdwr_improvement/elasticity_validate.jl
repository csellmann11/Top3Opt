include("elasticity_control.jl")
using Test

function elasticity_bilinear_grad(A,B,lambda,mu)
    ea,eb = (A+transpose(A))/2,(B+transpose(B))/2
    return 2mu*sum(ea.*eb)+lambda*tr(A)*tr(B)
end

function elasticity_navier_difference(problem,x;step=1e-4)
    shift(i,sign) = ntuple(d->x[d]+(d==i ? sign*step : 0.0),3)
    function second(component,i,j)
        if i == j
            return (problem.exact(shift(i,1))[component]-2problem.exact(x)[component]+
                problem.exact(shift(i,-1))[component])/step^2
        end
        value = 0.0
        for a in (-1,1),b in (-1,1)
            point = ntuple(d->x[d]+(d==i ? a*step : 0.0)+(d==j ? b*step : 0.0),3)
            value += a*b*problem.exact(point)[component]
        end
        return value/(4step^2)
    end
    return ntuple(i->-problem.mu*sum(second(i,j,j) for j in 1:3)-
        (problem.lambda+problem.mu)*sum(second(j,i,j) for j in 1:3),3)
end

"""Independent direct quadrature of all three original affine DWR terms."""
function elasticity_sdwr_direct(mesh,fan,coarse,problem,uh,zh,U,Z,name;order=8)
    rule = tetra_rule(order)
    uvalues,zvalues = reshape(U,3,:),reshape(Z,3,:)
    result = 0.0
    for tet in eachindex(fan.volumes)
        cell = fan.tet_cells[tet]
        data = coarse.cells[cell]
        ids = fan.tet_nodes[:,tet]
        uc,zc = reshape(uh[elasticity_dofs(mesh.cells[cell])],3,8),
            reshape(zh[elasticity_dofs(mesh.cells[cell])],3,8)
        gpu,gpz = uc*transpose(data.G),zc*transpose(data.G)
        local_u,local_z = uvalues[:,ids],zvalues[:,ids]
        gu,gz = local_u*transpose(fan.gradients[tet]),local_z*transpose(fan.gradients[tet])
        xyz = fan.points[:,ids]*rule.barycentric
        for q in eachindex(rule.weights)
            x,N = view(xyz,:,q),view(rule.barycentric,:,q)
            u,z = local_u*N,local_z*N
            p = vec(sum(uc;dims=2))/8+gpu*(x-data.center)
            qfield = vec(sum(zc;dims=2))/8+gpz*(x-data.center)
            f = collect(problem.rhs(x))
            goal_term = name == :integral_u ? p[1]-u[1] :
                name == :integral_u2 ? 2u[1]*(p[1]-u[1]) :
                elasticity_bilinear_grad(gu,gpu-gu,problem.lambda,problem.mu)
            density = dot(f,z)-elasticity_bilinear_grad(gu,gz,problem.lambda,problem.mu)+
                (dot(f,qfield-z)-elasticity_bilinear_grad(gu,gpz-gz,problem.lambda,problem.mu))/2+
                (goal_term-elasticity_bilinear_grad(gpu-gu,gz,problem.lambda,problem.mu))/2
            result += fan.volumes[tet]*rule.weights[q]*density
        end
    end
    return result
end

function validate_elasticity_control(;output=joinpath(@__DIR__,"..","results","sdwr_elasticity_control"))
    BLAS.set_num_threads(1)
    mkpath(output)
    records = NamedTuple[]
    @testset "Coupled elasticity constitutive matrix and DWR controls" begin
        for m in (1,3),x in ((0.17,0.31,0.43),(0.41,0.22,0.67),(0.73,0.61,0.29))
            problem = elasticity_problem(m)
            @test collect(elasticity_navier_difference(problem,x)) ≈ collect(problem.rhs(x)) rtol=2e-6 atol=1e-7
        end
        G = [0.2 -0.3 0.1 0.7; -0.4 0.1 0.3 -0.2; 0.5 0.2 -0.1 0.4]
        u = sin.(collect(1:12))
        gradient = reshape(u,3,4)*transpose(G)
        @test dot(u,elasticity_matrix(G,1.0,1.0)*u) ≈
            elasticity_bilinear_grad(gradient,gradient,1.0,1.0) rtol=1e-13
        mesh = build_mesh(:cube,2)
        fan = build_fan(mesh)
        geometry = hierarchical_geometry(fan)
        coarse = elasticity_coarse_operator(mesh,1.0,1.0)
        enriched = elasticity_enriched_operator(fan,geometry,1.0,1.0)
        P = kron(hierarchical_prolongation(mesh,fan,geometry),sparse(Matrix{Float64}(I,3,3)))
        @test norm(enriched.K-transpose(enriched.K)) < 1e-11
        @test norm(coarse.K-transpose(coarse.K)) < 1e-12
        @test all(>(0),enriched.diagonal)
        problem = elasticity_problem(1)
        source = elasticity_loads(mesh,fan,geometry,coarse,problem;order=8)
        low = elasticity_loads(mesh,fan,geometry,coarse,problem;order=6)
        @test source.projected ≈ low.projected rtol=2e-9 atol=1e-11
        @test source.enriched ≈ low.enriched rtol=2e-9 atol=1e-11
        uh = elasticity_solve(coarse,source.projected)
        U = P*uh
        bu = source.enriched-enriched.K*U
        A = enriched.K[enriched.free,enriched.free]
        factor = cholesky(Symmetric(A))
        du = zeros(length(U)); du[enriched.free] = factor\bu[enriched.free]
        reference = sine_uniform_reference(;order=4)
        for name in (:integral_u,:integral_u2,:energy)
            goal = elasticity_goal(enriched,U,problem,name)
            zh = elasticity_solve(coarse,transpose(P)*goal.rhs)
            Z = P*zh
            bz = goal.rhs-enriched.K*Z
            baseline = elasticity_sdwr(mesh,coarse,enriched,source,uh,zh,U,Z,name,reference)
            direct = elasticity_sdwr_direct(mesh,fan,coarse,problem,uh,zh,U,Z,name)
            @test baseline.estimate ≈ direct rtol=3e-11 atol=2e-12
            results = elasticity_estimates(enriched,P,coarse,bu,bz,U,Z,source.enriched,goal,name;
                checkpoints=(4,8),direct=true)
            fine_goal = elasticity_goal(enriched,U+du,problem,name)
            correction = last(results).estimate
            @test correction ≈ fine_goal.value-goal.value rtol=1e-11 atol=1e-12
            @test norm((coarse.K*uh-source.projected)[coarse.free])/norm(source.projected[coarse.free]) < 1e-12
            push!(records,(;goal=String(name),baseline=baseline.estimate,direct_quadrature=direct,
                baseline_relative_difference=abs(baseline.estimate-direct)/abs(direct),
                enriched_estimate=correction,enriched_goal_difference=fine_goal.value-goal.value))
        end
    end
    write_csv(joinpath(output,"validation.csv"),records)
    return records
end

if abspath(PROGRAM_FILE) == @__FILE__
    validate_elasticity_control()
end
