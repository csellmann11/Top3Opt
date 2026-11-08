# julia --project=. --startup-file=no Tests/pu_body_loads.jl
isdefined(@__MODULE__, :run_optimization) || include("regularization_benchmark.jl")

function body_fixture(n=2;hanging=false)
    mesh = create_rectangular_mesh(n,n,n,1.,1.,1.,StandardEl{1})
    if hanging
        Ju3VEM.VEMGeo._refine!(first(RootIterator{4}(mesh.topo)),mesh.topo)
        mesh = Mesh(mesh.topo,StandardEl{1}())
    end
    cv = CellValues{3}(mesh)
    states = DesignVarInfo{3}(cv,0.6)
    lambda,mu = E_ν_to_lame(1.,0.3)
    pars = SimPars(Helmholtz{3,3}(Ψlin_totopt,(lambda,mu,1.)),lambda,mu,1e-3,15.,1.,0.6)
    return cv,states,pars
end

function body_values(cv,field)
    u = zeros(3length(cv.dh.dof_mapping))
    for (node,dofs) in cv.dh.dof_mapping
        u[dofs] = field(cv.mesh.nodes[node].coords)
    end
    return u
end

@testset "Projected 3D body loads preserve force and affine work" begin
    for hanging in (false,true)
        cv,states,pars = body_fixture(;hanging)
        base = SA[0.7,-0.3,0.5]
        A = SA[0.2 -0.1 0.;0.0 0.0 0.4;-0.3 0.2 0.0]
        f(x) = base+A*x
        K,rhs,ed = assembly(cv,states,f,pars)
        resultant = base+A*SA[0.5,0.5,0.5]
        @test SVector{3}(sum(rhs[c:3:end]) for c in 1:3) ≈ resultant atol=3e-13
        b = SA[0.4,-0.2,0.1]
        G = SA[0.3 -0.1 0.2;0.2 0.4 -0.3;-0.1 0.2 0.1]
        displacement = body_values(cv,x -> b+G*x)
        moments = SMatrix{3,3}(ntuple(9) do k
            i,j = mod1(k,3),div(k-1,3)+1
            base[i]/2 + sum(A[i,l]*(l == j ? 1/3 : 1/4) for l in 1:3)
        end)
        expected_work = dot(resultant,b)+sum(G.*moments)
        @test dot(rhs,displacement) ≈ expected_work atol=3e-13
        ch,loads = ConstraintHandler{3}(cv.mesh),PU3DBoundaryData()
        cache = build_pu_dwr_cache(cv,states,ed,ch,pars;boundary_data=loads,rhs_fun=f)
        pu3d_reconstruct!(cache,displacement,ed)
        @test sum(pu3d_discrete_load_work(cache.loads,e,displacement)
            for e in eachindex(cache.geometry.element_ids)) ≈ expected_work atol=3e-13
        @test sum(pu3d_continuous_load_work(cache.loads,i,t,cache.values)
            for (i,t) in enumerate(cache.geometry.tets)) ≈ expected_work atol=3e-13
        _,zero_rhs,_ = assembly(cv,states,ZeroBodyForce(),pars)
        @test all(iszero,zero_rhs)
        add_face_set!(cv.mesh,"clamp",x -> x[1] ≈ 0.)
        pu3d_add_dirichlet!(loads,ch,cv.dh,cv.facedata_col,"clamp",x -> SA[0.,0.,0.])
        apply!(K,rhs,ch)
        u = cholesky(Symmetric(K))\rhs
        @test norm(K*u-rhs) < 1e-10
        cache = build_pu_dwr_cache(cv,states,ed,ch,pars;boundary_data=loads,rhs_fun=f)
        estimate = estimate_pu_dwr!(cache,u,states,ed,pars)
        @test isfinite(estimate.signed)
        @test sum(estimate.indicators) ≈ estimate.magnitude rtol=5e-13
    end
end

@testset "Quadratic manufactured elasticity state converges" begin
    energy_errors = Float64[]
    estimates = Float64[]
    for n in (2,3,4)
        cv,states,pars = body_fixture(n)
        ch,loads = ConstraintHandler{3}(cv.mesh),PU3DBoundaryData()
        scale = states.χ_vec[1]^3
        add_face_set!(cv.mesh,"clamp",x -> x[1] ≈ 0.)
        pu3d_add_dirichlet!(loads,ch,cv.dh,cv.facedata_col,"clamp",x -> SA[0.,0.,0.])
        for (axis,side) in ((1,1.),(2,0.),(2,1.),(3,0.),(3,1.))
            name = "natural_$(axis)_$(side)"
            add_face_set!(cv.mesh,name,x -> x[axis] ≈ side)
            normal = SVector{3}(ntuple(k -> k == axis ? 2side-1 : 0.,3))
            traction(x) = scale*(pars.λ*2x[1]*SA[1. 0. 0.;0. 1. 0.;0. 0. 1.]+
                4pars.μ*x[1]*SA[1. 0. 0.;0. 0. 0.;0. 0. 0.])*normal
            pu3d_add_neumann!(loads,ch,cv.dh,cv.facedata_col,name,traction)
        end
        f(x) = SA[-2scale*(pars.λ+2pars.μ),0.,0.]
        K,rhs,ed = assembly(cv,states,f,pars)
        apply!(K,rhs,ch)
        u = cholesky(Symmetric(K))\rhs
        exact_energy = 2scale*(pars.λ+2pars.μ)/3
        discrete_energy = dot(u,rhs)/2
        cache = build_pu_dwr_cache(cv,states,ed,ch,pars;boundary_data=loads,rhs_fun=f)
        estimate = estimate_pu_dwr!(cache,u,states,ed,pars)
        push!(energy_errors,abs(exact_energy-discrete_energy))
        push!(estimates,estimate.signed)
        @test isfinite(estimate.signed)
        @test 0.9 < estimate.signed/energy_errors[end] < 1.1
        @test norm(K*u-rhs) < 1e-10
        @test sum(estimate.indicators) ≈ estimate.magnitude rtol=5e-13
    end
    @test all(diff(energy_errors).<0)
    @test energy_errors[end] < 0.4energy_errors[1]
    println("PU_BODY_MANUFACTURED energy_errors=",energy_errors," signed_estimates=",estimates)
end
