using Test, LinearAlgebra, SparseArrays, StaticArrays, OrderedCollections
using Ju3VEM
include("../src/mat_states.jl")
include("../src/diamond_flux.jl")

@testset "3D diamond face flux" begin
    points = [SA[0.,-1.,-1.], SA[0.,1.,-1.], SA[0.,1.,1.], SA[0.,-1.,1.]]
    xL = SA[-1.,0.,0.]; xR = SA[0.5,0.3,-0.2]
    T,c = diamond_face_coefficients(points,xL,xR,2.,8.)
    @test T ≈ 4/(1/2 + 0.5/8)
    @test sum(c) ≈ 0 atol=1e-14
    # Continuous trace and normal flux, discontinuous normal gradient.
    q = 1.7; tangent = SA[0.,0.6,-0.9]; intercept = 0.4
    chiL = intercept + q/2*xL[1] + dot(tangent,xL)
    chiR = intercept + q/8*xR[1] + dot(tangent,xR)
    chiF = [intercept + dot(tangent,p) for p in points]
    @test T*(chiR-chiL) + dot(c,chiF) ≈ 4q
    Tr,cr = diamond_face_coefficients(reverse(points),xL,xR,2.,8.)
    @test Tr*(chiR-chiL) + dot(cr,reverse(chiF)) ≈ 4q
    Ts,cs = diamond_face_coefficients(points,xR,xL,8.,2.)
    @test Ts*(chiL-chiR) + dot(cs,chiF) ≈ -4q
    _,ct = diamond_face_coefficients(points,xL,xR,2.,8.; diamond=false)
    @test all(iszero,ct)
    _,co = diamond_face_coefficients(points,xL,SA[0.5,0.,0.],2.,8.)
    @test all(iszero,co)
end

@testset "Assembled uniform and adaptive mesh" begin
    mesh = create_rectangular_mesh(3,3,3,1.,1.,1.,StandardEl{1})
    for adaptive in (false,true)
        if adaptive
            el = collect(RootIterator{4}(mesh.topo))[14]
            Ju3VEM.VEMGeo._refine!(el,mesh.topo)
            mesh = Mesh(mesh.topo,StandardEl{1}())
        end
        cv = CellValues{3}(mesh)
        states = DesignVarInfo{3}(cv,0.3)
        pars = (β0=1.0,)
        R = compute_flux_operator_mat(cv,states,pars)
        P = compute_flux_operator_mat(cv,states,pars; scheme=:tpfa)
        N = length(states.χ_vec)
        @test norm(R*ones(N),Inf) < 1e-10
        @test norm(states.area_vec' * R,Inf) < 1e-11
        @test norm(P*ones(N),Inf) < 1e-10
        @test norm(states.area_vec' * P,Inf) < 1e-11
        @test all(isfinite,nonzeros(R))
        owners = regularization_faces(cv,states)
        weights = build_diamond_node_weights(cv,states,owners)
        affine(x) = 0.7 + dot(SA[0.2,-0.4,0.8],x)
        chi = affine.(states.x_vec)
        for nid in eachindex(weights)
            isempty(weights[nid]) && continue
            @test sum(w for (_,w) in weights[nid]) ≈ 1
            x = cv.mesh.topo.nodes[nid]
            if all(t -> 1e-8 < t < 1-1e-8,x)
                @test sum(w*chi[sid] for (sid,w) in weights[nid]) ≈ affine(x) atol=1e-12
            end
        end
        if adaptive
            @test norm(R-P,Inf) > 1e-4
            @test length(unique(states.h_vec)) > 1
        else
            @test norm(R-P,Inf) < 1e-10
        end
        @test norm(compute_flux_operator_mat(cv,states,(β0=0.,)),Inf) == 0
    end
end
