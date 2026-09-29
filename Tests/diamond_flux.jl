using Test, LinearAlgebra, SparseArrays, StaticArrays, OrderedCollections
using Ju3VEM, Ju3VEM.FixedSizeArrays
include("../src/mat_states.jl")
include("../src/diamond_flux.jl")
include("../src/density_timestepping.jl")
include("../src/utils/mesh_processing_utils.jl")

@testset "Nearly coplanar reconstruction and fixed substeps" begin
    # An exactly planar stencil shifted off the target vertex, perturbed only
    # by centroid integration roundoff. Default pinv fits that noise with 1e11 weights.
    A = hcat(ones(4), [-1.,1.,-1.,1.], [-1.,-1.,1.,1.],
        [0.2+1e-12,0.2-1e-12,0.2-1e-12,0.2+1e-12])
    w = diamond_reconstruction_weights(A)
    @test all(isfinite,w)
    @test sum(w) ≈ 1
    @test norm(w,1) < 2
    @test dot(w, 0.7 .+ 0.2A[:,2] .- 0.4A[:,3]) ≈ 0.7 atol=1e-12
    @test density_substeps(spdiagm(0=>[-75.]),15.,1.;beta_in_operator=true) == (8,75.)
    @test density_substeps(spzeros(2,2),15.,0.;beta_in_operator=true) == (1,0.)
    @test density_substeps(spdiagm(0=>[-1e13]),15.,1.;beta_in_operator=true) == (8,1e13)
    @test_throws ErrorException density_substeps(spdiagm(0=>[NaN]),15.,1.;beta_in_operator=true)
end

@testset "Direct coarse/fine interfaces up to 32:1" begin
    mesh = create_rectangular_mesh(2,1,1,2.,1.,1.,StandardEl{1})
    target = SA[1.,0.,0.]
    for level in 1:5
        cv = CellValues{3}(mesh)
        states = DesignVarInfo{3}(cv,0.3)
        candidates = findall(x->x[1]<1.,states.x_vec)
        sid = candidates[argmin([norm(states.x_vec[i]-target) for i in candidates])]
        elid = get_el_id(states,sid)
        el = only(el for el in RootIterator{4}(mesh.topo) if el.id == elid)
        Ju3VEM.VEMGeo._refine!(el,mesh.topo)
        mesh = Mesh(mesh.topo,StandardEl{1}())
        cv = CellValues{3}(mesh)
        states = DesignVarInfo{3}(cv,0.3)
        R = compute_flux_operator_mat(cv,states,(β0=1.,))
        row_bound = maximum(vec(sum(abs,R;dims=2)))
        @test maximum(states.h_vec)/minimum(states.h_vec) ≈ 2.0^level
        @test all(isfinite,nonzeros(R))
        @test norm(R*ones(length(states.χ_vec)),Inf) < 1e-9
        @test norm(states.area_vec'*R,Inf) < 1e-10
        # Nonorthogonal corrections can grow with the size ratio without
        # producing a large eigenvalue; this is not a time-step selection rule.
        @test row_bound < 40*2.0^level
        @test density_substeps(R,15.,1.;beta_in_operator=true)[1] == 8
        # The diffusion-only explicit step must not develop a growing mode on
        # these deliberately unbalanced meshes with the original eight steps.
        amplification = eigvals(I + Matrix(R)/(8*15))
        @test maximum(abs,amplification) <= 1+1e-10
    end
end

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

@testset "Extruded Voronoi faces" begin
    mesh2d = create_voronoi_mesh((0.,0.),(1.,1.),3,3,StandardEl{1})
    topo = mesh2d.topo
    for _ in 1:3
        topo = remove_short_edges(topo)
    end
    mesh = extrude_to_3d(2,Mesh(topo,StandardEl{1}()),1.)
    cv = CellValues{3}(mesh)
    states = DesignVarInfo{3}(cv,0.3)
    R = compute_flux_operator_mat(cv,states,(β0=1.,))
    @test all(isfinite,nonzeros(R))
    @test norm(R*ones(length(states.χ_vec)),Inf) < 1e-9
    @test norm(states.area_vec'*R,Inf) < 1e-10
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
