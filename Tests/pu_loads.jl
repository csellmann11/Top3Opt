# Independent continuous/discrete load checks, including nonuniform tractions.
include("pu_dwr.jl")

function pu_load_totals(c)
    resultant = zero(SVector{3,Float64})
    for load in c.loads.tetra
        for i in 1:10
            resultant += SA[load.force[3i-2],load.force[3i-1],load.force[3i]]
        end
    end
    return resultant
end

function pu_load_work(c,u,ed)
    pu3d_reconstruct!(c,u,ed)
    return sum(pu3d_continuous_load_work(c.loads,i,t,c.values)
        for (i,t) in enumerate(c.geometry.tets))
end

@testset "Nonuniform physical face traction and discrete load work" begin
    cv,states,pars = pu_fixture(2;hanging=true)
    ch,loads = ConstraintHandler{3}(cv.mesh),PU3DBoundaryData()
    add_face_set!(cv.mesh,"load",x -> x[1] ≈ 1)
    # Exact unit-face integrals: (2, 1/3, -5/6).
    traction(x) = SA[1+x[2]+x[3],x[2]*x[3]+x[2]^2*x[3]/2,-1+x[3]^2/2]
    pu3d_add_neumann!(loads,ch,cv.dh,cv.facedata_col,"load",traction)
    _,_,ed = assembly(cv,states,x -> SA[0.,0.,0.],pars)
    c = build_pu_dwr_cache(cv,states,ed,ch,pars;boundary_data=loads)
    @test pu_load_totals(c) ≈ SA[2.,1/3,-5/6] atol=5e-13
    for load in c.loads.tetra
        @test vec(sum(load.weighted;dims=2)) ≈ load.force atol=3e-15
    end
    # The exact displacement trace is affine. Integrate its work analytically
    # against the polynomial traction, independently of all PU quadrature.
    u = pu_interpolate(cv,x -> SA[2+x[2],0.,0.])
    @test pu_load_work(c,u,ed) ≈ 61/12 atol=5e-13
    # Cached discrete work must equal the actual assembled mechanics load;
    # a non-affine trace distinguishes it from the continuous work.
    u = pu_interpolate(cv,x -> SA[sin(x[2]),x[2]^2,x[3]^3])
    discrete_work = sum(pu3d_discrete_load_work(c.loads,k,u) for k in eachindex(states.χ_vec))
    @test discrete_work ≈ sum(value*u[dof] for (dof,value) in ch.n_bcs) atol=2e-13
end

@testset "Physical edge load is counted once across shared face fans" begin
    for hanging in (false,true)
        cv,states,pars = pu_fixture(2;hanging)
        ch,loads = ConstraintHandler{3}(cv.mesh),PU3DBoundaryData()
        add_edge_set!(cv.mesh,"line",x -> x[1] ≈ 0 && x[3] ≈ 0)
        traction(x) = SA[1+x[2]^2,2x[2],-1.]
        pu3d_add_neumann!(loads,ch,cv.dh,"line",traction)
        _,_,ed = assembly(cv,states,x -> SA[0.,0.,0.],pars)
        c = build_pu_dwr_cache(cv,states,ed,ch,pars;boundary_data=loads)
        @test pu_load_totals(c) ≈ SA[4/3,1.,-1.] atol=2e-13
        u = pu_interpolate(cv,x -> SA[1+x[2],0.,0.])
        @test pu_load_work(c,u,ed) ≈ 25/12 atol=2e-13
        discrete_work = sum(pu3d_discrete_load_work(c.loads,k,u) for k in eachindex(states.χ_vec))
        @test discrete_work ≈ sum(value*u[dof] for (dof,value) in ch.n_bcs) atol=2e-13
        # Endpoint load quadrature is intentionally retained for the discrete
        # mechanics; the continuous quadratic residual uses higher order.
        @test abs(discrete_work-pu_load_work(c,u,ed)) > 1e-4
    end
end

@testset "Reject incomplete loads and inhomogeneous essential data" begin
    cv,states,pars = pu_fixture(2)
    _,_,ed = assembly(cv,states,x -> SA[0.,0.,0.],pars)
    ch = ConstraintHandler{3}(cv.mesh)
    add_face_set!(cv.mesh,"load",x -> x[1] ≈ 1)
    add_neumann_bc!(ch,cv.dh,cv.facedata_col,"load",x -> SA[0.,0.,-1.])
    @test_throws ArgumentError build_pu_dwr_cache(cv,states,ed,ch,pars)
    ch = ConstraintHandler{3}(cv.mesh)
    loads = PU3DBoundaryData()
    add_face_set!(cv.mesh,"prescribed",x -> x[1] ≈ 0)
    pu3d_add_dirichlet!(loads,ch,cv.dh,cv.facedata_col,"prescribed",x -> SA[0.01,0.,0.])
    @test_throws ArgumentError build_pu_dwr_cache(cv,states,ed,ch,pars;boundary_data=loads)
end

@testset "Point supports do not prescribe auxiliary edge values" begin
    for kind in (:point,:edge,:face)
        cv,states,pars = pu_fixture(1)
        _,_,ed = assembly(cv,states,ZeroBodyForce(),pars)
        ch,loads = ConstraintHandler{3}(cv.mesh),PU3DBoundaryData()
        on_line(x) = x[1] ≈ 0 && x[3] ≈ 0
        if kind == :point
            add_node_set!(cv.mesh,"support",on_line)
            pu3d_add_dirichlet!(loads,ch,cv.dh,"support",x -> SA[0.];c_dofs=SA[1])
        elseif kind == :edge
            add_edge_set!(cv.mesh,"support",on_line)
            pu3d_add_dirichlet!(loads,ch,cv.dh,"support",x -> SA[0.];c_dofs=SA[1])
        else
            add_face_set!(cv.mesh,"support",x -> x[1] ≈ 0)
            pu3d_add_dirichlet!(loads,ch,cv.dh,cv.facedata_col,"support",x -> SA[0.];c_dofs=SA[1])
        end
        c = build_pu_dwr_cache(cv,states,ed,ch,pars;boundary_data=loads)
        g = c.geometry
        ti = findfirst(t -> on_line(g.node_points[t.nodes[1]]) &&
            on_line(g.node_points[t.nodes[2]]),g.tets)
        @test ti !== nothing
        t = g.tets[ti]
        @test c.fixed[t.nodes[1]][1] && c.fixed[t.nodes[2]][1]
        # Local P2 node5 is the midpoint of original edge(1,2). In the
        # cell-center patch its physical-boundary trace is otherwise free.
        midpoint_mask = pu3d_local_fixed(c,t,5,4)
        @test midpoint_mask == (kind == :point ? SA[false,false,false] : SA[true,false,false])
        face = findfirst(ids -> all(i -> g.node_points[i][1] ≈ 0,ids),g.face_nodes)
        @test c.fixed[length(g.vertex_ids)+face] ==
            (kind == :face ? SA[true,false,false] : SA[false,false,false])
    end
end
