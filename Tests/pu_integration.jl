include("pu_dwr.jl")

@testset "PU-DWR cache follows refinement and coarsening" begin
    cv,states,pars,ch,loads,u,K,ed,c = pu_clamped_problem(2)
    original_count = length(states.χ_vec)
    estimate_pu_dwr!(c,u,states,ed,pars)
    workers = c.workspaces
    parent = get_el_id(states,1)
    transfer = DensityTransferWorkspace()
    for refine_now in (true,false)
        ref = falses(length(get_volumes(cv.mesh.topo)))
        coarse = copy(ref)
        if refine_now
            ref[parent] = true
        else
            coarse[collect(get_volumes(cv.mesh.topo)[parent].childs)] .= true
        end
        cv = adapt_mesh(cv,coarse,ref)
        update_states_after_mesh_adaption!(states,cv,ref,coarse;workspace=transfer)
        ch = ConstraintHandler{3}(cv.mesh)
        loads = PU3DBoundaryData()
        add_face_set!(cv.mesh,"clamp",x -> x[1] ≈ 0)
        add_face_set!(cv.mesh,"traction",x -> x[1] ≈ 1)
        pu3d_add_dirichlet!(loads,ch,cv.dh,cv.facedata_col,"clamp",x -> SA[0.,0.,0.])
        pu3d_add_neumann!(loads,ch,cv.dh,cv.facedata_col,"traction",x -> SA[0.,0.,-1.])
        u,K,ed = compute_displacement(cv,ch,states,x -> SA[0.,0.,0.],pars)
        c = build_pu_dwr_cache(cv,states,ed,ch,pars;boundary_data=loads,previous=c)
        @test c.workspaces === workers
        result = estimate_pu_dwr!(c,u,states,ed,pars)
        @test length(result.indicators) == length(states.χ_vec)
        @test isfinite(result.signed) && result.magnitude > 0
        @test sum(result.indicators) ≈ result.magnitude rtol=2e-14
        @test refine_now ? length(states.χ_vec) > original_count : length(states.χ_vec) == original_count
    end
end

@testset "PU-DWR production adaptive runs" begin
    output_root = mktempdir(;prefix="toopt3-pu-dwr-",cleanup=false)
    for regularization in (:mean,:local_max,:helmholtz)
        mesh = create_rectangular_mesh(12,2,2,3.,0.5,1.,StandardEl{1})
        lambda,mu = E_ν_to_lame(210.e3,0.33)
        pars = SimPars(Helmholtz{3,3}(Ψlin_totopt,(lambda,mu,1.)),lambda,mu,1e-3,15.,1.,0.3)
        destination = joinpath(output_root,string(regularization))
        result = run_optimization(mesh,x -> SA[0.,0.,0.],pars;
            vtk_folder_name=destination,MAX_OPT_STEPS=3,MAX_REF_LEVEL=2,
            error_estimator=:pu_dwr,regularization,projection=true,
            update_mode=regularization == :helmholtz ? :explicit : :implicit,
            filter_radius=0.15,take_snapshots_at=[1],do_adaptivity=true)
        @test result.error_estimator == :pu_dwr
        @test result.pu_marking_exponent == 1.0
        @test length(result.error_history) == 3
        @test all(e -> isfinite(e.signed) && e.magnitude > 0 && abs(e.signed) <= e.magnitude,result.error_history)
        @test all(x -> isfinite(x) && x>0,result.strain_energy)
        @test all(row -> abs(row.design_volume-pars.ρ_init)<2e-9 &&
            abs(row.physical_volume-pars.ρ_init)<2e-9,result.density_history)
        @test isfile(joinpath(destination,"temp_res_1.vtu"))
        @test isfile(joinpath(destination,"final_res.vtu"))
        @test maximum(result.number_of_states) > first(result.number_of_states)
    end
    @info "PU-DWR production output" output_root
end
