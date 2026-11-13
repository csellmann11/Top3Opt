# julia --project=. --startup-file=no Tests/density_update_modes.jl
isdefined(@__MODULE__, :run_optimization) || include("regularization_benchmark.jl")

function density_fixture()
    mesh = create_rectangular_mesh(12,2,2,3.,0.5,1.,StandardEl{1})
    cv = CellValues{3}(mesh)
    states = DesignVarInfo{3}(cv,0.3)
    lam,mu = E_ν_to_lame(210.e3,0.33)
    pars = SimPars(Helmholtz{3,3}(Ψlin_totopt,(lam,mu,1.)),lam,mu,1e-3,15.,1.,0.3)
    trial = [0.3+0.13sin(2x[1])+0.04cos(5x[3]) for x in states.x_vec]
    project_volume!(states.design_vec,trial,states.area_vec,pars.ρ_init,pars.χmin)
    return cv,states,pars
end

@testset "Regularization coefficient 2*beta0*cbrt(volume)^2" begin
    cv,states,pars = density_fixture()
    coefficient = 2pars.β0 .* cbrt.(states.area_vec).^2
    @test prepare_density_cache(nothing,cv,states,pars,DensityOptions(),0.0).base_beta ≈ coefficient
    # Radius zero gives Helmholtz the same coefficient as r²_i.
    helmholtz = DensityOptions(regularization=:helmholtz)
    @test prepare_density_cache(nothing,cv,states,pars,helmholtz,0.0).radius2 ≈ coefficient
end

@testset "Conservative persistent design and remapping" begin
    unequal = DensityTransferWorkspace(Dict(11=>1,12=>2),[0.2,0.8],[1.0,3.0])
    @test transferred_design((id=10,parent_id=0,childs=[11,12]),unequal) ≈ 0.65
    for regularization in (:mean,:local_max,:helmholtz), projected in (false,true)
        cv,states,pars = density_fixture()
        options = DensityOptions(regularization=regularization)
        proj = DensityProjection(DensityProjectionOptions(enabled=projected),states.area_vec;rho_min=pars.χmin)
        cache = prepare_density_cache(nothing,cv,states,pars,options,0.15)
        map_design!(states,cache,proj,options)
        design_mass = dot(states.area_vec,states.design_vec)
        initial_design = copy(states.design_vec)
        @test dot(states.area_vec,states.χ_vec) ≈ design_mass atol=2e-10
        ref = falses(length(get_volumes(cv.mesh.topo)))
        coarse = copy(ref)
        parent = get_el_id(states,20)
        ref[parent] = true
        cv = adapt_mesh(cv,coarse,ref)
        transfer = DensityTransferWorkspace()
        update_states_after_mesh_adaption!(states,cv,ref,coarse;workspace=transfer)
        @test dot(states.area_vec,states.design_vec) ≈ design_mass atol=2e-12
        @test length(states.design_vec) > length(initial_design)
        cache = prepare_density_cache(cache,cv,states,pars,options,0.15)
        projection_rebuild!(proj,states.area_vec)
        map_design!(states,cache,proj,options)
        @test dot(states.area_vec,states.χ_vec) ≈ design_mass atol=2e-10
        @test all(x -> pars.χmin-1e-10 <= x <= 1+1e-10,states.χ_vec)
        # Coarsening averages the latent design with old physical cell volumes.
        ref = falses(length(get_volumes(cv.mesh.topo)))
        coarse = copy(ref)
        children = collect(get_volumes(cv.mesh.topo)[parent].childs)
        for (k,id) in enumerate(children)
            states.design_vec[get_state_id(states,id)] = 0.1+0.07k
            coarse[id] = true
        end
        design_mass = dot(states.area_vec,states.design_vec)
        cv = adapt_mesh(cv,coarse,ref)
        update_states_after_mesh_adaption!(states,cv,ref,coarse;workspace=transfer)
        @test dot(states.area_vec,states.design_vec) ≈ design_mass atol=2e-12
        @test length(states.design_vec) == length(initial_design)
        cache = prepare_density_cache(cache,cv,states,pars,options,0.15)
        projection_rebuild!(proj,states.area_vec)
        map_design!(states,cache,proj,options)
        @test dot(states.area_vec,states.χ_vec) ≈ design_mass atol=2e-10
    end
end

@testset "All production 3D regularization combinations" begin
    output_root = mktempdir(;prefix="toopt3-regularization-",cleanup=false)
    for regularization in (:mean,:local_max,:helmholtz), projected in (false,true), adaptive in (false,true)
        modes = regularization == :helmholtz ? (:explicit,) : (:explicit,:implicit)
        for mode in modes
            @testset "$regularization projection=$projected adaptive=$adaptive mode=$mode" begin
                cv,states,pars = density_fixture()
                destination = joinpath(output_root,"$(regularization)_$(projected)_$(adaptive)_$(mode)")
                reset_timer!(to)
                result = run_optimization(cv.mesh,x -> SA[0.,0.,0.],pars;
                    vtk_folder_name=destination,MAX_OPT_STEPS=3,MAX_REF_LEVEL=2,
                    regularization=regularization,projection=projected,update_mode=mode,
                    do_adaptivity=adaptive,filter_radius=0.15,take_snapshots_at=[1])
                @test length(result.strain_energy) == 3
                @test all(x -> isfinite(x) && x>0,result.strain_energy)
                @test length(result.projection_history) == 3
                @test all(row -> abs(row.design_volume-pars.ρ_init)<2e-9 &&
                    abs(row.physical_volume-pars.ρ_init)<2e-9,result.density_history)
                @test all(x -> pars.χmin-1e-10 <= x <= 1+1e-10,result.final_density)
                @test all(x -> pars.χmin <= x <= 1,result.final_design)
                @test isfile(joinpath(destination,"final_res.vtu"))
                @test isfile(joinpath(destination,"temp_res_1.vtu"))
                @test isfinite(result.final_strain_energy) && result.final_strain_energy > 0
                @test last(result.projection_history).beta == last(result.projection_history).next_beta
                @test result.projection_history[1].beta == (projected ? 1.0 : 0.0)
                @test adaptive || all(==(first(result.number_of_states)),result.number_of_states)
                @test result.simulation_times.state_update_time > 0
            end
        end
    end
    @info "3D regularization smoke outputs" output_root
end
