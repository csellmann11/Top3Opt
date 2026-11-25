# julia --project=. --startup-file=no Tests/density_update_modes.jl
isdefined(@__MODULE__, :run_optimization) || include("regularization_benchmark.jl")

"""The gradient penalty evolves the physical density itself: `separate_design=false`."""
function density_fixture(;separate_design=true)
    mesh = create_rectangular_mesh(12,2,2,3.,0.5,1.,StandardEl{1})
    cv = CellValues{3}(mesh)
    states = DesignVarInfo{3}(cv,0.3;separate_design)
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
    # Only the Helmholtz design has a projection; the gradient penalty shares one vector.
    for label in (:mean,:local_max,:local_lambda,:local_free,:helmholtz), projected in (false,true)
        helmholtz = label == :helmholtz
        helmholtz || !projected || continue
        cv,states,pars = density_fixture(separate_design=helmholtz)
        @test (states.design_vec === states.χ_vec) == !helmholtz
        options = helmholtz ? DensityOptions(regularization=:helmholtz) : DensityOptions(scaling=label)
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

@testset "Sharpening follows the regularization; a g step keeps the secant" begin
    @test sharpening(false,:gradient,[1.0],1e-3) == (nothing,nothing)
    @test sharpening(false,:helmholtz,[1.0],1e-3) == (nothing,nothing)
    # The settings of the unused mechanism are ignored (an invalid one does not throw).
    projection,obstacle = sharpening(true,:gradient,[1.0,2.0],1e-3;max_weight=5.0,
        imbalance_tolerance=0.2,gray_tolerance=0.05,growth_rate=-1.0,beta_initial=0.0)
    @test projection === nothing && obstacle isa ObstacleContinuation
    @test (obstacle.weight,obstacle.max_weight,obstacle.imbalance_tolerance,obstacle.gray_tolerance) == (0.0,5.0,0.2,0.05)
    projection,obstacle = sharpening(true,:helmholtz,[1.0,2.0],1e-3;max_weight=-1.0,
        imbalance_tolerance=0.2,gray_tolerance=0.05,growth_rate=0.5,beta_initial=2.0)
    @test obstacle === nothing && projection.beta == 2.0
    @test (projection.options.max_growth_rate,projection.options.imbalance_tolerance,
        projection.options.gray_tolerance) == (0.5,0.2,0.05)

    # Gradient penalty with damping: the density update measures the balance; a g step
    # keeps the secant history (g does not enter the driving force).
    cv,states,pars = density_fixture(separate_design=false)
    options = DensityOptions(diag=true,substeps=1,measure_balance=true)
    cache = prepare_density_cache(nothing,cv,states,pars,options,0.0)
    _,obstacle = sharpening(true,:gradient,states.area_vec,pars.χmin)
    force! = p -> (@. p = 2-3states.χ_vec)
    outer!() = (cache.obstacle = obstacle.weight; advance_density!(states.χ_vec,zeros(length(states.χ_vec)),
        force!,cache;options,eta0=pars.η0,rho=pars.ρ_init,lower=pars.χmin))
    outer!(); outer!()
    @test cache.secant_valid && cache.damping_active
    @test isfinite(cache.balance.imbalance) && cache.balance.obstacle_unit > 0
    # A balanced last substep (imbalance 0) gives g its full budget.
    cache.balance = merge(cache.balance,(imbalance=0.0,))
    damping = copy(cache.damping)
    record = advance_sharpening!(states,cache,nothing,obstacle,options,pars.χmin;iteration=2)
    @test record.mechanism == :obstacle && record.advanced && record.value_used == 0.0
    @test record.value_next == obstacle.weight ≈ 0.1*(cache.balance.force+cache.balance.diffusion)/cache.balance.obstacle_unit
    @test isnan(record.threshold) && record.imbalance == 0.0
    @test record.grayness ≈ grayness_metrics(states.χ_vec,states.area_vec,pars.χmin).grayness
    @test cache.secant_valid && cache.damping_active && cache.damping == damping
    outer!()
    @test cache.obstacle == obstacle.weight > 0 && cache.damping_active
    # The last iteration does not advance.
    cache.balance = merge(cache.balance,(imbalance=0.0,))
    record = advance_sharpening!(states,cache,nothing,obstacle,options,pars.χmin;iteration=3,allow_advance=false)
    @test !record.advanced && record.value_used == record.value_next
    @test advance_sharpening!(states,cache,nothing,nothing,options,pars.χmin;iteration=3) === nothing

    # Helmholtz: beta grows by the imbalance rule and the physical density is remapped.
    cv,states,pars = density_fixture()
    options = DensityOptions(regularization=:helmholtz,measure_balance=true)
    projection,_ = sharpening(true,:helmholtz,states.area_vec,pars.χmin)
    cache = prepare_density_cache(nothing,cv,states,pars,options,0.15)
    map_design!(states,cache,projection,options)
    advance_filtered_design!(states.design_vec,states.χ_vec,zeros(length(states.χ_vec)),
        p -> (@. p = 2-3states.χ_vec),cache;options,eta0=pars.η0,rho=pars.ρ_init,lower=pars.χmin,
        density_projection=projection)
    @test isfinite(cache.balance.imbalance)
    cache.balance = merge(cache.balance,(imbalance=0.05,))
    before = copy(states.χ_vec)
    record = advance_sharpening!(states,cache,projection,nothing,options,pars.χmin;iteration=1)
    @test record.mechanism == :projection && record.advanced
    @test record.value_used == 1.0 && record.value_next ≈ exp(0.3*0.5) && record.threshold > 0
    @test states.χ_vec != before
    @test dot(states.area_vec,states.χ_vec) ≈ dot(states.area_vec,before) atol=2e-10
end

@testset "All production 3D regularization combinations" begin
    output_root = mktempdir(;prefix="toopt3-regularization-",cleanup=false)
    # Labels are the former regularization names: a scaling of the gradient penalty, or helmholtz.
    # Damping and the move limit run on the adaptive meshes (remesh resets the secant history).
    for label in (:mean,:local_max,:local_lambda,:local_free,:helmholtz), projected in (false,true), adaptive in (false,true)
        integrators = label in (:helmholtz,:local_lambda,:local_free) ? (:explicit,) : (:explicit,:semi_implicit)
        extras = label == :helmholtz ? ((false,Inf),(false,0.05)) :
            adaptive ? ((false,Inf),(true,Inf),(true,0.05)) : ((false,Inf),)
        for integrator in integrators, (diag,move_limit) in extras
            @testset "$label projection=$projected adaptive=$adaptive $integrator diag=$diag move_limit=$move_limit" begin
                cv,states,pars = density_fixture()
                destination = joinpath(output_root,"$(label)_$(projected)_$(adaptive)_$(integrator)_$(diag)_$(move_limit)")
                reset_timer!(to)
                setting = label == :helmholtz ? (regularization=:helmholtz,) : (scaling=label,)
                result = run_optimization(cv.mesh,x -> SA[0.,0.,0.],pars;
                    vtk_folder_name=destination,MAX_OPT_STEPS=3,MAX_REF_LEVEL=2,
                    setting...,integrator,diag,move_limit,projection=projected,
                    do_adaptivity=adaptive,filter_radius=0.15,take_snapshots_at=[1])
                @test result.scaling == (label == :helmholtz ? :mean : label)
                @test result.integrator == integrator && result.diag == diag
                @test all(row -> row.move_limited >= 0,result.density_history)
                @test isfinite(move_limit) || all(row -> row.move_limited == 0,result.density_history)
                @test length(result.strain_energy) == 3
                @test all(x -> isfinite(x) && x>0,result.strain_energy)
                @test length(result.projection_history) == (projected ? 3 : 0)
                @test all(row -> abs(row.design_volume-pars.ρ_init)<2e-9 &&
                    abs(row.physical_volume-pars.ρ_init)<2e-9,result.density_history)
                @test all(x -> pars.χmin-1e-10 <= x <= 1+1e-10,result.final_density)
                @test all(x -> pars.χmin <= x <= 1,result.final_design)
                @test isfile(joinpath(destination,"final_res.vtu"))
                @test isfile(joinpath(destination,"temp_res_1.vtu"))
                @test isfinite(result.final_strain_energy) && result.final_strain_energy > 0
                if projected
                    history = result.projection_history
                    mechanism = label == :helmholtz ? :projection : :obstacle
                    @test all(row -> row.mechanism == mechanism,history)
                    @test [row.iteration for row in history] == 1:3
                    @test last(history).value_used == last(history).value_next && !last(history).advanced
                    @test first(history).value_used == (label == :helmholtz ? 1.0 : 0.0)
                    @test all(row -> isnan(row.imbalance) || 0 <= row.imbalance <= 1+1e-12,history)
                end
                label == :helmholtz || @test result.final_design == result.final_density
                @test adaptive || all(==(first(result.number_of_states)),result.number_of_states)
                @test result.simulation_times.state_update_time > 0
            end
        end
    end
    @info "3D regularization smoke outputs" output_root
end
