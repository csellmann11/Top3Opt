# CPU integration checks for the complete density update, including VEM driving
# forces, all existing regularization operators, and an actual mesh refinement.
# Run: julia --project=. --startup-file=no Tests/density_update_modes.jl
include("regularization_benchmark.jl")
include("../src/utils/general_utils.jl")
include("../src/postprocessing/sim_data.jl")
include("../src/postprocessing/topopt_vtk_export.jl")
include("../src/optim_run.jl")

function update_mode_fixture()
    # The MBB traction occupies x in [0,0.251]; resolve that patch so the
    # fixture has a nonzero load even before protected-boundary refinement.
    mesh = create_rectangular_mesh(12,2,2,3.,0.5,1.,StandardEl{1})
    cv = CellValues{3}(mesh)
    states = DesignVarInfo{3}(cv,0.3)
    lam,mu = E_ν_to_lame(210.e3,0.33)
    pars = SimPars(Helmholtz{3,3}(Ψlin_totopt,(lam,mu,1.)),
        lam,mu,1e-3,15.,2.,0.3)
    # Nonuniform initial densities exercise diffusion as well as the force.
    trial = [0.3+0.13sin(2x[1])+0.04cos(5x[3]) for x in states.x_vec]
    states.χ_vec .= reference_volume_projection(trial,states.area_vec,pars)
    return cv,states,pars
end

function reference_volume_projection(trial,volumes,pars)
    lo,hi = minimum(trial)-1,maximum(trial)-pars.χmin
    for _ in 1:100
        shift = (lo+hi)/2
        mass = dot(volumes,clamp.(trial .- shift,pars.χmin,1.))/sum(volumes)
        mass > pars.ρ_init ? (lo=shift) : (hi=shift)
    end
    return clamp.(trial .- (lo+hi)/2,pars.χmin,1.)
end

function update_mode_operator(scheme,cv,states,pars)
    if scheme in (:diamond,:tpfa)
        return compute_flux_operator_mat(cv,states,pars;scheme)
    elseif scheme == :taylor
        return compute_taylor_operator_mat(cv,states,pars)
    elseif scheme == :strong
        neighbors,ghosts = create_neigh_list(states,cv)
        return compute_laplace_operator_mat(cv.mesh.topo,neighbors,ghosts,states,true)
    end
    error("Unknown scheme $scheme")
end

function direct_imex_reference(states,cv,pars,R,u,ed;beta_in_operator)
    chi = states.χ_vec
    psi = compute_strain_energy(cv.dh,ed,u,states,pars)
    force = -3 .* chi.^2 .* psi
    g = (chi .- pars.χmin) .* (1 .- chi)
    p_avg = abs(dot(states.area_vec,g.*force)/dot(states.area_vec,g))
    @assert isfinite(p_avg) && p_avg > 0
    row_beta = beta_in_operator ? ones(length(chi)) : 2pars.β0 .* states.h_vec.^2
    A = Matrix{Float64}(I,length(chi),length(chi)) -
        Diagonal(row_beta/pars.η0)*Matrix(R)
    unbounded = A \ (chi-force/(pars.η0*p_avg))
    return reference_volume_projection(unbounded,states.area_vec,pars)
end

function check_update_density(states,pars)
    @test all(isfinite,states.χ_vec)
    @test all(x -> pars.χmin <= x <= 1.,states.χ_vec)
    @test abs(dot(states.area_vec,states.χ_vec)/sum(states.area_vec)-pars.ρ_init) <= 1e-8
end

@testset "Density update modes: CPU integration" begin
    for scheme in (:tpfa,:diamond,:taylor,:strong)
        @testset "$scheme" begin
            cv,states,pars = update_mode_fixture()
            beta_in_operator = scheme != :strong
            ch = create_constraint_handler(cv,:MBB_sym)
            u,_,ed = compute_displacement(cv,ch,states,x -> SA[0.,0.,0.],pars)
            R = update_mode_operator(scheme,cv,states,pars)
            initial = copy(states.χ_vec)

            # The public default continues to be the old explicit path exactly.
            default_change = state_update!(states,cv,pars,R,u,ed;beta_in_operator)
            default_result = copy(states.χ_vec)
            copyto!(states.χ_vec,initial)
            explicit_change = state_update!(states,cv,pars,R,u,ed;
                beta_in_operator,update_mode=:explicit)
            @test states.χ_vec == default_result
            @test explicit_change == default_change
            check_update_density(states,pars)

            # With beta=0 both modes have exactly one source-only step.
            pars0 = SimPars(pars.mat_law,pars.λ,pars.μ,pars.χmin,
                pars.η0,0.,pars.ρ_init)
            R0 = update_mode_operator(scheme,cv,states,pars0)
            copyto!(states.χ_vec,initial)
            state_update!(states,cv,pars0,R0,u,ed;beta_in_operator,update_mode=:explicit)
            source_only = copy(states.χ_vec)
            copyto!(states.χ_vec,initial)
            state_update!(states,cv,pars0,R0,u,ed;beta_in_operator,update_mode=:implicit)
            @test states.χ_vec ≈ source_only atol=1e-12 rtol=0
            check_update_density(states,pars0)

            # Preserve the existing zero-energy convention: p_avg=0 means
            # neither the source nor regularization changes the density.
            for mode in (:explicit,:implicit)
                copyto!(states.χ_vec,initial)
                change = state_update!(states,cv,pars,R,zero(u),ed;
                    beta_in_operator,update_mode=mode)
                @test states.χ_vec == initial
                @test all(iszero,change)
            end

            # Compare the complete implicit update with independent dense linear
            # algebra and a plain bisection, including the force normalization.
            copyto!(states.χ_vec,initial)
            expected = direct_imex_reference(states,cv,pars,R,u,ed;beta_in_operator)
            cache = DensityImplicitCache(R,pars.η0,pars.β0,states.h_vec;beta_in_operator)
            changed = state_update!(states,cv,pars,R,u,ed;
                beta_in_operator,update_mode=:implicit,implicit_cache=cache)
            @test states.χ_vec ≈ expected atol=1e-6 rtol=0
            @test changed == states.χ_vec-initial
            check_update_density(states,pars)
            cached_result = copy(states.χ_vec)

            copyto!(states.χ_vec,initial)
            state_update!(states,cv,pars,R,u,ed;beta_in_operator,update_mode=:implicit)
            @test states.χ_vec ≈ cached_result atol=1e-12 rtol=0

            # Reuse the same sparse matrix/workspace for another density step.
            expected = direct_imex_reference(states,cv,pars,R,u,ed;beta_in_operator)
            state_update!(states,cv,pars,R,u,ed;
                beta_in_operator,update_mode=:implicit,implicit_cache=cache)
            @test states.χ_vec ≈ expected atol=1e-6 rtol=0
            check_update_density(states,pars)

            # A same-sized reassembled matrix must not silently use stale data.
            before_failure = copy(states.χ_vec)
            @test_throws ArgumentError state_update!(states,cv,pars,copy(R),u,ed;
                beta_in_operator,update_mode=:implicit,implicit_cache=cache)
            @test states.χ_vec == before_failure
            @test_throws ArgumentError state_update!(states,cv,pars,R,u,ed;
                beta_in_operator,update_mode=:invalid)
            @test states.χ_vec == before_failure

            # Exercise the production refinement and density-transfer routines.
            # One marked cell guarantees an actual new mesh, independent of
            # the optimization's current density/error marking thresholds.
            old_count = length(states.χ_vec)
            old_mass = dot(states.area_vec,states.χ_vec)
            ref = fill(false,length(get_volumes(cv.mesh.topo)))
            coarse = copy(ref)
            sid = argmin([norm(x-SA[1.4,0.25,0.5]) for x in states.x_vec])
            ref[get_el_id(states,sid)] = true
            clear_up_topo!(cv.mesh.topo)
            cv = adapt_mesh(cv,coarse,ref)
            update_states_after_mesh_adaption!(states,cv,ed,ref,coarse)
            @test length(states.χ_vec) > old_count
            @test dot(states.area_vec,states.χ_vec) ≈ old_mass atol=1e-12
            @test maximum(states.h_vec)/minimum(states.h_vec) > 1.5
            Rnew = update_mode_operator(scheme,cv,states,pars)
            ch = create_constraint_handler(cv,:MBB_sym)
            u,_,ed = compute_displacement(cv,ch,states,x -> SA[0.,0.,0.],pars)
            before_failure = copy(states.χ_vec)
            @test_throws ArgumentError state_update!(states,cv,pars,Rnew,u,ed;
                beta_in_operator,update_mode=:implicit,implicit_cache=cache)
            @test states.χ_vec == before_failure

            cache_new = DensityImplicitCache(Rnew,pars.η0,pars.β0,states.h_vec;
                beta_in_operator)
            expected = direct_imex_reference(states,cv,pars,Rnew,u,ed;beta_in_operator)
            state_update!(states,cv,pars,Rnew,u,ed;
                beta_in_operator,update_mode=:implicit,implicit_cache=cache_new)
            @test states.χ_vec ≈ expected atol=1e-6 rtol=0
            check_update_density(states,pars)
        end
    end
end

@testset "Production optimization update-mode wiring" begin
    # Every run receives a fresh nonexistent child directory. The driver never
    # reaches its existing-output deletion branch; keep these small artifacts
    # in the system temp directory for inspection after the test.
    output_root = mktempdir(;prefix="toopt3-density-modes-",cleanup=false)
    for mode in (:explicit,:implicit), adaptive in (false,true)
        @testset "$mode adaptive=$adaptive" begin
            mesh = create_rectangular_mesh(6,2,2,3.,0.5,1.,StandardEl{1})
            lam,mu = E_ν_to_lame(210.e3,0.33)
            pars = SimPars(Helmholtz{3,3}(Ψlin_totopt,(lam,mu,1.)),
                lam,mu,1e-3,15.,2.,0.3)
            destination = joinpath(output_root,"$(mode)_adaptive_$(adaptive)")
            @test !ispath(destination)
            reset_timer!(to)
            result = run_optimization(mesh,x -> SA[0.,0.,0.],pars;
                vtk_folder_name=destination,MAX_OPT_STEPS=3,MAX_REF_LEVEL=2,
                flux_scheme=:tpfa,update_mode=mode,do_adaptivity=adaptive,
                take_snapshots_at=Int[])
            @test length(result.mod) == 3
            @test all(isfinite,result.mod)
            @test all(x -> isfinite(x) && x > 0,result.strain_energy)
            @test all(>(0),result.number_of_states)
            @test adaptive || all(==(first(result.number_of_states)),result.number_of_states)
            @test isfile(joinpath(destination,"final_res.vtu"))

            # Density timing includes implicit matrix/workspace preparation,
            # without charging that same work to adaptation. A fixed mesh reuses
            # its cache; adaptive runs rebuild whenever the operator is assembled.
            density_timer = to["state_update"]
            @test TimerOutputs.ncalls(density_timer) == 3
            @test result.simulation_times.state_update_time == TimerOutputs.time(density_timer)/1e9
            @test !haskey(to["adaptivity"],"prepare_implicit_density")
            if mode == :implicit
                @test haskey(density_timer,"prepare_implicit_density")
                preparation_timer = density_timer["prepare_implicit_density"]
                @test TimerOutputs.ncalls(preparation_timer) == (adaptive ? 3 : 1)
                @test 0 < TimerOutputs.time(preparation_timer) <= TimerOutputs.time(density_timer)
            else
                @test !haskey(density_timer,"prepare_implicit_density")
            end
        end
    end
    @info "Density update production smoke outputs" output_root
end
