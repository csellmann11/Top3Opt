include("density_timestepping.jl")
include("density_projection.jl")
include("density_implicit.jl")

function compute_driving_force!(
    pχv::Vector{Float64},
    Ψvec::Vector{Float64},
    states::DesignVarInfo) 

    for (state_id,(χ,Ψ0)) in enumerate(zip(states.χ_vec,Ψvec))
        pχv[state_id]    = -3χ^2 * Ψ0
    end
    pχv
end


function compute_strain_energy(
    dh::DofHandler{D,U},
    eldata_col::Dict{Int,<:ElData},
    u::AbstractVector{Float64},
    states::DesignVarInfo{D}, 
    sim_pars::SimPars) where {D,U}
    mat_law = sim_pars.mat_law

    Ψvec = zeros(length(states.χ_vec))
    base = get_base(BaseInfo{3,1,3}())
    e2s = states.el_id_to_state_id
    dofs = CachedVector(Int)

    for (el_id,elem_data) in eldata_col
        state_id = e2s[el_id]
        proj_s = stretch(elem_data.proj_s,Val(U))

        node_ids = elem_data.node_ids 
        bc   = states.x_vec[state_id]  
        h    = states.h_vec[state_id]

        setsize!(dofs,(length(node_ids)*U,))
        get_dofs!(dofs.array,dh,node_ids)
        uel = @view u[dofs]
        uπ = sol_proj(base,uel,proj_s)
        ∇u   = ∇x(uπ,h,zero(bc))
        Ψ0 = eval_psi_fun(mat_law,∇u,(sim_pars.λ,sim_pars.μ,1.0)) 

        Ψvec[state_id]    = Ψ0 
    end
    Ψvec
end


function get_avarage_driving_force(states::DesignVarInfo,
    p_χ::Vector{Float64},
    sim_pars::SimPars)

    ∑g_pχ = ∑g = 0.0
    for (χi,area,pχi) in zip(states.χ_vec,states.area_vec,p_χ)
        gχ = (χi-sim_pars.χmin)*(1-χi)
        ∑g_pχ += gχ * pχi * area
        ∑g    += gχ * area
    end
    # At a completely clipped design g vanishes everywhere. Use the volume
    # average in that degenerate case so the next update remains defined.
    return ∑g > 0 ? ∑g_pχ/∑g : dot(states.area_vec,p_χ)/sum(states.area_vec)
end



function state_update!(states::DesignVarInfo,
    cv::CellValues,
    sim_pars::SimPars, 
    laplace_operator::SparseMatrixCSC,
    u::AbstractVector{Float64},
    eldata_col::Dict{Int64, <:ElData};
    beta_in_operator::Bool = false,
    update_mode::Symbol = :explicit,
    implicit_cache::Union{Nothing,DensityImplicitCache} = nothing)


    dh = cv.dh
    χ_min = sim_pars.χmin
    

    hmin,hmax = extrema(states.h_vec)#./sqrt(3)
    n_steps, row_bound = density_substeps(laplace_operator,sim_pars.η0,sim_pars.β0;
        beta_in_operator,update_mode)
    println("[density] substeps=$n_steps, operator row bound=$row_bound, hmin=$hmin, hmax=$hmax, mode=$update_mode")
    flush(stdout)
    dt = 1.0/n_steps

    if update_mode == :implicit
        if isnothing(implicit_cache)
            implicit_cache = DensityImplicitCache(laplace_operator,sim_pars.η0,sim_pars.β0,
                states.h_vec; beta_in_operator)
        else
            validate_implicit_cache(implicit_cache,laplace_operator,sim_pars.η0,sim_pars.β0,
                states.h_vec; beta_in_operator)
        end
    end

    Δχ          = update_mode == :explicit ? zero(states.χ_vec) : Float64[]
    p_χ         = zero(states.χ_vec)
    χv          = states.χ_vec 
    χv_trial    = similar(χv)
    unconstrained = similar(χv)
    areav       = states.area_vec
    hv          = states.h_vec
    Ψvec        = compute_strain_energy(dh,eldata_col,u,states,sim_pars)


    state_initial = copy(states.χ_vec)

    for _ in 1:n_steps

        update_mode == :explicit && mul!(Δχ,laplace_operator,states.χ_vec)
        compute_driving_force!(p_χ,Ψvec,states)
        p_avg = get_avarage_driving_force(states,p_χ,sim_pars) |> abs
    
        isfinite(p_avg) || error("Nonfinite average driving force")
        # Zero strain energy gives no driving or regularization force in the
        # original scaling (both beta and eta contain p_avg).
        p_avg == 0 && break
        if update_mode == :implicit
            for i in eachindex(χv)
                unconstrained[i] = χv[i] - p_χ[i]/(sim_pars.η0*p_avg)
            end
            solve_info = implicit_density_solve!(unconstrained,unconstrained,implicit_cache)
            println("[density] GMRES iterations=$(solve_info.iterations), relative residual=$(solve_info.relative_residual)")
        else
            for i in eachindex(χv)
                beta_hat = beta_in_operator ? 1.0 : 2hv[i]^2*sim_pars.β0
                unconstrained[i] = χv[i] + dt/sim_pars.η0*(-p_χ[i]/p_avg + beta_hat*Δχ[i])
            end
        end
        # Rhat*ones == 0, so an unconstrained implicit multiplier response is
        # uniform. Apply the reference's separate bounds/volume projection.
        project_density_volume!(χv_trial,unconstrained,areav,sim_pars.ρ_init,χ_min)
        copyto!(states.χ_vec,χv_trial)
    end
    state_changed = (states.χ_vec .- state_initial) 
    return state_changed
end


