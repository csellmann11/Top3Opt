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
    return ∑g_pχ/∑g
end



function lower_upper_bound(p_χ::Vector{Float64},η::Float64,dt::Float64)

    min_val,max_val = extrema(p_χ)
    return min_val - η/dt, max_val + η/dt
end


function state_update!(states::DesignVarInfo,
    dh::DofHandler,
    sim_pars::SimPars, 
    laplace_operator::SparseMatrixCSC,
    u::AbstractVector{Float64},
    eldata_col::Dict{Int64, <:ElData})


    χ_min = sim_pars.χmin
    

    MAX_ITER = 1000
 
    hmin,hmax = extrema(states.h_vec)#./sqrt(3)
    β0 = 2*hmax^2 * sim_pars.β0
 
    #TODO: hmin should be the minimal distance between two nodes --> very large n_steps for voronoi?
    n_steps = 4*ceil(Int,12/sim_pars.η0 * β0/hmin^2)
    dt = 1.0/n_steps

    Δχ          = zero(states.χ_vec)
    p_χ         = zero(states.χ_vec)
    χv          = states.χ_vec 
    χv_trial    = similar(χv)
    areav       = states.area_vec
    hv          = states.h_vec
    Ψvec        = compute_strain_energy(dh,eldata_col,u,states,sim_pars)


    state_initial = copy(states.χ_vec)

    for _ in 1:n_steps

        Δχ .= laplace_operator * states.χ_vec
        compute_driving_force!(p_χ,Ψvec,states)
        p_avg = get_avarage_driving_force(states,p_χ,sim_pars) 
    
        η    = sim_pars.η0 * p_avg
        
        iter = 0
        λ_trial = 0.0
        copyto!(χv_trial,states.χ_vec)
        λ_lower, λ_upper = lower_upper_bound(p_χ,η,dt)
        ρ_trial = 1.0

        while abs(sim_pars.ρ_init - ρ_trial) > 1e-8

            iter += 1
            ∑χ = 0.0  
            ∑Ω = 0.0    
            
            for (state_id,(χi,area,h,pχi,Δχi)) in enumerate(zip(χv,areav,hv,p_χ,Δχ))
                 
                β = 2*max(h^2,hmin^2)*p_avg*sim_pars.β0
    
                dχ = dt/η * (pχi - λ_trial + β * Δχi)
                χv_trial[state_id] = clamp(χi + dχ,χ_min,1.0)

                ∑Ω += area
                ∑χ += area * χv_trial[state_id]
            end

            ρ_trial = ∑χ/∑Ω


            ρ_trial > sim_pars.ρ_init ? λ_lower = λ_trial : λ_upper = λ_trial 

            λ_trial = (λ_lower + λ_upper)/2.0

            if iter > MAX_ITER
                error("Max iterations reached")
                break
            end
        end

        copyto!(states.χ_vec,χv_trial)
    end
    state_changed = (states.χ_vec .- state_initial) 
    return state_changed
end


