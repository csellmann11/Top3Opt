# Plain include entry for reusable density kernels; no module or global solver.
isdefined(@__MODULE__, :DensityOptions) || include("regularization/load.jl")

"""Mechanical descent force per unit cell volume, with displacement held fixed."""
function compute_driving_force!(p,energy,states::DesignVarInfo)
    @inbounds for i in eachindex(p,energy,states.χ_vec)
        p[i] = 3states.χ_vec[i]^2*energy[i]
    end
    return p
end

struct DensityMechanicalWorkspace{C}
    energy::Vector{Float64}
    driving::Vector{Float64}
    dofs::C
end
DensityMechanicalWorkspace(n) = DensityMechanicalWorkspace(zeros(n),zeros(n),CachedVector(Int))

function compute_strain_energy!(
    Ψvec::Vector{Float64},
    dofs,
    dh::DofHandler{D,U},
    eldata_col::Dict{Int,<:ElData},
    u::AbstractVector{Float64},
    states::DesignVarInfo{D}, 
    sim_pars::SimPars) where {D,U}
    mat_law = sim_pars.mat_law

    resize!(Ψvec,length(states.χ_vec))
    base = get_base(BaseInfo{3,1,3}())
    e2s = states.el_id_to_state_id

    for (el_id,elem_data) in eldata_col
        state_id = e2s[el_id]
        proj_s = stretch(elem_data.proj_s,Val(U))

        node_ids = elem_data.node_ids 
        bc   = states.x_vec[state_id]  
        h    = elem_data.hvol # VEM scaling length (cell diameter)

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



function compute_strain_energy(dh,eldata_col,u,states,sim_pars)
    return compute_strain_energy!(zeros(length(states.χ_vec)),CachedVector(Int),
        dh,eldata_col,u,states,sim_pars)
end

"""Build on mesh changes only. Helmholtz retains its work arrays across remeshing.
A new gradient cache starts without secant damping history.

All regularizations share the per-cell coefficient 2*beta0*h_i^2: the gradient
penalties use it as beta_i, Helmholtz as r²_i. A positive `radius` overrides
Helmholtz with a uniform physical radius; zero keeps the beta0 scaling.
The regularization length is h_i = cbrt(cell volume), the edge length on cubes.
"""
function prepare_density_cache(cache,cv,states,pars,options,radius)
    faces = build_tpfa_faces(cv,states)
    beta = [2pars.β0*h^2 for h in states.h_vec]
    if options.regularization == :helmholtz
        radius2 = radius > 0 ? fill(Float64(radius)^2,length(beta)) : beta
        return cache isa HelmholtzCache ? rebuild_filter!(cache,faces,radius2) :
            HelmholtzCache(faces,radius2)
    end
    return DensityCache(faces,beta)
end

"""Map the persistent design to the physical density. The gradient penalty evolves the
physical density itself (`design_vec === χ_vec`): there is nothing to map."""
map_design!(states,cache::DensityCache,::Nothing,options) = nothing
function map_design!(states,cache::HelmholtzCache,projection,options)
    map_filtered_design!(states.χ_vec,states.design_vec,cache,projection;
        tolerance=options.filter_tolerance)
    return nothing
end

"""Advance design with reusable force/energy buffers.

Returns the update result: the physical activity mask `changed` and the
move-limit counter `move_limited` (cells ending a substep on a box bound
tighter than [lower, 1], summed over the substeps). The gradient penalty updates
the physical density directly; its states share one vector for design and
physical density (`DesignVarInfo(...; separate_design=false)`).
"""
function state_update!(states::DesignVarInfo,cv::CellValues,pars::SimPars,
        cache,u,eldata_col;options=DensityOptions(),density_projection=nothing,
        workspace=DensityMechanicalWorkspace(length(states.χ_vec)))
    resize!(workspace.driving,length(states.χ_vec))
    compute_strain_energy!(workspace.energy,workspace.dofs,cv.dh,eldata_col,u,states,pars)
    refresh!(p) = compute_driving_force!(p,workspace.energy,states)
    if cache isa HelmholtzCache
        result = advance_filtered_design!(states.design_vec,states.χ_vec,workspace.driving,
            refresh!,cache;options,eta0=pars.η0,rho=pars.ρ_init,lower=pars.χmin,density_projection)
    else
        states.design_vec === states.χ_vec && density_projection === nothing || throw(ArgumentError(
            "The gradient penalty evolves the physical density: no projection, separate_design=false"))
        result = advance_density!(states.χ_vec,workspace.driving,refresh!,cache;
            options,eta0=pars.η0,rho=pars.ρ_init,lower=pars.χmin)
    end
    return result
end

"""Sharpening of a run (`projection = true`); the mechanism follows the regularization:
the smooth volume-preserving projection of the Helmholtz design (beta), or the
double-obstacle term of the gradient penalty (weight g, see obstacle.jl). Returns
(density_projection, obstacle); the unused one is `nothing`, and the settings of the
unused mechanism are ignored. Both continuations share the imbalance and gray tolerances.
"""
function sharpening(projection,regularization,area,lower;imbalance_tolerance=0.1,
        gray_tolerance=0.01,growth_rate=0.3,beta_initial=1.0,max_weight=Inf)
    projection || return (nothing,nothing)
    regularization == :helmholtz ||
        return (nothing,ObstacleContinuation(;max_weight,imbalance_tolerance,gray_tolerance))
    options = DensityProjectionOptions(;enabled=true,beta_initial,max_growth_rate=growth_rate,
        gray_tolerance,imbalance_tolerance)
    return DensityProjection(options,area;rho_min=lower),nothing
end

"""Prepare the next projection strength from the imbalance of the last design step,
remap, and include the continuation in the activity. The filter input is unchanged:
changing the projection steepness needs no Helmholtz solve."""
function advance_projection!(states,cache::HelmholtzCache,projection,options,imbalance;allow_advance=true)
    changed = projection_advance!(projection,imbalance;allow_advance)
    if changed
        projection_apply!(states.χ_vec,projection,cache.filtered)
        mark_physical_activity!(cache.changed,states.χ_vec,cache.activity_reference,options)
    end
    return changed
end

"""Advance the sharpening after the density update of outer iteration `iteration` and
return its history record (`nothing` without sharpening):

    (iteration, mechanism = :projection | :obstacle, value_used, value_next,
     threshold (NaN for the obstacle), grayness, imbalance, advanced).

The new beta or g takes effect in the next density update. A g step leaves the density
and the sensitivity unchanged, so neither a remap nor a secant reset follows it.
"""
function advance_sharpening!(states,cache,projection,obstacle,options,lower;iteration,allow_advance=true)
    imbalance = cache.balance.imbalance
    if projection !== nothing
        beta,threshold,grayness = projection.beta,projection.eta,projection.grayness
        advanced = advance_projection!(states,cache,projection,options,imbalance;allow_advance)
        return (;iteration,mechanism=:projection,value_used=beta,value_next=projection.beta,
            threshold,grayness,imbalance,advanced)
    end
    obstacle === nothing && return nothing
    weight = obstacle.weight
    grayness = grayness_metrics(states.χ_vec,states.area_vec,lower).grayness
    advanced = obstacle_advance!(obstacle,cache.balance,grayness;allow_advance)
    return (;iteration,mechanism=:obstacle,value_used=weight,value_next=obstacle.weight,
        threshold=NaN,grayness,imbalance,advanced)
end
