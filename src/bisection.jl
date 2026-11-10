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



function compute_strain_energy(dh,eldata_col,u,states,sim_pars)
    return compute_strain_energy!(zeros(length(states.χ_vec)),CachedVector(Int),
        dh,eldata_col,u,states,sim_pars)
end

"""Build on mesh changes only. Helmholtz retains its work arrays across remeshing.

All regularizations share the per-cell coefficient 2*beta0*h_i^2: the gradient
penalties use it as beta_i, Helmholtz as r²_i. A positive `radius` overrides
Helmholtz with a uniform physical radius; zero keeps the beta0 scaling.
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

function map_design!(states,cache::DensityCache,projection,options)
    projection_apply!(states.χ_vec,projection,states.design_vec)
    return nothing
end
function map_design!(states,cache::HelmholtzCache,projection,options)
    map_filtered_design!(states.χ_vec,states.design_vec,cache,projection;
        tolerance=options.filter_tolerance)
    return nothing
end

"""Advance design with reusable force/energy buffers and return physical activity."""
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
        legacy_substeps = options.integrator == :semi_implicit ? 1 : max(1,4ceil(Int,24pars.β0/pars.η0))
        result = advance_density!(states.design_vec,workspace.driving,refresh!,cache;
            options,eta0=pars.η0,rho=pars.ρ_init,lower=pars.χmin,legacy_substeps,
            physical=states.χ_vec,density_projection)
    end
    return result.changed
end

"""Prepare the next projection strength, remap, and include continuation in activity."""
function advance_projection!(states,cache,projection,options,objective;allow_advance=true)
    changed = projection_advance!(projection,objective;allow_advance)
    if changed
        # Continuation changes only the scalar projection, not the filter input.
        input = cache isa HelmholtzCache ? cache.filtered : states.design_vec
        projection_apply!(states.χ_vec,projection,input)
        mark_physical_activity!(cache.changed,states.χ_vec,cache.activity_reference,options)
    end
    return changed
end
