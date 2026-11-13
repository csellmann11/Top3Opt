using Ju3VEM.VEMGeo: is_active_root


struct SimPars{H<:Helmholtz}
    mat_law::H
    λ::Float64
    μ::Float64 
    
    χmin::Float64
    η0::Float64 
    β0::Float64
    ρ_init::Float64
end


@inline function Ψlin_totopt(∇u::M,λ,μ,χ) where {M<:AbstractMatrix}
    ε = 1/2*(∇u + ∇u')
    W = λ/2 * tr(ε)^2 + μ*tr(ε*ε)
    return W*χ[1]^3
end

"""Cellwise physical density and persistent design with active-cell geometry.

`χ_vec` is used by mechanics; `design_vec` is the unfiltered, unprojected
optimization variable. `area_vec` contains cell volumes in 3D. Both fields are
initialized together; after editing a custom design, call `map_design!` before
solving mechanics. Adaptivity transfers design and then rebuilds the mapping.
"""
struct DesignVarInfo{D}
    χ_vec::Vector{Float64} # physical density used by mechanics and adaptivity
    design_vec::Vector{Float64} # persistent unfiltered, unprojected design
    x_vec::Vector{SVector{D,Float64}}
    area_vec::Vector{Float64} #TODO: rename to elsize_vec
    h_vec::Vector{Float64} # regularization length cbrt(cell volume); mechanics uses ElData.hvol

    el_id_to_state_id::OrderedDict{Int,Int}
end

function get_el_id(states::DesignVarInfo,state_id::Integer)
    return states.el_id_to_state_id.keys[state_id]
end
function get_state_id(states::DesignVarInfo,el_id::Integer)
    return states.el_id_to_state_id[el_id]
end


function fill_states!(states::DesignVarInfo{D},cv::CellValues{D,U},ρ_init) where {D,U}

    topo = cv.mesh.topo 
    for (state_id,element) in enumerate(RootIterator{D+1}(topo))
        vol_data = precompute_volume_monomials(element.id,topo,cv.facedata_col,Val(1))
        bc                     = vol_data.vol_bc
        area                   = vol_data.integrals[1]

        states.χ_vec[state_id]               = ρ_init
        states.design_vec[state_id]          = ρ_init
        states.x_vec[state_id]               = bc
        states.area_vec[state_id]            = area
        states.h_vec[state_id]               = cbrt(area)
        states.el_id_to_state_id[element.id] = state_id
    end
end

function resize_states!(states::DesignVarInfo{D},
    n_states::Int) where {D}
    
    resize!(states.χ_vec,n_states)
    resize!(states.design_vec,n_states)
    resize!(states.x_vec,n_states)
    resize!(states.area_vec,n_states)
    resize!(states.h_vec,n_states)
    empty!(states.el_id_to_state_id)
end


function DesignVarInfo{D}(cv::CellValues{D,U},ρ_init) where {D,U}


    mesh = cv.mesh
    topo = mesh.topo

    n_states = length(RootIterator{D+1}(topo))

    χ_vec              = Vector{Float64}(undef,n_states)
    x_vec              = Vector{SVector{D,Float64}}(undef, n_states)
    area_vec           = Vector{Float64}(undef,n_states)
    h_vec              = Vector{Float64}(undef,n_states)
    el_id_to_state_id  = OrderedDict{Int,Int}()

    states = DesignVarInfo(χ_vec,similar(χ_vec),x_vec,area_vec,h_vec,el_id_to_state_id)
    fill_states!(states,cv,ρ_init)

    return states
end



"""Reusable conservative transfer storage, independent of filtering/projection."""
struct DensityTransferWorkspace
    element_to_state::Dict{Int,Int}
    design::Vector{Float64}
    volume::Vector{Float64}
end
DensityTransferWorkspace() = DensityTransferWorkspace(Dict{Int,Int}(),Float64[],Float64[])

function backup_design!(work::DensityTransferWorkspace, states::DesignVarInfo)
    empty!(work.element_to_state)
    merge!(work.element_to_state,states.el_id_to_state_id)
    resize!(work.design,length(states.design_vec))
    resize!(work.volume,length(states.area_vec))
    copyto!(work.design,states.design_vec)
    copyto!(work.volume,states.area_vec)
    return work
end

"""Inherit a parent's design or volume-average old children during coarsening."""
function transferred_design(element, work::DensityTransferWorkspace)
    old = work.element_to_state
    haskey(old,element.id) && return work.design[old[element.id]]
    haskey(old,element.parent_id) && return work.design[old[element.parent_id]]
    mass,total = 0.0,0.0
    for child in element.childs
        sid = get(old,child,0)
        sid != 0 || error("Missing design transfer child $child")
        mass += work.volume[sid]*work.design[sid]
        total += work.volume[sid]
    end
    total > 0 || error("Cannot transfer design to element $(element.id)")
    return mass/total
end

"""Transfer persistent design; remap physical density before the next mechanical solve.

Physical density is temporarily initialized from design. The driver then applies
its rebuilt filter and projection; transferring physical density alone would
lose the design history. The optional workspace avoids new field backups.
"""
function update_states_after_mesh_adaption!(states::DesignVarInfo,
        cv::CellValues{D,U},ref_marker::AbstractVector{Bool},coarse_marker::AbstractVector{Bool};
        workspace::DensityTransferWorkspace=DensityTransferWorkspace()) where {D,U}
    backup_design!(workspace,states)
    topo = cv.mesh.topo
    resize_states!(states,length(RootIterator{D+1}(topo)))
    fill_states!(states,cv,0.0)
    for (id,sid) in states.el_id_to_state_id
        value = transferred_design(get_volumes(topo)[id],workspace)
        states.design_vec[sid] = value
        states.χ_vec[sid] = value
    end
    return states
end

# Retain the previous positional call used by existing mesh scripts.
function update_states_after_mesh_adaption!(states::DesignVarInfo,cv::CellValues,
        eldata_col::Dict,ref_marker::AbstractVector{Bool},coarse_marker::AbstractVector{Bool};kwargs...)
    return update_states_after_mesh_adaption!(states,cv,ref_marker,coarse_marker;kwargs...)
end


function measure_of_nondiscreteness(
    states::DesignVarInfo,
    pars::SimPars)


    ∑Ω  = 0.0 
    ∑Ωχ = 0.0 
    χmin = pars.χmin
    for (χ,el_size) in zip(states.χ_vec,states.area_vec) 
        ∑Ω  += el_size
        ∑Ωχ += el_size * (1 - χ)*(χ-χmin)
    end
    return 4∑Ωχ/∑Ω
end