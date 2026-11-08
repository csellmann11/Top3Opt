"""Stable diagnostic schema for an active projection and the identity mapping."""
density_projection_metadata(p) = (density_projection_enabled=p !== nothing,
    density_projection_beta=p === nothing ? 0.0 : p.beta,
    density_projection_threshold=p === nothing ? 0.5 : p.eta)

projection_threshold(p) = p === nothing ? 0.5 : p.eta
diagnostic_timer(options) = options.diagnostics ? time_ns() : UInt64(0)
diagnostic_seconds(options,start) = options.diagnostics ? (time_ns()-start)*1e-9 : 0.0

"""Each internal face occurs once; distances are centroid-to-face normal distances.
The dimension-independent names `area` and `length` mean cell volume and face
area in 3D. Cell volumes may share mesh storage. Rebuild on remeshing.
"""
struct TPFAFaces
    area::Vector{Float64}
    left::Vector{Int}
    right::Vector{Int}
    length::Vector{Float64}
    distance_left::Vector{Float64}
    distance_right::Vector{Float64}
end

Base.@kwdef struct DensityOptions
    integrator::Symbol = :explicit
    regularization::Symbol = :mean
    # Zero retains the old subdivision of one optimization time unit.
    substeps::Int = 0
    diagnostics::Bool = false
    volume_tolerance::Float64 = 1e-10
    max_substeps::Int = 10000
    filter_tolerance::Float64 = 1e-12
    implicit_tolerance::Float64 = 1e-10
    implicit_max_iterations::Int = 1000
    # Absolute physical-density change used by adaptive mesh marking.
    activity_tolerance::Float64 = 1e-3
end

density_activity_tolerance(options::DensityOptions) =
    max(options.activity_tolerance,
        10max(options.volume_tolerance,options.filter_tolerance,eps(Float64)))

"""Merge physical-density changes into the cached per-iteration activity mask.
The reference is the physical field at the start of the outer iteration. Keeping
it fixed across substeps and continuation detects cumulative changes, while the
OR retains any excursion above the threshold even if the field later returns.
"""
function mark_physical_activity!(changed,physical,reference,options::DensityOptions)
    tolerance = density_activity_tolerance(options)
    @inbounds for i in eachindex(changed,physical,reference)
        changed[i] |= abs(physical[i]-reference[i]) > tolerance
    end
    return changed
end

struct DiagnosticWorkspace
    reg_before::Vector{Float64}
    reg_after::Vector{Float64}
    trial::Vector{Float64}
    projected::Vector{Float64}
    physical_before::Vector{Float64}
end
DiagnosticWorkspace(n::Int) = DiagnosticWorkspace(ntuple(_ -> zeros(n),5)...)

mutable struct DensityCache
    faces::TPFAFaces
    base_beta::Vector{Float64}
    beta_relative::Vector{Float64}
    stiffness::SparseMatrixCSC{Float64,Int}
    diagonal_indices::Vector{Int}
    face_indices::Vector{NTuple{2,Int}}
    system::Union{Nothing,SparseMatrixCSC{Float64,Int}}
    preconditioner::Diagonal{Float64,Vector{Float64}}
    solver::Union{Nothing,CgWorkspace{Float64,Float64,Vector{Float64}}}
    system_dirty::Bool
    system_dt::Float64
    system_eta0::Float64
    old::Vector{Float64}
    rhs::Vector{Float64}
    trial::Vector{Float64}
    design_driving::Vector{Float64}
    activity_reference::Vector{Float64}
    changed::BitVector
    diagnostic_work::Union{Nothing,DiagnosticWorkspace}
    fields::NamedTuple
end

"""Reusable coordinate/CSR storage for rebuilding a changing TPFA graph.
`sparse!` reuses rows/cols/values as the returned CSC arrays. The matrix therefore
belongs to this workspace and is invalidated by its next assembly.
"""
struct TPFAAssemblyWorkspace
    rows::Vector{Int}
    cols::Vector{Int}
    values::Vector{Float64}
    last_touch::Vector{Int}
    csr_rowptr::Vector{Int}
    csr_cols::Vector{Int}
    csr_values::Vector{Float64}
end
TPFAAssemblyWorkspace() = TPFAAssemblyWorkspace(Int[],Int[],Float64[],Int[],Int[],Int[],Float64[])

struct FilterDiagnostics
    product::Vector{Float64}
    trial::Vector{Float64}
    projected::Vector{Float64}
    filtered_before::Vector{Float64}
end
FilterDiagnostics(n) = FilterDiagnostics(ntuple(_ -> zeros(n),4)...)

"""
Helmholtz operator B = M + K(r²), with the baseline coefficient chosen by
regularization_coefficient(r0²,h_i); currently uniform r²_i = r0².
The caller owns the persistent unfiltered design; this cache owns only geometry,
solver work, and update scratch. Rebuild after remapping the design on a new mesh.

Jacobi-preconditioned CG avoids a sparse factorization on every adaptive step.
Both forward and adjoint applications reuse the same matrix and solver workspace.
"""
mutable struct HelmholtzCache
    faces::TPFAFaces
    radius2::Vector{Float64}
    assembly::TPFAAssemblyWorkspace
    system::SparseMatrixCSC{Float64,Int}
    preconditioner::Diagonal{Float64,Vector{Float64}}
    solver::CgWorkspace{Float64,Float64,Vector{Float64}}
    rhs::Vector{Float64}
    q::Vector{Float64}
    old::Vector{Float64}
    design_before::Vector{Float64}
    trial::Vector{Float64}
    filtered::Vector{Float64}
    filtered_driving::Vector{Float64}
    activity_reference::Vector{Float64}
    changed::BitVector
    diagnostic_work::Union{Nothing,FilterDiagnostics}
    fields::NamedTuple
end
