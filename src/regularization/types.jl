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

"""Density update settings. Three orthogonal choices define a gradient-penalty update:

- `scaling` the boost b_i: `:mean` (b = 1), `:local_max`, `:local_lambda` or `:local_free`
  (both explicit only), see `update_coefficients!`;
- `integrator` `:explicit` (the fewest equal substeps inside the measured CFL limit)
  or `:semi_implicit` (implicit diffusion, one step);
- `diag` the secant reaction damping, see `update_secant_damping!`.

`regularization=:helmholtz` selects the filter update instead; it has neither a
boost nor damping. `move_limit` bounds |x_i^p - x_old,i^p| per unit pseudo-time in
every cell, p = `simp_exponent`, for every update (`Inf` disables it): a per-cell
box in the volume projection, see `move_box!`.
"""
Base.@kwdef struct DensityOptions
    regularization::Symbol = :gradient
    scaling::Symbol = :mean
    integrator::Symbol = :explicit
    diag::Bool = false
    move_limit::Float64 = Inf
    # SIMP exponent of the move-limit measure; the mechanics uses chi^3.
    simp_exponent::Float64 = 3.0
    # Zero (default): substeps only from the measured CFL limit (one semi-implicit
    # step). n > 0 additionally caps the substep length at 1/n.
    substeps::Int = 0
    diagnostics::Bool = false
    volume_tolerance::Float64 = 1e-10
    # Hard cap per optimization step; an explicit step may then exceed its CFL bound.
    max_substeps::Int = 1000
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
    weights::Vector{Float64}
end
DiagnosticWorkspace(n::Int) = DiagnosticWorkspace(ntuple(_ -> zeros(n),6)...)

"""
`stiffness` is always the baseline TPFA operator K(base_beta). `beta_relative`
is the local boost b_i >= 1/2 of `:local_max` (b_i = max(m_i,|lambda|/s)/2 after a
`:local_lambda` step, the boost b_i(l) the update used after a `:local_free` step;
all ones for `:mean`): it scales
both beta_i and the viscosity eta_i, so the update divides the driving force and
the multipliers of a cell by b_i and never changes `stiffness` or the system.
`volume_weight` is the response of the unconstrained update to a unit volume
multiplier, used by `project_volume!`. `push_boost` holds the boost of `:local_free`
for cells that gain material (see `update_coefficients!`); it is written only by that
scaling and stays empty otherwise.

The secant damping history (`damping` D_i, the previous entry state
`secant_design`/`secant_force`) is empty until `diag` is used; `damping_ratio`
r_i = D_i/(s*eta*b_i) belongs to the current substep. A new cache (run start,
every remesh) has no history; `reset_secant!` clears it on a projection beta step.
`move_lower`/`move_upper` hold the move-limit box of the current substep; they
stay empty without a move limit.
"""
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
    system_damped::Bool
    damping::Vector{Float64}
    damping_ratio::Vector{Float64}
    secant_design::Vector{Float64}
    secant_force::Vector{Float64}
    secant_valid::Bool
    damping_active::Bool
    old::Vector{Float64}
    rhs::Vector{Float64}
    trial::Vector{Float64}
    volume_weight::Vector{Float64}
    design_driving::Vector{Float64}
    move_lower::Vector{Float64}
    move_upper::Vector{Float64}
    activity_reference::Vector{Float64}
    changed::BitVector
    diagnostic_work::Union{Nothing,DiagnosticWorkspace}
    fields::NamedTuple
    push_boost::Vector{Float64}
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
r²_i = 2*beta0*h_i^2, h_i = cbrt(cell volume), by default (as for the penalties),
or a uniform r0² override.
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
    move_lower::Vector{Float64}
    move_upper::Vector{Float64}
    activity_reference::Vector{Float64}
    changed::BitVector
    diagnostic_work::Union{Nothing,FilterDiagnostics}
    fields::NamedTuple
end
