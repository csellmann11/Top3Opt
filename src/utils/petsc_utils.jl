################################################################################
# PETSc linear-solver backend: CG + GAMG (smoothed aggregation) with a rigid-body
# near-null space, for the 3-D linear-elasticity systems of ToOpt3.
#
# Serial only (MPI.COMM_SELF, one rank). Ported from the PETSc backend of
# AndersonPlasticity (src/linsolve/petsc_gamg.jl, benchmarks/petsc/REPORT.md):
#   * block size 3 + rigid-body near-null space (3 translations + 3 rotations
#     about the node centroid, Dirichlet DOFs zeroed, orthonormalised),
#   * GAMG aggregation threshold 0.01, set through the C API (PETSc.jl option
#     kwargs do not all reach the PC),
#   * optional l1-Jacobi Chebyshev level smoothers with a fixed interval,
#   * every PETSc object is destroyed explicitly, in reverse creation order:
#     the C heap is invisible to Julia's GC.
#
# The assembled CSC matrix is symmetric, so its arrays are also its CSR arrays:
# PETSc borrows them zero-copy (MatCreateSeqAIJWithArrays); only the index
# arrays are copied (0-based Int64).
#
# ENV knobs (read at every solve):
#   PETSC_CG_RTOL (1e-4)   PETSC_CG_MAXIT (1000)   PETSC_GAMG_THRESHOLD (0.01)
#   PETSC_GAMG_RBM (1)     PETSC_GAMG_L1CHEB (0)   PETSC_GAMG_CHEB_EMIN (0.1)
#   PETSC_GAMG_CHEB_EMAX (1.0)   PETSC_GAMG_VIEW (0)   PETSC_VERBOSE (1)
################################################################################
using LinearAlgebra
using SparseArrays
using Printf
import Libdl
using PETSc
using MPI

const PETSC_LIB = PETSc.petsclibs[findfirst(
    l -> PETSc.scalartype(l) == Float64 && PETSc.inttype(l) == Int64, PETSc.petsclibs)]

_petsc_env_f(name, default) = parse(Float64, get(ENV, name, default))
_petsc_env_i(name, default) = parse(Int, get(ENV, name, default))
_petsc_env_b(name, default) = get(ENV, name, default) == "1"

"""
    petsc_init!()

Initialise MPI (if needed) and the Float64/Int64 PETSc library. Idempotent.
"""
function petsc_init!()
    PETSc.initialized(PETSC_LIB) && return nothing
    MPI.Initialized() || MPI.Init()
    PETSc.initialize(PETSC_LIB)
    return nothing
end

# ── Direct C calls. The LibPETSc autowrap mistypes some output-pointer / string
#    arguments, so the GAMG / smoother knobs go through dlsym. PetscInt = Int64
#    and PetscReal = Cdouble for PETSC_LIB. ──────────────────────────────────────
const _PETSC_DLH = Ref{Ptr{Cvoid}}(C_NULL)
function _petsc_dlh()
    if _PETSC_DLH[] == C_NULL
        _PETSC_DLH[] = Libdl.dlopen(PETSC_LIB.petsc_library)
    end
    return _PETSC_DLH[]
end
_petsc_sym(name::Symbol) = Libdl.dlsym(_petsc_dlh(), name)
_petsc_check(r::Integer, what::String) = r == 0 || error("PETSc: $what failed (error code $r)")

function _petsc_ksp_get_pc(ksp_ptr::Ptr{Cvoid})
    pc = Ref{Ptr{Cvoid}}(C_NULL)
    _petsc_check(ccall(_petsc_sym(:KSPGetPC), Cint,
                       (Ptr{Cvoid}, Ptr{Ptr{Cvoid}}), ksp_ptr, pc), "KSPGetPC")
    return pc[]
end
_petsc_gamg_set_threshold(pc::Ptr{Cvoid}, thr::Float64) =
    _petsc_check(ccall(_petsc_sym(:PCGAMGSetThreshold), Cint,
                       (Ptr{Cvoid}, Ptr{Cdouble}, Int64), pc, [thr], 1), "PCGAMGSetThreshold")
function _petsc_mg_nlevels(pc::Ptr{Cvoid})
    n = Ref{Int64}(0)
    _petsc_check(ccall(_petsc_sym(:PCMGGetLevels), Cint,
                       (Ptr{Cvoid}, Ptr{Int64}), pc, n), "PCMGGetLevels")
    return Int(n[])
end
function _petsc_mg_smoother(pc::Ptr{Cvoid}, level::Int)
    k = Ref{Ptr{Cvoid}}(C_NULL)
    _petsc_check(ccall(_petsc_sym(:PCMGGetSmoother), Cint,
                       (Ptr{Cvoid}, Int64, Ptr{Ptr{Cvoid}}), pc, Int64(level), k), "PCMGGetSmoother")
    return k[]
end
function _petsc_ksp_type(ksp_ptr::Ptr{Cvoid})
    s = Ref{Ptr{Cchar}}(C_NULL)
    _petsc_check(ccall(_petsc_sym(:KSPGetType), Cint,
                       (Ptr{Cvoid}, Ptr{Ptr{Cchar}}), ksp_ptr, s), "KSPGetType")
    return s[] == C_NULL ? "" : unsafe_string(s[])
end
_petsc_pc_set_type(pc::Ptr{Cvoid}, name::String) =
    _petsc_check(ccall(_petsc_sym(:PCSetType), Cint, (Ptr{Cvoid}, Cstring), pc, name), "PCSetType($name)")
function _petsc_ksp_view(ksp_ptr::Ptr{Cvoid}, comm::MPI.Comm)
    viewer = ccall(_petsc_sym(:PETSC_VIEWER_STDOUT_), Ptr{Cvoid}, (MPI.MPI_Comm,), comm.val)
    _petsc_check(ccall(_petsc_sym(:KSPView), Cint, (Ptr{Cvoid}, Ptr{Cvoid}), ksp_ptr, viewer), "KSPView")
    return nothing
end

"""
    _petsc_configure_l1_chebyshev!(pc) -> number of levels configured

Put every Chebyshev level smoother (levels 1 … nlevels-1; level 0 is the coarse
solver) on l1-Jacobi with the fixed interval [emin, emax]. With
D = diag(Σ_j |a_ij|) the bound λ_max(D⁻¹A) ≤ 1 holds for every SPD A, so no
eigenvalue estimate is needed (AndersonPlasticity recipe, PETSC_GAMG_L1CHEB=1).
Must run after KSPSetUp (the level KSPs exist only then).
"""
function _petsc_configure_l1_chebyshev!(pc::Ptr{Cvoid})
    emin = _petsc_env_f("PETSC_GAMG_CHEB_EMIN", "0.1")
    emax = _petsc_env_f("PETSC_GAMG_CHEB_EMAX", "1.0")
    0.0 < emin < emax || error("PETSC_GAMG_CHEB_EMIN/EMAX: need 0 < emin < emax (got $emin, $emax)")
    n_cfg = 0
    for level in 1:_petsc_mg_nlevels(pc)-1
        k = _petsc_mg_smoother(pc, level)
        _petsc_ksp_type(k) == "chebyshev" || continue
        sub = _petsc_ksp_get_pc(k)
        _petsc_pc_set_type(sub, "none")     # type flip ⇒ diagonal recomputed with the l1 rule
        _petsc_pc_set_type(sub, "jacobi")
        _petsc_check(ccall(_petsc_sym(:PCJacobiSetType), Cint,
                           (Ptr{Cvoid}, Cint), sub, Cint(1)), "PCJacobiSetType(ROWL1)")
        _petsc_check(ccall(_petsc_sym(:PCJacobiSetRowl1Scale), Cint,
                           (Ptr{Cvoid}, Cdouble), sub, 1.0), "PCJacobiSetRowl1Scale")
        # KSPChebyshevSetEigenvalues(ksp, emax, emin): emax FIRST; also disables the estimator
        _petsc_check(ccall(_petsc_sym(:KSPChebyshevSetEigenvalues), Cint,
                           (Ptr{Cvoid}, Cdouble, Cdouble), k, emax, emin), "KSPChebyshevSetEigenvalues")
        n_cfg += 1
    end
    return n_cfg
end

"""
    rigid_body_modes(cv, ch, n) -> Vector{Vector{Float64}} or nothing

Six orthonormal rigid-body vectors (3 translations, 3 rotations about the
centroid of the active nodes) in DOF order, with the Dirichlet-constrained DOFs
zeroed so that they match the BC-applied operator. Returns `nothing` when the
DOF space is not made of node-contiguous 3-vectors (e.g. moment unknowns), in
which case GAMG runs without a near-null space.
"""
function rigid_body_modes(cv::CellValues{D,U}, ch::ConstraintHandler, n::Int) where {D,U}
    (D == 3 && U == 3) || return nothing
    dof_mapping = cv.dh.dof_mapping
    n_nodes = length(dof_mapping)
    n == 3n_nodes || return nothing
    xyz = zeros(3, n_nodes)
    for node in cv.mesh.nodes
        is_active(node) || continue
        dofs = dof_mapping[node.id]
        ((dofs[1] - 1) % 3 == 0 && dofs[2] == dofs[1] + 1 && dofs[3] == dofs[1] + 2) || return nothing
        k = (dofs[1] - 1) ÷ 3 + 1
        xyz[1, k] = node.coords[1]; xyz[2, k] = node.coords[2]; xyz[3, k] = node.coords[3]
    end
    c = vec(sum(xyz, dims=2)) ./ n_nodes
    W = zeros(n, 6)
    @inbounds for k in 1:n_nodes
        dx = xyz[1, k] - c[1]; dy = xyz[2, k] - c[2]; dz = xyz[3, k] - c[3]
        r = 3 * (k - 1)
        W[r+1, 1] = 1.0;  W[r+2, 2] = 1.0;  W[r+3, 3] = 1.0    # translations
        W[r+2, 4] = -dz;  W[r+3, 4] =  dy                       # rotation about x
        W[r+1, 5] =  dz;  W[r+3, 5] = -dx                       # rotation about y
        W[r+1, 6] = -dy;  W[r+2, 6] =  dx                       # rotation about z
    end
    for d in keys(ch.d_bcs)
        1 <= d <= n && (W[d, :] .= 0.0)
    end
    F = cholesky(Symmetric(W' * W); check=false)          # Gram orthonormalisation
    if !issuccess(F)
        @warn "rigid_body_modes: Gram matrix not SPD (degenerate mesh / constraints); GAMG runs without a near-null space"
        return nothing
    end
    W = W / F.U
    return [W[:, j] for j in 1:6]
end

# Guarded destroy (reference pattern from AndersonPlasticity): one failing destroy
# must not skip the remaining ones, or the rest of the C objects would leak.
function _petsc_try_destroy(f::Function, what::String)
    try
        f()
    catch err
        err isa InterruptException && rethrow()
        @warn "PETSc: destroying the $what failed; continuing with the remaining objects" exception = (err, catch_backtrace())
    end
    return nothing
end

# Julia-side buffers PETSc borrows during a solve; kept in one struct so they
# provably outlive the PETSc objects.
mutable struct _PetscSolveBuffers
    ia::Vector{Int64}
    ja::Vector{Int64}
    a::Vector{Float64}
    b::Vector{Float64}
    x::Vector{Float64}
    nns::Vector{Vector{Float64}}
end

"""
    solve_lse_petsc(k_global, rhs_global, cv, ch) -> u::Vector{Float64}

Solve `k_global * u = rhs_global` (SPD, BCs already applied) with PETSc CG
preconditioned by GAMG (block size 3, rigid-body near-null space). All PETSc
objects are created and destroyed inside this call.
"""
function solve_lse_petsc(k_global::SparseMatrixCSC, rhs_global::AbstractVector,
                         cv::CellValues, ch::ConstraintHandler)
    petsc_init!()
    LP   = PETSc.LibPETSc
    lib  = PETSC_LIB
    comm = MPI.COMM_SELF
    n    = size(k_global, 1)
    @assert size(k_global, 2) == n == length(rhs_global)
    rtol  = _petsc_env_f("PETSC_CG_RTOL", "1e-4")
    maxit = _petsc_env_i("PETSC_CG_MAXIT", "1000")
    t0 = time_ns()

    # CSC arrays of the symmetric matrix == CSR arrays; PETSc wants 0-based PetscInt
    ia = Vector{Int64}(undef, n + 1)
    @inbounds for i in 1:n+1
        ia[i] = k_global.colptr[i] - 1
    end
    ja = Vector{Int64}(undef, nnz(k_global))
    @inbounds for p in eachindex(ja)
        ja[p] = k_global.rowval[p] - 1
    end
    a = k_global.nzval isa Vector{Float64} ? k_global.nzval : Vector{Float64}(k_global.nzval)
    b = Vector{Float64}(rhs_global)
    x = zeros(n)
    buffers = _PetscSolveBuffers(ia, ja, a, b, x, Vector{Float64}[])

    A = nothing; ksp = nothing; vb = nothing; vx = nothing; nullsp = nothing
    nns_vecs = Any[]
    bs = 1; rbm_on = false
    its = 0; reason = 0; t_setup = 0.0; t_solve = 0.0
    try
        A = LP.MatCreateSeqAIJWithArrays(lib, comm, Int64(n), Int64(n), buffers.ia, buffers.ja, buffers.a)
        if n % 3 == 0
            try
                LP.MatSetBlockSize(lib, A, Int64(3))
                bs = 3
            catch err
                @warn "PETSc: MatSetBlockSize(3) failed; continuing with bs=1 and no near-null space" err
            end
        end
        if bs == 3 && _petsc_env_b("PETSC_GAMG_RBM", "1")
            rbm = rigid_body_modes(cv, ch, n)
            if rbm !== nothing
                buffers.nns = rbm
                for col in buffers.nns
                    push!(nns_vecs, LP.VecCreateSeqWithArray(lib, comm, Int64(3), Int64(n), col))
                end
                nullsp = LP.MatNullSpaceCreate(lib, comm, LP.PETSC_FALSE, Int64(6), [v.ptr for v in nns_vecs])
                LP.MatSetNearNullSpace(lib, A, nullsp)
                rbm_on = true
            end
        end
        vb = LP.VecCreateSeqWithArray(lib, comm, Int64(bs), Int64(n), buffers.b)
        vx = LP.VecCreateSeqWithArray(lib, comm, Int64(bs), Int64(n), buffers.x)

        ksp = PETSc.KSP(A; ksp_type = "cg", ksp_norm_type = "unpreconditioned",
                        ksp_rtol = rtol, ksp_max_it = maxit, pc_type = "gamg")
        pc = _petsc_ksp_get_pc(ksp.ptr)
        _petsc_gamg_set_threshold(pc, _petsc_env_f("PETSC_GAMG_THRESHOLD", "0.01"))

        t1 = time_ns()
        LP.KSPSetUp(lib, ksp)                          # builds the GAMG hierarchy
        _petsc_env_b("PETSC_GAMG_L1CHEB", "0") && _petsc_configure_l1_chebyshev!(pc)
        t_setup = (time_ns() - t1) / 1e9
        _petsc_env_b("PETSC_GAMG_VIEW", "0") && _petsc_ksp_view(ksp.ptr, comm)

        t2 = time_ns()
        PETSc.solve!(vx, ksp, vb)
        t_solve = (time_ns() - t2) / 1e9
        its    = Int(LP.KSPGetIterationNumber(lib, ksp))
        reason = Int(LP.KSPGetConvergedReason(lib, ksp))
    finally
        # Reverse creation order; C-side memory is invisible to the Julia GC.
        # Every destroy is guarded individually (see _petsc_try_destroy).
        _petsc_try_destroy("KSP") do
            ksp === nothing || PETSc.destroy(ksp)
        end
        _petsc_try_destroy("KSP options object") do        # per-solve PetscOptions (finalizer-guarded)
            (ksp === nothing || ksp.opts === nothing) || PETSc.destroy(ksp.opts)
        end
        _petsc_try_destroy("solution Vec") do
            vx === nothing || PETSc.destroy(vx)
        end
        _petsc_try_destroy("rhs Vec") do
            vb === nothing || PETSc.destroy(vb)
        end
        _petsc_try_destroy("MatNullSpace") do
            nullsp === nothing || LP.MatNullSpaceDestroy(lib, nullsp)
        end
        for v in nns_vecs
            _petsc_try_destroy("near-null-space Vec") do
                PETSc.destroy(v)
            end
        end
        _petsc_try_destroy("Mat") do
            A === nothing || PETSc.destroy(A)
        end
    end

    if reason < 0
        if reason == -3                                # KSP_DIVERGED_ITS
            @warn "PETSc CG+GAMG hit the iteration limit without reaching rtol" its maxit rtol
        else
            error("PETSc CG+GAMG diverged: KSPConvergedReason = $reason after $its iterations")
        end
    end
    if _petsc_env_b("PETSC_VERBOSE", "1")
        r = k_global * buffers.x
        r .-= buffers.b
        relres = norm(r) / max(norm(buffers.b), eps())
        @printf("[petsc] n=%d nnz=%d bs=%d rbm=%d  CG+GAMG its=%d reason=%d relres=%.2e  setup=%.2fs solve=%.2fs total=%.2fs\n",
                n, nnz(k_global), bs, Int(rbm_on), its, reason, relres,
                t_setup, t_solve, (time_ns() - t0) / 1e9)
        flush(stdout)
    end
    return buffers.x
end

"""
    solve_lse(k_global, rhs_global, cv, ch)

Linear-solver dispatcher: `LINEAR_SOLVER == :petsc` (default) → CG + GAMG via
PETSc, `:hypre` → PCG + BoomerAMG via HYPRE (`solve_lse_hypre`).
"""
function solve_lse(k_global::SparseMatrixCSC, rhs_global::AbstractVector,
                   cv::CellValues, ch::ConstraintHandler)
    backend = (@isdefined LINEAR_SOLVER) ? LINEAR_SOLVER : :petsc
    if backend === :petsc
        return @timeit to "petsc_solver" solve_lse_petsc(k_global, rhs_global, cv, ch)
    elseif backend === :hypre
        return solve_lse_hypre(k_global, rhs_global)
    else
        error("Unknown LINEAR_SOLVER = $backend (use :petsc or :hypre)")
    end
end
