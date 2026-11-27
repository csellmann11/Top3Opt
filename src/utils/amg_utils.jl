using LinearAlgebra
using SparseArrays
using Printf
include("hypre_conversion.jl")

# Size, iteration count and native setup/solve time of the most recent linear solve
# (either backend); the driver reads it for the per-step calibration record.
const LAST_SOLVE = Ref((n=0, nnz=0, iterations=0, setup=0.0, solve=0.0))

function solve_lse_hypre(
    k_global::SparseMatrixCSC,
    rhs_global::AbstractVector;
    workspace::HypreConversionWorkspace = _hypre_conversion_workspace())

    t_start = time_ns()
    size(k_global,1) == length(rhs_global) ||
        throw(DimensionMismatch("HYPRE matrix and right-hand side sizes must agree"))
    precond = solver = A = b = x = nothing
    u = Vector{Float64}(undef, length(rhs_global))
    t_setup = t_solve = relres = 0.0
    iterations = 0
    try
        # Refill task-local Julia conversion buffers, but create a fresh native
        # matrix for this step. No old connectivity or AMG hierarchy is reused.
        @timeit to "hypre_matrix_conversion" A = _hypre_matrix(k_global,workspace)
        precond = HYPRE.BoomerAMG(;
            NumFunctions=3,       # 3 DOFs for elasticity
            CoarsenType=10,       # HMIS (High-Parallel/Low-Memory Coarsening)
            RelaxType=6,          # Sym G.S./Jacobi
            # Aggressive coarsening on the first levels: 2-4x cheaper setup, about
            # twice the iterations, lower total (measured 2026-11-25). 0 disables it.
            AggNumLevels=parse(Int, get(ENV, "HYPRE_AGG_NUM_LEVELS", "1")),
            NumSweeps=1,
            MaxIter=1,
            Tol=0.0
        )

        solver = HYPRE.PCG(;
            MaxIter=1000,
            Tol=1e-4,
            PrintLevel=1,
            Precond=precond      # Attach the AMG preconditioner
        )

        # Keep explicit handles so the C-side allocations can be released at
        # the end of this step instead of waiting for Julia's garbage collector.
        b = HYPRE.HYPREVector(convert(Vector{Float64}, rhs_global))
        x = zero(b)
        @timeit to "hypre_solver" begin
            # These are the same two calls made by HYPRE.solve!. PCG setup
            # invokes the attached BoomerAMG preconditioner's setup, so this
            # measures PCG + AMG setup rather than preconditioner-only work.
            t_setup_start = time_ns()
            @timeit to "hypre_setup" HYPRE.LibHYPRE.@check HYPRE.LibHYPRE.HYPRE_ParCSRPCGSetup(solver, A, b, x)
            t_setup = (time_ns() - t_setup_start) / 1e9

            t_solve_start = time_ns()
            @timeit to "hypre_solve" HYPRE.LibHYPRE.@check HYPRE.LibHYPRE.HYPRE_ParCSRPCGSolve(solver, A, b, x)
            t_solve = (time_ns() - t_solve_start) / 1e9
            iterations = HYPRE.GetNumIterations(solver)
            relres = HYPRE.GetFinalRelativeResidualNorm(solver)
        end
        copy!(u, x)
    finally
        # Deterministically release every HYPRE C object created above.
        # Base.finalize runs the C *Destroy now and is idempotent (guards on
        # pointer != C_NULL), so the atexit/GC sweep later is a no-op.
        for object in (solver,precond,x,b,A)
            object === nothing || Base.finalize(object)
        end
    end

    @printf("[hypre] n=%d nnz=%d PCG+BoomerAMG its=%d relres=%.2e setup(PCG+AMG)=%.3fs solve=%.3fs total=%.3fs\n",
            length(rhs_global), nnz(k_global), iterations, relres,
            t_setup, t_solve, (time_ns() - t_start) / 1e9)
    flush(stdout)
    LAST_SOLVE[] = (n=length(rhs_global), nnz=nnz(k_global), iterations=Int(iterations),
                    setup=t_setup, solve=t_solve)
    return u

end
