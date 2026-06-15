using LinearAlgebra
using SparseArrays

function solve_lse(
    k_global::SparseMatrixCSC,
    rhs_global::AbstractVector)


    precond = HYPRE.BoomerAMG(;
        NumFunctions=3,       # 3 DOFs for elasticity
        CoarsenType=10,       # HMIS (High-Parallel/Low-Memory Coarsening)
        RelaxType=6,          # Sym G.S./Jacobi
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

    # Build the HYPRE objects explicitly. The convenience call
    # HYPRE.solve(solver, ::SparseMatrixCSC, ::Vector) hides the HYPREMatrix,
    # the rhs HYPREVector and the solution vector it allocates internally, so
    # their C handles can never be freed → the BoomerAMG hierarchy + matrix
    # leak on the C heap every step (Julia GC underestimates the ~32 B wrapper).
    A = HYPRE.HYPREMatrix(k_global)                              # COMM_SELF, rows 1:n
    b = HYPRE.HYPREVector(convert(Vector{Float64}, rhs_global))
    x = zero(b)                                                  # solution vector

    u = Vector{Float64}(undef, length(rhs_global))
    try
        @timeit to "hypre_solver" HYPRE.solve!(solver, x, A, b)
        copy!(u, x)
    finally
        # Deterministically release every HYPRE C object created above.
        # Base.finalize runs the C *Destroy now and is idempotent (guards on
        # pointer != C_NULL), so the atexit/GC sweep later is a no-op.
        foreach(Base.finalize, (A, b, x, solver, precond))
    end

    return u

end