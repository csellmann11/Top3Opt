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
        Tol=1e-6,
        PrintLevel=1,
        Precond=precond      # Attach the AMG preconditioner
    )

    @timeit to "hypre_solver" u = HYPRE.solve(solver, k_global, rhs_global)

    return u

end