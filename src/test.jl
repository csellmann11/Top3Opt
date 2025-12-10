using HYPRE
using SparseArrays
using LinearAlgebra

HYPRE.Init()
# 1. Load or Generate your system
# Replace this with your actual stiffness matrix A and load vector b.
# A must be a SparseMatrixCSC (standard Julia sparse matrix).
# b must be a standard Vector.
n = 10
N = n^3 * 3
A = sprand(N, N, 5/N) + I * 10.0 # Dummy matrix (ensure yours is SPD for PCG)
A = A + A' # Make it symmetric for this example
b = rand(N)

# 2. Configure the Preconditioner (BoomerAMG)
# "NumFunctions = 3" is crucial for 3D elasticity to handle vector unknowns efficiently.
# "RelaxType = 6" (Symmetric Hybrid Gauss-Seidel/Jacobi) is typically chosen 
# when using AMG as a preconditioner for Conjugate Gradient.
precond = HYPRE.BoomerAMG(; 
    NumFunctions = 3,      # 3 degrees of freedom per node (u,v,w)
    CoarsenType = 10,      # HMIS coarsening (often faster for larger 3D problems)
    RelaxType = 6,         # Symmetric smoothing for PCG
    NumSweeps = 1,         # 1 sweep of smoothing
    MaxIter = 1,           # Perform 1 V-cycle per preconditioner application
    Tol = 0.0              # Do not stop on tolerance inside preconditioner
)

# 3. Configure the Solver (PCG)
# We use Conjugate Gradient (PCG) as the outer solver.
# If your matrix is not Symmetric Positive Definite, use HYPRE.GMRES or HYPRE.BiCGSTAB instead.
solver = HYPRE.PCG(; 
    MaxIter = 1000, 
    Tol = 1e-8,            # Converge when relative residual < 1e-8
    Precond = precond      # Attach the AMG preconditioner
)

# # 4. Solve
# # HYPRE.jl accepts SparseMatrixCSC and Vector directly.
println("Solving system with $(N) unknowns...")
@time "hypre_solver" x = HYPRE.solve(solver, A, b);

@time "julia_solver" x = A \ b;

# println("Solved!")

# Verification (optional)
res_norm = norm(b - A * x) / norm(b)
println("Relative Residual: $res_norm")