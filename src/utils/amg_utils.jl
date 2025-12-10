using AlgebraicMultigrid
using LinearAlgebra
using SparseArrays
import LinearSolve
using LinearSolve: LinearProblem, KrylovJL_CG
function create_elasticity_nullspace_matrix(topo::Topology,
    fixed_dofs::Vector{Int}; eltype::Type{T}=Float64) where T<:Real

    active_nodes = filter(is_active, get_nodes(topo))
    n_active_nodes = length(active_nodes)
    n_dofs = 3 * n_active_nodes

    # Initialize the Null Space Matrix (N_dofs x 6 modes)
    # Mode 1-3: Translations, Mode 4-6: Rotations
    B = zeros(eltype, n_dofs, 6)

    for (i, node) in enumerate(active_nodes)
        dofx = 3 * (i - 1) + 1
        dofy = 3 * (i - 1) + 2
        dofz = 3 * (i - 1) + 3

        x, y, z = node
        # --- Translation Modes ---
        B[dofx, 1] = one(eltype)
        B[dofy, 2] = one(eltype)
        B[dofz, 3] = one(eltype)
        # --- Rotation Modes ---
        # Rotation around X (y -> z, z -> -y)
        B[dofy, 4] = -z
        B[dofz, 4] = y

        # Rotation around Y (z -> x, x -> -z)
        B[dofx, 5] = z
        B[dofz, 5] = -x

        # Rotation around Z (x -> y, y -> -x)
        B[dofx, 6] = -y
        B[dofy, 6] = x
    end

    # Zero out fixed DOFs (Dirichlet BCs)
    # This is important so the AMG doesn't try to interpolate values into fixed boundaries
    if !isempty(fixed_dofs)
        B[fixed_dofs, :] .= zero(eltype)
    end

    return B
end


import AlgebraicMultigrid as AMG
# 1. Define the Wrapper Struct



function solve_lse_amg(k_global::SparseMatrixCSC,
    rhs_global::AbstractVector{Float64},
    cv::CellValues,
    ch::ConstraintHandler)

    # 1. Setup Nullspace
    fixed_dofs = collect(keys(ch.d_bcs))
    B = create_elasticity_nullspace_matrix(cv.mesh.topo,
        fixed_dofs; eltype=Float64)


    n_dofs = size(k_global, 2)

    function smooth_fun(A, T, ::Any, ::Any)

        n_reduces_dofs = size(T, 1)
        (n_dofs == n_reduces_dofs) && return T


        D_inv_S = AlgebraicMultigrid.weight(
            AlgebraicMultigrid.LocalWeighting(), A, 4.0 / 3.0)
        return (I - D_inv_S) * T
    end


    GC.gc()
    # --- MEMORY OPTIMIZATION 2: Tune Parameters ---
    @timeit to "amg build" ml = smoothed_aggregation(k_global,
        B=B,
        smooth=smooth_fun,
        improve_candidates=GaussSeidel(iter=4),
        strength=SymmetricStrength(0.0)
    )

    # 2. Setup Problem (Keep k_global as Float64 for the actual solve)
    prob = LinearProblem(Symmetric(k_global), rhs_global)

    reltol = 1e-06
    strategy = LinearSolve.KrylovJL_GMRES(gmres_restart=50)
    # strategy = LinearSolve.KrylovJL_CG()
    @timeit to "krylov solve" sol = LinearSolve.solve(prob, strategy;
        Pl=aspreconditioner(ml),
        reltol=reltol, itmax=500, verbose=true)

    @show sol.iters

    return sol.u
end


function solve_lse_hypre(k_global::SparseMatrixCSC,
    rhs_global::AbstractVector{Float64})


    precond = HYPRE.BoomerAMG(;
        NumFunctions=3,      # 3 degrees of freedom per node (u,v,w)
        CoarsenType=10,      # HMIS coarsening (often faster for larger 3D problems)
        RelaxType=6,         # Symmetric smoothing for PCG
        NumSweeps=1,         # 1 sweep of smoothing
        MaxIter=1,           # Perform 1 V-cycle per preconditioner application
        Tol=0.0              # Do not stop on tolerance inside preconditioner
    )

    # 3. Configure the Solver (PCG)
    # We use Conjugate Gradient (PCG) as the outer solver.
    # If your matrix is not Symmetric Positive Definite, use HYPRE.GMRES or HYPRE.BiCGSTAB instead.
    solver = HYPRE.PCG(;
        MaxIter=1000,
        Tol=1e-6,            
        PrintLevel=0,
        Precond=precond      # Attach the AMG preconditioner
    )

    @timeit to "hypre_solver" u = HYPRE.solve(solver, k_global, rhs_global);

    return u

end