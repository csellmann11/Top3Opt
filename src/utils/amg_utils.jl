using AlgebraicMultigrid
using LinearAlgebra
using SparseArrays
import LinearSolve
using LinearSolve: LinearProblem, KrylovJL_CG
function create_elasticity_nullspace_matrix(topo::Topology, fixed_dofs::Vector{Int})
    active_nodes   = filter(is_active, get_nodes(topo))
    n_active_nodes = length(active_nodes)
    n_dofs = 3 * n_active_nodes

    # Initialize the Null Space Matrix (N_dofs x 6 modes)
    # Mode 1-3: Translations, Mode 4-6: Rotations
    B = zeros(Float64, n_dofs, 6)

    for (i, node) in enumerate(active_nodes)
        dofx = 3 * (i - 1) + 1
        dofy = 3 * (i - 1) + 2
        dofz = 3 * (i - 1) + 3

        x, y, z = node
        # --- Translation Modes ---
        B[dofx, 1] = 1.0
        B[dofy, 2] = 1.0
        B[dofz, 3] = 1.0
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
        B[fixed_dofs, :] .= 0.0
    end

    return B
end


using SymRCM

function solve_lse_amg(k_global::SparseMatrixCSC,
    rhs_global::AbstractVector{Float64},
    cv::CellValues,
    ch::ConstraintHandler)

    # 1. Setup Nullspace
    fixed_dofs = collect(keys(ch.d_bcs))
    B = create_elasticity_nullspace_matrix(cv.mesh.topo, fixed_dofs)


    n_dofs = size(k_global, 2)

    function smooth_fun(A, T, S, B, weighting=AlgebraicMultigrid.LocalWeighting())

        n_reduces_dofs = size(T, 1)
        (n_dofs == n_reduces_dofs || n_reduces_dofs >= 1_000_000) && return T

        D_inv_S = AlgebraicMultigrid.weight(weighting, A, 4.0 / 3.0)

        (I - D_inv_S) * T
    end


    GC.gc()

    perm = symrcm(k_global) 
    inv_perm = similar(perm); inv_perm[perm] = 1:length(perm)
    k_global_perm = k_global[perm,perm]
    rhs_global_perm = rhs_global[perm]
    B_perm = B[perm,:]

    # --- MEMORY OPTIMIZATION 2: Tune Parameters ---
    @timeit to "amg build" ml = smoothed_aggregation(k_global_perm,
        B=B_perm,
        smooth=(A, T, S, B) -> smooth_fun(A, T, S, B),
        improve_candidates = GaussSeidel(iter=8) 
        )

    # Convert the MultiLevel object into a LinearOperator/Preconditioner
    P = aspreconditioner(ml)
    # 2. Setup Problem (Keep k_global as Float64 for the actual solve)
    prob = LinearProblem(k_global_perm, rhs_global_perm)

    # 3. Solve
    strategy = KrylovJL_CG()
    @timeit to "krylov solve" sol = LinearSolve.solve(prob, strategy;
        Pl=P,
        abstol=1e-6, reltol=1e-6, itmax=500, verbose=true)

    @show sol.iters

    return sol.u[inv_perm]
end