using AlgebraicMultigrid
using LinearAlgebra
using SparseArrays
import LinearSolve
using LinearSolve: LinearProblem, KrylovJL_CG
function create_elasticity_nullspace_matrix(topo::Topology, fixed_dofs::Vector{Int})
    active_nodes = filter(is_active, get_nodes(topo))
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

# function solve_lse_amg(k_global::SparseMatrixCSC,
#                         rhs_global::AbstractVector{Float64},
#                         cv::CellValues,
#                         ch::ConstraintHandler)

#     # 1. Setup Nullspace 
#     fixed_dofs = collect(keys(ch.d_bcs))
#     @timeit to "build nullspace" B = create_elasticity_nullspace_matrix(cv.mesh.topo, fixed_dofs)


#     prob = LinearProblem(k_global, rhs_global)
#     amg_builder = SmoothedAggregationPreconBuilder(
#         B = B,
#         strength = SymmetricStrength(0.00),
#         smooth = JacobiProlongation(4.0/3.0),
#         presmoother = GaussSeidel(SymmetricSweep(),1),
#         postsmoother = GaussSeidel(SymmetricSweep(),1),
#         max_levels = 20
#     )
#     strategy = KrylovJL_CG(precs = amg_builder)
#     @timeit to "krylov solve" sol = LinearSolve.solve(prob, strategy; 
#              abstol = 1e-6, reltol = 1e-6, itmax = 500, verbose = true)

#     u = sol.u


#     return u
# end

struct AMGCoarseSolver
    lu_dec::SparseArrays.UMFPACK.UmfpackLU{Float64, Int64}
    function AMGCoarseSolver(A::SparseMatrixCSC{Float64, Int64}) 
        return new(lu(A))
    end
end

(p::AMGCoarseSolver)(x,b) = LinearAlgebra.ldiv!(x,p.lu_dec,b)

function truncated_smoother(threshold::Float64=0.15; damping::Float64=4.0/3.0)
    # Returns a lambda that accepts the 4 arguments required by AMG
    return (A, T, S, B) -> begin
        # 1. Apply Standard Smoothing (Captures Physics)
        #    This creates a high-quality but DENSE prolongator P
        P = AlgebraicMultigrid.JacobiProlongation(damping)(A, T, S, B)
        
        # 2. Aggressive Truncation (Saves Memory)
        #    Removes small entries (e.g., < 5% of max) that cause fill-in
        #    without contributing significantly to convergence.
        droptol!(P, threshold)
        
        return P
    end
end


function hybrid_smoother(; damping::Float64=4.0/3.0, filter_strength::Float64=0.02)
    return (A, T, S, B) -> begin
        # 1. Create a Filtered Strength Matrix just for smoothing
        #    We keep S (full) for aggregation, but use S_filt for smoothing.
        S_filt = copy(S)
        
        # This removes weak connections from the smoothing graph.
        # It prevents P from spreading to weakly coupled neighbors.
        droptol!(S_filt, filter_strength)
        
        # 2. Compute P using the SPARSER S_filt
        P = AlgebraicMultigrid.JacobiProlongation(damping)(A, T, S_filt, B)
        
        # 3. Final cleanup (optional, keeps P extremely tidy)
        droptol!(P, 0.01) 
        
        return P
    end
end

function solve_lse_amg(k_global::SparseMatrixCSC,
    rhs_global::AbstractVector{Float64},
    cv::CellValues,
    ch::ConstraintHandler)

    # 1. Setup Nullspace
    fixed_dofs = collect(keys(ch.d_bcs))
    @timeit to "build nullspace" B = create_elasticity_nullspace_matrix(cv.mesh.topo, fixed_dofs)

    # --- MEMORY OPTIMIZATION 2: Tune Parameters ---
    # We call smoothed_aggregation directly on the Float32 matrix.
    # strength=0.02 is CRITICAL. It drops weak connections (sparsity).
    # 0.0 creates fully dense coarse grids -> Out of Memory.
    @timeit to "amg build" ml = smoothed_aggregation(k_global,
        B=B,
        strength=SymmetricStrength(0.0),
        smooth=(A, T, S, B) -> T,#JacobiProlongation(4.0/3.0),
        presmoother=GaussSeidel(SymmetricSweep(), 1),
        postsmoother=GaussSeidel(SymmetricSweep(), 1),
        max_levels=25, max_coarse=25,
        diagonal_dominance = true,
        coarse_solver=AMGCoarseSolver
    )

    # Convert the MultiLevel object into a LinearOperator/Preconditioner
    P = aspreconditioner(ml)

    # 2. Setup Problem (Keep k_global as Float64 for the actual solve)
    prob = LinearProblem(k_global, rhs_global)

    # 3. Solve
    # We pass 'P' explicitly to Pl (Left Preconditioner). 
    # LinearSolve will now use the Float32 AMG to precondition the Float64 CG.
    # P = Diagonal(diag(k_global))
    strategy = KrylovJL_CG()
    @timeit to "krylov solve" sol = LinearSolve.solve(prob, strategy;
        Pl=P,
        abstol=1e-6, reltol=1e-6, itmax=500, verbose=true)

    @show sol.iters

    return sol.u
end