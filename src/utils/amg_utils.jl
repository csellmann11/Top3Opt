using AlgebraicMultigrid
using LinearAlgebra
using SparseArrays
import LinearSolve
using LinearSolve: LinearProblem, SmoothedAggregationPreconBuilder, KrylovJL_CG

function create_elasticity_nullspace_matrix(topo::Topology, fixed_dofs::Vector{Int})
    active_nodes = filter(is_active, get_nodes(topo))
    n_active_nodes = length(active_nodes)
    n_dofs = 3 * n_active_nodes

    # Initialize the Null Space Matrix (N_dofs x 6 modes)
    # Mode 1-3: Translations, Mode 4-6: Rotations
    B = zeros(Float64, n_dofs, 6)

    for (i, node) in enumerate(active_nodes)
        dofx = 3*(i-1)+1
        dofy = 3*(i-1)+2
        dofz = 3*(i-1)+3

        x, y, z = node[1], node[2], node[3]

        # --- Translation Modes ---
        B[dofx, 1] = 1.0
        B[dofy, 2] = 1.0
        B[dofz, 3] = 1.0
        
        # --- Rotation Modes ---
        # Rotation around X (y -> z, z -> -y)
        B[dofy, 4] = -z
        B[dofz, 4] =  y
        
        # Rotation around Y (z -> x, x -> -z)
        B[dofx, 5] =  z
        B[dofz, 5] = -x
        
        # Rotation around Z (x -> y, y -> -x)
        B[dofx, 6] = -y
        B[dofy, 6] =  x
    end

    # Zero out fixed DOFs (Dirichlet BCs)
    # This is important so the AMG doesn't try to interpolate values into fixed boundaries
    if !isempty(fixed_dofs)
        B[fixed_dofs, :] .= 0.0
    end

    return B
end

function solve_lse_amg(k_global::SparseMatrixCSC,
                        rhs_global::AbstractVector{Float64},
                        cv::CellValues,
                        ch::ConstraintHandler)

    # 1. Setup Nullspace (Crucial for 5M DOFs Elasticity)
    fixed_dofs = collect(keys(ch.d_bcs))
    @timeit to "build nullspace" B = create_elasticity_nullspace_matrix(cv.mesh.topo, fixed_dofs)


    prob = LinearProblem(k_global, rhs_global)
    amg_builder = SmoothedAggregationPreconBuilder(
        B = B,
        strength = SymmetricStrength(0.0),
        smooth = JacobiProlongation(4.0/3.0),
        presmoother = GaussSeidel(SymmetricSweep(),1),
        postsmoother = GaussSeidel(SymmetricSweep(),1),
        max_levels = 20
    )
    strategy = KrylovJL_CG(precs = amg_builder)
    @timeit to "krylov solve" sol = LinearSolve.solve(prob, strategy; 
             abstol = 1e-6, reltol = 1e-6, itmax = 500, verbose = true)

    u = sol.u


    return u
end