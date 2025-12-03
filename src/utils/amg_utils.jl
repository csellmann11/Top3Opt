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
struct Float32AMGWrapper{PType,BufType}
    real_preconditioner::PType  # This holds your Float32 AMG
    buf_x::BufType              # Pre-allocated Float32 buffer for input
    buf_b::BufType              # Pre-allocated Float32 buffer for output
end

# 2. Constructor
function Float32AMGWrapper(p::AMG.Preconditioner)
    # Get the size from the top level of the hierarchy
    n = size(p.ml.levels[1].A, 1)

    # Pre-allocate Float32 vectors ONCE.
    buf_x = Vector{Float32}(undef, n)
    buf_b = Vector{Float32}(undef, n)

    return Float32AMGWrapper(p, buf_x, buf_b)
end

# 3. The magic ldiv! function
#    This is what the Krylov solver calls.
#    y is Float64 (output), x is Float64 (input)
function LinearAlgebra.ldiv!(y::AbstractVector{Float64},
    wrapper::Float32AMGWrapper,
    x::AbstractVector{Float64})

    # A. Fast Copy & Cast: Float64 x -> Float32 buffer
    #    This uses the pre-allocated buffer, no new memory is asked from OS.
    wrapper.buf_b .= x

    # B. Call the inner AMG solve on Float32 data
    #    We use the definition you provided: ldiv!(x, p, b)
    #    Target: wrapper.buf_x (solution), Source: wrapper.buf_b (RHS)
    ldiv!(wrapper.buf_x, wrapper.real_preconditioner, wrapper.buf_b)

    # C. Fast Copy & Cast Back: Float32 buffer -> Float64 y
    y .= wrapper.buf_x

    return y
end

# 4. Helper for the '\' operator just in case
function Base.:\(wrapper::Float32AMGWrapper, x::AbstractVector{Float64})
    y = similar(x)
    ldiv!(y, wrapper, x)
    return y
end



function solve_lse_amg(k_global::SparseMatrixCSC,
    rhs_global::AbstractVector{Float64},
    cv::CellValues,
    ch::ConstraintHandler)

    # 1. Setup Nullspace
    fixed_dofs = collect(keys(ch.d_bcs))
    B = create_elasticity_nullspace_matrix(cv.mesh.topo,
        fixed_dofs; eltype=Float32)


    n_dofs = size(k_global, 2)

    function smooth_fun(A, T, ::Any, ::Any)

        n_reduces_dofs = size(T, 1)
        (n_dofs == n_reduces_dofs) && return T


        D_inv_S = AlgebraicMultigrid.weight(
            AlgebraicMultigrid.LocalWeighting(), A, 4.0 / 3.0)
        return (I - D_inv_S) * T
    end


    GC.gc()



    k_global_f32 = SparseMatrixCSC(
        k_global.m, k_global.n, k_global.colptr, k_global.rowval, Float32.(k_global.nzval)
    )

    # --- MEMORY OPTIMIZATION 2: Tune Parameters ---
    @timeit to "amg build" ml = smoothed_aggregation(k_global_f32,
        B=B,
        smooth=smooth_fun,
        improve_candidates=GaussSeidel(iter=4),
        strength=SymmetricStrength(0.0)
    )

    # Convert the MultiLevel object into a LinearOperator/Preconditioner
    P = aspreconditioner(ml) |> Float32AMGWrapper
    # 2. Setup Problem (Keep k_global as Float64 for the actual solve)
    prob = LinearProblem(Symmetric(k_global), rhs_global)

    # 3. Solve
    strategy = LinearSolve.KrylovJL_GMRES(gmres_restart = 50)
    @timeit to "krylov solve" sol = LinearSolve.solve(prob, strategy;
        Pl=P,
        abstol=1e-6, reltol=1e-6, itmax=500, verbose=true)

    @show sol.iters

    return sol.u
end