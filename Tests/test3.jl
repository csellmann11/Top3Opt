
ENV["OMP_NUM_THREADS"] = Threads.nthreads()
using LinearAlgebra, SparseArrays, Pardiso

println("Number of threads: $(Threads.nthreads())")
println("Number of BLAS threads: $(BLAS.get_num_threads())")

# --- 1. Helper function to create 3D Poisson matrix ---
# This creates the sparse matrix for a 3D Poisson problem
# on an n x n x n grid with Dirichlet boundary conditions.
# The resulting matrix is (n^3) x (n^3) and is symmetric positive definite.
function create_poisson_3d(n)
    N = n * n * n
    # We use spzeros and fill it, which is efficient for CSC format
    #  A = spzeros(Float64, PETSc.inttype(PETSc.petsclibs[1]), N, N) # Use PETSc's Int type
    cols = Int[]
    rows = Int32[]
    vals = Float64[]

    
    # Mapping from 3D index (i, j, k) to 1D index (idx)
    idx(i, j, k) = (k - 1) * n * n + (j - 1) * n + i

    counter = 1
    for k in 1:n, j in 1:n, i in 1:n
        current_idx = idx(i, j, k)
        push!(cols, current_idx)
        push!(rows, current_idx)
        push!(vals, 6.0)

        if i > 1 
            push!(cols, current_idx - 1)
            push!(rows, current_idx)
            push!(vals, -1.0)
        end
        if i < n 
            push!(cols, current_idx + 1)
            push!(rows, current_idx)
            push!(vals, -1.0)
        end
        if j > 1 
            push!(cols, current_idx - n)
            push!(rows, current_idx)
            push!(vals, -1.0)
        end
        if j < n 
            push!(cols, current_idx + n)
            push!(rows, current_idx)
            push!(vals, -1.0)
        end
        if k > 1 
            push!(cols, current_idx - n * n)
            push!(rows, current_idx)
            push!(vals, -1.0)
        end
        if k < n 
            push!(cols, current_idx + n * n)
            push!(rows, current_idx)
            push!(vals, -1.0)
        end

    end
    A = sparse(rows, cols, vals, N, N)
    return A
end

# --- 2. Main comparison function ---
function run_comparison(n)
    @time "created A" A = create_poisson_3d(n)
    b = rand(Float64, n^3)


    
    ps = MKLPardisoSolver()


    set_matrixtype!(ps, 1)

    set_nprocs!(ps, 8) # Sets the number of threads to use
    @show get_nprocs(ps) # Gets the number of threads being used
    xs = zero(b)
    
    trilA = tril(A)
    @time "Pardiso solve" Pardiso.solve!(ps,xs,A,b)
    # @time "Julia solve" x = cholesky(A)\b
    
    # res_chol  = (A*x - b) |> norm
    res_pardiso = (A*xs - b) |> norm
   
    println("Residuals:")
    # println("Cholesky: $res_chol")
    println("Pardiso: $res_pardiso")
    nothing
end

run_comparison(20)