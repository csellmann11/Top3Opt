using PETSc, Libdl, MPI

# Helper to find the function symbol in the loaded PETSc library
function get_petsc_symbol(petsclib, sym::Symbol)
    return Libdl.dlsym(Libdl.dlopen(petsclib.petsc_library), sym)
end

# Wrapper for VecDuplicate (Missing in PETSc.jl)
function VecDuplicate(v::PETSc.AbstractVec{T}) where T
    # 1. Allocate a pointer to hold the new vector handle
    new_ptr = Ref{Ptr{Cvoid}}()
    
    # 2. Call PETSc C API
    sym = get_petsc_symbol(PETSc.petsclibs[1], :VecDuplicate)
    err = ccall(sym, PETSc.PetscErrorCode, (Ptr{Cvoid}, Ptr{Ptr{Cvoid}}), v.ptr, new_ptr)
    @assert err == 0 "VecDuplicate failed"
    
    # 3. Wrap the new pointer in a generic PETSc.Vec struct
    #    (Note: We use PETSc.Vec, not PETSc.VecSeq, because we don't have a Julia array backing it)
    new_vec = PETSc.Vec{T}(new_ptr[])
    
    # 4. Register finalizer for cleanup
    finalizer(PETSc.destroy, new_vec)
    return new_vec
end

# Wrapper for VecCopy (Much faster than broadcasting .=)
function VecCopy!(x::PETSc.AbstractVec, y::PETSc.AbstractVec)
    sym = get_petsc_symbol(PETSc.petsclibs[1], :VecCopy)
    err = ccall(sym, PETSc.PetscErrorCode, (Ptr{Cvoid}, Ptr{Cvoid}), x.ptr, y.ptr)
    @assert err == 0 "VecCopy failed"
end

# 1. Wrapper for MatSetBlockSize
function MatSetBlockSize!(mat::PETSc.AbstractMat, bs::Integer, petsclib=PETSc.petsclibs[1])
    sym = get_petsc_symbol(petsclib, :MatSetBlockSize)
    
    # Hardcoded Int64 for the integer argument 'bs'
    err = ccall(sym, PETSc.PetscErrorCode, 
        (Ptr{Cvoid}, Int64), 
        mat.ptr, Int64(bs)
    )
    @assert err == 0 "MatSetBlockSize failed with error $err"
end

# 2. Wrapper for MatSetNearNullSpace
function MatSetNearNullSpace!(mat::PETSc.AbstractMat, nullsp::PETSc.MatNullSpace, petsclib=PETSc.petsclibs[1])
    sym = get_petsc_symbol(petsclib, :MatSetNearNullSpace)
    
    # Arguments are just pointers (Mat, MatNullSpace)
    err = ccall(sym, PETSc.PetscErrorCode, 
        (Ptr{Cvoid}, Ptr{Cvoid}), 
        mat.ptr, nullsp.ptr
    )
    @assert err == 0 "MatSetNearNullSpace failed with error $err"
end

# 3. Custom Constructor for MatNullSpace (Accepting Vectors)
function MatNullSpaceCreate(comm::MPI.Comm, has_constant::Bool, vecs::AbstractVector, petsclib=PETSc.petsclibs[1])
    # Extract pointers from the PETSc vectors
    vec_ptrs = [v.ptr for v in vecs]
    
    # Allocate the wrapper struct (we need a place to put the resulting C pointer)
    # We use Float64 as the default scalar type for the struct parameter
    nullsp = PETSc.MatNullSpace{Float64}(C_NULL, comm)
    
    sym = get_petsc_symbol(petsclib, :MatNullSpaceCreate)
    
    # We need a Ref to hold the output pointer returned by PETSc
    out_ptr = Ref{Ptr{Cvoid}}(C_NULL)
    
    # Hardcoded Int64 for the integer argument 'n' (number of vectors)
    err = ccall(sym, PETSc.PetscErrorCode, 
        (MPI.MPI_Comm, PETSc.PetscBool, Int64, Ptr{Ptr{Cvoid}}, Ptr{Ptr{Cvoid}}),
        comm, 
        has_constant ? PETSc.PETSC_TRUE : PETSc.PETSC_FALSE, 
        Int64(length(vecs)), 
        vec_ptrs, 
        out_ptr
    )
    
    @assert err == 0 "MatNullSpaceCreate failed with error $err"
    
    # Assign the valid C pointer to our Julia struct
    nullsp.ptr = out_ptr[]
    
    # Register a finalizer so PETSc cleans up memory when Julia GCs this object
    finalizer(PETSc.destroy, nullsp)
    
    return nullsp
end

# Helper to destroy PETSc object and invalidates the pointer
# to prevent the GC from causing a Double Free later.
function safe_destroy!(obj)
    if isdefined(obj, :ptr) && obj.ptr != C_NULL
        PETSc.destroy(obj)
        obj.ptr = C_NULL # Defuse the finalizer
    end
end


function create_elasticity_nullspace(topo::Topology, fixed_dofs::Vector{Int}, petsclib=PETSc.petsclibs[1]) 
    
    active_nodes = filter(is_active,get_nodes(topo))

    n_active_nodes = length(active_nodes)
    n_dofs = 3 * n_active_nodes


    # 2. Get direct array views into the PETSc memory
    #    write=true, read=false tells PETSc we are overwriting, which can be faster
    rbm_arrays = [zeros(Float64, n_dofs) for _ in 1:6]

    # 3. Fill the arrays directly (Your Logic)
    #    Note: 'arrays' is a Vector of Julia Arrays that point to C memory.
    # for i in 1:n_active_nodes 
    for (i,node) in enumerate(active_nodes)
        dofx = 3*(i-1)+1
        dofy = 3*(i-1)+2
        dofz = 3*(i-1)+3

        # Assuming coords is compacted/ordered correctly for 1:n_active_nodes
        x = node[1]
        y = node[2]
        z = node[3]

        # Translation Modes
        rbm_arrays[1][dofx] = 1.0
        rbm_arrays[2][dofy] = 1.0
        rbm_arrays[3][dofz] = 1.0
        
        # Rotation Modes
        rbm_arrays[4][dofy] = -z;  rbm_arrays[4][dofz] =  y
        rbm_arrays[5][dofx] =  z;  rbm_arrays[5][dofz] = -x
        rbm_arrays[6][dofx] = -y;  rbm_arrays[6][dofy] =  x
    end

    # 4. Zero out fixed DOFs directly in PETSc memory
    if !isempty(fixed_dofs)
        for arr in rbm_arrays
            arr[fixed_dofs] .= 0.0
        end
    end

    # 5. CRITICAL: Restore the arrays
    #    This tells PETSc "I am done writing to your memory".
    #    If you skip this, PETSc will crash during Solve.
    petsc_vecs = [PETSc.VecSeq(v) for v in rbm_arrays]
    # for v in petsc_vecs
    #     PETSc.assemble(v)
    # end
    
    # 7. Create NullSpace
    nullsp = MatNullSpaceCreate(MPI.COMM_SELF, false, petsc_vecs, petsclib)
    
    return nullsp, rbm_arrays
end



function solve_lse_petsc(k_global::SparseMatrixCSC,
    rhs_global::AbstractVector{Float64},
    cv::CellValues{D,U},
    ch::ConstraintHandler{D,U}) where {D,U}
    
    println("starting petsc")
    flush(stdout)

    k_petsc = PETSc.MatSeqAIJ(k_global)
    b_petsc = PETSc.VecSeq(rhs_global)
    u       = zero(rhs_global)
    u_petsc = PETSc.VecSeq(u)

    
    println("created petsc quantities")
    flush(stdout)

    # 1. Set Block Size (Crucial for vector structure)
    MatSetBlockSize!(k_petsc, U)
    PETSc.MatSetOption(k_petsc, PETSc.MAT_SYMMETRIC, true)
    PETSc.MatSetOption(k_petsc, PETSc.MAT_SPD, true) # If it is positive definite

    fixed_dofs = collect(keys(ch.d_bcs))
    # 2. Attach Null Space (Crucial for AMG convergence)
    @timeit to "create_nullspace" nullspace, null_vecs = create_elasticity_nullspace(cv.mesh.topo, fixed_dofs)
    MatSetNearNullSpace!(k_petsc, nullspace)
    
    println("set up null spaces")
    flush(stdout)

    ksp = PETSc.KSP(k_petsc; 
        ksp_type="cg",
        ksp_rtol=1e-6,
        ksp_atol=1e-6,

        mat_block_size=3,
        # PC Settings
        pc_type="gamg",
        pc_gamg_type="agg",
        
        # PERFORMANCE FIXES:
        pc_gamg_square_graph=1,        
        pc_gamg_agg_nsmooths=1,        # Disable smoothed aggregation (use standard aggregation)
        pc_gamg_threshold=0.05,        # Slightly higher threshold to ignore weak links
        
        # Smoother settings
        mg_levels_ksp_type="chebyshev",
        mg_levels_pc_type="sor",    # Jacobi is faster than SOR/ASM
        mg_levels_ksp_max_it=2,        # Reduce from 4 to 2.
        
        # Use exact solver only on the very coarsest level
        #mg_coarse_pc_type="redundant",
        mg_coarse_redundant_pc_type="cholesky",
        ksp_monitor=true,
        ksp_view = false
    )
    # PETSc.setfromoptions!(ksp)
    
    PETSc.solve!(u_petsc, ksp, b_petsc)
    
    println("solve finished") 
    flush(stdout)  # <--- Essential for debugging crashes

    
    println("cleaning up...")
    flush(stdout)

    
    println("cleanup finished")
    flush(stdout)


    return u # Return result
end