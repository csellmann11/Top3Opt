using PETSc, Libdl, MPI

# Helper to find the function symbol in the loaded PETSc library
function get_petsc_symbol(petsclib, sym::Symbol)
    return Libdl.dlsym(Libdl.dlopen(petsclib.petsc_library), sym)
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
function MatNullSpaceCreate(comm::MPI.Comm, has_constant::Bool, vecs::Vector{<:PETSc.AbstractVec}, petsclib=PETSc.petsclibs[1])
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


# Modified function to zero out BCs
function create_elasticity_nullspace(coords::AbstractVector{<:StaticVector{3, T}}, fixed_dofs::Vector{Int}, petsclib=PETSc.petsclibs[1]) where T

    
    # 2. Extract coordinates and shift to centroid
    #    (Centering decouples rotations from translations, improving numerical stability)

    n_active_nodes = count(is_active,coords)  
    # x = FixedSizeVector{Float64}(undef,n_active_nodes) 
    # y = FixedSizeVector{Float64}(undef,n_active_nodes) 
    # z = FixedSizeVector{Float64}(undef,n_active_nodes)  
    # counter = 1 
    # for c in coords
    #     if is_active(c)
    #         x[counter] = c[1]
    #         y[counter] = c[2]
    #         z[counter] = c[3]
    #         counter += 1
    #     end
    # end
    
    # cx, cy, cz = sum(x)/n_active_nodes, sum(y)/n_active_nodes, sum(z)/n_active_nodes
    # x .-= cx
    # y .-= cy
    # z .-= cz

    # N = length(coords)
    dofs = 3 * n_active_nodes

    # 1. Prepare vectors for Rigid Body Modes (RBM)
    #    We create 6 vectors of length 'dofs' initialized to 0.0
    rbm = [zeros(Float64, dofs) for _ in 1:6]

    for i in 1:n_active_nodes 

        dofx = 3*(i-1)+1
        dofy = 3*(i-1)+2
        dofz = 3*(i-1)+3

        x    = coords[i][1]
        y    = coords[i][2]
        z    = coords[i][3]

        rbm[1][dofx] = 1.0
        rbm[2][dofy] = 1.0
        rbm[3][dofz] = 1.0
        
        rbm[4][dofx] = -z
        rbm[4][dofy] =  y
        rbm[5][dofx] =  z
        rbm[5][dofz] = -x
        rbm[6][dofx] = -y
        rbm[6][dofy] =  x
    end

    
   
    
    # 3. Fill Translations (x, y, z)
    # #    Pattern: 1 0 0 | 1 0 0 | ...
    # rbm[1][1:3:end] .= 1.0  # Trans X
    # rbm[2][2:3:end] .= 1.0  # Trans Y
    # rbm[3][3:3:end] .= 1.0  # Trans Z
    
    # # 4. Fill Rotations (small angle approximation)
    # #    Rot X (about x-axis): 0, -z,  y
    # rbm[4][2:3:end] .= -z
    # rbm[4][3:3:end] .=  y
    
    # #    Rot Y (about y-axis): z,  0, -x
    # rbm[5][1:3:end] .=  z
    # rbm[5][3:3:end] .= -x
    
    # #    Rot Z (about z-axis): -y, x,  0
    # rbm[6][1:3:end] .= -y
    # rbm[6][2:3:end] .=  x
    
    # 5. CRITICAL: Zero out rows corresponding to Dirichlet BCs
    #    If a DOF is fixed in the system matrix, it must NOT move in the null space.
    if !isempty(fixed_dofs)
        for v in rbm
            v[fixed_dofs] .= 0.0
        end
    end
    
    # 6. Convert to PETSc Vectors and Assemble
    petsc_vecs = [PETSc.VecSeq(v) for v in rbm]
    for v in petsc_vecs
        PETSc.assemble(v)
    end
    
    # 7. Create the NullSpace Object
    #    has_constant=false because we are explicitly providing the translational modes
    nullsp = MatNullSpaceCreate(MPI.COMM_SELF, false, petsc_vecs, petsclib)
    
    return nullsp, petsc_vecs # return petsc_vecs for gc 
end



function solve_lse_petsc(k_global::SparseMatrixCSC,
    rhs_global::AbstractVector{Float64},
    cv::CellValues{D,U},
    ch::ConstraintHandler{D,U}) where {D,U}

    k_petsc = PETSc.MatSeqAIJ(k_global)
    b_petsc = PETSc.VecSeq(rhs_global)
    u_petsc = PETSc.VecSeq(zero(rhs_global))


    # 1. Set Block Size (Crucial for vector structure)
    MatSetBlockSize!(k_petsc, U)
    PETSc.MatSetOption(k_petsc, PETSc.MAT_SYMMETRIC, true)
    PETSc.MatSetOption(k_petsc, PETSc.MAT_SPD, true) # If it is positive definite

    fixed_dofs = collect(keys(ch.d_bcs))
    # 2. Attach Null Space (Crucial for AMG convergence)
    nullspace, null_vecs = create_elasticity_nullspace(cv.mesh.nodes, fixed_dofs)
    MatSetNearNullSpace!(k_petsc, nullspace)

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
        mg_levels_ksp_max_it=4,        # Reduce from 4 to 2.
        
        # Use exact solver only on the very coarsest level
        mg_coarse_pc_type="redundant",
        mg_coarse_redundant_pc_type="cholesky",
        ksp_monitor=true,
        ksp_view = false
    )
    # PETSc.setfromoptions!(ksp)
    PETSc.solve!(u_petsc, ksp, b_petsc)
    # Extract solution to Julia Array
    result = collect(u_petsc)

    # --- MANUAL CLEANUP (Crucial for loops) ---
    # Destroy in reverse order of dependency
    PETSc.destroy(ksp)
    
    # NullSpace does NOT destroy its vectors automatically in PETSc
    PETSc.destroy(nullspace) 
    for v in null_vecs
        PETSc.destroy(v) # Destroy the vectors we kept alive
    end
    
    PETSc.destroy(u_petsc)
    PETSc.destroy(b_petsc)
    PETSc.destroy(k_petsc)

    return result # Return result
end