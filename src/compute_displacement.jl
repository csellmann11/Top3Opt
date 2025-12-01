using IterativeSolvers, AlgebraicMultigrid
using Ju3VEM.VEMUtils.Octavian: matmul!
# import Libdl
# function get_symbol(petsclib, func_name::Symbol)
#     handle = Libdl.dlopen(petsclib.petsc_library)
#     return Libdl.dlsym(handle, func_name)
# end

# function get_petsc_types(petsclib)
#     return PETSc.scalartype(petsclib), PETSc.inttype(petsclib)
# end

# # --- The Missing Function Implementation ---
# function MatSetBlockSize(mat, bs, petsclib)
#     # 1. Get the correct Integer type (Int32 or Int64) for this PETSc build
#     # _, PetscInt = get_petsc_types(petsclib)
    
#     # 2. Look up the C function symbol dynamically
#     sym = get_symbol(petsclib, :MatSetBlockSize)
    
#     # 3. Call it
#     # Signature: PetscErrorCode MatSetBlockSize(Mat mat, PetscInt bs)
#     err = ccall(sym, 
#                 PETSc.PetscErrorCode,      # Return type
#                 (Ptr{Cvoid}, Int64),    # Argument types
#                 mat.ptr, bs)               # Arguments
                
#     @assert err == 0 "MatSetBlockSize failed with error code $err"
# end

# --- 1. Wrapper for MatSetNearNullSpace (Missing in PETSc.jl) ---
# --- 1. Wrapper for MatSetNearNullSpace ---
# function MatSetNearNullSpace(mat, nullsp, petsclib)
#     sym = get_symbol(petsclib, :MatSetNearNullSpace)
#     # Note: PetscErrorCode is a constant, so PETSc.PetscErrorCode is fine.
#     err = ccall(sym, PETSc.PetscErrorCode, (Ptr{Cvoid}, Ptr{Cvoid}), mat.ptr, nullsp)
#     @assert err == 0 "MatSetNearNullSpace failed"
# end

#### INFO: 2nd try 

struct ElData{D}
    el_id ::Int 
    proj_s::FixedSizeMatrixDefault{Float64}
    proj  ::FixedSizeMatrixDefault{Float64}
    node_ids ::FixedSizeVectorDefault{Int}
    
    γ_stab ::Float64
    hvol   ::Float64
    volume ::Float64
    bc     ::SVector{D,Float64}
end

function collect_nodes_by_order(map::Dict{I,<:Integer}) where I <: Integer
    node_ids = FixedSizeVector{I}(undef,length(map))
    for (node_id,i) in map
        node_ids[i] = node_id
    end
    return node_ids
end


function build_k_poly_space(
    mat_law::H,
    pars::Tuple{Float64,Float64,Float64}
) where H<:Helmholtz

    base3d = get_base(BaseInfo{3,1,3}())

    ℂ0 = eval_hessian(mat_law,zero(SMatrix{3,3,Float64,9}),pars)
    ∇x_base = SVector{length(base3d)}(∇(m,1.0)(SA[0.0,0.0,0.0]) for m in base3d)
    k_poly_space = MMatrix{length(base3d),length(base3d),Float64} |> zero
    for (i,∇mx) in enumerate(∇x_base)
        ∇mxℂ0 = ℂ0 ⊡₂ ∇mx
        for (j,∇nx) in enumerate(∇x_base)
            k_poly_space[i,j] += ∇mxℂ0 ⋅ ∇nx
        end
    end
    return SMatrix(k_poly_space)
end


function project_matrix!(
    dest::AbstractMatrix{Float64},
    g::AbstractMatrix{Float64},
    proj::AbstractMatrix{Float64}
    )

    @no_escape begin
        destretch_cache = @alloc(Float64,size(proj,1),size(proj,2))
        mul_cache       = @alloc(Float64,size(g,1),size(proj,2))
        Ju3VEM.VEMGeo.destretch!(destretch_cache,stretch(proj,Val(U)))
        matmul!(mul_cache,g,destretch_cache)
        matmul!(dest,destretch_cache',mul_cache)
    end
    dest
end

function build_local_kel_and_f_topo!(
    k_poly_space::SMatrix{L,L,Float64},
    kelement::CachedMatrix{Float64},
    rhs_element::CachedVector{Float64},
    cv::CellValues{D,U,ET},
    element_id::Int,
    cache1::CachedMatrix{Float64},
    cache2::CachedMatrix{Float64},
    γ::Float64,
    χ::Float64) where {L,D,U,ET<:ElType{1}}

    hvol = cv.volume_data.hvol
    dΩ   = cv.volume_data.integrals[1]

    proj_s, proj = create_volume_vem_projectors(
        element_id,cv.mesh,cv.volume_data,cv.facedata_col,cv.vnm)

    n_nodes = size(proj,1)
    n_dofs = n_nodes*U

    setsize!(kelement,(n_dofs,n_dofs))
    setsize!(rhs_element,(n_dofs,))


    # project_matrix!(kelement.array,k_poly_space,stretch(proj_s,Val(U)))
    begin # computes proj_s' * k_poly_space * proj_s
        # full_proj_s = FixedSizeMatrix{Float64}(undef,L,n_dofs)
        setsize!(cache2,(L,n_dofs))
        Ju3VEM.VEMGeo.destretch!(cache2.array,stretch(proj_s,Val(U)))
        setsize!(cache1,(L,n_dofs)) 
        matmul!(cache1.array,k_poly_space,cache2.array)
        matmul!(kelement.array,cache2.array',cache1.array)

        # kelement.array .= full_proj_s' * (k_poly_space .* dΩ/(hvol^2)) * full_proj_s
    end
    kelement .*= dΩ/(hvol^2) * χ^3

    setsize!(cache1,(n_nodes,n_nodes))
    setsize!(cache2,(n_nodes,n_nodes))


    cache1 .= I - proj
    matmul!(cache2.array,cache1.array',cache1.array)
    cache2.array .*= hvol*γ
    # cache2.array .= (I(n_nodes) .- proj)' * (I(n_nodes) .- proj) * hvol*γ
    kelement .+= stretch(cache2.array,Val(U))

    return proj_s, proj
end




function assembly(cv::CellValues{D,U,ET},
    states::DesignVarInfo{D},
    f::F,
    sim_pars::SimPars) where {D,U,F<:Function,K,ET<:ElType{K}}

    mat_law = sim_pars.mat_law
    @timeit to "set up assembler" k_global = get_sparsity_pattern(cv)
    # ass = FR.start_assemble(Symmetric(k_global),zeros(size(k_global,1))) #!
    ass = FR.start_assemble(k_global,zeros(size(k_global,1)))



    Is = SMatrix{U,U,Float64}(I)
    γ = Is ⊡₂ eval_hessian(mat_law,Is,(sim_pars.λ,sim_pars.μ,1.0)) ⊡₂ Is
    rhs_element = DefCachedVector{Float64}()
    kelement    = DefCachedMatrix{Float64}()
    cache1      = DefCachedMatrix{Float64}()
    cache2      = DefCachedMatrix{Float64}()

    eldata_col  = Dict{Int,ElData{D}}()

    e2s = states.el_id_to_state_id

    k_poly_space = build_k_poly_space(mat_law,(sim_pars.λ,sim_pars.μ,1.0))


    for element in RootIterator{4}(cv.mesh.topo)
        χ = states.χ_vec[e2s[element.id]]
        @timeit to "reinit" reinit!(element.id,cv)

        γ_stab = γ/4 * χ^3

        @timeit to "build_local_kel_and_f" proj_s, proj = build_local_kel_and_f_topo!(
            k_poly_space,kelement, 
            rhs_element,
            cv,element.id,cache1,cache2,γ_stab,χ)

    
        node_ids = FixedSizeVector{Int}(undef,length(cv.vnm.map))
        copyto!(node_ids,cv.vnm.map.keys)

        dofs = FixedSizeVector{Int}(undef,length(node_ids)*U)
        get_dofs!(dofs,cv.dh,node_ids)
  
        vol_data = cv.volume_data
        bc_vol   = vol_data.vol_bc
        hvol     = vol_data.hvol 
        volume   = vol_data.integrals[1]
        eldata_col[element.id] = ElData(
            element.id,proj_s,proj,
            node_ids,
            γ_stab,hvol,volume,bc_vol)
  

        @timeit to "local_assembly" FR.assemble!(ass,dofs,kelement.array,rhs_element.array)
    end

    kglobal, rhsglobal = FR.finish_assemble(ass)

    kglobal, rhsglobal, eldata_col
end


function compute_displacement(cv::CellValues{D,U,ET},
    ch::ConstraintHandler,
    states::DesignVarInfo{D},
    f::F,
    sim_pars::SimPars) where {D,U,F<:Function,K,ET<:ElType{K}}

    @timeit to "assembly" k_global,rhs_global, eldata_col = assembly(cv,states,f,sim_pars)

    @timeit to "apply" apply!(k_global,rhs_global,ch)
  
    n = size(k_global, 1)
    @timeit to "solver" u = begin 
        # u_chol = cholesky(Symmetric(k_global)) \ rhs_global
        # u = zero(rhs_global)
        # Pardiso.pardiso(ps, u,tril(k_global), rhs_global)
        # u

        u = solve_lse_petsc(k_global,rhs_global,cv,ch)
        u
    end



    return u, k_global, eldata_col
end


