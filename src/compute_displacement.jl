struct ElementData{D}
    el_id ::Int 
    proj_s::FixedSizeMatrixDefault{Float64}
    proj  ::FixedSizeMatrixDefault{Float64}
    kel   ::Matrix{Float64}
    node_ids ::FixedSizeVectorDefault{Int}
    dofs  ::FixedSizeVectorDefault{Int}
    
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
    return k_poly_space
end


function build_local_kel_and_f_topo!(
    k_poly_space::MMatrix{L,L,Float64},
    kelement::CachedMatrix{Float64},
    rhs_element::CachedVector{Float64},
    cv::CellValues{D,U,ET},
    element_id::Int,
    γ::Float64 = 1/4) where {L,D,U,ET<:ElType{1}}

    hvol = cv.volume_data.hvol
    dΩ   = cv.volume_data.integrals[1]

    proj_s, proj = create_volume_vem_projectors(
        element_id,cv.mesh,cv.volume_data,cv.facedata_col,cv.vnm)

    n_dofs = size(proj,1)*U

    setsize!(kelement,(n_dofs,n_dofs))
    setsize!(rhs_element,(n_dofs,))


    # computes proj_s' * k_poly_space * proj_s
    _temp = k_poly_space * stretch(proj_s,Val(U)) .* dΩ/(hvol^2)
    mul!(kelement,stretch(proj_s',Val(U)),_temp)


    _Δmat = (I-proj)
    stab       = _Δmat'*_Δmat*hvol*γ
    kelement .+= stretch(stab,Val(U))

    return proj_s, proj
end




function assembly(cv::CellValues{D,U,ET},
    states::TopStates{D},
    f::F,
    sim_pars::SimParameter) where {D,U,F<:Function,K,ET<:ElType{K}}

    mat_law = sim_pars.mat_law
    ass = Assembler{Float64}(cv)


    Is = SMatrix{U,U,Float64}(I)
    γ = Is ⊡₂ eval_hessian(mat_law,Is,(sim_pars.λ,sim_pars.μ,1.0)) ⊡₂ Is
    rhs_element = DefCachedVector{Float64}()
    kelement    = DefCachedMatrix{Float64}()

    eldata_col  = Dict{Int,ElementData{D}}()

    e2s = states.el_id_to_state_id

    k_poly_space = build_k_poly_space(mat_law,(sim_pars.λ,sim_pars.μ,1.0))


    for element in RootIterator{4}(cv.mesh.topo)
        χ = states.χ_vec[e2s[element.id]]
        reinit!(element.id,cv)

        γ_stab = γ/4 #* χ^3

        proj_s, proj = build_local_kel_and_f_topo!(
            k_poly_space,kelement,
            rhs_element,
            cv,element.id,γ_stab)

        
        node_ids = collect_nodes_by_order(cv.vnm.map)
        cell_dofs = FixedSizeVector{Int}(undef,length(node_ids)*U)
        get_dofs!(cell_dofs,cv.dh,node_ids)

  
        vol_data = cv.volume_data
        bc_vol   = vol_data.vol_bc
        hvol     = vol_data.hvol 
        volume   = vol_data.integrals[1]
        eldata_col[element.id] = ElementData(
            element.id,proj_s,proj,
            kelement.array[:,:],node_ids,cell_dofs,
            γ_stab*χ^3,hvol,volume,bc_vol)
  
        kelement .*= χ^3
        local_assembly!(ass,kelement,rhs_element)
    end

    kglobal, rhsglobal = assemble!(ass)


    kglobal, rhsglobal, eldata_col
end


function compute_displacement(cv::CellValues{D,U,ET},
    ch::ConstraintHandler,
    states::TopStates{D},
    f::F,
    sim_pars::SimParameter) where {D,U,F<:Function,K,ET<:ElType{K}}

    @timeit to "assembly" k_global,rhs_global, eldata_col = assembly(cv,states,f,sim_pars)
    # @timeit to "assembly" k_global, rhs_global, eldata_col = FEM_assembly(cv,states,sim_pars) #! not working
    apply!(k_global,rhs_global,ch)
  
    @timeit to "solver" u = cholesky(Symmetric(k_global)) \ rhs_global



    return u, k_global, eldata_col
end


