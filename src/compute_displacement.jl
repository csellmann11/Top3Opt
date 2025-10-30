struct ElData{D}
    el_id ::Int 
    proj_s::FixedSizeMatrixDefault{Float64}
    proj  ::FixedSizeMatrixDefault{Float64}
    dofs  ::FixedSizeVectorDefault{Int}
    
    γ_stab ::Float64
    hvol   ::Float64
    volume ::Float64
    bc     ::SVector{D,Float64}
end

function assembly(cv::CellValues{D,U,ET},
    states::TopStates{D},
    f::F,
    sim_pars::SimParameter) where {D,U,F<:Function,K,ET<:ElType{K}}

    mat_law = sim_pars.mat_law
    mat_pars = (sim_pars.λ,sim_pars.μ)
    ass = Assembler{Float64}(cv)


    Is = SMatrix{U,U,Float64}(I)
    γ = Is ⊡₂ eval_hessian(mat_law,Is) ⊡₂ Is
    rhs_element = DefCachedVector{Float64}()
    kelement    = DefCachedMatrix{Float64}()

    eldata_col  = Dict{Int,ElData{D}}()

    e2s = states.el_id_to_state_id


    for element in RootIterator{4}(cv.mesh.topo)
        χ = states.χ_vec[e2s[element.id]]
        reinit!(element.id,cv)

        γ_stab = γ * χ^3

        proj_s, proj = build_local_kel_and_f!(kelement,
        rhs_element,cv,element.id,f,mat_law,γ_stab,(sim_pars.λ,sim_pars.μ,χ))

        


  
        cell_dofs = Ju3VEM.VEMUtils.get_cell_dofs(cv)

  
        vol_data = cv.volume_data
        bc_vol   = vol_data.vol_bc
        hvol     = vol_data.hvol 
        volume   = vol_data.integrals[1]
        eldata_col[element.id] = ElData(element.id,proj_s,proj,cell_dofs,γ_stab,hvol,volume,bc_vol)
  
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
    apply!(k_global,rhs_global,ch)
  
    @timeit to "solver" u = cholesky(Symmetric(k_global)) \ rhs_global



    return u, k_global, eldata_col
end


