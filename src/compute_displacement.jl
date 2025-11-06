struct ElData{D}
    el_id ::Int 
    proj_s::FixedSizeMatrixDefault{Float64}
    proj  ::FixedSizeMatrixDefault{Float64}
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

    eldata_col  = Dict{Int,ElData{D}}()

    e2s = states.el_id_to_state_id


    for element in RootIterator{4}(cv.mesh.topo)
        χ = states.χ_vec[e2s[element.id]]
        reinit!(element.id,cv)

        γ_stab = γ/4 * χ^3

        proj_s, proj = build_local_kel_and_f!(kelement,
        rhs_element,cv,element.id,f,mat_law,γ_stab,(sim_pars.λ,sim_pars.μ,χ))

        


  
        # cell_dofs = Ju3VEM.VEMUtils.get_cell_dofs(cv)
        # node_ids = collect(keys(cv.vnm.map))
        node_ids = collect_nodes_by_order(cv.vnm.map)
        cell_dofs = FixedSizeVector{Int}(undef,length(node_ids)*U)
        get_dofs!(cell_dofs,cv.dh,node_ids)

  
        vol_data = cv.volume_data
        bc_vol   = vol_data.vol_bc
        hvol     = vol_data.hvol 
        volume   = vol_data.integrals[1]
        eldata_col[element.id] = ElData(element.id,proj_s,proj,node_ids,cell_dofs,γ_stab,hvol,volume,bc_vol)
  
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
    # @timeit to "assembly" k_global, rhs_global, eldata_col = FEM_assembly(cv,states,sim_pars)
    apply!(k_global,rhs_global,ch)
  
    @timeit to "solver" u = cholesky(Symmetric(k_global)) \ rhs_global



    return u, k_global, eldata_col
end


