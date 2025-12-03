using Ju3VEM
import Ju3VEM.FR as FR

function assembly(cv::CellValues{D,U,ET},
    states::DesignVarInfo{D},
    f::F,
    sim_pars::SimPars) where {D,U,F<:Function,ET<:ElType{2}}


    K = 2
    mat_law = sim_pars.mat_law
    k_global = get_sparsity_pattern(cv)
    ass = FR.start_assemble(k_global,zeros(size(k_global,1)))

    fe_ip = FR.Lagrange{FR.RefTetrahedron, 1}()^U
    fe_qr = FR.QuadratureRule{FR.RefTetrahedron}(K)
    fe_cv = FR.CellValues(fe_qr,fe_ip)



    Is = SMatrix{U,U,Float64}(I)
    γ = Is ⊡₂ eval_hessian(mat_law,Is,(sim_pars.λ,sim_pars.μ,1.0)) ⊡₂ Is
    rhs_element = DefCachedVector{Float64}()
    kelement    = DefCachedMatrix{Float64}()


    eldata_col = Dict{Int,ElData{D}}()

    e2s = states.el_id_to_state_id


    for element in RootIterator{4}(cv.mesh.topo)
        χ = states.χ_vec[e2s[element.id]]
        reinit!(element.id,cv)
        material_pars = (sim_pars.λ,sim_pars.μ,χ)

        γ_stab = γ/4 * χ^3 

        proj_s, proj = build_local_kel_and_f!(
            kelement, 
            rhs_element,
            cv, 
            element.id,
            f,
            fe_cv, 
            mat_law,
            γ, 
            material_pars
        )

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

        FR.assemble!(ass,dofs,kelement.array,rhs_element.array)
    end

    kglobal, rhsglobal = FR.finish_assemble(ass)

    kglobal, rhsglobal, eldata_col

end





function compute_second_order_solution(
    cv_o1::CellValues{D,U},
    states::DesignVarInfo,
    sim_pars::SimPars,
    b_case::Symbol) where {D,U}


    mesh_o1 = cv_o1.mesh 

    mesh = Mesh(mesh_o1.topo,StandardEl{2}())
    cv   = CellValues{3}(mesh)
    ch   = create_constraint_handler(cv,b_case)


    f = x -> zeros(3)
    kglobal, rhsglobal, eldata_col = assembly(cv, states, f, sim_pars)
    apply!(kglobal,rhsglobal,ch) 



    u = zero(rhsglobal)
    Pardiso.pardiso(ps, u,tril(kglobal), rhsglobal)
    
    n_o1_nodes = length(get_nodes(mesh_o1.topo))

    @show mesh.node_sets["roller_bearing"]

    write_vtu_file(cv_o1,eldata_col,"delete_me",u[1:n_o1_nodes*3])

    return u
end


function test_script()


    mesh_o1 =  create_rectangular_mesh(
        12, 2, 4,
        3.0, 0.5, 1.0, StandardEl{1}
    )

    E = 210.e03
    ν = 0.33
    λ, μ = E_ν_to_lame(E, ν)
    mat_law = Helmholtz{3,3}(Ψlin_totopt, (λ, μ, 1.0))

    χmin = 1e-03
    η0 = 15.0
    sim_pars = SimPars(mat_law, λ, μ, χmin, η0, 1.0, 0.3)

    cv_o1 = CellValues{3}(mesh_o1)
    states = DesignVarInfo{3}(cv_o1, 0.3)

    compute_second_order_solution(cv_o1, states, sim_pars,:MBB_sym)

end
test_script()