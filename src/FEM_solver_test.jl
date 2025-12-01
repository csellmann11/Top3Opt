import Ju3VEM.FR as FR
using Ju3VEM

function order_node_to_ferrite_hex_order(nodes::AbstractVector)
    @assert length(nodes) == 8 "Expected 8 nodes for a hexahedron"
    
    # Find the center of the hex
    center = sum(nodes) / 8
    
    # Classify each node by its position relative to center
    # Store both node and its original index
    node_data = map(enumerate(nodes)) do (idx, node)
        rel = node - center
        # Determine sign of each coordinate relative to center
        sx = rel[1] >= 0 ? 1 : -1
        sy = rel[2] >= 0 ? 1 : -1
        sz = rel[3] >= 0 ? 1 : -1
        (node, idx, sx, sy, sz)
    end
    
    # Ferrite hex ordering pattern: (sign_x, sign_y, sign_z)
    # Bottom face (z=-1): counterclockwise when viewed from above
    # Top face (z=+1): same pattern
    target_order = [
        (-1, -1, -1),  # v1: bottom-back-left
        ( 1, -1, -1),  # v2: bottom-back-right
        ( 1,  1, -1),  # v3: bottom-front-right
        (-1,  1, -1),  # v4: bottom-front-left
        (-1, -1,  1),  # v5: top-back-left
        ( 1, -1,  1),  # v6: top-back-right
        ( 1,  1,  1),  # v7: top-front-right
        (-1,  1,  1),  # v8: top-front-left
    ]
    
    # Match each target position to a node
    ordered_nodes = similar(nodes)
    mapping = Vector{Int}(undef, 8)
    
    for (new_idx, (target_sx, target_sy, target_sz)) in enumerate(target_order)
        idx = findfirst(node_data) do (node, old_idx, sx, sy, sz)
            sx == target_sx && sy == target_sy && sz == target_sz
        end
        @assert !isnothing(idx) "Could not find node at position ($target_sx, $target_sy, $target_sz)"
        
        ordered_nodes[new_idx] = node_data[idx][1]
        mapping[new_idx] = node_data[idx][2]  # old index
    end
    
    return ordered_nodes, mapping
end

function FEM_assembly(cv::CellValues{3},
    states::DesignVarInfo{3},
    sim_pars::SimPars)

    ip = FR.Lagrange{FR.RefHexahedron,1}()^3
    qr = FR.QuadratureRule{FR.RefHexahedron}(2)


    ass = Assembler{Float64}(cv)
    Is = SMatrix{3,3,Float64}(I)
    CC = eval_hessian(sim_pars.mat_law,Is,(sim_pars.λ,sim_pars.μ,1.0)) 


    fe_cv = FR.CellValues(qr,ip)

    mesh = cv.mesh; topo = mesh.topo;

    eldata_col  = Dict{Int,ElData{3}}()

    for element in RootIterator{4}(topo)

        χ = states.χ_vec[states.el_id_to_state_id[element.id]]

        reinit!(element.id,cv)
        node_ids = get_volume_node_ids(topo,element.id)

        _nodes = [FR.Tensors.Vec{3}(topo.nodes[node_id].coords) for node_id in node_ids] 
        nodes, mapping = order_node_to_ferrite_hex_order(_nodes)

        dof_mapping = FixedSizeVector{Int}(undef,24)
        for i in eachindex(mapping) 
           for u in 1:3
                dof_mapping[3*(i-1)+u] = 3*(mapping[i]-1)+u
           end
        end


        FR.reinit!(fe_cv,nodes)

        k = FixedSizeMatrix{Float64}(undef,24,24)
        k .= 0.0
        rhs_element = FixedSizeVector{Float64}(undef,24)
        rhs_element .= 0.0

        for q_point in 1:FR.getnquadpoints(fe_cv)
            dΩ = FR.getdetJdV(fe_cv,q_point)
            for i in 1:FR.getnbasefunctions(fe_cv)
                ∇Ni = FR.shape_gradient(fe_cv,q_point,i)
                i_idx = dof_mapping[i]
                C∇Ni = SMatrix{3,3,Float64}(∇Ni) ⊡₂ CC 
                for j in 1:FR.getnbasefunctions(fe_cv)
                    ∇Nj = FR.shape_gradient(fe_cv,q_point,j)
                    j_idx = dof_mapping[j]
                    k[i_idx,j_idx] += ∇Nj ⋅ C∇Ni * dΩ * χ^3
                end
            end
        end

          
        vol_data = cv.volume_data
        bc_vol   = vol_data.vol_bc
        hvol     = vol_data.hvol 
        volume   = vol_data.integrals[1]

        proj_s, proj = create_volume_vem_projectors(
            element.id,cv.mesh,cv.volume_data,cv.facedata_col,cv.vnm)
    

        local_assembly!(ass,k,rhs_element)
        cell_dofs = Ju3VEM.VEMUtils.get_cell_dofs(cv)

        eldata_col[element.id] = ElData(
            element.id,proj_s,proj,cell_dofs,1.0,hvol,volume,bc_vol)
  


    end

    kglobal, rhsglobal = assemble!(ass)


    return kglobal, rhsglobal, eldata_col
end