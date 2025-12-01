using Ju3VEM
using Ju3VEM.FixedSizeArrays
using StaticArrays
using LinearAlgebra
using SparseArrays
using Chairmarks
import Ju3VEM.FR as FR
using Statistics
using Random
using Bumper
using TimerOutputs
using OrderedCollections
using Dates
using Bumper
using JLD2
using Infiltrator
using Dates: today
using Pardiso

const to = TimerOutput()
const ps = MKLPardisoSolver()
set_matrixtype!(ps, 2)
set_nprocs!(ps, Threads.nthreads()) # Sets the number of threads to use




include("../src/general_utils.jl")
include("../src/hash_output.jl")
args = parse_commandline()
include("../src/mat_states.jl")
include("../src/laplace_operator.jl")
include("../src/compute_displacement.jl")
include("../src/bisection.jl")
include("../src/mesh_processing_utils.jl")
include("../src/neighbor_search.jl")
include("../src/refinement_utils/estiamte_element_error.jl")
include("../src/postprocessing/sim_data.jl")
include("../src/postprocessing/topopt_vtk_export.jl")
include("../src/optim_run.jl")
include("../src/get_sparsity_pattern.jl")



"""
Function to define the boundary of a Christmas tree with zig-zag tiers.

Returns true if point (x, y) is inside the tree, false otherwise.
Tree is centered at x=0, with base at y=0.
"""
function is_inside_christmas_tree(x, y)
    # Tree parameters
    tree_height = 10.0
    tree_base_width = 6.0
    trunk_width = 1.0
    trunk_height = 1.5
    
    # Number of tiers (zig-zag levels)
    n_tiers = 4
    
    # Check trunk (at the bottom)
    if y >= 0.0 && y <= trunk_height && abs(x) <= trunk_width/2
        return true
    end
    
    # Main tree body with zig-zag tiers
    if y > trunk_height && y <= tree_height
        # Calculate which tier we're in
        tier_height = (tree_height - trunk_height) / n_tiers
        current_tier = floor((y - trunk_height) / tier_height)
        
        # Position within current tier (0 at bottom of tier, 1 at top)
        tier_progress = ((y - trunk_height) - current_tier * tier_height) / tier_height
        
        # Height from bottom of tree (for overall taper)
        height_from_base = y - trunk_height
        
        # Overall triangular envelope (gets narrower going up)
        max_width_at_y = tree_base_width * (1.0 - height_from_base / (tree_height - trunk_height))
        
        # Zig-zag effect: each tier tapers in, then next tier starts wider
        if current_tier < n_tiers
            # Width at bottom of tier
            tier_bottom_fraction = 0.95  # Each tier starts at 95% of envelope
            # Width at top of tier
            tier_top_fraction = 0.65     # Each tier ends at 65% of envelope
            
            # Interpolate width fraction within tier
            width_fraction = tier_bottom_fraction - (tier_bottom_fraction - tier_top_fraction) * tier_progress
            
            # Actual width at this height
            current_width = max_width_at_y * width_fraction
            
            if abs(x) < current_width / 2
                return true
            end
        end
    end
    
    return false
end

function is_inside_christmas_tree(x::StaticVector)
    return is_inside_christmas_tree(x[1],x[2]) 
end

function christmas_constrain_handler(cv::CellValues{3})

    mesh = cv.mesh
    topo = mesh.topo

    ch = ConstraintHandler{3}(mesh)
    
    add_face_set!(
        mesh,"trunk_bottom",x -> x[2] ≈ 0.0)
    
    add_face_set!(mesh,"tree_top",x -> x[2] ≈ 10.0 && -0.5 < x[1] < 0.5)
    add_face_set!(mesh,"tree_body",is_inside_christmas_tree)

    add_dirichlet_bc!(ch, cv.dh, cv.facedata_col,"trunk_bottom", x -> SA[0.0, 0.0, 0.0], c_dofs=SA[1, 2, 3])
    add_neumann_bc!(ch,cv.dh,cv.facedata_col,"tree_top",x -> SA[0.0, 0.0, 10.0])    
    # add_dirichlet_bc!(ch, cv.dh, cv.facedata_col,"tree_top", x -> SA[0.0, 0.0, 0.0], c_dofs=SA[1, 2, 3])
    # add_neumann_bc!(ch,cv.dh,cv.facedata_col,"trunk_bottom",x -> SA[0.0, 0.0, -10.0])   
    # add_neumann_bc!(ch,cv.dh,cv.facedata_col,"tree_body",x -> SA[0.0, 0.0, -10.0])

    return ch
end


function adjust_states!(states::DesignVarInfo{3},tree_state::Dict{Int,Int})
    for (el_id,bc_info) in tree_state

        state_id = states.el_id_to_state_id[el_id]
        if bc_info == 1
            states.χ_vec[state_id] = 1.0 
        elseif bc_info == 2 # is inside the tree
            states.χ_vec[state_id] = 0.2 
        elseif bc_info == 3 # is outside the tree
            states.χ_vec[state_id] = 1e-03
        end
    end
end

function compute_von_mises_stresses(
    u::AbstractVector{Float64},
    cv::CellValues{3},
    eldata_col::Dict{Int,<:ElData},
    states::DesignVarInfo{3},
    sim_pars::SimPars{H}) where H<:Helmholtz

    von_mises_stresses = Dict{Int,Float64}()
    base = get_base(BaseInfo{3,1,3}())
    for (el_id,elem_data) in eldata_col
        state_id = states.el_id_to_state_id[el_id]
        χ = states.χ_vec[state_id]
        
        proj_s   = elem_data.proj_s
        node_ids = elem_data.node_ids
        dofs     = get_dofs(cv.dh,node_ids)
        uel      = @view u[dofs]
        h        = states.h_vec[state_id]
        uπ       = sol_proj(base,uel,stretch(proj_s,Val(3)))
        bc       = states.x_vec[state_id]
        ∇u       = ∇x(uπ,h,zero(bc))
        ℂ        = eval_hessian(sim_pars.mat_law,∇u,(sim_pars.λ,sim_pars.μ,χ))
        σ        = ℂ ⊡₂ (1/2*(∇u + ∇u'))
        s        = σ  - 1/3*tr(σ)*I
        von_mises_stress = sqrt(1/2 * tr(s*s))
        von_mises_stresses[el_id] = von_mises_stress
    end

    return von_mises_stresses
end


function create_christ_tree_mesh()
    n = 10
    # mesh = create_rectangular_mesh(
    #     6,15,1,
    #     6.0,10.0,0.1,StandardEl{K}
    # )

    mesh2d = create_voronoi_mesh(
    (0.0,0.0),
    (7.0,10.0),
    6,10,StandardEl{1})


    mesh = extrude_to_3d(1,mesh2d,2.0);

    topo = mesh.topo
    for i in eachindex(topo.nodes)
        old_coords = topo.nodes[i].coords 
        new_coords = SA[old_coords[1]-3.5,old_coords[2],old_coords[3]]
        topo.nodes[i] = Node(i,new_coords)
    end

    function is_boundary_element(element_id::Int,topo::Topology{3})
        node_ids = get_volume_node_ids(topo, element_id)
        has_nodes_outside = false
        has_nodes_inside = false
        
        # Check node positions
        for node_id in node_ids 
            node = topo.nodes[node_id] 
            x,y,_ = node.coords 
            status = is_inside_christmas_tree(x,y)
            if status == 1
                has_nodes_inside = true 
            elseif status == 0
                has_nodes_outside = true
            else
                has_nodes_inside = true 
                has_nodes_outside = true
            end
        end
        
        # If nodes are on different sides, it's definitely a boundary element
        if has_nodes_outside && has_nodes_inside
            return true, true, true
        end
        
        # Even if all nodes are on the same side, check if boundary crosses any edge
        # by sampling points along each edge
        n_samples = 100  # Number of sample points along each edge
        checked_edges = Set{Tuple{Int,Int}}()  # Track edges we've already checked
        boundary_crosses = false
        
        Ju3VEM.VEMGeo.iterate_volume_areas(topo, element_id) do face, _
            boundary_crosses && return  # Early exit if we already found a crossing
            
            # Get face node IDs to iterate over consecutive pairs
            face_node_ids = Ju3VEM.VEMGeo.get_area_node_ids(topo, face.id)
            
            for (counter, v_id) in enumerate(face_node_ids)
                boundary_crosses && break  # Break if we found a crossing
                
                # Get next node (wrapping around)
                counter_p1 = mod1(counter + 1, length(face_node_ids))
                v_id_p1 = face_node_ids[counter_p1]
                
                # Create edge key (sorted to avoid duplicates)
                edge_key = v_id < v_id_p1 ? (v_id, v_id_p1) : (v_id_p1, v_id)
                
                # Skip if we've already checked this edge
                if edge_key in checked_edges
                    continue
                end
                push!(checked_edges, edge_key)
                
                # Get node coordinates
                n1 = topo.nodes[v_id]
                n2 = topo.nodes[v_id_p1]
                
                x1, y1, _ = n1.coords
                x2, y2, _ = n2.coords
                
                # Check endpoints
                status1 = is_inside_christmas_tree(x1, y1)
                status2 = is_inside_christmas_tree(x2, y2)
                
                # If endpoints have different status, boundary crosses the edge
                if status1 != status2
                    boundary_crosses = true
                    break
                end
                
                # Sample intermediate points to catch cases where boundary crosses
                # but endpoints happen to be on the same side
                for i in 1:(n_samples-1)
                    t = i / n_samples
                    x_sample = x1 + t * (x2 - x1)
                    y_sample = y1 + t * (y2 - y1)
                    status_sample = is_inside_christmas_tree(x_sample, y_sample)
                    
                    # If any sample point has different status than endpoints, boundary crosses
                    if status_sample != status1
                        boundary_crosses = true
                        break
                    end
                end
            end
        end
        
        return boundary_crosses, has_nodes_outside, has_nodes_inside
    end

    max_iter = 5

    tree_state = Dict{Int,Int}()
    for iter in 1:max_iter
        println("Refinement iteration: $iter")
        coarse_marker = zeros(Bool, length(get_volumes(mesh.topo)))
        ref_marker = zeros(Bool, length(get_volumes(mesh.topo)))
        for element in RootIterator{4}(mesh.topo) 
            node_ids = get_volume_node_ids(mesh.topo, element.id)

            is_b, _, _ = is_boundary_element(element.id,mesh.topo)
            if is_b
                # Ju3VEM.VEMGeo._refine!(element,mesh.topo)
                ref_marker[element.id] = true
            else 
                # Ju3VEM.VEMGeo._coarsen!(element,mesh.topo)
                coarse_marker[element.id] = true
            end

        end
        for element in RootIterator{4}(mesh.topo) 
            if ref_marker[element.id]
                Ju3VEM.VEMGeo._refine!(element,mesh.topo)
            elseif coarse_marker[element.id]
                # check if all children are marked for coarseing
                parent_id = element.parent_id
                if parent_id != 0 && all(coarse_marker[element.childs])
                    coarse_marker[parent_id] = true
                end
            end
        end

        clear_up_topo!(mesh.topo)
    end

    for element in RootIterator{4}(mesh.topo) 
        is_b, has_nodes_outside, has_nodes_inside = is_boundary_element(element.id,mesh.topo)
        if is_b 
            tree_state[element.id] = 1
        elseif has_nodes_inside 
            tree_state[element.id] = 2
        elseif has_nodes_outside 
            tree_state[element.id] = 3
        end
    end

    mesh = Mesh(mesh.topo,StandardEl{1}())
    cv = CellValues{3}(mesh)
    states = DesignVarInfo{3}(cv,1.0)
    adjust_states!(states,tree_state)
    E = 210.e03
    ν = 0.33
    λ, μ = E_ν_to_lame(E, ν)
    mat_law = Helmholtz{3,3}(Ψlin_totopt, (λ, μ, 1.0))

    χmin = 1e-03
    η0 = 15.0
    sim_pars = SimPars(mat_law, λ, μ, χmin, η0, 1.0, 1.0)
    ch = christmas_constrain_handler(cv)

    tree_state_vect = el_dict_to_state_vec(tree_state,states)

    println("Computing displacement")
    u,_,eldata_col = compute_displacement(cv,ch,states,x -> SA[0.0, 0.0, 0.0],sim_pars)
    von_mises_stresses = compute_von_mises_stresses(u,cv,eldata_col,states,sim_pars)
    von_mises_stresses_vect = el_dict_to_state_vec(von_mises_stresses,states)

    println("Writing vtu file")
    @time "vtu_file" write_vtu_file(cv,eldata_col,"Results/christ_tree",u;cell_data_col = (tree_state_vect,von_mises_stresses_vect))
    println("Number of elements: $(length(get_volumes(mesh.topo)))")
    println("Number of nodes: $(length(get_nodes(mesh.topo)))")
end

create_christ_tree_mesh()