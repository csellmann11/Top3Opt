isdefined(@__MODULE__,:PU3DBoundaryData) || include("../refinement_utils/pu_loads.jl")

function find_maximal_cell_diameter(topo::Topology{3})
    maximal_diameter = -Inf

    for element in RootIterator{4}(topo)
        vol_node_ids = get_volume_node_ids(topo, element.id)

        diameter = max_node_distance(topo.nodes, vol_node_ids)
        maximal_diameter = max(maximal_diameter, diameter)
    end
    return maximal_diameter
end

function find_minimal_cell_diameter(topo::Topology{3})
    minimal_diameter = Inf

    for element in RootIterator{4}(topo)
        vol_node_ids = get_volume_node_ids(topo, element.id)

        diameter = max_node_distance(topo.nodes, vol_node_ids)
        minimal_diameter = min(minimal_diameter, diameter)
    end
    return minimal_diameter
end


function permute_coord_dimensions(mesh::Mesh{D,ET}, perm::SVector{D,Int}) where {D,ET}

    nodes = mesh.topo.nodes
    for i in eachindex(nodes)
        node = nodes[i]
        nodes[i] = Node(node.id, node.coords[perm])
    end
    return Mesh(mesh.topo, StandardEl{1}())
end




function refine_sets(mesh::Mesh{3},
    sets_to_refine::T,
    MAX_REF_LEVEL::Int) where T<:Tuple

    topo = mesh.topo


    ref_marker = zeros(Bool, length(get_volumes(topo)))

    n_els = length(get_volumes(topo))

    for element in RootIterator{4}(topo)

        Ju3VEM.VEMGeo.iterate_volume_areas(
            topo, element.id) do face, _

            Ju3VEM.VEMGeo.iterate_element_edges(
                topo, face.id) do n1_id, _, _
                node = topo.nodes[n1_id]
                if any(set_func(node) for set_func in sets_to_refine)
                    ref_marker[element.id] = true
                end
                nothing
            end
        end
    end

    for element in RootIterator{4}(topo)

        if element.refinement_level >= MAX_REF_LEVEL
            ref_marker[element.id] = false
            continue
        end
        ref_marker[element.id] && _refine!(element, topo)
    end


    for _ in 3:MAX_REF_LEVEL
        ref_marker = resize!(ref_marker, length(get_volumes(topo)))

        for element in RootIterator{4}(topo)
            element.id <= n_els && continue
            _refine!(element, topo)
            ref_marker[element.id] = true
        end

    end

    mesh = Mesh(topo, StandardEl{1}())

    return mesh, ref_marker
end


function mark_elements_which_are_part_of_sets(cv::CellValues{3}, ch::ConstraintHandler{3})
    mesh = cv.mesh
    topo = mesh.topo
    forbid_coarsening = zeros(Bool, length(get_volumes(topo)))

    for element in RootIterator{4}(topo)

        reinit!(element.id, cv)
        dofs = Ju3VEM.VEMUtils.get_cell_dofs(cv)
        forbidden = false

        for dof in dofs
            continue #! remove
            if haskey(ch.n_bcs, dof)
                forbidden = true
                break
            end
            if haskey(ch.d_bcs, dof)
                forbidden = true
                break
            end
        end

        forbid_coarsening[element.id] = forbidden
    end

    return forbid_coarsening
end


function get_sets_to_refine(b_case::Symbol)
    if b_case == :MBB_sym
        return (x -> (0 ≤ x[1] ≤ 0.251) && x[3] ≈ 1.0,)
        # return (x -> x[1] ≈ 0.0 && x[3] ≈ 1.0,)
    elseif b_case == :Cantilever_sym
        return (x -> x[1] ≈ 2.0 && (0.4 ≤ x[3] ≤ 0.6) && (0.4 ≤ x[2] ≤ 0.6),)
    elseif b_case == :Bending_Beam_sym
        return (x -> x[1] ≈ 3.0 && x[3] ≈ 0.0,)
    elseif b_case == :simple_lever
        return (x -> x[1] ≈ 3.0 && x[3] >= 2.5 && x[2] >= 0.4,)
    elseif b_case == :pressure_plate
        return (x -> x[1] > 2.8 && x[2] > 2.8 && x[3] ≈ 1.0,)
    elseif b_case == :L_cantilever
        return (x ->(1.749 ≤ x[1] ≤ 2.0) && x[2] ≈ 1.0 && x[3] >= 0.374,)
    else
        ()
    end
end


function create_constraint_handler(cv::CellValues{3}, b_case::Symbol; boundary_data=nothing)
    mesh = cv.mesh
    ch = ConstraintHandler{U}(mesh)

    if b_case == :MBB_sym

        add_face_set!(mesh, "symmetry_bc", x -> x[1] ≈ 0.0)
        add_face_set!(mesh, "symmetry_bc_2", x -> x[2] ≈ 0.5)
        add_node_set!(mesh, "roller_bearing", x -> x[1] ≈ 3.0 && x[3] ≈ 0.0 && x[2] ≈ 0.0)
        add_face_set!(mesh, "middle_traction", x -> (0 ≤ x[1] ≤ 0.251) && x[3] ≈ 1.0)
        # add_edge_set!(mesh, "middle_traction", x -> (x[1] ≈ 0.0) && x[3] ≈ 1.0)


        pu3d_add_dirichlet!(boundary_data,ch, cv.dh, cv.facedata_col, "symmetry_bc", x -> SA[0.0], c_dofs=SA[1])
        pu3d_add_dirichlet!(boundary_data,ch, cv.dh, cv.facedata_col, "symmetry_bc_2", x -> SA[0.0], c_dofs=SA[2])
        pu3d_add_dirichlet!(boundary_data,ch, cv.dh, "roller_bearing", x -> SA[0.0, 0.0], c_dofs=SA[2, 3])
        pu3d_add_neumann!(boundary_data,ch, cv.dh, cv.facedata_col, "middle_traction", x -> SA[0.0, 0.0, -1.0])
        # add_neumann_bc!(ch, cv.dh, "middle_traction", x -> SA[0.0, 0.0, -1.0])

    elseif b_case == :Cantilever_sym

        add_face_set!(mesh, "symmetry_bc", x -> x[2] ≈ 0.5)
        add_face_set!(mesh, "left_clamp", x -> x[1] ≈ 0.0)
        add_face_set!(mesh, "right_traction", x -> x[1] ≈ 2.0 && (0.4 ≤ x[3] ≤ 0.6) && (0.4 ≤ x[2] ≤ 0.6))

        pu3d_add_dirichlet!(boundary_data,ch, cv.dh, cv.facedata_col, "symmetry_bc", x -> SA[0.0], c_dofs=SA[2])
        pu3d_add_dirichlet!(boundary_data,ch, cv.dh, cv.facedata_col, "left_clamp", x -> SA[0.0, 0.0, 0.0], c_dofs=SA[1, 2, 3])
        pu3d_add_neumann!(boundary_data,ch, cv.dh, cv.facedata_col, "right_traction", x -> SA[0.0, 0.0, -1.0])
    elseif b_case == :Bending_Beam_sym
        add_face_set!(mesh, "symmetry_bc", x -> x[2] ≈ 0.5)
        add_node_set!(mesh, "left_clamp1", x -> x[1] ≈ 0.0 && x[2] ≈ 0.0 && x[3] ≈ 0.0)
        add_node_set!(mesh, "left_clamp2", x -> x[1] ≈ 0.0 && x[2] ≈ 0.0 && x[3] ≈ 1.0)
        add_edge_set!(mesh, "right_traction", x -> x[1] ≈ 3.0 && x[3] ≈ 0.0)
        # add_face_set!(mesh, "right_traction", x -> x[1] ≈ 3.0 && x[3] <= 0.1)

        pu3d_add_dirichlet!(boundary_data,ch, cv.dh, cv.facedata_col, "symmetry_bc", x -> SA[0.0], c_dofs=SA[2])
        pu3d_add_dirichlet!(boundary_data,ch, cv.dh, "left_clamp1", x -> SA[0.0, 0.0, 0.0], c_dofs=SA[1, 2, 3])
        pu3d_add_dirichlet!(boundary_data,ch, cv.dh, "left_clamp2", x -> SA[0.0, 0.0, 0.0], c_dofs=SA[1, 2, 3])
        pu3d_add_neumann!(boundary_data,ch, cv.dh, "right_traction", x -> SA[0.0, 0.0, -1.0])
        # add_neumann_bc!(ch, cv.dh, cv.facedata_col, "right_traction", x -> SA[0.0, 0.0, -100.0])^
    elseif b_case == :simple_lever
        add_node_set!(mesh, "bottom_clamp1", x -> x[1] ≈ 0.0 && x[2] ≈ 0.0 && x[3] ≈ 0.0)
        add_node_set!(mesh, "bottom_clamp2", x -> x[1] ≈ 1.0 && x[2] ≈ 0.0 && x[3] ≈ 0.0)
        add_face_set!(mesh, "right_traction", x -> x[1] ≈ 3.0 && x[3] >= 2.5 && x[2] >= 0.4)
        add_face_set!(mesh, "symmetry_bc", x -> x[2] ≈ 0.5)
        add_face_set!(mesh, "second_traction", x -> x[1] < 1.0 && x[3] ≈ 3.0)

        pu3d_add_dirichlet!(boundary_data,ch, cv.dh, "bottom_clamp1", x -> SA[0.0, 0.0, 0.0], c_dofs=SA[1, 2, 3])
        pu3d_add_dirichlet!(boundary_data,ch, cv.dh, "bottom_clamp2", x -> SA[0.0, 0.0, 0.0], c_dofs=SA[1, 2, 3])
        pu3d_add_dirichlet!(boundary_data,ch, cv.dh, cv.facedata_col, "symmetry_bc", x -> SA[0.0], c_dofs=SA[2])
        pu3d_add_neumann!(boundary_data,ch, cv.dh, cv.facedata_col, "right_traction", x -> SA[0.0, 0.0, -1.0])
        pu3d_add_neumann!(boundary_data,ch, cv.dh, cv.facedata_col, "second_traction", x -> SA[0.0, 0.0, -1.0])
    elseif b_case == :pressure_plate
        add_node_set!(mesh, "bottom_clamp", x -> x[1] ≈ 0.0 && x[2] ≈ 0.0 && x[3] ≈ 0.0)
        add_face_set!(mesh, "symmetry_bc1", x -> x[1] ≈ 3.0)
        add_face_set!(mesh, "symmetry_bc2", x -> x[2] ≈ 3.0)
        add_face_set!(mesh, "pressure", x -> x[1] > 2.8 && x[2] > 2.8 && x[3] ≈ 1.0)

        pu3d_add_dirichlet!(boundary_data,ch, cv.dh, "bottom_clamp", x -> SA[0.0], c_dofs=SA[3])
        pu3d_add_dirichlet!(boundary_data,ch, cv.dh, cv.facedata_col, "symmetry_bc1", x -> SA[0.0], c_dofs=SA[1])
        pu3d_add_dirichlet!(boundary_data,ch, cv.dh, cv.facedata_col, "symmetry_bc2", x -> SA[0.0], c_dofs=SA[2])
        pu3d_add_neumann!(boundary_data,ch, cv.dh, cv.facedata_col, "pressure", x -> SA[0.0, 0.0, -1.0])
    elseif b_case == :L_cantilever
        add_face_set!(mesh, "symmetry_bc", x -> x[3] ≈ 0.5)
        add_node_set!(mesh, "top_clamp1", x -> x[2] ≈ 2.0 && x[3] ≈ 0.0 && x[1] ≈ 0.0)
        add_node_set!(mesh, "top_clamp2", x -> x[2] ≈ 2.0 && x[3] ≈ 0.0 && x[1] ≈ 1.0)
        add_face_set!(mesh, "traction", x -> (1.749 ≤ x[1] ≤ 2.0) && x[2] ≈ 1.0 && x[3] >= 0.374)

        pu3d_add_dirichlet!(boundary_data,ch, cv.dh, cv.facedata_col, "symmetry_bc", x -> SA[0.0], c_dofs=SA[3])
        pu3d_add_dirichlet!(boundary_data,ch, cv.dh, "top_clamp1", x -> SA[0.0, 0.0], c_dofs=SA[1, 2])
        pu3d_add_dirichlet!(boundary_data,ch, cv.dh, "top_clamp2", x -> SA[0.0, 0.0], c_dofs=SA[1, 2])
        pu3d_add_neumann!(boundary_data,ch, cv.dh, cv.facedata_col, "traction", x -> SA[0.0, -1.0, 0.0])
    else
        error("Invalid boundary value problem: $b_case")
    end
    return ch
end


function clear_up_topo!(
    topo::Topology{3})

    do_face_coarse_markers = ones(Bool, length(get_areas(topo)))
    do_edge_coarse_markers = ones(Bool, length(get_edges(topo)))
    for element in RootIterator{4}(topo)

        Ju3VEM.VEMGeo.iterate_volume_areas(
            topo, element.id) do face, _

            face_ref_level = face.refinement_level
            do_coarse = do_face_coarse_markers[face.id] && (face_ref_level > element.refinement_level)

            do_face_coarse_markers[face.id] = do_coarse


            Ju3VEM.VEMGeo.iterate_element_edges(
                topo, face.id) do _, edge_id, _

                edge_ref_level = get_edges(topo)[edge_id].refinement_level
                do_coarse = do_edge_coarse_markers[edge_id] && (edge_ref_level > element.refinement_level)
                do_edge_coarse_markers[edge_id] = do_coarse

            end
        end
    end

    for face in RootIterator{3}(topo)
        do_coarse = do_face_coarse_markers[face.id]
        do_coarse || continue
        _coarsen!(face, topo)
    end
    for edge in RootIterator{2}(topo)
        do_coarse = do_edge_coarse_markers[edge.id]
        do_coarse || continue
        _coarsen!(edge, topo)
    end
end






function mesh_cleanup_signed_area2(ids,topo)
    origin = topo.nodes[first(ids)].coords
    result = 0.0
    for i in eachindex(ids)
        a = topo.nodes[ids[i]].coords-origin
        b = topo.nodes[ids[mod1(i+1,length(ids))]].coords-origin
        result += a[1]*b[2]-a[2]*b[1]
    end
    return result
end

face_area(ids::AbstractVector{<:Integer},topo::Topology{2}) =
    abs(mesh_cleanup_signed_area2(ids,topo))/2

using Ju3VEM.VEMGeo: get_area_edge_ids

function create_dense_node_id_map(topo::Topology{D}) where {D}
    mapping = FixedSizeVector{Int32}(undef,length(topo.nodes))
    fill!(mapping,0)
    next = 1
    for node in topo.nodes
        is_active(node) || continue
        mapping[node.id] = next
        next += 1
    end
    return mapping
end

function mesh_cleanup_replace(ids,removed,survivor)
    result = Int[]
    for id in ids
        mapped = id==removed ? survivor : id
        (isempty(result) || last(result)!=mapped) && push!(result,mapped)
    end
    length(result)>1 && first(result)==last(result) && pop!(result)
    return result
end

"""All-vertex half-plane checks exclude crossings as well as nonconvex proposals."""
function mesh_cleanup_valid_polygon(ids,topo,orientation)
    length(ids)>=3 && length(unique(ids))==length(ids) || return false
    origin = topo.nodes[first(ids)].coords
    scale2 = maximum(sum(abs2,topo.nodes[id].coords-origin) for id in ids)
    tolerance = 128eps(Float64)*scale2
    orientation*mesh_cleanup_signed_area2(ids,topo)>tolerance || return false
    for i in eachindex(ids)
        a = topo.nodes[ids[i]].coords
        edge = topo.nodes[ids[mod1(i+1,length(ids))]].coords-a
        sum(abs2,edge)>eps(Float64)^2*scale2 || return false
        for id in ids
            offset = topo.nodes[id].coords-a
            orientation*(edge[1]*offset[2]-edge[2]*offset[1]) >= -tolerance || return false
        end
    end
    return true
end

function mesh_cleanup_edge_changes!(counts,signs,ids,orientation,increment)
    for i in eachindex(ids)
        a,b = ids[i],ids[mod1(i+1,length(ids))]
        edge = minmax(a,b)
        counts[edge] = get(counts,edge,0)+increment
        signs[edge] = get(signs,edge,0)+increment*orientation*(a<b ? 1 : -1)
    end
end

function mesh_cleanup_update_edges!(owners,signs,ids,face,orientation,increment)
    for i in eachindex(ids)
        a,b = ids[i],ids[mod1(i+1,length(ids))]
        edge = minmax(a,b)
        if increment==1
            push!(get!(owners,edge,Set{Int}()),face)
        else
            delete!(owners[edge],face)
        end
        signs[edge] = get(signs,edge,0)+increment*orientation*(a<b ? 1 : -1)
        if isempty(owners[edge])
            delete!(owners,edge)
            delete!(signs,edge)
        end
    end
end

"""Allow deletion only between the two current collinear boundary neighbours."""
function mesh_cleanup_boundary_other(topo,neighbors,removed,survivor)
    adjacent = neighbors[removed]
    length(adjacent)==2 && survivor in adjacent || return 0
    other = only(id for id in adjacent if id!=survivor)
    a = topo.nodes[survivor].coords-topo.nodes[removed].coords
    b = topo.nodes[other].coords-topo.nodes[removed].coords
    scale2 = max(sum(abs2,a),sum(abs2,b))
    abs(a[1]*b[2]-a[2]*b[1])<=128eps(Float64)*scale2 && dot(a,b)<0 || return 0
    return other
end

"""Validate every affected face and edge before changing current connectivity."""
function mesh_cleanup_proposal(topo,polygons,orientations,areas,node_faces,
    edge_faces,edge_signs,boundary_neighbors,removed,survivor)
    boundary = !isempty(boundary_neighbors[removed])
    other = boundary ? mesh_cleanup_boundary_other(topo,boundary_neighbors,removed,survivor) : 0
    boundary && other==0 && return nothing
    affected = sort!(collect(node_faces[removed]))
    replacements = [mesh_cleanup_replace(polygons[face],removed,survivor) for face in affected]
    all(mesh_cleanup_valid_polygon(ids,topo,orientations[face])
        for (face,ids) in zip(affected,replacements)) || return nothing
    newareas = [face_area(ids,topo) for ids in replacements]
    oldarea = sum(areas[affected])
    abs(sum(newareas)-oldarea)<=512eps(Float64)*oldarea || return nothing
    counts,signs = Dict{Tuple{Int,Int},Int}(),Dict{Tuple{Int,Int},Int}()
    for (face,ids) in zip(affected,replacements)
        mesh_cleanup_edge_changes!(counts,signs,polygons[face],orientations[face],-1)
        mesh_cleanup_edge_changes!(counts,signs,ids,orientations[face],1)
    end
    old_boundary,new_boundary = Set{Tuple{Int,Int}}(),Set{Tuple{Int,Int}}()
    for (edge,change) in counts
        oldcount = haskey(edge_faces,edge) ? length(edge_faces[edge]) : 0
        newcount = oldcount+change
        newsign = get(edge_signs,edge,0)+signs[edge]
        0<=newcount<=2 || return nothing
        (newcount==2 ? newsign==0 : abs(newsign)==newcount) || return nothing
        oldcount==1 && push!(old_boundary,edge)
        newcount==1 && push!(new_boundary,edge)
    end
    if boundary
        delete!(old_boundary,minmax(removed,survivor))
        delete!(old_boundary,minmax(removed,other))
        push!(old_boundary,minmax(survivor,other))
    end
    old_boundary==new_boundary || return nothing
    return (;affected,replacements,newareas,other)
end

function mesh_cleanup_rebuild(topo,polygons)
    used = falses(length(topo.nodes))
    for ids in polygons
        used[ids] .= true
    end
    mapping = zeros(Int,length(topo.nodes))
    result = Topology{2}()
    for id in findall(used)
        is_active(topo.nodes[id]) || error("Cleanup retained an inactive node $id")
        mapping[id] = add_node!(topo.nodes[id].coords,result)
    end
    for ids in polygons
        mapped = mapping[ids]
        all(>(0),mapped) || error("Cleanup produced an unmapped polygon vertex")
        add_area!(mapped,result)
    end
    length(RootIterator{3}(result))==length(polygons) || error("Cleanup merged distinct polygons")
    return result
end

"""Remove safe short edges and return a fresh, densely numbered topology.

The input is unchanged. Active leaf polygons include hanging edge vertices;
rebuilding flattens their hierarchy and retains only referenced active nodes.
Candidates keep length^2 < area/8 on every incident polygon. Current incidence
is updated after each accepted contraction, including faces acquired by its
survivor. Degenerate, inverted, nonconvex, nonmanifold, or boundary-changing
proposals are skipped. Original polygon orientations are preserved. New edges
can be considered by the next cleanup pass.
"""
function remove_short_edges(topo::Topology{2})
    polygons = [collect(Ju3VEM.VEMGeo.get_iterative_area_vertex_ids(face,topo))
        for face in RootIterator{3}(topo)]
    isempty(polygons) && return Topology{2}()
    node_faces = [Set{Int}() for _ in topo.nodes]
    edge_faces,edge_signs = Dict{Tuple{Int,Int},Set{Int}}(),Dict{Tuple{Int,Int},Int}()
    orientations,areas = Int[],Float64[]
    for (face,ids) in enumerate(polygons)
        length(ids)>=3 && length(unique(ids))==length(ids) ||
            throw(ArgumentError("Invalid cleanup input polygon"))
        for id in ids
            1<=id<=length(topo.nodes) && is_active(topo.nodes[id]) ||
                throw(ArgumentError("Cleanup input refers to an invalid or inactive node $id"))
            all(isfinite,topo.nodes[id].coords) || throw(ArgumentError("Nonfinite cleanup coordinates"))
            push!(node_faces[id],face)
        end
        signed = mesh_cleanup_signed_area2(ids,topo)
        isfinite(signed) && signed!=0 || throw(ArgumentError("Zero-area cleanup input polygon"))
        push!(orientations,signed>0 ? 1 : -1)
        push!(areas,abs(signed)/2)
        mesh_cleanup_update_edges!(edge_faces,edge_signs,ids,face,last(orientations),1)
    end
    boundary_neighbors = [Set{Int}() for _ in topo.nodes]
    for ((a,b),owners) in edge_faces
        length(owners)<=2 || throw(ArgumentError("Nonmanifold cleanup input edge"))
        length(owners)==2 && edge_signs[(a,b)]!=0 &&
            throw(ArgumentError("Overlapping cleanup input polygons"))
        if length(owners)==1
            push!(boundary_neighbors[a],b)
            push!(boundary_neighbors[b],a)
        end
    end
    candidates = [(sum(abs2,topo.nodes[a].coords-topo.nodes[b].coords),a,b)
        for ((a,b),owners) in edge_faces
        if all(length(polygons[face])>3 &&
            sum(abs2,topo.nodes[a].coords-topo.nodes[b].coords)<areas[face]/8 for face in owners)]
    sort!(candidates)
    removed_count = 0
    for (length2,a,b) in candidates
        owners = get(edge_faces,(a,b),nothing)
        owners===nothing && continue
        all(length(polygons[face])>3 && length2<areas[face]/8 for face in owners) || continue
        boundary_a,boundary_b = !isempty(boundary_neighbors[a]),!isempty(boundary_neighbors[b])
        boundary_a && boundary_b && length(owners)!=1 && continue
        directions = boundary_a && !boundary_b ? ((b,a),) :
            boundary_b && !boundary_a ? ((a,b),) : ((b,a),(a,b))
        for (removed,survivor) in directions
            proposal = mesh_cleanup_proposal(topo,polygons,orientations,areas,node_faces,
                edge_faces,edge_signs,boundary_neighbors,removed,survivor)
            proposal===nothing && continue
            for face in proposal.affected
                mesh_cleanup_update_edges!(edge_faces,edge_signs,polygons[face],face,orientations[face],-1)
            end
            for (face,ids,area) in zip(proposal.affected,proposal.replacements,proposal.newareas)
                polygons[face],areas[face] = ids,area
                push!(node_faces[survivor],face)
                mesh_cleanup_update_edges!(edge_faces,edge_signs,ids,face,orientations[face],1)
            end
            empty!(node_faces[removed])
            if proposal.other!=0
                other = proposal.other
                delete!(boundary_neighbors[survivor],removed)
                delete!(boundary_neighbors[other],removed)
                push!(boundary_neighbors[survivor],other)
                push!(boundary_neighbors[other],survivor)
                empty!(boundary_neighbors[removed])
            end
            removed_count += 1
            break
        end
    end
    println("removed $removed_count short edges")
    return mesh_cleanup_rebuild(topo,polygons)
end
