using LinearAlgebra

"""Actual tetrahedral face ownership, exterior faces, and face connectivity.

This uses vertex triples, never `fan.tet_faces`, which can be provenance only.
Every interior triangle must have two owners. Exterior triangles have one.
"""
function poisson_refinement_face_audit(tet_nodes)
    nt = size(tet_nodes, 2)
    nt > 0 || throw(ArgumentError("A tetrahedral mesh must contain tetrahedra"))
    owners = Dict{NTuple{3,Int},Tuple{Int,Int}}()
    neighbors = [Int[] for _ in 1:nt]
    for tet in 1:nt, omitted in 1:4
        face = Tuple(sort([tet_nodes[j, tet] for j in 1:4 if j != omitted]))
        first_owner, second_owner = get(owners, face, (0, 0))
        if first_owner == 0
            owners[face] = (tet, 0)
        else
            second_owner == 0 || error("Nonmanifold tetrahedral face $face")
            owners[face] = (first_owner, tet)
            push!(neighbors[first_owner], tet)
            push!(neighbors[tet], first_owner)
        end
    end
    boundary_triangles = Set(face for (face, pair) in owners if pair[2] == 0)
    reached = falses(nt)
    reached[1] = true
    queue = [1]
    cursor = 1
    while cursor <= length(queue)
        tet = queue[cursor]
        cursor += 1
        for neighbor in neighbors[tet]
            reached[neighbor] && continue
            reached[neighbor] = true
            push!(queue, neighbor)
        end
    end
    all(reached) || error("The tetrahedral mesh is not connected through faces")
    return (; boundary_triangles, interior_faces=length(owners)-length(boundary_triangles),
        total_faces=length(owners), face_connected=true)
end

"""Rebuild incidence without assuming that tetrahedra are grouped by parent cell."""
function poisson_refinement_incidence(tet_nodes, tet_cells, number_of_nodes)
    stars = [Int[] for _ in 1:number_of_nodes]
    node_cells = [Int[] for _ in 1:number_of_nodes]
    for tet in axes(tet_nodes, 2), node in view(tet_nodes, :, tet)
        push!(stars[node], tet)
        push!(node_cells[node], tet_cells[tet])
    end
    for cells in node_cells
        sort!(cells)
        unique!(cells)
    end
    return (; stars, node_cells)
end

"""Compute geometry from the actual tetrahedra and integrate parent-cell moments."""
function poisson_refinement_geometry(points, tet_nodes, tet_cells, number_of_cells)
    nt = size(tet_nodes, 2)
    gradients = Vector{Matrix{Float64}}(undef, nt)
    volumes = zeros(nt)
    determinants = zeros(nt)
    cell_volumes = zeros(number_of_cells)
    cell_centers = zeros(3, number_of_cells)
    minimum_scaled_jacobian = Inf
    for tet in 1:nt
        ids = view(tet_nodes, :, tet)
        xyz = points[:, ids]
        J = xyz[:, 2:4] .- xyz[:, 1]
        determinants[tet] = det(J)
        scaled = abs(determinants[tet]) / prod(norm(view(J, :, j)) for j in 1:3)
        isfinite(scaled) && scaled > 0 || error("Invalid refined tetrahedron $tet")
        minimum_scaled_jacobian = min(minimum_scaled_jacobian, scaled)
        gradients[tet], volumes[tet] = poisson_tetra_geometry(points, ids)
        cell = tet_cells[tet]
        cell_volumes[cell] += volumes[tet]
        cell_centers[:, cell] .+= volumes[tet] .* vec(sum(xyz; dims=2)) ./ 4
    end
    all(>(0), cell_volumes) || error("An empty reporting cell was produced")
    cell_centers ./= transpose(cell_volumes)
    return (; gradients, volumes, determinants, cell_volumes, cell_centers,
        minimum_scaled_jacobian)
end

"""Split every `stride`-th tetrahedron into four using its interior barycenter.

Input is the named tuple returned by `irregular_tetrahedral_mesh`. All old
vertices and all exterior triangles remain unchanged, so the new mesh is
conforming without hanging interfaces. This changes tetrahedral topology and
star valences; it does not establish validity of any polyhedral VEM method.

Parent-cell labels, volumes, centers, and the original fan categories
`vertex_count`, `face_count`, and `cell_nodes` are retained. New interior points
are listed in `fan.refined_nodes`; generic assembly must use `points`, `stars`,
and `node_cells` rather than assume the original categories exhaust all nodes.

`tet_faces` is inherited solely as parent-face provenance. After refinement it
is NOT an identifier for an actual child boundary face. Physical boundary
triangles are checked from tetrahedral vertex triples. `boundary_edges` remains
unchanged. `refinement_parent_tetrahedra` refers to the immediate input mesh.

The original coordinate-perturbation orientation statistic is retained;
`minimum_child_orientation_ratio` separately checks the new child/parent map.
"""
function locally_refined_tetrahedral_mesh(generated; stride::Integer=7)
    stride > 0 || throw(ArgumentError("Refinement stride must be positive"))
    mesh, source = generated.mesh, generated.fan
    nt, nn = length(source.volumes), size(source.points, 2)
    selected = collect(stride:stride:nt)
    added = length(selected)
    points = Matrix{Float64}(undef, 3, nn + added)
    points[:, 1:nn] .= source.points
    tet_nodes = Matrix{Int}(undef, 4, nt + 3added)
    tet_cells = Vector{Int}(undef, nt + 3added)
    tet_faces = similar(tet_cells)
    refinement_parent_tetrahedra = similar(tet_cells)
    original_determinants = zeros(nt)
    index, new_node = 0, nn
    for tet in 1:nt
        ids = view(source.tet_nodes, :, tet)
        xyz = source.points[:, ids]
        original_determinants[tet] = det(xyz[:, 2:4] .- xyz[:, 1])
        if tet % stride == 0
            new_node += 1
            points[:, new_node] .= vec(sum(xyz; dims=2)) ./ 4
            for replaced in 1:4
                index += 1
                # Replacing a vertex in place preserves the parent's orientation.
                tet_nodes[:, index] .= ntuple(j -> j == replaced ? new_node : ids[j], 4)
                tet_cells[index] = source.tet_cells[tet]
                tet_faces[index] = source.tet_faces[tet]
                refinement_parent_tetrahedra[index] = tet
            end
        else
            index += 1
            tet_nodes[:, index] .= ids
            tet_cells[index] = source.tet_cells[tet]
            tet_faces[index] = source.tet_faces[tet]
            refinement_parent_tetrahedra[index] = tet
        end
    end
    index == size(tet_nodes, 2) && new_node == size(points, 2) ||
        error("Refinement count mismatch")

    geometry = poisson_refinement_geometry(points, tet_nodes, tet_cells, length(mesh.cells))
    accumulated_parent_volumes = zeros(nt)
    minimum_child_orientation_ratio = 1.0
    for tet in eachindex(geometry.volumes)
        parent = refinement_parent_tetrahedra[tet]
        ratio = geometry.determinants[tet] / original_determinants[parent]
        expected_ratio = parent % stride == 0 ? 0.25 : 1.0
        isfinite(ratio) && ratio > 0 || error("Refinement inverted tetrahedron $tet")
        isapprox(ratio, expected_ratio; rtol=5e-12, atol=0.0) ||
            error("Unexpected child geometry in tetrahedron $tet")
        minimum_child_orientation_ratio = min(minimum_child_orientation_ratio, ratio)
        accumulated_parent_volumes[parent] += geometry.volumes[tet]
    end
    for tet in 1:nt
        isapprox(accumulated_parent_volumes[tet], source.volumes[tet]; rtol=5e-12, atol=0.0) ||
            error("Refinement changed parent tetrahedron $tet volume")
    end
    for cell in eachindex(mesh.volumes)
        isapprox(geometry.cell_volumes[cell], mesh.volumes[cell]; rtol=5e-12, atol=0.0) ||
            error("Refinement changed reporting-cell $cell volume")
        norm(geometry.cell_centers[:, cell] - mesh.centers[:, cell]) <=
            5e-12 * max(1.0, norm(mesh.centers[:, cell])) ||
            error("Refinement changed reporting-cell $cell center")
    end
    isapprox(sum(geometry.volumes), generated.geometry_stats.expected_volume;
        rtol=5e-12, atol=0.0) || error("Refinement changed domain volume")

    old_faces = poisson_refinement_face_audit(source.tet_nodes)
    new_faces = poisson_refinement_face_audit(tet_nodes)
    new_faces.boundary_triangles == old_faces.boundary_triangles ||
        error("Refinement changed exterior triangles")
    new_faces.interior_faces == old_faces.interior_faces + 6added ||
        error("Unexpected number of interior faces after refinement")
    boundary_nodes = falses(size(points, 2))
    actual_boundary_edges = Set{Tuple{Int,Int}}()
    for (a, b, c) in new_faces.boundary_triangles
        boundary_nodes[[a, b, c]] .= true
        push!(actual_boundary_edges, minmax(a, b), minmax(b, c), minmax(c, a))
    end
    boundary_nodes[1:nn] == source.boundary_nodes || error("Physical boundary node labels changed")
    any(view(boundary_nodes, nn+1:length(boundary_nodes))) &&
        error("An added barycenter lies on the physical boundary")
    actual_boundary_edges == source.boundary_edges || error("Physical boundary edges changed")
    points[:, 1:nn] == source.points || error("An existing vertex moved")

    incidence = poisson_refinement_incidence(tet_nodes, tet_cells, size(points, 2))
    previous_nodes = hasproperty(source, :refined_nodes) ? source.refined_nodes : Int[]
    refined_nodes = vcat(previous_nodes, collect(nn+1:size(points, 2)))
    fan = merge(source, (; points, tet_nodes, tet_cells, tet_faces,
        gradients=geometry.gradients, volumes=geometry.volumes,
        stars=incidence.stars, node_cells=incidence.node_cells, boundary_nodes,
        boundary_edges=copy(source.boundary_edges), refined_nodes,
        refinement_parent_tetrahedra))
    geometry_stats = merge(generated.geometry_stats, (; tetrahedra=size(tet_nodes, 2),
        vertices=size(points, 2), moved_vertices=count(!, boundary_nodes),
        minimum_scaled_jacobian=geometry.minimum_scaled_jacobian,
        domain_volume=sum(geometry.volumes), refinement_stride=Int(stride),
        input_tetrahedra=nt, refined_tetrahedra=added, added_vertices=added,
        minimum_child_orientation_ratio, face_connected=new_faces.face_connected,
        interior_triangles=new_faces.interior_faces,
        boundary_triangles=length(new_faces.boundary_triangles),
        maximum_parent_volume_relative_error=maximum(abs.(geometry.cell_volumes-mesh.volumes) ./ mesh.volumes),
        maximum_parent_center_error=maximum(norm(geometry.cell_centers[:, cell]-mesh.centers[:, cell])
            for cell in eachindex(mesh.volumes))))
    return (; mesh, fan, geometry_stats)
end
