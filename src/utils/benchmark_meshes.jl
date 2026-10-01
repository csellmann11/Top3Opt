# Clip a convex polygon against an axis-aligned half-plane. Intersections are
# snapped to the cutting line so adjacent cells share exactly the same boundary.
function clip_mesh_polygon(points, axis, cut, keep_lower, tol)
    result = SVector{2,Float64}[]
    isempty(points) && return result
    previous = last(points)
    previous_inside = keep_lower ? previous[axis] <= cut : previous[axis] >= cut
    for point in points
        inside = keep_lower ? point[axis] <= cut : point[axis] >= cut
        if inside != previous_inside
            t = (cut - previous[axis]) / (point[axis] - previous[axis])
            intersection = previous + t * (point - previous)
            push!(result, setindex(intersection, cut, axis))
        end
        inside && push!(result, point)
        previous, previous_inside = point, inside
    end
    # A vertex on the cut can be emitted twice when its neighbour is outside.
    polygon = SVector{2,Float64}[]
    for point in result
        (isempty(polygon) || norm(point - last(polygon)) > tol) && push!(polygon, point)
    end
    if length(polygon) > 1 && norm(first(polygon) - last(polygon)) <= tol
        pop!(polygon)
    end
    return polygon
end

"""
    create_L_voronoi_mesh(left, right, nx, ny, ElT, smooth=true; rel_x=0.5, rel_y=0.5)

Create a Voronoi-derived polygon mesh of a rectangle with its upper-right
corner removed. The notch starts at `left + (rel_x, rel_y) .* (right-left)`.
Uses the same deterministic seeds, smoothing and short-edge cleanup as the
rectangular benchmark. Cells crossing the horizontal notch line are split into
convex pieces so extrusion and VEM refinement retain convex cells.
"""
function create_L_voronoi_mesh(left::Tuple{Float64,Float64},
    right::Tuple{Float64,Float64}, nx::Int, ny::Int,
    ::Type{ElT}, smooth::Bool=true; rel_x=0.5, rel_y=0.5) where {ElT<:ElType}

    all(right[i] > left[i] for i in 1:2) || throw(ArgumentError("right must be above and right of left"))
    nx > 0 && ny > 0 || throw(ArgumentError("nx and ny must be positive"))
    0 < rel_x < 1 && 0 < rel_y < 1 || throw(ArgumentError("notch fractions must lie strictly between zero and one"))
    cut_x = left[1] + rel_x * (right[1] - left[1])
    cut_y = left[2] + rel_y * (right[2] - left[2])
    tol = 1e-10 * max(right[1] - left[1], right[2] - left[2])

    source = create_voronoi_mesh(left, right, nx, ny, ElT, smooth).topo
    for _ in 1:3
        source = remove_short_edges(source)
    end

    topo = Topology{2}()
    node_buckets = Dict{Tuple{Int,Int},Vector{Int}}()
    function shared_node(point)
        key = Tuple(round.(Int, (point - SVector(left)) / tol))
        # Check neighbouring buckets too: rounding can put two evaluations of
        # the same edge intersection on opposite sides of a bucket boundary.
        for dx in -1:1, dy in -1:1
            for id in get(node_buckets, (key[1]+dx, key[2]+dy), Int[])
                norm(topo.nodes[id].coords - point) <= tol && return id
            end
        end
        id = add_node!(point, topo)
        push!(get!(node_buckets, key, Int[]), id)
        return id
    end

    function add_polygon(points)
        length(points) >= 3 || return
        # Compute area relative to a vertex to avoid cancellation on translated domains.
        origin = first(points)
        area2 = sum(eachindex(points)) do i
            a = points[i] - origin
            b = points[mod1(i+1, length(points))] - origin
            a[1]*b[2] - a[2]*b[1]
        end
        abs(area2) > tol^2 || return
        area2 < 0 && reverse!(points)
        ids = shared_node.(points)
        length(unique(ids)) == length(ids) || error("Degenerate clipped Voronoi cell")
        add_area!(ids, topo)
    end

    for face in RootIterator{3}(source)
        points = [source.nodes[id].coords for id in get_area_node_ids(source, face.id)]
        bottom = clip_mesh_polygon(points, 2, cut_y, true, tol)
        upper = clip_mesh_polygon(points, 2, cut_y, false, tol)
        upper_left = clip_mesh_polygon(upper, 1, cut_x, true, tol)

        # The upper-left clip introduces the reentrant corner on the seam.
        # Split the matching bottom edge there too, avoiding a T-junction.
        for i in eachindex(bottom)
            a, b = bottom[i], bottom[mod1(i+1, length(bottom))]
            if abs(a[2]-cut_y) <= tol && abs(b[2]-cut_y) <= tol &&
               min(a[1],b[1]) + tol < cut_x < max(a[1],b[1]) - tol
                insert!(bottom, i+1, SVector(cut_x, cut_y))
                break
            end
        end
        add_polygon(bottom)
        add_polygon(upper_left)
    end

    # Do not collapse short edges after clipping: that can move the notch or
    # delete the corner nodes used by the cantilever's supports.
    return Mesh(topo, ElT())
end

"""Build the benchmark geometry, honouring the requested Voronoi mesh family."""
function create_benchmark_mesh(b_case::Symbol, mesh_type::Symbol,
    lx, ly, lz, nx::Int, ny::Int, nz::Int, ::Type{ElT}) where {ElT<:ElType}

    if b_case == :L_cantilever
        mesh2d = if mesh_type == :Voronoi
            create_L_voronoi_mesh((0.0, 0.0), (lx, ly), nx, ny, ElT)
        elseif mesh_type in (:Hexahedra, :Lquad_mesh)
            create_L_mesh((0.0, 0.0), (lx, ly), nx, ny, ElT(), 0.5, 0.5)
        else
            error("Invalid MeshType: $mesh_type")
        end
        return extrude_to_3d(nz, mesh2d, lz)
    elseif mesh_type == :Hexahedra
        return create_rectangular_mesh(nx, ny, nz, lx, ly, lz, ElT)
    elseif mesh_type == :Voronoi
        mesh2d = create_voronoi_mesh((0.0, 0.0), (lx, lz), nx, nz, ElT)
        topo = mesh2d.topo
        for _ in 1:3
            topo = remove_short_edges(topo)
        end
        mesh = extrude_to_3d(ny, Mesh(topo, ElT()), ly)
        dim_permute = b_case == :pressure_plate ? SA[1,2,3] : SA[1,3,2]
        return permute_coord_dimensions(mesh, dim_permute)
    elseif mesh_type == :Lquad_mesh
        mesh2d = create_L_mesh((0.0, 0.0), (2.0, 2.0), nx, ny, ElT(), 0.5, 0.5)
        return extrude_to_3d(nz, mesh2d, lz)
    else
        error("Invalid MeshType: $mesh_type")
    end
end
