using LinearAlgebra

function sdwr_goal_box_tolerance(vertices,box)
    size(vertices) == (3,4) || throw(DimensionMismatch("tetrahedron vertices"))
    length(box) == 3 || throw(DimensionMismatch("box bounds"))
    all(isfinite,vertices) || throw(ArgumentError("finite tetrahedron required"))
    all(bounds -> length(bounds)==2 && all(isfinite,bounds) && bounds[1]<bounds[2],box) ||
        throw(ArgumentError("box bounds must be finite and strictly increasing"))
    scale = max(maximum(abs,vertices),maximum(abs(b) for bounds in box for b in bounds),1.0)
    return 64eps(Float64)*scale
end

"""Positive four-point tetrahedral rule, exact for total degree two."""
function sdwr_quadratic_tetra_barycentric()
    small = (5-sqrt(5.0))/20
    large = (5+3sqrt(5.0))/20
    result = fill(small,4,4)
    for i in 1:4
        result[i,i] = large
    end
    return result
end

"""Remove repeated barycentric polygon vertices without changing their order."""
function sdwr_unique_cut_vertices(points; tolerance=2e-12)
    result = Vector{Float64}[]
    for point in points
        any(other -> maximum(abs.(point-other)) <= tolerance, result) && continue
        push!(result,point)
    end
    return result
end

"""Clip one convex face against a Cartesian half-space in physical coordinates."""
function sdwr_clip_goal_face(face, vertices, axis, bound, side, tolerance)
    result = Vector{Float64}[]
    previous = last(face)
    previous_distance = side*(dot(view(vertices,axis,:),previous)-bound)
    abs(previous_distance) <= tolerance && (previous_distance=0.0)
    for current in face
        distance = side*(dot(view(vertices,axis,:),current)-bound)
        abs(distance) <= tolerance && (distance=0.0)
        if (previous_distance >= 0) != (distance >= 0)
            fraction = previous_distance/(previous_distance-distance)
            push!(result,(1-fraction).*previous+fraction.*current)
        end
        distance >= 0 && push!(result,current)
        previous,previous_distance = current,distance
    end
    return sdwr_unique_cut_vertices(result)
end

"""Order a clipping cap in its Cartesian plane; orientation is immaterial."""
function sdwr_order_goal_cap(points, vertices, axis)
    axes = [d for d in 1:3 if d != axis]
    first_axis,second_axis = axes
    first_coordinates = [dot(view(vertices,first_axis,:),point) for point in points]
    second_coordinates = [dot(view(vertices,second_axis,:),point) for point in points]
    center1,center2 = sum(first_coordinates)/length(points),sum(second_coordinates)/length(points)
    order = sortperm(eachindex(points);by=i->atan(second_coordinates[i]-center2,
                                                first_coordinates[i]-center1))
    return points[order]
end

"""Convex tetrahedron/box intersection represented by barycentric face polygons.

The coordinates stay barycentric in the original tetrahedron, so any original
P1 field and hat can be evaluated on the clipped pieces without reconstruction.
Faces retain cyclic vertex order and each new clipping plane adds one cap.
"""
function sdwr_clip_tetra_box(vertices, box)
    tolerance = sdwr_goal_box_tolerance(vertices,box)
    basis = [Float64[i==j for i in 1:4] for j in 1:4]
    faces = [[basis[i] for i in face] for face in ((2,3,4),(1,4,3),(1,2,4),(1,3,2))]
    cut = false
    for axis in 1:3, (bound,side) in ((box[axis][1],1.0),(box[axis][2],-1.0))
        points = sdwr_unique_cut_vertices(vcat(faces...))
        distances = [side*(dot(view(vertices,axis,:),point)-bound) for point in points]
        minimum(distances) >= -tolerance && continue
        maximum(distances) < -tolerance && return (;faces=Vector{Vector{Float64}}[],cut=true,tolerance)
        cut = true
        clipped = Vector{Vector{Float64}}[]
        for face in faces
            polygon = sdwr_clip_goal_face(face,vertices,axis,bound,side,tolerance)
            length(polygon) >= 3 && push!(clipped,polygon)
        end
        isempty(clipped) && return (;faces=clipped,cut=true,tolerance)
        plane_points = Vector{Float64}[]
        for face in clipped,point in face
            abs(dot(view(vertices,axis,:),point)-bound) <= 2tolerance && push!(plane_points,point)
        end
        cap = sdwr_unique_cut_vertices(plane_points)
        length(cap) >= 3 && push!(clipped,sdwr_order_goal_cap(cap,vertices,axis))
        faces = clipped
    end
    return (;faces,cut,tolerance)
end

"""Positive physical quadrature for a tetrahedron clipped by an axis-aligned box.

Returns barycentric coordinates relative to the ORIGINAL tetrahedron and
physical weights. The rule integrates every polynomial of degree <=2 exactly.
An empty or tangent intersection gives a 4-by-0 matrix and no weights.
"""
function sdwr_cut_tetra_rule(vertices, box)
    tolerance = sdwr_goal_box_tolerance(vertices,box)
    original_volume = abs(det(vertices[:,2:4].-vertices[:,1]))/6
    original_volume > 0 || throw(ArgumentError("degenerate tetrahedron"))
    empty_rule = (;barycentric=zeros(4,0),weights=Float64[],volume=0.0,cut=true)
    fully_inside = true
    for axis in 1:3
        lo,hi = extrema(view(vertices,axis,:))
        (hi < box[axis][1]-tolerance || lo > box[axis][2]+tolerance) && return empty_rule
        fully_inside &= lo >= box[axis][1]-tolerance && hi <= box[axis][2]+tolerance
    end
    basic = sdwr_quadratic_tetra_barycentric()
    fully_inside && return (;barycentric=basic,weights=fill(original_volume/4,4),
        volume=original_volume,cut=false)
    clipped = sdwr_clip_tetra_box(vertices,box)
    isempty(clipped.faces) && return empty_rule
    !clipped.cut && return (;barycentric=basic,weights=fill(original_volume/4,4),
        volume=original_volume,cut=false)
    points = sdwr_unique_cut_vertices(vcat(clipped.faces...))
    length(points) >= 4 || return empty_rule
    center = sum(points)/length(points)
    barycentric_columns = Vector{Float64}[]
    weights = Float64[]
    for face in clipped.faces, i in 2:length(face)-1
        sub_barycentric = hcat(center,first(face),face[i],face[i+1])
        sub_vertices = vertices*sub_barycentric
        volume = abs(det(sub_vertices[:,2:4].-sub_vertices[:,1]))/6
        volume > 0 || continue
        quadrature = sub_barycentric*basic
        for q in 1:4
            push!(barycentric_columns,quadrature[:,q])
            push!(weights,volume/4)
        end
    end
    isempty(weights) && return empty_rule
    barycentric = reduce(hcat,barycentric_columns)
    volume = sum(weights)
    return (;barycentric,weights,volume,cut=true)
end

"""Cut rule for the localized LINEAR goal; quadratic PU goals need degree three.

This helper is deliberately restricted to the current localized integral.
Other goals keep the benchmark's full-domain quadrature.
"""
function sdwr_goal_tetra_rule(vertices,goal)
    goal.name == :localized_u && goal.power == 1 ||
        throw(ArgumentError("cut rules are for the localized linear goal"))
    return sdwr_cut_tetra_rule(vertices,goal.box)
end

"""Build cut rules once for a fan and its localized linear goal."""
function sdwr_cut_goal_rules(fan,goal)
    return [sdwr_goal_tetra_rule(fan.points[:,fan.tet_nodes[:,tet]],goal)
        for tet in eachindex(fan.volumes)]
end
