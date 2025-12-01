
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


include("../src/general_utils.jl")
include("../src/hash_output.jl")
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



const K = 1
const U = 3




n = 10
mesh = create_rectangular_mesh(
    n,1,3n,
    1.0,0.2,3.0,StandardEl{K}
)

# --- Christmas tree boundary refinement (drawn in X–Z plane; Y is extrusion) ---
# Distance from point p to segment a-b in 2D (generic helper)
function _dist_point_to_segment(px, py, ax, ay, bx, by)
    abx = bx - ax
    aby = by - ay
    apx = px - ax
    apy = py - ay
    denom = abx*abx + aby*aby
    if denom == 0.0
        return sqrt(apx*apx + apy*apy)
    end
    t = (apx*abx + apy*aby) / denom
    t = clamp(t, 0.0, 1.0)
    projx = ax + t*abx
    projy = ay + t*aby
    dx = px - projx
    dy = py - projy
    return sqrt(dx*dx + dy*dy)
end

# Build zigzag polyline points along a side from (ax,az) to (bx,bz) in X–Z
function _zigzag_side_points(ax, az, bx, bz; teeth::Int=12, amp::Float64=0.02, decay::Bool=true)
    vx = bx - ax
    vz = bz - az
    len = hypot(vx, vz)
    if len == 0.0
        return [(ax, az), (bx, bz)]
    end
    tx = vx/len
    tz = vz/len
    nx = -tz
    nz = tx
    pts = Vector{Tuple{Float64,Float64}}(undef, teeth + 2)
    pts[1] = (ax, az)
    for k in 1:teeth
        t = k/teeth
        bxk = ax + vx*t
        bzk = az + vz*t
        sgn = isodd(k) ? 1.0 : -1.0
        a = amp * len * (decay ? (1 - 0.6*t) : 1.0)
        pxk = bxk + sgn * a * nx
        pzk = bzk + sgn * a * nz
        pts[k+1] = (pxk, pzk)
    end
    pts[end] = (bx, bz)
    return pts
end

# Convert a polyline of points into segment tuples in X–Z: [(x1,z1,x2,z2), ...]
function _polyline_to_segments(pts::Vector{Tuple{Float64,Float64}})
    segs = Tuple{Float64,Float64,Float64,Float64}[]
    for i in 1:(length(pts)-1)
        (x1, z1) = pts[i]
        (x2, z2) = pts[i+1]
        push!(segs, (x1, z1, x2, z2))
    end
    return segs
end

# Return iterable of 2D segments [(ax,az,bx,bz), ...] forming a zigzag tree in X–Z
function _christmas_tree_segments(topo)
    # Domain bounds
    xmin = Inf; xmax = -Inf; zmin = Inf; zmax = -Inf
    for node in topo.nodes
        x = node.coords[1]
        z = node.coords[3]
        x < xmin && (xmin = x)
        x > xmax && (xmax = x)
        z < zmin && (zmin = z)
        z > zmax && (zmax = z)
    end
    cx = 0.5*(xmin + xmax)
    dx = xmax - xmin
    dz = zmax - zmin

    # levels: (z_base_rel, z_top_rel, base_width_rel) all relative to domain height
    levels = (
        (0.15, 0.45, 0.90),
        (0.35, 0.70, 0.65),
        (0.58, 0.92, 0.42),
    )

    teeth_per_level = (12, 10, 8)
    amp_rel = (0.06, 0.05, 0.04) # relative to side length

    segs = Tuple{Float64,Float64,Float64,Float64}[]
    for (i, (zb_rel, zt_rel, w_rel)) in enumerate(levels)
        zb = zmin + dz * zb_rel
        zt = zmin + dz * zt_rel
        w  = dx  * w_rel
        lx = cx - w/2
        rx = cx + w/2

        # left side zigzag from (lx,zb) to (cx,zt)
        lpts = _zigzag_side_points(lx, zb, cx, zt; teeth=teeth_per_level[i], amp=amp_rel[i])
        # right side zigzag from (rx,zb) to (cx,zt)
        rpts = _zigzag_side_points(rx, zb, cx, zt; teeth=teeth_per_level[i], amp=amp_rel[i])

        append!(segs, _polyline_to_segments(lpts))
        append!(segs, _polyline_to_segments(rpts))
    end

    # trunk rectangle in X–Z
    trunk_w = 0.10 * dx
    trunk_h = 0.10 * dz
    tz0 = zmin + 0.06 * dz
    tz1 = tz0 + trunk_h
    lx = cx - trunk_w/2
    rx = cx + trunk_w/2
    push!(segs, (lx, tz0, rx, tz0)) # bottom
    push!(segs, (rx, tz0, rx, tz1)) # right
    push!(segs, (rx, tz1, lx, tz1)) # top
    push!(segs, (lx, tz1, lx, tz0)) # left

    return segs
end

# Boundary indicator with thickness eps in X–Z plane (Y ignored)
function _is_on_tree_boundary_xz(x, eps, segs)
    px = x[1]; py = x[3]
    mind = Inf
    @inbounds for (ax, ay, bx, by) in segs
        d = _dist_point_to_segment(px, py, ax, ay, bx, by)
        d < mind && (mind = d)
        if mind < eps
            return true
        end
    end
    return mind < eps
end

# Refine once with a set function (re-mark each time to follow boundary)
function _refine_once_with_set(mesh::Mesh{3}, set_fun::F) where {F<:Function}
    topo = mesh.topo
    ref_marker = zeros(Bool, length(get_volumes(topo)))

    for element in RootIterator{4}(topo)
        Ju3VEM.VEMGeo.iterate_volume_areas(topo, element.id) do face, _
            Ju3VEM.VEMGeo.iterate_element_edges(topo, face.id) do n1_id, _, _
                node = topo.nodes[n1_id]
                if set_fun(node)
                    ref_marker[element.id] = true
                end
                nothing
            end
            nothing
        end
    end

    for element in RootIterator{4}(topo)
        ref_marker[element.id] && Ju3VEM.VEMGeo._refine!(element, topo)
    end
    return Mesh(topo, StandardEl{1}())
end

# Iterative boundary-following refinement (updates marking each iteration)
let mesh = mesh
    for iter in 1:3
        println("Refinement iteration: $iter")
        h = find_minimal_cell_diameter(mesh.topo)
        segs = _christmas_tree_segments(mesh.topo)
        tree_boundary_fun = x -> _is_on_tree_boundary_xz(x, 0.8*h, segs)
        mesh = _refine_once_with_set(mesh, tree_boundary_fun)
    end

    write_vtk(mesh.topo,"Results/vtk/christ_tree")
end