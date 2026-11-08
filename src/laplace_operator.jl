using LinearAlgebra, SparseArrays

isdefined(@__MODULE__, :TPFAFaces) || include("regularization/load.jl")

"""Positive normal distances of two cell centers on opposite sides of a face.
The orientation of the supplied normal does not matter.
"""
function tpfa_face_distances(point, normal, left_center, right_center)
    magnitude = norm(normal)
    isfinite(magnitude) && magnitude > 0 || throw(ArgumentError("Invalid TPFA face normal"))
    left = dot(point-left_center,normal)/magnitude
    right = dot(right_center-point,normal)/magnitude
    isfinite(left) && isfinite(right) && left != 0 && right != 0 &&
        signbit(left) == signbit(right) ||
        throw(ArgumentError("TPFA cell centers must lie on opposite sides of their face"))
    return abs(left), abs(right)
end

"""
    build_tpfa_faces(cv, states)

Collect each active interior leaf face once, including every hanging subface.
Geometry uses the mesh's cached face areas and planes; only a scalar owner map
is needed, with no vertex reconstruction or extra spatial neighbor search.
Boundary faces are omitted to impose zero normal flux.

The dimension-independent TPFA storage calls cell measures `area` and face
measures `length`; in 3D these contain cell volumes and face areas respectively.
Call after mesh changes, before constructing or rebuilding a density cache.
"""
function build_tpfa_faces(cv, states)
    volumes = states.area_vec
    !isempty(volumes) && all(v -> isfinite(v) && v > 0,volumes) ||
        throw(ArgumentError("TPFA needs finite positive cell volumes"))
    owners = Dict{Int,Int}()
    sizehint!(owners,length(cv.facedata_col))
    left = Int[]; right = Int[]
    areas = Float64[]; distance_left = Float64[]; distance_right = Float64[]
    for vector in (left,right,areas,distance_left,distance_right)
        sizehint!(vector,3length(volumes))
    end
    for (element_id,sid) in states.el_id_to_state_id
        Ju3VEM.VEMGeo.iterate_volume_areas(cv.facedata_col,cv.mesh.topo,element_id) do face,fd,_
            previous = get(owners,face.id,0)
            if previous == 0
                owners[face.id] = sid
            else
                previous > 0 && previous != sid ||
                    error("TPFA face $(face.id) has repeated or more than two incident cells")
                owners[face.id] = -sid
                area = Ju3VEM.VEMGeo.get_area(fd)
                isfinite(area) && area > 0 || error("Invalid TPFA face area at face $(face.id)")
                dl,dr = tpfa_face_distances(fd.dΩ.plane.p0,fd.dΩ.plane.n,
                    states.x_vec[previous],states.x_vec[sid])
                push!(left,previous); push!(right,sid); push!(areas,area)
                push!(distance_left,dl); push!(distance_right,dr)
            end
        end
    end
    return TPFAFaces(volumes,left,right,areas,distance_left,distance_right)
end

"""Face ownership for mesh diagnostics, including single-owner boundary faces."""
function regularization_faces(cv, states)
    owners = Dict{Int,Vector{Int}}()
    for (element_id,sid) in states.el_id_to_state_id
        Ju3VEM.VEMGeo.iterate_volume_areas(cv.facedata_col,cv.mesh.topo,element_id) do face,_,_
            cells = get!(owners,face.id) do
                Int[]
            end
            push!(cells,sid)
            length(cells) <= 2 || error("Nonmanifold regularization face $(face.id)")
        end
    end
    return owners
end

"""
    compute_flux_operator_mat(cv, states, sim_pars; scheme=:tpfa)

Compatibility assembly for the conservative strong-form operator `-M^-1 K`,
with the existing 3D baseline coefficient `2*sim_pars.β0*h_i^2`.
The optimization kernels retain `K` and update its values in place instead.
TPFA is the only supported gradient approximation.
"""
function compute_flux_operator_mat(cv, states, sim_pars; scheme::Symbol=:tpfa)
    scheme == :tpfa || throw(ArgumentError("Only flux_scheme=:tpfa is supported"))
    isfinite(sim_pars.β0) && sim_pars.β0 >= 0 ||
        throw(ArgumentError("beta0 must be finite and nonnegative"))
    faces = build_tpfa_faces(cv,states)
    n = length(faces.area)
    sim_pars.β0 == 0 && return spzeros(n,n)
    operator = diffusion_matrix(faces,2sim_pars.β0 .* states.h_vec.^2)
    rows,values = rowvals(operator),nonzeros(operator)
    for k in eachindex(values)
        values[k] = -values[k]/faces.area[rows[k]]
    end
    return operator
end
