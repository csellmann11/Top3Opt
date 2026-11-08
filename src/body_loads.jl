using StaticArrays, LinearAlgebra

"""Explicit zero body force, allowing mechanics and PU setup to skip quadrature."""
struct ZeroBodyForce <: Function end
@inline (::ZeroBodyForce)(x) = zero(SVector{3,Float64})
@inline pu3d_body_iszero(f) = false
@inline pu3d_body_iszero(::ZeroBodyForce) = true
@inline pu3d_body_iszero(::Nothing) = true

# Degree-four tetrahedral rule, normalized to unit physical volume.
const PU3D_TETRA_POINTS = let
    a,b = 0.3994035761667992,0.1005964238332008
    points = SVector{4,Float64}[SA[0.25,0.25,0.25,0.25]]
    for i in 1:4
        push!(points,SVector{4,Float64}(ntuple(j -> i == j ? 11/14 : 1/14,4)))
    end
    for i in 1:3,j in i+1:4
        push!(points,SVector{4,Float64}(ntuple(k -> k == i || k == j ? a : b,4)))
    end
    Tuple(points)
end
const PU3D_TETRA_WEIGHTS = (-0.07893333333333333,
    0.04573333333333333,0.04573333333333333,0.04573333333333333,0.04573333333333333,
    0.14933333333333333,0.14933333333333333,0.14933333333333333,
    0.14933333333333333,0.14933333333333333,0.14933333333333333)

@inline function pu3d_body_moment(point,value,bc,h,measure)
    x = (point-bc)/h
    return (measure*value)*transpose(SA[1.0,x[1],x[2],x[3]])
end

function pu3d_body_load!(rhs,cv,element_id,f,projection)
    fill!(rhs,0.0)
    pu3d_body_iszero(f) && return rhs
    center = cv.volume_data.vol_bc
    h = cv.volume_data.hvol
    moments = zeros(MMatrix{3,4,Float64,12})
    Ju3VEM.VEMGeo.iterate_volume_areas(cv.facedata_col,cv.mesh.topo,element_id) do _,fd,_
        facecenter = Ju3VEM.VEMGeo.project_to_3d(fd.dΩ.bc,fd.dΩ.plane)
        ids = fd.face_node_ids.v.args[1]
        for j in eachindex(ids)
            a,b = cv.mesh.nodes[ids[j]].coords,cv.mesh.nodes[ids[mod1(j+1,length(ids))]].coords
            volume = abs(dot(a-center,cross(b-center,facecenter-center)))/6
            for (N,w) in zip(PU3D_TETRA_POINTS,PU3D_TETRA_WEIGHTS)
                point = N[1]*a+N[2]*b+N[3]*facecenter+N[4]*center
                value = SVector{3,Float64}(f(point))
                if !iszero(value)
                    addition = pu3d_body_moment(point,value,center,h,volume*w)
                    for k in eachindex(moments)
                        moments[k] += addition[k]
                    end
                end
            end
        end
    end
    @inbounds for j in axes(projection,2),component in 1:3
        value = 0.0
        for k in 1:4
            value += moments[component,k]*projection[k,j]
        end
        rhs[3j-3+component] = value
    end
    return rhs
end
