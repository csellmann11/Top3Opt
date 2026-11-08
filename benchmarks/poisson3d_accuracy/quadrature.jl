using LinearAlgebra

"""Gauss–Legendre nodes and weights on [0,1], using only the standard library."""
function gauss_legendre01(order::Integer)
    order >= 1 || throw(ArgumentError("Gauss–Legendre order must be positive"))
    n = Int(order)
    offdiagonal = [i/sqrt(4.0i*i-1.0) for i in 1:n-1]
    decomposition = eigen(SymTridiagonal(zeros(n),offdiagonal))
    nodes = (decomposition.values .+ 1.0)./2.0
    weights = vec(decomposition.vectors[1,:].^2)
    return (;nodes,weights)
end

"""Tensor Gauss rule mapped to a tetrahedron by the Duffy transformation.

`order` is the number of Gauss points per cube direction, not a polynomial
degree. It must be at least two to integrate the Jacobian a²b exactly. Columns
of `barycentric` are (1-a,a(1-b),ab(1-c),abc); weights sum to one. The returned
rule can be reused for every tetrahedron with the same integration order.
"""
function tetra_rule(order::Integer)
    order >= 2 || throw(ArgumentError("Tensor tetrahedral quadrature requires order >= 2"))
    nodes,weights1d = gauss_legendre01(order)
    n = length(nodes)
    barycentric = Matrix{Float64}(undef,4,n^3)
    weights = Vector{Float64}(undef,n^3)
    q = 0
    @inbounds for i in 1:n,j in 1:n,k in 1:n
        a,b,c = nodes[i],nodes[j],nodes[k]
        q += 1
        barycentric[1,q] = 1-a
        barycentric[2,q] = a*(1-b)
        barycentric[3,q] = a*b*(1-c)
        barycentric[4,q] = a*b*c
        weights[q] = 6a*a*b*weights1d[i]*weights1d[j]*weights1d[k]
    end
    # Remove roundoff in the normalization, not a quadrature-order defect.
    weights ./= sum(weights)
    return (;barycentric,weights)
end

"""Physical points and integration weights for a 3×4 vertex matrix."""
function tetra_quadrature(vertices::AbstractMatrix,rule::NamedTuple)
    size(vertices) == (3,4) || throw(DimensionMismatch("Expected a 3-by-4 tetrahedron vertex matrix"))
    all(isfinite,vertices) || throw(ArgumentError("Tetrahedron coordinates must be finite"))
    points = Matrix{Float64}(vertices)
    edges = points[:,2:4] .- points[:,1]
    volume = abs(det(edges))/6
    isfinite(volume) && volume > 0 || throw(ArgumentError("Degenerate tetrahedron"))
    return (;points=points*rule.barycentric,
        weights=volume.*rule.weights,barycentric=rule.barycentric)
end

tetra_quadrature(vertices::AbstractMatrix,order::Integer) =
    tetra_quadrature(vertices,tetra_rule(order))
