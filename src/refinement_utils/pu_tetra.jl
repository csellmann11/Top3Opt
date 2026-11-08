using LinearAlgebra, StaticArrays

const PU3D_P2_EDGES = ((1,2),(1,3),(1,4),(2,3),(2,4),(3,4))
const PU3D_FREE = SVector(false,false,false)
const PU3D_FIXED = SVector(true,true,true)

@inline pu3d_strain(v,g) = (v*transpose(g)+g*transpose(v))/2
@inline pu3d_stress(strain,lambda,mu) = lambda*tr(strain)*one(strain)+2mu*strain

@inline function pu3d_p2_values(N)
    SVector{10,Float64}(ntuple(Val(10)) do i
        i <= 4 && return N[i]*(2N[i]-1)
        a,b = PU3D_P2_EDGES[i-4]
        return 4N[a]*N[b]
    end)
end

@inline function pu3d_p2_gradients(t,N)
    SVector{10,SVector{3,Float64}}(ntuple(Val(10)) do i
        i <= 4 && return (4N[i]-1)*t.gradients[i]
        a,b = PU3D_P2_EDGES[i-4]
        return 4(N[a]*t.gradients[b]+N[b]*t.gradients[a])
    end)
end

# Isotropic strain contraction without a 6x30 B matrix or a 30x30 tetra matrix.
@inline pu3d_elastic_entry(g,h,c,d,lambda,mu) =
    lambda*g[c]*h[d]+mu*((c == d)*dot(g,h)+g[d]*h[c])

"""Exact integral divided by tetra volume of a P2 elastic stiffness entry."""
@inline function pu3d_stiffness_entry(t,i,j,c,d,lambda,mu)
    g = t.gradients
    if i <= 4 && j <= 4
        return ((4(i == j)-1)/5)*pu3d_elastic_entry(g[i],g[j],c,d,lambda,mu)
    elseif i <= 4
        a,b = PU3D_P2_EDGES[j-4]
        return ((4(i == a)-1)*pu3d_elastic_entry(g[i],g[b],c,d,lambda,mu)+
                (4(i == b)-1)*pu3d_elastic_entry(g[i],g[a],c,d,lambda,mu))/5
    elseif j <= 4
        a,b = PU3D_P2_EDGES[i-4]
        return ((4(j == a)-1)*pu3d_elastic_entry(g[b],g[j],c,d,lambda,mu)+
                (4(j == b)-1)*pu3d_elastic_entry(g[a],g[j],c,d,lambda,mu))/5
    end
    a,b = PU3D_P2_EDGES[i-4]
    k,l = PU3D_P2_EDGES[j-4]
    return 0.8*((1+(a == k))*pu3d_elastic_entry(g[b],g[l],c,d,lambda,mu)+
                (1+(a == l))*pu3d_elastic_entry(g[b],g[k],c,d,lambda,mu)+
                (1+(b == k))*pu3d_elastic_entry(g[a],g[l],c,d,lambda,mu)+
                (1+(b == l))*pu3d_elastic_entry(g[a],g[k],c,d,lambda,mu))
end

"""Exact mean gradients of phi_i and of the product N_hat*phi_i in 3D."""
@inline function pu3d_internal_gradients(t,i,hat)
    g = t.gradients
    if i <= 4
        return zero(g[1]), ((4(i == hat)-1)*g[i]-g[hat])/20
    end
    a,b = PU3D_P2_EDGES[i-4]
    return g[a]+g[b], ((1+(hat == a))*g[b]+(1+(hat == b))*g[a]+g[hat])/5
end

"""Exact Cholesky solve on the active lower triangle, without copying its storage.

Small patches avoid library-call overhead; larger patches use the optimized
LAPACK factorization with the workspace's actual leading dimension.
"""
function pu3d_solve!(matrix,rhs,n)
    if n >= 32
        active_matrix = @view matrix[1:n,1:n]
        active_rhs = @view rhs[1:n]
        _,info = LAPACK.potrf!('L',active_matrix)
        info == 0 || throw(PosDefException(info))
        # Preserve the small kernel's explicit rejection of nonfinite pivots.
        @inbounds for i in 1:n
            isfinite(matrix[i,i]) && matrix[i,i] > 0 || throw(PosDefException(i))
        end
        LAPACK.potrs!('L',active_matrix,active_rhs)
        return nothing
    end
    @inbounds for j in 1:n
        diagonal = matrix[j,j]
        for k in 1:j-1
            diagonal -= matrix[j,k]^2
        end
        isfinite(diagonal) && diagonal > 0 || throw(PosDefException(j))
        matrix[j,j] = sqrt(diagonal)
        for i in j+1:n
            value = matrix[i,j]
            for k in 1:j-1
                value -= matrix[i,k]*matrix[j,k]
            end
            matrix[i,j] = value/matrix[j,j]
        end
    end
    @inbounds for i in 1:n
        value = rhs[i]
        for j in 1:i-1
            value -= matrix[i,j]*rhs[j]
        end
        rhs[i] = value/matrix[i,i]
    end
    @inbounds for i in n:-1:1
        value = rhs[i]
        for j in i+1:n
            value -= matrix[j,i]*rhs[j]
        end
        rhs[i] = value/matrix[i,i]
    end
    return nothing
end
