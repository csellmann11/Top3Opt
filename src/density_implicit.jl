using LinearAlgebra, SparseArrays
import Krylov

"""
    DensityImplicitCache(operator, eta0, beta0, h; beta_in_operator=false)

Cache for one implicit regularization step of duration one. The system is
`A = I - Rhat/eta0`, where `Rhat` is the existing regularization operator,
including `2*beta0*h^2` on each row of the legacy strong-form Laplacian.
The mechanical force remains explicit. Bounds and volume are projected after
the solve, as in AndersonPlasticity's IMEX update; this is a split update, not
a coupled bound-constrained backward-Euler solve.

The operator must remain unchanged while this cache is used. Rebuild the cache
when assembling a new operator, including mesh changes with the same cell count.
"""
struct DensityImplicitCache{Ti,Tj,W}
    operator::SparseMatrixCSC{Float64,Ti}
    eta0::Float64
    beta0::Float64
    beta_in_operator::Bool
    h::Vector{Float64}
    row_scale::Vector{Float64}
    scaled_matrix::SparseMatrixCSC{Float64,Tj}
    scale::Vector{Float64}
    scaled_rhs::Vector{Float64}
    solution::Vector{Float64}
    residual::Vector{Float64}
    workspace::W
end

function DensityImplicitCache(operator::SparseMatrixCSC{Float64}, eta0, beta0,
        h::AbstractVector; beta_in_operator::Bool=false)
    isfinite(eta0) && eta0 > 0 || throw(ArgumentError("eta0 must be finite and positive"))
    isfinite(beta0) && beta0 >= 0 || throw(ArgumentError("beta0 must be finite and nonnegative"))
    n = length(h)
    n > 0 && size(operator) == (n,n) || throw(DimensionMismatch("Density operator and cell sizes must agree"))
    all(isfinite, nonzeros(operator)) || error("Nonfinite implicit regularization operator")
    all(x -> isfinite(x) && x > 0, h) || throw(ArgumentError("Cell sizes must be finite and positive"))
    row_scale = beta_in_operator ? fill(1.0/eta0,n) : [2hi^2*beta0/eta0 for hi in h]
    all(isfinite, row_scale) || error("Nonfinite implicit regularization scaling")

    A = copy(operator)
    rows = rowvals(A)
    values = nonzeros(A)
    for col in 1:n, k in nzrange(A,col)
        values[k] *= -row_scale[rows[k]]
    end
    A += spdiagm(0 => ones(n))
    d = Vector(diag(A))
    all(x -> isfinite(x) && x > 0, d) ||
        error("Implicit density matrix requires a finite positive diagonal for Jacobi scaling")
    scale = 1.0 ./ sqrt.(d)
    rows = rowvals(A)
    values = nonzeros(A)
    for col in 1:n, k in nzrange(A,col)
        values[k] *= scale[rows[k]]*scale[col]
    end
    all(isfinite, values) || error("Nonfinite Jacobi-scaled density matrix")
    dropzeros!(A)
    rhs = zeros(n)
    workspace = Krylov.GmresWorkspace(A,rhs; memory=min(30,n))
    return DensityImplicitCache(operator,Float64(eta0),Float64(beta0),beta_in_operator,
        Float64.(h),row_scale,A,scale,rhs,zeros(n),zeros(n),workspace)
end

function validate_implicit_cache(cache::DensityImplicitCache, operator, eta0, beta0, h;
        beta_in_operator::Bool=false)
    valid = cache.operator === operator && cache.eta0 == eta0 &&
        cache.beta0 == beta0 && cache.beta_in_operator == beta_in_operator && cache.h == h
    valid || throw(ArgumentError("Stale implicit density cache: rebuild after changing the mesh, operator, eta0, or beta0"))
    return cache
end

"""
    implicit_density_solve!(result, rhs, cache; rtol=1e-7, itmax=200)

Solve `(I - Rhat/eta0) * result = rhs` using Jacobi-scaled GMRES. The restarted
workspace is reused, with at most 30 Krylov vectors. Check the residual of the
original unscaled equation before changing `result`; there is no LU or explicit
fallback. `result` may alias `rhs`.
"""
function implicit_density_solve!(result::AbstractVector, rhs::AbstractVector,
        cache::DensityImplicitCache; rtol::Float64=1e-7, itmax::Int=200)
    length(result) == length(rhs) == length(cache.scale) ||
        throw(DimensionMismatch("Implicit density vectors have inconsistent lengths"))
    isfinite(rtol) && 0 < rtol < 1 || throw(ArgumentError("GMRES rtol must be in (0,1)"))
    itmax > 0 || throw(ArgumentError("GMRES itmax must be positive"))
    all(isfinite, rhs) || error("Nonfinite implicit density right-hand side")
    rhs_norm = norm(rhs)
    isfinite(rhs_norm) || error("Nonfinite implicit density right-hand-side norm")
    if rhs_norm == 0
        fill!(result,0.0)
        return (iterations=0, relative_residual=0.0)
    end
    @. cache.scaled_rhs = cache.scale * rhs
    # Convert the requested unscaled tolerance to a sufficient scaled tolerance:
    # ||r|| <= ||S*r|| / minimum(diag(S)).
    scaled_rtol = rtol * minimum(cache.scale) * rhs_norm / norm(cache.scaled_rhs)
    Krylov.gmres!(cache.workspace,cache.scaled_matrix,cache.scaled_rhs;
        restart=true,atol=0.0,rtol=scaled_rtol,itmax)
    stats = cache.workspace.stats
    stats.solved || error("Implicit density GMRES failed after $(stats.niter) iterations: $(stats.status)")
    @. cache.solution = cache.scale * cache.workspace.x
    mul!(cache.residual,cache.operator,cache.solution)
    @. cache.residual = cache.solution - cache.row_scale * cache.residual - rhs
    relative_residual = norm(cache.residual)/rhs_norm
    isfinite(relative_residual) && relative_residual <= max(1.01rtol,100eps(Float64)) ||
        error("Implicit density solve failed its unscaled residual check: $relative_residual (tolerance $rtol)")
    copyto!(result,cache.solution)
    return (iterations=stats.niter, relative_residual=relative_residual)
end
