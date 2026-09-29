"""
    density_substeps(operator, eta0, beta0; beta_in_operator=false)

Use the original density substep rule. The operator row norm is returned only
for diagnostics and never changes the number of substeps.
"""
function density_substeps(operator, eta0, beta0;
        beta_in_operator::Bool=false)
    isfinite(eta0) && eta0 > 0 || throw(ArgumentError("eta0 must be finite and positive"))
    isfinite(beta0) && beta0 >= 0 || throw(ArgumentError("beta0 must be finite and nonnegative"))
    # Equivalent to the legacy 4*ceil(12/eta0 * (2*hmin^2*beta0)/hmin^2).
    required = max(1, 4ceil(Int,24beta0/eta0))
    row_bound = 0.0
    worst_row = 0
    if beta_in_operator
        row_bound, worst_row = findmax(vec(sum(abs, operator; dims=2)))
    end
    if !isfinite(row_bound)
        error("Nonfinite regularization coefficients at state $worst_row")
    end
    return required, row_bound
end
