isdefined(@__MODULE__, :selected_weighted_patch_system) ||
    include(joinpath(@__DIR__, "selected_weighted_patches.jl"))

regularized_relative_norm(numerator, denominator) =
    iszero(denominator) ? (iszero(numerator) ? 0.0 : Inf) : numerator / denominator

@inline function regularized_quartic_mean(a, b, c, d)
    a == b == c == d && return 1 / 35
    (a == b == c || a == b == d || a == c == d || b == c == d) && return 1 / 140
    ((a == b && c == d) || (a == c && b == d) || (a == d && b == c)) && return 1 / 210
    (a == b || a == c || a == d || b == c || b == d || c == d) && return 1 / 420
    return 1 / 840
end

"""Exact energy Gram entry of two cubic weighted edge test functions."""
function regularized_test_energy_entry(gram, volume, hat, first, second)
    a, b = POISSON_P2_EDGES[first]
    c, d = POISSON_P2_EDGES[second]
    terms_first = ((a, b, hat), (hat, a, b), (hat, b, a))
    terms_second = ((c, d, hat), (hat, c, d), (hat, d, c))
    value = 0.0
    for p in terms_first, q in terms_second
        value += regularized_quartic_mean(p[1], p[2], q[1], q[2]) * gram[p[3], q[3]]
    end
    return 16volume * value
end

function regularized_weighted_patch_system(fan, rhs_data, node; readout_data=rhs_data)
    system = selected_weighted_patch_system(fan, rhs_data, node; readout_data)
    H = zeros(length(system.b), length(system.b))
    for tet in system.tets
        ids, G = view(fan.tet_nodes, :, tet), fan.gradients[tet]
        hat = findfirst(==(node), ids)
        gram = transpose(G) * G
        rows = [get(system.edge_rows, poisson_ordered_edge(ids[a], ids[b]), 0)
            for (a, b) in POISSON_P2_EDGES]
        for first in 1:6, second in 1:6
            row, column = rows[first], rows[second]
            (row == 0 || column == 0) && continue
            H[row, column] += regularized_test_energy_entry(
                gram, fan.volumes[tet], hat, first, second)
        end
    end
    return merge(system, (; H))
end

"""Factor both energy Grams and prepare each unweighted/normalized SVD once.

B=inv(Lh)*A*inv(Lk'), c=inv(Lh)*b and y=Lk'*delta. Thus ||c|| is the dual norm of
the weighted residual, ||y|| is correction energy norm, and ||B|| <= 1.
Raw Euclidean least squares and energy-normalized least squares are different
metrics when the equations are incompatible; this is separate from damping.
"""
function regularized_weighted_prepare(system; numerical_rank_tolerance=1e-12)
    0 < numerical_rank_tolerance < 1 || throw(ArgumentError("Invalid numerical rank tolerance"))
    n = length(system.b)
    if n == 0
        Lk, Lh = LowerTriangular(zeros(0, 0)), LowerTriangular(zeros(0, 0))
        decomposition = (; U=zeros(0, 0), V=zeros(0, 0), S=Float64[])
        raw = energy = (; B=zeros(0, 0), c=Float64[], decomposition)
        return (; system, Lk, Lh, raw, energy, numerical_rank_tolerance,
            normalized_rhs_norm=0.0, normalized_left_null_fraction=0.0,
            normalized_numerical_rank=0, readout_dual_norm=0.0)
    end
    Lk, Lh = cholesky(Symmetric(system.K)).L, cholesky(Symmetric(system.H)).L
    B, c = Lh \ (system.A / transpose(Lk)), Lh \ system.b
    raw = (; B=system.A, c=system.b, decomposition=svd(system.A; full=true))
    energy = (; B, c, decomposition=svd(B; full=true))
    singular = energy.decomposition.S
    rank = count(s -> s > numerical_rank_tolerance * maximum(singular), singular)
    left_null_part = transpose(energy.decomposition.U[:, rank+1:end]) * c
    return (; system, Lk, Lh, raw, energy, numerical_rank_tolerance,
        normalized_rhs_norm=norm(c),
        normalized_left_null_fraction=regularized_relative_norm(norm(left_null_part), norm(c)),
        normalized_numerical_rank=rank, readout_dual_norm=norm(Lk \ system.g))
end

function regularized_tikhonov_coefficients(singular, projected, alpha)
    return [s == 0 ? 0.0 : s * projected[i] / (s^2 + alpha^2)
        for (i, s) in enumerate(singular)]
end

"""Data-only correction-energy cap, using the smallest admissible Tikhonov alpha."""
function regularized_gain_cap(singular, projected, rhs_norm, gain, numerical_cutoff)
    isfinite(gain) && gain > 0 || throw(ArgumentError("Gain cap must be positive"))
    active_singular = [s > numerical_cutoff ? s : 0.0 for s in singular]
    initial = [s > 0 ? projected[i] / s : 0.0 for (i, s) in enumerate(active_singular)]
    target = gain * rhs_norm
    norm(initial) <= target && return (; spectral=initial, alpha=0.0, active=false)
    lower, upper = 0.0, 1 / (2gain)
    while norm(regularized_tikhonov_coefficients(active_singular, projected, upper)) > target
        upper *= 2
    end
    for _ in 1:64
        middle = (lower + upper) / 2
        if norm(regularized_tikhonov_coefficients(active_singular, projected, middle)) > target
            lower = middle
        else
            upper = middle
        end
    end
    return (; spectral=regularized_tikhonov_coefficients(active_singular, projected, upper),
        alpha=upper, active=true)
end

"""Apply one spec (name, normalization, filter, parameter), retaining original readout.

Normalizations are :raw/:energy; filters :tsvd/:tikhonov use parameter*sigma_max.
:gain_cap is energy-normalized only and uses parameter=G. Its alpha=0 branch
uses the numerical-rank pseudoinverse. Every filter, including positive-alpha
Tikhonov, first removes numerical-null modes below the common rank floor;
this prevents roundoff singular values from being amplified as weak modes.
retained_rank counts modes above the transition, not hard support of a
Tikhonov filter; effective_rank reports the corresponding soft spectral count.
"""
function regularized_weighted_solution(prepared, spec)
    spec.normalization in (:raw, :energy) || throw(ArgumentError("Unknown normalization"))
    spec.filter in (:tsvd, :tikhonov, :gain_cap) || throw(ArgumentError("Unknown filter"))
    isfinite(spec.parameter) && spec.parameter > 0 || throw(ArgumentError("Parameter must be positive"))
    selected = getproperty(prepared, spec.normalization)
    decomposition, singular = selected.decomposition, selected.decomposition.S
    largest = isempty(singular) ? 0.0 : maximum(singular)
    numerical_cutoff = prepared.numerical_rank_tolerance * largest
    active_singular = [s > numerical_cutoff ? s : 0.0 for s in singular]
    projected = transpose(decomposition.U) * selected.c
    cutoff, alpha, cap_active = spec.parameter * largest, 0.0, false
    if spec.filter == :tsvd
        spectral = [s > cutoff ? projected[i] / s : 0.0 for (i, s) in enumerate(active_singular)]
    elseif spec.filter == :tikhonov
        alpha = cutoff
        spectral = regularized_tikhonov_coefficients(active_singular, projected, alpha)
    else
        spec.normalization == :energy || throw(ArgumentError("Gain caps require energy normalization"))
        capped = regularized_gain_cap(singular, projected, norm(selected.c),
            spec.parameter, numerical_cutoff)
        spectral, alpha, cap_active = capped.spectral, capped.alpha, capped.active
        cutoff = iszero(alpha) ? numerical_cutoff : alpha
    end
    y = decomposition.V * spectral
    coefficients = spec.normalization == :energy ? transpose(prepared.Lk) \ y : y
    system = prepared.system
    residual = system.A * coefficients - system.b
    normalized_residual = prepared.Lh \ residual
    energy_norm = norm(transpose(prepared.Lk) * coefficients)
    numerical_rank = count(>(numerical_cutoff), singular)
    retained_rank = count(>(cutoff), active_singular)
    effective_rank = spec.filter == :tsvd || iszero(alpha) ? Float64(retained_rank) :
        sum(s^2 / (s^2 + alpha^2) for s in active_singular)
    actual_gain = regularized_relative_norm(energy_norm, prepared.normalized_rhs_norm)
    gain_bound = spec.filter == :gain_cap ?
        spec.parameter * prepared.readout_dual_norm * prepared.normalized_rhs_norm : NaN
    return (; name=spec.name, method=String(spec.name), normalization=spec.normalization,
        filter=spec.filter, parameter=spec.parameter, coefficients, residual,
        rhs_norm=norm(system.b), residual_norm=norm(residual),
        relative_residual=regularized_relative_norm(norm(residual), norm(system.b)),
        normalized_rhs_norm=prepared.normalized_rhs_norm,
        normalized_residual_norm=norm(normalized_residual),
        relative_normalized_residual=regularized_relative_norm(
            norm(normalized_residual), prepared.normalized_rhs_norm),
        normalized_left_null_fraction=prepared.normalized_left_null_fraction,
        correction_energy=energy_norm^2 / 2, correction_energy_norm=energy_norm,
        coefficient_norm=norm(coefficients), readout=dot(system.g, coefficients),
        chosen_alpha=alpha, spectral_cutoff=cutoff, actual_gain,
        retained_rank, numerical_rank, numerical_null_modes=length(singular)-numerical_rank,
        numerical_singular_cutoff=numerical_cutoff,
        effective_rank, largest_singular_value=largest,
        readout_dual_norm=prepared.readout_dual_norm, readout_gain_bound=gain_bound,
        gain_cap_active=cap_active)
end

function regularized_weighted_default_specs()
    specs = NamedTuple[]
    for (normalization, filter, powers) in
            ((:raw, :tsvd, (-11, -8, -6, -4)),
             (:energy, :tsvd, (-12, -8, -6, -4, -2)),
             (:energy, :tikhonov, (-8, -6, -4, -2)))
        for power in powers
            push!(specs, (; name=Symbol(normalization, "_", filter, "_1em", -power),
                normalization, filter, parameter=10.0^power))
        end
    end
    for gain in (10.0, 100.0)
        push!(specs, (; name=Symbol("energy_gain_cap_", Int(gain)),
            normalization=:energy, filter=:gain_cap, parameter=gain))
    end
    return specs
end

"""Compare fixed regularization specs in one pass over patches.

No exact solution or benchmark effectivity enters parameter selection. Each
patch factors K/H and each normalization only once. Diagnostics are scalar CSV
rows; original and physically normalized residuals are both retained.
"""
function regularized_weighted_patch_estimates(mesh, fan, rhs_data;
        readout_data=rhs_data, factor=0.5, include_corrections=true,
        specs=regularized_weighted_default_specs(), numerical_rank_tolerance=1e-12,
        retain_coefficients=false)
    names = [spec.name for spec in specs]
    length(unique(names)) == length(names) || throw(ArgumentError("Method names must be unique"))
    nn = length(rhs_data.U)
    base = include_corrections ? readout_data.conversion : zeros(nn)
    correction = include_corrections ? readout_data.correction : zeros(length(mesh.cells))
    nodal = Dict(name => copy(base) for name in names)
    counts = zeros(Int, nn)
    ranks = Dict(name => zeros(Int, nn) for name in names)
    diagnostics = NamedTuple[]
    retained = retain_coefficients ? NamedTuple[] : nothing
    for node in 1:nn
        system = regularized_weighted_patch_system(fan, rhs_data, node; readout_data)
        prepared = regularized_weighted_prepare(system; numerical_rank_tolerance)
        counts[node] = length(system.b)
        for spec in specs
            state = regularized_weighted_solution(prepared, spec)
            nodal[spec.name][node] += factor * state.readout
            ranks[spec.name][node] = state.retained_rank
            scalar_names = filter(name -> name ∉ (:coefficients, :residual), propertynames(state))
            scalars = NamedTuple{Tuple(scalar_names)}(Tuple(getproperty(state, name) for name in scalar_names))
            push!(diagnostics, merge((; node, dofs=counts[node]), scalars))
            retain_coefficients && push!(retained,
                (; node, method=spec.name, edges=system.edges, coefficients=state.coefficients))
        end
    end
    estimates = Dict{Symbol,Any}()
    for spec in specs
        statistics = (; total_dofs=sum(counts), max_dofs=maximum(counts), patch_count=nn,
            normalization=spec.normalization, filter=spec.filter, parameter=spec.parameter,
            numerical_rank_tolerance, convention="regularized weighted P2 patches with free artificial edges")
        estimates[spec.name] = poisson_collect_estimate(mesh, fan, nodal[spec.name],
            correction, copy(counts), ranks[spec.name], statistics)
    end
    return (; estimates, diagnostics, coefficients=retained)
end
