# 3D density regularization

ToOpt3 uses TPFA for all density face gradients. Load the reusable numerical
kernels with `include("src/ToOpt3.jl")` from the repository root. This is a plain
include entry point for `src/regularization/load.jl`; it defines small functions
and reusable caches in the caller's scope, without a package module. Launch the
full 3D optimization through `src/main_optim_runs.jl`.
Both the existing sDWR and the optional [3D PU-DWR estimator](pu_dwr.md) work with
all three regularizations and optional projection.
`Project.toml` is an unnamed script environment, so dependency precompilation
does not expect a `ToOpt3` package module.

| Option | Meaning | Default |
| --- | --- | --- |
| `--regularization mean` | Gradient penalty with mean-force scaling | `mean` |
| `--regularization local_max` | Gradient penalty with direct-neighbor sensitivity boost | |
| `--regularization helmholtz` | Forward/adjoint Helmholtz filter | |
| `--projection true` | Smooth volume-preserving projection and automatic continuation | `false` |
| `--update_mode explicit` / `implicit` | Explicit or semi-implicit gradient-penalty update | `explicit` |
| `--filter_radius` | Uniform Helmholtz radius override; zero uses r²_i = 2·β0·h_i², the same coefficient as the gradient penalties | `0.0` |
| `--density_substeps` | Requested subdivisions of one optimization time unit | `0` |
| `--filter_tolerance` | Relative tolerance of Helmholtz forward/adjoint solves | `1e-12` |
| `--density_activity_tolerance` | Physical-density change used by adaptive marking | `1e-3` |

`--flux_scheme tpfa` is accepted for compatibility. The other face-gradient
approximations have been removed. `--laplace_rescale` is accepted as a deprecated
no-op so existing TPFA cluster scripts continue to work.

Run from the repository root, for example:

```text
julia --project=. src/main_optim_runs.jl --regularization mean --projection true
julia --project=. src/main_optim_runs.jl --regularization local_max --update_mode implicit --projection true
julia --project=. src/main_optim_runs.jl --regularization helmholtz --filter_radius 0.02 --projection true
```

The same option names are keywords of `run_optimization`. Helmholtz uses an
explicit design update with elliptic forward/adjoint solves; requesting its
`implicit` update is an error. Zero substeps retains the original gradient
subdivision count (one step for implicit) and selects one Helmholtz design step.
Explicit gradient updates additionally satisfy the current TPFA diffusion CFL
bound, including the local sensitivity boost, up to at most 1000 substeps per
optimization step (`DensityOptions.max_substeps`). Beyond that the step exceeds
the bound and a `[density] explicit update capped` line is printed. The
`local_max` boost is capped at 10 (`max_boost` keyword of `update_coefficients!`).

## Design and physical density

`states.design_vec` is the persistent design `a`. `states.χ_vec` is the physical
density used by mechanics, VTK output, and adaptive marking. With projection off,
the gradient formulations use `χ=a`; Helmholtz uses `χ=F*a`. With projection on,
these become `χ=P(a)` and `χ=P(F*a)` respectively. Custom starting designs should
be assigned to `design_vec` and mapped to physical density before mechanics.

Mesh refinement inherits the parent design. Coarsening takes a volume-weighted
average of child designs. The operator is rebuilt and physical density is
recomputed after transfer, before the next mechanical solve. The filter radius
chosen at startup stays fixed through these mesh changes.

## Gradient penalties

The current 3D baseline is retained:

```text
base_i = 2 * beta0 * h_i²
mean:      beta_hat_i = base_i
local_max: beta_hat_i = base_i * clamp(max(abs(q_j), j=i and face neighbors)/s, 1, 10)
beta_i = s * beta_hat_i
eta = eta0 * s
```

Here `q` is the physical force pulled back through projection, and `s` is its
volume/grayness-weighted mean magnitude with binary and zero-force safeguards.
The 2D reference uses a globally constant baseline; the 3D `h_i²` scaling is
deliberately preserved. The maximum reads original forces, so traversal order
cannot propagate a peak beyond direct face neighbors. Coefficients are frozen
within each substep and refreshed afterwards.

The penalty acts on the design field, including when projection is enabled.
With `K` the positive TPFA stiffness and `M=diag(cell volumes)`, explicit evolution
uses `a + dt/eta0 * (q/s - M⁻¹*K*a)`. The implicit option solves
`(M + dt/eta0*K)*trial = M*(a + dt/eta0*q/s)`. Both enforce design bounds and
volume afterwards. This split implicit update is not a coupled constrained
backward-Euler solve. Neither force-dependent coefficient derivatives nor a
hard minimum member-size guarantee are introduced.

Implicit CG uses `DensityOptions.implicit_tolerance=1e-10` and
`implicit_max_iterations=1000`. These kernel settings are independent of the
Helmholtz tolerance and are not additional command-line options.

## Helmholtz filter

The filter uses `B=M+K(r²)` with a uniform physical radius, zero boundary flux,
and no gradient penalty. Forward filtering solves `B*phi=M*a`; the adjoint solves
`B*q=M*p` after the projection pullback. The mass factors are essential on
unequal-volume cells. The same cached, Jacobi-preconditioned CG workspace serves
both solves, without a sparse direct factorization.

The design trial is `a + dt*q/(eta0*s)`, followed by the box/volume constraint,
forward filter, and optional projection. The filter preserves constants,
volume-weighted mass, and bounds to solve tolerance. Its radius is independent
of `beta0`, `eta0`, and the design step size. TPFA retains its consistency
limitation on nonorthogonal faces.

## Projection and continuation

Normalize `t=(phi-χmin)/(1-χmin)` and apply the normalized tanh projection. Its
threshold is found by a scalar safeguarded Newton solve so that physical and
input material volumes agree. The exact pullback differentiates this implicit
threshold, including its global rank-one correction. Therefore the existing
design volume constraint also enforces physical volume.

Projection beta is separate from the gradient-penalty `beta0`. Defaults in
`DensityProjectionOptions` set initial beta `1`, growth factor `1e-3`, relative
growth cap `0.5`, and grayness target `0.01`. After a completed iteration, improving
objectives increase beta according to objective progress, capped at the relative
growth limit. A nonzero plateau uses the cap; worsening objectives and sufficiently
low physical grayness hold beta. Beta is fixed during each mechanical solve and
its density substeps. A continuation change remaps physical density and contributes
to adaptive activity before marking. Helmholtz reuses its stored filtered field
for this remapping, so sharpening needs no additional filter solve. Continuation
history survives remeshing. The final iteration records its projection state but
does not advance beta for an unused next iteration.

## Results and field alignment

The regular `strain_energy` history records the equilibrium objective before
each density update. The corresponding intermediate VTK snapshots use that same
pre-update physical density and displacement; their change mask records physical
density activity during the subsequent update and continuation.

After the last density update, the driver performs one additional equilibrium
solve. The final VTK displacement and error indicators therefore match the
accepted physical density. `final_strain_energy` records this final objective;
`final_design`, `final_density`, and `final_volumes` retain the accepted fields.
The final solve contributes to the reported mechanical-solver timings but does
not add another optimization iteration to the regular history.

## Performance and verification

Face geometry and sparse structure are prepared on mesh changes. Local-max
updates change sparse values in place. Density kernels reuse dense scratch and
iterative-solver workspaces; optional diagnostics allocate only when requested.
Geometry includes every active leaf subface of coarse/fine interfaces once.

Run the focused tests from the repository root:

```text
julia --project=. --startup-file=no Tests/density_options.jl
julia --project=. --startup-file=no Tests/regularization/runtests.jl
julia --project=. --startup-file=no Tests/tpfa_geometry.jl
julia --project=. --startup-file=no Tests/density_update_modes.jl
```

Older approximation studies in `Tests/*.md` are retained as historical records.
Their recommendations and removed comparison commands are not current options.
