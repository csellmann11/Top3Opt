# 3D density regularization

ToOpt3 uses TPFA for all density face gradients. Load the reusable numerical
kernels with `include("src/ToOpt3.jl")` from the repository root. This is a plain
include entry point for `src/regularization/load.jl`; it defines small functions
and reusable caches in the caller's scope, without a package module. Launch the
full 3D optimization through `src/main_optim_runs.jl`.
Both the existing sDWR and the optional [3D PU-DWR estimator](pu_dwr.md) work with
all regularizations and optional sharpening.
`Project.toml` is an unnamed script environment, so dependency precompilation
does not expect a `ToOpt3` package module.

| Option | Meaning | Default |
| --- | --- | --- |
| `--regularization gradient` / `helmholtz` | Gradient penalty or forward/adjoint Helmholtz filter | `gradient` |
| `--scaling mean` | Gradient penalty with mean-force scaling (boost `b_i = 1`) | `mean` |
| `--scaling local_max` | Direct-neighbor sensitivity boost | |
| `--scaling local_lambda` | `local_max` without the mean floor, current `abs(lambda)` in the boost (explicit only) | |
| `--scaling local_free` | `local_lambda`, but a gaining cell counts neighbor forces only by their grayness (explicit only) | |
| `--integrator explicit` / `semi_implicit` | Explicit (automatic CFL substeps) or semi-implicit gradient-penalty update | `explicit` |
| `--diag true` | Secant reaction damping of the gradient penalty | `false` |
| `--move_limit` | Per-cell box `|x³ - x_old³| <= move_limit*dt` per substep, all updates | `Inf` |
| `--projection true` | Sharpening with continuation: projection of the Helmholtz design, double-obstacle term of the gradient penalty | `false` |
| `--sharpen_imbalance_tolerance` | Both continuations grow only below this imbalance of the last density substep | `0.1` |
| `--sharpen_gray_tolerance` | Both continuations stop at this grayness | `0.01` |
| `--projection_growth_rate` | Projection only: largest growth of `ln(beta)` per optimization step | `0.3` |
| `--projection_beta_initial` | Projection only: initial `beta` | `1.0` |
| `--obstacle_max_weight` | Obstacle only: cap of the weight `g` (`Inf`: none) | `Inf` |
| `--beta0` | Regularization strength; per-cell coefficient `2*beta0*h_i²` with `h_i = cbrt(cell volume)` | `3.0` |
| `--eta0` | Viscosity of the density update | `15.0` |
| `--filter_radius` | Uniform Helmholtz radius override; zero uses r²_i = 2·β0·h_i², the same coefficient as the gradient penalties | `0.0` |
| `--density_substeps` | Requested subdivisions of one optimization time unit | `0` |
| `--filter_tolerance` | Relative tolerance of Helmholtz forward/adjoint solves | `1e-12` |
| `--density_activity_tolerance` | Physical-density change used by adaptive marking | `1e-3` |

`--flux_scheme tpfa` is accepted for compatibility. The other face-gradient
approximations have been removed. `--laplace_rescale` is accepted as a deprecated
no-op so existing TPFA cluster scripts continue to work.

Run from the repository root, for example:

```text
julia --project=. src/main_optim_runs.jl --scaling mean --projection true
julia --project=. src/main_optim_runs.jl --scaling local_max --integrator semi_implicit --diag true --projection true
julia --project=. src/main_optim_runs.jl --scaling local_lambda --integrator explicit --move_limit 0.1 --projection true
julia --project=. src/main_optim_runs.jl --regularization helmholtz --filter_radius 0.02 --projection true
```

The same option names are keywords of `run_optimization` and fields of
`DensityOptions`. The three gradient-penalty settings are orthogonal: every
scaling works with both integrators and with or without damping, except that
`local_lambda` and `local_free` are explicit only. Helmholtz uses an explicit design update with
elliptic forward/adjoint solves and has neither a boost nor damping; requesting
`semi_implicit`, a scaling other than `mean`, or `--diag true` for it is an error.
The former `--regularization mean|local_max|local_lambda` (and `local_free`) and `--update_mode`
(`update_mode` keyword) were removed and are rejected; run fingerprints of runs without damping and
move limit are unchanged. A finite move limit appends `;M=<value>box` to the fingerprint,
so these runs do not reuse directories of the former eta-increase rule. With zero
substeps (the default) an explicit gradient update takes the fewest equal substeps
of the remaining time inside the measured TPFA diffusion CFL bound
dt*max_i(K_ii/area_i)/eta0 <= 0.95, recomputed every substep; there is no fixed
count any more, and such runs append `;N=cfl` to the fingerprint. The
`local_max` boost does not enter this bound: it is the same as for `mean`. At
most 1000 substeps are taken per optimization step (`DensityOptions.max_substeps`);
beyond that the step exceeds the bound and a `[density] explicit update capped`
line is printed. The semi-implicit integrator takes one step, and Helmholtz one
design step. `--density_substeps n > 0` caps the substep length at 1/n.

## Design and physical density

`states.design_vec` is the persistent design `a`. `states.χ_vec` is the physical
density used by mechanics, VTK output, and adaptive marking. Helmholtz uses
`χ=F*a`, or `χ=P(F*a)` with projection. The gradient penalties evolve the physical
density itself and have no projection: the driver builds their states with
`separate_design=false`, so `design_vec` and `χ_vec` are one vector. Custom
starting designs should be assigned to `design_vec` and mapped to physical density
(`map_design!`) before mechanics.

Mesh refinement inherits the parent design. Coarsening takes a volume-weighted
average of child designs. The operator is rebuilt and physical density is
recomputed after transfer, before the next mechanical solve. A positive
`--filter_radius` stays fixed through these mesh changes; the default Helmholtz
radius r²_i = 2·β0·h_i² follows the cell sizes of the new mesh.

## Gradient penalties

The baseline uses the volume-equivalent cell size `h_i = cbrt(volume_i)`, the edge
length on cubes; no cell diameter enters the regularization:

```text
base_i = 2 * beta0 * h_i²
mean:      beta_hat_i = base_i
local_max: m_i = max(abs(q_j), j=i and face neighbors)/s
           b_i = max(1, m_i)/2
           beta_hat_i = base_i * b_i,   eta_i = eta0 * s * b_i
beta_i = s * beta_hat_i
eta = eta0 * s            (mean)
```

The fixed factor 1/2 (`LOCAL_MAX_BOOST_SCALE`) calibrates `local_max` against `mean`:
with `max(1, m_i)` alone the boost exceeds one along most interfaces and the scheme
acts like `mean` with roughly twice `beta0` and `eta0`. Because the factor is a
constant, the normalized force `(q_i/s - lambda)/b_i` of a cell with `q_i >= lambda`
is at most 2 for every force field. The value was calibrated on 2D benchmarks.

On a uniform grid a 0→1 density jump across one face gives the normalized diffusion
`base_i/h_i² = 2*beta0` in the solid cell against a normalized `local_max` force of
at most 2. For `beta0 > 1` such an unresolved interface therefore cannot be stationary.

Migration note: a `beta0 = x` that was used with the cell diameter as `h_i`
corresponds to `beta0 = 3x` with `h_i = cbrt(volume_i)`. This holds on hexahedral
meshes (cubic cells) and only approximately on Voronoi meshes. Run fingerprints and
cluster job names (`_B<beta0>`) do not record which length was used. Default runs
keep their fingerprint, while a result stored with another `beta0 = x` under the cell
diameter can carry the name of a present run with the same `x` and has to be compared
with `3x`.

For `local_max` the boost `b_i` scales the regularization and the viscosity of a
cell together. With the baseline TPFA operator `K` (coefficient `base_i`), the
volume multiplier `lambda` and the bound multiplier `gamma_i`,

```text
eta0 * b_i * dchi_i/dt = q_i/s - lambda - gamma_i - b_i * (K chi)_i / volume_i
```

and, divided by `b_i`, the form that is integrated:

```text
eta0 * dchi_i/dt = (q_i/s - lambda - gamma_i) / b_i - (K chi)_i / volume_i
```

- The stationary state is that of a locally boosted `beta`: at a force peak the
  regularization is `b_i` times stronger relative to the force.
- Diffusion keeps the baseline operator, so the explicit stability limit and the
  implicit system are those of `mean`. One force peak does not shrink the time
  step, and the boost needs no cap. On cubes the limit is `eta0 >= 12*beta0*dt`
  (up to `14*beta0*dt` for a fine cell at a 2:1 interface).
- A peak cell with `b_i = abs(q_i)/(2s)` has the normalized force `+-2`: it moves
  like a cell with twice the mean force. Whether a cell gains material is still decided by
  comparing its full force with `lambda`.
- The volume multiplier is divided by `b_i` as well: the projection uses
  `chi_i = clamp(y_i - shift*w_i, lower, 1)` with `w_i = 1/b_i` (explicit) or
  `(M + dt/eta0*K) w = M/b` (semi-implicit, one extra CG solve per substep).

### local_lambda

`local_lambda` replaces the mean floor 1 of `local_max` by the current normalized
volume multiplier `l` (the `lambda` of the equations above; diagnostics report `s*l`):

```text
b_i(l) = max(m_i, abs(l), 1e-12)/2          (1e-12 = LOCAL_LAMBDA_GUARD, guards 0/0 only)
chi_i  = clamp(chi_i + a*((q_i/s - l)/b_i(l) - (K chi)_i/volume_i), lower, 1),   a = dt/eta0
```

- Because `b_i` depends on `l`, the multiplier is the fixed point of boost and
  volume constraint: `project_volume_floored!` re-evaluates `b_i(l)` inside a
  bisection on `l`. For `q >= 0` the volume is continuous and decreasing in `l >= 0`,
  but the pull toward void saturates at `2a`, so the bracket is grown from `l = 0` by
  doubling in the direction of the volume excess. The final boost `max(m_i, abs(l))/2`
  is stored as `beta_boost`.
- For `q >= 0` and `l >= 0` the normalized force satisfies
  `abs((q_i/s - l)/b_i) <= 2` in every cell and both directions, so one step moves a
  cell by at most `2a` plus diffusion. Cells whose force and face neighbors' forces
  are below `abs(l)` carry `b_i = abs(l)/2`.
- `s` cancels from the update, `(q_i/s - l)/b_i = 2*(q_i - s*l)/max(max_j abs(q_j), abs(s*l))`
  (j = i and face neighbors). Diffusion keeps the baseline operator, so the CFL limit
  is that of `mean`; a capped explicit step that runs past it can make the volume
  target unreachable, which is an error here.
- With signed forces (`q_i < 0`) or `l < 0` the bound
  is 4 and monotonicity is not guaranteed; the bisection keeps a sign change and
  stops with an error if a step cannot reach the volume target.
- Only the explicit update is available, because the boost needs the multiplier of
  the current step.

### local_free

At an interface the solid neighbor's large force sits in `m_i` of a gray cell and
slows its growth under `local_lambda`. `local_free` removes that for cells that gain
material and changes nothing else:

```text
theta_j  = clamp((chi_j - lower)/(1 - lower), 0, 1),   g_j = 4*theta_j*(1 - theta_j)
m_i^free = max(abs(q_i), g_j*abs(q_j), j = face neighbors)/s   (own force unweighted)
b_i(l)   = max(m_i^free, abs(l))/2   if q_i/s > l    (push)
b_i(l)   = max(m_i, abs(l))/2        otherwise       (pull: local_lambda)
```

- `g` is taken from the design at the start of the substep (`cache.old`). A clamped
  neighbor (`g = 0`) does not count in the push of a cell; a cell below the
  multiplier keeps the relief by any loaded neighbor, which lets a front advance.
- The force is that of `local_lambda` or larger for `q_i/s > l` and identical
  otherwise, continuous at the switch (its numerator vanishes there) and bounded by
  2 for `q >= 0`, `l >= 0`.
- Bisection, diffusion, secant damping, move limit and the stored boost are those of
  `local_lambda` with this `b_i`; `project_volume_floored!` takes the push boost
  `m_i^free/2` as `push_boost`. The stronger regularization around a force peak
  only acts on a gaining cell while the peak cell is gray. The double-obstacle term
  enters the explicit part like the diffusion and is not divided by the boost.
  Explicit only.

Here `q` is the physical force, and `s` is its
volume/grayness-weighted mean magnitude with binary and zero-force safeguards.
The 2D code uses the same gradient-penalty convention with `h_i = sqrt(cell area)`.
The maximum reads original forces, so traversal order
cannot propagate a peak beyond direct face neighbors. Coefficients are frozen
within each substep and refreshed afterwards.

The penalty acts on the physical density `χ`.
With `K` the positive TPFA stiffness and `M=diag(cell volumes)`, explicit evolution
uses `χ + dt/eta0 * (q/s - M⁻¹*K*χ)`. The implicit option solves
`(M + dt/eta0*K)*trial = M*(χ + dt/eta0*q/s)`. Both enforce the bounds and
volume afterwards. This split implicit update is not a coupled constrained
backward-Euler solve. Neither force-dependent coefficient derivatives nor a
hard minimum member-size guarantee are introduced.

Implicit CG uses `DensityOptions.implicit_tolerance=1e-10` and
`implicit_max_iterations=1000`. These kernel settings are independent of the
Helmholtz tolerance and are not additional command-line options.

### Secant reaction damping (`--diag`)

A cell whose force falls steeply when it grows (a stiff, single-cell-dominated
objective, force `-k (x - x*)`) overshoots its maximum when `dt*k/eta > 2` and
two-cycles. The damping measures this curvature per cell from the entry states
`k-1, k` of two consecutive optimization steps (density `x`, force `p`):

```text
D_new_i = max(-(p_i^k - p_i^(k-1)) / (x_i^k - x_i^(k-1)), 0)    (0 if |dx_i| <= 1e-12)
D_i     = D_new_i/2 + D_i/2                                       (D = 0 without history)
```

and adds it to the viscosity over the full optimization step `T = 1` (not the
substep length, which would dilute it to `D/n`): with `q = p/s`,

```text
(eta0*b_i + D_i/s) * dchi_i/dt = q_i - lambda - b_i * (K chi)_i / volume_i.
```

Explicitly this divides the step of cell `i` by `1 + r_i`, `r_i = D_i/(s*eta0*b_i)`,
and the volume response becomes `w_i = 1/(b_i*(1+r_i))`; semi-implicitly the mass
term is `M*diag(1+r)` in the system and in the volume-response solve (rebuilt each
substep while damping is active). For `local_lambda` the denominator
`b_i(l) + D_i/(s*eta0)` is evaluated inside the floored bisection. The right-hand
side is unchanged, so stationary points depend neither on `D` nor on `eta0`;
without history the update is bitwise the undamped one. The history is reset at
the start of a run and on every remesh (new cache). A step of the obstacle weight
`g` keeps it: `g` does not enter the force whose secant `D` measures.

### Double-obstacle term (`--projection true`)

The gradient penalty sharpens with a double-obstacle potential instead of a density
projection. Its force per unit volume, positive = grow, is

```text
g * kappa_i * (chi_i - c),   c = (1 + lower)/2,   kappa_i = base_i / h_i² (= 2*beta0)
```

`kappa_i` is the normalized diffusion per face of a uniform grid (`t_f/volume_i = 2*beta0`
on cubes), so `g` is mesh independent: a mesh-aligned planar interface stays gray for
`g < 2` and becomes binary above. The term enters exactly like the diffusion,
`K*chi` becomes `K*chi - G` with the integrated force `G_i = g*base_i*h_i*(chi_i - c)`:
it is boosted together with `beta` (not divided by `b_i`), divided by `1 + r_i` with
damping, and explicit in both integrators (evaluated at the substep start; an implicit
anti-diffusive reaction would amplify without bound). `g = 0` is bitwise the plain update.

The driver continues `g` from zero by the imbalance of the last density substep. On
its free cells (strictly inside `[lower, 1]` and the move-limit box), with volume-weighted
L2 norms, `F = (q/s - l)/b`, `D = -(K chi)/volume`, `U = kappa*(chi_old - c)` and the
net term `N = (chi - chi_old)*(1 + r)/a` actually taken (`N = F + D + g*U` explicitly),

```text
imbalance = |N| / (|F| + |D| + g*|U|)
dg        = max(0, imbalance_tolerance - imbalance) * (|F| + |D| + g*|U|) / |U|,
```

capped at `--obstacle_max_weight`; `g` stops for good at the gray tolerance.

### Move limit (`--move_limit`)

For every update (gradient penalty and Helmholtz) and each substep of length `dt`,
the design may move only inside a per-cell box around the substep's start state
`x_old` (exponent `p = DensityOptions.simp_exponent`, the SIMP exponent 3 of the
mechanics, and `m = move_limit*dt`):

```text
lo_i = max(lower, max(x_old,i^p - m, 0)^(1/p)),   hi_i = min(1, (x_old,i^p + m)^(1/p)),
```

with `lo_i <= x_old,i <= hi_i` guarded against rounding, so `|x_i^p - x_old,i^p| <= m`
in every cell. The box replaces the density bounds `[lower, 1]` in the volume
bisection, like MMA move limits: after the semi-implicit solve, inside the
floored `local_lambda` bisection, and in the projection of the latent Helmholtz
design step. eta, force, scale, boost and damping are those of the unlimited
update, and each substep is taken once. If the box makes the volume target
unreachable, the projection stops with an error that names the move limit.
The number of cells that end a substep on a box bound tighter than `[lower, 1]`,
summed over the substeps, is printed per optimization step and stored in
`density_history.move_limited` and (per substep) in the diagnostic records.
`Inf` (default) is bitwise the unlimited update.

## Helmholtz filter

The filter uses `B=M+K(r²)` with r²_i = 2·β0·h_i² by default or the uniform
physical radius of `--filter_radius`, zero boundary flux,
and no gradient penalty. Forward filtering solves `B*phi=M*a`; the adjoint solves
`B*q=M*p` after the projection pullback. The mass factors are essential on
unequal-volume cells. The same cached, Jacobi-preconditioned CG workspace serves
both solves, without a sparse direct factorization.

The design trial is `a + dt*q/(eta0*s)`, followed by the box/volume constraint,
forward filter, and optional projection. The filter preserves constants,
volume-weighted mass, and bounds to solve tolerance. Its radius is independent
of `eta0` and the design step size; only the default radius scales with `beta0`.
TPFA retains its consistency
limitation on nonorthogonal faces.

## Projection and continuation

Normalize `t=(phi-χmin)/(1-χmin)` and apply the normalized tanh projection. Its
threshold is found by a scalar safeguarded Newton solve so that physical and
input material volumes agree. The exact pullback differentiates this implicit
threshold, including its global rank-one correction. Therefore the existing
design volume constraint also enforces physical volume. The projection belongs to
the Helmholtz design; the gradient penalties sharpen with the double-obstacle term.

Projection beta is separate from the gradient-penalty `beta0`. It starts at
`--projection_beta_initial` and grows by the imbalance of the last design step: on
the free design cells (inside `[lower, 1]` and the move-limit box), with `F = q/s`
and the normalized multiplier `l`,

```text
imbalance  = |F - l| / (|F| + |l|*sqrt(V_free))
ln(beta)  += projection_growth_rate * max(0, 1 - imbalance/sharpen_imbalance_tolerance)
```

Beta holds while the design still moves (imbalance at or above the tolerance), and
stops for good once the physical grayness is at most `--sharpen_gray_tolerance`. The
objective does not enter. Beta is fixed during each mechanical solve and
its density substeps. A continuation change remaps physical density and contributes
to adaptive activity before marking. Helmholtz reuses its stored filtered field
for this remapping, so sharpening needs no additional filter solve. Continuation
history survives remeshing. The final iteration records its sharpening state but
advances neither beta nor `g` for an unused next iteration. A beta or `g` step blocks
the convergence count of that iteration, and convergence requires the active
mechanism to be converged. `projection_history` holds one record per sharpened
optimization step: `(iteration, mechanism, value_used, value_next, threshold, grayness,
imbalance, advanced)` with `mechanism` `:projection` or `:obstacle` (`threshold = NaN`).
The settings of the inactive mechanism are ignored, so switching the regularization
alone gives a working run.

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
julia +release --project=. --startup-file=no Tests/density_options.jl
julia +release --project=. --startup-file=no Tests/regularization/runtests.jl
julia --project=. --startup-file=no Tests/tpfa_geometry.jl
julia --project=. --startup-file=no Tests/density_update_modes.jl
```

Older approximation studies in `Tests/*.md` are retained as historical records.
Their recommendations and removed comparison commands are not current options.
