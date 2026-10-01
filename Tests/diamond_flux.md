# 3D density regularization

The default is now `--flux_scheme diamond`. For comparisons, select
`--flux_scheme tpfa`, `--flux_scheme taylor`, or `--flux_scheme strong`. The Julia
entry point accepts the corresponding symbols, for example
`run_optimization(...; flux_scheme=:taylor)`. `laplace_rescale` applies only to
the legacy `strong` scheme. Output directory identifiers include the flux scheme
and density update mode.

Density updates default to `--update_mode explicit`. Select
`--update_mode implicit` for implicit regularization, or use
`run_optimization(...; update_mode=:implicit)` in Julia. For cluster sweeps:

```bash
UPDATE_MODE=implicit bash cluster/run_sweeps.sh
```

The `UPDATE_MODE` setting in `cluster/run_sweeps.sh` accepts either `explicit`
or `implicit`; job names and logs record the chosen mode.

The `taylor` scheme in `src/taylor_operator.jl` computes the variable-beta product
rule `beta_hat * laplacian(chi) + grad(beta_hat) ⋅ grad(chi)` with nine Taylor
derivatives and ridge stabilization. It includes beta in the assembled operator
and uses physical neighbour locations. It preserves constants but is not exactly
conservative on graded meshes. The tests and benchmarks use this same implementation.

The face schemes assemble `div(beta_hat * grad(chi))`, with the existing 3D
coefficient `beta_hat[i] = 2 * beta0 * h[i]^2`. The common `p_avg` scaling of
regularization and viscosity cancels in the density update. The operator is
rebuilt after mesh adaptation, so both geometry and the spatially varying
coefficient follow the current mesh.

Each active interior subface contributes one flux with opposite signs to its
two neighboring cells. Its transmissibility is
`area / (delta_left/beta_left + delta_right/beta_right)`. Diamond adds the two
tangential gradient components reconstructed from face vertex values. Vertex
values use local linear least squares and mirrored samples on boundary faces,
following the 2D reference. Boundary flux is zero. The dimensionless least-squares
matrix uses a rank-revealing SVD with relative cutoff `sqrt(eps(Float64))`.
Numerically unresolved directions are discarded and the weights are normalized
to preserve constants. Resolved, full-rank stencils reproduce affine fields.
There is no arbitrary clipping of flux coefficients.

This cutoff matters on adaptive hex meshes: a coplanar group of cell centres can
appear full rank because of centroid integration roundoff. In the reproduced
MBB level-3 case, the old default `pinv` inverted a singular value of `1.84e-15`,
producing vertex weights of order `1e14`. The operator row bound jumped from
roughly 100 to `2.176e14`. The corrected assembly on that same mesh has row bound
`100.3946`. See [validation results](regularization_validation.md).

Reconstruction is only built for vertices of nonorthogonal interior faces.
Orthogonal faces use their exact two-point flux directly; uniform hex meshes
therefore require no vertex SVDs. Zero sparse entries are removed. Density
updates reuse this assembled sparse operator.

The operator preserves constants and conserves the volume-weighted total.
It reduces to TPFA on orthogonal faces. Like the reference correction, it is not
guaranteed symmetric or monotone. Planar faces and cell centers on opposite sides
of each interior face are required. The regression suite tests fixed-step
diffusion stability on specific unbalanced hex meshes up to a 32:1 direct size
ratio; this is not a stability proof for arbitrary distorted polyhedra.

Explicit mode preserves the original substep rule:
`max(1, 4*ceil(Int, 24*beta0/eta0))`, which gives eight substeps for `beta0=1`
and `eta0=15`. The operator row norm is logged only as a diagnostic and **never
increases the substep count**. A nonfinite operator causes an immediate error.

The scalar multiplier is found by projecting the complete unconstrained explicit
Euler update onto `[chi_min,1]` with the requested volume-weighted mean. The bracket
includes both the driving and regularization terms. Safeguarded Newton steps
use the volume of currently unclipped cells, with bisection as a fallback and
a finite iteration limit. In explicit mode, this solves the same clipped update
as bisection.
If every cell is at a bound, the undefined `g`-weighted force average falls back
to its volume-weighted average. Zero average driving force leaves density unchanged.

Implicit mode uses one `dt=1` step per optimization iteration. With the current
mechanical driving force held fixed, it solves

```text
(I - Rhat/eta0) y = chi - p_chi/(eta0*p_avg)
chi_new = weighted_volume_and_bounds_projection(y)
```

Here `Rhat` is the same regularization operator used by explicit mode. For
`strong`, it is `diag(2*beta0*h.^2) * L`; the other schemes already incorporate
that coefficient. This changes the time discretization, without introducing a
filter or changing the regularization functional. The solve uses Jacobi-scaled
GMRES with a cached matrix and workspace, reused while the mesh and operator
remain unchanged and rebuilt at operator assembly. There are no additional
density substeps or LU fallback.

This is an IMEX update followed by the existing volume/bounds projection, as in
the AndersonPlasticity implementation. It is not an exact coupled solution of
the bound-constrained backward-Euler equations. Implicit time integration also
does not automatically remove spatial nonmonotonicity in Diamond or Taylor.

Run the focused regression tests with:

```powershell
julia --project=. --startup-file=no Tests/diamond_flux.jl
julia --project=. --startup-file=no Tests/density_projection.jl
julia --project=. --startup-file=no Tests/density_implicit.jl
julia --project=. --startup-file=no Tests/density_update_modes.jl
julia --project=. --startup-file=no Tests/taylor_reference_tests.jl
```

They check two-material flux with a discontinuous normal gradient, tangential
correction, orientation independence, uniform and locally refined meshes,
constant preservation, conservation, nodal affine reconstruction away from the
boundary, zero regularization, near-coplanar stencils, direct coarse/fine
interfaces, and Voronoi face geometry. The volume projection is compared with
an independent bisection solve over unequal cell volumes and strongly clipped
updates. See the validation report for integration and timing commands.
