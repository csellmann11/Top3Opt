# 3D density regularization

The default is now `--flux_scheme diamond`. For comparisons, select
`--flux_scheme tpfa` or `--flux_scheme strong`. The Julia entry point accepts
`run_optimization(...; flux_scheme=:diamond)`. `laplace_rescale` applies only to
the legacy `strong` scheme. Output directory identifiers include the flux scheme.

The face schemes assemble `div(beta_hat * grad(chi))`, with the existing 3D
coefficient `beta_hat[i] = 2 * beta0 * h[i]^2`. The density update multiplies this
by `p_avg` once. The operator is rebuilt after mesh adaptation, so both geometry
and the spatially varying coefficient follow the current mesh.

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
substeps reuse this assembled sparse operator with `mul!`.

The operator preserves constants and conserves the volume-weighted total.
It reduces to TPFA on orthogonal faces. Like the reference correction, it is not
guaranteed symmetric or monotone. Planar faces and cell centers on opposite sides
of each interior face are required. The regression suite tests fixed-step
diffusion stability on specific unbalanced hex meshes up to a 32:1 direct size
ratio; this is not a stability proof for arbitrary distorted polyhedra.

The original substep rule is preserved:
`max(1, 4*ceil(Int, 24*beta0/eta0))`, which gives eight substeps for `beta0=1`
and `eta0=15`. The operator row norm is logged only as a diagnostic and **never
increases the substep count**. A nonfinite operator causes an immediate error.

The scalar multiplier is found by projecting the complete unconstrained Euler
update onto `[chi_min,1]` with the requested volume-weighted mean. The bracket
includes both the driving and regularization terms. Safeguarded Newton steps
use the volume of currently unclipped cells, with bisection as a fallback and
a finite iteration limit. This solves the same clipped explicit update as
bisection; it does not change the evolution equation or introduce implicit steps.
If every cell is at a bound, the undefined `g`-weighted force average falls back
to its volume-weighted average. Zero average driving force leaves density unchanged.

Run the focused regression tests with:

```powershell
julia --project=. --startup-file=no Tests/diamond_flux.jl
julia --project=. --startup-file=no Tests/density_projection.jl
julia --project=. --startup-file=no Tests/taylor_reference_tests.jl
```

They check two-material flux with a discontinuous normal gradient, tangential
correction, orientation independence, uniform and locally refined meshes,
constant preservation, conservation, nodal affine reconstruction away from the
boundary, zero regularization, near-coplanar stencils, direct coarse/fine
interfaces, and Voronoi face geometry. The volume projection is compared with
an independent bisection solve over unequal cell volumes and strongly clipped
updates. See the validation report for integration and timing commands.
