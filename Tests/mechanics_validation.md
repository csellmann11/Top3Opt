# 3D mechanics regression, September 2026

The reproduced failure is in the elastic projection/assembly path, before the
density update or diamond regularization. Fixes are needed in **both ToOpt3 and
its locally developed Ju3VEM dependency** (`../Ju3VEM`). Updating ToOpt3 alone
does not repair the projectors.

## Reproduction and cause

With Julia 1.12.6 and the current dependency environment, the calls
`Octavian.matmul!(destination, inverse_static_matrix, B)` in Ju3VEM's face and
volume projectors return incorrect coefficients. Dense reference multiplication
with the same operands returns the correct answer. This isolates the observed
failure to that multiplication path; the precise upstream compiler/package
change that first introduced it has not been bisected.

For a unit cube with linear elements, the production volume projector gives
`Pi_star * D = diag(1, 1/9, 1/9, 1/9)` instead of identity. The face projector
gives `diag(1, 1/4, 1/4)`. Constants survive, so constant-field tests alone miss
the error. Linear fields, rotations, and elastic strains do not survive.

For an affine vector field at density 0.3, with E=210000 and nu=0.33:

| Quantity | Before | Corrected |
|---|---:|---:|
| Unit-cube elastic energy (exact: 17.058900928792568) | 221.0165160777152 | 17.05890092879302 |
| Unit-cube affine energy relative error | 11.9561 | 2.64e-14 |
| Relative rigid-rotation force, `norm(K*u)/(norm(K)*norm(u))` | 0.1330 | below 2.4e-17 |
| Affine energy relative error on a locally refined mesh | 16.7817 | 1.60e-13 |

In a 768-cell uniform MBB solve, the projected physical energy represented only
about 1.86e-7 of the computed elastic energy before the fix. Stabilization was
effectively carrying the response. Penalizing a rigid rotation and suppressing
the physical strains explains why a converged linear solve could still feed a
physically wrong load-to-support topology into the optimizer.

The saved September 16 run already has a different trajectory from the saved
January 6 run, before diamond flux was introduced on September 29. This also
rules out diamond as the sole explanation of this regression.

## Changes

- Ju3VEM: use `LinearAlgebra.mul!` for the immutable static inverse operands in
  `src/face_projector.jl` and `src/volume_projector.jl`.
- ToOpt3: use `mul!` for the static polynomial stiffness operand in
  `src/compute_displacement.jl`, avoiding the same multiplication path.
- Check face and volume polynomial reproduction on the first active cell at
  optimization startup. An outdated/broken linked VEM library now stops with
  an explicit error instead of silently producing an invalid optimization.
- Add upstream projector tests and application elasticity tests. The existing
  boundary conditions, material law, density update, and flux are unchanged.

## Regression checks

From the ToOpt3 root:

```sh
julia --project=. --startup-file=no Tests/mechanics.jl
julia --project=. --startup-file=no ../Ju3VEM/tests/projector_regression.jl
```

The application suite passes **608 assertions**, including all six rigid
motions, affine strains and analytical energies, face/volume reproduction, and
rejection of an intentionally corrupted projector. Meshes include a cube,
small anisotropic cells, hanging nodes, skew cells, and extruded Voronoi cells.
The upstream suite passes **189 assertions**, including quadratic projectors.

Both fixed-mesh MBB and L-cantilever validations completed **200 iterations**
on 6,144-cell meshes with HYPRE and diamond regularization. Density bounds and
volume fractions passed at every iteration. Final energies relative to the
uniform initial design were 0.0839494 (MBB) and 0.0178689 (L). At the initial
and final designs, HYPRE's energies agreed with direct solves to better than
9e-9 relative error.

With the original `:strong` regularization, the corrected 20-step MBB run
reproduced the saved January 6, 2026 energy trajectory within **3.32e-7 relative
error** and nondiscreteness within **2.08e-7 absolute error**. This is a comparison
against an existing historical run, not just internal consistency checks.

The full cluster sweep and long adaptive optimization reruns remain to be
checked on the cluster. Elasticity on locally refined/hanging-node meshes is
covered by the passing regression suite above.

## Separate sensitivity limitation

The existing thermodynamic update uses the projected physical strain energy.
It omits the density derivative of VEM stabilization, so it is not the exact
gradient of the full discrete compliance. This predates the multiplication
regression and has been left unchanged to preserve the existing formulation.
In the corrected 768-cell MBB probe, stabilization contributes 27.5% of the
energy; one tested volume-preserving directional derivative differs by 39.4%
from the full-discrete finite difference. Switching to a discrete-compliance
gradient would be a separate modeling change, not a fix for the lost polynomial
reproduction.

Local exploratory scripts, logs, density CSVs, and VTK snapshots are retained
under the ignored `Results/` directory. The checks above do not require those
artifacts or a cluster connection.
