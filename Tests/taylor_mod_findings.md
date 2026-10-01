# Taylor regularization and adaptive MOD: controlled investigation

Date: 2026-10-01. Production ToOpt3 revision: `365ad0ae3889d92dca6118295e684bfc241db473`.
No production solver, regularization, mesh-transfer, or cluster settings were changed.

## Finding

The lower adaptive MOD is reproducible. For the controlled MBB hexahedral
case at refinement level 3, most of the difference comes from Taylor's added
coefficient-gradient term. Physical versus rescaled neighbor distances make
a smaller contribution. The rescaled strong-form control nearly restores the
constant-mesh MOD, consistent with the December results.

Taylor is not using a smaller scalar `beta0`. Both schemes use the same
normalized coefficient `beta_hat[i] = 2 * beta0 * h[i]^2`. The discrete operators
are different on a graded mesh:

\[
R_{\rm Taylor}\chi
=\widehat\beta L_{\rm physical}\chi
+G\widehat\beta\cdot G\chi,
\qquad
R_{\rm strong}\chi=\widehat\beta L\chi.
\]

For the strong scheme, `L` can use physical or rescaled neighbor positions.
Taylor always uses physical positions. On a uniform hexahedral mesh,
`G * beta_hat` vanishes and the three operators agree numerically. On an
adaptive mesh, the cell diameter and therefore `beta_hat` jump at coarse/fine
transitions. A 2:1 diameter ratio gives a 4:1 coefficient ratio.

## Controlled 100-iteration optimization

The diagnostic uses the production MBB geometry (3 by 0.5 by 1), an initially
uniform 6144-cell hex mesh at refinement level 3, target density 0.3,
`chi_min=0.001`, `eta0=15`, `beta0=1`, and eight density substeps. Assembly,
driving force, volume projection, marking, and density transfer use the existing
production functions. All runs execute exactly 100 iterations.

A sparse direct displacement solve replaces the cluster HYPRE solve. All
other settings are shared. One fixed-mesh baseline suffices because the
initial uniform operators agree: maximum absolute matrix differences are
`1.42e-14` (physical strong form) and `1.99e-11` (rescaled strong form).

| Mesh and operator | MOD at iteration 100 | Relative to fixed mesh | Final cells |
|---|---:|---:|---:|
| Fixed, Taylor; equivalent uniform strong operators | 0.2896702754 | 0.000% | 6144 |
| Adaptive, Taylor | 0.2703643241 | -6.665% | 4037 |
| Adaptive, strong with physical distances | 0.2861093331 | -1.229% | 4198 |
| Adaptive, strong with rescaled distances | 0.2898678012 | +0.068% | 4170 |

Removing the coefficient-gradient term while retaining physical distances
raises MOD by 0.0157450090. Adding rescaling then raises it by another
0.0037584682. The first change accounts for about 81% of the total Taylor to
rescaled-strong endpoint difference in this experiment. This is a sequential
comparison of coupled optimization trajectories, not a universal linear
decomposition: the meshes and density fields evolve differently after the
operator is changed.

The replay matches the cluster Taylor endpoint MOD to `1.99e-7` on the fixed
mesh and `9.31e-8` on the adaptive mesh. Across all 100 recorded iterations,
the largest absolute differences are `6.38e-6` and `4.66e-6`, respectively.

Across all four runs, density transfer changes MOD by at most `3.33e-16`
and the volume fraction by at most `1.67e-16`. Transfer itself therefore
does not explain the observed gap in this hexahedral test.

Raw results: `Results/taylor_mod_diagnostics/optimization_r3_s100.csv`.
Compact endpoints: `Results/taylor_mod_diagnostics/optimization_summary.csv`.

## Interpretation of the coefficient-gradient term

The sign of `G(beta_hat) dot G(chi)` is not fixed. For a fine gray cell next
to a coarse void region, the coefficient increases toward the coarse cell,
while density decreases in that direction. Their gradient dot product can
be negative, pulling density toward the void value and opposing part of
the spreading from the Laplacian term. Next to a coarse solid region, the
corresponding contribution can pull density toward the solid value. This
provides a mechanism for shorter gray tails and lower integrated MOD.

This is a local explanation of the measured effect, not a claim that Taylor
is uniformly weaker everywhere. The coefficient is actually larger in coarse
cells, and some manufactured profiles are smoothed more strongly than on a
uniform fine mesh. MOD also depends on the total amount and geometry of gray
material, not solely on a single interface width.

For smooth coefficients, the product-rule term belongs to
`div(beta_hat * grad(chi))`. However, the old row-scaled strong Laplacian and
this variable-coefficient divergence model are different operators. Also,
mesh-dependent `h^2` changes the coefficient field when the mesh adapts;
using the same `beta0` does not make the two mesh treatments equivalent.
Voronoi cells can already have different diameters on a fixed mesh, so a
coefficient gradient is not exclusive to adaptive Voronoi meshes.

## Additional isolated checks

The planar-profile test uses identical fine cells around the transition and
coarser cells farther away. For a tanh profile centered at x=1.25 with width
0.08, the initial graded-mesh MOD is 0.0507643813. One diffusion-only update
(eight substeps, including the usual volume projection) increases MOD by:

| Operator on the same graded mesh | Increase in MOD |
|---|---:|
| Taylor, default ridge 1e-4 | 0.0445293216 |
| Taylor, ridge 1e-8 | 0.0446114894 |
| Strong, physical distances | 0.0502091975 |
| Strong, rescaled distances | 0.0502535901 |

Taylor gives about 11.4% less smoothing by this measure than rescaled strong
form. Reducing ridge stabilization by a factor of 10000 changes the Taylor
increment by only about 0.18%. Ridge is not the leading explanation on this
fixture. Results vary with interface position; the script also tests centers
at x=1.0 and x=1.5 and a wider profile.

A second test applies the different operators to the same saved 5290-cell
MBB state and the same displacement solution. It confirms small differences
in a single early density update (Taylor MOD 0.5150707 versus rescaled strong
0.5151260). This older saved state is not a snapshot of the cluster batch.
The 100-iteration experiment is the evidence for the accumulated effect.

The Taylor operator preserves constants but does not exactly conserve the
volume-weighted integral on the graded fixtures, and it has some negative
off-diagonal coefficients. Those are numerical limitations worth retaining
in future operator validation. They are not, by themselves, proof of the MOD
cause; the operator-removal comparison above is the more direct evidence.
The strong schemes also have conservation defects, and the corrected diamond
scheme is not generally monotone either.

## Historical comparison

December filenames encode `l1` (rescaling enabled). The `s500` filename denotes
the requested limit, but the actual CSV histories contain 70--107 iterations.
The following comparison uses the same recorded iteration for each pair:

| RL | Iteration | December fixed | December adaptive | Current fixed | Current adaptive Taylor |
|---|---:|---:|---:|---:|---:|
| 3 | 99 | 0.28963465 | 0.28981492 | 0.28962805 | 0.27040452 |
| 4 | 75 | 0.15495556 | 0.15552254 | 0.15495239 | 0.14209720 |
| 5 | 70 | 0.09706700 | 0.09903728 | 0.09707595 | 0.07686011 |

This supports the user's observation: the fixed-mesh histories remain close,
while the adaptive results changed substantially.

The L-cantilever driver intentionally forces `Lquad_mesh`; jobs labeled
Hexahedra and Voronoi have identical MOD histories in the supplied batch.
They are not independent mesh-family evidence. The two MBB families are.

## Scope and practical implication

The controlled optimization result covers MBB hex at RL=3. It identifies a
causal operator difference there; the relative contribution has not been
remeasured at RL=4/5 or on Voronoi meshes. None of these comparisons depends
on a convergence declaration: all new optimization endpoints are at iteration 100.

For reproducing the old adaptive/fixed correspondence, the verified reference
is `--flux_scheme strong --laplace_rescale true`. The current cluster sweep
hardcodes `--laplace_rescale false`, so changing only `FLUX_SCHEME` would not
select that reference. Taylor ignores the rescaling flag.

Increasing `beta0` globally would change the already matching fixed-mesh
result too. A Taylor modification should instead explicitly address the
intended mesh dependence of the coefficient and coarse/fine treatment, and
be validated against both fixed and adaptive results.

## Reproduction and files

Run from the ToOpt3 root with its existing Julia environment. The diagnostic
sets BLAS to one thread; setting `OPENBLAS_NUM_THREADS=1` before launching
Julia also limits startup memory when other Julia processes are active.

```sh
julia --project=. --startup-file=no Tests/taylor_mod_diagnostics.jl operators
julia --project=. --startup-file=no Tests/taylor_mod_diagnostics.jl optimize 100 3
# Optional: requires the existing Results/bad_diamond_stencil.bin fixture.
julia --project=. --startup-file=no Tests/taylor_mod_diagnostics.jl frozen
```

Generated outputs are under the ignored `Results/taylor_mod_diagnostics/`
directory. The investigation also saved `historical_comparison.csv` and
`optimization_summary.csv` there as summaries of existing and generated data.

Relevant production code:

- `src/taylor_operator.jl`: physical positions, local beta, and coefficient-gradient term.
- `src/laplace_operator.jl`: strong Laplacian and neighbor-distance rescaling.
- `src/bisection.jl`: shared normalized density update and volume projection.
- `src/density_timestepping.jl`: common eight-substep rule for these parameters.
- `src/mat_states.jl`: volume-weighted MOD and density transfer.
