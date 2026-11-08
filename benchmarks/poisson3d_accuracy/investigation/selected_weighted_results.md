# Selected weighted-patch reconstructions: measured results

These experiments test the proposed local equation with free artificial-boundary edge values, followed by three explicit rules for choosing a reconstructed field. They are actual accuracy comparisons, including an independent dual problem, irregular geometry, changed tetrahedral connectivity, and controls. **The results do not support these three selectors as a reliable improvement across the tested cases.** This is not a rejection of nonunique PU reconstructions: the most damaging measured patch is full rank, and all three selectors therefore give the same unstable solution there.

The code remains in the isolated Poisson laboratory. It uses ordinary Julia functions and no point fitting. No production topology-optimization implementation was changed by these experiments, and no runtime advantage is claimed.

## Local equation and three selection rules

Let `U` be the explicit conforming P1 primal field on the tetrahedral fan, `N_i` its central hat function, and `omega_i` the corresponding tetrahedral star. With `a(w,v)=integral grad(w).grad(v)` and `R_U(v)=F(v)-a(U,v)`, the proposed equation is

\[
 a_{\omega_i}(U+\delta_i,N_i v)=F_{\omega_i}(N_i v).
\]

The correction is continuous piecewise P2 on the patch. **Every existing P1 value is fixed**, including original vertex, face-center, cell-center, and subsequently inserted refinement nodes. Every added P2 edge midpoint is free except those on the physical Dirichlet boundary. In particular, edge midpoints on artificial patch boundaries are free. Multiplication by `N_i` makes the final PU correction vanish at those artificial boundaries. This is a finite-element space defined by mesh connectivity, not a polynomial fit to neighboring samples.

For its free edge shapes `phi_j`, assemble

\[
 A_{jk}=a(\phi_k,N_i\phi_j),\qquad
 b_j=R_{U_{\rm rhs}}(N_i\phi_j),\qquad
 g_j=R_U(N_i\phi_j),
\]
\[
 K_{jk}=a(\phi_k,\phi_j),\qquad
 \ell_j=a(U_{\rm rhs},\phi_j).
\]

`A` is generally nonsymmetric. `K` is positive definite on the edge-correction space: a constant correction cannot vanish at every existing P1 node unless it is zero. For energy, `U_rhs=U` and `b=g`; for the linear goal, the solve uses the dual field and dual forcing while `g` remains the primal residual.

Write the retained singular-value decomposition as `A_tau=L_r Sigma_r V_r'`, let `Z` contain the discarded right singular vectors, and set

\[
 d_0=V_r\Sigma_r^{-1}L_r^Tb,\qquad b_{\rm proj}=L_rL_r^Tb.
\]

The three representatives of `d_0+Zc` are:

1. **Minimum coefficient norm:**
   \[
   d_C=d_0=\arg\min_{d\in d_0+\operatorname{range}Z}\|d\|_2.
   \]
2. **Minimum correction energy:**
   \[
   d_E=d_0-Z(Z^TKZ)^{-1}Z^TKd_0,
   \qquad \min \tfrac12d^TKd.
   \]
3. **Minimum total Dirichlet energy:**
   \[
   d_T=d_0-Z(Z^TKZ)^{-1}Z^T(Kd_0+\ell),
   \qquad \min \bigl(\tfrac12d^TKd+\ell^Td\bigr).
   \]

The third objective is the change in `a(U_rhs+delta,U_rhs+delta)/2`, **without** a body-load potential. It is not minimization of the full Poisson potential energy. The Euclidean coefficient norm and least-squares norm use the standard, unscaled P2 edge basis; there is no row or column equilibration.

Compatibility is not assumed. The solve projects `b` onto the retained left-singular space and reports `norm(b-b_proj)`, together with the actual residual `A*d-b` for each representative. When discarded singular values are genuinely zero, these are least-squares minimizers for the original operator. Discarding small nonzero singular values instead gives a truncated-SVD regularization of that operator. The main cutoff is `sigma/sigma_max > 1e-11`.

**The DWR readout is not projected:** it remains `g'*d`, using the original residual. For the energy goal,

\[
 J(u)-J_h=B+\tfrac12a(u-U,u-U),\qquad
 B=F(U)-\tfrac12a(U,U)-J_h,
\]
\[
 \eta=B+\tfrac12\sum_i g_i^Td_i.
\]

This retains the same load, consistency, and stabilization conversion as the other lab estimators. For `L(u)=integral u`, an independent dual solves `-Delta z=1`; the estimator is

\[
 \eta_L=R_U(Z_h)+\sum_i R_U(N_i\delta z_i).
\]

The coarse dual residual is included once and there is no energy-goal half factor. The direct P2 linear reference is also checked against the independently integrated primal correction by reciprocity.

A unique local field is unnecessary. With a true nullspace `Z`, the readout is invariant across representatives exactly when `Z'*g=0`. That condition makes the estimator well-defined but does not establish its accuracy. A deliberately chosen stable representative can also be useful without invariance, provided its DWR accuracy is validated. The following tests assess three such choices rather than assuming uniqueness is required.

## Completed studies and independent checks

The smooth cube uses `u=x(1-x)y(1-y)z(1-z)` and its exact manufactured forcing `f=-Delta u`; its exact energy is `1/1800` and integral goal is `1/216`. The singular L-beam uses the existing manufactured singular solution and forcing from [problems.jl](../problems.jl). Only the energy goal is evaluated for that singular case.

| Data directory | Completed accuracy rows | Scope |
| --- | ---: | --- |
| [selected_weighted](../results/selected_weighted/accuracy.csv) | 112 | `n=4,6`, cube energy/integral and singular L-beam energy, all three primal sources |
| [selected_weighted_smoke](../results/selected_weighted_smoke/accuracy.csv) | 37 | `n=2` cube, both goals, all three sources |
| [selected_weighted_regular_fem](../results/selected_weighted_regular_fem/accuracy.csv) | 24 | Regular, unperturbed conforming P1 FEM, cube `n=4,6` |
| [selected_weighted_tolerances](../results/selected_weighted_tolerances/accuracy.csv) | 48 | Perturbed/refined cube `n=4`, cutoffs `1e-9,1e-11,1e-13` |
| [selected_weighted_quadrature](../results/selected_weighted_quadrature/accuracy.csv) | 6 | Perturbed singular L-beam `n=4`, increased quadrature |
| [selected_weighted_fixed_vertices](../results/selected_weighted_fixed_vertices/accuracy.csv) | 14 | Seven paired global P2 solves, with and without fixed existing P1 values |

The first five studies have no nonfinite effectivities or duplicate source/problem/level/goal/method/cutoff keys. Each main study directory also records cell contributions, patch diagnostics, and settings. The default load quadrature order is 6 and truth integration order is 10. Patch `A` is integrated using exact cubic barycentric moments; the quadrature control concerns the forcing and independent truth integrations, not an approximate integration of that matrix.

The **315 independent checks** in [test_selected_weighted_patches.jl](test_selected_weighted_patches.jl) passed. They cover direct quadrature versus the independently assembled matrix, stiffness, forcing, and internal-energy vector; reconstructed-field readout; physical boundary traces; residual preservation; all three selection stationarity conditions; distinct primal/dual forcing; and rank/projection behavior. On regular, perturbed, and locally refined meshes, an admissible exact quadratic correction satisfies the weighted equation to about `4e-16` absolute residual. A selected representative need not reproduce that quadratic field: equation consistency, field reproduction, and DWR accuracy are separate properties. The affine tests likewise check the readout separately from the selected field.

The quadratic readout check directly demonstrates why full-field uniqueness is
unnecessary. On the regular interior patch, the selected edge coefficients
differ from the exact quadratic correction by norm `0.03118`, yet all three
raw DWR readout defects are only `1.42e-17`. On the perturbed patch, the relative
readout defects are about `0.0213%`, `0.0105%`, and `0.997%` for coefficient,
correction-energy, and total-energy selection. On the locally refined patch
they are `0.124%`, `0.640%`, and `12.63%`. Thus accurate readout without field
reproduction is possible, but is not guaranteed across these patch geometries.

The sources must be distinguished:

- **Cartesian VEM:** the original first-order hexahedral VEM, including projected load and stabilization, with an explicit tetrahedral-fan field for residual evaluation.
- **Perturbed FEM:** conforming P1 FEM on a deterministically perturbed tetrahedral fan, amplitude `0.08`, with physical boundary points fixed.
- **Refined FEM:** that perturbed FEM mesh with every seventh tetrahedron split into four at its interior barycenter. Outer triangular faces are unchanged, so the new connectivity is conforming and has variable star valence.

The two FEM controls do **not** validate arbitrary-polyhedral VEM. Parent-cell labels are used only for aggregation. These experiments also do not test coefficient jumps or 3D elasticity.

## Accuracy: the representative choice changes little

Effectivity is the signed estimate divided by the true signed goal error. A range below is the minimum and maximum across the three selectors, not uncertainty or a confidence interval. `full PU` denotes the previous independent Dirichlet-patch reconstruction; `global P2` is the directly solved coupled reference.

| Primal source | Problem / goal | Selected, `n=4` | Selected, `n=6` | Full PU, `n=6` | Global P2, `n=6` |
| --- | --- | ---: | ---: | ---: | ---: |
| Cartesian VEM | Cube / energy | 0.70008 | 0.78708 | 0.82509 | 0.99878 |
| Cartesian VEM | Cube / integral | 0.66954 | 0.75407 | 0.79273 | 0.99845 |
| Cartesian VEM | Singular L / energy | 0.69275 | 0.75146 | 0.81763 | 0.99205 |
| Perturbed FEM | Cube / energy | 1.14047–1.14191 | **0.25061–0.25201** | 0.74314 | 0.99520 |
| Perturbed FEM | Cube / integral | 0.92239–0.92389 | 0.89654–0.89796 | 0.72968 | 0.99413 |
| Perturbed FEM | Singular L / energy | 1.07937–1.08082 | 0.91411–0.91549 | 0.72426 | 0.97351 |
| Refined FEM | Cube / energy | 0.93097–0.93369 | 0.86456–0.86802 | 0.68550 | 0.99523 |
| Refined FEM | Cube / integral | 0.89053–0.89323 | 0.89514–0.89838 | 0.67068 | 0.99415 |
| Refined FEM | Singular L / energy | 1.06499–1.06785 | 0.88424–0.88647 | 0.66280 | 0.97355 |

On the Cartesian cases the three readouts coincide to the reported precision. On the irregular cases their spread is small compared with the change between geometries and refinement levels. Several individual cases improve over the old full PU, but the perturbed-cube energy result deteriorates sharply on refinement. The integral goal does not show the same collapse, demonstrating why an energy-only comparison cannot settle general DWR behavior.

The original cheap sDWR is retained in the Cartesian energy data; its cube effectivity is `0.86659` at `n=4` and `1.01495` at `n=6`. These two levels neither establish its asymptotic effectivity nor make it a reference truth.

## What the controls isolate

**Regular FEM versus VEM.** Without coordinate perturbation, all selectors give FEM cube energy effectivities `0.94349 -> 0.95349` and integral effectivities `0.93592 -> 0.94481` from `n=4` to `n=6`. Thus the weaker Cartesian VEM results cannot be attributed simply to regular geometry. The primal discretization, reconstructed residual, and consistency contribution also matter. Perturbing the FEM geometry exposes an additional stability problem.

**Compatibility versus accuracy.** The combined relative projected-away energy right-hand side is small but nonzero: for the perturbed cube it is `0.00318` at `n=4` and `0.00168` at `n=6`. The cube dual problems have combined projected-away residuals around `1e-15`. Small equation residuals and compatible dual loads do not imply accurate DWR estimates. The CSV field `incompatible_patches` is a diagnostic count using relative residual greater than `1e-8` under the chosen numerical rank, not an exact mathematical classification.

**Rank cutoff and quadrature.** On the perturbed cube at `n=4`, changing the cutoff from `1e-11` to `1e-9` changes the minimum-coefficient energy effectivity from `1.14185` to `1.06066`. Cutoffs `1e-11` and `1e-13` agree there, and the refined `n=4` case is unchanged across all three cutoffs. This is sensitivity to retained near-null modes, not evidence that a particular cutoff is generally correct. Raising the singular L-beam load/truth quadrature from `6/10` to `8/12` changes the selected effectivities by about `1.21e-7`; it does not explain their much larger estimator errors.

**A full-rank patch causes the largest measured cancellation.** In the perturbed cube at `n=6`, node 998 has 16 free edge unknowns and rank 16. Its condition estimate is `3.94e8`, coefficient norm is `41.77`, and unhalved readout is `-2.18956e-5`. Its energy contribution alone is about `-0.64514` times the true goal error, while the whole minimum-coefficient estimate is only `4.27656e-6`. Node 521 is another full-rank patch with a negative contribution. Consequently, changing a representative within the exact nullspace cannot repair these dominant contributions: all three selectors already coincide on those patches.

The worst patch is an interior face-center star with eight tetrahedra, three physical-boundary vertices, and no free edge on the physical boundary. Independent direct quadrature reproduces its matrix and right-hand side to relative differences `1.02e-15` and `1.04e-15`. Direct integration of the readout and an independent LU solve reproduce the same large negative value. This supports a conditioning problem in the specified weighted patch equation rather than an assembly or SVD implementation error. The reproducible audit is [diagnose_selected_weighted_instability.jl](diagnose_selected_weighted_instability.jl).

The [whole-case cutoff audit](../results/selected_weighted_instability/global_n6.csv)
quantifies the effect of regularizing those weak modes on the same perturbed
cube at `n=6`:

| Relative singular-value cutoff | Minimum coefficient | Minimum correction energy | Minimum total Dirichlet energy |
| --- | ---: | ---: | ---: |
| `1e-11` | 0.252012 | 0.252010 | 0.250612 |
| `1e-8` | 0.841869 | 0.841866 | 0.840467 |
| `1e-6` | 0.958404 | 0.958402 | 0.956993 |

At node 998, the weakest singular value is `2.537705e-9` times the largest.
That single mode contributes `-0.645619` effectivity. At cutoff `1e-8`, its
retained rank drops from 16 to 15, coefficient norm falls from `41.7655` to
`0.000873706`, and raw readout changes to `1.62423e-8`. The relative residual
of the original equations is then `0.000560787`; the regularized solve is
explicitly allowed to leave those equations slightly unsatisfied. Detailed
[patch](../results/selected_weighted_instability/nodes_n6.csv) and
[mode](../results/selected_weighted_instability/modes_n6.csv) diagnostics are saved.
This is an exploratory stability result on one case, not a generally justified
cutoff or a validated replacement estimator. In particular, changing a rank
cutoff and selecting a representative of an exact nullspace are different
operations.

**Freezing every P1 value also loses correction space.** The global control solves the ordinary SPD residual equation with all coupling retained, either allowing every free P2 degree of freedom or allowing only edge corrections. The latter retains the same fixed-existing-node restriction as the proposed local reconstruction.

| Cube primal source | Global P2, `n=4` | Fixed-P1 global P2, `n=4` | Global P2, `n=6` | Fixed-P1 global P2, `n=6` |
| --- | ---: | ---: | ---: | ---: |
| Cartesian VEM | 0.99707 | 0.78352 | 0.99878 | 0.84284 |
| Perturbed FEM | 0.98931 | 0.94162 | 0.99520 | 0.94642 |
| Refined FEM | 0.98936 | 0.88907 | 0.99523 | 0.89474 |

For the perturbed singular L-beam at `n=4`, the corresponding values are `0.97350` and `0.92379`. All restricted solves have relative residual at most `5.5e-16`. The restriction has a measurable accuracy cost even without local patch boundaries, especially for VEM and the refined mesh. These values are **not an upper bound** on the weighted PU estimator: `sum N_i*delta_i` is piecewise cubic, and the weighted estimator is not that global P2 Galerkin correction.

## Spatial indicators are a separate test

The signed estimate and marking indicators are recorded separately. Each complete nodal contribution is converted to its magnitude and distributed to incident parent cells in proportion to their volumes; cell consistency terms retain the existing lab treatment. Distribution error is the L1 distance between normalized indicators and normalized positive reconstructed-field energy errors. Top-20% capture measures the fraction of that true positive error in the cells with the largest indicators. It is not the fraction of signed VEM goal error captured.

For the perturbed cube at `n=6`, the minimum-coefficient selector has distribution L1 error about **0.839**, versus **0.0583** for old full PU. Yet its top-20% capture is **36.21%**, versus **36.80%** for full PU and **36.81%** for the ideal ranking. Thus there is substantial distribution distortion without a comparably large collapse in this particular marking-capture metric. For the refined cube at `n=6`, selected distribution L1 is about `0.205`, versus `0.0581` for full PU, while capture is `36.60%` versus `36.93%`. For Cartesian singular L at `n=6`, selected capture falls to `32.09%`, versus `35.39%` for full PU and `41.09%` for the ideal ranking.

Good global effectivity does not establish good cell indicators, and the converse is also false. These are fixed-mesh diagnostics, not completed adaptive refinement sequences. No spatial accuracy claim is made for the linear goal without an independently justified spatial target.

## Reproduction and interpretation

Run from the repository root with the lab project. The first command runs the independent checks; the second reproduces the complete main comparison in a separate output directory.

```powershell
julia --project=benchmarks/poisson3d_accuracy --startup-file=no benchmarks/poisson3d_accuracy/investigation/test_selected_weighted_patches.jl
julia --project=benchmarks/poisson3d_accuracy --startup-file=no benchmarks/poisson3d_accuracy/investigation/run_selected_weighted.jl --levels=4,6 --cases=cube_smooth,lshape_singular --sources=cartesian_vem,irregular_tet_fem,refined_tet_fem --tolerances=1e-11 --output=benchmarks/poisson3d_accuracy/results/reproduce_selected_weighted
```

Use the same runner with the following settings to reproduce the controls; give each a separate `--output` directory.

| Control | Runner arguments |
| --- | --- |
| Smoke | `--levels=2 --cases=cube_smooth` |
| Regular FEM | `--levels=4,6 --cases=cube_smooth --sources=irregular_tet_fem --amplitude=0` |
| Cutoffs | `--levels=4 --cases=cube_smooth --sources=irregular_tet_fem,refined_tet_fem --tolerances=1e-9,1e-11,1e-13` |
| Singular quadrature | `--levels=4 --cases=lshape_singular --sources=irregular_tet_fem --quadrature=8 --truth-order=12` |

The fixed-value control is [diagnose_fixed_vertices.jl](diagnose_fixed_vertices.jl), and the representative implementation is [selected_weighted_patches.jl](selected_weighted_patches.jl). The default standalone diagnostic commands regenerate their respective diagnostic result files:

```powershell
julia --project=benchmarks/poisson3d_accuracy --startup-file=no benchmarks/poisson3d_accuracy/investigation/diagnose_fixed_vertices.jl
julia --project=benchmarks/poisson3d_accuracy --startup-file=no benchmarks/poisson3d_accuracy/investigation/diagnose_selected_weighted_instability.jl
```

The next question is stable approximation of the DWR readout, not uniqueness by itself. The three current representative rules do not address dangerous retained modes of a full-rank weighted operator, and the fixed-P1 restriction has a separate approximation cost. The earlier [coupled-enrichment study](README.md) therefore remains a candidate direction, while [the research notes](options_research.md) describe Neumann, equilibrated, and output-directed alternatives with their assumptions. Any future performance comparison must include setup and coarse/global solve costs each time the mesh changes; these experiments establish no runtime win.
