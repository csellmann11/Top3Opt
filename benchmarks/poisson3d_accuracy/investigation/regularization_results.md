# Regularizing the weighted PU patch equation

This follow-up tests whether controlling weakly determined patch modes improves the [selected weighted-patch reconstructions](selected_weighted_results.md). The completed studies show that regularization can suppress the previously observed large contributions and improve effectivity on perturbed tetrahedral FEM. It does not repair every approximation error: the Cartesian VEM estimates remain substantially below one. The implementation is an isolated accuracy experiment, with no production changes or performance claim.

## Equation, normalization, and parameter selection

The patch space is unchanged: all existing P1 nodal values are fixed, added P2 edge values are free on artificial patch boundaries, and physical Dirichlet edge values are fixed. There is no point fitting. For the central PU hat `N_i`, trial edge shapes `phi_j`, and residual `R_U(v)=F(v)-a(U,v)`, assemble

\[
 A_{jk}=a(\phi_k,N_i\phi_j),\quad b_j=R_{U_{\rm rhs}}(N_i\phi_j),\quad
 g_j=R_U(N_i\phi_j),
\]
\[
 K_{jk}=a(\phi_k,\phi_j),\qquad H_{jk}=a(N_i\phi_k,N_i\phi_j).
\]

`K` and `H` are the trial and weighted-test energy Gram matrices. With their Cholesky factors `K=L_k L_k^T` and `H=L_h L_h^T`, define

\[
 B=L_h^{-1}AL_k^{-T},\qquad c=L_h^{-1}b,\qquad y=L_k^Td.
\]

Then `norm(y)` is the correction energy norm and `norm(c)` is the residual dual norm on the weighted test space. The singular values of `B` lie in `[0,1]` and measure the alignment of the trial and test spaces. They are invariant under invertible changes of their finite-element bases. The normalization changes the least-squares residual metric when the equations are incompatible; that change is separate from regularization. Exact barycentric integration is used for `A`, `K`, and `H`, with polynomial degrees three, two, and four respectively.

For `B=P Sigma Q^T`, the normalized Tikhonov correction is

\[
 y_\alpha=\sum_j \frac{\sigma_j}{\sigma_j^2+\alpha^2}
 (p_j^Tc)q_j,\qquad d_\alpha=L_k^{-T}y_\alpha.
\]

It minimizes `norm(B*y-c)^2 + alpha^2*norm(y)^2` on the numerically retained space. The declared numerical-rank floor is `sigma/sigma_max > 1e-12` for all filters. Modes below that floor are excluded before deliberate filtering; this numerical safeguard is reported separately from the chosen regularization strength.

The **15 default specifications were fixed before evaluating the comparison**:

| Normalization | Filter | Parameters |
| --- | --- | --- |
| Raw coefficient basis | TSVD | Relative cutoffs `1e-11, 1e-8, 1e-6, 1e-4` |
| Energy | TSVD | Relative cutoffs `1e-12, 1e-8, 1e-6, 1e-4, 1e-2` |
| Energy | Tikhonov | `alpha/sigma_max = 1e-8, 1e-6, 1e-4, 1e-2` |
| Energy | Adaptive gain cap | Predetermined `G=10,100` |

The normalized TSVD at the numerical floor is the undamped reference. Fixed spectral parameters are relative to each patch's largest singular value. The gain-cap rule instead chooses the **smallest** `alpha` such that

\[
 \|y_\alpha\|\le G\|c\|.
\]

It accepts the numerical-rank pseudoinverse if this already satisfies the bound; otherwise it finds `alpha` by bisection. The filter bound `sigma/(sigma^2+alpha^2) <= 1/(2 alpha)` supplies the upper bracket `alpha=1/(2G)`. Zero forcing returns zero correction. Neither the exact solution nor effectivity enters this decision.

The cap guarantees a bound on the computed readout,

\[
 |g^Td|\le G\sqrt{g^TK^{-1}g}\sqrt{b^TH^{-1}b},
\]

but **does not bound estimator error**. The test-space residual norm is only a lower bound on the true local error norm; a legitimate weakly observed correction can therefore be damped. `G=10,100` are explicit stability assumptions, not universally validated constants or estimates of a known noise level. No parameter is selected by proximity of effectivity to one.

## DWR readout and diagnostics

The original residual and readout are retained after filtering. For energy,

\[
 \eta=B_{\rm common}+\tfrac12\sum_i g_i^Td_i,\qquad
 B_{\rm common}=F(U)-\tfrac12a(U,U)-J_h.
\]

For `L(u)=integral u`, patches use the independently solved dual problem `-Delta z=1` and its residual as forcing, while the readout remains the primal residual. The estimator is `R_U(Z_h)+sum_i R_U(N_i delta z_i)`, with the coarse term included once and no half factor.

Both `A*d-b` and its energy-normalized norm are recorded. A small regularized equation residual is not an accuracy certificate. Combined residuals and sums of patch correction energies are overlapping-patch diagnostics, not global residual or reconstructed-field energy norms. Raw patch readouts precede the energy half factor and common correction. Spatial marking is assessed separately from signed effectivity.

## Validation and study scope

**All 359 independent checks passed** in [test_regularized_weighted_patches.jl](test_regularized_weighted_patches.jl). They check the energy Gram integration, normalization and filter algebra, basis and forcing scaling, original-residual/readout evaluation, and the gain constraint. The earlier selected-patch tests independently check the weighted equation and separate primal/dual handling.

A separate [45-row consistency diagnostic](../results/patch_regularization_consistency.csv), reproduced by [diagnose_regularization_consistency.jl](diagnose_regularization_consistency.jl), separates field error from DWR readout error on exact quadratic data. The normalized undamped readout defects are approximately `0%, 0.010462%, 0.640342%` on the regular, perturbed, and refined patches; Tikhonov `1e-4` gives `0.000003423%, 0.013935%, 0.680420%`. Stronger damping at `1e-2` raises the refined defect to `4.968619%`, demonstrating possible bias. Both gain caps are inactive on these particular quadratic patches, so this test does not measure their bias when active. Affine readouts remain below `7.674e-33`. The reused 315 selected-patch checks also passed.

The predeclared main study uses `n=4,6`, smooth cube energy and integral goals, and singular L-beam energy, with all three primal sources. **The main study and all independent validation runs completed: 690 accuracy rows, all finite effectivities, and no duplicate case/method keys.** Every case evaluates the same fixed parameter grid plus the existing baselines.

| Completed data | Rows | Scope |
| --- | ---: | --- |
| [Main](../results/patch_regularization/accuracy.csv) | 328 | `n=4,6`, cube and singular L-beam, all three sources |
| [Smoke](../results/patch_regularization_smoke/accuracy.csv) | 109 | Cube `n=2`, all three sources |
| [Finer mesh](../results/patch_regularization_fine/accuracy.csv) | 73 | Cube `n=8`, Cartesian VEM and perturbed FEM |
| [Amplitude 0](../results/patch_regularization_amplitude_0.0/accuracy.csv), [0.04](../results/patch_regularization_amplitude_0.04/accuracy.csv), [0.12](../results/patch_regularization_amplitude_0.12/accuracy.csv) | 36 each | FEM cube `n=6` |
| [Smooth L-beam](../results/patch_regularization_smooth_l/accuracy.csv) | 72 | `n=4`, perturbed and locally refined FEM |

- `cartesian_vem` is the original first-order hexahedral VEM with projected loads and stabilization, evaluated through its explicit fan reconstruction.
- `irregular_tet_fem` is conforming P1 FEM on a deterministically perturbed tetrahedral fan, with physical boundary nodes fixed.
- `refined_tet_fem` is that FEM mesh with every seventh tetrahedron split into four at its interior barycenter, preserving conforming exterior faces.

The FEM controls are not arbitrary-polyhedral VEM validation. Coefficient jumps, 3D elasticity, and completed adaptive refinement sequences are outside this experiment. The exact smooth cube solution is `x(1-x)y(1-y)z(1-z)`; all forcing and goal references come from the existing manufactured problems, not fitted data.

## Accuracy and stability results

The following completed rows are from [the main accuracy data](../results/patch_regularization/accuracy.csv) and [the finer-mesh validation](../results/patch_regularization_fine/accuracy.csv). They compare the same primal solutions and unchanged DWR readout. They are examples from the predeclared grid, not a rule selecting the best method from the exact error.

| Case / effectivity | Raw TSVD `1e-11` | Normalized Tikhonov `1e-4` | Gain cap `G=10` |
| --- | ---: | ---: | ---: |
| Perturbed FEM cube, `n=6`, energy | 0.252012 | 0.959695 | 0.957174 |
| Perturbed FEM cube, `n=6`, integral | 0.897953 | 0.949853 | 0.946825 |
| Refined FEM cube, `n=6`, energy | 0.868024 | 0.931521 | 0.928805 |
| Refined FEM cube, `n=6`, integral | 0.898380 | 0.917651 | 0.914179 |
| Perturbed FEM singular L, `n=6`, energy | 0.915366 | 0.919748 | 0.918651 |
| Perturbed FEM cube, `n=8`, energy | 0.988998 | 0.966111 | 0.964276 |

For the perturbed cube energy case at `n=6`, the maximum patch energy amplification falls from about `91,999` for the raw method to at most `10` for the gain-capped method. The combined energy-normalized equation residual changes only from about `0.003897` to `0.004128`. A small change in residual can therefore accompany a large reduction in an unstable correction.

The `n=8` holdout shows that regularization does **not** monotonically improve effectivity case by case: the raw value is closer to one there. On cube FEM `n=6`, gain cap 10 gives energy effectivities `0.956475`, `0.957174`, and `0.956336` for perturbation amplitudes `0.04`, `0.08`, and `0.12`. This supports stability across these geometry changes, not an accuracy theorem.

On the smooth L-beam holdout, gain cap 10 changes perturbed-FEM energy effectivity from `0.868819` to `0.943497` and refined-FEM energy from `0.855697` to `0.906996`. The corresponding integral values change from `0.938049` to `0.939561` and from `0.914124` to `0.907875`. The last value is another case where regularization does not improve effectivity. On the unperturbed FEM cube `n=6`, the cap leaves energy/integral effectivities at `0.953491/0.944812`.

For the perturbed cube at `n=6`, normalized indicator-distribution L1 error improves from `0.83904` to `0.06189` with fixed Tikhonov and `0.06232` with cap 10. Top-20% capture changes from `36.21%` to `36.78%` and `36.79%`. These compare indicators with positive reconstructed-field energy error; they do not measure capture of signed VEM goal error or prove an adaptive-convergence result.

For Cartesian VEM cube energy at `n=6`, gain cap 10 leaves effectivity at `0.787076`: the undamped normalized maximum gain is only about `2.59`, so the cap is inactive. The corresponding singular L-beam VEM effectivity remains about `0.751464`, and the finer cube remains about `0.823723`. Stabilizing weak modes alone does not remove the separate approximation limitations of the patch space and reconstruction.

The refined singular L-beam at `n=6` likewise retains appreciable underestimation: cap 10 gives `0.885634`, versus `0.973548` for the global P2 reference. **Energy-normalized Tikhonov with the residual-based gain cap 10 is a defensible candidate for controlling patch instability.** The fixed relative parameter `1e-4` gives similar accuracy in these comparisons. Neither is established as a universally accurate DWR estimator; further work must address approximation bias as well as stability.

## Reproduction

Run from the repository root. The default comparison writes settings, per-case patch/cell diagnostics, and the accumulating accuracy table. Separate output directories preserve amplitude controls.

```powershell
julia --project=benchmarks/poisson3d_accuracy --startup-file=no benchmarks/poisson3d_accuracy/investigation/test_regularized_weighted_patches.jl
julia --project=benchmarks/poisson3d_accuracy --startup-file=no benchmarks/poisson3d_accuracy/investigation/run_patch_regularization.jl --output=benchmarks/poisson3d_accuracy/results/reproduce_patch_regularization
```

For the same runner, use these argument changes and a distinct `--output` for each:

| Validation case | Arguments |
| --- | --- |
| Finer cube | `--levels=8 --cases=cube_smooth --sources=cartesian_vem,irregular_tet_fem` |
| Perturbation control | `--levels=6 --cases=cube_smooth --sources=irregular_tet_fem --amplitude=0` and repeat with `0.04,0.12` separately |
| Smooth L-beam | `--levels=4 --cases=lshape_smooth --sources=irregular_tet_fem,refined_tet_fem` |

The implementation is [regularized_weighted_patches.jl](regularized_weighted_patches.jl), driven by [run_patch_regularization.jl](run_patch_regularization.jl). Each patch reuses its Gram factorizations and SVDs across all parameter specifications. Primal and dual passes currently rebuild them. Setup on every changed mesh, any reusable geometry work, and the primal/dual costs must all be counted in a later performance study.
