# Investigation of local and coupled DWR reconstructions

This directory extends the isolated Poisson accuracy laboratory. It uses plain Julia functions and standard libraries, includes no production topology-optimization code, and performs no point fitting. The implementations favor transparent equations and independent checks. **No runtime advantage has been established.**

The follow-up [selected weighted-patch experiments](selected_weighted_results.md)
test three explicit rules for selecting a local field when artificial boundary
edge values are free. They include full accuracy runs, independent assembly and
consistency tests, rank-cutoff controls, and a comparison with globally coupled
corrections that keep the original vertex values fixed.

The subsequent [regularization study](regularization_results.md) compares
singular-value cutoffs, energy-normalized Tikhonov damping, and residual-based
amplification limits on the same patch space and on additional validation meshes.

The measured results support **coupled conforming P2 correction with a small number of coarse-assisted residual cycles, followed by PU localization, as a candidate for further investigation**. They do not establish a universally sufficient cycle count, better adaptive marking, or suitability for arbitrary polyhedral VEM and 3D elasticity.

## What was compared

- `full_pu`, `reduced_pu`: the existing independent Dirichlet-patch reconstructions.
- `sdwr`: the existing projected VEM substitution, with its load and stabilization corrections. It is included only for the Cartesian VEM energy goal.
- `buffer_1`, `buffer_2`: enlarge a patch by one or two layers of face-adjacent tetrahedra, solve there, and retain the original inner PU readout.
- `direct`: one globally conforming P2 residual solve, with only physical Dirichlet constraints.
- `sgs_k`: k coupled symmetric Gauss-Seidel sweeps on that same P2 system.
- `two_level_k`: k cycles of forward smoothing, a global P1 coarse correction, and backward smoothing, again on the same P2 system.

Three primal controls are distinguished:

| Source | Discretization and geometry |
| --- | --- |
| `cartesian_vem` | The existing first-order VEM on Cartesian hexahedra, including its stabilization and projected load. |
| `irregular_tet_fem` | Conforming P1 FEM on the tetrahedral fan after deterministic interior-coordinate perturbation; physical boundary points are fixed. |
| `refined_tet_fem` | The irregular P1 FEM mesh with every seventh tetrahedron split into four at its interior barycenter; exterior triangles remain unchanged. |

The last two are **tetrahedral FEM controls, not arbitrary-polyhedral VEM implementations**. Parent-cell labels are retained for reporting and indicator aggregation. Local refinement changes connectivity and star valence without introducing hanging interfaces. It does not validate a general polyhedral VEM discretization.

## Equations retained by the comparison

Let U be the explicit conforming primal reconstruction, e=u-U, and R_U(v)=F(v)-a(U,v). For the energy goal J(u)=a(u,u)/2 and the reported discrete energy J_h,

```math
J(u)-J_h=B+\tfrac12a(e,e),\qquad
B=F(U)-\tfrac12a(U,U)-J_h.
```

This is exact for linear Poisson; there is no omitted Taylor remainder. The VEM conversion and load/stabilization terms remain in B. For the conforming P1 controls, B is zero up to algebraic and quadrature error.

The coupled correction solves

```math
a(\delta_+,v)=R_U(v)\quad\forall v\in V_{P2,0}.
```

All fan vertices and shared edge midpoints participate. Existing interior vertices are not frozen. U+delta_plus is the enriched Galerkin field. The estimator still uses the same DWR substitution and PU localization:

```math
\eta=B+\tfrac12R_U(\delta_+)
     =B+\tfrac12\sum_iR_U(N_i\delta_+).
```

Here the PU localizes one coupled field. Both terms in grad(N_i*delta_plus) are evaluated. This differs from constructing a field by blending independently truncated patch corrections. The direct P2 solve remains a finite-dimensional approximation, not exact truth.

For an inexact correction x with residual r=b-Kx, two quantities are exported separately:

```math
\eta_{DWR}=B+\tfrac12b^Tx,\qquad
\eta_{var}=B+b^Tx-\tfrac12x^TKx,\qquad
\eta_{var}-\eta_{DWR}=\tfrac12x^Tr.
```

The variational quantity increases under the implemented coordinate/coarse energy-minimization steps. It is a diagnostic, not the DWR quantity used for marking, and this monotonicity is not a generic linear-goal guarantee.

The independent linear goal is L(u)=integral(u), with dual -Delta(z)=1 and homogeneous Dirichlet data. The runner solves that dual rather than reusing the primal solution. If Z is its coarse reconstruction,

```math
L(u)-L(U)=R_U(Z)+R_U(z-Z).
```

R_U(Z) is retained, including for VEM, where fan-field Galerkin orthogonality cannot be assumed. Linear-goal patch solves use the dual residual as forcing and the primal residual as readout, without the energy goal's half factor. Exact integral goals are 1/216 for the smooth cube and 1/96 for the smooth L-beam. No linear-goal result is reported for the singular case.

The coarse interpolation P is exact P1-to-P2 interpolation: midpoint values average their endpoints. Its operator is K_c=P^T K P, the conforming fan P1 stiffness, not the VEM stiffness. Each cycle applies

```math
x\leftarrow x+P K_c^{-1}P^T(b-Kx)
```

between forward and backward coordinate passes. The current prototype factors K_c directly and reuses it for additional right-hand sides on the same mesh.

## Saved studies and measured effectivities

The following snapshots were complete when this note was prepared. Every listed row has finite effectivity, and each study has unique source/problem/n/goal/method keys.

| Results directory | Coverage | Rows |
| --- | --- | ---: |
| [investigation](../results/investigation/accuracy.csv) | n=2,4; all three manufactured problems; Cartesian VEM and irregular FEM; buffers and SGS, before coarse-cycle rows were added | 186 |
| [investigation_coupling](../results/investigation_coupling/accuracy.csv) | n=4,6; smooth cube and singular L; Cartesian VEM and irregular FEM; includes coarse cycles | 148 |
| [investigation_quadrature](../results/investigation_quadrature/accuracy.csv) | n=4 singular L, irregular FEM; load quadrature 8 and truth quadrature 12 | 9 |
| [investigation_refined](../results/investigation_refined/accuracy.csv) | n=4,6; smooth cube and singular L; locally refined tetrahedral topology | 72 |
| [investigation_fine_cube](../results/investigation_fine_cube/accuracy.csv) | n=10 smooth cube, Cartesian VEM; both goals | 17 |

Each directory also contains the settings used. Effectivity below means the **signed** estimate divided by the exact signed goal error; it is not the sum of absolute marking indicators.

At n=6, rounded to four decimal places:

| Primal control | Problem / goal | Full PU | Reduced PU | Buffer 2 | Four coarse cycles | Direct P2 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Cartesian VEM | Cube / energy | 0.8251 | 0.7689 | 0.8652 | 0.9976 | 0.9988 |
| Cartesian VEM | Cube / integral | 0.7927 | 0.7380 | 0.8471 | 0.9968 | 0.9984 |
| Cartesian VEM | Singular L / energy | 0.8176 | 0.7506 | 0.8767 | 0.9894 | 0.9920 |
| Irregular FEM | Cube / energy | 0.7431 | 0.7053 | 0.9156 | 0.9946 | 0.9952 |
| Irregular FEM | Cube / integral | 0.7297 | 0.6853 | 0.9000 | 0.9933 | 0.9941 |
| Irregular FEM | Singular L / energy | 0.7243 | 0.6882 | 0.8877 | 0.9724 | 0.9735 |
| Refined FEM | Cube / energy | 0.6855 | 0.6415 | 0.8658 | 0.9814 | 0.9952 |
| Refined FEM | Cube / integral | 0.6707 | 0.6208 | 0.8503 | 0.9766 | 0.9942 |
| Refined FEM | Singular L / energy | 0.6628 | 0.6188 | 0.8438 | 0.9527 | 0.9735 |

The refined-topology rows are an important limitation: four cycles track the direct solution less closely there. A universal four-cycle prescription is not justified. The n=6 refined cube has 7,404 tetrahedra and 740 added barycenters; the refined L has 22,215 tetrahedra and 2,221 added barycenters.

The initial n=4 smooth-L study gives the same qualitative comparison: Cartesian-VEM direct effectivities are 0.9966 for energy and 0.9963 for the integral, versus full-PU values 0.6929 and 0.6830. The irregular-FEM direct values are 0.9876 and 0.9868. These are additional measured cases, not a general convergence theorem.

At n=10 on the Cartesian-VEM cube:

| Goal | Full PU | Buffer 2 | Eight SGS sweeps | Two coarse cycles | Four coarse cycles | Direct P2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Energy | 0.8986 | 0.9313 | 0.9581 | 0.9868 | 0.9984 | 0.9996 |
| Integral | 0.8715 | 0.9158 | 0.9527 | 0.9834 | 0.9978 | 0.9995 |

sDWR's Cartesian-VEM energy effectivity is 1.0149 on the n=6 cube and 1.1015 at n=10; it is 0.9191 for the n=6 singular L. Its earlier proximity to one does not establish asymptotic exactness. No sDWR linear-goal comparison was implemented in this investigation.

For the singular irregular-FEM n=4 control, increasing load/truth quadrature from 6/10 to 8/12 changes every reported effectivity by less than 7.1e-9. The independent exact-energy integration discrepancy decreases from 2.05e-7 to 1.31e-7. This tested quadrature change does not explain the observed PU underestimation; it is not a universal quadrature-error bound.

## Effectivity, marking, and cost are separate questions

Better global effectivity did not automatically improve which cells were marked. On the n=6 Cartesian cube, full PU and direct P2 both capture 20.82% of the reconstruction energy error in the top 20% of parent cells; the reference best selection captures 26.23%. On the irregular cube, the normalized distribution L1 error is 0.0583 for full PU and 0.0745 for direct P2, despite the latter's much better signed effectivity.

The spatial reference here is the positive reconstruction energy a_K(u-U,u-U)/2. It is not a unique signed goal-error decomposition. Linear-goal spatial-truth columns are deliberately NaN: the primal energy-error distribution is not presented as the exact localization of an arbitrary linear goal.

There is also a substantial global cost to measure. At n=10 the Cartesian-VEM cube has 729 primal free DOFs, whereas this prototype has 4,429 coarse and 31,659 enriched free DOFs, with 783,359 entries in the free P2 operator. Coarse assembly/factorization, P2 assembly, interpolation, residual cycles, dual work, and localization all belong in a runtime comparison. The mesh-dependent setup must be counted again when frequent remeshing invalidates it. These accuracy results establish no speed advantage over current PU-DWR or sDWR.

## Nonunique local fields: the relevant distinction

A unique reconstructed field is not necessary for DWR. Without a rule selecting a representative, the output c^T x of a singular system Ax=b is well defined precisely when c annihilates ker(A), provided b satisfies the left-nullspace compatibility condition. One may then work on a quotient space. This gives output uniqueness, not estimator accuracy.

A different legitimate possibility is to specify a stable representative and validate its DWR accuracy even when the output is not invariant across all solutions. A minimum-norm or other selection rule is therefore a candidate construction, not automatically an invalid idea; its stability and accuracy still need evidence.

The [weighted-patch audit](../results/investigation/weighted_patch_audit.csv) tests the proposed operator a(phi_j,Ni*phi_k) on an interior face-center star with seven fixed existing P1 vertices and eighteen free P2 edge midpoint values, including artificial-boundary edge values. It uses no point fitting.

| Geometry | Rank / 18 | Right nullity | Actual RHS left-null defect / RHS norm | Readout right-null component / RHS norm |
| --- | ---: | ---: | ---: | ---: |
| Regular star | 15 | 3 | 9.1e-17 | 5.0e-16 |
| Perturbed star, amplitude 0.08 | 16 | 2 | 3.56e-4 | 5.38e-2 |

These are algebraic diagnostics, not global effectivity errors. The audit also constructs a compatible body-load moment vector for which two admissible solutions of the same local equations give different DWR readouts. Thus the local weighted equations alone do not provide a general compatibility/output-invariance guarantee on irregular stars. This does not prove that every explicitly selected representative would be inaccurate.

Compatible Neumann patches and equilibrated flux reconstructions remain alternatives. They require boundary-flux information, compatibility, VEM consistency treatment, and a separate dual for generic goals. An energy-norm flux bound must not silently replace signed DWR. The primary literature, limitations, and further experiments are collected in [options_research.md](options_research.md).

## Validation completed

- `test_coupled_enrichment.jl`: 58 checks passed. Shared P2 assembly, free-system residual, independent full-quadratic boundary lifting, direct residual/energy identities, monotone variational SGS correction, PU summation, indicator conservation, and zero-buffer equality with existing node and cell indicators.
- `test_irregular_tetrahedra.jl`: 1,559 checks passed. Valid perturbed geometry, domain volume, affine FEM reproduction, barycentric-gradient identities, energy consistency, and existing patch-estimator integration.
- `test_local_tetra_refinement.jl`: 1,136 checks passed. Preserved exterior triangles/edges and original coordinates, oriented children, per-cell volume/centroid conservation, face connectivity, changed incidence, affine reproduction, free-hat residual compatibility, and both patch kernels.
- `two_level_correction.jl`: standalone assertions passed. Galerkin coarse operator against independently assembled P1 stiffness, monotone variational correction, and convergence to a polynomial containing all quadratic terms with exact physical trace.
- `audit_weighted_patch.jl`: standalone assertions passed for regular and perturbed stars, including analytical harmonic null modes and a compatible load realization.
- The study runner checks energy localization and independent load-side identities for the linear goal. The five saved CSVs contain 186/148/9/72/17 complete rows with no nonfinite effectivities or duplicate keys.
- The original laboratory's 3,825 checks passed before these new isolated files were added; that historical pass is not represented as a rerun of the expanded investigation.

Unvalidated extensions include arbitrary polyhedral VEM, coefficient jumps/high contrast, mixed physical boundaries, 3D elasticity, and adaptive optimization. No accuracy claim here removes those validation requirements.

## Running

Run from the project root with Julia 1.10 or later. The commands below require no external packages.

```text
julia --project=benchmarks/poisson3d_accuracy --startup-file=no benchmarks/poisson3d_accuracy/investigation/test_coupled_enrichment.jl
julia --project=benchmarks/poisson3d_accuracy --startup-file=no benchmarks/poisson3d_accuracy/investigation/test_irregular_tetrahedra.jl
julia --project=benchmarks/poisson3d_accuracy --startup-file=no benchmarks/poisson3d_accuracy/investigation/test_local_tetra_refinement.jl
julia --project=benchmarks/poisson3d_accuracy --startup-file=no benchmarks/poisson3d_accuracy/investigation/two_level_correction.jl
julia --project=benchmarks/poisson3d_accuracy --startup-file=no benchmarks/poisson3d_accuracy/investigation/audit_weighted_patch.jl
```

For a current CLI study:

```text
julia --project=benchmarks/poisson3d_accuracy --startup-file=no benchmarks/poisson3d_accuracy/investigation/run_investigation.jl --levels=4,6 --cases=cube_smooth,lshape_singular --sources=cartesian_vem,irregular_tet_fem --buffers=1,2 --sweeps=1,2,4,8 --cycles=1,2,4 --output=benchmarks/poisson3d_accuracy/results/reproduce_coupling
```

To reproduce all five study configurations, start `julia --project=benchmarks/poisson3d_accuracy --startup-file=no` and execute the following Julia statements. Separate output directories preserve the recorded snapshots; empty cycle tuples reproduce the two studies made before coarse-cycle rows were introduced.

```julia
include("benchmarks/poisson3d_accuracy/investigation/run_investigation.jl")
out = "benchmarks/poisson3d_accuracy/results"
run_investigation(levels=(2,4), buffers=(1,2), cycles=(),
    output=joinpath(out,"reproduce_initial"))
run_investigation(levels=(4,6), cases=(:cube_smooth,:lshape_singular),
    buffers=(1,2), output=joinpath(out,"reproduce_coupling"))
run_investigation(levels=(4,), cases=(:lshape_singular,),
    sources=(:irregular_tet_fem,), buffers=(1,2), cycles=(),
    quadrature_order=8, truth_order=12, output=joinpath(out,"reproduce_quadrature"))
run_investigation(levels=(4,6), cases=(:cube_smooth,:lshape_singular),
    sources=(:refined_tet_fem,), buffers=(1,2),
    output=joinpath(out,"reproduce_refined"))
run_investigation(levels=(10,), cases=(:cube_smooth,), sources=(:cartesian_vem,),
    buffers=(1,2), sweeps=(8,), cycles=(2,4), output=joinpath(out,"reproduce_fine_cube"))
```

`accuracy.csv` contains signed estimates, effectivities, separate variational diagnostics, geometry/DOF information, algebraic residuals, and available spatial diagnostics. `settings.toml` records each configuration. The weighted audit additionally exports its nullspace coefficients. No runtime ranking is inferred from these files.
