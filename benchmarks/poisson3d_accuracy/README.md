# Isolated 3D Poisson accuracy experiment

This directory is a standalone accuracy laboratory. It does not load or modify
the topology-optimization driver, its estimators, its package environment, or its
output paths. All Julia code uses plain includes and small functions; dependencies
are Julia standard libraries. Its explicit local matrices and SVDs favor clarity.
It is **not a performance implementation or a runtime comparison**.

The separate [reconstruction investigation](investigation/README.md) compares
buffered patches and coupled P2 corrections, including irregular tetrahedral
controls, an independent linear goal, and an audit of nonunique weighted patches.

The experiment solves scalar Poisson with unit diffusion and compares:

1. **Full PU-DWR:** local conforming P2 Dirichlet-patch corrections on a
   vertex/face-center/cell-center tetrahedral fan. This is the scalar analogue of
   the current 3D elasticity PU formulation, not a call to the production code.
2. **Reduced PU-DWR:** the same patch equations restricted to the hat + linear
   directional + curvature space. The scalar space has at most ten modes; this
   corresponds to the thirty-coefficient vector space discussed for elasticity.
3. **sDWR:** replace the enriched primal and adjoint fields by the elementwise
   linear VEM projection, retaining the same goal conversion, cell consistency
   correction and PU localization. This requires no additional global or local
   PDE solve. No effectivity calibration is applied.

## Running

From the project root, using Julia 1.10 or later:

```text
julia --project=benchmarks/poisson3d_accuracy --startup-file=no benchmarks/poisson3d_accuracy/runtests.jl
julia --project=benchmarks/poisson3d_accuracy --startup-file=no benchmarks/poisson3d_accuracy/run.jl
```

No package installation is needed. The default run uses n=4,6,8,10 cells per unit
length for all three manufactured problems. For a smaller or different run:

```text
julia --project=benchmarks/poisson3d_accuracy --startup-file=no benchmarks/poisson3d_accuracy/run.jl --levels=2,4 --cases=cube_smooth,lshape_singular --quadrature=6 --truth-order=10 --output=benchmarks/poisson3d_accuracy/results/custom
```

Every case compares the estimators on the **same primal solution and mesh**.
Mesh levels are uniform: n changes spacing, not the selected estimator's marking.
There is no adaptive optimization or density update in this laboratory.

## Domains and manufactured solutions

All physical boundaries have homogeneous Dirichlet conditions. Set f=-Delta(u).
The goal is

```math
J(u)=\frac12\int_\Omega |\nabla u|^2\,dx.
```

The domains are the unit cube and the extruded reentrant L:

```math
\Omega_{L}=\left((-1,1)^2\setminus\{x>0,y<0\}\right)\times(0,1).
```

The L-beam has volume 3 and a reentrant edge at x=y=0. The three cases are:

| Name | Exact solution | Reference energy |
| --- | --- | --- |
| `cube_smooth` | x(1-x)y(1-y)z(1-z) | 1/1800, analytic |
| `lshape_smooth` | xy(1-x^2)(1-y^2)z(1-z) | 0.008997732426303856, analytic |
| `lshape_singular` | r^(2/3) sin(2 theta/3) (1-x^2)(1-y^2) z(1-z) | approximately 0.05699269777506444 |

For the singular case, theta is in [0,3pi/2]. Its gradient behaves as r^(-1/3)
near the reentrant edge. Exact values and body force are finite there; the
interior-edge gradient is deliberately undefined. Volume quadrature never samples
that edge. The reference energy is integrated independently in polar coordinates,
with r=R(theta)t^3, angular segments at multiples of pi/4, and analytic integration
in z. Orders 32 and 48 estimate its integration uncertainty; this is an observed
convergence estimate, not a certified bound.

`--quadrature` controls the tensor Gauss order of a Duffy tetrahedral rule, not
its polynomial degree. Order 6 integrates the smooth manufactured loads used by
both estimators exactly up to roundoff. `--truth-order` controls independent
integration of the exact gradient for the cell diagnostics. The singular case
also needs quadrature-convergence checks, provided by the tests.

## Primal discretization and goal-error convention

To retain a meaningful stabilization-based comparison, the primal method is a
simple scalar first-order VEM on uniform Cartesian hexahedra, not tetrahedral P1
FEM with a zero stabilization defect. Each cell has its eight vertex values.
For side h and volume V, its affine projection is

```math
\Pi_K v=\operatorname{mean}(v)+(x-x_K)^T G_Kv,
\qquad
(G_K)_{a,j}=\frac{\operatorname{sign}(x_{j,a}-x_{K,a})}{4h}.
```

With P_K the projection evaluated at the vertices, the cell operator is

```math
K_K=V G_K^T G_K+h(I-P_K)^T(I-P_K).
```

The body load is projected consistently: F_h,K(v)=integral_K f Pi_K(v).
The solve retains the original unconstrained operator for energy calculation.
It reports

```math
J_h=\tfrac12 u_h^T K_hu_h,\qquad \Delta J=J(u)-J_h.
```

J_h includes VEM stabilization. It is **not** the energy of the explicit
piecewise-linear fan field U. Face-center values of U are shared face averages;
cell-center values are cell averages, preserving the cell projection mean.

## Full and reduced patch equations

Each hexahedron is split into 24 tetrahedra: four triangles per face, connected
to the cell center. Each PU node i has an incident tetrahedral star. The full
patch solves

```math
a(\delta_i,v)=R_U(v)=F(v)-a(U,v).
```

Corrections vanish on artificial patch boundaries and on physical Dirichlet
boundaries. The signed node contribution is

```math
\eta_i=\tfrac12R_U(N_i\delta_i)
 +\tfrac12\int N_i\left(|\nabla U|^2-|\nabla\Pi u_h|^2\right)
 -\tfrac12S_i.
```

S_i assigns each nodal squared stabilization defect to its original vertex.
The separate cell correction is

```math
C_K=F_K(U)-a_K(U,U)-\left[F_{h,K}(u_h)-a_{h,K}(u_h,u_h)\right].
```

The global signed estimate is sum_i eta_i + sum_K C_K. Absolute values are taken
only after assembling each whole node contribution. Each |eta_i| is distributed
to its adjacent cells by normalized volume; |C_K| is added to cell K. All
densities equal one, so volume*density^q weighting reduces exactly to volume.

For the reduced space, xi=(x-x_i)/h_i and d_j=(x_j-x_i)/h_i. Its ten scalar
candidate functions are N_i, N_i xi_1:3, and six radial P2 aggregates with edge
features [dx^2,dy^2,dz^2,2dxdy,2dxdz,2dydz]. In the full P2 nodal representation,
the central row is [1,0,...,0], and radial midpoint rows are

```text
[1/2, dx/4, dy/4, dz/4, dx^2, dy^2, dz^2, 2dxdy, 2dxdz, 2dydz].
```

Physical Dirichlet P2 rows are removed first. At physical boundaries this is
nodal clipping of the candidate functions: they remain conforming and admissible
but need not equal the original unmodified N_i times polynomial. This experiment
uses that explicit boundary construction, **without silently falling back to full
patches**. SVD removes linearly dependent columns using relative tolerance 1e-11.
The columns of Q span the remaining space; reduced equations are

```math
K_r=Q^TK_iQ,\quad b_r=Q^Tb_i,\quad
\delta_i^r=QK_r^{-1}b_r.
```

Both methods share K_i, b_i, the PU-weighted residual vector, and all consistency
corrections. The reduced indicator still evaluates R_U(N_i delta_i^r)/2; it does
not substitute the correction energy. There is no displacement polynomial fit.
The shared full assembly and SVD are intentional for accuracy verification.

## sDWR by projected enrichment

Let p=Pi u_h. The sDWR estimator makes the literal substitution W=p in the
same residual expression used by the PU estimators: its error enrichment is
p-U. Each signed nodal contribution is

```math
\eta_i^{\mathrm{sDWR}}=\tfrac12R_U\!\left(N_i(p-U)\right)
 +\tfrac12\int N_i\left(|\nabla U|^2-|\nabla p|^2\right)
 -\tfrac12S_i.
```

The separate C_K above is unchanged. Although U is continuous, p is generally
discontinuous across element faces. This is a formal elementwise substitution
into the volume residual, not a conforming higher-order reconstruction. The
implementation evaluates p-U on each tetrahedron belonging to its parent cell;
it does not solve a patch problem for sDWR.

For an equilibrated discrete solution, the global sum simplifies to

```math
\eta_{\mathrm{sDWR}}=
 \tfrac12\left[F(p)+F(U)-a(U,p)-a_h(u_h,u_h)\right].
```

Here F_h(u_h)=F(p), a_h(u_h,u_h)=F_h(u_h), and the Cartesian fan preserves the
cell mean gradient, so a(U,p)=a(p,p). Writing
S=a_h(u_h,u_h)-a(p,p), this gives the independent check

```math
\eta_{\mathrm{sDWR}}=
 \tfrac12\left[S+F(U)-F_h(u_h)\right].
```

The factor one-half follows from the energy goal. All physical boundaries are
Dirichlet, so there is no traction term, but the manufactured body loads are
nonzero: their defect F(U)-F_h(u_h) is retained. The plus sign follows from this
benchmark's projected loading. With exact discrete loading F_h(u_h)=F(U), the
same substitution instead yields the paper's pre-neglect expression
one-half times [S-F(U-p)]. Neglecting that volume-load contribution is a further
approximation, whose stated smoothness assumptions do not cover the singular
reentrant-edge solution.

Marking uses the same localization as the full and reduced estimators:
distribute each assembled |eta_i| by normalized adjacent-cell volume and add
|C_K| separately. It is not the cellwise quantity S_K/2, and the signed sDWR
estimate need not be nonnegative. Earlier benchmark reports using unscaled S
as the sDWR comparator are superseded: they omitted the energy factor and this
projected-enrichment construction. Comparisons require results regenerated
with the formulation above; production estimators are unaffected.

## What the outputs mean

The default output is `results/latest/` (ignored by Git):

- `report.md`: readable comparison tables, with no runtime ranking.
- `summary.csv`: signed/magnitude effectivities, goal errors, primal residual,
  reference-integration diagnostics, and spatial marking diagnostics.
- `comparison.csv`: change in the reduced signed estimate and cell indicators
  relative to the full method.
- `*_cells.csv`: exact/discrete cell energies, their signed difference, positive
  reconstruction error and all three marking indicators.
- `*_nodes.csv`: full/reduced signed PU contributions and actual patch dimensions,
  including the geometric node type and physical-boundary flag.
- `*.vtk`: the same cell fields and the primal nodal solution for ParaView.
- `settings.toml`: cases, levels, numerical integration orders and conventions.

Signed effectivity is eta/DeltaJ. Magnitude effectivity is sum(indicators)/|DeltaJ|.
The former includes cancellation and sign; the latter measures the marking
magnitude and need not approach one. Near-zero denominators produce NaN rather
than a misleading ratio. All three methods report their signed residual estimate
separately from their nonnegative marking indicators.

The positive cell reference for **spatial** comparisons is

```math
E_K^{\mathrm{rec}}=\tfrac12\int_K|\nabla u-\nabla U|^2.
```

Its sum is not DeltaJ for this VEM discretization. The normalized-distribution
L1 distance and error captured by the top 20% of marked cells refer to E_rec.
The CSV also contains the signed local goal differences, whose sum approaches
DeltaJ as exact-field quadrature converges. Thus neither spatial diagnostic is
misrepresented as a unique positive decomposition of the goal error.

## Validation and scope

`runtests.jl` checks quadrature moments and manufactured derivatives, topology and
fan conformity, affine consistency of the primal operator, physical constraints,
P2 stiffness and weighted residual against independent quadrature, reduced
Galerkin orthogonality, unchanged corrections, indicator conservation, and the
goal identity using the exact manufactured reconstruction error. Independent
quadrature checks the projected sDWR substitution and its global energy formula.
The tests also check mesh convergence and load-quadrature refinement on the
singular problem.

No assertion forces either PU method's effectivity toward one or forces the
reduced indicators to agree with full P2: those are the quantities under study.
The first experiment uses constant diffusion, uniform hexahedral meshes and
homogeneous Dirichlet conditions. Hanging interfaces, variable coefficients,
mixed boundaries, elasticity and topology optimization remain separate future
validation stages. The scalar VEM stabilization scaling alpha=h is explicit and
is not tuned to the manufactured answers.

## Diagnosing the existing PU reconstruction

Run `julia --project=benchmarks/poisson3d_accuracy --startup-file=no benchmarks/poisson3d_accuracy/diagnose_pu_reconstruction.jl`
to inspect the smooth cube at n=4,6,10. This retains the existing patch solves
and writes `results/pu_reconstruction_diagnostics.csv`; it does not overwrite
`results/latest` or change either estimator.

For e=u-U and e_hat=sum_i N_i delta_i, the diagnostic measures the energy-norm
ratio r=||e_hat||_a/||e||_a and alignment
c=a(e,e_hat)/(||e||_a ||e_hat||_a). Their product is exactly the effectivity of
the reconstruction correction, Q/E_rec, rather than the total goal estimate.
It also separates grad(e_hat) into sum_i N_i grad(delta_i) and
sum_i delta_i grad(N_i). These are diagnostic contributions, not alternative
estimators: both terms belong in the gradient.

Order-7 quadrature integrates the smooth cube quantities exactly up to floating
point error. Assertions check the complete goal identity and agreement between
the original patch-residual evaluation and independent exact-gradient
integration. No point fitting is used.

## Source map

`geometry.jl` constructs meshes and PU fans; `problems.jl` defines exact solutions;
`quadrature.jl` supplies integration; `primal.jl` solves scalar VEM;
`estimators.jl` contains both patch spaces and sDWR; `metrics.jl` evaluates truth
and exports results; `run.jl` selects cases and levels. There are no modules and
no references from production code into this directory.
