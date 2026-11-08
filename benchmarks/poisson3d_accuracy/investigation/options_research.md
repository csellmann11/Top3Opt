# Reconstruction options for DWR on irregular meshes

Research assessment, 7 November 2026. This note concerns the isolated Poisson laboratory. It does not change the production topology-optimization code or propose a runtime winner. The existing laboratory README was read before preparing this assessment.

## Recommendation and scope

Use a globally coupled conforming P2 correction as the accuracy reference, then investigate how much of that coupling is needed. In parallel, test one- and two-layer buffered Dirichlet patches with the original inner PU readout. Treat equilibrated Neumann/flux reconstruction as a separate, more substantial research route. Keep an output-invariant quotient construction experimental until its compatibility, output invariance, and stability pass on the actual constrained spaces.

None of these routes requires an unchecked polynomial fit to neighboring samples. Their assumptions concern finite-element spaces, geometry, boundary conditions, and operator stability instead.

| Route | What it changes | Established foundation | Unresolved question here |
| --- | --- | --- | --- |
| Coupled P2 enrichment, possibly solved inexactly | Communicates corrections through the mesh | Conforming Galerkin enrichment; goal-oriented algebraic error control | How few iterations retain accurate signed DWR and marking? |
| Buffered Dirichlet patches | Moves artificial boundaries away from the original PU support | Standard well-posed local Galerkin solves | Does buffering materially improve DWR per unit work on irregular meshes? |
| Compatible Neumann/equilibrated flux patches | Supplies equilibrium and boundary flux information instead of zero potential correction | Patch equilibration and mixed methods, subject to compatibility | VEM consistency, dual reconstruction, and 3D elasticity cost |
| Output-invariant quotient solve | Permits nonunique local fields if the requested output is unique | Exact finite-dimensional algebra; standard Neumann gauges | Whether this specific weighted-patch operator gives a stable, useful DWR approximation |

## Preserve the goal-error identity

The following identities are direct derivations for the laboratory, not additional estimator assumptions. For its conforming reconstruction U, let e=u-U and R_U(v)=F(v)-a(U,v). Its quadratic energy goal satisfies exactly

```math
J(u)-J_h=B+\tfrac12a(e,e),\qquad
B=F(U)-\tfrac12a(U,U)-J_h.
```

There is no Taylor remainder for linear Poisson with this quadratic goal. A reconstruction e_plus gives the current signed estimator B+R_U(e_plus)/2. VEM stabilization and load consistency remain in B and its existing localization; they must not disappear when the reconstruction changes.

For a general linear goal L, the dual solves a(v,z)=L(v). With a discrete reported goal L_h, the exact identity is

```math
L(u)-L_h=L(U)-L_h+R_U(z).
```

For any coarse dual Z, this is L(U)-L_h+R_U(Z)+R_U(z-Z). The term R_U(Z) cannot be dropped merely because the original VEM equations were solved: the explicit fan field U is not necessarily Galerkin-orthogonal in the continuous fan bilinear form. A generic goal therefore needs its own dual reconstruction and goal-consistency accounting. The energy-goal shortcut z=u does not carry over.

## 1. Coupled hierarchical P2 enrichment

**Concrete experiment.** On the same conforming tetrahedral fan, solve

```math
a(e_+,v)=R_U(v)\quad\forall v\in V_{+,0},
```

where V_plus is the full continuous piecewise P2 space with physical Dirichlet conditions. U+e_plus is then the corresponding enriched Galerkin solution. No artificial patch boundary is imposed. This is a finite-element solve, not sample fitting. For a generic linear goal, also solve the enriched dual with right-hand side L.

Hierarchical DWR theory supplies two-sided estimates under a goal saturation assumption, not an unconditional assertion that a chosen enrichment has effectivity one. The assumption can fail before the asymptotic regime, and cancellation can make a general goal error zero. See Endtmayer, Langer and Wick, Sections 2–4 of [Two-side a posteriori error estimates for the DWR method](https://www.ricam.oeaw.ac.at/files/reports/18/rep18-32.pdf).

**Implementation inference.** A hierarchical vertex/edge basis exposes the coupling:

```math
\begin{bmatrix}K_{cc}&K_{ce}\\K_{ec}&K_{ee}\end{bmatrix}
\begin{bmatrix}d_c\\d_e\end{bmatrix}
=\begin{bmatrix}b_c\\b_e\end{bmatrix}.
```

Freezing every existing fan vertex and solving only for new edge values gives K_ee d_e=b_e, not the full enriched problem. Eliminating the coarse block instead gives a Schur complement involving K_cc inverse. This identifies a useful middle ground: a few globally coupled, preconditioned correction steps, rather than assuming independent patches reproduce that coupling. An appropriate coarse operator or preconditioner must be checked; the original VEM stiffness cannot automatically substitute for the fan P1 block.

**Inexact solves need output control.** Dolejsi and Tichy explicitly study primal and dual algebraic errors and goal-aware stopping, including loss of Galerkin orthogonality under incomplete solves. Their algorithm is a useful foundation, not a measured speed result for this code. See [On efficient numerical solution of linear algebraic systems arising in goal-oriented error estimates](https://arxiv.org/html/2001.01929), Sections 2.4–3.4 and 5.

For an enriched linear system Kx=b and output c^T x, the following algebra provides a transparent laboratory diagnostic. If r=b-Kx_tilde and K^T y=c, then

```math
c^T(x-x_{\rm tilde})=y^Tr.
```

With an approximate dual y_tilde and s=c-K^T y_tilde, the remaining correction is s^T K^{-1}r. A small primal Euclidean residual alone is therefore insufficient evidence that the DWR output is converged. Compare exact enriched solves with 1, 2, 4, and 8 correction steps, tracking both primal and dual output errors. A small fixed iteration count is a hypothesis to test, not a guarantee.

## 2. Compatible Neumann patches and equilibrated fluxes

For scalar Poisson, a Neumann patch solve has the form

```math
a_{\omega_i}(Q_i,v)=F_{\omega_i}(v)+\int_{\partial\omega_i}g_i v,
\qquad \int_{\omega_i}Q_i=m_i.
```

For a connected pure Neumann patch, compatibility requires integral(f)+integral(g_i)=0. The mean removes the constant gauge. Physical Dirichlet portions change the space and can remove that gauge. A quadratic q is reproduced when its forcing, normal derivative, and mean are supplied exactly in adequate spaces; merely choosing an equilibrated flux close to the coarse gradient does not imply those data are polynomial-exact.

Ern and Vohralik prove stable polynomial potential/flux extensions on shape-regular three-dimensional tetrahedral vertex patches with compatible data, including specified boundary cases. Their Section 4 makes the connection between patch compatibility and hat-function orthogonality explicit. These are not results for arbitrary unvalidated polyhedron fans. See [Stable broken H1 and H(div) polynomial extensions](https://arxiv.org/html/1701.02161v4), Theorems 2.2–2.5 and equation (4.5).

**A VEM obstacle that must be addressed.** For sigma=-grad(u), the familiar localized divergence datum is approximately Ni*f-grad(Ni)·grad(U). Its patch integral is R_U(Ni). This need not vanish for the laboratory's reconstructed U, especially at added face and cell centers. Thus setting zero artificial-boundary normal flux on every patch can create incompatible problems. A compatible discrete gradient, a justified residual correction, or a coupled construction is required. Subtracting a patch mean without retaining the removed residual changes the estimator.

There is direct evidence against assuming automatic improvement from equilibration: Dassi, Gedicke and Mascotto prove reliability of a localized VEM flux estimator but document loss of efficiency related to mismatched projection operators, including an adaptive low-order example with hanging nodes. Their analysis is two-dimensional and includes stabilization effects. This warns against a blind 3D transplant; it does not rule out all flux methods. See [Adaptive virtual element methods with equilibrated fluxes](https://arxiv.org/html/2004.11220), Sections 5.2 and 6.3–6.4.

**Keep the connection to generic DWR.** An energy-norm flux bound is not a signed DWR estimator for an arbitrary output. For homogeneous Dirichlet data and div(sigma_h)=f_h, direct integration by parts gives

```math
R_U(z)=(f-f_h,z)-(\sigma_h+\nabla U,\nabla z).
```

The dual z is still needed. Substituting an enriched dual gives a computable signed quantity; controlling the remainder needs dual accuracy and, potentially, a second flux reconstruction. Data oscillation and physical Neumann inconsistencies must be retained. Mozolevski and Prudhomme combine primal/dual flux reconstructions with DWR and analyze asymptotic exactness under approximation-order assumptions; this is a relevant generic-goal route, not proof for our VEM scheme. See [Goal-oriented error estimation based on equilibrated-flux reconstruction](https://www.sciencedirect.com/science/article/pii/S004578251400348X).

**Decisive test.** First use a globally coupled mixed-flux reference to verify divergence, boundary flux, load oscillation, and the complete goal identity. Then localize while tracking compatibility defects. Compare signed DWR and an energy bound as separate outputs. Do not label the bound a replacement DWR effectivity.

## 3. Output-invariant patch quotient solves

The user is correct that field uniqueness is not necessary by itself. The relevant requirement is uniqueness and stability of the requested residual output.

For a constrained patch system Ax=b and output c^T x, direct linear algebra gives:

- Solvability: b is orthogonal to ker(A^T).
- Output invariance: c is orthogonal to ker(A).
- If both hold, solve A^T y=c on a quotient space and evaluate c^T x=y^T b, independently of the primal gauge.

For the proposed weighted-test correction equation a(delta,Ni*v)=R_U(Ni*v), its energy-type output has c=b/2 when trial and unweighted test coefficients use the same free basis. This is distinct from the existing full-PU solve, whose forcing is R_U(v) and whose readout is R_U(Ni*delta)/2. A nonsymmetric weighted-test operator does not make the two orthogonality conditions equivalent. For a generic goal, c also changes with the dual weighting.

Standard compatible Neumann problems on mean-zero spaces are a principled quotient construction; the spaces in equation (4.6) of the [Ern–Vohralik paper](https://arxiv.org/html/1701.02161v4) provide an example. However, in the inspected primary literature I did not find a reliability/convergence result for the exact proposed operator a(phi_j,Ni*phi_k), with the particular fixed-vertex/free-edge constraints and PU readout used here. That extension remains a research proposal.

**Concrete gate before accuracy comparisons.** Build the actual constrained operator on irregular interior, face, cell, and physical-boundary stars. Check left and right nullspaces separately, b compatibility, c invariance, and response to controlled quadrature perturbations. Fixing all fan-P1 vertices is different from fixing only original VEM vertices; null modes from the latter space cannot be used as counterexamples for the former.

An SVD minimum-norm solution is not a cure when output invariance fails: it silently chooses one of several output values. Even when exact null modes are harmless, near-null modes can amplify errors. The norm of a minimum-norm output dual solving A^T y=c is a useful sensitivity diagnostic after consistent scaling. Passing these algebraic tests still does not establish accurate approximation of the true goal error.

## 4. Buffered Dirichlet patches: a lower-change experiment

**Proposal, not an imported theorem.** Enlarge each current patch by one or two face-adjacent element layers. Solve the same coercive P2 residual problem on the enlarged patch, with zero correction only at its outer artificial boundary. Read out R_U(Ni*delta_i)/2 on the original support of Ni. Keep the original PU, conversion terms, physical boundary conditions, and cell distribution unchanged.

This moves the artificial boundary away from the region used by the estimator and avoids Neumann compatibility or point-fitting issues. It does not guarantee monotone improvement of the signed DWR estimate. Overlap and dense patch-solve work can grow sharply in 3D.

There are exponential localization results for specialized multiscale correctors. Målqvist and Peterseim construct correctors in the kernel of a suitable coarse interpolation operator and a coupled modified coarse space. Their theorem does not apply directly to independent unrestricted residual patches. See [Localization of elliptic multiscale problems](https://arxiv.org/html/1110.0692), Sections 2.2–2.4 and Theorem 3.6. Here buffering is an empirical isolation test for boundary truncation, not a claimed exponential-decay method.

## Geometry, boundaries, materials, and changing meshes

The following are implementation requirements and limitations inferred for this project:

- **General polygon/polyhedron fans:** verify that face triangulations are shared and conforming, tetrahedra have positive volume, and the center-based fan actually covers the cell without overlaps. A non-star-shaped cell or face need not admit the assumed fan. Polyhedral admissibility alone does not guarantee good tetrahedral aspect ratios; slivers can destroy conditioning and the usefulness of shape-regular estimates.
- **Hanging interfaces:** all methods need a common interface subdivision and consistent global IDs for shared P2 edge degrees of freedom. Merely sharing the face center is insufficient.
- **Physical boundaries:** distinguish artificial and physical boundaries. Enforce actual Dirichlet traces and Neumann loads in primal and dual problems; do not count clipping of candidate modes as preservation of their original polynomial form. Pure Neumann gauges and compatibility must be stated explicitly.
- **Coefficient jumps:** use the actual coefficient in the residual, energy metric, and flux continuity condition. Resolve interfaces or integrate them consistently. Constant-coefficient quadratic reproduction is not a general transmission-property guarantee. Test contrast dependence instead of assuming that a constant-diffusion theorem is robust for near-void topology-optimization coefficients.
- **Frequent remeshing:** count connectivity construction, patch growth, factorization/preconditioner setup, quadrature, and output assembly in eventual runtime comparisons. Mesh-dependent global factorizations and per-patch rank decompositions cannot be treated as reusable indefinitely. Reuse within one mesh is legitimate; amortization across changing meshes must be measured.
- **Three-dimensional elasticity:** a pure Neumann displacement patch has six rigid-body modes, requiring force and moment compatibility. Fluxes become stresses, and symmetry or a controlled weak-symmetry condition matters. Bertrand et al. give a rigorous weakly symmetric stress-reconstruction method in 2D/3D for a specified stable displacement-pressure discretization; it is not a proof for our low-order VEM elasticity. See [Weakly symmetric stress equilibration](https://arxiv.org/html/1808.02655), Sections 3–4. Coupled vector P2 enrichment also requires appropriate displacement constraints and material robustness checks.

## Decisive experiment sequence

1. **Establish coupled references.** Keep the same primal U, load quadrature, and goal corrections. Compare full P2 enrichment against another enrichment/refinement to estimate reference error; the first P2 result is a reference method, not exact truth.
2. **Break mesh symmetry deliberately.** Use valid irregular meshes with reproducible perturbations, varying face counts and aspect ratios, boundary and reentrant-edge patches, and conforming representations of hanging interfaces. Report actual fan quality alongside results. A mean effectivity over favorable stars is insufficient.
3. **Separate goal classes.** Retain energy, add a manufactured dual distinct from the primal by defining L(v)=a(v,z_exact), and add a localized volume goal with an independently resolved dual. Record signed error and cancellation; suppress ratios near zero denominators.
4. **Compare localizations.** Test current patches and buffered patches against the coupled solution. Measure the reconstructed-field alignment, PU product-gradient contributions, signed effectivity, and actual marking performance. No term of grad(Ni*delta_i) may be omitted to improve a number.
5. **Audit quotient proposals first.** Apply the algebraic gate above before interpreting their effectivity. Deliberately perturb compatible data and geometry to expose near-null output sensitivity.
6. **Then reduce coupled solve work.** Vary iteration count and monitor both primal and dual algebraic contributions to the goal estimate. Choose tolerances from observed output convergence, rather than a primal residual norm alone.
7. **Only then compare full costs.** Include all setup per mesh and all dual/reconstruction work. No source inspected establishes a universal ten-times-sDWR target, a fixed number of sufficient Krylov steps, or superiority of any route on this code base.

No production changes or estimator implementations were made for this research note.
