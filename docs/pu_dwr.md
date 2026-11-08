# 3D PU-DWR error estimation

The adaptive driver supports `error_estimator=:sdwr` (the existing default) and
`error_estimator=:pu_dwr`. PU-DWR follows the local quadratic Dirichlet-patch
formulation of the 2D `TTO adaptive DWR` implementation. It estimates error in the
elastic strain energy and supplies nonnegative cell indicators for mesh marking.
The estimator is independent of the density regularization and optional projection.
All implementation files use plain includes and small functions, without modules.

```text
julia --project=. src/main_optim_runs.jl --error_estimator pu_dwr --do_adaptivity true --do_adaptivity_at_the_start false
julia --project=. src/main_optim_runs.jl --error_estimator pu_dwr --pu_marking_exponent 3 --regularization helmholtz --projection true
```

The same names are keywords of `run_optimization`. The marking exponent defaults
to `1.0`, matching the current 2D top-level `pu_dwr_estimator`; it must be finite
and nonnegative. It changes redistribution of nodal indicator magnitudes, not the
SIMP material exponent, which remains three. Both options enter the run hash and
are recorded in `SimulationResults`.

## Geometry and local reconstruction

Every active leaf face is triangulated by connecting its ordered boundary edges
to a shared face center. Connecting each face triangle to its cell center creates
tetrahedra. This also covers coarse/fine interfaces through their leaf subfaces.
The scalar partition of unity consists of affine tetrahedral hats at original
vertices, face centers, and cell centers. Each hat has a local star of incident
tetrahedra. Invalid face fans, degenerate tetrahedra, and inconsistent coverage are
rejected when geometry is built.

The continuous coarse field `U` uses the displacement at original VEM vertices.
Face-center values preserve the mean of the face VEM projection; cell-center
values preserve the mean of the volume VEM projection. Shared faces therefore
have identical traces on their two sides. Affine displacements are reproduced.

For each PU node `i`, a conforming quadratic correction `delta_i` is solved on its
star:

```text
a(delta_i, v) = f(v) - a(U, v) = rho_U(v).
```

The correction vanishes on artificial patch boundaries. Physical natural
boundaries remain free, and prescribed displacement components remain fixed.
The strain-energy identity requires homogeneous prescribed displacements.
Quadratic stiffness and PU-weighted internal residuals use exact tetrahedral
barycentric moments; no global enriched mechanical solve is introduced.

## Localization and marking

The signed nodal contribution is

```text
eta_i = 0.5*rho_U(N_i*delta_i)
      + 0.5*integral(N_i*(energy(U)-energy(p))) - 0.5*S_i,
```

where `p` is the affine VEM projection, and `S_i` assigns the VEM stabilization to
original vertex hats. The element consistency correction stays separate:

```text
C_K = rho_U(U)|K - (f_h,K(u_h)-a_h,K(u_h,u_h)).
```

Continuous and discrete loads must both correspond to the solved mechanical
problem. The signed estimate is `sum(eta_i)+sum(C_K)`. Magnitudes are formed only
after assembling each complete signed node contribution. Each `abs(eta_i)` is
then distributed to its distinct adjacent cells with normalized weights
`volume_K * physical_density_K^q`. All-zero weights fall back to cell volumes.
Face-center contributions reach the two adjacent cells (one on the boundary);
cell-center contributions reach exactly one cell. Each `abs(C_K)` is added
directly to its own cell. Consequently, the sum of the marking indicators equals
`sum(abs(eta_i))+sum(abs(C_K))`.

PU-DWR uses the 2D marking thresholds: refine above twice the mean cell indicator,
and consider coarsening below one eighth of the mean. The existing sDWR marking
thresholds are retained. Density activity, maximum refinement, protected cells,
and complete coarsening families still participate in the final marking decision.

## Cost and cache lifetime

Frequent mesh changes make rebuild cost part of the design constraint. Setup
stores compact geometry, interpolation weights, boundary/load information, and
node-to-tetrahedron incidence. Patch DOF numbering is built on demand while
visiting a node. Neither patch stiffness matrices nor their factorizations are
cached across the mesh. The active entries are assembled directly, avoiding a
stored 30-by-30 operator for every tetrahedron.

The current implementation reduces work within each patch without changing the
quadratic reconstruction or residual:

- For a radial star, only the central vertex and its three incident P2 edge
  shapes can be active on a tetrahedron. Four explicit maps cover the possible
  central vertices. A dense node lookup replaces repeated tuple-key searches;
  only entries touched by the preceding patch are cleared. Central displacement
  DOFs are numbered last. Cell stars reaching a physical natural boundary use
  the general numbering path, which retains their additional free shapes.
- Assembly forms each unordered pair of active scalar shapes once as a 3-by-3
  displacement block. It scatters the block into the active lower triangle,
  respecting component constraints and arbitrary local DOF ordering.
- Patches with at least 32 free DOFs use LAPACK Cholesky and triangular solves
  on views of the active workspace. Smaller patches use the direct scalar
  kernel. The active matrix is neither copied nor retained as a cached factor.

Each patch uses a dense local factorization. Its cost depends on the number of
free patch DOFs, so unusually large vertex stars or faces with many edges can
cost more than the ordinary benchmark meshes. Dense matrix storage is bounded by
the largest visited patch per worker. Each worker also retains a node lookup
whose size follows the largest PU node count it has encountered.

Worker-local scratch grows to the largest visited patch and is reused for later
patches and mesh rebuilds; it is not shared between workers. Start Julia with
`--threads=4` to use four workers for the independent patch solves. Mesh/cache
setup, coarse-field reconstruction, and final localization collection remain
serial. The benchmarks set BLAS to one thread so local LAPACK calls do not
create nested parallelism. Localization is collected in deterministic cell order. Performance
measurements should report both rebuild and estimation time, including one-use
caches, alongside repeated evaluations on unchanged meshes.

Rebuild the cache after geometry, topology, material constants, physical
constraints, or loads change. Density and displacement changes reuse the cache.
The optimization driver rebuilds after adaptation and evaluates the estimator
before the density update, so the displacement and physical density agree. That
evaluation is reused for marking and snapshots. The final export uses the final
equilibrium solve and accepted physical density.

## Reusable interface and results

After loading the mechanical types, include
`src/refinement_utils/pu_dwr.jl` for the estimator functions. Build a cache with
`build_pu_dwr_cache(cv, states, eldata, ch, sim_pars; ...)`, then call
`estimate_pu_dwr!(cache, u, states, eldata, sim_pars; q=1.0)`.
Its optional `density` argument must be the physical density used for `u`.

The returned `PU3DEstimate` contains state-ordered `indicators`, signed `nodal`
and `element_correction` arrays, and scalar `signed` and `magnitude` values.
Arrays are borrowed from the cache and remain valid until its next evaluation;
copy them to retain a historical field. The optional observer executes serially
and exposes each solved local quadratic correction for validation.

Physical boundary definitions must be retained when constructing a custom
problem. Create `boundary_data=PU3DBoundaryData()` and use
`pu3d_add_neumann!(boundary_data, ch, cv.dh, cv.facedata_col, name, traction)`
for a face traction, or its overload without `cv.facedata_col` for an edge load.
Use `pu3d_add_dirichlet!(boundary_data, ch, cv.dh, cv.facedata_col, name, value;
c_dofs=...)` for prescribed face components, or its overload without face data
for an edge or node set. Original point restraints stay point restraints:
fixing both ends of an edge does not prescribe its quadratic midpoint. Only
explicit edge/face data constrain auxiliary values along that physical entity.
Pass this `boundary_data` to `build_pu_dwr_cache`. The standard benchmark boundary
conditions are wired automatically. Cache setup verifies that the reconstructed
discrete loads agree with the mechanical load vector. Untracked nodal loads are
rejected because an assembled nodal vector does not identify a unique continuous
traction. Body forces passed through `rhs_fun` are assembled in both the mechanical
problem and the PU residual using matching tetrahedral integration moments.

`SimulationResults.error_history` records `(step, signed, magnitude)` whenever
PU-DWR is evaluated, including the final accepted state. It is not a mandatory
per-iteration history when adaptivity and snapshots are disabled. The recorded
`error_estimator` and `pu_marking_exponent` identify the selected configuration.

The method assumes first-order 3D VEM mechanics, isotropic linear elasticity,
an equilibrated discrete displacement, star-shaped face/cell fans, and positive
stiffness on every solved patch.
Singular patches are reported rather than silently regularized. The estimator
is an approximation to the error, not a guaranteed error bound.

## Verification

The focused validation covers exact tetrahedral moments and polynomial
reproduction, shared-face conformity, hanging interfaces, local constraints,
signed localization and magnitude conservation, density scaling, cache rebuilds,
and adaptive optimization. The manufactured unit-cube solution `u=(x²,0,0)` uses
matching body force and tractions, with `E=1`, `nu=0.3`, and uniform density `0.6`.
Its true strain-energy error and signed PU estimate are:

| Uniform mesh | True error | Signed PU estimate | Estimate / true error |
| --- | ---: | ---: | ---: |
| 2 × 2 × 2 | 0.01196093 | 0.01156020 | 0.9665 |
| 3 × 3 × 3 | 0.00533842 | 0.00539471 | 1.0105 |
| 4 × 4 × 4 | 0.00300973 | 0.00311670 | 1.0355 |

The true errors decay approximately as `h²`; the estimates are within 4% on
this test. This is a verification case, not a general effectivity guarantee.
Additional tests use affine traction equilibria, unequal densities, hanging
interfaces, nonuniform face/edge loads, and all three regularizations with
projection during adaptive runs. A production bending-beam run also exercises
line loading and isolated point supports, including snapshot and final exports.
Serial and two-thread patch localization agree within the test tolerances.
Estimator arrays remain borrowed from the cache. Current warm allocation checks
on 8-cell and 64-cell meshes measure zero bytes with one Julia thread and
2,528 bytes with four threads when discarding the returned wrapper. The latter
is fixed task-launch overhead on these meshes, not storage growing with the
cell count. The profiling wrappers retain additional small temporary objects;
their allocation figures are separate from these direct kernel checks.
The existing mechanics suite and all 20 WP1 fixed/adaptive regularization
combinations also pass with the original sDWR selection.

The optimized kernels are checked against independent or frozen reference
implementations: 5,938 local map/DOF comparisons, 128 block-assembly checks,
400 active-view solve checks, and 301 independent tetrahedral checks. In the
512-cell numbering microbenchmark, the numbering stage decreased from 12.41 ms
to 3.47 ms with zero measured allocation. The block-assembly microbenchmark
showed a 4–5× kernel speedup. These are individual stage measurements, not
whole-estimator speedups.

Run the checks and the separate rebuild-plus-estimate benchmark from the
repository root:

```text
julia --project=. --startup-file=no Tests/pu_options.jl
julia --project=. --startup-file=no Tests/pu_tetra.jl
julia --project=. --startup-file=no Tests/pu_geometry.jl
julia --project=. --startup-file=no Tests/pu_dwr.jl
julia --project=. --startup-file=no Tests/pu_loads.jl
julia --project=. --startup-file=no Tests/pu_body_loads.jl
julia --project=. --startup-file=no Tests/pu_integration.jl
julia --project=. --startup-file=no Tests/pu_boundary_problem.jl
julia --project=. --startup-file=no Tests/pu_performance.jl
julia --project=. --startup-file=no Tests/pu_numbering_minimal.jl
julia --project=. --startup-file=no Tests/pu_assembly_benchmark.jl
julia --project=. --startup-file=no Tests/pu_solver_benchmark.jl
julia --project=. --startup-file=no Tests/pu_optimization.jl
julia --project=. --startup-file=no Tests/pu_benchmark.jl
```

The benchmark script's default run rebuilds unchanged meshes. For actual
alternating refinement/coarsening, start Julia with this project and run:

```julia
include("Tests/pu_benchmark.jl")
benchmark_pu_remeshing(8; cycles=6, warmup=2)
```

This starts with 512 cells, alternates between 512 and 519 cells, and reports
four measured cycles after two warmup cycles. It records mesh adaptation,
mechanical solve, PU setup, and PU evaluation separately. Garbage collection is
included in wall times and also reported separately. The full-cycle time
includes all four stages, not only estimator work on an unchanged mesh.

### Initial implementation: remeshing baseline

The following measurements precede the block assembly, radial-star numbering,
and active-view LAPACK optimizations described above. They are retained as a
historical baseline, not as timings for the current implementation. The initial
implementation was measured on Windows with Julia 1.12.6, one Julia thread and
one BLAS thread, using the four measured 512/519-cell cycles above:

| Stage | Mean time | Observed range |
| --- | ---: | ---: |
| PU setup after mesh changes | 5.3 ms | 3.4–7.0 ms |
| PU evaluation | 156.1 ms | 139.9–193.8 ms |
| PU setup + evaluation | 161.4 ms | — |
| Full mesh/mechanics/PU cycle, including GC | 438.2 ms | 169.9–1013.8 ms |

Setup accounts for 3.3% of the measured PU setup-plus-evaluation time. The full
cycle has a median of 284.5 ms; garbage collection during mechanics contributed
up to 784.9 ms in one cycle and explains much of its range. No PU-stage garbage
collection occurred in this run. Peak retained cache size was 5.19 MiB, including
246 KiB of reusable worker scratch, and stayed bounded through the alternating
refinement/coarsening cycles. Each retained estimator result added 48 bytes.

## Direct sDWR overhead comparison

`Tests/dwr_overhead_benchmark.jl` compares the production estimators on
identical solved meshes, physical densities, boundary conditions and topology
histories. sDWR includes its nodal averaging and dictionary-to-state-vector
conversion; PU-DWR includes a fresh cache rebuild when reporting its total.
Compilation and initial scratch sizing are warmed; garbage collection remains
included. The integration fixture uses a CPU direct mechanical solver.

### Current estimator cost relative to the primal solve

The current production estimators were also timed alongside the primal assembly,
boundary-condition application, and linear solve on the same fixed meshes. These
measurements explicitly warm the cache-reuse construction path and alternate the
estimator timing order. The following four-thread results are medians of seven
observations, with GC included. PU-DWR includes rebuilding its cache for each
estimate. BLAS uses one thread.

| Cells | Primal assembly | Boundary application | Linear solve | Primal total |
| ---: | ---: | ---: | ---: | ---: |
| 512 | 6.04 ms | 0.326 ms | 21.7 ms | 28.9 ms |
| 1,728 | 20.3 ms | 0.984 ms | 78.5 ms | 137.7 ms |

Each stage column is its own median. The primal total is timed around the whole
sequence, including allocation, GC, and instrumentation overhead; it is not the
sum of those medians.

| Cells | sDWR | sDWR / primaPLeasel | PU rebuild + estimate | PU / primal |
| ---: | ---: | ---: | ---: | ---: |
| 512 | 0.561 ms | 1.94% | 17.6 ms | 61.0% |
| 1,728 | 2.15 ms | 1.56% | 52.6 ms | 38.2% |

The preceding paired run measured sDWR at 2.3%/2.6% and PU-DWR at 48.1%/53.2%
of primal time for the same two sizes. Across these runs, a useful approximate
description is **1.5–3% for sDWR and 40–60% for rebuilt PU-DWR**. These
percentages divide estimator time by primal time. In the latest table, PU-DWR
accounts for approximately 28–38% of their combined time.

The primal fixture uses a CPU direct Cholesky solver with one BLAS thread, so
this comparison does not predict the overhead relative to a production
cluster's PETSc/HYPRE solve. Density updates, exports, and mesh adaptation are
outside the fixed-mesh primal timings. Measurements vary with runtime and GC;
ratios are formed from the reported medians, not from a universal cost model.

In a separate six-cycle comparison alternating between 512 and 519 cells,
the medians were 0.920 ms for sDWR, 31.6 ms for PU setup plus evaluation, and
54.4 ms for mechanics including boundary-handler construction. The median
shared mesh/mechanics stage sum plus sDWR was 59.2 ms, versus 93.3 ms plus PU
(1.57×). This is a stage-sum comparison using the same topology history, not
two separately optimized mesh trajectories. Mechanics timing was particularly
variable, and a shared run may collect allocations from either estimator.

### Current kernel comparison

The frozen original patch kernel and the optimized kernel were compared in the
same process at the same thread count. The following medians use seven samples
per configuration on Windows with Julia 1.12.6 and one BLAS thread. The sDWR
column remains serial in both runs. PU total includes a fresh cache rebuild and
an immediately following estimate.

| Julia threads | Cells | sDWR | Original PU evaluation | Optimized PU evaluation | PU rebuild + estimate | PU total / sDWR |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 512 | 0.975 ms | 99.2 ms | 54.8 ms | 43.6 ms | 44.7× |
| 1 | 1,728 | 4.37 ms | 453 ms | 143 ms | 211 ms | 48.3× |
| 4 | 512 | 0.670 ms | 31.0 ms | 11.6 ms | 14.1 ms | 21.0× |
| 4 | 1,728 | 2.86 ms | 85.0 ms | 33.4 ms | 53.0 ms | 18.5× |

Within the one-thread run, the original-to-optimized evaluation ratios were
1.81× and 3.16× for 512 and 1,728 cells; in the four-thread run they were 2.66×
and 2.55×. Setup medians were 5.34/17.1 ms in the one-thread run and
2.79/12.9 ms in the four-thread run. Setup is serial; these differing setup
times reflect run variability, not setup parallelism.

Each timing column is a median of its own observations. Cached evaluation and
freshly rebuilt evaluation are separate calls, so their medians are not
additive: the 512-cell one-thread rebuilt total is below its cached-evaluation
median. Estimator kernels were warmed, but this original/current comparison did
not separately warm the first cache-reuse constructor call; its first 512-cell
setup observation may include first-time compilation of that path. The raw
samples also show substantial timing variation, including a
912 ms collection during one 1,728-cell setup. No samples or GC times were
removed. These measurements support the reduction in patch cost, but do not
provide a controlled one-to-four-thread scaling ratio. The estimator remains
more expensive than sDWR because it solves local quadratic elasticity problems.

The original/current kernel comparison can be repeated in Julia with the
desired `--threads` setting:

```julia
include("Tests/pu_dwr.jl")
include("Tests/pu_stage_profile.jl")
benchmark_pu_optimization(8; repeats=7)
benchmark_pu_optimization(12; repeats=7)
```

### Initial implementation: sDWR comparison baseline

The table below predates the current patch optimizations. Its approximately
120× overhead is a historical measurement, not a claim about the optimized
implementation. These measurements used Julia 1.12.6 on Windows, one Julia
thread and one BLAS thread.

| Case | sDWR median | PU-DWR rebuild + estimate median | Ratio |
| --- | ---: | ---: | ---: |
| 64 fixed cells, 7 samples | 0.118 ms | 18.4 ms | 156× |
| 512 fixed cells, 7 samples | 1.50 ms | 168 ms | 112× |
| 1,728 fixed cells, 7 samples | 5.01 ms | 604 ms | 120× |
| Alternating 512/519 cells, 6 measured cycles | 0.744 ms | 90.5 ms | 122× |

The earlier setup percentage describes setup's share of PU time, not PU overhead
relative to sDWR. The additional cost is chiefly local quadratic patch assembly
and dense solves. sDWR evaluates a stabilization defect without these solves.
Although PU reuses its numerical scratch, that does not remove its arithmetic cost.
Absolute timings vary across phases; estimator ratios are more consistent here.

In the alternating-mesh case, the median shared mesh/mechanics stage sum plus
sDWR was 25.6 ms, compared with 132.7 ms plus PU-DWR (5.17×). These are stage-sum
comparisons on the same mesh history, not separately timed complete optimization
runs. Mechanics garbage collection can also include allocations from either
estimator. Overall optimization cost depends on the solver and on the different
adaptive meshes selected by the two indicators.

Benchmark the current implementation with:

```text
julia --threads=1 --project=. --startup-file=no Tests/dwr_overhead_benchmark.jl
julia --threads=4 --project=. --startup-file=no Tests/dwr_overhead_benchmark.jl
```

The script prints all samples, GC time and timing boundaries. Its PU closure
captures the cache and returns a borrowed vector, so its small wrapper allocation
is not a direct production-kernel allocation measurement. The four-thread run
parallelizes PU patch evaluation only; sDWR remains serial. Whole-estimator
speedup therefore also depends on the serial setup, reconstruction, and
localization work.
