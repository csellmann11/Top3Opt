# ToOpt3 cluster runs

`run_sweeps.sh` generates one SLURM job per parameter combination using
`job_template.sh`, following the basic layout of AndersonPlasticity's
`run_small_solver_exp.sh`. Edit the experiment axes and fixed parameters near the
top of the generator.
The configured sweep has 32 combinations: `MBB_sym` and `L_cantilever`, four
refinement levels, two meshes, and adaptivity on/off. It uses 200 optimization
steps and HYPRE; the smoke run selects PETSc and five steps.
`FLUX_SCHEME=tpfa` is the supported face-gradient approximation for all jobs.
The former `diamond`, `taylor`, and `strong` options have been removed.
The selected scheme appears in each job's arguments and name; it is not a sweep
axis. Older generated jobs may pass `--laplace_rescale false`; this deprecated
option remains accepted as a no-op. Regenerate jobs that request removed schemes.

The Julia driver accepts `--regularization mean`, `local_max`, or `helmholtz`,
and `--projection true` enables volume-preserving projection and continuation
for any of them. Both gradient penalties also accept `--update_mode implicit`.
Helmholtz requires `--update_mode explicit`. All three use the per-cell coefficient
2·β0·h_i² (Helmholtz as r²_i); `--filter_radius` overrides Helmholtz with a uniform
radius. See [regularization options](../docs/regularization.md) for complete examples.
The generator defaults to `REGULARIZATION=mean`, `PROJECTION=false` and
`UPDATE_MODE=implicit`; override them in the environment, e.g.
`REGULARIZATION=helmholtz PROJECTION=true bash cluster/run_sweeps.sh`. Helmholtz with
the implicit update prints a warning and switches to explicit. The available values
are listed as comments at the top of `run_sweeps.sh`.

## Run on the cluster

From the repository root:

```bash
# Preview one small PETSc run; does not submit anything.
bash cluster/run_sweeps.sh --smoke --dry-run

# Submit the small run (level 1, Hexahedra, MBB_sym, five steps).
bash cluster/run_sweeps.sh --smoke

# Preview the configured sweep, then submit it.
bash cluster/run_sweeps.sh --dry-run
bash cluster/run_sweeps.sh
```

Every invocation creates a new suite. A dry run writes reviewable job scripts;
the subsequent command generates a fresh suite. You can instead submit an
individual generated script with `sbatch /absolute/path/to/job.sh`.
Generate on the cluster checkout: scripts contain absolute paths to that checkout.
The generator also works when called from another directory.

Load your cluster's Julia environment before submission. Use Julia compatible
with the project (the current manifest was created with Julia 1.12.6), make the
locally developed `../Ju3VEM` dependency available, and instantiate/precompile the
repository environment once before launching a sweep:

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'
```

The job uses `--project=<repository root>`. PETSc currently uses `MPI.COMM_SELF`,
so jobs request one task and launch one Julia process, without `mpiexecjl`.
Julia threads follow `--cpus-per-task`; BLAS threads are set to one. No external
PETSc build or AndersonPlasticity-specific MPI setup is copied into this project.

Both AVX2 and AVX-512 nodes remain eligible. The exported HostCPUFeatures preference
in `Project.toml` enables runtime invalidation: when a shared precompile cache was
built on a different CPU, the package refreshes its CPU feature information on
the executing node. This can add startup compilation time. It does not force
AVX-512 on AVX2 hardware or guarantee that every kernel uses the widest vectors.
After pulling this preference change, run the instantiate/precompile command above
once before starting the sweep. A `LocalPreferences.toml` override can supersede
the exported preference.

## Files and names

```text
cluster/
  run_sweeps.sh
  job_template.sh
  jobs/<suite>/<job-name>.sh
  jobs/<suite>/submissions.tsv
  logs/<suite>/<job-name>_<slurm-id>.out
  logs/<suite>/<job-name>_<slurm-id>.err
  results/<suite>/<job-name>/<slurm-id>/
    run_info.txt
    exit_status.txt           # Completion/failure/cancellation, when the trap runs
    SimData/                 # JLD2 and CSV
    vtk/Adaptive_Runs/        # VTK output
```

Example job name:
`toopt_MBB_sym_hypre_Hexahedra_r3_atrue_btrue_ftpfa_uimplicit_lfalse_dtrue_s200`.
Here `r` is refinement level, `a` adaptivity, `b` initial refinement, `f` flux
scheme, `u` density update mode, `l` the deprecated compatibility switch,
`d` density marking, and `s` maximum
optimization steps.
Result-directory identities include a deterministic fingerprint of all exposed
density controls, so differing regularization, projection, radius, substeps, or
tolerances do not reuse the same result directory.
For `L_cantilever`, `Voronoi` creates an L-shaped polygon mesh in the x–y plane
and extrudes it through the thickness. `Hexahedra` and `Lquad_mesh` both select
the structured L mesh; choose only one of those two to avoid duplicate runs.

Logs are created separately from results, with parent directories prepared before
submission. Generated files and outputs are ignored by Git. Each run records its
CLI arguments, PETSc options, source revision and working-tree status; the source
itself is not snapshotted, so keep the checkout unchanged while jobs are queued/running.
The exit trap reports failures as well as successful completion.

## Cancellation and cleanup

Stop a run with `scancel <job-id>`. The job script handles `TERM`, `INT`, and `HUP`:
it sends `TERM` to its own Julia child, waits up to ten seconds, then sends `KILL`
if needed and reaps that child. Julia runs asynchronously only so Bash can react
to signals while waiting; the batch job still waits for Julia to finish.
PETSc is inside that Julia process. Cleanup does not search for or kill other
Julia jobs, delete shared precompile locks, or remove logs and partial results.
SLURM remains responsible for job-wide process cleanup, including descendants.

Normal failures retain Julia's exit code. A signal caught by the shell records
143 (`TERM`), 130 (`INT`), or 129 (`HUP`) in the log and `exit_status.txt`.
A forced `SIGKILL`, node failure, or SLURM's shorter termination deadline can
prevent the handler from finishing; absence of that file does not imply success.
Use SLURM accounting to confirm the job's final state.

Regenerate job scripts after changing the generator or template. Previously
generated or already submitted scripts retain their old settings and behavior.

Run `bash cluster/test_cleanup.sh` on Linux to check success, solver failure,
`TERM`/`HUP` cancellation, and escalation for a child that ignores `TERM`.
The test uses a fake Julia executable and checks that an unrelated process survives;
it does not submit jobs or run a numerical simulation.

`TOOPT_RESULTS_DIR` redirects all output from `main_optim_runs.jl`; ordinary local
runs still default to the existing repository `Results/` directory.

## Resources and solver options

The full sweep uses partition `smp` and 16 CPUs. Total job RAM (`--mem`) and
time limits (`--time`) are selected automatically per mesh and refinement level:

| Refinement level | Hexahedra RAM | Hexahedra time | Voronoi RAM | Voronoi time |
| --- | --- | --- | --- | --- |
| 1–3 | 4G | 30 min | 5G | 38 min |
| 4 | 8G | 1 hour | 10G | 1 h 15 min |
| 5 | 16G | 6 hours | 20G | 7 h 30 min |
| 6 | 100G | 48 hours | 125G | 60 hours |

These are provisional reservations with headroom, based on the supplied hex
measurements: level 3 took 3m24s / 1563.13M peak RSS; level 4 took 24m26s /
2833.36M. Levels 5/6 use extrapolated budgets, not measured bounds. They assume
comparable workload and eight CPUs; changing solver, step count, benchmark, or
CPU count does not automatically rescale the table. Validate level 5 before level 6.
Voronoi receives 25% extra time and memory, rounded up to whole minutes/GiB.
Adaptive and nonadaptive runs use the same budgets. CPU count does not affect
the default total RAM request. Both selected resources are printed per job.
Other meshes or levels require explicit overrides for both memory and time.

Set `MEM_PER_JOB` to override total RAM and `TIME_LIMIT` to override time.
Overrides are used exactly as supplied, without the Voronoi multiplier.
The legacy `MEM_PER_CPU` override is still accepted, but cannot be combined with
`MEM_PER_JOB`.
The smoke run uses two CPUs, 4G total RAM, and 30 minutes; override these with
`SMOKE_CPUS`, `SMOKE_MEM_PER_JOB`, and `SMOKE_TIME_LIMIT`. A smoke-specific memory
override takes precedence over a general override; otherwise the general override
also applies to smoke runs. `SMOKE_MEM_PER_CPU` remains supported as an alternative.

```bash
CPUS_PER_TASK=8 MEM_PER_JOB=32G TIME_LIMIT=1-00:00:00 \
    bash cluster/run_sweeps.sh --dry-run
CONSTRAINT='' MAIL_USER=you@example.org bash cluster/run_sweeps.sh --dry-run
PETSC_CG_RTOL=1e-5 bash cluster/run_sweeps.sh --smoke --dry-run
```

`PARTITION`, `CONSTRAINT`, and optional `MAIL_USER` are configurable. Email defaults
to off. PETSc options listed in the generator are pinned at generation time,
using the current solver defaults unless overridden in the environment.
Set `SOLVERS=(petsc hypre)` to compare the two supported backends.
For PETSc setup comparisons, set `SOLVERS=(petsc)` and change
`PETSC_GAMG_SQUARE_GRAPH` near the top of `run_sweeps.sh`: `1` keeps squared-graph
aggressive coarsening (the existing default), while `0` selects MIS-2 without
graph squaring. You can also override it when generating or submitting a suite:

```bash
PETSC_GAMG_SQUARE_GRAPH=0 bash cluster/run_sweeps.sh --dry-run
PETSC_GAMG_SQUARE_GRAPH=0 bash cluster/run_sweeps.sh
```

The setting is pinned in each generated script and recorded in `run_info.txt`.
It only affects PETSc; the hierarchy is still rebuilt for every solve, including
when the mesh changes. Set `PETSC_GAMG_VIEW=1` to print the effective hierarchy
and coarsening configuration.

Monitor with `squeue -u "$USER"`; job IDs are recorded in `submissions.tsv`.
If submission fails partway through, earlier jobs remain submitted and recorded.

## What the old files did

| Old file | Role |
| --- | --- |
| `run_sweeps.sh` | Submitted a 16-task SLURM array and exported parameter lists to the worker. |
| `run_optimization.sh` | Array worker: decoded the task index and launched `main_optim_runs.jl`. Its mention of `submit_sweep.sh` was a stale name for the submission script. |
| `simple_lever.sh` | Standalone five-step test at refinement level 5. Despite its name, it selected `MBB_sym`, not `simple_lever`. |

These are different roles, not three interchangeable versions. Their contents
alone do not establish which was most recently maintained. The sweep/worker pair
contains the more complete workflow, but assumes submission from the parent of
`ToOpt3` and writes logs outside a project-owned cluster folder. The standalone
test also uses an array-job variable despite not being an array. The new workflow
replaces those path and naming conventions and explicitly selects the current solver.
