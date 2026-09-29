#!/bin/bash
# One descriptive SLURM job per combination, using job_template.sh.
set -euo pipefail

DRY_RUN=0
SMOKE=0
for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=1 ;;
        --smoke) SMOKE=1 ;;
        -h|--help)
            echo "Usage: bash cluster/run_sweeps.sh [--dry-run] [--smoke]"
            echo "Edit the axes below. --smoke selects one small, five-step MBB run."
            echo "--dry-run generates scripts without submitting jobs."
            exit 0 ;;
        *) echo "Unknown argument: $arg" >&2; exit 2 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Experiment axes. Defaults reproduce the 16 combinations in the old sweep.
BENCHMARK_CASES=(MBB_sym L_cantilever)        # Options: MBB_sym | Cantilever_sym | Bending_Beam_sym | simple_lever | pressure_plate | L_cantilever
SOLVERS=(hypre)                  # petsc | hypre
MAX_REF_LEVELS=(3 4 5 6)
MESH_TYPES=(Hexahedra Voronoi)
ADAPTIVITY_OPTIONS=(true false)
DENSITY_MARKING_OPTIONS=(true)
MAX_OPT_STEPS=400
ADAPTIVITY_AT_START=true

# Cluster resources; override these through the submitting environment.
PARTITION=${PARTITION:-smp}
CPUS_PER_TASK=${CPUS_PER_TASK:-16}
# Empty overrides select total job RAM automatically below.
MEM_PER_JOB=${MEM_PER_JOB:-}
MEM_PER_CPU=${MEM_PER_CPU:-}
TIME_LIMIT=${TIME_LIMIT:-}       # Empty selects a limit by mesh/refinement.
CONSTRAINT=${CONSTRAINT-'[CPU_ARCH:avx512|CPU_ARCH:avx2]'}
MAIL_USER=${MAIL_USER:-}         # Empty means no email notifications.

if (( SMOKE )); then
    BENCHMARK_CASES=(MBB_sym)
    SOLVERS=(petsc)
    MAX_REF_LEVELS=(1)
    MESH_TYPES=(Hexahedra)
    ADAPTIVITY_OPTIONS=(true)
    DENSITY_MARKING_OPTIONS=(true)
    MAX_OPT_STEPS=5
    CPUS_PER_TASK=${SMOKE_CPUS:-2}
    if [[ -n "${SMOKE_MEM_PER_JOB:-}" || -n "${SMOKE_MEM_PER_CPU:-}" ]]; then
        MEM_PER_JOB=${SMOKE_MEM_PER_JOB:-}
        MEM_PER_CPU=${SMOKE_MEM_PER_CPU:-}
    fi
    TIME_LIMIT=${SMOKE_TIME_LIMIT:-0-00:30:00}
fi

if [[ -n "$MEM_PER_JOB" && -n "$MEM_PER_CPU" ]]; then
    echo "Set either MEM_PER_JOB or MEM_PER_CPU, not both." >&2
    exit 2
fi

# Provisional limits with headroom over measured hex levels 3/4 and extrapolated
# levels 5/6. Time is in minutes, memory in total GiB (independent of CPU count).
# Same limits for adaptive/nonadaptive runs; no assumed saving from adaptivity.
suggested_resource() {
    local mesh=$1 ref=$2 resource=$3 memory minutes value
    case "$ref" in
        1|2|3) memory=4; minutes=30 ;;
        4) memory=8; minutes=60 ;;
        5) memory=16; minutes=360 ;;
        6) memory=100; minutes=2880 ;;
        *) echo "No $resource estimate for level $ref; set MEM_PER_JOB/TIME_LIMIT explicitly." >&2; return 2 ;;
    esac
    case "$resource" in
        memory) value=$memory ;;
        time) value=$minutes ;;
    esac
    case "$mesh" in
        Hexahedra) ;;
        Voronoi) value=$(( (value * 5 + 3) / 4 )) ;; # +25%, rounded up
        *) echo "No $resource estimate for mesh $mesh; set MEM_PER_JOB/TIME_LIMIT explicitly." >&2; return 2 ;;
    esac
    if [[ "$resource" == memory ]]; then
        printf '%sG\n' "$value"
    else
        printf '%d-%02d:%02d:00\n' "$((value / 1440))" "$((value / 60 % 24))" "$((value % 60))"
    fi
}

if (( ! DRY_RUN )) && ! command -v sbatch >/dev/null 2>&1; then
    echo "sbatch is unavailable. Use --dry-run to generate scripts locally." >&2
    exit 1
fi

# Pin current PETSc defaults/overrides into the generated scripts.
PETSC_KEYS=(PETSC_CG_RTOL PETSC_CG_MAXIT PETSC_GAMG_THRESHOLD PETSC_GAMG_RBM
    PETSC_GAMG_L1CHEB PETSC_GAMG_CHEB_EMIN PETSC_GAMG_CHEB_EMAX PETSC_GAMG_VIEW PETSC_VERBOSE)
PETSC_DEFAULTS=(1e-4 1000 0.01 1 0 0.1 1.0 0 1)

# mktemp provides a unique suite even when two submissions start simultaneously.
mkdir -p "$SCRIPT_DIR/jobs" "$SCRIPT_DIR/logs" "$SCRIPT_DIR/results"
JOB_DIR=$(mktemp -d "$SCRIPT_DIR/jobs/$(date +%Y%m%d-%H%M%S)_XXXXXX")
SUITE=$(basename "$JOB_DIR")
LOG_DIR="$SCRIPT_DIR/logs/$SUITE"
RESULTS_DIR="$SCRIPT_DIR/results/$SUITE"
# SLURM opens log files before running the script: parents must exist now.
mkdir -p "$LOG_DIR" "$RESULTS_DIR"
printf 'job_name\tjob_script\tsubmission\n' > "$JOB_DIR/submissions.tsv"

COUNT=0
for ref in "${MAX_REF_LEVELS[@]}"; do
for benchmark in "${BENCHMARK_CASES[@]}"; do
for solver in "${SOLVERS[@]}"; do
for mesh in "${MESH_TYPES[@]}"; do
for adapt in "${ADAPTIVITY_OPTIONS[@]}"; do
for density in "${DENSITY_MARKING_OPTIONS[@]}"; do
    NAME="toopt_${benchmark}_${solver}_${mesh}_r${ref}_a${adapt}_b${ADAPTIVITY_AT_START}_fdiamond_lfalse_d${density}_s${MAX_OPT_STEPS}"
    JOB_SCRIPT="$JOB_DIR/$NAME.sh"
    if [[ -n "$MEM_PER_CPU" ]]; then
        MEMORY_OPTION=--mem-per-cpu
        MEMORY_VALUE=$MEM_PER_CPU
    else
        MEMORY_OPTION=--mem
        MEMORY_VALUE=${MEM_PER_JOB:-$(suggested_resource "$mesh" "$ref" memory)}
    fi
    JOB_TIME_LIMIT=${TIME_LIMIT:-$(suggested_resource "$mesh" "$ref" time)}
    ARGS=(-c "$benchmark" --solver "$solver" -r "$ref" -m "$mesh"
        -a "$adapt" -b "$ADAPTIVITY_AT_START" --flux_scheme diamond --laplace_rescale false
        -d "$density" -s "$MAX_OPT_STEPS")
    {
        printf '#!/bin/bash -l\n'
        printf '#SBATCH --job-name=%s\n' "$NAME"
        printf '#SBATCH --nodes=1\n#SBATCH --ntasks=1\n'
        printf '#SBATCH --cpus-per-task=%s\n' "$CPUS_PER_TASK"
        printf '#SBATCH %s=%s\n' "$MEMORY_OPTION" "$MEMORY_VALUE"
        printf '#SBATCH --time=%s\n#SBATCH --partition=%s\n' "$JOB_TIME_LIMIT" "$PARTITION"
        if [[ -n "$CONSTRAINT" ]]; then printf '#SBATCH --constraint=%s\n' "$CONSTRAINT"; fi
        if [[ -n "$MAIL_USER" ]]; then
            printf '#SBATCH --mail-user=%s\n#SBATCH --mail-type=END,FAIL\n' "$MAIL_USER"
        fi
        printf '#SBATCH --output="%s/%s_%%j.out"\n' "$LOG_DIR" "$NAME"
        printf '#SBATCH --error="%s/%s_%%j.err"\n' "$LOG_DIR" "$NAME"
        printf '\nPROJECT_ROOT=%q\nRESULTS_BASE=%q\n' "$PROJECT_ROOT" "$RESULTS_DIR/$NAME"
        printf 'ARGS=('; printf ' %q' "${ARGS[@]}"; printf ' )\n'
        for i in "${!PETSC_KEYS[@]}"; do
            key=${PETSC_KEYS[$i]}
            printf 'export %s=%q\n' "$key" "${!key-${PETSC_DEFAULTS[$i]}}"
        done
        cat "$SCRIPT_DIR/job_template.sh"
    } > "$JOB_SCRIPT"
    bash -n "$JOB_SCRIPT"
    if (( DRY_RUN )); then
        SUBMISSION=dry-run
    else
        SUBMISSION=$(sbatch --parsable "$JOB_SCRIPT")
    fi
    printf '%s\t%s\t%s\n' "$NAME" "$JOB_SCRIPT" "$SUBMISSION" >> "$JOB_DIR/submissions.tsv"
    echo "$SUBMISSION: $NAME ($MEMORY_OPTION=$MEMORY_VALUE, --time=$JOB_TIME_LIMIT)"
    COUNT=$((COUNT + 1))
done; done; done; done; done; done

echo "$COUNT jobs; scripts: $JOB_DIR"
echo "Logs: $LOG_DIR"
echo "Results: $RESULTS_DIR"
