# Runtime body appended to each generated SLURM script by run_sweeps.sh.
# Project, output paths, solver settings and ARGS are pinned by the generator.
set -euo pipefail

JOB_T0=$(date +%s)
JULIA_PID=""
STOP_REASON="exit"

cleanup() {
    local status=$?
    trap - EXIT
    # Do not interrupt cleanup if SLURM sends another catchable signal.
    trap '' TERM INT HUP
    set +e
    if [[ -n "$JULIA_PID" ]]; then
        if kill -0 "$JULIA_PID" 2>/dev/null; then
            echo "Stopping Julia process $JULIA_PID ($STOP_REASON)"
            kill -TERM "$JULIA_PID" 2>/dev/null
            # Bound our grace period; SLURM may enforce a shorter KillWait.
            for ((attempt=0; attempt<10; attempt++)); do
                kill -0 "$JULIA_PID" 2>/dev/null || break
                sleep 1
            done
            if kill -0 "$JULIA_PID" 2>/dev/null; then
                echo "Julia did not stop within 10 seconds; sending SIGKILL"
                kill -KILL "$JULIA_PID" 2>/dev/null
            fi
        fi
        wait "$JULIA_PID" 2>/dev/null
    fi
    local summary="Finished: $(date -Is); elapsed: $(( $(date +%s) - JOB_T0 ))s; reason: $STOP_REASON; exit: $status"
    echo "$summary"
    if [[ -n "${TOOPT_RESULTS_DIR:-}" && -d "$TOOPT_RESULTS_DIR" ]]; then
        printf '%s\n' "$summary" > "$TOOPT_RESULTS_DIR/exit_status.txt"
    fi
    exit "$status"
}
trap cleanup EXIT
trap 'STOP_REASON=SIGTERM; exit 143' TERM
trap 'STOP_REASON=SIGINT; exit 130' INT
trap 'STOP_REASON=SIGHUP; exit 129' HUP
cd "$PROJECT_ROOT"

# One Julia process: the current PETSc backend uses MPI.COMM_SELF.
export JULIA_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS="$JULIA_NUM_THREADS"
export TOOPT_RESULTS_DIR="$RESULTS_BASE/${SLURM_JOB_ID:?Run the generated script with sbatch}"
mkdir -p "$TOOPT_RESULTS_DIR"

echo "Job: $SLURM_JOB_NAME ($SLURM_JOB_ID)"
echo "Node: ${SLURM_JOB_NODELIST:-unknown}; started: $(date -Is)"
echo "Project: $PROJECT_ROOT"
echo "Results: $TOOPT_RESULTS_DIR"
echo "Julia threads: $JULIA_NUM_THREADS; BLAS threads: 1"
printf 'Arguments:'; printf ' %q' "${ARGS[@]}"; printf '\n'
# Record the source revision and effective PETSc options alongside each run.
{
    git -C "$PROJECT_ROOT" rev-parse HEAD 2>/dev/null || true
    git -C "$PROJECT_ROOT" status --short 2>/dev/null || true
    printf 'Arguments:'; printf ' %q' "${ARGS[@]}"; printf '\n'
    env | sort | grep '^PETSC_' || true
} > "$TOOPT_RESULTS_DIR/run_info.txt"

julia --startup-file=no --project="$PROJECT_ROOT" --threads="$JULIA_NUM_THREADS" \
    "$PROJECT_ROOT/src/main_optim_runs.jl" "${ARGS[@]}" &
JULIA_PID=$!
# Waiting on an asynchronous child lets Bash process cancellation immediately.
# Preserve Julia's exit code, including solver failures, rather than wait's status
# from any later cleanup. The EXIT handler owns the child until this wait finishes.
STATUS=0
wait "$JULIA_PID" || STATUS=$?
JULIA_PID=""
exit "$STATUS"
