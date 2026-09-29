#!/bin/bash
# Local lifecycle test with a fake Julia executable; no SLURM or solver required.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEST_DIR=$(mktemp -d)
TEST_PID=""
SENTINEL_PID=""
finish() {
    [[ -z "$TEST_PID" ]] || kill -TERM "$TEST_PID" 2>/dev/null || true
    [[ -z "$SENTINEL_PID" ]] || kill -TERM "$SENTINEL_PID" 2>/dev/null || true
    wait 2>/dev/null || true
    rm -rf -- "$TEST_DIR"
}
trap finish EXIT
mkdir -p "$TEST_DIR/bin"
cat > "$TEST_DIR/bin/julia" <<'MOCK'
#!/bin/bash
case "$TEST_MODE" in
    success) exit 0 ;;
    failure) exit 7 ;;
    stubborn) trap '' TERM ;;
esac
printf '%s\n' "$$" > "$PID_FILE"
exec sleep 120
MOCK
chmod +x "$TEST_DIR/bin/julia"
export PATH="$TEST_DIR/bin:$PATH"
export PROJECT_ROOT="$SCRIPT_DIR/.."
export TEMPLATE="$SCRIPT_DIR/job_template.sh"
export SLURM_CPUS_PER_TASK=1 SLURM_JOB_NAME=cleanup-test
sleep 120 &
SENTINEL_PID=$!

run_case() {
    local mode=$1 signal=$2 expected=$3 status=0 child_pid=""
    export TEST_MODE="$mode" SLURM_JOB_ID="$mode-$signal"
    export RESULTS_BASE="$TEST_DIR/results" PID_FILE="$TEST_DIR/$mode-$signal.pid"
    bash -c 'ARGS=(-c MBB_sym); source "$TEMPLATE"' > "$TEST_DIR/$mode-$signal.log" 2>&1 &
    TEST_PID=$!
    if [[ "$signal" != none ]]; then
        for ((i=0; i<100; i++)); do
            [[ -s "$PID_FILE" ]] && break
            sleep 0.1
        done
        [[ -s "$PID_FILE" ]] || { echo "Fake Julia failed to start" >&2; return 1; }
        child_pid=$(cat "$PID_FILE")
        kill -"$signal" "$TEST_PID"
    fi
    wait "$TEST_PID" || status=$?
    TEST_PID=""
    [[ "$status" == "$expected" ]] || { cat "$TEST_DIR/$mode-$signal.log"; return 1; }
    grep -q "exit: $expected$" "$RESULTS_BASE/$SLURM_JOB_ID/exit_status.txt"
    [[ -f "$RESULTS_BASE/$SLURM_JOB_ID/run_info.txt" ]]
    if [[ -n "$child_pid" ]] && kill -0 "$child_pid" 2>/dev/null; then
        echo "Child $child_pid survived cleanup" >&2
        kill -KILL "$child_pid" 2>/dev/null || true
        return 1
    fi
    kill -0 "$SENTINEL_PID" # An unrelated process must remain alive.
    if [[ "$mode" == stubborn ]]; then
        grep -q 'sending SIGKILL' "$TEST_DIR/$mode-$signal.log"
    fi
    echo "PASS: $mode / $signal (exit $status)"
}

run_case success none 0
run_case failure none 7
run_case running TERM 143
run_case running HUP 129
run_case stubborn TERM 143
