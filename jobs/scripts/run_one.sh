#!/usr/bin/env bash
# Run one system on one prepared sequence and evaluate it.
#
#   jobs/scripts/run_one.sh SYSTEM DATASET SEQUENCE
#   SYSTEM in: orbslam3 | droid | mast3r | dpvo
#
# Input : $PREPARED_ROOT/<dataset>/<sequence>/ (scripts/convert/prepare_sequence.py)
# Output: $RESULTS_ROOT/<system>/<dataset>/<sequence>/
#           traj.txt      TUM trajectory (timestamps in the dataset time base)
#           status.txt    key=value: status (ok|failed|no_trajectory|timeout), exit_code, runtime_s, ...
#           log.txt       stdout/stderr of the system
#           results.json  eval/evaluate.py output (if GT is available)
set -uo pipefail
if [[ $# -lt 3 ]]; then
    echo "usage: $0 SYSTEM DATASET SEQUENCE" >&2
    exit 2
fi
SYSTEM="$1" DATASET="$2" SEQ="$3"

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
export REPO_ROOT
# shellcheck disable=SC1091
source "$REPO_ROOT/jobs/common.sh"

IN="$PREPARED_ROOT/$DATASET/$SEQ"
OUT="$RESULTS_ROOT/$SYSTEM/$DATASET/$SEQ"
SCRIPTS="$REPO_ROOT/jobs/scripts"
mkdir -p "$OUT"
LOG="$OUT/log.txt"
STATUS="$OUT/status.txt"
TRAJ="$OUT/traj.txt"
rm -f "$TRAJ" "$OUT/results.json"

if [[ ! -f "$IN/rgb.txt" ]]; then
    echo "[run_one] prepared input missing: $IN (run scripts/convert/prepare_sequence.py)" >&2
    printf 'status=failed\nreason=missing_input\n' > "$STATUS"
    exit 1
fi

sys_commit() { git -C "$1" rev-parse --short HEAD 2>/dev/null || echo unknown; }

write_status() {  # write_status STATUS EXIT_CODE
    local n_poses=0
    [[ -s "$TRAJ" ]] && n_poses=$(grep -vc '^#' "$TRAJ" || true)
    cat > "$STATUS" <<EOF
status=$1
exit_code=$2
system=$SYSTEM
dataset=$DATASET
sequence=$SEQ
runtime_s=$((END - START))
start=$(date -d "@$START" -Is 2>/dev/null || echo "$START")
host=$(hostname)
slurm_job_id=${SLURM_JOB_ID:-}
system_commit=$SYS_COMMIT
bench_commit=$(sys_commit "$REPO_ROOT")
n_poses=$n_poses
n_frames=$(grep -vc '^#' "$IN/rgb.txt")
EOF
}

# --------------------------------------------------------------------------- dispatch
START=$(date +%s)
RC=0
case "$SYSTEM" in
    orbslam3)
        SYS_COMMIT=$(sys_commit "$SYSTEMS_DIR/ORB_SLAM3")
        # mono_tum reads <seq>/rgb.txt and writes KeyFrameTrajectory.txt (keyframes only, TUM, s)
        # into the cwd. It opens a Pangolin viewer -> needs a display: use xvfb-run if present.
        # TODO(vamr fork): headless flag + per-frame SaveTrajectoryEuRoC output for full coverage.
        WRAP=()
        if [[ -z "${DISPLAY:-}" ]] && command -v xvfb-run >/dev/null; then WRAP=(xvfb-run -a); fi
        (cd "$OUT" && "${WRAP[@]}" "$ORB_MONO_BIN" "$ORB_VOCAB" "$IN/orbslam3.yaml" "$IN") >"$LOG" 2>&1 || RC=$?
        [[ -s "$OUT/KeyFrameTrajectory.txt" ]] && cp "$OUT/KeyFrameTrajectory.txt" "$TRAJ"
        ;;
    droid)
        SYS_COMMIT=$(sys_commit "$SYSTEMS_DIR/DROID-SLAM")
        conda_activate "$CONDA_ENV_DROID" || RC=$?
        [[ $RC == 0 ]] && python "$SCRIPTS/droid_run.py" --prepared "$IN" --out "$TRAJ" \
            --droid-dir "$SYSTEMS_DIR/DROID-SLAM" --weights "$DROID_WEIGHTS" \
            --stride "${DROID_STRIDE:-1}" >"$LOG" 2>&1 || RC=$?
        ;;
    mast3r)
        SYS_COMMIT=$(sys_commit "$SYSTEMS_DIR/MASt3R-SLAM")
        conda_activate "$CONDA_ENV_MAST3R" || RC=$?
        [[ $RC == 0 ]] && python "$SCRIPTS/mast3r_run.py" --prepared "$IN" --out "$TRAJ" \
            --mast3r-dir "$SYSTEMS_DIR/MASt3R-SLAM" --config "$MAST3R_CONFIG" >"$LOG" 2>&1 || RC=$?
        ;;
    dpvo)
        SYS_COMMIT=$(sys_commit "$SYSTEMS_DIR/DPVO")
        conda_activate "$CONDA_ENV_DPVO" || RC=$?
        [[ $RC == 0 ]] && python "$SCRIPTS/dpvo_run.py" --prepared "$IN" --out "$TRAJ" \
            --dpvo-dir "$SYSTEMS_DIR/DPVO" --weights "$DPVO_WEIGHTS" >"$LOG" 2>&1 || RC=$?
        ;;
    *)
        echo "[run_one] unknown system '$SYSTEM'" >&2
        exit 2
        ;;
esac
END=$(date +%s)

if [[ $RC != 0 ]]; then
    STATE=failed
elif [[ ! -s "$TRAJ" ]]; then
    STATE=no_trajectory
else
    STATE=ok
fi
write_status "$STATE" "$RC"
echo "[run_one] $SYSTEM $DATASET/$SEQ -> $STATE (rc=$RC, $((END - START)) s)"
tail -n 20 "$LOG" 2>/dev/null | sed 's/^/  | /'

# --------------------------------------------------------------------------- evaluate
if [[ -f "$IN/gt_tum.txt" ]] && conda_activate "$CONDA_ENV_EVAL"; then
    EXTRA=()
    [[ -f "$OUT/lost.txt" ]] && EXTRA+=(--lost "$OUT/lost.txt")
    [[ "$DATASET" == lamaria ]] && EXTRA+=(--interpolate)   # sparse pseudo-GT
    (cd "$REPO_ROOT" && python -m eval.evaluate --est "$TRAJ" --gt "$IN/gt_tum.txt" \
        --frames "$IN/rgb.txt" --out "$OUT/results.json" \
        --system "$SYSTEM" --dataset "$DATASET" --sequence "$SEQ" --plot "${EXTRA[@]}") \
        || echo "[run_one] evaluation failed" >&2
fi
[[ "$STATE" == ok ]]
