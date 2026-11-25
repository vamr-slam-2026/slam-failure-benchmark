#!/usr/bin/env bash
# Shared environment for all Euler jobs. Source it, don't execute it:
#   source "$(dirname "$0")/common.sh"
# Every variable can be overridden from the calling environment.

# ---------------------------------------------------------------- paths
export REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export SYSTEMS_DIR="${SYSTEMS_DIR:-$REPO_ROOT/systems}"
export DATA_ROOT="${DATA_ROOT:-$SCRATCH/data}"
export RESULTS_ROOT="${RESULTS_ROOT:-$SCRATCH/results}"
export LOG_ROOT="${LOG_ROOT:-$SCRATCH/logs}"
export PREPARED_ROOT="${PREPARED_ROOT:-$DATA_ROOT/prepared}"

# ---------------------------------------------------------------- modules (Euler new software stack)
# Check with `module spider cuda` / `module avail` if versions change.
EULER_MODULES="${EULER_MODULES:-stack/2024-06 gcc/12.2.0 cuda/12.1.1 cmake/3.27.7 eth_proxy}"
if command -v module >/dev/null 2>&1; then
    # shellcheck disable=SC2086
    module load $EULER_MODULES >/dev/null 2>&1 || echo "[common.sh] warning: module load '$EULER_MODULES' failed" >&2
fi

# ---------------------------------------------------------------- conda
export CONDA_ROOT="${CONDA_ROOT:-$HOME/miniforge3}"
export CONDA_ENV_EVAL="${CONDA_ENV_EVAL:-slambench-eval}"   # evo, numpy, opencv, pyyaml, matplotlib
export CONDA_ENV_DROID="${CONDA_ENV_DROID:-droidenv}"
export CONDA_ENV_MAST3R="${CONDA_ENV_MAST3R:-mast3r-slam}"
export CONDA_ENV_DPVO="${CONDA_ENV_DPVO:-dpvo}"

conda_activate() {
    # usage: conda_activate <env>
    if [[ -f "$CONDA_ROOT/etc/profile.d/conda.sh" ]]; then
        # shellcheck disable=SC1091
        source "$CONDA_ROOT/etc/profile.d/conda.sh"
    else
        echo "[common.sh] no conda at $CONDA_ROOT (set CONDA_ROOT)" >&2
        return 1
    fi
    conda activate "$1"
}

# ---------------------------------------------------------------- systems
export ORB_VOCAB="${ORB_VOCAB:-$SYSTEMS_DIR/ORB_SLAM3/Vocabulary/ORBvoc.txt}"
export ORB_MONO_BIN="${ORB_MONO_BIN:-$SYSTEMS_DIR/ORB_SLAM3/Examples/Monocular/mono_tum}"
export DROID_WEIGHTS="${DROID_WEIGHTS:-$SYSTEMS_DIR/DROID-SLAM/droid.pth}"
export DPVO_WEIGHTS="${DPVO_WEIGHTS:-$SYSTEMS_DIR/DPVO/dpvo.pth}"
export MAST3R_CONFIG="${MAST3R_CONFIG:-config/calib.yaml}"     # relative to the MASt3R-SLAM repo

# ---------------------------------------------------------------- misc
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-4}}"
export PYTHONNOUSERSITE=1
export MPLBACKEND=Agg

is_gpu_system() {
    case "$1" in droid|mast3r|dpvo) return 0 ;; *) return 1 ;; esac
}

mkdir -p "$RESULTS_ROOT" "$LOG_ROOT" 2>/dev/null || true
