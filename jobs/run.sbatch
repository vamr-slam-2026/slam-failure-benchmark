#!/usr/bin/env bash
# Generic single-run job:  sbatch [extra flags] jobs/run.sbatch SYSTEM DATASET SEQUENCE
# GPU is NOT requested here; jobs/submit_all.sh adds --gpus/--gres for droid/mast3r/dpvo.
# Manual GPU submit example:
#   sbatch --gpus=1 --gres=gpumem:20g jobs/run.sbatch droid euroc MH_04_difficult
#SBATCH --job-name=slambench
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.out

set -euo pipefail
if [[ $# -lt 3 ]]; then
    echo "usage: sbatch jobs/run.sbatch SYSTEM DATASET SEQUENCE" >&2
    exit 2
fi
SYSTEM="$1" DATASET="$2" SEQ="$3"

# Under SLURM the script runs from a spool copy: locate the repo via the submit dir.
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}}"
[[ -f "$REPO_ROOT/jobs/common.sh" ]] || REPO_ROOT="$(cd "$REPO_ROOT/.." && pwd)"
export REPO_ROOT
# shellcheck disable=SC1091
source "$REPO_ROOT/jobs/common.sh"

echo "[run.sbatch] job=${SLURM_JOB_ID:-local} host=$(hostname) system=$SYSTEM dataset=$DATASET seq=$SEQ"
nvidia-smi -L 2>/dev/null || true
exec bash "$REPO_ROOT/jobs/scripts/run_one.sh" "$SYSTEM" "$DATASET" "$SEQ"
