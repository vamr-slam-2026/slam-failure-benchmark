#!/usr/bin/env bash
# Submit one SLURM job per row of jobs/runs.csv (system,dataset,sequence).
#
#   jobs/submit_all.sh                 # all runs without a finished traj.txt
#   SYSTEMS=droid jobs/submit_all.sh   # only some systems (comma separated)
#   DATASETS=euroc,tum jobs/submit_all.sh
#   FORCE=1 jobs/submit_all.sh         # resubmit even if status=ok exists
#   DRY_RUN=1 jobs/submit_all.sh       # print sbatch commands only
#
# GPU flags for droid/mast3r/dpvo come from GPU_FLAGS.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export REPO_ROOT
# shellcheck disable=SC1091
source "$REPO_ROOT/jobs/common.sh"

RUNS_CSV="${RUNS_CSV:-$REPO_ROOT/jobs/runs.csv}"
GPU_FLAGS="${GPU_FLAGS:---gpus=1 --gres=gpumem:20g}"
CPU_FLAGS="${CPU_FLAGS:-}"
SYSTEMS="${SYSTEMS:-}"
DATASETS="${DATASETS:-}"
FORCE="${FORCE:-0}"
DRY_RUN="${DRY_RUN:-0}"

in_list() {  # in_list item "a,b,c"  (empty list = everything)
    [[ -z "$2" ]] && return 0
    [[ ",$2," == *",$1,"* ]]
}

n_sub=0 n_skip=0
cd "$REPO_ROOT"   # SLURM_SUBMIT_DIR -> run.sbatch finds the repo
while IFS=, read -r system dataset sequence _rest; do
    [[ -z "$system" || "$system" == \#* || "$system" == "system" ]] && continue
    in_list "$system" "$SYSTEMS" || continue
    in_list "$dataset" "$DATASETS" || continue

    out="$RESULTS_ROOT/$system/$dataset/$sequence"
    if [[ "$FORCE" != 1 && -f "$out/status.txt" ]] && grep -q '^status=ok$' "$out/status.txt"; then
        n_skip=$((n_skip + 1))
        continue
    fi
    flags="$CPU_FLAGS"
    is_gpu_system "$system" && flags="$GPU_FLAGS"
    name="sb-${system}-${dataset}-${sequence}"
    # shellcheck disable=SC2086
    cmd=(sbatch --parsable --job-name="$name" --output="$LOG_ROOT/${name}-%j.out" $flags
         "$REPO_ROOT/jobs/run.sbatch" "$system" "$dataset" "$sequence")
    if [[ "$DRY_RUN" == 1 ]]; then
        echo "${cmd[*]}"
    else
        jid=$("${cmd[@]}")
        echo "submitted $jid  $system $dataset $sequence"
    fi
    n_sub=$((n_sub + 1))
done < "$RUNS_CSV"
echo "submitted=$n_sub skipped(ok)=$n_skip"
