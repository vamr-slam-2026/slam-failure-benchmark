#!/usr/bin/env bash
# Pull results from Euler to the local repo (trajectories, status, metrics, plots, logs).
# Large intermediates (e.g. mast3r_raw/ point clouds) are skipped unless FULL=1.
#
#   EULER_USER=<nethz> jobs/scripts/sync_results.sh            # -> ./results/
#   EULER_USER=<nethz> jobs/scripts/sync_results.sh /tmp/res   # custom destination
#   DRY_RUN=1 ...                                              # show what would be copied
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
EULER_USER="${EULER_USER:?set EULER_USER to your nethz username}"
EULER_HOST="${EULER_HOST:-euler.ethz.ch}"
REMOTE_RESULTS="${REMOTE_RESULTS:-/cluster/scratch/$EULER_USER/results}"
DEST="${1:-$REPO_ROOT/results}"
mkdir -p "$DEST"

FILTER=()
if [[ "${FULL:-0}" != 1 ]]; then
    FILTER=(--include='*/'
            --include='traj.txt' --include='KeyFrameTrajectory.txt' --include='lost.txt'
            --include='status.txt' --include='log.txt' --include='results.json'
            --include='timeline.csv' --include='timeline.png'
            --exclude='*' --prune-empty-dirs)
fi
OPTS=(-avz)
[[ "${DRY_RUN:-0}" == 1 ]] && OPTS+=(--dry-run)

rsync "${OPTS[@]}" "${FILTER[@]}" "$EULER_USER@$EULER_HOST:$REMOTE_RESULTS/" "$DEST/"
echo "synced -> $DEST"
echo "next: python -m eval.aggregate --results-root $DEST"
