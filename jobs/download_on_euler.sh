#!/usr/bin/env bash
# Download the 12 benchmark sequences into $DATA_ROOT/raw/<dataset>/... (layout expected by
# configs/sequences.yaml), then prepare them with scripts/convert/prepare_sequence.py.
#
#   jobs/download_on_euler.sh                 # everything
#   jobs/download_on_euler.sh euroc tum       # selected datasets
#   PREPARE=0 jobs/download_on_euler.sh       # download only
#
# Large downloads are better run as a job, e.g.:
#   sbatch --time=08:00:00 --mem-per-cpu=4G --wrap "bash jobs/download_on_euler.sh"
# (eth_proxy is loaded by common.sh, compute nodes need it for internet access.)
#
# URLs marked VERIFY were not checked; dataset hosts move occasionally.
set -uo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export REPO_ROOT
# shellcheck disable=SC1091
source "$REPO_ROOT/jobs/common.sh"

RAW="$DATA_ROOT/raw"
DL="$DATA_ROOT/downloads"
mkdir -p "$RAW" "$DL"
FAILED=()

fetch() {  # fetch URL [FILENAME] -> path in $DL ; resumes partial downloads
    local url="$1" name="${2:-$(basename "$1")}"
    if [[ -s "$DL/$name" && ! -f "$DL/$name.part" ]]; then
        echo "$DL/$name"; return 0
    fi
    touch "$DL/$name.part"
    if wget -c -q --show-progress -O "$DL/$name" "$url" >&2; then
        rm -f "$DL/$name.part"; echo "$DL/$name"
    else
        echo "[download] FAILED $url" >&2; return 1
    fi
}

extract() {  # extract ARCHIVE DEST_DIR
    mkdir -p "$2"
    case "$1" in
        *.zip) unzip -q -o "$1" -d "$2" ;;
        *.tgz|*.tar.gz) tar -xzf "$1" -C "$2" ;;
        *.tar) tar -xf "$1" -C "$2" ;;
        *) echo "unknown archive $1" >&2; return 1 ;;
    esac
}

done_marker() { [[ -f "$1/.downloaded" ]]; }
mark_done()   { touch "$1/.downloaded"; }

# ------------------------------------------------------------------ EuRoC
# The old ASL server (robotics.ethz.ch/~asl-datasets) is the URL used by most SLAM repos.
# VERIFY: EuRoC was moved to the ETH Research Collection (doi:10.3929/ethz-b-000690084);
# set EUROC_BASE if the old server is unreachable.
dl_euroc() {
    local base="${EUROC_BASE:-http://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset}"
    local s room
    for s in MH_04_difficult V2_03_difficult; do
        local dst="$RAW/euroc/$s"
        done_marker "$dst" && { echo "[euroc] $s present"; continue; }
        case "$s" in MH_*) room=machine_hall ;; V1_*) room=vicon_room1 ;; V2_*) room=vicon_room2 ;; esac
        local f; f=$(fetch "$base/$room/$s/$s.zip") || { FAILED+=("euroc/$s"); continue; }
        extract "$f" "$dst" && mark_done "$dst" || FAILED+=("euroc/$s")
    done
}

# ------------------------------------------------------------------ TUM RGB-D
dl_tum() {
    local s
    for s in rgbd_dataset_freiburg1_360 rgbd_dataset_freiburg1_room; do
        local dst="$RAW/tum/$s"
        done_marker "$dst" && { echo "[tum] $s present"; continue; }
        local f; f=$(fetch "https://cvg.cit.tum.de/rgbd/dataset/freiburg1/$s.tgz") || { FAILED+=("tum/$s"); continue; }
        extract "$f" "$RAW/tum" && mark_done "$dst" || FAILED+=("tum/$s")   # tgz contains $s/
    done
}

# ------------------------------------------------------------------ Bonn RGB-D Dynamic
dl_bonn() {
    local s
    for s in rgbd_bonn_crowd rgbd_bonn_moving_obstructing_box; do
        local dst="$RAW/bonn/$s"
        done_marker "$dst" && { echo "[bonn] $s present"; continue; }
        local f; f=$(fetch "https://www.ipb.uni-bonn.de/html/projects/rgbd_dynamic2019/$s.zip") || { FAILED+=("bonn/$s"); continue; }
        extract "$f" "$RAW/bonn" && mark_done "$dst" || FAILED+=("bonn/$s")  # zip contains $s/
    done
}

# ------------------------------------------------------------------ TartanAir ECCV 2020 SLAM challenge (mono)
# VERIFY: test images + GT archives. The image archive holds mono/MH000..MH007 (hard: MH*).
# GT ("mono_gt/MHxxx.txt" expected by sequences.yaml) is a separate release; set TARTANAIR_GT_URL.
dl_tartanair() {
    local dst="$RAW/tartanair"
    if ! done_marker "$dst"; then
        local f; f=$(fetch "${TARTANAIR_MONO_URL:-https://tartanair.blob.core.windows.net/tartanair-testing1/tartanair-test-mono-release.tar.gz}") \
            && extract "$f" "$dst" && mark_done "$dst" || FAILED+=("tartanair/images")
    fi
    if [[ ! -d "$dst/mono_gt" ]]; then
        if [[ -n "${TARTANAIR_GT_URL:-}" ]]; then
            local g; g=$(fetch "$TARTANAIR_GT_URL") && extract "$g" "$dst" || FAILED+=("tartanair/gt")
        else
            echo "[tartanair] set TARTANAIR_GT_URL to the test-set GT archive (needs $dst/mono_gt/MH004.txt)" >&2
            FAILED+=("tartanair/gt")
        fi
    fi
}

# ------------------------------------------------------------------ UZH-FPV (Snapdragon, with GT)
# VERIFY: URLs and archive layout (download page: https://fpv.ifi.uzh.ch/datasets/).
dl_uzhfpv() {
    local base="${UZHFPV_BASE:-https://download.ifi.uzh.ch/rpg/web/datasets/uzh-fpv-newer-versions/v3}"
    local cbase="${UZHFPV_CALIB_BASE:-https://download.ifi.uzh.ch/rpg/web/datasets/uzh-fpv/calib}"
    local s c
    for s in indoor_forward_9_snapdragon_with_gt indoor_45_2_snapdragon_with_gt; do
        local dst="$RAW/uzhfpv/$s"
        if ! done_marker "$dst"; then
            local f; f=$(fetch "$base/$s.zip") && extract "$f" "$dst" && mark_done "$dst" || FAILED+=("uzhfpv/$s")
        fi
    done
    for c in indoor_forward_calib_snapdragon indoor_45_calib_snapdragon; do
        local dst="$RAW/uzhfpv/$c"
        if ! done_marker "$dst"; then
            local f; f=$(fetch "$cbase/$c.zip") && extract "$f" "$dst" && mark_done "$dst" || FAILED+=("uzhfpv/$c")
        fi
    done
}

# ------------------------------------------------------------------ LaMAria (ASL pinhole export)
# VERIFY: no public direct URL wired in. Set LAMARIA_BASE to the directory that serves
# <seq>.zip in ASL format (with the pGT), or place the sequences in $RAW/lamaria/<seq>/ by hand.
dl_lamaria() {
    local s
    for s in R_06_medium R_11_5cp; do
        local dst="$RAW/lamaria/$s"
        done_marker "$dst" && { echo "[lamaria] $s present"; continue; }
        if [[ -z "${LAMARIA_BASE:-}" ]]; then
            echo "[lamaria] LAMARIA_BASE not set; put $s (ASL format + pGT) into $dst" >&2
            FAILED+=("lamaria/$s"); continue
        fi
        local f; f=$(fetch "$LAMARIA_BASE/$s.zip") && extract "$f" "$dst" && mark_done "$dst" || FAILED+=("lamaria/$s")
    done
}

DATASETS=("$@")
[[ ${#DATASETS[@]} -eq 0 ]] && DATASETS=(euroc tum bonn tartanair uzhfpv lamaria)
for d in "${DATASETS[@]}"; do
    echo "=== $d"
    "dl_$d"
done

if [[ "${PREPARE:-1}" == 1 ]]; then
    echo "=== prepare"
    if conda_activate "$CONDA_ENV_EVAL"; then
        python "$REPO_ROOT/scripts/convert/prepare_sequence.py" --all --data-root "$DATA_ROOT" || FAILED+=("prepare")
    else
        FAILED+=("prepare(no eval env)")
    fi
fi

if [[ ${#FAILED[@]} -gt 0 ]]; then
    echo "incomplete: ${FAILED[*]}" >&2
    exit 1
fi
echo "all sequences downloaded and prepared under $DATA_ROOT"
