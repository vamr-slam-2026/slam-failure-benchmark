# slam-failure-benchmark

Benchmark of monocular SLAM failures (ORB-SLAM3, DROID-SLAM, MASt3R-SLAM; DPVO optional) on
12 sequences from EuRoC, TUM RGB-D, TartanAir, LaMAria, UZH-FPV and Bonn Dynamic.
Forks live in `systems/` (git submodules, `vamr` branch). All trajectories use TUM format
(`timestamp tx ty tz qx qy qz qw`).

## Layout

| path | content |
|---|---|
| `configs/sequences.yaml` | sequences: raw path, format, frame list, GT, calibration |
| `scripts/convert/prepare_sequence.py` | raw sequence -> undistorted pinhole images + `rgb.txt`, calibs, `gt_tum.txt` |
| `eval/` | GT conversion, evaluation (ATE/RPE/coverage, raw + filtered), RPE timeline, spike frame export, aggregation |
| `jobs/` | Euler/SLURM: `common.sh`, `run.sbatch`, `runs.csv`, `submit_all.sh`, `download_on_euler.sh`, `scripts/` |

## Working with the SLAM systems (submodules)

The folders in `systems/` are git submodules. Each one points to a fork in our
organization (branch `vamr`), pinned to a specific commit:

| Folder | Fork | Upstream |
|---|---|---|
| `systems/ORB_SLAM3` | vamr-slam-2026/ORB_SLAM3 | UZ-SLAMLab/ORB_SLAM3 |
| `systems/DROID-SLAM` | vamr-slam-2026/DROID-SLAM | princeton-vl/DROID-SLAM |
| `systems/MASt3R-SLAM` | vamr-slam-2026/MASt3R-SLAM | rmurai0610/MASt3R-SLAM |
| `systems/DPVO` | vamr-slam-2026/DPVO | princeton-vl/DPVO (not used in current scope) |

The team repo stores only *which commit* of each fork is used. The code itself
lives in the forks, with all changes documented in each fork's `CHANGES.md`.

### Clone everything

    git clone --recursive git@github.com:vamr-slam-2026/slam-failure-benchmark.git

Already cloned without `--recursive`, or after pulling changes from `main`:

    git submodule update --init --recursive

### Change code in a system

Changes to a system are made **in its fork**, not in the team repo:

    cd systems/DROID-SLAM
    git checkout vamr            # submodules start in "detached HEAD"; switch to the branch first
    git pull
    # ... edit, test ...
    git add <files>
    git commit -m "Add --seed flag"
    git push                     # pushes to the fork
    # add the change to the fork's CHANGES.md

### Use the new version in the team repo

Pushing to the fork does **not** update the team repo automatically. Update the
pointer on a branch and open a pull request:

    cd <repo root>
    git checkout -b systems/update-droid
    git submodule update --remote systems/DROID-SLAM   # moves pointer to latest vamr
    git add systems/DROID-SLAM
    git commit -m "Update DROID-SLAM to vamr@<short-hash>: seed flag"
    git push -u origin systems/update-droid

### Rules

- Never edit system code without checking out `vamr` inside the submodule first,
  otherwise commits end up on a detached HEAD and get lost.
- Only build-/compatibility fixes and our instrumentation (seed, output format,
  status logging) go into the forks. Each change is logged in the fork's `CHANGES.md`.
- After `git pull` on `main`, always run `git submodule update --init --recursive`,
  so your `systems/` match what `main` expects.
- `git status` in the team repo shows `modified: systems/<name> (new commits)` when a
  submodule is at a different commit than recorded. Either commit the pointer
  (intended update) or run `git submodule update` (go back to the recorded version).

## Euler workflow

```bash
bash jobs/download_on_euler.sh          # -> $SCRATCH/data/raw, then prepares $SCRATCH/data/prepared
DRY_RUN=1 bash jobs/submit_all.sh       # check, then without DRY_RUN
# each job: jobs/scripts/run_one.sh SYSTEM DATASET SEQ -> $SCRATCH/results/<system>/<dataset>/<seq>/
EULER_USER=<nethz> bash jobs/scripts/sync_results.sh   # local ./results
python -m eval.aggregate --results-root results          # results/summary.{csv,md}
python -m eval.export_frames --results-root results --prepared-root <prepared> --out labeling
```

Labels go into a copy of `eval/labels_template.csv`
(`system,sequence,t_start,t_end,failure_type,cause,notes`).

## Local setup / tests

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements-eval.txt
.venv/bin/python -m pytest        # with ROS sourced: env -u PYTHONPATH PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ...
```
