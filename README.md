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
