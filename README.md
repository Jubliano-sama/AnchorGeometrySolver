# AnchorGeometrySolver

Bundled workspace for the SmartClicker anchor geometry solver experiments.

The repository preserves the original experimental scripts under `work/` so older command lines and imports keep working. It also includes the patched `SmartClicker-GUI` source snapshot used by the experiments, especially `work/SmartClicker-GUI/uwb_capture/anchor_geometry.py`.

## Layout

- `anchor_geometry_solver/` - reusable benchmark harness package for generated cases, solver methods, sweeps, summaries, and CLI runs.
- `configs/` - benchmark and sweep configs.
- `work/anchor_solver_*.py` - simulation, seed, graph-scaffold, ML, PPO, and evaluation scripts from the solver exploration.
- `work/agent_*.py` - parallel sweep/probe helpers used during graph-shortest and ML-distance experiments.
- `work/SmartClicker-GUI/` - cloned SmartClicker GUI source with the current anchor geometry implementation.
- `docs/results/` - compact CSV result snapshots worth keeping in git.
- `outputs/` - local generated figures, logs, checkpoints, and large result files. This directory is intentionally ignored except for `.gitkeep`.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install -r requirements.txt
```

For CUDA training, install a CUDA-enabled PyTorch wheel that matches the local driver instead of the default CPU wheel if needed.

## Harness

New solver work should go through the `anchor_geometry_solver` package. The legacy scripts in `work/` remain available, but the harness gives us reusable case generation, solver method specs, grid expansion, parallel execution, and standard CSV output.

Config files live under `configs/`. A benchmark config has:

- `layouts`: generated layout buckets or explicit random/grid shape requests.
- `methods`: solver pipelines; add `grid` under a method to sweep parameters.
- `runner`: `serial`, `thread`, or `process` execution plus output location.

Run a tiny dependency-light smoke, or a graph-shortest YAML sweep after installing `requirements.txt`:

```powershell
python -m anchor_geometry_solver bench configs/sweeps/tiny_serial.json
python -m anchor_geometry_solver bench configs/sweeps/graph_scaffold_smoke.yaml
```

Each run writes:

```text
outputs/<benchmark-name>/
  detail.csv
  summary.csv
```

Use `thread` or `process` backends for expensive sweeps. The nonlinear solver still sets BLAS thread counts to one in worker processes so case-level parallelism does not fight with library-level threading.

## Useful Legacy Entry Points

```powershell
python work/anchor_solver_graph_scaffold_weight_sweep_buckets.py --help
python work/anchor_solver_weighted_ppo_strong.py --help
python work/anchor_solver_weighted_ppo_curriculum.py --help
python work/anchor_solver_p95_solver_case_compare.py --help
```

The recent graph-shortest baseline has usually been run with settings close to:

```powershell
python work/anchor_solver_p95_solver_case_compare.py --solver graph_shortest --max-hops 2 --relative-sigma 0.30 --scaffold-weight 2.0 --hop-weight-base 1.4
```

For PPO curriculum work on Windows, prefer `--solve-threads 1` on strong evaluation/training branches until the native threaded nonlinear solve crash is fully eliminated.

## Current Notes

- Sparse anchor edges are generated from noisy known ranges; fair layouts should enforce at least three known connections per anchor.
- Graph-shortest scaffold distances remain the most robust non-ML baseline in the grid tail at the moment.
- The weighted PPO model is useful for experimentation, but the last comparison still had graph-shortest ahead or tied on the most difficult grid cases.
- Checkpoints and generated figures stay out of git. Put durable conclusions in `docs/` and regenerable artifacts in `outputs/`.
