from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
from pathlib import Path
import sys
from typing import Any, Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from anchor_geometry_solver.benchmark import run_benchmark, write_csv
from anchor_geometry_solver.types import BenchmarkConfig, BenchmarkRow, LayoutSpec, MethodSpec, RunnerSpec


DEFAULT_BRANCHING: dict[str, Any] = {
    "iterations": 45,
    "beam_width": 32,
    "optimizer_seeds": 16,
    "radio_radius_m": 8.0,
    "missing_margin_m": 0.0,
    "missing_weight": 4.0,
    "missing_sigma_m": 0.75,
    "graph_upper_weight": 0.35,
    "graph_upper_factor": 1.0,
    "graph_upper_slack_m": 0.75,
    "graph_upper_sigma_m": 1.0,
    "final_visibility_weight": 1.0,
    "constrained_polish": True,
    "constrained_iterations": 55,
    "constrained_known_weight": 1.0,
}

DEFAULT_SDP: dict[str, Any] = {
    "iterations": 45,
    "radio_radius_m": 8.0,
    "missing_margin_m": 0.0,
    "known_weight": 1.0,
    "missing_weight": 0.35,
    "missing_sigma_m": 0.75,
    "graph_upper_weight": 0.15,
    "graph_upper_factor": 1.0,
    "graph_upper_slack_m": 0.75,
    "graph_upper_sigma_m": 1.0,
    "constrained_polish": True,
    "constrained_iterations": 55,
    "sdp_solver": "SCS",
    "sdp_max_iters": 3000,
    "sdp_eps": 0.0002,
}

BRANCHING_STAGE1_VALUES: dict[str, list[Any]] = {
    "radio_radius_m": [7.8, 8.0, 8.2],
    "missing_margin_m": [-0.25, 0.0, 0.25],
    "missing_weight": [0.5, 1.0, 4.0, 12.0],
    "missing_sigma_m": [0.5, 0.75, 1.0],
    "graph_upper_weight": [0.1, 0.35, 1.0],
    "graph_upper_factor": [0.98, 1.0, 1.04, 1.08],
    "graph_upper_slack_m": [0.5, 0.75, 1.0, 1.5],
    "graph_upper_sigma_m": [0.75, 1.0, 1.5],
    "final_visibility_weight": [0.25, 1.0, 4.0],
    "constrained_polish": [False, True],
    "constrained_known_weight": [0.5, 1.0, 2.0],
}

SDP_STAGE1_VALUES: dict[str, list[Any]] = {
    "radio_radius_m": [7.8, 8.0, 8.2],
    "missing_margin_m": [-0.25, 0.0, 0.25],
    "known_weight": [0.5, 1.0, 2.0],
    "missing_weight": [0.03, 0.1, 0.35, 1.0],
    "missing_sigma_m": [0.5, 0.75, 1.0],
    "graph_upper_weight": [0.05, 0.15, 0.5],
    "graph_upper_factor": [0.98, 1.0, 1.04, 1.08],
    "graph_upper_slack_m": [0.5, 0.75, 1.0, 1.5],
    "graph_upper_sigma_m": [0.75, 1.0, 1.5],
    "constrained_polish": [False, True],
}

BASE_GRAPH_PARAMS: dict[str, Any] = {
    "seed_count": 6,
    "iterations": 45,
    "max_hops": 2,
    "relative_sigma": 0.30,
    "scaffold_weight": 2.0,
    "hop_weight_base": 1.4,
    "constrained_polish": False,
}

RUNTIME_KEYS = {
    "beam_width",
    "optimizer_seeds",
    "iterations",
    "constrained_iterations",
    "sdp_solver",
    "sdp_max_iters",
    "sdp_eps",
}


def layouts(cases_per_bucket: int) -> tuple[LayoutSpec, ...]:
    return (
        LayoutSpec(
            name="random_16_24",
            bucket="Random 16-24",
            cases=cases_per_bucket,
            family="random",
            shapes=("rectangle",),
            count_range=(16, 24),
            width_range_m=(16.0, 28.0),
            height_range_m=(16.0, 28.0),
            min_vertex_connectivity=3,
        ),
        LayoutSpec(
            name="grid_16_24",
            bucket="Grid 16-24",
            cases=cases_per_bucket,
            family="grid",
            shapes=("rectangle",),
            count_range=(16, 24),
            width_range_m=(18.0, 32.0),
            height_range_m=(18.0, 32.0),
            min_vertex_connectivity=3,
        ),
        LayoutSpec(
            name="office_16_24",
            bucket="Office 16-24",
            cases=cases_per_bucket,
            family="grid",
            shapes=("corridor", "l_shape", "t_shape", "u_shape", "hollow_square", "rooms"),
            count_range=(16, 24),
            width_range_m=(18.0, 34.0),
            height_range_m=(18.0, 34.0),
            min_vertex_connectivity=3,
        ),
    )


def _param_key(params: dict[str, Any]) -> str:
    parts = []
    for key in sorted(params):
        if key in RUNTIME_KEYS:
            continue
        value = params[key]
        if isinstance(value, float):
            parts.append(f"{key}={value:g}")
        else:
            parts.append(f"{key}={value}")
    return ";".join(parts)


def _dedupe_params(candidates: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for params in candidates:
        key = _param_key(params)
        if key in seen:
            continue
        seen.add(key)
        out.append(params)
    return out


def randomized_discrete_sweep(
    default: dict[str, Any],
    values_by_key: dict[str, list[Any]],
    *,
    count: int,
    seed: int,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    candidates: list[dict[str, Any]] = [dict(default)]

    keys = list(values_by_key)
    for key in keys:
        for value in values_by_key[key]:
            params = dict(default)
            params[key] = value
            candidates.append(params)

    while len(_dedupe_params(candidates)) < count:
        params = dict(default)
        for key, values in values_by_key.items():
            params[key] = rng.choice(values)
        candidates.append(params)

    return _dedupe_params(candidates)[:count]


def _local_numeric_values(value: float, *, minimum: float | None = None, maximum: float | None = None) -> list[float]:
    if value == 0.0:
        raw = [-0.25, 0.0, 0.25]
    elif abs(value) < 0.75:
        raw = [value * 0.5, value, value * 2.0]
    else:
        raw = [value - 0.25, value, value + 0.25]
    out = []
    for item in raw:
        if minimum is not None:
            item = max(minimum, item)
        if maximum is not None:
            item = min(maximum, item)
        out.append(round(float(item), 6))
    return sorted(set(out))


def local_values_around_best(best: dict[str, Any], stage1_values: dict[str, list[Any]]) -> dict[str, list[Any]]:
    local: dict[str, list[Any]] = {}
    for key, stage_values in stage1_values.items():
        value = best[key]
        if isinstance(value, bool):
            local[key] = [False, True]
        elif key == "radio_radius_m":
            local[key] = _local_numeric_values(float(value), minimum=7.4, maximum=8.6)
        elif key == "missing_margin_m":
            local[key] = _local_numeric_values(float(value), minimum=-0.5, maximum=0.5)
        elif key == "missing_weight":
            local[key] = sorted(set(round(float(v), 6) for v in [float(value) * 0.5, float(value), float(value) * 2.0]))
        elif key in {"graph_upper_weight", "final_visibility_weight", "known_weight", "constrained_known_weight"}:
            local[key] = sorted(set(round(float(v), 6) for v in [float(value) * 0.5, float(value), float(value) * 2.0]))
        elif key == "graph_upper_factor":
            local[key] = _local_numeric_values(float(value), minimum=0.95, maximum=1.1)
        elif key == "graph_upper_slack_m":
            local[key] = _local_numeric_values(float(value), minimum=0.25, maximum=2.0)
        elif key in {"missing_sigma_m", "graph_upper_sigma_m"}:
            local[key] = _local_numeric_values(float(value), minimum=0.25, maximum=2.0)
        else:
            local[key] = list(stage_values)
    return local


def local_sweep(
    best: dict[str, Any],
    values_by_key: dict[str, list[Any]],
    *,
    count: int,
    seed: int,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    candidates: list[dict[str, Any]] = [dict(best)]

    for key, values in values_by_key.items():
        for value in values:
            params = dict(best)
            params[key] = value
            candidates.append(params)

    keys = list(values_by_key)
    while len(_dedupe_params(candidates)) < count:
        params = dict(best)
        changed = rng.sample(keys, k=min(len(keys), rng.randint(2, 4)))
        for key in changed:
            params[key] = rng.choice(values_by_key[key])
        candidates.append(params)

    return _dedupe_params(candidates)[:count]


def methods_for_stage(
    branching_params: list[dict[str, Any]],
    sdp_params: list[dict[str, Any]],
    *,
    include_graph: bool = True,
) -> tuple[MethodSpec, ...]:
    methods: list[MethodSpec] = []
    if include_graph:
        methods.append(MethodSpec("graph_scaffold_base", "graph_shortest", dict(BASE_GRAPH_PARAMS)))
    for index, params in enumerate(branching_params):
        methods.append(MethodSpec(f"branching_{index:03d}", "visibility_branching", dict(params)))
    for index, params in enumerate(sdp_params):
        methods.append(MethodSpec(f"sdp_{index:03d}", "visibility_sdp", dict(params)))
    return tuple(methods)


def score_rows(rows: list[BenchmarkRow], *, fold_threshold_m: float) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[BenchmarkRow]] = {}
    for row in rows:
        groups.setdefault((row.method, row.solver), []).append(row)

    scored: list[dict[str, Any]] = []
    for (method, solver), part in groups.items():
        ok = [row for row in part if row.status == "ok" and math.isfinite(row.max_offset_m)]
        offsets = np.array([row.max_offset_m for row in ok], dtype=float)
        rmse = np.array([row.known_rmse_m for row in ok], dtype=float)
        if offsets.size:
            folded_rate = float(np.mean(offsets > fold_threshold_m))
            error_rate = 1.0 - len(ok) / max(len(part), 1)
            score = (
                float(np.quantile(offsets, 0.95))
                + 0.25 * float(np.median(offsets))
                + 0.10 * float(np.max(offsets))
                + 2.0 * folded_rate
                + 10.0 * error_rate
            )
            row = {
                "method": method,
                "solver": solver,
                "cases": len(part),
                "ok_cases": len(ok),
                "error_cases": len(part) - len(ok),
                "score": score,
                "median_max_offset_m": float(np.median(offsets)),
                "p90_max_offset_m": float(np.quantile(offsets, 0.90)),
                "p95_max_offset_m": float(np.quantile(offsets, 0.95)),
                "max_offset_m": float(np.max(offsets)),
                "folded_case_rate": folded_rate,
                "under_20cm": float(np.mean(offsets <= 0.2)),
                "under_50cm": float(np.mean(offsets <= 0.5)),
                "under_1m": float(np.mean(offsets <= 1.0)),
                "median_known_rmse_m": float(np.median(rmse)) if rmse.size else math.nan,
                "median_runtime_s": float(np.median([row.runtime_s for row in part])),
            }
        else:
            row = {
                "method": method,
                "solver": solver,
                "cases": len(part),
                "ok_cases": 0,
                "error_cases": len(part),
                "score": math.inf,
            }
        if ok:
            row.update({f"param_{key}": value for key, value in ok[0].params.items()})
        scored.append(row)
    return sorted(scored, key=lambda item: (float(item["score"]), str(item["method"])))


def best_params(rows: list[BenchmarkRow], solver: str, *, fold_threshold_m: float) -> dict[str, Any]:
    leaderboard = [row for row in score_rows(rows, fold_threshold_m=fold_threshold_m) if row["solver"] == solver]
    if not leaderboard:
        raise RuntimeError(f"No successful rows for solver {solver!r}.")
    best_method = str(leaderboard[0]["method"])
    for row in rows:
        if row.method == best_method and row.params:
            return dict(row.params)
    raise RuntimeError(f"Could not recover params for best method {best_method!r}.")


def write_selected(path: Path, *, stage1_branching: dict[str, Any], stage1_sdp: dict[str, Any], stage2_branching: dict[str, Any], stage2_sdp: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "stage1_best": {
                    "visibility_branching": stage1_branching,
                    "visibility_sdp": stage1_sdp,
                },
                "stage2_best": {
                    "visibility_branching": stage2_branching,
                    "visibility_sdp": stage2_sdp,
                },
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def run_stage(
    *,
    name: str,
    seed: int,
    cases_per_bucket: int,
    methods: tuple[MethodSpec, ...],
    workers: int,
    output_dir: Path,
    fold_threshold_m: float,
) -> tuple[list[BenchmarkRow], list[dict[str, Any]]]:
    config = BenchmarkConfig(
        name=name,
        seed=seed,
        layouts=layouts(cases_per_bucket),
        methods=methods,
        runner=RunnerSpec(
            backend="process" if workers > 1 else "serial",
            workers=workers,
            fold_threshold_m=fold_threshold_m,
            output_dir=output_dir,
            write_outputs=True,
        ),
        device="cpu",
    )
    print(
        f"running {name}: cases={cases_per_bucket * 3} methods={len(methods)} "
        f"jobs={cases_per_bucket * 3 * len(methods)} workers={workers}",
        flush=True,
    )
    rows, _summary = run_benchmark(config)
    leaderboard = score_rows(rows, fold_threshold_m=fold_threshold_m)
    write_csv(output_dir / name / "leaderboard.csv", leaderboard)
    return rows, leaderboard


def copy_result_snapshots(output_dir: Path, names: Iterable[str], docs_dir: Path) -> None:
    docs_dir.mkdir(parents=True, exist_ok=True)
    for name in names:
        run_dir = output_dir / name
        for file_name in ("summary.csv", "leaderboard.csv"):
            src = run_dir / file_name
            if src.exists():
                shutil.copy2(src, docs_dir / f"{name}_{file_name}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Two-stage visibility solver parameter sweep.")
    parser.add_argument("--seed", type=int, default=2026070220)
    parser.add_argument("--stage1-cases-per-bucket", type=int, default=15)
    parser.add_argument("--stage2-cases-per-bucket", type=int, default=15)
    parser.add_argument("--stage1-candidates", type=int, default=32)
    parser.add_argument("--stage2-candidates", type=int, default=32)
    parser.add_argument("--workers", type=int, default=max(1, min(12, (os.cpu_count() or 4) - 1)))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--fold-threshold-m", type=float, default=1.65)
    parser.add_argument("--copy-docs", action="store_true")
    return parser.parse_args()


def print_top(label: str, leaderboard: list[dict[str, Any]], solver: str, limit: int = 5) -> None:
    print(f"\n{label} top {solver}", flush=True)
    for row in [item for item in leaderboard if item["solver"] == solver][:limit]:
        print(
            f"{row['method']} score={float(row['score']):.3f} "
            f"p95={float(row.get('p95_max_offset_m', math.nan)):.3f} "
            f"median={float(row.get('median_max_offset_m', math.nan)):.3f} "
            f"max={float(row.get('max_offset_m', math.nan)):.3f} "
            f"folded={float(row.get('folded_case_rate', math.nan)):.3f}",
            flush=True,
        )


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()

    stage1_branching = randomized_discrete_sweep(
        DEFAULT_BRANCHING,
        BRANCHING_STAGE1_VALUES,
        count=args.stage1_candidates,
        seed=args.seed + 101,
    )
    stage1_sdp = randomized_discrete_sweep(
        DEFAULT_SDP,
        SDP_STAGE1_VALUES,
        count=args.stage1_candidates,
        seed=args.seed + 202,
    )
    stage1_rows, stage1_leaderboard = run_stage(
        name="visibility_stage1_high_sample_sweep",
        seed=args.seed,
        cases_per_bucket=args.stage1_cases_per_bucket,
        methods=methods_for_stage(stage1_branching, stage1_sdp),
        workers=args.workers,
        output_dir=output_dir,
        fold_threshold_m=args.fold_threshold_m,
    )
    print_top("stage 1", stage1_leaderboard, "visibility_branching")
    print_top("stage 1", stage1_leaderboard, "visibility_sdp")

    best_branching_stage1 = best_params(stage1_rows, "visibility_branching", fold_threshold_m=args.fold_threshold_m)
    best_sdp_stage1 = best_params(stage1_rows, "visibility_sdp", fold_threshold_m=args.fold_threshold_m)

    stage2_branching_values = local_values_around_best(best_branching_stage1, BRANCHING_STAGE1_VALUES)
    stage2_sdp_values = local_values_around_best(best_sdp_stage1, SDP_STAGE1_VALUES)
    stage2_branching = local_sweep(
        best_branching_stage1,
        stage2_branching_values,
        count=args.stage2_candidates,
        seed=args.seed + 303,
    )
    stage2_sdp = local_sweep(
        best_sdp_stage1,
        stage2_sdp_values,
        count=args.stage2_candidates,
        seed=args.seed + 404,
    )
    stage2_rows, stage2_leaderboard = run_stage(
        name="visibility_stage2_local_refinement",
        seed=args.seed + 1,
        cases_per_bucket=args.stage2_cases_per_bucket,
        methods=methods_for_stage(stage2_branching, stage2_sdp),
        workers=args.workers,
        output_dir=output_dir,
        fold_threshold_m=args.fold_threshold_m,
    )
    print_top("stage 2", stage2_leaderboard, "visibility_branching")
    print_top("stage 2", stage2_leaderboard, "visibility_sdp")

    best_branching_stage2 = best_params(stage2_rows, "visibility_branching", fold_threshold_m=args.fold_threshold_m)
    best_sdp_stage2 = best_params(stage2_rows, "visibility_sdp", fold_threshold_m=args.fold_threshold_m)
    write_selected(
        output_dir / "visibility_two_stage_selected_params.json",
        stage1_branching=best_branching_stage1,
        stage1_sdp=best_sdp_stage1,
        stage2_branching=best_branching_stage2,
        stage2_sdp=best_sdp_stage2,
    )
    print("\nselected stage 2 params", flush=True)
    print(json.dumps({"visibility_branching": best_branching_stage2, "visibility_sdp": best_sdp_stage2}, indent=2, sort_keys=True), flush=True)

    if args.copy_docs:
        copy_result_snapshots(
            output_dir,
            ("visibility_stage1_high_sample_sweep", "visibility_stage2_local_refinement"),
            Path("docs") / "results",
        )


if __name__ == "__main__":
    main()
