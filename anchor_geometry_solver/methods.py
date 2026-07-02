from __future__ import annotations

import math
import os
import random
import time
from typing import Any

from anchor_geometry_solver.compat import ensure_legacy_paths
from anchor_geometry_solver.types import BenchmarkRow, CaseContext, MethodSpec

ensure_legacy_paths()
import anchor_solver_fold_rescue_experiment as exp  # noqa: E402
import anchor_solver_ml_distance_completion as dc  # noqa: E402


os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")


def _int_param(params: dict[str, Any], key: str, default: int) -> int:
    return int(params.get(key, default))


def _float_param(params: dict[str, Any], key: str, default: float) -> float:
    return float(params.get(key, default))


def _solve_positions(context: CaseContext, method: MethodSpec, *, rng_seed: int) -> dict[str, tuple[float, float]]:
    solver = method.solver.replace("_", "-").lower()
    params = method.params
    known_pairs = list(context.known_pairs)
    if solver in {"production", "default", "default-production", "production-priors"}:
        return exp.production_solve(
            known_pairs,
            seed_count=_int_param(params, "seed_count", 6),
            iterations=_int_param(params, "iterations", 45),
            rng_seed=rng_seed,
        )
    if solver in {"graph-shortest", "graph-shortest-scaffold", "graph"}:
        return exp.graph_shortest_scaffold_solve(
            known_pairs,
            seed_count=_int_param(params, "seed_count", 6),
            iterations=_int_param(params, "iterations", 45),
            rng_seed=rng_seed,
            max_hops=_int_param(params, "max_hops", 2),
            relative_sigma=_float_param(params, "relative_sigma", 0.30),
            scaffold_weight=_float_param(params, "scaffold_weight", 1.0),
            hop_weight_base=_float_param(params, "hop_weight_base", 1.0),
        )
    if solver in {"distance-only", "known-only"}:
        return exp.best_distance_only_solve(
            known_pairs,
            seed_count=_int_param(params, "seed_count", 6),
            iterations=_int_param(params, "iterations", 45),
            rng=random.Random(rng_seed),
        )
    if solver in {"triangulated", "known-triangulated"}:
        return dc.known_only_solution(
            known_pairs,
            max_iterations=_int_param(params, "iterations", 45),
        )
    raise ValueError(f"Unknown solver {method.solver!r}")


def solve_context(
    benchmark_name: str,
    context: CaseContext,
    method: MethodSpec,
    *,
    fold_threshold_m: float,
    rng_seed: int,
) -> BenchmarkRow:
    started = time.perf_counter()
    try:
        positions = _solve_positions(context, method, rng_seed=rng_seed)
        result = exp.metrics_row(
            context.bucket,
            context.case_index,
            context.family,
            context.shape,
            method.name,
            context.truth,
            positions,
            list(context.known_pairs),
            fold_threshold_m=fold_threshold_m,
        )
        return BenchmarkRow(
            benchmark=benchmark_name,
            bucket=result.bucket,
            case_index=result.case_index,
            family=result.family,
            shape=result.shape,
            method=method.name,
            solver=method.solver,
            anchors=result.anchors,
            known_pairs=result.known_pairs,
            max_offset_m=result.max_offset_m,
            median_offset_m=result.median_offset_m,
            p95_offset_m=result.p95_offset_m,
            known_rmse_m=result.known_rmse_m,
            known_max_residual_m=result.known_max_residual_m,
            min_pair_distance_m=result.min_pair_distance_m,
            close_pair_count=result.close_pair_count,
            fold_cluster_count=result.fold_cluster_count,
            runtime_s=time.perf_counter() - started,
            status="ok",
            params=dict(method.params),
            diagnostics=dict(context.diagnostics),
        )
    except Exception as exc:
        return BenchmarkRow(
            benchmark=benchmark_name,
            bucket=context.bucket,
            case_index=context.case_index,
            family=context.family,
            shape=context.shape,
            method=method.name,
            solver=method.solver,
            anchors=context.anchor_count,
            known_pairs=context.known_pair_count,
            max_offset_m=math.inf,
            median_offset_m=math.inf,
            p95_offset_m=math.inf,
            known_rmse_m=math.inf,
            known_max_residual_m=math.inf,
            min_pair_distance_m=math.inf,
            close_pair_count=0,
            fold_cluster_count=0,
            runtime_s=time.perf_counter() - started,
            status="error",
            error=f"{type(exc).__name__}: {exc}",
            params=dict(method.params),
            diagnostics=dict(context.diagnostics),
        )
