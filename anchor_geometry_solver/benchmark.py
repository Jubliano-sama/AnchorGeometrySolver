from __future__ import annotations

import csv
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from anchor_geometry_solver.layouts import generate_case_contexts
from anchor_geometry_solver.methods import solve_context
from anchor_geometry_solver.types import BenchmarkConfig, BenchmarkRow, CaseContext, MethodSpec


def _job(job: tuple[str, CaseContext, MethodSpec, float, int]) -> BenchmarkRow:
    benchmark_name, context, method, fold_threshold_m, rng_seed = job
    return solve_context(
        benchmark_name,
        context,
        method,
        fold_threshold_m=fold_threshold_m,
        rng_seed=rng_seed,
    )


def make_jobs(config: BenchmarkConfig, contexts: list[CaseContext]) -> list[tuple[str, CaseContext, MethodSpec, float, int]]:
    jobs: list[tuple[str, CaseContext, MethodSpec, float, int]] = []
    for context in contexts:
        for method_index, method in enumerate(config.methods):
            seed = config.seed + context.case_index * 1009 + method_index * 9173 + 17
            jobs.append((config.name, context, method, config.runner.fold_threshold_m, seed))
    return jobs


def _run_serial(jobs: list[tuple[str, CaseContext, MethodSpec, float, int]]) -> list[BenchmarkRow]:
    return [_job(job) for job in jobs]


def run_jobs(
    jobs: list[tuple[str, CaseContext, MethodSpec, float, int]],
    *,
    backend: str,
    workers: int,
    progress_every: int = 10,
) -> list[BenchmarkRow]:
    backend = backend.lower()
    if backend == "serial" or workers <= 1:
        return _run_serial(jobs)
    executor_cls = ThreadPoolExecutor if backend == "thread" else ProcessPoolExecutor
    rows: list[BenchmarkRow] = []
    started = time.perf_counter()
    with executor_cls(max_workers=workers) as executor:
        futures = [executor.submit(_job, job) for job in jobs]
        for index, future in enumerate(as_completed(futures), 1):
            rows.append(future.result())
            if index == 1 or index % progress_every == 0 or index == len(jobs):
                print(f"progress solved={index}/{len(jobs)} elapsed_s={time.perf_counter() - started:.1f}", flush=True)
    return rows


def summarize_rows(rows: list[BenchmarkRow], *, thresholds_m: tuple[float, ...] = (0.05, 0.10, 0.20, 0.50, 1.0, 2.0)) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    keys = sorted({(row.bucket, row.method, row.solver) for row in rows})
    for bucket, method, solver in keys:
        part = [row for row in rows if row.bucket == bucket and row.method == method and row.solver == solver]
        ok = [row for row in part if row.status == "ok" and math.isfinite(row.max_offset_m)]
        offsets = np.array([row.max_offset_m for row in ok], dtype=float)
        rmse = np.array([row.known_rmse_m for row in ok], dtype=float)
        runtimes = np.array([row.runtime_s for row in part], dtype=float)
        out: dict[str, Any] = {
            "bucket": bucket,
            "method": method,
            "solver": solver,
            "cases": len(part),
            "ok_cases": len(ok),
            "error_cases": len(part) - len(ok),
            "median_runtime_s": float(np.median(runtimes)) if runtimes.size else math.nan,
        }
        if offsets.size:
            out.update(
                {
                    "median_max_offset_m": float(np.median(offsets)),
                    "p90_max_offset_m": float(np.quantile(offsets, 0.90)),
                    "p95_max_offset_m": float(np.quantile(offsets, 0.95)),
                    "max_offset_m": float(np.max(offsets)),
                    "median_known_rmse_m": float(np.median(rmse)) if rmse.size else math.nan,
                    "p95_known_rmse_m": float(np.quantile(rmse, 0.95)) if rmse.size else math.nan,
                }
            )
            for threshold in thresholds_m:
                label = f"under_{int(round(threshold * 100))}cm" if threshold < 1.0 else f"under_{threshold:g}m"
                out[label] = float(np.mean(offsets <= threshold))
        summary.append(out)
    return summary


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run_benchmark(config: BenchmarkConfig) -> tuple[list[BenchmarkRow], list[dict[str, Any]]]:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    contexts = generate_case_contexts(config.layouts, seed=config.seed, device_name=config.device)
    jobs = make_jobs(config, contexts)
    print(
        f"benchmark={config.name} cases={len(contexts)} methods={len(config.methods)} jobs={len(jobs)} "
        f"backend={config.runner.backend} workers={config.runner.workers}",
        flush=True,
    )
    rows = run_jobs(jobs, backend=config.runner.backend, workers=config.runner.workers)
    rows = sorted(rows, key=lambda row: (row.bucket, row.case_index, row.method))
    summary = summarize_rows(rows)
    if config.runner.write_outputs:
        output_dir = config.runner.output_dir / config.name
        write_csv(output_dir / "detail.csv", [row.to_flat_dict() for row in rows])
        write_csv(output_dir / "summary.csv", summary)
    return rows, summary


def with_output_dir(config: BenchmarkConfig, output_dir: Path) -> BenchmarkConfig:
    return replace(config, runner=replace(config.runner, output_dir=output_dir))
