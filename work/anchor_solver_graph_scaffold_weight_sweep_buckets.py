from __future__ import annotations

import argparse
import csv
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
import os
import random
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
sys.path.insert(0, str(ROOT / "work"))

import anchor_solver_fold_rescue_experiment as exp
import anchor_solver_ml_distance_completion as dc
import anchor_solver_p95_ml_cases as p95

TOKENS = dc.TOKENS
BLUE = dc.BLUE
GOLD = dc.GOLD
OLIVE = dc.OLIVE


def parse_ints(text: str) -> list[int]:
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def parse_floats(text: str) -> list[float]:
    return [float(part.strip()) for part in text.split(",") if part.strip()]


def case_context(case: p95.CaseSpec, case_index: int, device: torch.device) -> dict[str, object]:
    batch = p95.graph_batch_from_cases([case], device)
    truth = dc.graph_to_truth(batch, 0)
    known_pairs = dc.known_pairs_from_batch(batch, 0)
    diag = dc.measured_graph_diagnostics(case.points)
    return {
        "bucket": case.bucket,
        "family": case.family,
        "shape": case.shape,
        "case_index": case_index,
        "truth": truth,
        "known_pairs": known_pairs,
        "diag": diag,
    }


def solve_job(job: dict[str, object]) -> dict[str, str | int | float]:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    ctx = job["context"]  # type: ignore[assignment]
    assert isinstance(ctx, dict)
    known_pairs = ctx["known_pairs"]
    truth = ctx["truth"]
    method = str(job["method"])
    if method == "production-priors":
        positions = exp.production_solve(
            known_pairs,  # type: ignore[arg-type]
            seed_count=int(job["seed_count"]),
            iterations=int(job["iterations"]),
            rng_seed=int(job["rng_seed"]),
        )
    else:
        positions = exp.graph_shortest_scaffold_solve(
            known_pairs,  # type: ignore[arg-type]
            seed_count=int(job["seed_count"]),
            iterations=int(job["iterations"]),
            rng_seed=int(job["rng_seed"]),
            max_hops=int(job["max_hops"]),
            relative_sigma=float(job["relative_sigma"]),
            scaffold_weight=float(job["scaffold_weight"]),
            hop_weight_base=float(job["hop_weight_base"]),
        )
    row = exp.result_to_dict(
        exp.metrics_row(
            str(ctx["bucket"]),
            int(ctx["case_index"]),
            str(ctx["family"]),
            str(ctx["shape"]),
            method,
            truth,  # type: ignore[arg-type]
            positions,
            known_pairs,  # type: ignore[arg-type]
            fold_threshold_m=float(job["fold_threshold_m"]),
        )
    )
    diag = ctx["diag"]
    assert isinstance(diag, dict)
    row.update(
        {
            "max_hops": job["max_hops"],
            "relative_sigma": job["relative_sigma"],
            "scaffold_weight": job["scaffold_weight"],
            "hop_weight_base": job["hop_weight_base"],
            "min_degree": diag["min_degree"],
            "mean_degree": diag["mean_degree"],
            "rigidity_surplus": diag["rigidity_surplus"],
            "vertex_connectivity_capped3": diag["vertex_connectivity_capped3"],
        }
    )
    return row


def summarize(rows: list[dict[str, str | int | float]]) -> list[dict[str, str | int | float]]:
    summary: list[dict[str, str | int | float]] = []
    keys = sorted(
        {
            (
                str(row["bucket"]),
                str(row["method"]),
                str(row["max_hops"]),
                str(row["relative_sigma"]),
                str(row["scaffold_weight"]),
                str(row["hop_weight_base"]),
            )
            for row in rows
        }
    )
    for bucket, method, max_hops, relative_sigma, scaffold_weight, hop_weight_base in keys:
        part = [
            row
            for row in rows
            if str(row["bucket"]) == bucket
            and str(row["method"]) == method
            and str(row["max_hops"]) == max_hops
            and str(row["relative_sigma"]) == relative_sigma
            and str(row["scaffold_weight"]) == scaffold_weight
            and str(row["hop_weight_base"]) == hop_weight_base
        ]
        offsets = np.array([float(row["max_offset_m"]) for row in part], dtype=float)
        rmses = np.array([float(row["known_rmse_m"]) for row in part], dtype=float)
        close = np.array([float(row["close_pair_count"]) for row in part], dtype=float)
        vc = np.array([float(row["vertex_connectivity_capped3"]) for row in part], dtype=float)
        summary.append(
            {
                "bucket": bucket,
                "method": method,
                "max_hops": max_hops,
                "relative_sigma": relative_sigma,
                "scaffold_weight": scaffold_weight,
                "hop_weight_base": hop_weight_base,
                "cases": len(part),
                "median_max_offset_m": float(np.median(offsets)),
                "p90_max_offset_m": float(np.quantile(offsets, 0.90)),
                "p95_max_offset_m": float(np.quantile(offsets, 0.95)),
                "max_offset_m": float(np.max(offsets)),
                "under_20cm": float(np.mean(offsets <= 0.20)),
                "under_50cm": float(np.mean(offsets <= 0.50)),
                "under_1m": float(np.mean(offsets <= 1.00)),
                "under_2m": float(np.mean(offsets <= 2.00)),
                "median_known_rmse_m": float(np.median(rmses)),
                "p95_known_rmse_m": float(np.quantile(rmses, 0.95)),
                "folded_case_rate": float(np.mean(close > 0)),
                "mean_close_pair_count": float(np.mean(close)),
                "min_vertex_connectivity_capped3": float(np.min(vc)),
            }
        )
    return summary


def write_csv(path: Path, rows: list[dict[str, str | int | float]]) -> None:
    if not rows:
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fields)
        writer.writeheader()
        writer.writerows(rows)


def make_figure(path: Path, summary: list[dict[str, str | int | float]], *, cases: int) -> None:
    graph_rows = [row for row in summary if row["method"] == "graph-shortest-scaffold"]
    top = sorted(graph_rows, key=lambda row: float(row["p95_max_offset_m"]))[: min(14, len(graph_rows))]
    if not top:
        return
    plt.rcParams.update(
        {
            "figure.facecolor": TOKENS["surface"],
            "savefig.facecolor": TOKENS["surface"],
            "font.family": ["Segoe UI", "DejaVu Sans", "Arial", "sans-serif"],
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(18.6, 6.2), dpi=170)
    fig.text(0.035, 0.98, "Graph-shortest scaffold parameter sweep", ha="left", va="top", fontsize=18, fontweight="bold", color=TOKENS["ink"])
    fig.text(0.035, 0.925, f"{cases} fixed generated cases per bucket. Office cases require capped vertex-connectivity 3; graph springs are removed before measured-range polish.", ha="left", va="top", fontsize=9, color=TOKENS["muted"])
    labels = [
        f"{row['bucket']}\nh{row['max_hops']} sw={float(row['scaffold_weight']):.2g} hb={float(row['hop_weight_base']):.2g}\nrs={float(row['relative_sigma']):.2f}"
        for row in top
    ]
    specs = [
        ("p95_max_offset_m", "p95 max offset (m)", BLUE["base"]),
        ("under_1m", "share under 1 m", OLIVE["base"]),
        ("median_known_rmse_m", "median known RMSE (m)", GOLD["base"]),
    ]
    for ax, (metric, title, color) in zip(axes, specs):
        values = [float(row[metric]) for row in top]
        x = np.arange(len(values))
        ax.set_facecolor(TOKENS["panel"])
        ax.bar(x, values, color=color, edgecolor=TOKENS["ink"], linewidth=0.35)
        ax.set_title(title, loc="left", fontsize=10.5, fontweight="bold", color=TOKENS["ink"])
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=0, fontsize=6.2)
        ax.tick_params(axis="y", labelsize=7.5, colors=TOKENS["muted"], length=0)
        ax.grid(True, axis="y", color=TOKENS["grid"], linewidth=0.5)
        for spine in ax.spines.values():
            spine.set_color(TOKENS["axis"])
    fig.subplots_adjust(left=0.045, right=0.985, top=0.82, bottom=0.28, wspace=0.20)
    fig.savefig(path, bbox_inches="tight", dpi=170)
    plt.close(fig)


def run_jobs(jobs: list[dict[str, object]], *, backend: str, workers: int, started: float) -> list[dict[str, str | int | float]]:
    detail: list[dict[str, str | int | float]] = []
    if backend == "serial" or workers <= 1:
        for index, job in enumerate(jobs, 1):
            detail.append(solve_job(job))
            if index == 1 or index % 10 == 0 or index == len(jobs):
                print(f"progress {index}/{len(jobs)} elapsed_s={time.perf_counter() - started:.1f}", flush=True)
        return detail

    executor_cls = ThreadPoolExecutor if backend == "thread" else ProcessPoolExecutor
    with executor_cls(max_workers=workers) as executor:
        futures = [executor.submit(solve_job, job) for job in jobs]
        for index, future in enumerate(as_completed(futures), 1):
            detail.append(future.result())
            if index == 1 or index % 10 == 0 or index == len(jobs):
                print(f"progress {index}/{len(jobs)} elapsed_s={time.perf_counter() - started:.1f}", flush=True)
    return detail


def main() -> None:
    parser = argparse.ArgumentParser(description="Bucketed graph-shortest scaffold parameter sweep.")
    parser.add_argument("--buckets", default="random,grid,office")
    parser.add_argument("--cases", type=int, default=2)
    parser.add_argument("--seed", type=int, default=2026062745)
    parser.add_argument("--seed-count", type=int, default=4)
    parser.add_argument("--iterations", type=int, default=35)
    parser.add_argument("--max-hops", default="2,3")
    parser.add_argument("--relative-sigma", default="0.30,0.42")
    parser.add_argument("--scaffold-weight", default="0.5,1.0,2.0")
    parser.add_argument("--hop-weight-base", default="0.7,1.0,1.4")
    parser.add_argument("--backend", choices=("thread", "process", "serial"), default="thread")
    parser.add_argument("--workers", type=int, default=max(1, min(8, (os.cpu_count() or 4) - 1)))
    parser.add_argument("--fold-threshold", type=float, default=1.65)
    parser.add_argument("--prefix", default="anchor_solver_graph_scaffold_weight_sweep_buckets")
    args = parser.parse_args()

    OUTPUTS.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed % (2**32 - 1))
    torch.manual_seed(args.seed)
    device = torch.device("cpu")

    jobs: list[dict[str, object]] = []
    global_case_index = 0
    combos = [
        (max_hops, relative_sigma, scaffold_weight, hop_weight_base)
        for max_hops in parse_ints(args.max_hops)
        for relative_sigma in parse_floats(args.relative_sigma)
        for scaffold_weight in parse_floats(args.scaffold_weight)
        for hop_weight_base in parse_floats(args.hop_weight_base)
    ]
    for bucket in [part.strip() for part in args.buckets.split(",") if part.strip()]:
        print(f"generating bucket={bucket} cases={args.cases}", flush=True)
        cases = p95.generate_cases(bucket, args.cases, device)
        for local_index, case in enumerate(cases):
            case_index = global_case_index + local_index
            ctx = case_context(case, case_index, device)
            jobs.append(
                {
                    "context": ctx,
                    "method": "production-priors",
                    "seed_count": args.seed_count,
                    "iterations": args.iterations,
                    "rng_seed": args.seed + case_index * 101 + 17,
                    "max_hops": "production",
                    "relative_sigma": "production",
                    "scaffold_weight": "production",
                    "hop_weight_base": "production",
                    "fold_threshold_m": args.fold_threshold,
                }
            )
            for max_hops, relative_sigma, scaffold_weight, hop_weight_base in combos:
                jobs.append(
                    {
                        "context": ctx,
                        "method": "graph-shortest-scaffold",
                        "seed_count": args.seed_count,
                        "iterations": args.iterations,
                        "rng_seed": args.seed + case_index * 1009 + max_hops * 100 + int(relative_sigma * 1000) + int(scaffold_weight * 10000) + int(hop_weight_base * 100000),
                        "max_hops": max_hops,
                        "relative_sigma": relative_sigma,
                        "scaffold_weight": scaffold_weight,
                        "hop_weight_base": hop_weight_base,
                        "fold_threshold_m": args.fold_threshold,
                    }
                )
        global_case_index += len(cases)

    print(f"jobs={len(jobs)} combos={len(combos)} backend={args.backend} workers={args.workers}", flush=True)
    started = time.perf_counter()
    try:
        detail = run_jobs(jobs, backend=args.backend, workers=args.workers, started=started)
    except PermissionError as exc:
        if args.backend != "process":
            raise
        print(f"process_backend_failed={exc}; retrying backend=thread", flush=True)
        detail = run_jobs(jobs, backend="thread", workers=args.workers, started=started)

    summary = summarize(detail)
    detail_path = OUTPUTS / f"{args.prefix}_detail.csv"
    summary_path = OUTPUTS / f"{args.prefix}_summary.csv"
    figure_path = OUTPUTS / f"{args.prefix}.png"
    write_csv(detail_path, detail)
    write_csv(summary_path, summary)
    make_figure(figure_path, summary, cases=args.cases)

    for row in sorted([r for r in summary if r["method"] == "graph-shortest-scaffold"], key=lambda item: float(item["p95_max_offset_m"]))[:18]:
        print(
            f"summary bucket={row['bucket']} h={row['max_hops']} rs={row['relative_sigma']} "
            f"sw={row['scaffold_weight']} hb={row['hop_weight_base']} p95={float(row['p95_max_offset_m']):.3f}m "
            f"median={float(row['median_max_offset_m']):.3f}m under1={float(row['under_1m']):.3f} "
            f"folded={float(row['folded_case_rate']):.3f} rmse={float(row['median_known_rmse_m']):.4f}m",
            flush=True,
        )
    print(f"elapsed_s={time.perf_counter() - started:.1f}", flush=True)
    print(f"Wrote {detail_path}", flush=True)
    print(f"Wrote {summary_path}", flush=True)
    print(f"Wrote {figure_path}", flush=True)


if __name__ == "__main__":
    main()