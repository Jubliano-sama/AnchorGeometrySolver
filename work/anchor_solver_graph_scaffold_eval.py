from __future__ import annotations

import argparse
import csv
from pathlib import Path
import random
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
sys.path.insert(0, str(ROOT / "work"))

import anchor_solver_fold_rescue_experiment as exp  # noqa: E402
import anchor_solver_p95_ml_cases as p95  # noqa: E402
import anchor_solver_ml_distance_completion as dc  # noqa: E402

TOKENS = dc.TOKENS
BLUE = dc.BLUE
PINK = dc.PINK
OLIVE = dc.OLIVE
NEUTRAL = dc.NEUTRAL
BUCKET_LABELS = exp.BUCKET_LABELS


def parse_buckets(text: str) -> list[str]:
    buckets = [part.strip().lower() for part in text.split(",") if part.strip()]
    unknown = [bucket for bucket in buckets if bucket not in BUCKET_LABELS]
    if unknown:
        raise ValueError(f"Unknown buckets: {unknown}")
    return buckets


def generate_limited_cases(bucket_key: str, count: int, device: torch.device, *, max_nodes: int | None) -> list[p95.CaseSpec]:
    cases: list[p95.CaseSpec] = []
    attempts = 0
    max_attempts = max(200, count * 80)
    while len(cases) < count and attempts < max_attempts:
        attempts += 1
        case = p95.generate_cases(bucket_key, 1, device)[0]
        if max_nodes is not None and int(case.points.shape[0]) > max_nodes:
            continue
        cases.append(case)
    if len(cases) < count:
        raise RuntimeError(f"Only generated {len(cases)}/{count} cases for bucket={bucket_key} with max_nodes={max_nodes}.")
    return cases


def result_dict(row: exp.MethodResult) -> dict[str, str | int | float]:
    return exp.result_to_dict(row)


def summarize(rows: list[dict[str, str | int | float]]) -> list[dict[str, str | int | float]]:
    summary: list[dict[str, str | int | float]] = []
    for bucket, method in sorted({(str(row["bucket"]), str(row["method"])) for row in rows}):
        part = [row for row in rows if row["bucket"] == bucket and row["method"] == method]
        offsets = np.array([float(row["max_offset_m"]) for row in part], dtype=float)
        rmse = np.array([float(row["known_rmse_m"]) for row in part], dtype=float)
        close = np.array([float(row["close_pair_count"]) for row in part], dtype=float)
        summary.append(
            {
                "bucket": bucket,
                "method": method,
                "cases": len(part),
                "median_max_offset_m": float(np.median(offsets)),
                "p90_max_offset_m": float(np.quantile(offsets, 0.90)),
                "p95_max_offset_m": float(np.quantile(offsets, 0.95)),
                "under_20cm": float(np.mean(offsets <= 0.20)),
                "under_50cm": float(np.mean(offsets <= 0.50)),
                "under_1m": float(np.mean(offsets <= 1.00)),
                "median_known_rmse_m": float(np.median(rmse)),
                "folded_case_rate": float(np.mean(close > 0)),
            }
        )
    return summary


def write_csv(path: Path, rows: list[dict[str, str | int | float]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def make_figure(path: Path, summary: list[dict[str, str | int | float]], *, cases_per_bucket: int) -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": TOKENS["surface"],
            "savefig.facecolor": TOKENS["surface"],
            "font.family": ["Segoe UI", "DejaVu Sans", "Arial", "sans-serif"],
        }
    )
    buckets = [bucket for bucket in BUCKET_LABELS.values() if any(row["bucket"] == bucket for row in summary)]
    methods = ["production-priors", "graph-shortest-scaffold"]
    colors = {"production-priors": BLUE["base"], "graph-shortest-scaffold": "#2C9C8F"}
    fig, axes = plt.subplots(1, 3, figsize=(15.8, 5.6), dpi=170)
    fig.text(0.035, 0.98, "Graph-shortest scaffold eval", ha="left", va="top", fontsize=18, fontweight="bold", color=TOKENS["ink"])
    fig.text(0.035, 0.925, f"{cases_per_bucket} fair generated cases per bucket. Temporary graph shortest-path springs are removed before measured-range polish.", ha="left", va="top", fontsize=9.0, color=TOKENS["muted"])
    metric_specs = [("p95_max_offset_m", "p95 max offset (m)"), ("under_1m", "share under 1 m"), ("median_known_rmse_m", "median known RMSE (m)")]
    for ax, (metric, title) in zip(axes, metric_specs):
        labels: list[str] = []
        values: list[float] = []
        bar_colors: list[str] = []
        for bucket in buckets:
            for method in methods:
                row = next((item for item in summary if item["bucket"] == bucket and item["method"] == method), None)
                if row is None:
                    continue
                labels.append(f"{bucket.split()[0]}\n{method.replace('-', ' ')}")
                values.append(float(row[metric]))
                bar_colors.append(colors[method])
        x = np.arange(len(values))
        ax.set_facecolor(TOKENS["panel"])
        ax.bar(x, values, color=bar_colors, edgecolor=TOKENS["ink"], linewidth=0.35)
        ax.set_title(title, loc="left", fontsize=10.5, fontweight="bold", color=TOKENS["ink"])
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=6.8)
        ax.tick_params(axis="y", labelsize=7.5, colors=TOKENS["muted"], length=0)
        ax.grid(True, axis="y", color=TOKENS["grid"], linewidth=0.5)
        for spine in ax.spines.values():
            spine.set_color(TOKENS["axis"])
    fig.subplots_adjust(left=0.055, right=0.985, top=0.82, bottom=0.24, wspace=0.22)
    fig.savefig(path, bbox_inches="tight", dpi=170)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate production vs graph-shortest scaffold on generated fair layouts.")
    parser.add_argument("--buckets", default="random,grid,office")
    parser.add_argument("--cases-per-bucket", type=int, default=6)
    parser.add_argument("--seed", type=int, default=2026062707)
    parser.add_argument("--seed-count", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=60)
    parser.add_argument("--max-nodes", type=int, default=0, help="Optional cap on generated anchor count; 0 disables the cap.")
    parser.add_argument("--max-hops", type=int, default=2)
    parser.add_argument("--relative-sigma", type=float, default=0.36)
    parser.add_argument("--scaffold-weight", type=float, default=1.0)
    parser.add_argument("--hop-weight-base", type=float, default=1.0)
    parser.add_argument("--fold-threshold", type=float, default=1.65)
    parser.add_argument("--prefix", default="anchor_solver_graph_scaffold_eval")
    args = parser.parse_args()
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed % (2**32 - 1))
    torch.manual_seed(args.seed)
    device = torch.device("cpu")
    detail: list[dict[str, str | int | float]] = []
    case_counter = 0
    max_nodes = args.max_nodes if args.max_nodes > 0 else None
    for bucket_key in parse_buckets(args.buckets):
        cap_text = f" max_nodes={max_nodes}" if max_nodes is not None else ""
        print(f"generating bucket={bucket_key} cases={args.cases_per_bucket}{cap_text}", flush=True)
        cases = generate_limited_cases(bucket_key, args.cases_per_bucket, device, max_nodes=max_nodes)
        for local_index, case in enumerate(cases):
            batch = p95.graph_batch_from_cases([case], device)
            truth = dc.graph_to_truth(batch, 0)
            known_pairs = dc.known_pairs_from_batch(batch, 0)
            print(f"solving bucket={bucket_key} case={local_index + 1}/{len(cases)} anchors={len(truth)}", flush=True)
            production = exp.production_solve(known_pairs, seed_count=args.seed_count, iterations=args.iterations, rng_seed=args.seed + case_counter * 101 + 17)
            graph = exp.graph_shortest_scaffold_solve(
                known_pairs,
                seed_count=args.seed_count,
                iterations=args.iterations,
                rng_seed=args.seed + case_counter * 1009 + 19,
                max_hops=args.max_hops,
                relative_sigma=args.relative_sigma,
                scaffold_weight=args.scaffold_weight,
                hop_weight_base=args.hop_weight_base,
            )
            for method, positions in [("production-priors", production), ("graph-shortest-scaffold", graph)]:
                row = exp.metrics_row(case.bucket, case_counter, case.family, case.shape, method, truth, positions, known_pairs, fold_threshold_m=args.fold_threshold)
                detail.append(result_dict(row))
            case_counter += 1
    summary = summarize(detail)
    detail_path = OUTPUTS / f"{args.prefix}_detail.csv"
    summary_path = OUTPUTS / f"{args.prefix}_summary.csv"
    figure_path = OUTPUTS / f"{args.prefix}.png"
    write_csv(detail_path, detail)
    write_csv(summary_path, summary)
    make_figure(figure_path, summary, cases_per_bucket=args.cases_per_bucket)
    for row in summary:
        print(
            f"summary bucket={row['bucket']} method={row['method']} p95={float(row['p95_max_offset_m']):.3f}m "
            f"median={float(row['median_max_offset_m']):.3f}m under1m={float(row['under_1m']):.3f} "
            f"folded={float(row['folded_case_rate']):.3f} rmse={float(row['median_known_rmse_m']):.4f}m",
            flush=True,
        )
    print(f"Wrote {detail_path}", flush=True)
    print(f"Wrote {summary_path}", flush=True)
    print(f"Wrote {figure_path}", flush=True)


if __name__ == "__main__":
    main()

