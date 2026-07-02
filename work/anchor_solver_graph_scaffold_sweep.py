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
GOLD = dc.GOLD
PINK = dc.PINK
OLIVE = dc.OLIVE
NEUTRAL = dc.NEUTRAL


def parse_ints(text: str) -> list[int]:
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def parse_floats(text: str) -> list[float]:
    return [float(part.strip()) for part in text.split(",") if part.strip()]


def row_dict(row: exp.MethodResult, *, max_hops: int | str, relative_sigma: float | str, scaffold_weight: float | str, hop_weight_base: float | str) -> dict[str, str | int | float]:
    data = exp.result_to_dict(row)
    data["max_hops"] = max_hops
    data["relative_sigma"] = relative_sigma
    data["scaffold_weight"] = scaffold_weight
    data["hop_weight_base"] = hop_weight_base
    return data


def summarize(rows: list[dict[str, str | int | float]]) -> list[dict[str, str | int | float]]:
    summary: list[dict[str, str | int | float]] = []
    keys = sorted({(str(row["method"]), str(row["max_hops"]), str(row["relative_sigma"]), str(row["scaffold_weight"]), str(row["hop_weight_base"])) for row in rows})
    for method, max_hops, relative_sigma, scaffold_weight, hop_weight_base in keys:
        part = [
            row for row in rows
            if row["method"] == method
            and str(row["max_hops"]) == max_hops
            and str(row["relative_sigma"]) == relative_sigma
            and str(row["scaffold_weight"]) == scaffold_weight
            and str(row["hop_weight_base"]) == hop_weight_base
        ]
        offsets = np.array([float(row["max_offset_m"]) for row in part], dtype=float)
        rmses = np.array([float(row["known_rmse_m"]) for row in part], dtype=float)
        close = np.array([float(row["close_pair_count"]) for row in part], dtype=float)
        summary.append(
            {
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
                "under_50cm": float(np.mean(offsets <= 0.50)),
                "under_1m": float(np.mean(offsets <= 1.00)),
                "median_known_rmse_m": float(np.median(rmses)),
                "folded_case_rate": float(np.mean(close > 0)),
            }
        )
    return summary


def write_csv(path: Path, rows: list[dict[str, str | int | float]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def make_figure(path: Path, summary: list[dict[str, str | int | float]], *, cases: int) -> None:
    graph_rows = [row for row in summary if row["method"] == "graph-shortest-scaffold"]
    graph_rows = sorted(graph_rows, key=lambda row: float(row["p95_max_offset_m"]))
    baseline = next((row for row in summary if row["method"] == "production-priors"), None)
    top = graph_rows[: min(12, len(graph_rows))]
    plt.rcParams.update(
        {
            "figure.facecolor": TOKENS["surface"],
            "savefig.facecolor": TOKENS["surface"],
            "font.family": ["Segoe UI", "DejaVu Sans", "Arial", "sans-serif"],
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(17.4, 5.7), dpi=170)
    fig.text(0.035, 0.98, "Graph-shortest scaffold sweep", ha="left", va="top", fontsize=18, fontweight="bold", color=TOKENS["ink"])
    fig.text(0.035, 0.925, f"Fixed {cases} fair grid cases; temporary shortest-path springs are removed before measured-range polish.", ha="left", va="top", fontsize=9, color=TOKENS["muted"])
    labels = [f"h{row['max_hops']} sw={float(row['scaffold_weight']):.2g}\nrs={float(row['relative_sigma']):.2f}\nhb={float(row['hop_weight_base']):.2f}" for row in top]
    for ax, metric, title, color in [
        (axes[0], "p95_max_offset_m", "p95 max offset (m)", BLUE["base"]),
        (axes[1], "under_1m", "share under 1 m", OLIVE["base"]),
        (axes[2], "median_known_rmse_m", "median known RMSE (m)", GOLD["base"]),
    ]:
        values = [float(row[metric]) for row in top]
        x = np.arange(len(values))
        ax.set_facecolor(TOKENS["panel"])
        ax.bar(x, values, color=color, edgecolor=TOKENS["ink"], linewidth=0.35)
        if baseline is not None and metric in baseline:
            ax.axhline(float(baseline[metric]), color=PINK["dark"], linestyle="--", linewidth=1.0, label="production")
        ax.set_title(title, loc="left", fontsize=10.5, fontweight="bold", color=TOKENS["ink"])
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=0, fontsize=6.8)
        ax.tick_params(axis="y", labelsize=7.5, colors=TOKENS["muted"], length=0)
        ax.grid(True, axis="y", color=TOKENS["grid"], linewidth=0.5)
        for spine in ax.spines.values():
            spine.set_color(TOKENS["axis"])
    fig.subplots_adjust(left=0.05, right=0.985, top=0.82, bottom=0.20, wspace=0.20)
    fig.savefig(path, bbox_inches="tight", dpi=170)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep graph-shortest scaffold parameters on fixed generated grid cases.")
    parser.add_argument("--cases", type=int, default=6)
    parser.add_argument("--seed", type=int, default=2026062706)
    parser.add_argument("--seed-count", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=60)
    parser.add_argument("--max-hops", default="2,3,4")
    parser.add_argument("--relative-sigma", default="0.22,0.30,0.42")
    parser.add_argument("--scaffold-weight", default="1.0")
    parser.add_argument("--hop-weight-base", default="1.0")
    parser.add_argument("--fold-threshold", type=float, default=1.65)
    parser.add_argument("--prefix", default="anchor_solver_graph_scaffold_sweep")
    args = parser.parse_args()
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed % (2**32 - 1))
    torch.manual_seed(args.seed)
    device = torch.device("cpu")
    print(f"generating fixed grid cases={args.cases}", flush=True)
    cases = p95.generate_cases("grid", args.cases, device)
    detail: list[dict[str, str | int | float]] = []
    for case_index, case in enumerate(cases):
        batch = p95.graph_batch_from_cases([case], device)
        truth = dc.graph_to_truth(batch, 0)
        known_pairs = dc.known_pairs_from_batch(batch, 0)
        print(f"case={case_index + 1}/{len(cases)} anchors={len(truth)} known={len(known_pairs)}", flush=True)
        production = exp.production_solve(known_pairs, seed_count=args.seed_count, iterations=args.iterations, rng_seed=args.seed + case_index * 101 + 17)
        detail.append(row_dict(exp.metrics_row(case.bucket, case_index, case.family, case.shape, "production-priors", truth, production, known_pairs, fold_threshold_m=args.fold_threshold), max_hops="production", relative_sigma="production", scaffold_weight="production", hop_weight_base="production"))
        for max_hops in parse_ints(args.max_hops):
            for relative_sigma in parse_floats(args.relative_sigma):
                for scaffold_weight in parse_floats(args.scaffold_weight):
                    for hop_weight_base in parse_floats(args.hop_weight_base):
                        positions = exp.graph_shortest_scaffold_solve(
                            known_pairs,
                            seed_count=args.seed_count,
                            iterations=args.iterations,
                            rng_seed=(
                                args.seed
                                + case_index * 1009
                                + max_hops * 100
                                + int(relative_sigma * 1000)
                                + int(scaffold_weight * 10000)
                                + int(hop_weight_base * 100000)
                            ),
                            max_hops=max_hops,
                            relative_sigma=relative_sigma,
                            scaffold_weight=scaffold_weight,
                            hop_weight_base=hop_weight_base,
                        )
                        row = exp.metrics_row(case.bucket, case_index, case.family, case.shape, "graph-shortest-scaffold", truth, positions, known_pairs, fold_threshold_m=args.fold_threshold)
                        detail.append(row_dict(row, max_hops=max_hops, relative_sigma=relative_sigma, scaffold_weight=scaffold_weight, hop_weight_base=hop_weight_base))
    summary = summarize(detail)
    detail_path = OUTPUTS / f"{args.prefix}_detail.csv"
    summary_path = OUTPUTS / f"{args.prefix}_summary.csv"
    figure_path = OUTPUTS / f"{args.prefix}.png"
    write_csv(detail_path, detail)
    write_csv(summary_path, summary)
    make_figure(figure_path, summary, cases=args.cases)
    for row in sorted(summary, key=lambda item: float(item["p95_max_offset_m"]))[:12]:
        print(
            f"summary method={row['method']} h={row['max_hops']} rs={row['relative_sigma']} "
            f"sw={row['scaffold_weight']} hb={row['hop_weight_base']} "
            f"p95={float(row['p95_max_offset_m']):.3f}m median={float(row['median_max_offset_m']):.3f}m "
            f"under1m={float(row['under_1m']):.3f} folded={float(row['folded_case_rate']):.3f} rmse={float(row['median_known_rmse_m']):.4f}m",
            flush=True,
        )
    print(f"Wrote {detail_path}", flush=True)
    print(f"Wrote {summary_path}", flush=True)
    print(f"Wrote {figure_path}", flush=True)


if __name__ == "__main__":
    main()
