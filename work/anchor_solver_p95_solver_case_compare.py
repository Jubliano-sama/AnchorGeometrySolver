from __future__ import annotations

import argparse
import csv
import math
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

import anchor_solver_fold_rescue_experiment as exp
import anchor_solver_ml_distance_completion as dc
import anchor_solver_p95_ml_cases as p95
import anchor_solver_weighted_ppo_strong as wp

BUCKET_KEYS = ("random", "grid", "office")
BUCKET_LABELS = {"random": "Random 16-32", "grid": "Grid >=16", "office": "Office >=16"}
METHODS = ("default-production", "graph-shortest", "weighted-ppo")

TOKENS = dc.TOKENS
BLUE = dc.BLUE
GOLD = dc.GOLD
PINK = dc.PINK
OLIVE = dc.OLIVE
NEUTRAL = dc.NEUTRAL
ORANGE = dc.ORANGE


def write_csv(path: Path, rows: list[dict[str, str | int | float]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def p95_index(values: list[float]) -> int:
    order = np.argsort(np.array(values, dtype=float))
    return int(order[min(len(order) - 1, int(math.ceil(0.95 * len(order))) - 1)])


def solve_weighted_ppo(
    model: torch.nn.Module,
    case: p95.CaseSpec,
    *,
    device: torch.device,
    args: argparse.Namespace,
) -> tuple[dict[str, tuple[float, float]], dict[str, float]]:
    batch = p95.graph_batch_from_cases([case], device)
    model.eval()
    with torch.no_grad():
        pred, weight = model(batch.node_features, batch.edge_features, batch.mask, batch.measured_mask, batch.pair_mask)
    pairs = wp.weighted_pairs(batch, pred, weight, 0, args)
    known = dc.known_pairs_from_batch(batch, 0)
    positions = dc.completion_solution(
        pairs,
        known,
        max_iterations=args.solver_iterations,
        polish_known_iterations=args.polish_iterations,
    )
    n = batch.node_counts[0]
    pair_mask_np = batch.pair_mask[0, :n, :n].detach().cpu().numpy()
    measured_np = batch.measured_mask[0, :n, :n].detach().cpu().numpy()
    w_np = weight[0, :n, :n].detach().cpu().numpy()
    upper = np.triu(np.ones((n, n), dtype=bool), 1)
    known_w = w_np[upper & pair_mask_np & measured_np]
    pred_w = w_np[upper & pair_mask_np & ~measured_np]
    stats = {
        "known_weight_p05": float(np.quantile(known_w, 0.05)) if known_w.size else 1.0,
        "known_weight_p50": float(np.quantile(known_w, 0.50)) if known_w.size else 1.0,
        "known_weight_p95": float(np.quantile(known_w, 0.95)) if known_w.size else 1.0,
        "pred_weight_p05": float(np.quantile(pred_w, 0.05)) if pred_w.size else 1.0,
        "pred_weight_p50": float(np.quantile(pred_w, 0.50)) if pred_w.size else 1.0,
        "pred_weight_p95": float(np.quantile(pred_w, 0.95)) if pred_w.size else 1.0,
    }
    return positions, stats


def summarize(rows: list[dict[str, str | int | float]]) -> list[dict[str, str | int | float]]:
    out: list[dict[str, str | int | float]] = []
    for bucket in [BUCKET_LABELS[key] for key in BUCKET_KEYS]:
        for method in METHODS:
            part = [r for r in rows if r["bucket"] == bucket and r["method"] == method]
            offsets = np.array([float(r["max_offset_m"]) for r in part], dtype=float)
            rmse = np.array([float(r["known_rmse_m"]) for r in part], dtype=float)
            out.append(
                {
                    "bucket": bucket,
                    "method": method,
                    "cases": len(part),
                    "median_max_offset_m": float(np.median(offsets)),
                    "p95_max_offset_m": float(np.quantile(offsets, 0.95)),
                    "under_1m": float(np.mean(offsets <= 1.0)),
                    "median_known_rmse_m": float(np.median(rmse)),
                }
            )
    return out


def draw_case(ax, row: dict[str, str | int | float | object], *, color: str, edge_color: str) -> None:
    truth = row["_truth"]
    estimate = dc.aligned_estimate(truth, row["_positions"])
    known_pairs = row["_known_pairs"]
    ax.set_facecolor(TOKENS["panel"])
    for pair in known_pairs:
        ax.plot(
            [truth[pair.anchor_a_id][0], truth[pair.anchor_b_id][0]],
            [truth[pair.anchor_a_id][1], truth[pair.anchor_b_id][1]],
            color=NEUTRAL["light"],
            linewidth=0.55,
            alpha=0.55,
            zorder=1,
        )
    for anchor_id, true_point in truth.items():
        solved = estimate[anchor_id]
        ax.plot(
            [true_point[0], solved[0]],
            [true_point[1], solved[1]],
            color=ORANGE["dark"],
            linewidth=0.75,
            alpha=0.45,
            zorder=2,
        )
    ax.scatter(
        [point[0] for point in truth.values()],
        [point[1] for point in truth.values()],
        s=19,
        facecolors=TOKENS["panel"],
        edgecolors=NEUTRAL["dark"],
        linewidths=0.8,
        zorder=3,
    )
    ax.scatter(
        [point[0] for point in estimate.values()],
        [point[1] for point in estimate.values()],
        s=25,
        color=color,
        edgecolors=edge_color,
        linewidths=0.65,
        zorder=4,
    )
    xs = [x for x, _ in truth.values()] + [x for x, _ in estimate.values()]
    ys = [y for _, y in truth.values()] + [y for _, y in estimate.values()]
    span = max(max(xs) - min(xs), max(ys) - min(ys), 1.0)
    pad = span * 0.14
    ax.set_xlim(min(xs) - pad, max(xs) + pad)
    ax.set_ylim(min(ys) - pad, max(ys) + pad)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, color=TOKENS["grid"], linewidth=0.55)
    ax.tick_params(labelsize=6.8, colors=TOKENS["muted"], length=0)
    for spine in ax.spines.values():
        spine.set_color(TOKENS["axis"])
    ax.set_title(
        f"{row['method']} | {row['bucket']}\n"
        f"p95 case offset {float(row['max_offset_m']):.2f} m, RMSE {float(row['known_rmse_m']):.3f} m",
        loc="left",
        fontsize=8.8,
        fontweight="semibold",
        color=TOKENS["ink"],
    )
    ax.text(
        0.012,
        0.02,
        f"case {int(row['case_index'])}, {int(row['anchors'])} anchors, {row['shape']}, {int(row['known_pairs'])} measured",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=6.9,
        color=TOKENS["muted"],
        bbox={"facecolor": TOKENS["panel"], "edgecolor": "none", "alpha": 0.84, "pad": 1.6},
    )


def make_figure(path: Path, rows: list[dict[str, str | int | float | object]], *, cases_per_bucket: int) -> None:
    selected: list[dict[str, str | int | float | object]] = []
    for method in METHODS:
        for bucket in [BUCKET_LABELS[key] for key in BUCKET_KEYS]:
            part = [r for r in rows if r["bucket"] == bucket and r["method"] == method]
            selected.append(part[p95_index([float(r["max_offset_m"]) for r in part])])

    colors = {
        "default-production": (BLUE["base"], BLUE["dark"]),
        "graph-shortest": (OLIVE["base"], OLIVE["dark"]),
        "weighted-ppo": (PINK["base"], PINK["dark"]),
    }
    plt.rcParams.update(
        {
            "figure.facecolor": TOKENS["surface"],
            "savefig.facecolor": TOKENS["surface"],
            "font.family": ["Segoe UI", "DejaVu Sans", "Arial", "sans-serif"],
        }
    )
    fig, axes = plt.subplots(3, 3, figsize=(16.2, 15.8), dpi=170)
    fig.text(
        0.035,
        0.986,
        "P95 anchor-layout cases by solver",
        ha="left",
        va="top",
        fontsize=20,
        fontweight="bold",
        color=TOKENS["ink"],
    )
    fig.text(
        0.035,
        0.958,
        f"Each panel shows the p95 max-coordinate-offset case within {cases_per_bucket} same-seed generated layouts per bucket. "
        "Truth and solved layouts are Procrustes-aligned; orange lines are per-anchor offsets.",
        ha="left",
        va="top",
        fontsize=9.1,
        color=TOKENS["muted"],
    )
    for index, row in enumerate(selected):
        r = index // 3
        c = index % 3
        color, edge_color = colors[str(row["method"])]
        draw_case(axes[r, c], row, color=color, edge_color=edge_color)
    handles = [
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=TOKENS["panel"], markeredgecolor=NEUTRAL["dark"], label="Ground truth", markersize=6),
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=BLUE["base"], markeredgecolor=BLUE["dark"], label="Default solved", markersize=6),
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=OLIVE["base"], markeredgecolor=OLIVE["dark"], label="Graph-shortest solved", markersize=6),
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=PINK["base"], markeredgecolor=PINK["dark"], label="Weighted PPO solved", markersize=6),
        plt.Line2D([0], [0], color=ORANGE["dark"], linewidth=1.0, label="Offset"),
    ]
    fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.975, 0.988), frameon=False, ncol=3, fontsize=8.2)
    fig.subplots_adjust(left=0.035, right=0.985, top=0.91, bottom=0.04, hspace=0.28, wspace=0.14)
    fig.savefig(path, bbox_inches="tight", dpi=170)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Show p95 cases for default, graph-shortest, and weighted PPO solvers.")
    parser.add_argument("--cases-per-bucket", type=int, default=12)
    parser.add_argument("--seed", type=int, default=2026062732)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--ppo-checkpoint", type=Path, default=OUTPUTS / "anchor_solver_weighted_ppo_strong_known_weight_long_20260627_1930_best.pt")
    parser.add_argument("--seed-count", type=int, default=6)
    parser.add_argument("--iterations", type=int, default=45)
    parser.add_argument("--solver-iterations", type=int, default=80)
    parser.add_argument("--polish-iterations", type=int, default=80)
    parser.add_argument("--max-hops", type=int, default=2)
    parser.add_argument("--relative-sigma", type=float, default=0.36)
    parser.add_argument("--scaffold-weight", type=float, default=1.0)
    parser.add_argument("--hop-weight-base", type=float, default=1.0)
    parser.add_argument("--predicted-sigma", type=float, default=0.55)
    parser.add_argument("--predicted-sigma-slope", type=float, default=0.65)
    parser.add_argument("--closest-predicted-pairs-per-anchor", type=float, default=5.0)
    parser.add_argument("--weight-known-springs", action="store_true", default=True)
    parser.add_argument("--prefix", default="anchor_solver_p95_solver_case_compare")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed % (2**32 - 1))
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    ppo_model, _source = wp.load_weighted_model(args.ppo_checkpoint, device)

    rows: list[dict[str, str | int | float | object]] = []
    case_counter = 0
    for bucket_key in BUCKET_KEYS:
        print(f"generating bucket={bucket_key} cases={args.cases_per_bucket}", flush=True)
        cases = p95.generate_cases(bucket_key, args.cases_per_bucket, torch.device("cpu"))
        for local_index, case in enumerate(cases):
            batch = p95.graph_batch_from_cases([case], torch.device("cpu"))
            truth = dc.graph_to_truth(batch, 0)
            known_pairs = dc.known_pairs_from_batch(batch, 0)
            rng_seed = args.seed + case_counter * 1009
            print(f"solving bucket={bucket_key} case={local_index + 1}/{len(cases)} anchors={len(truth)}", flush=True)
            solved = {
                "default-production": exp.production_solve(known_pairs, seed_count=args.seed_count, iterations=args.iterations, rng_seed=rng_seed + 17),
                "graph-shortest": exp.graph_shortest_scaffold_solve(
                    known_pairs,
                    seed_count=args.seed_count,
                    iterations=args.iterations,
                    rng_seed=rng_seed + 19,
                    max_hops=args.max_hops,
                    relative_sigma=args.relative_sigma,
                    scaffold_weight=args.scaffold_weight,
                    hop_weight_base=args.hop_weight_base,
                ),
            }
            ppo_positions, ppo_weight_stats = solve_weighted_ppo(ppo_model, case, device=device, args=args)
            solved["weighted-ppo"] = ppo_positions
            for method, positions in solved.items():
                max_offset, median_offset, p95_offset = dc.offset_summary(truth, positions)
                known_rmse, known_max = dc.pair_metrics(positions, known_pairs)
                weight_stats = ppo_weight_stats if method == "weighted-ppo" else {}
                rows.append(
                    {
                        "bucket": BUCKET_LABELS[bucket_key],
                        "case_index": local_index,
                        "family": case.family,
                        "shape": case.shape,
                        "method": method,
                        "anchors": len(truth),
                        "known_pairs": len(known_pairs),
                        "max_offset_m": max_offset,
                        "median_offset_m": median_offset,
                        "p95_offset_m": p95_offset,
                        "known_rmse_m": known_rmse,
                        "known_max_residual_m": known_max,
                        "known_weight_p05": weight_stats.get("known_weight_p05", ""),
                        "known_weight_p50": weight_stats.get("known_weight_p50", ""),
                        "known_weight_p95": weight_stats.get("known_weight_p95", ""),
                        "pred_weight_p05": weight_stats.get("pred_weight_p05", ""),
                        "pred_weight_p50": weight_stats.get("pred_weight_p50", ""),
                        "pred_weight_p95": weight_stats.get("pred_weight_p95", ""),
                        "_truth": truth,
                        "_positions": positions,
                        "_known_pairs": known_pairs,
                    }
                )
            case_counter += 1

    public_rows = [{k: v for k, v in row.items() if not str(k).startswith("_")} for row in rows]
    detail_path = OUTPUTS / f"{args.prefix}_detail.csv"
    summary_path = OUTPUTS / f"{args.prefix}_summary.csv"
    figure_path = OUTPUTS / f"{args.prefix}.png"
    write_csv(detail_path, public_rows)
    write_csv(summary_path, summarize(public_rows))
    make_figure(figure_path, rows, cases_per_bucket=args.cases_per_bucket)
    for row in summarize(public_rows):
        print(
            f"summary bucket={row['bucket']} method={row['method']} p95={float(row['p95_max_offset_m']):.3f}m "
            f"median={float(row['median_max_offset_m']):.3f}m under1={float(row['under_1m']):.3f} rmse={float(row['median_known_rmse_m']):.4f}",
            flush=True,
        )
    print(f"Wrote {detail_path}", flush=True)
    print(f"Wrote {summary_path}", flush=True)
    print(f"Wrote {figure_path}", flush=True)


if __name__ == "__main__":
    main()
