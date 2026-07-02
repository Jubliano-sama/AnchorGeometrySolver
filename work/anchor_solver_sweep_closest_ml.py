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

import anchor_solver_ml_distance_completion as dc  # noqa: E402
import anchor_solver_p95_ml_cases as p95  # noqa: E402


TOKENS = {
    "surface": "#FCFCFD",
    "panel": "#FFFFFF",
    "ink": "#1F2430",
    "muted": "#6F768A",
    "grid": "#E6E8F0",
    "axis": "#D7DBE7",
}
BLUE = {"base": "#A3BEFA", "mid": "#5477C4", "dark": "#2E4780"}
PINK = {"base": "#F390CA", "mid": "#BD569B", "dark": "#8A3A6F"}
GOLD = {"base": "#FFE15B", "mid": "#B8A037", "dark": "#736422"}
ORANGE = {"base": "#F0986E", "mid": "#CC6F47", "dark": "#804126"}
OLIVE = {"base": "#A3D576", "mid": "#71B436", "dark": "#386411"}
NEUTRAL = {"light": "#E2E5EA", "base": "#C5CAD3", "dark": "#464C55"}


BUCKET_KEYS = ("random", "grid", "office")
BUCKET_LABELS = {
    "random": "Random 16-32",
    "grid": "Grid >=16",
    "office": "Office >=16",
}
METHOD_LABELS = {
    "ML completed": "ML completed",
    "ML weak polish": "ML weak polish",
}


def parse_cap_list(text: str) -> list[float]:
    caps: list[float] = []
    for part in text.split(","):
        token = part.strip().lower()
        if not token:
            continue
        if token in {"all", "inf", "none", "0"}:
            caps.append(0.0)
        else:
            value = float(token)
            if value <= 0.0:
                raise ValueError("Caps must be positive, or use 'all'.")
            caps.append(value)
    if not caps:
        raise ValueError("At least one cap is required.")
    return caps


def cap_label(cap: float) -> str:
    if cap <= 0.0:
        return "all"
    if abs(cap - round(cap)) < 1e-9:
        return str(int(round(cap)))
    return f"{cap:g}"


def cap_sort_value(cap: float) -> float:
    return 20.0 if cap <= 0.0 else cap


def summarize(values: list[float]) -> dict[str, float]:
    data = np.array(values, dtype=float)
    return {
        "median_max_offset_m": float(np.median(data)),
        "p90_max_offset_m": float(np.quantile(data, 0.90)),
        "p95_max_offset_m": float(np.quantile(data, 0.95)),
        "max_offset_m": float(np.max(data)),
        "under_0_2m": float(np.mean(data <= 0.20)),
        "under_0_5m": float(np.mean(data <= 0.50)),
        "under_1m": float(np.mean(data <= 1.00)),
    }


def row_dict(row: p95.SolvedCase, cap: float) -> dict[str, float | str | int]:
    return {
        "cap_per_anchor": cap_label(cap),
        "cap_numeric": cap,
        "bucket": row.bucket,
        "method": row.method,
        "case_index": row.case_index,
        "family": row.family,
        "shape": row.shape,
        "anchors": row.anchors,
        "known_pairs": row.known_pairs,
        "full_pairs": row.full_pairs,
        "missing_mae_m": row.missing_mae_m,
        "known_rmse_m": row.known_rmse_m,
        "known_max_residual_m": row.known_max_residual_m,
        "max_offset_m": row.max_offset_m,
        "median_offset_m": row.median_offset_m,
        "p95_offset_m": row.p95_offset_m,
    }


def write_csv(path: Path, rows: list[dict[str, float | str | int]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def make_summary(rows: list[dict[str, float | str | int]]) -> list[dict[str, float | str | int]]:
    summary: list[dict[str, float | str | int]] = []
    keys = sorted(
        {(float(row["cap_numeric"]), str(row["cap_per_anchor"]), str(row["bucket"]), str(row["method"])) for row in rows},
        key=lambda item: (str(item[2]), str(item[3]), cap_sort_value(item[0])),
    )
    for cap, label, bucket, method in keys:
        part = [row for row in rows if float(row["cap_numeric"]) == cap and row["bucket"] == bucket and row["method"] == method]
        values = [float(row["max_offset_m"]) for row in part]
        known_rmse = [float(row["known_rmse_m"]) for row in part]
        missing_mae = [float(row["missing_mae_m"]) for row in part]
        summary.append(
            {
                "cap_per_anchor": label,
                "cap_numeric": cap,
                "bucket": bucket,
                "method": method,
                "cases": len(part),
                **summarize(values),
                "median_known_rmse_m": float(np.median(known_rmse)),
                "median_missing_mae_m": float(np.median(missing_mae)),
            }
        )
    return summary


def best_by_bucket(summary: list[dict[str, float | str | int]], *, metric: str) -> list[dict[str, float | str | int]]:
    best: list[dict[str, float | str | int]] = []
    for bucket in BUCKET_LABELS.values():
        for method in METHOD_LABELS.values():
            part = [row for row in summary if row["bucket"] == bucket and row["method"] == method]
            if not part:
                continue
            best.append(min(part, key=lambda row: float(row[metric])))
    return best


def make_figure(path: Path, summary: list[dict[str, float | str | int]], *, cases_per_bucket: int) -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": TOKENS["surface"],
            "savefig.facecolor": TOKENS["surface"],
            "font.family": ["Segoe UI", "DejaVu Sans", "Arial", "sans-serif"],
        }
    )
    fig, axes = plt.subplots(3, 2, figsize=(15.8, 14.2), dpi=170, sharex=True)
    fig.text(
        0.035,
        0.985,
        "Closest ML-predicted distance cap sweep",
        ha="left",
        va="top",
        fontsize=20,
        fontweight="bold",
        color=TOKENS["ink"],
    )
    fig.text(
        0.035,
        0.958,
        (
            f"Same {cases_per_bucket} fair generated cases per bucket are reused for every cap. "
            "The x-axis is the number of closest ML-predicted missing distances admitted per anchor; "
            "'all' is the uncapped complete predicted graph. All measured <=8 m edges are always included."
        ),
        ha="left",
        va="top",
        fontsize=9.2,
        color=TOKENS["muted"],
    )
    metric_specs = [
        ("p95_max_offset_m", "p95 max offset (m)", ORANGE),
        ("under_1m", "share under 1 m", OLIVE),
    ]
    method_styles = {
        "ML completed": {"linestyle": "-", "marker": "o", "color": BLUE["mid"]},
        "ML weak polish": {"linestyle": "--", "marker": "s", "color": PINK["mid"]},
    }
    bucket_order = [BUCKET_LABELS[key] for key in BUCKET_KEYS]
    for row_index, bucket in enumerate(bucket_order):
        for col_index, (metric, ylabel, _family) in enumerate(metric_specs):
            ax = axes[row_index, col_index]
            ax.set_facecolor(TOKENS["panel"])
            for method, style in method_styles.items():
                part = [row for row in summary if row["bucket"] == bucket and row["method"] == method]
                part = sorted(part, key=lambda row: cap_sort_value(float(row["cap_numeric"])))
                xs = [cap_sort_value(float(row["cap_numeric"])) for row in part]
                ys = [float(row[metric]) for row in part]
                labels = [str(row["cap_per_anchor"]) for row in part]
                ax.plot(
                    xs,
                    ys,
                    color=style["color"],
                    linestyle=style["linestyle"],
                    marker=style["marker"],
                    markersize=4.2,
                    linewidth=1.2,
                    label=method,
                )
                if metric == "p95_max_offset_m":
                    best = min(part, key=lambda row: float(row[metric]))
                    ax.scatter(
                        [cap_sort_value(float(best["cap_numeric"]))],
                        [float(best[metric])],
                        s=58,
                        facecolors=GOLD["base"],
                        edgecolors=GOLD["dark"],
                        linewidths=0.9,
                        zorder=5,
                    )
            all_caps = sorted({float(row["cap_numeric"]) for row in summary}, key=cap_sort_value)
            ax.set_xticks([cap_sort_value(cap) for cap in all_caps], [cap_label(cap) for cap in all_caps], fontsize=7.8)
            ax.set_title(bucket, loc="left", fontsize=10.5, fontweight="semibold", color=TOKENS["ink"])
            ax.set_ylabel(ylabel, fontsize=8.4, color=TOKENS["muted"])
            if row_index == len(bucket_order) - 1:
                ax.set_xlabel("closest predicted distances per anchor", fontsize=8.4, color=TOKENS["muted"])
            ax.grid(True, color=TOKENS["grid"], linewidth=0.55)
            ax.tick_params(colors=TOKENS["muted"], labelsize=7.8, length=0)
            if metric == "under_1m":
                ax.set_ylim(-0.03, 1.03)
                ax.yaxis.set_major_formatter(lambda value, _pos: f"{value:.0%}")
            for spine in ax.spines.values():
                spine.set_color(TOKENS["axis"])
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(0.975, 0.987), frameon=False, ncol=2, fontsize=8.5)
    fig.text(
        0.035,
        0.026,
        "Gold dots mark the lowest p95 offset within each bucket/method. Offset is after translation/rotation/mirror alignment, with no scale fit.",
        ha="left",
        va="bottom",
        fontsize=8.1,
        color=TOKENS["muted"],
    )
    fig.subplots_adjust(left=0.065, right=0.985, top=0.91, bottom=0.075, hspace=0.32, wspace=0.16)
    fig.savefig(path, bbox_inches="tight", dpi=170)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep closest ML-predicted distance caps for anchor solves.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--checkpoint", type=Path, default=OUTPUTS / "anchor_solver_ml_distance_completion_fair_diag_cuda.pt")
    parser.add_argument("--cases-per-bucket", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--caps", default="0.5,1,2,3,4,5,6,8,12,all")
    parser.add_argument("--seed", type=int, default=20260628)
    parser.add_argument("--solver-iterations", type=int, default=80)
    parser.add_argument("--polish-iterations", type=int, default=80)
    parser.add_argument("--weak-polish-iterations", type=int, default=80)
    parser.add_argument("--predicted-sigma", type=float, default=0.55)
    parser.add_argument("--predicted-sigma-slope", type=float, default=0.65)
    parser.add_argument("--weak-completion-max-distance", type=float, default=18.0)
    parser.add_argument("--weak-completion-sigma-multiplier", type=float, default=2.5)
    parser.add_argument("--prefix", default="anchor_solver_ml_closest_cap_sweep")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    caps = parse_cap_list(args.caps)
    random.seed(args.seed)
    np.random.seed(args.seed % (2**32 - 1))
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = p95.choose_device(args.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
    print(f"device={device} cases_per_bucket={args.cases_per_bucket} caps={','.join(cap_label(cap) for cap in caps)}", flush=True)
    model = p95.load_model(args.checkpoint, device)
    chunks_by_bucket: dict[str, list[tuple[dc.GraphBatch, torch.Tensor, list[p95.CaseSpec]]]] = {}
    for bucket_key in BUCKET_KEYS:
        print(f"generating bucket={bucket_key}", flush=True)
        cases = p95.generate_cases(bucket_key, args.cases_per_bucket, device)
        node_counts = [int(case.points.shape[0]) for case in cases]
        print(
            f"generated bucket={bucket_key} min_nodes={min(node_counts)} "
            f"median_nodes={np.median(node_counts):.0f} max_nodes={max(node_counts)}",
            flush=True,
        )
        chunks_by_bucket[bucket_key] = p95.predict_batches(model, cases, device=device, batch_size=args.batch_size)

    detail_rows: list[dict[str, float | str | int]] = []
    for cap in caps:
        print(f"sweep cap={cap_label(cap)}", flush=True)
        for bucket_key in BUCKET_KEYS:
            rows = p95.solve_bucket(
                BUCKET_LABELS[bucket_key],
                chunks_by_bucket[bucket_key],
                predicted_sigma=args.predicted_sigma,
                predicted_sigma_slope=args.predicted_sigma_slope,
                closest_predicted_pairs_per_anchor=cap,
                solver_iterations=args.solver_iterations,
                polish_iterations=args.polish_iterations,
                weak_polish_iterations=args.weak_polish_iterations,
                weak_completion_max_distance=args.weak_completion_max_distance,
                weak_completion_sigma_multiplier=args.weak_completion_sigma_multiplier,
            )
            detail_rows.extend(row_dict(row, cap) for row in rows)
        summary = make_summary(detail_rows)
        latest = [row for row in summary if float(row["cap_numeric"]) == cap]
        for row in latest:
            print(
                f"summary cap={row['cap_per_anchor']} bucket={row['bucket']} method={row['method']} "
                f"median={float(row['median_max_offset_m']):.3f}m p95={float(row['p95_max_offset_m']):.3f}m "
                f"under1m={float(row['under_1m']):.0%}",
                flush=True,
            )

    summary_rows = make_summary(detail_rows)
    metrics_path = OUTPUTS / f"{args.prefix}_metrics.csv"
    summary_path = OUTPUTS / f"{args.prefix}_summary.csv"
    figure_path = OUTPUTS / f"{args.prefix}.png"
    write_csv(metrics_path, detail_rows)
    write_csv(summary_path, summary_rows)
    make_figure(figure_path, summary_rows, cases_per_bucket=args.cases_per_bucket)
    for row in best_by_bucket(summary_rows, metric="p95_max_offset_m"):
        print(
            f"best_p95 bucket={row['bucket']} method={row['method']} cap={row['cap_per_anchor']} "
            f"p95={float(row['p95_max_offset_m']):.3f}m median={float(row['median_max_offset_m']):.3f}m "
            f"under1m={float(row['under_1m']):.0%}",
            flush=True,
        )
    print(f"Wrote {metrics_path}", flush=True)
    print(f"Wrote {summary_path}", flush=True)
    print(f"Wrote {figure_path}", flush=True)


if __name__ == "__main__":
    main()
