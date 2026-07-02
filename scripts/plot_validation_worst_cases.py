from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from anchor_geometry_solver.benchmark import make_jobs  # noqa: E402
from anchor_geometry_solver.config import parse_config  # noqa: E402
from anchor_geometry_solver.layouts import generate_case_contexts  # noqa: E402
from anchor_geometry_solver.methods import solve_context  # noqa: E402
from anchor_geometry_solver.types import BenchmarkRow, CaseContext, MethodSpec  # noqa: E402

from anchor_geometry_solver.compat import ensure_legacy_paths  # noqa: E402

ensure_legacy_paths()
import anchor_solver_ml_distance_completion as dc  # noqa: E402


TOKENS = dc.TOKENS
BLUE = dc.BLUE
GOLD = dc.GOLD
ORANGE = dc.ORANGE
PINK = dc.PINK
NEUTRAL = dc.NEUTRAL
OLIVE = dc.OLIVE


def load_worst_rows(detail_path: Path) -> list[dict[str, str]]:
    rows = list(csv.DictReader(detail_path.open(newline="", encoding="utf-8")))
    worst: list[dict[str, str]] = []
    for method in sorted({row["method"] for row in rows}):
        method_rows = [row for row in rows if row["method"] == method and row["status"] == "ok"]
        if not method_rows:
            continue
        worst.append(max(method_rows, key=lambda row: float(row["max_offset_m"])))
    return worst


def metrics(
    truth: dict[str, tuple[float, float]],
    estimate: dict[str, tuple[float, float]],
    pairs: list[Any],
) -> dict[str, float]:
    max_offset, median_offset, p95_offset = dc.offset_summary(truth, estimate)
    known_rmse, known_max = dc.pair_metrics(estimate, pairs)
    return {
        "max_offset_m": max_offset,
        "median_offset_m": median_offset,
        "p95_offset_m": p95_offset,
        "known_rmse_m": known_rmse,
        "known_max_residual_m": known_max,
    }


def degrees(context: CaseContext) -> dict[str, int]:
    out = {anchor_id: 0 for anchor_id in context.truth}
    for pair in context.known_pairs:
        out[pair.anchor_a_id] += 1
        out[pair.anchor_b_id] += 1
    return out


def solve_case_for_methods(
    *,
    config_path: Path,
    case_indices: set[int],
) -> tuple[dict[int, CaseContext], tuple[MethodSpec, ...], dict[tuple[int, str], BenchmarkRow]]:
    config = parse_config(config_path)
    contexts = {context.case_index: context for context in generate_case_contexts(config.layouts, seed=config.seed, device_name=config.device)}
    jobs_by_key = {
        (context.case_index, method.name): (benchmark_name, context, method, fold_threshold_m, rng_seed)
        for benchmark_name, context, method, fold_threshold_m, rng_seed in make_jobs(config, list(contexts.values()))
        if context.case_index in case_indices
    }
    solved: dict[tuple[int, str], BenchmarkRow] = {}
    for method in config.methods:
        for case_index in sorted(case_indices):
            job = jobs_by_key[(case_index, method.name)]
            benchmark_name, context, _method, fold_threshold_m, rng_seed = job
            solved[(case_index, method.name)] = solve_context(
                benchmark_name,
                context,
                method,
                fold_threshold_m=fold_threshold_m,
                rng_seed=rng_seed,
            )
    return contexts, config.methods, solved


def solve_positions_for_panel(
    *,
    config_path: Path,
    case_indices: set[int],
) -> tuple[dict[int, CaseContext], tuple[MethodSpec, ...], dict[tuple[int, str], dict[str, tuple[float, float]]]]:
    config = parse_config(config_path)
    contexts = {context.case_index: context for context in generate_case_contexts(config.layouts, seed=config.seed, device_name=config.device)}
    positions: dict[tuple[int, str], dict[str, tuple[float, float]]] = {}
    for context in contexts.values():
        if context.case_index not in case_indices:
            continue
        for method_index, method in enumerate(config.methods):
            rng_seed = config.seed + context.case_index * 1009 + method_index * 9173 + 17
            from anchor_geometry_solver.methods import _solve_positions

            positions[(context.case_index, method.name)] = _solve_positions(context, method, rng_seed=rng_seed)
    return contexts, config.methods, positions


def method_color(method_name: str) -> str:
    if "sdp" in method_name:
        return BLUE["base"]
    if "branch" in method_name:
        return GOLD["base"]
    return OLIVE["base"]


def set_axis_limits(ax: Any, truth: dict[str, tuple[float, float]], aligned: dict[str, tuple[float, float]]) -> None:
    xs = [point[0] for point in truth.values()] + [point[0] for point in aligned.values()]
    ys = [point[1] for point in truth.values()] + [point[1] for point in aligned.values()]
    span = max(max(xs) - min(xs), max(ys) - min(ys), 1.0)
    pad = span * 0.12
    ax.set_xlim(min(xs) - pad, max(xs) + pad)
    ax.set_ylim(min(ys) - pad, max(ys) + pad)
    ax.set_aspect("equal", adjustable="box")


def draw_panel(
    ax: Any,
    *,
    context: CaseContext,
    positions: dict[str, tuple[float, float]],
    method: MethodSpec,
    row_label: str,
    highlight: bool,
) -> dict[str, float]:
    truth = context.truth
    pairs = list(context.known_pairs)
    aligned = dc.aligned_estimate(truth, positions)
    summary = metrics(truth, positions, pairs)
    offset_by_anchor = {
        anchor_id: math.dist(truth[anchor_id], aligned[anchor_id])
        for anchor_id in truth
    }
    worst_anchor = max(offset_by_anchor, key=offset_by_anchor.get)

    ax.set_facecolor(TOKENS["panel"])
    for pair in pairs:
        ax.plot(
            [truth[pair.anchor_a_id][0], truth[pair.anchor_b_id][0]],
            [truth[pair.anchor_a_id][1], truth[pair.anchor_b_id][1]],
            color=NEUTRAL["light"],
            linewidth=0.48,
            alpha=0.48,
            zorder=1,
        )
    for anchor_id, true_point in truth.items():
        solved = aligned[anchor_id]
        is_worst = anchor_id == worst_anchor
        ax.plot(
            [true_point[0], solved[0]],
            [true_point[1], solved[1]],
            color=PINK["dark"] if is_worst else ORANGE["dark"],
            linewidth=1.15 if is_worst else 0.55,
            alpha=0.72 if is_worst else 0.38,
            zorder=2,
        )
    ax.scatter(
        [point[0] for point in truth.values()],
        [point[1] for point in truth.values()],
        s=28,
        facecolors=TOKENS["panel"],
        edgecolors=NEUTRAL["dark"],
        linewidths=0.8,
        zorder=3,
        label="truth",
    )
    ax.scatter(
        [point[0] for point in aligned.values()],
        [point[1] for point in aligned.values()],
        s=34,
        color=method_color(method.name),
        edgecolors=TOKENS["ink"],
        linewidths=0.62,
        zorder=4,
        label="solved",
    )
    deg = degrees(context)
    for anchor_id in (worst_anchor,):
        x, y = truth[anchor_id]
        ax.text(x + 0.12, y + 0.12, f"{anchor_id}\nd{deg[anchor_id]}", fontsize=6.3, color=PINK["dark"], weight="bold")
    set_axis_limits(ax, truth, aligned)
    ax.grid(True, color=TOKENS["grid"], linewidth=0.55)
    ax.tick_params(labelsize=7, colors=TOKENS["muted"], length=0)
    for spine in ax.spines.values():
        spine.set_color(PINK["dark"] if highlight else TOKENS["axis"])
        spine.set_linewidth(1.3 if highlight else 0.8)
    ax.set_title(
        f"{method.name.replace('_', ' ')}\nmax {summary['max_offset_m']:.3f} m | p95 {summary['p95_offset_m']:.3f} m | RMSE {summary['known_rmse_m']:.4f} m",
        loc="left",
        fontsize=9.3,
        color=TOKENS["ink"],
        fontweight="bold" if highlight else "semibold",
    )
    ax.text(
        0.01,
        0.015,
        row_label,
        transform=ax.transAxes,
        fontsize=7.4,
        color=TOKENS["muted"],
        ha="left",
        va="bottom",
    )
    return summary


def write_case_metrics(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot worst validation cases for selected methods.")
    parser.add_argument("--config", type=Path, default=Path("configs/sweeps/visibility_tuned_validation.json"))
    parser.add_argument("--detail", type=Path, default=Path("outputs/visibility_tuned_validation/detail.csv"))
    parser.add_argument("--output", type=Path, default=Path("outputs/visibility_tuned_validation/worst_cases_all_methods.png"))
    parser.add_argument("--metrics", type=Path, default=Path("outputs/visibility_tuned_validation/worst_cases_all_methods_metrics.csv"))
    args = parser.parse_args()

    worst_rows = load_worst_rows(args.detail)
    case_indices = {int(row["case_index"]) for row in worst_rows}
    contexts, methods, positions = solve_positions_for_panel(config_path=args.config, case_indices=case_indices)

    row_defs = sorted(worst_rows, key=lambda row: (row["method"]))
    plt.rcParams.update(
        {
            "figure.facecolor": TOKENS["surface"],
            "savefig.facecolor": TOKENS["surface"],
            "font.family": ["Segoe UI", "DejaVu Sans", "Arial", "sans-serif"],
        }
    )
    fig, axes = plt.subplots(len(row_defs), len(methods), figsize=(16.2, 13.0), dpi=180)
    if len(row_defs) == 1:
        axes = np.expand_dims(axes, axis=0)
    metric_rows: list[dict[str, Any]] = []
    for row_i, worst_row in enumerate(row_defs):
        case_index = int(worst_row["case_index"])
        context = contexts[case_index]
        worst_method = worst_row["method"]
        row_label = (
            f"{context.bucket} case {case_index} | {context.shape} | "
            f"{context.anchor_count} anchors | worst for {worst_method.replace('_', ' ')}"
        )
        for col_i, method in enumerate(methods):
            summary = draw_panel(
                axes[row_i, col_i],
                context=context,
                positions=positions[(case_index, method.name)],
                method=method,
                row_label=row_label,
                highlight=method.name == worst_method,
            )
            metric_rows.append(
                {
                    "worst_for_method": worst_method,
                    "case_index": case_index,
                    "bucket": context.bucket,
                    "shape": context.shape,
                    "anchors": context.anchor_count,
                    "method": method.name,
                    **summary,
                }
            )
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(0.985, 0.965), frameon=False, fontsize=8)
    fig.suptitle(
        "Worst Validation Cases, Cross-Solved By All Selected Methods",
        x=0.055,
        y=0.992,
        ha="left",
        fontsize=18,
        fontweight="bold",
        color=TOKENS["ink"],
    )
    fig.text(
        0.055,
        0.956,
        "Gray rings are ground truth, colored dots are solved coordinates after no-scale rigid alignment; pink spoke marks the worst anchor in each panel.",
        ha="left",
        va="top",
        fontsize=9.5,
        color=TOKENS["muted"],
    )
    fig.subplots_adjust(left=0.055, right=0.985, top=0.895, bottom=0.055, wspace=0.16, hspace=0.30)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight")
    plt.close(fig)
    write_case_metrics(args.metrics, metric_rows)
    print(f"wrote {args.output}")
    print(f"wrote {args.metrics}")


if __name__ == "__main__":
    main()
