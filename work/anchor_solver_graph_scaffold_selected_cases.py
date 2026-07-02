from __future__ import annotations

import csv
import random
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
sys.path.insert(0, str(ROOT / "work"))

import anchor_solver_graph_scaffold_eval as graph_eval  # noqa: E402
import anchor_solver_fold_rescue_experiment as exp  # noqa: E402
import anchor_solver_ml_distance_completion as dc  # noqa: E402
import anchor_solver_p95_ml_cases as p95  # noqa: E402

TOKENS = dc.TOKENS
NEUTRAL = dc.NEUTRAL
BLUE = dc.BLUE
OLIVE = dc.OLIVE
ORANGE = dc.ORANGE

SEED = 2026062707
CASES_PER_BUCKET = 8
MAX_NODES = 32
SEED_COUNT = 6
ITERATIONS = 45
SELECTED_CASES = {
    2: "Random p95 graph case",
    12: "Grid hard old-solver case",
    23: "Office hardest graph case",
}


def solve_case(
    case: p95.CaseSpec,
    case_index: int,
    truth: dict[str, tuple[float, float]],
    known_pairs: list[dc.AnchorPairDistance],
) -> tuple[exp.MethodResult, exp.MethodResult]:
    production_positions = exp.production_solve(
        known_pairs,
        seed_count=SEED_COUNT,
        iterations=ITERATIONS,
        rng_seed=SEED + case_index * 101 + 17,
    )
    graph_positions = exp.graph_shortest_scaffold_solve(
        known_pairs,
        seed_count=SEED_COUNT,
        iterations=ITERATIONS,
        rng_seed=SEED + case_index * 1009 + 19,
        max_hops=2,
        relative_sigma=0.36,
    )
    production = exp.metrics_row(case.bucket, case_index, case.family, case.shape, "production-priors", truth, production_positions, known_pairs, fold_threshold_m=1.65)
    graph = exp.metrics_row(case.bucket, case_index, case.family, case.shape, "graph-shortest-scaffold", truth, graph_positions, known_pairs, fold_threshold_m=1.65)
    return production, graph


def draw_truth(ax, row: exp.MethodResult, title: str) -> None:
    truth = row.truth
    ax.set_facecolor(TOKENS["panel"])
    for pair in row.known_pair_list:
        ax.plot(
            [truth[pair.anchor_a_id][0], truth[pair.anchor_b_id][0]],
            [truth[pair.anchor_a_id][1], truth[pair.anchor_b_id][1]],
            color=NEUTRAL["light"],
            linewidth=0.55,
            alpha=0.55,
            zorder=1,
        )
    ax.scatter(
        [point[0] for point in truth.values()],
        [point[1] for point in truth.values()],
        s=22,
        facecolors=TOKENS["panel"],
        edgecolors=NEUTRAL["dark"],
        linewidths=0.75,
        zorder=3,
    )
    format_axes(ax, truth, truth)
    ax.set_title(title, loc="left", fontsize=9.4, fontweight="bold", color=TOKENS["ink"])


def draw_solution(ax, row: exp.MethodResult, *, color: str, title: str) -> None:
    truth = row.truth
    estimate = dc.aligned_estimate(truth, row.positions)
    ax.set_facecolor(TOKENS["panel"])
    for pair in row.known_pair_list:
        ax.plot(
            [truth[pair.anchor_a_id][0], truth[pair.anchor_b_id][0]],
            [truth[pair.anchor_a_id][1], truth[pair.anchor_b_id][1]],
            color=NEUTRAL["light"],
            linewidth=0.48,
            alpha=0.40,
            zorder=1,
        )
    for anchor_id, true_point in truth.items():
        solved = estimate[anchor_id]
        ax.plot([true_point[0], solved[0]], [true_point[1], solved[1]], color=ORANGE["dark"], linewidth=0.72, alpha=0.46, zorder=2)
    ax.scatter([x for x, _ in truth.values()], [y for _, y in truth.values()], s=17, facecolors=TOKENS["panel"], edgecolors=NEUTRAL["dark"], linewidths=0.65, zorder=3)
    ax.scatter([x for x, _ in estimate.values()], [y for _, y in estimate.values()], s=23, color=color, edgecolors=TOKENS["ink"], linewidths=0.50, zorder=4)
    format_axes(ax, truth, estimate)
    ax.set_title(
        f"{title}\nmax {row.max_offset_m:.2f} m | RMSE {row.known_rmse_m:.3f} m | close {row.close_pair_count}",
        loc="left",
        fontsize=8.7,
        fontweight="bold",
        color=TOKENS["ink"],
    )


def format_axes(ax, truth: dict[str, tuple[float, float]], estimate: dict[str, tuple[float, float]]) -> None:
    xs = [x for x, _ in truth.values()] + [x for x, _ in estimate.values()]
    ys = [y for _, y in truth.values()] + [y for _, y in estimate.values()]
    span = max(max(xs) - min(xs), max(ys) - min(ys), 1.0)
    pad = span * 0.14
    ax.set_xlim(min(xs) - pad, max(xs) + pad)
    ax.set_ylim(min(ys) - pad, max(ys) + pad)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, color=TOKENS["grid"], linewidth=0.50)
    ax.tick_params(labelsize=6.3, colors=TOKENS["muted"], length=0)
    for spine in ax.spines.values():
        spine.set_color(TOKENS["axis"])


def result_dict(row: exp.MethodResult) -> dict[str, str | int | float]:
    return exp.result_to_dict(row)


def main() -> None:
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    random.seed(SEED)
    np.random.seed(SEED % (2**32 - 1))
    torch.manual_seed(SEED)
    device = torch.device("cpu")
    selected: list[tuple[str, exp.MethodResult, exp.MethodResult]] = []
    detail_rows: list[dict[str, str | int | float]] = []
    case_counter = 0
    for bucket_key in graph_eval.parse_buckets("random,grid,office"):
        cases = graph_eval.generate_limited_cases(bucket_key, CASES_PER_BUCKET, device, max_nodes=MAX_NODES)
        for case in cases:
            batch = p95.graph_batch_from_cases([case], device)
            truth = dc.graph_to_truth(batch, 0)
            known_pairs = dc.known_pairs_from_batch(batch, 0)
            if case_counter in SELECTED_CASES:
                print(f"solving selected case={case_counter} label={SELECTED_CASES[case_counter]} anchors={case.points.shape[0]}", flush=True)
                production, graph = solve_case(case, case_counter, truth, known_pairs)
                selected.append((SELECTED_CASES[case_counter], production, graph))
                detail_rows.append(result_dict(production))
                detail_rows.append(result_dict(graph))
            case_counter += 1
    fig, axes = plt.subplots(len(selected), 3, figsize=(13.8, 10.6), dpi=170)
    fig.patch.set_facecolor(TOKENS["surface"])
    fig.text(0.035, 0.985, "Graph-shortest scaffold: selected solved layouts", ha="left", va="top", fontsize=18, fontweight="bold", color=TOKENS["ink"])
    fig.text(
        0.035,
        0.952,
        "Same generated fair cases as the capped eval. Solved layouts are aligned to truth only by rigid transform/mirror; orange segments are coordinate offsets.",
        ha="left",
        va="top",
        fontsize=9.0,
        color=TOKENS["muted"],
    )
    for row_index, (label, production, graph) in enumerate(selected):
        draw_truth(axes[row_index, 0], production, f"{label}\ntruth + measured edges")
        draw_solution(axes[row_index, 1], production, color=BLUE["base"], title="production priors")
        draw_solution(axes[row_index, 2], graph, color=OLIVE["dark"], title="graph-shortest scaffold")
    fig.subplots_adjust(left=0.055, right=0.985, top=0.900, bottom=0.055, hspace=0.38, wspace=0.18)
    figure_path = OUTPUTS / "anchor_solver_graph_scaffold_selected_cases.png"
    csv_path = OUTPUTS / "anchor_solver_graph_scaffold_selected_cases.csv"
    fig.savefig(figure_path, bbox_inches="tight", dpi=170)
    plt.close(fig)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, detail_rows[0].keys())
        writer.writeheader()
        writer.writerows(detail_rows)
    print(f"Wrote {figure_path}", flush=True)
    print(f"Wrote {csv_path}", flush=True)


if __name__ == "__main__":
    main()

