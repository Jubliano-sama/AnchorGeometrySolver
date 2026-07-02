
from __future__ import annotations

import importlib.util
import math
from pathlib import Path
import random
import sys
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SWEEP_PATH = ROOT / "work" / "anchor_solver_large_hop_swap_sweep.py"
OUTPUTS = ROOT / "outputs"

spec = importlib.util.spec_from_file_location("large_swap", SWEEP_PATH)
large_swap = importlib.util.module_from_spec(spec)
sys.modules["large_swap"] = large_swap
assert spec.loader is not None
spec.loader.exec_module(large_swap)

pilot = large_swap.pilot
TOKENS = large_swap.TOKENS
BLUE = large_swap.BLUE
GOLD = large_swap.GOLD
ORANGE = large_swap.ORANGE
OLIVE = large_swap.OLIVE
PINK = large_swap.PINK
NEUTRAL = large_swap.NEUTRAL

CASE_COUNT = 7
METHODS = [
    ("stock hops", 24, 10, 0.35, False, False, False),
    ("larger hops", 24, 10, 1.50, False, False, False),
    ("large + swap", 24, 10, 1.50, True, False, False),
    ("corner large + swap", 24, 10, 1.50, True, True, False),
    ("corner prior + swap", 24, 10, 1.50, True, True, True),
]


def make_case_for_index(case_index: int) -> large_swap.GridCase:
    rng = random.Random(large_swap.RNG_SEED + 900 + case_index * 17)
    discarded = 0
    while True:
        positions = large_swap.irregular_grid_positions(rng)
        exact_pairs = large_swap.exact_pairs_from_positions(positions)
        if pilot.accepted_graph(positions, exact_pairs):
            noisy_pairs, nlos_count = large_swap.noisy_pairs_from_exact(
                exact_pairs,
                random.Random(large_swap.RNG_SEED + 1900 + case_index * 17),
            )
            return large_swap.GridCase(
                positions=positions,
                rows=4,
                cols=4,
                exact_pairs=exact_pairs,
                noisy_pairs=noisy_pairs,
                nlos_count=nlos_count,
                discarded_before=discarded,
            )
        discarded += 1


def solve_metrics(case: large_swap.GridCase, measurement: str, pairs, method):
    positions, energy, pair_rmse, max_pair = large_swap.solve_search(case, pairs, method)
    max_offset, median_offset, p95_offset = pilot.position_error_summary(case.positions, positions)
    return {
        "measurement": measurement,
        "method": method[0],
        "positions": positions,
        "energy": energy,
        "pair_rmse_m": pair_rmse,
        "max_pair_residual_m": max_pair,
        "max_coord_offset_m": max_offset,
        "median_coord_offset_m": median_offset,
        "p95_coord_offset_m": p95_offset,
    }


def align_positions(truth, estimate):
    ids = sorted(set(truth) & set(estimate))
    target = np.array([truth[a] for a in ids], dtype=float)
    source = np.array([estimate[a] for a in ids], dtype=float)
    sc = source.mean(axis=0)
    tc = target.mean(axis=0)
    u, _s, vt = np.linalg.svd((source - sc).T @ (target - tc))
    transform = u @ vt
    aligned = (source - sc) @ transform + tc
    return {anchor_id: tuple(point) for anchor_id, point in zip(ids, aligned)}


def layout_limits(case, aligned_layouts):
    xs = [p[0] for p in case.positions.values()]
    ys = [p[1] for p in case.positions.values()]
    for layout in aligned_layouts:
        xs.extend(p[0] for p in layout.values())
        ys.extend(p[1] for p in layout.values())
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    span = max(max_x - min_x, max_y - min_y, 1.0)
    pad = 0.08 * span
    return min_x - pad, max_x + pad, min_y - pad, max_y + pad


def draw_panel(ax, case, result, limits, color, edge):
    truth = case.positions
    aligned = align_positions(truth, result["positions"])
    for pair in case.exact_pairs:
        ax.plot(
            [truth[pair.anchor_a_id][0], truth[pair.anchor_b_id][0]],
            [truth[pair.anchor_a_id][1], truth[pair.anchor_b_id][1]],
            color=NEUTRAL["light"],
            linewidth=0.55,
            alpha=0.6,
            zorder=1,
        )
    for anchor_id, true_point in truth.items():
        solved_point = aligned[anchor_id]
        ax.plot(
            [true_point[0], solved_point[0]],
            [true_point[1], solved_point[1]],
            color=ORANGE["mid"],
            linewidth=0.7,
            alpha=0.42,
            zorder=2,
        )
    ax.scatter(
        [p[0] for p in truth.values()],
        [p[1] for p in truth.values()],
        s=26,
        facecolors=TOKENS["panel"],
        edgecolors=NEUTRAL["dark"],
        linewidths=0.85,
        zorder=3,
        label="truth",
    )
    ax.scatter(
        [p[0] for p in aligned.values()],
        [p[1] for p in aligned.values()],
        s=23,
        color=color,
        edgecolors=edge,
        linewidths=0.85,
        zorder=4,
        label="solved",
    )
    ax.set_title(
        f"{result['method']}\nRMSE {result['pair_rmse_m']:.3f} m | max offset {result['max_coord_offset_m']:.2f} m",
        loc="left",
        fontsize=8.7,
        color=TOKENS["ink"],
        fontweight="semibold",
    )
    min_x, max_x, min_y, max_y = limits
    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, color=TOKENS["grid"], linewidth=0.55)
    ax.tick_params(labelsize=7, colors=TOKENS["muted"], length=0)
    for spine in ax.spines.values():
        spine.set_color(TOKENS["axis"])


def main():
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    stock_method = METHODS[0]
    candidates = []
    for case_index in range(CASE_COUNT):
        case = make_case_for_index(case_index)
        stock = solve_metrics(case, "noisy+nlos", case.noisy_pairs, stock_method)
        candidates.append((case_index, case, stock))
        print(
            f"candidate {case_index}: stock noisy RMSE={stock['pair_rmse_m']:.4f} max_offset={stock['max_coord_offset_m']:.3f} nlos={case.nlos_count}/{len(case.noisy_pairs)}",
            flush=True,
        )
    offsets = sorted(item[2]["max_coord_offset_m"] for item in candidates)
    median = offsets[len(offsets) // 2]
    case_index, case, _stock = min(candidates, key=lambda item: abs(item[2]["max_coord_offset_m"] - median))
    print(f"selected median candidate {case_index} with stock noisy max offset near {median:.3f}", flush=True)

    results = []
    for measurement, pairs in (("exact", case.exact_pairs), ("noisy+nlos", case.noisy_pairs)):
        for method in METHODS:
            result = solve_metrics(case, measurement, pairs, method)
            results.append(result)
            print(
                f"{measurement} | {result['method']}: RMSE={result['pair_rmse_m']:.4f} max_offset={result['max_coord_offset_m']:.3f}",
                flush=True,
            )

    csv_path = OUTPUTS / "anchor_solver_updated_median_case.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "selected_case_index", "measurement", "method", "pair_rmse_m", "max_pair_residual_m",
            "max_coord_offset_m", "median_coord_offset_m", "p95_coord_offset_m", "energy",
            "anchors", "pairs", "nlos_count", "nlos_share", "discarded_before",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for result in results:
            writer.writerow({
                "selected_case_index": case_index,
                "measurement": result["measurement"],
                "method": result["method"],
                "pair_rmse_m": result["pair_rmse_m"],
                "max_pair_residual_m": result["max_pair_residual_m"],
                "max_coord_offset_m": result["max_coord_offset_m"],
                "median_coord_offset_m": result["median_coord_offset_m"],
                "p95_coord_offset_m": result["p95_coord_offset_m"],
                "energy": result["energy"],
                "anchors": len(case.positions),
                "pairs": len(case.exact_pairs),
                "nlos_count": case.nlos_count,
                "nlos_share": case.nlos_count / max(len(case.noisy_pairs), 1),
                "discarded_before": case.discarded_before,
            })

    plt.rcParams.update({
        "figure.facecolor": TOKENS["surface"],
        "savefig.facecolor": TOKENS["surface"],
        "font.family": ["Segoe UI", "DejaVu Sans", "Arial", "sans-serif"],
    })
    fig, axes = plt.subplots(2, len(METHODS), figsize=(19, 7.6), dpi=180)
    fig.patch.set_facecolor(TOKENS["surface"])
    fig.text(0.025, 0.975, "Updated Median Grid-Like Case", ha="left", va="top", fontsize=22, fontweight="bold", color=TOKENS["ink"])
    fig.text(
        0.025,
        0.93,
        f"Selected case {case_index} as median of {CASE_COUNT} corrected irregular-grid candidates by stock noisy/NLOS worst-anchor offset. Gray=truth, colored=solved after no-scale alignment, orange spokes=label error.",
        ha="left",
        va="top",
        fontsize=10,
        color=TOKENS["muted"],
    )
    fig.text(
        0.025,
        0.895,
        f"Anchors {len(case.positions)} | true edges {len(case.exact_pairs)} using <=8 m | noisy case NLOS {case.nlos_count}/{len(case.noisy_pairs)} ({case.nlos_count / len(case.noisy_pairs):.0%}) | known corners A00/A03/A12 for corner modes",
        ha="left",
        va="top",
        fontsize=9.5,
        color=TOKENS["muted"],
    )
    color_map = {
        "stock hops": (NEUTRAL["light"], NEUTRAL["dark"]),
        "larger hops": (BLUE["base"], BLUE["dark"]),
        "large + swap": (GOLD["base"], GOLD["dark"]),
        "corner large + swap": (PINK["base"], PINK["dark"]),
        "corner prior + swap": (OLIVE["base"], OLIVE["dark"]),
    }
    all_aligned = [align_positions(case.positions, r["positions"]) for r in results]
    limits = layout_limits(case, all_aligned)
    for row, measurement in enumerate(["exact", "noisy+nlos"]):
        fig.text(0.025, 0.778 - row * 0.382, measurement, ha="left", va="center", fontsize=11, fontweight="bold", color=TOKENS["ink"])
        for col, method in enumerate(METHODS):
            label = method[0]
            result = next(r for r in results if r["measurement"] == measurement and r["method"] == label)
            color, edge = color_map[label]
            draw_panel(axes[row, col], case, result, limits, color, edge)
            if row == 1:
                axes[row, col].set_xlabel("x (m)", fontsize=8, color=TOKENS["muted"])
            if col == 0:
                axes[row, col].set_ylabel("y (m)", fontsize=8, color=TOKENS["muted"])
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower left", bbox_to_anchor=(0.025, 0.025), frameon=False, ncol=2, fontsize=9)
    fig.text(0.19, 0.04, "RMSE is final pair-distance RMSE. Max offset is the worst labeled anchor after translation/rotation/optional mirror alignment; scale is not changed.", ha="left", va="center", fontsize=8.5, color=TOKENS["muted"])
    fig.subplots_adjust(left=0.055, right=0.99, top=0.84, bottom=0.105, wspace=0.18, hspace=0.43)
    png_path = OUTPUTS / "anchor_solver_updated_median_case.png"
    fig.savefig(png_path, bbox_inches="tight", dpi=180)
    plt.close(fig)
    print(f"Wrote {png_path}")
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
