from __future__ import annotations

import csv
import importlib.util
import math
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT / "work" / "SmartClicker-GUI"
SWEEP_PATH = ROOT / "work" / "anchor_solver_large_hop_swap_sweep.py"
OUTPUTS = ROOT / "outputs"
sys.path.insert(0, str(REPO))

spec = importlib.util.spec_from_file_location("large_swap", SWEEP_PATH)
large_swap = importlib.util.module_from_spec(spec)
sys.modules["large_swap"] = large_swap
assert spec.loader is not None
spec.loader.exec_module(large_swap)

from uwb_capture.anchor_geometry import (  # noqa: E402
    _Parameterization,
    _anchor_degrees,
    _anchor_ids,
    _degree_shell_seed,
    _inferred_outer_shell_count,
    _layout_priors,
    _layout_scale,
    _local_minimize,
    _preprocess_pairs,
    pair_residuals,
)


pilot = large_swap.pilot
TOKENS = large_swap.TOKENS
BLUE = large_swap.BLUE
GOLD = large_swap.GOLD
ORANGE = large_swap.ORANGE
OLIVE = large_swap.OLIVE
PINK = large_swap.PINK
NEUTRAL = large_swap.NEUTRAL


def align_positions(truth, estimate):
    ids = sorted(set(truth) & set(estimate))
    target = np.array([truth[anchor_id] for anchor_id in ids], dtype=float)
    source = np.array([estimate[anchor_id] for anchor_id in ids], dtype=float)
    target_center = target.mean(axis=0)
    source_center = source.mean(axis=0)
    target_centered = target - target_center
    source_centered = source - source_center
    best_aligned = None
    best_offsets = None
    best_max = math.inf
    for reflect_y in (1.0, -1.0):
        reflected = source_centered.copy()
        reflected[:, 1] *= reflect_y
        u, _s, vt = np.linalg.svd(reflected.T @ target_centered)
        aligned = reflected @ (u @ vt) + target_center
        offsets = np.linalg.norm(aligned - target, axis=1)
        max_offset = float(offsets.max())
        if max_offset < best_max:
            best_max = max_offset
            best_aligned = aligned
            best_offsets = offsets
    assert best_aligned is not None
    assert best_offsets is not None
    return (
        {anchor_id: tuple(point) for anchor_id, point in zip(ids, best_aligned)},
        {anchor_id: float(offset) for anchor_id, offset in zip(ids, best_offsets)},
    )


def pair_metrics(positions, processed):
    residuals = pair_residuals(positions, processed)
    rmse = math.sqrt(sum(value * value for value in residuals.values()) / len(residuals))
    max_residual = max(abs(value) for value in residuals.values())
    return rmse, max_residual


def solve_example():
    case = large_swap.make_case()
    processed = _preprocess_pairs(case.exact_pairs, min_sigma_m=0.02, min_distance_m=0.05)
    anchor_ids = _anchor_ids(processed)
    parameterization = _Parameterization(anchor_ids)
    scale = _layout_scale(processed)
    seed_params = _degree_shell_seed(parameterization, processed, scale)
    seed_positions = parameterization.to_positions(seed_params)

    distance_params, _distance_energy = _local_minimize(
        seed_params,
        parameterization,
        processed,
        None,
        max_iterations=100,
    )
    distance_positions = parameterization.to_positions(distance_params)

    priors = _layout_priors(
        anchor_ids,
        processed,
        scale=scale,
        min_anchor_spacing_m=2.0,
        anchor_spacing_sigma_m=0.35,
        unmeasured_pair_min_distance_m=0.0,
        unmeasured_pair_sigma_m=1.0,
        boundary_degree_prior_sigma_m=12.0,
    )
    prior_params, _prior_energy = _local_minimize(
        seed_params,
        parameterization,
        processed,
        priors,
        max_iterations=100,
    )
    prior_polished_params, _polished_energy = _local_minimize(
        prior_params,
        parameterization,
        processed,
        None,
        max_iterations=100,
    )
    prior_positions = parameterization.to_positions(prior_polished_params)

    degrees = _anchor_degrees(anchor_ids, processed)
    outer_count = _inferred_outer_shell_count(len(anchor_ids))
    outer_ids = set(sorted(anchor_ids, key=lambda anchor_id: (degrees[anchor_id], anchor_id))[:outer_count])
    return case, processed, degrees, outer_ids, seed_positions, distance_positions, prior_positions


def layout_limits(case, layouts):
    xs = [x for x, _y in case.positions.values()]
    ys = [y for _x, y in case.positions.values()]
    for layout in layouts:
        aligned, _offsets = align_positions(case.positions, layout)
        xs.extend(x for x, _y in aligned.values())
        ys.extend(y for _x, y in aligned.values())
    span = max(max(xs) - min(xs), max(ys) - min(ys), 1.0)
    pad = 0.10 * span
    return min(xs) - pad, max(xs) + pad, min(ys) - pad, max(ys) + pad


def draw_truth(ax, case, degrees, outer_ids, limits):
    truth = case.positions
    ax.set_facecolor(TOKENS["panel"])
    for pair in case.exact_pairs:
        ax.plot(
            [truth[pair.anchor_a_id][0], truth[pair.anchor_b_id][0]],
            [truth[pair.anchor_a_id][1], truth[pair.anchor_b_id][1]],
            color=NEUTRAL["base"],
            linewidth=0.9,
            alpha=0.82,
            zorder=1,
        )
    colors = [BLUE["base"] if anchor_id in outer_ids else GOLD["base"] for anchor_id in truth]
    edges = [BLUE["dark"] if anchor_id in outer_ids else GOLD["dark"] for anchor_id in truth]
    ax.scatter(
        [point[0] for point in truth.values()],
        [point[1] for point in truth.values()],
        s=48,
        c=colors,
        edgecolors=edges,
        linewidths=1.0,
        zorder=3,
    )
    for anchor_id, (x_m, y_m) in truth.items():
        ax.text(x_m + 0.07, y_m + 0.07, f"{anchor_id}\\nd{degrees[anchor_id]}", fontsize=6.2, color=TOKENS["ink"])
    ax.set_title("Ground truth degrees\\nblue=outer-shell selected, gold=inner", loc="left", fontsize=10, fontweight="semibold", color=TOKENS["ink"])
    format_axis(ax, limits)


def draw_layout(ax, case, processed, layout, outer_ids, limits, title):
    truth = case.positions
    aligned, offsets = align_positions(truth, layout)
    rmse, max_residual = pair_metrics(layout, processed)
    worst_anchor = max(offsets, key=offsets.get)
    ax.set_facecolor(TOKENS["panel"])
    for pair in case.exact_pairs:
        ax.plot(
            [truth[pair.anchor_a_id][0], truth[pair.anchor_b_id][0]],
            [truth[pair.anchor_a_id][1], truth[pair.anchor_b_id][1]],
            color=NEUTRAL["base"],
            linewidth=0.75,
            alpha=0.70,
            zorder=1,
        )
    for anchor_id, true_point in truth.items():
        solved_point = aligned[anchor_id]
        is_worst = anchor_id == worst_anchor
        ax.plot(
            [true_point[0], solved_point[0]],
            [true_point[1], solved_point[1]],
            color=PINK["dark"] if is_worst else ORANGE["mid"],
            linewidth=1.5 if is_worst else 0.75,
            alpha=0.85 if is_worst else 0.38,
            zorder=2,
        )
    ax.scatter(
        [point[0] for point in truth.values()],
        [point[1] for point in truth.values()],
        s=30,
        facecolors=TOKENS["panel"],
        edgecolors=NEUTRAL["dark"],
        linewidths=0.9,
        zorder=3,
        label="truth",
    )
    colors = [BLUE["base"] if anchor_id in outer_ids else GOLD["base"] for anchor_id in aligned]
    edges = [BLUE["dark"] if anchor_id in outer_ids else GOLD["dark"] for anchor_id in aligned]
    ax.scatter(
        [point[0] for point in aligned.values()],
        [point[1] for point in aligned.values()],
        s=34,
        c=colors,
        edgecolors=edges,
        linewidths=0.9,
        zorder=4,
        label="layout",
    )
    ax.text(
        aligned[worst_anchor][0] + 0.08,
        aligned[worst_anchor][1],
        f"{worst_anchor}",
        fontsize=7,
        color=PINK["dark"],
        zorder=5,
    )
    ax.set_title(
        f"{title}\\nRMSE {rmse:.4f} m | max offset {max(offsets.values()):.3f} m",
        loc="left",
        fontsize=10,
        fontweight="semibold",
        color=TOKENS["ink"],
    )
    format_axis(ax, limits)


def format_axis(ax, limits):
    ax.set_xlim(limits[0], limits[1])
    ax.set_ylim(limits[2], limits[3])
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, color=TOKENS["grid"], linewidth=0.6)
    ax.tick_params(labelsize=7, colors=TOKENS["muted"], length=0)
    ax.set_xlabel("x (m)", fontsize=8, color=TOKENS["muted"])
    ax.set_ylabel("y (m)", fontsize=8, color=TOKENS["muted"])
    for spine in ax.spines.values():
        spine.set_color(TOKENS["axis"])


def write_csv(path, case, processed, degrees, outer_ids, layouts):
    rows = []
    for layout_name, layout in layouts:
        aligned, offsets = align_positions(case.positions, layout)
        rmse, max_residual = pair_metrics(layout, processed)
        for anchor_id in sorted(layout):
            rows.append(
                {
                    "layout": layout_name,
                    "anchor_id": anchor_id,
                    "degree": degrees[anchor_id],
                    "outer_shell": anchor_id in outer_ids,
                    "x_aligned_m": aligned[anchor_id][0],
                    "y_aligned_m": aligned[anchor_id][1],
                    "offset_m": offsets[anchor_id],
                    "pair_rmse_m": rmse,
                    "max_pair_residual_m": max_residual,
                }
            )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main():
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    case, processed, degrees, outer_ids, seed_positions, distance_positions, prior_positions = solve_example()
    limits = layout_limits(case, [seed_positions, distance_positions, prior_positions])
    plt.rcParams.update(
        {
            "figure.facecolor": TOKENS["surface"],
            "savefig.facecolor": TOKENS["surface"],
            "font.family": ["Segoe UI", "DejaVu Sans", "Arial", "sans-serif"],
        }
    )
    fig, axes = plt.subplots(1, 4, figsize=(19, 5.2), dpi=190)
    fig.patch.set_facecolor(TOKENS["surface"])
    fig.text(0.03, 0.965, "Degree-shell seed: starting layout and convergence", ha="left", va="top", fontsize=20, fontweight="bold", color=TOKENS["ink"])
    fig.text(
        0.03,
        0.915,
        "Deterministic irregular 4x4 grid, exact <=8 m anchor edges (42 measured links). Degree shell selects the 12 lowest-degree anchors for the outer circle and puts the 4 highest-degree anchors on an inner circle.",
        ha="left",
        va="top",
        fontsize=9.5,
        color=TOKENS["muted"],
    )
    draw_truth(axes[0], case, degrees, outer_ids, limits)
    draw_layout(axes[1], case, processed, seed_positions, outer_ids, limits, "Degree-shell starting seed")
    draw_layout(axes[2], case, processed, distance_positions, outer_ids, limits, "After distance-only LM")
    draw_layout(axes[3], case, processed, prior_positions, outer_ids, limits, "With close+edge priors, then polish")
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower left", bbox_to_anchor=(0.03, 0.02), frameon=False, ncol=2, fontsize=8.5)
    fig.text(
        0.20,
        0.04,
        "Orange/pink spokes show labeled-coordinate error after no-scale translation/rotation/mirror alignment. Pink marks the worst anchor.",
        ha="left",
        va="center",
        fontsize=8.5,
        color=TOKENS["muted"],
    )
    fig.subplots_adjust(left=0.045, right=0.992, top=0.82, bottom=0.12, wspace=0.18)
    png_path = OUTPUTS / "anchor_solver_degree_shell_example.png"
    csv_path = OUTPUTS / "anchor_solver_degree_shell_example.csv"
    fig.savefig(png_path, bbox_inches="tight", dpi=190)
    plt.close(fig)
    write_csv(
        csv_path,
        case,
        processed,
        degrees,
        outer_ids,
        [
            ("degree_shell_seed", seed_positions),
            ("distance_only_converged", distance_positions),
            ("close_edge_prior_then_polish", prior_positions),
        ],
    )
    for name, layout in [
        ("seed", seed_positions),
        ("distance-only", distance_positions),
        ("prior+polish", prior_positions),
    ]:
        aligned, offsets = align_positions(case.positions, layout)
        rmse, max_residual = pair_metrics(layout, processed)
        print(f"{name}: rmse={rmse:.6f} max_residual={max_residual:.6f} max_offset={max(offsets.values()):.3f}")
    print(f"Wrote {png_path}")
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
