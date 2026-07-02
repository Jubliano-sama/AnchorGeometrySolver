from __future__ import annotations

import csv
import importlib.util
import math
from pathlib import Path
import random
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT / "work" / "SmartClicker-GUI"
OUTPUTS = ROOT / "outputs"
BASE_EXAMPLE_PATH = ROOT / "work" / "anchor_solver_5x5_grid_completion_example.py"
sys.path.insert(0, str(REPO))

spec = importlib.util.spec_from_file_location("grid_example", BASE_EXAMPLE_PATH)
grid_example = importlib.util.module_from_spec(spec)
sys.modules["grid_example"] = grid_example
assert spec.loader is not None
spec.loader.exec_module(grid_example)

from uwb_capture.anchor_geometry import (  # noqa: E402
    AnchorPairDistance,
    _Parameterization,
    _anchor_ids,
    _initial_parameters,
    _layout_scale,
    _local_minimize,
    _positions_to_params,
    _preprocess_pairs,
    pair_residuals,
    rotate_layout_to_level,
    solve_anchor_layout,
)


dc = grid_example.dc
TOKENS = grid_example.TOKENS
BLUE = grid_example.BLUE
GOLD = grid_example.GOLD
ORANGE = grid_example.ORANGE
OLIVE = grid_example.OLIVE
PINK = grid_example.PINK
NEUTRAL = grid_example.NEUTRAL


def best_normal_distance_solve(known_pairs: list[AnchorPairDistance], *, seed_count: int = 12) -> dict[str, tuple[float, float]]:
    processed = _preprocess_pairs(known_pairs, min_sigma_m=0.02, min_distance_m=0.05)
    anchor_ids = _anchor_ids(processed)
    parameterization = _Parameterization(anchor_ids)
    scale = _layout_scale(processed)
    seeds = _initial_parameters(
        parameterization,
        processed,
        seed_count=seed_count,
        scale=scale,
        rng=random.Random(20260626),
    )
    best_params = None
    best_rmse = math.inf
    for seed in seeds:
        params, _energy = _local_minimize(seed, parameterization, processed, None, max_iterations=100)
        positions = parameterization.to_positions(params)
        rmse, _max_residual = pair_metrics(positions, known_pairs)
        if rmse < best_rmse:
            best_rmse = rmse
            best_params = params
    assert best_params is not None
    positions = parameterization.to_positions(best_params)
    return rotate_layout_to_level(positions, anchor_ids[0], anchor_ids[1])


def pair_metrics(positions: dict[str, tuple[float, float]], pairs: list[AnchorPairDistance]) -> tuple[float, float]:
    residuals = pair_residuals(positions, pairs)
    values = list(residuals.values())
    return math.sqrt(sum(v * v for v in values) / len(values)), max(abs(v) for v in values)


def layout_metrics(
    truth: dict[str, tuple[float, float]],
    estimate: dict[str, tuple[float, float]],
    known_pairs: list[AnchorPairDistance],
) -> dict[str, float]:
    max_offset, median_offset, p95_offset = dc.offset_summary(truth, estimate)
    rmse, max_residual = pair_metrics(estimate, known_pairs)
    return {
        "max_offset_m": max_offset,
        "median_offset_m": median_offset,
        "p95_offset_m": p95_offset,
        "known_rmse_m": rmse,
        "known_max_residual_m": max_residual,
    }


def radial_expand(
    positions: dict[str, tuple[float, float]],
    factor: float,
) -> dict[str, tuple[float, float]]:
    if factor == 1.0:
        return dict(positions)
    center_x = sum(x for x, _ in positions.values()) / len(positions)
    center_y = sum(y for _, y in positions.values()) / len(positions)
    return {
        anchor_id: (
            center_x + (x - center_x) * factor,
            center_y + (y - center_y) * factor,
        )
        for anchor_id, (x, y) in positions.items()
    }


def nearest_uniform_springs(
    positions: dict[str, tuple[float, float]],
    *,
    k: int,
    rest_length_m: float,
    sigma_m: float,
) -> list[AnchorPairDistance]:
    pairs: set[tuple[str, str]] = set()
    ids = sorted(positions)
    for anchor_id in ids:
        ax, ay = positions[anchor_id]
        distances: list[tuple[float, str]] = []
        for other_id in ids:
            if other_id == anchor_id:
                continue
            bx, by = positions[other_id]
            distances.append((math.hypot(ax - bx, ay - by), other_id))
        for _distance, other_id in sorted(distances)[:k]:
            pairs.add(tuple(sorted((anchor_id, other_id))))
    return [
        AnchorPairDistance(a, b, rest_length_m, sigma_m=sigma_m, enabled=True, source="explosion")
        for a, b in sorted(pairs)
    ]


def explosion_solve(
    known_pairs: list[AnchorPairDistance],
    *,
    cycles: int,
    k: int,
    rest_length_m: float,
    sigma_m: float,
    expand_factor: float,
    seed_count: int = 12,
) -> dict[str, tuple[float, float]]:
    known_processed = _preprocess_pairs(known_pairs, min_sigma_m=0.02, min_distance_m=0.05)
    anchor_ids = _anchor_ids(known_processed)
    parameterization = _Parameterization(anchor_ids)
    positions = best_normal_distance_solve(known_pairs, seed_count=seed_count)
    params = _positions_to_params(parameterization, positions)
    for _cycle in range(cycles):
        current_positions = parameterization.to_positions(params)
        springs = nearest_uniform_springs(
            current_positions,
            k=k,
            rest_length_m=rest_length_m,
            sigma_m=sigma_m,
        )
        expanded = radial_expand(current_positions, expand_factor)
        expanded_params = _positions_to_params(parameterization, expanded)
        augmented = _preprocess_pairs(
            [*known_pairs, *springs],
            min_sigma_m=0.02,
            min_distance_m=0.05,
        )
        exploded_params, _exploded_energy = _local_minimize(
            expanded_params,
            parameterization,
            augmented,
            None,
            max_iterations=70,
        )
        params, _energy = _local_minimize(
            exploded_params,
            parameterization,
            known_processed,
            None,
            max_iterations=100,
        )
    positions = parameterization.to_positions(params)
    return rotate_layout_to_level(positions, anchor_ids[0], anchor_ids[1])


def solve_current_production(known_pairs: list[AnchorPairDistance]) -> dict[str, tuple[float, float]]:
    result = solve_anchor_layout(
        known_pairs,
        seed_count=24,
        basin_hops=10,
        max_iterations=100,
        random_seed=20260626,
    )
    return result.positions_m


def solve_ml_completed(batch, known_pairs, *, device):
    checkpoint_path = OUTPUTS / "anchor_solver_ml_distance_completion_office_cuda.pt"
    model = grid_example.load_model(checkpoint_path, batch, device)
    with torch.no_grad():
        pred = model(batch.node_features, batch.edge_features, batch.mask, batch.measured_mask, batch.pair_mask)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    args = checkpoint["args"]
    completed_pairs, _pred_matrix, _scale = dc.completed_pairs_from_prediction(
        batch,
        pred,
        0,
        predicted_sigma_m=float(args.get("predicted_sigma", 0.55)),
        predicted_sigma_slope=float(args.get("predicted_sigma_slope", 0.65)),
    )
    return dc.completion_solution(
        completed_pairs,
        known_pairs,
        max_iterations=100,
        polish_known_iterations=100,
    )


def draw_panel(ax, truth, estimate, known_pairs, title, color):
    aligned = dc.aligned_estimate(truth, estimate)
    m = layout_metrics(truth, estimate, known_pairs)
    ax.set_facecolor(TOKENS["panel"])
    for pair in known_pairs:
        ax.plot(
            [truth[pair.anchor_a_id][0], truth[pair.anchor_b_id][0]],
            [truth[pair.anchor_a_id][1], truth[pair.anchor_b_id][1]],
            color=NEUTRAL["light"],
            linewidth=0.45,
            alpha=0.45,
            zorder=1,
        )
    for anchor_id, true_point in truth.items():
        solved = aligned[anchor_id]
        ax.plot(
            [true_point[0], solved[0]],
            [true_point[1], solved[1]],
            color=ORANGE["dark"],
            linewidth=0.75,
            alpha=0.50,
            zorder=2,
        )
    ax.scatter(
        [x for x, _ in truth.values()],
        [y for _, y in truth.values()],
        s=24,
        facecolors=TOKENS["panel"],
        edgecolors=NEUTRAL["dark"],
        linewidths=0.75,
        zorder=3,
    )
    ax.scatter(
        [x for x, _ in aligned.values()],
        [y for _, y in aligned.values()],
        s=30,
        color=color,
        edgecolors=TOKENS["ink"],
        linewidths=0.65,
        zorder=4,
    )
    format_axis(ax, truth, aligned)
    ax.set_title(
        f"{title}\nmax {m['max_offset_m']:.3f} m | med {m['median_offset_m']:.3f} m | RMSE {m['known_rmse_m']:.4f} m",
        loc="left",
        fontsize=10,
        fontweight="bold",
        color=TOKENS["ink"],
    )


def draw_truth(ax, truth, known_pairs):
    ax.set_facecolor(TOKENS["panel"])
    for pair in known_pairs:
        ax.plot(
            [truth[pair.anchor_a_id][0], truth[pair.anchor_b_id][0]],
            [truth[pair.anchor_a_id][1], truth[pair.anchor_b_id][1]],
            color=NEUTRAL["base"],
            linewidth=0.8,
            alpha=0.7,
            zorder=1,
        )
    ax.scatter(
        [x for x, _ in truth.values()],
        [y for _, y in truth.values()],
        s=42,
        color=GOLD["base"],
        edgecolors=GOLD["dark"],
        linewidths=0.9,
        zorder=3,
    )
    format_axis(ax, truth, truth)
    ax.set_title("Ground truth known graph\n25 anchors, local <=8 m links", loc="left", fontsize=10, fontweight="bold", color=TOKENS["ink"])


def format_axis(ax, truth, estimate):
    xs = [x for x, _ in truth.values()] + [x for x, _ in estimate.values()]
    ys = [y for _, y in truth.values()] + [y for _, y in estimate.values()]
    span = max(max(xs) - min(xs), max(ys) - min(ys), 1.0)
    pad = span * 0.10
    ax.set_xlim(min(xs) - pad, max(xs) + pad)
    ax.set_ylim(min(ys) - pad, max(ys) + pad)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, color=TOKENS["grid"], linewidth=0.55)
    ax.tick_params(labelsize=7, colors=TOKENS["muted"], length=0)
    for spine in ax.spines.values():
        spine.set_color(TOKENS["axis"])


def write_csv(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    device = dc.choose_device("cuda" if torch.cuda.is_available() else "cpu")
    truth = grid_example.make_irregular_5x5(20260626)
    batch = grid_example.graph_batch_from_positions(truth, device=device, seed=20260626 + 17)
    known_pairs = dc.known_pairs_from_batch(batch, 0)

    started = time.perf_counter()
    normal = best_normal_distance_solve(known_pairs, seed_count=12)
    current = solve_current_production(known_pairs)
    ml_completed = solve_ml_completed(batch, known_pairs, device=device)
    print(f"baselines done in {time.perf_counter() - started:.1f}s", flush=True)

    rest_lengths = [5.25, 5.75, 6.25, 6.75, 7.25]
    sigmas = [0.12, 0.25, 0.45, 0.75]
    expand_factors = [1.00, 1.15, 1.35]
    cycles_list = [1, 2, 4, 6]
    rows: list[dict[str, float | int | str]] = []
    best = None
    total = len(rest_lengths) * len(sigmas) * len(expand_factors) * len(cycles_list)
    index = 0
    for rest_length in rest_lengths:
        for sigma in sigmas:
            for expand_factor in expand_factors:
                for cycles in cycles_list:
                    index += 1
                    positions = explosion_solve(
                        known_pairs,
                        cycles=cycles,
                        k=3,
                        rest_length_m=rest_length,
                        sigma_m=sigma,
                        expand_factor=expand_factor,
                    )
                    m = layout_metrics(truth, positions, known_pairs)
                    row = {
                        "method": "explosion",
                        "cycles": cycles,
                        "k": 3,
                        "rest_length_m": rest_length,
                        "sigma_m": sigma,
                        "expand_factor": expand_factor,
                        **m,
                    }
                    rows.append(row)
                    if best is None or m["max_offset_m"] < best[0]["max_offset_m"]:
                        best = (row, positions)
                    if index % 20 == 0:
                        print(f"sweep {index}/{total}: best_max={best[0]['max_offset_m']:.3f}m", flush=True)
    assert best is not None
    best_row, best_positions = best

    for name, positions in [
        ("normal_distance_only", normal),
        ("current_solver", current),
        ("ml_completed", ml_completed),
        ("best_explosion", best_positions),
    ]:
        row = {
            "method": name,
            "cycles": "",
            "k": "",
            "rest_length_m": "",
            "sigma_m": "",
            "expand_factor": "",
            **layout_metrics(truth, positions, known_pairs),
        }
        rows.append(row)

    plt.rcParams.update(
        {
            "figure.facecolor": TOKENS["surface"],
            "savefig.facecolor": TOKENS["surface"],
            "font.family": ["Segoe UI", "DejaVu Sans", "Arial", "sans-serif"],
        }
    )
    fig = plt.figure(figsize=(18, 9.5), dpi=185)
    gs = fig.add_gridspec(2, 3, hspace=0.34, wspace=0.24)
    fig.text(0.035, 0.975, "Explosion Solver Prototype on Normal 5x5 Grid", ha="left", va="top", fontsize=20, fontweight="bold", color=TOKENS["ink"])
    fig.text(
        0.035,
        0.94,
        f"Cycle: known-range LM, nearest-3 temporary uniform springs, radial expansion, augmented LM, then real-range polish. Best sweep: cycles={best_row['cycles']}, rest={best_row['rest_length_m']:.2f}m, sigma={best_row['sigma_m']:.2f}m, expand={best_row['expand_factor']:.2f}.",
        ha="left",
        va="top",
        fontsize=9.3,
        color=TOKENS["muted"],
    )
    draw_truth(fig.add_subplot(gs[0, 0]), truth, known_pairs)
    draw_panel(fig.add_subplot(gs[0, 1]), truth, normal, known_pairs, "Normal distance-only LM", NEUTRAL["base"])
    draw_panel(fig.add_subplot(gs[0, 2]), truth, current, known_pairs, "Current solver", GOLD["base"])
    draw_panel(fig.add_subplot(gs[1, 0]), truth, best_positions, known_pairs, "Best explosion sweep", PINK["base"])
    draw_panel(fig.add_subplot(gs[1, 1]), truth, ml_completed, known_pairs, "ML completed distances", BLUE["base"])

    ax = fig.add_subplot(gs[1, 2])
    ax.set_facecolor(TOKENS["panel"])
    scatter_x = [float(row["rest_length_m"]) for row in rows if row["method"] == "explosion"]
    scatter_y = [float(row["max_offset_m"]) for row in rows if row["method"] == "explosion"]
    colors = [float(row["cycles"]) for row in rows if row["method"] == "explosion"]
    sc = ax.scatter(scatter_x, scatter_y, c=colors, cmap="viridis", s=28, edgecolors=TOKENS["ink"], linewidths=0.25, alpha=0.78)
    ax.axhline(layout_metrics(truth, current, known_pairs)["max_offset_m"], color=GOLD["dark"], linewidth=1.2, label="current")
    ax.axhline(layout_metrics(truth, ml_completed, known_pairs)["max_offset_m"], color=BLUE["dark"], linewidth=1.2, label="ML")
    ax.set_xlabel("temporary spring rest length (m)", fontsize=8, color=TOKENS["muted"])
    ax.set_ylabel("max offset after polish (m)", fontsize=8, color=TOKENS["muted"])
    ax.set_title("Explosion sweep results\ncolor = cycle count", loc="left", fontsize=10, fontweight="bold", color=TOKENS["ink"])
    ax.grid(True, color=TOKENS["grid"], linewidth=0.55)
    ax.legend(frameon=False, fontsize=8)
    fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
    for spine in ax.spines.values():
        spine.set_color(TOKENS["axis"])

    png_path = OUTPUTS / "anchor_solver_explosion_5x5.png"
    csv_path = OUTPUTS / "anchor_solver_explosion_5x5.csv"
    fig.savefig(png_path, bbox_inches="tight", dpi=185)
    plt.close(fig)
    write_csv(csv_path, rows)

    print("summary:")
    for name, positions in [
        ("normal_distance_only", normal),
        ("current_solver", current),
        ("best_explosion", best_positions),
        ("ml_completed", ml_completed),
    ]:
        m = layout_metrics(truth, positions, known_pairs)
        print(f"{name}: max={m['max_offset_m']:.3f}m median={m['median_offset_m']:.3f}m rmse={m['known_rmse_m']:.4f}m")
    print(f"Wrote {png_path}")
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
