from __future__ import annotations

import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import importlib.util
import math
import os
from pathlib import Path
import random
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
    _anchor_ids,
    _circle_seed,
    _degree_shell_seed,
    _layout_priors,
    _layout_scale,
    _local_minimize,
    _positions_to_params,
    _preprocess_pairs,
    _random_seed,
    _spring_energy,
    _triangulated_seed,
    _validate_connected,
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

SEED_MODES = ["triangulated", "circle", "degree-shell", "corner-frame", "random", "mixed"]
SPACING_SIGMAS = [0.0, 0.75, 0.35, 0.10]
BOUNDARY_SIGMAS = [0.0, 20.0, 12.0, 6.0]
MIN_ANCHOR_SPACING_M = 2.0
MEASUREMENT = "exact"
HOPS = 10
HOP_MULTIPLIER = 0.75
SWAP_PROBABILITY = 0.85
MAX_ITERATIONS = 60
MAX_WORKERS = max(1, min(os.cpu_count() or 4, 8))


@dataclass(frozen=True)
class SweepPoint:
    seed_mode: str
    spacing_sigma_m: float
    boundary_sigma_m: float


def stable_seed(point: SweepPoint) -> int:
    text = f"{point.seed_mode}:{point.spacing_sigma_m}:{point.boundary_sigma_m}"
    return large_swap.RNG_SEED + sum((idx + 1) * ord(char) for idx, char in enumerate(text))


def anchor_order(case, processed, seed_mode: str) -> list[str]:
    if seed_mode == "corner-frame":
        return large_swap.anchor_order(case, processed, corner_seed=True)
    return _anchor_ids(processed)


def corner_frame_seed(case, parameterization) -> list[float]:
    width = large_swap.NOMINAL_GRID_SPACING_M * (case.cols - 1)
    height = large_swap.NOMINAL_GRID_SPACING_M * (case.rows - 1)
    positions: dict[str, tuple[float, float]] = {}
    for row in range(case.rows):
        for col in range(case.cols):
            anchor_id = f"A{row * case.cols + col:02d}"
            positions[anchor_id] = (
                width * col / max(case.cols - 1, 1),
                height * row / max(case.rows - 1, 1),
            )
    return _positions_to_params(parameterization, positions)


def initial_seeds(case, parameterization, processed, scale: float, point: SweepPoint, rng: random.Random) -> list[list[float]]:
    if point.seed_mode == "triangulated":
        return [_triangulated_seed(parameterization, processed, scale)]
    if point.seed_mode == "circle":
        return [
            _circle_seed(parameterization, scale, flip_y=False),
            _circle_seed(parameterization, scale, flip_y=True),
        ]
    if point.seed_mode == "degree-shell":
        return [_degree_shell_seed(parameterization, processed, scale)]
    if point.seed_mode == "corner-frame":
        return [corner_frame_seed(case, parameterization)]
    if point.seed_mode == "random":
        return [_random_seed(parameterization, scale, rng) for _ in range(4)]
    if point.seed_mode == "mixed":
        return [
            _triangulated_seed(parameterization, processed, scale),
            _circle_seed(parameterization, scale, flip_y=False),
            _circle_seed(parameterization, scale, flip_y=True),
            _degree_shell_seed(parameterization, processed, scale),
            _random_seed(parameterization, scale, rng),
            _random_seed(parameterization, scale, rng),
        ]
    raise ValueError(f"Unknown seed mode: {point.seed_mode}")


def swappable_anchor_ids(case, parameterization, seed_mode: str) -> list[str]:
    protected = {case.top_left, case.top_right, case.bottom_left} if seed_mode == "corner-frame" else set()
    return [
        anchor_id
        for anchor_id in parameterization.anchor_ids
        if anchor_id not in protected
        and parameterization.derivative_index(anchor_id, "x") is not None
        and parameterization.derivative_index(anchor_id, "y") is not None
    ]


def swapped_params(params, parameterization, rng: random.Random, allowed_ids: list[str]) -> list[float]:
    if len(allowed_ids) < 2:
        return list(params)
    anchor_a, anchor_b = rng.sample(allowed_ids, 2)
    positions = parameterization.to_positions(params)
    positions[anchor_a], positions[anchor_b] = positions[anchor_b], positions[anchor_a]
    return _positions_to_params(parameterization, positions)


def proposed_params(params, parameterization, rng: random.Random, hop_sigma: float, allowed_ids: list[str]) -> list[float]:
    proposal = [value + rng.gauss(0.0, hop_sigma) for value in params]
    if allowed_ids and rng.random() < SWAP_PROBABILITY:
        proposal = swapped_params(proposal, parameterization, rng, allowed_ids)
    return proposal


def solve_point(point: SweepPoint) -> dict[str, float | int | str]:
    case = large_swap.make_case()
    pairs = case.exact_pairs if MEASUREMENT == "exact" else case.noisy_pairs
    processed = _preprocess_pairs(pairs, min_sigma_m=0.02, min_distance_m=0.05)
    order = anchor_order(case, processed, point.seed_mode)
    _validate_connected(_anchor_ids(processed), processed)
    parameterization = _Parameterization(order)
    scale = _layout_scale(processed)
    rng = random.Random(stable_seed(point))
    seeds = initial_seeds(case, parameterization, processed, scale, point, rng)
    min_spacing = 0.0 if point.spacing_sigma_m == 0.0 else MIN_ANCHOR_SPACING_M
    boundary_sigma = 0.0 if point.boundary_sigma_m == 0.0 else point.boundary_sigma_m
    priors = _layout_priors(
        order,
        processed,
        scale=scale,
        min_anchor_spacing_m=min_spacing,
        anchor_spacing_sigma_m=max(point.spacing_sigma_m, 1e-6),
        unmeasured_pair_min_distance_m=0.0,
        unmeasured_pair_sigma_m=1.0,
        boundary_degree_prior_sigma_m=boundary_sigma,
    )
    allowed_swaps = swappable_anchor_ids(case, parameterization, point.seed_mode)
    hop_sigma = max(scale * HOP_MULTIPLIER, 0.05)
    temperature = max(scale * scale * 1e-5, 1e-8)
    best_params = None
    best_prior_energy = math.inf
    accepted_hops = 0

    for seed_params in seeds:
        current_params, current_energy = _local_minimize(
            seed_params,
            parameterization,
            processed,
            priors,
            max_iterations=MAX_ITERATIONS,
        )
        if current_energy < best_prior_energy:
            best_params = current_params
            best_prior_energy = current_energy
        for _ in range(HOPS):
            candidate_start = proposed_params(current_params, parameterization, rng, hop_sigma, allowed_swaps)
            candidate_params, candidate_energy = _local_minimize(
                candidate_start,
                parameterization,
                processed,
                priors,
                max_iterations=MAX_ITERATIONS,
            )
            accept = candidate_energy <= current_energy
            if not accept:
                probability = math.exp(max(min((current_energy - candidate_energy) / temperature, 0.0), -60.0))
                accept = rng.random() < probability
            if accept:
                current_params = candidate_params
                current_energy = candidate_energy
                accepted_hops += 1
            if candidate_energy < best_prior_energy:
                best_params = candidate_params
                best_prior_energy = candidate_energy

    if best_params is None:
        raise ValueError("No solution")

    polished_params, distance_energy = _local_minimize(
        best_params,
        parameterization,
        processed,
        None,
        max_iterations=MAX_ITERATIONS,
    )
    positions = parameterization.to_positions(polished_params)
    residuals = pair_residuals(positions, processed)
    pair_rmse = math.sqrt(sum(value * value for value in residuals.values()) / len(residuals))
    max_pair = max(abs(value) for value in residuals.values())
    max_offset, median_offset, p95_offset = pilot.position_error_summary(case.positions, positions)
    return {
        "measurement": MEASUREMENT,
        "seed_mode": point.seed_mode,
        "spacing_sigma_m": point.spacing_sigma_m,
        "boundary_sigma_m": point.boundary_sigma_m,
        "min_anchor_spacing_m": min_spacing,
        "hops": HOPS,
        "hop_multiplier": HOP_MULTIPLIER,
        "swap_probability": SWAP_PROBABILITY,
        "accepted_hops": accepted_hops,
        "seed_count": len(seeds),
        "anchor_count": len(case.positions),
        "pair_count": len(pairs),
        "pair_rmse_m": pair_rmse,
        "max_pair_residual_m": max_pair,
        "max_coord_offset_m": max_offset,
        "median_coord_offset_m": median_offset,
        "p95_coord_offset_m": p95_offset,
        "prior_energy": best_prior_energy,
        "distance_energy": distance_energy,
    }


def all_points() -> list[SweepPoint]:
    return [
        SweepPoint(seed_mode, spacing_sigma, boundary_sigma)
        for seed_mode in SEED_MODES
        for spacing_sigma in SPACING_SIGMAS
        for boundary_sigma in BOUNDARY_SIGMAS
    ]


def write_csv(rows: list[dict[str, float | int | str]], path: Path) -> None:
    fields = [
        "measurement",
        "seed_mode",
        "spacing_sigma_m",
        "boundary_sigma_m",
        "min_anchor_spacing_m",
        "hops",
        "hop_multiplier",
        "swap_probability",
        "accepted_hops",
        "seed_count",
        "anchor_count",
        "pair_count",
        "pair_rmse_m",
        "max_pair_residual_m",
        "max_coord_offset_m",
        "median_coord_offset_m",
        "p95_coord_offset_m",
        "prior_energy",
        "distance_energy",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fields})


def write_summary(rows: list[dict[str, float | int | str]], path: Path) -> None:
    ordered = sorted(rows, key=lambda row: (float(row["max_coord_offset_m"]), float(row["pair_rmse_m"])))
    lines = [
        "# Seed And Loss Parameter Grid Search",
        "",
        "Mode: deterministic irregular 4x4 grid, exact measured edges, swap proposals enabled.",
        "Missing-edge prior is disabled to isolate the close-anchor and low-degree-to-edge penalties.",
        f"Fixed search: hops={HOPS}, hop multiplier={HOP_MULTIPLIER}, swap probability={SWAP_PROBABILITY}, max iterations={MAX_ITERATIONS}.",
        "",
        "## Top 20",
        "",
        "| rank | seed mode | close sigma | edge sigma | max offset | pair RMSE |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for rank, row in enumerate(ordered[:20], start=1):
        close = "off" if float(row["spacing_sigma_m"]) == 0.0 else f"{float(row['spacing_sigma_m']):.2f}"
        edge = "off" if float(row["boundary_sigma_m"]) == 0.0 else f"{float(row['boundary_sigma_m']):.1f}"
        lines.append(
            f"| {rank} | {row['seed_mode']} | {close} | {edge} | "
            f"{float(row['max_coord_offset_m']):.3f} m | {float(row['pair_rmse_m']):.6f} m |"
        )
    lines.extend(["", "## Best By Seed Mode", ""])
    lines.append("| seed mode | close sigma | edge sigma | max offset | pair RMSE |")
    lines.append("|---|---:|---:|---:|---:|")
    for seed_mode in SEED_MODES:
        best = min(
            [row for row in rows if row["seed_mode"] == seed_mode],
            key=lambda row: (float(row["max_coord_offset_m"]), float(row["pair_rmse_m"])),
        )
        close = "off" if float(best["spacing_sigma_m"]) == 0.0 else f"{float(best['spacing_sigma_m']):.2f}"
        edge = "off" if float(best["boundary_sigma_m"]) == 0.0 else f"{float(best['boundary_sigma_m']):.1f}"
        lines.append(
            f"| {seed_mode} | {close} | {edge} | {float(best['max_coord_offset_m']):.3f} m | {float(best['pair_rmse_m']):.6f} m |"
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def render_heatmap(rows: list[dict[str, float | int | str]], path: Path) -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": TOKENS["surface"],
            "savefig.facecolor": TOKENS["surface"],
            "font.family": ["Segoe UI", "DejaVu Sans", "Arial", "sans-serif"],
        }
    )
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 9.4), dpi=180)
    fig.patch.set_facecolor(TOKENS["surface"])
    fig.text(
        0.035,
        0.972,
        "Seed type and loss-parameter sweep",
        ha="left",
        va="top",
        fontsize=21,
        fontweight="bold",
        color=TOKENS["ink"],
    )
    fig.text(
        0.035,
        0.934,
        "Cell value is worst labeled-anchor coordinate offset in meters. Close sigma controls the 2 m minimum-spacing penalty; edge sigma controls the low-degree outward push. Lower sigma is stronger; off disables the term.",
        ha="left",
        va="top",
        fontsize=9.5,
        color=TOKENS["muted"],
    )
    lookup = {
        (row["seed_mode"], float(row["spacing_sigma_m"]), float(row["boundary_sigma_m"])): float(row["max_coord_offset_m"])
        for row in rows
    }
    values = np.array([float(row["max_coord_offset_m"]) for row in rows], dtype=float)
    vmax = max(float(np.quantile(values, 0.90)), 1.0)
    x_labels = ["off" if value == 0.0 else f"{value:.1f}" for value in BOUNDARY_SIGMAS]
    y_labels = ["off" if value == 0.0 else f"{value:.2f}" for value in SPACING_SIGMAS]
    last_image = None
    for ax, seed_mode in zip(axes.ravel(), SEED_MODES):
        matrix = np.array(
            [
                [
                    lookup[(seed_mode, spacing_sigma, boundary_sigma)]
                    for boundary_sigma in BOUNDARY_SIGMAS
                ]
                for spacing_sigma in SPACING_SIGMAS
            ],
            dtype=float,
        )
        last_image = ax.imshow(matrix, cmap="YlOrRd", vmin=0.0, vmax=vmax, aspect="auto")
        ax.set_title(seed_mode, fontsize=11, fontweight="semibold", color=TOKENS["ink"])
        ax.set_xticks(range(len(BOUNDARY_SIGMAS)), x_labels)
        ax.set_yticks(range(len(SPACING_SIGMAS)), y_labels)
        ax.set_xlabel("edge sigma (m)", fontsize=8, color=TOKENS["muted"])
        ax.set_ylabel("close sigma (m)", fontsize=8, color=TOKENS["muted"])
        ax.tick_params(labelsize=8, colors=TOKENS["muted"], length=0)
        for y in range(matrix.shape[0]):
            for x in range(matrix.shape[1]):
                value = matrix[y, x]
                label_color = TOKENS["panel"] if value > vmax * 0.55 else TOKENS["ink"]
                ax.text(x, y, f"{value:.2f}", ha="center", va="center", fontsize=7.4, color=label_color)
        for spine in ax.spines.values():
            spine.set_color(TOKENS["axis"])
    assert last_image is not None
    cbar = fig.colorbar(last_image, ax=axes.ravel().tolist(), shrink=0.78, pad=0.014)
    cbar.ax.tick_params(labelsize=8, colors=TOKENS["muted"])
    cbar.set_label("max coordinate offset (m)", color=TOKENS["muted"], fontsize=8)
    fig.subplots_adjust(left=0.07, right=0.915, top=0.875, bottom=0.08, hspace=0.36, wspace=0.28)
    fig.savefig(path, bbox_inches="tight", dpi=180)
    plt.close(fig)


def main() -> None:
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    points = all_points()
    print(f"Running {len(points)} seed/loss grid points with {MAX_WORKERS} workers", flush=True)
    rows: list[dict[str, float | int | str]] = []
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_map = {executor.submit(solve_point, point): point for point in points}
        for future in as_completed(future_map):
            point = future_map[future]
            row = future.result()
            rows.append(row)
            print(
                f"{point.seed_mode} close={point.spacing_sigma_m:.2f} edge={point.boundary_sigma_m:.1f}: "
                f"offset={float(row['max_coord_offset_m']):.3f} rmse={float(row['pair_rmse_m']):.6f}",
                flush=True,
            )
    rows.sort(
        key=lambda row: (
            SEED_MODES.index(str(row["seed_mode"])),
            float(row["spacing_sigma_m"]),
            float(row["boundary_sigma_m"]),
        )
    )
    csv_path = OUTPUTS / "anchor_solver_seed_loss_grid_search.csv"
    summary_path = OUTPUTS / "anchor_solver_seed_loss_grid_search.md"
    png_path = OUTPUTS / "anchor_solver_seed_loss_grid_search.png"
    write_csv(rows, csv_path)
    write_summary(rows, summary_path)
    render_heatmap(rows, png_path)
    best = min(rows, key=lambda row: (float(row["max_coord_offset_m"]), float(row["pair_rmse_m"])))
    print(
        f"BEST {best['seed_mode']} close={best['spacing_sigma_m']} edge={best['boundary_sigma_m']} "
        f"offset={float(best['max_coord_offset_m']):.3f} rmse={float(best['pair_rmse_m']):.6f}",
        flush=True,
    )
    print(f"Wrote {csv_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {png_path}")


if __name__ == "__main__":
    main()
