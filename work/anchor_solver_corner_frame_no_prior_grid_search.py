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
    _circle_seed,
    _degree_shell_seed,
    _random_seed,
    _triangulated_seed,
)


pilot = large_swap.pilot
TOKENS = large_swap.TOKENS
BLUE = large_swap.BLUE
GOLD = large_swap.GOLD
ORANGE = large_swap.ORANGE
OLIVE = large_swap.OLIVE
PINK = large_swap.PINK
NEUTRAL = large_swap.NEUTRAL


SEED_COUNTS = [8, 16, 24]
BASIN_HOPS = [4, 10, 20]
HOP_MULTIPLIERS = [0.75, 1.5, 3.0, 5.0]
SWAP_PROBABILITIES = [0.50, 0.85]
MAX_ITERATIONS = 60
MEASUREMENT = "exact"
MAX_WORKERS = max(1, min(os.cpu_count() or 4, 8))


@dataclass(frozen=True)
class GridPoint:
    seed_count: int
    basin_hops: int
    hop_multiplier: float
    swap_probability: float


def old_initial_parameters(
    parameterization,
    pairs,
    *,
    seed_count: int,
    scale: float,
    rng: random.Random,
) -> list[list[float]]:
    seeds = [
        _triangulated_seed(parameterization, pairs, scale),
        _circle_seed(parameterization, scale, flip_y=False),
        _circle_seed(parameterization, scale, flip_y=True),
        _degree_shell_seed(parameterization, pairs, scale),
    ]
    while len(seeds) < seed_count:
        seeds.append(_random_seed(parameterization, scale, rng))
    return seeds[:seed_count]


def stable_seed(point: GridPoint) -> int:
    return (
        large_swap.RNG_SEED
        + point.seed_count * 101
        + point.basin_hops * 1009
        + int(point.hop_multiplier * 100) * 9173
        + int(point.swap_probability * 100) * 13
    )


def proposed_params(
    params,
    parameterization,
    rng: random.Random,
    hop_sigma: float,
    allowed_ids: list[str],
    swap_probability: float,
) -> list[float]:
    proposal = [value + rng.gauss(0.0, hop_sigma) for value in params]
    if allowed_ids and rng.random() < swap_probability:
        proposal = large_swap.swapped_params(proposal, parameterization, rng, allowed_ids)
    return proposal


def solve_grid_point(point: GridPoint) -> dict[str, float | int | str]:
    case = large_swap.make_case()
    pairs = case.exact_pairs if MEASUREMENT == "exact" else case.noisy_pairs
    processed = pilot._preprocess_pairs(pairs, min_sigma_m=0.02, min_distance_m=0.05)
    order = large_swap.anchor_order(case, processed, corner_seed=True)
    pilot._validate_connected(pilot._anchor_ids(processed), processed)
    parameterization = pilot._Parameterization(order)
    scale = pilot._layout_scale(processed)
    rng = random.Random(stable_seed(point))

    seeds = [large_swap.corner_seed_params(case, parameterization)]
    seeds.extend(
        old_initial_parameters(
            parameterization,
            processed,
            seed_count=max(point.seed_count - 1, 1),
            scale=scale,
            rng=rng,
        )
    )
    seeds = seeds[: point.seed_count]

    allowed_swaps = large_swap.swappable_anchor_ids(case, parameterization, corner_seed=True)
    hop_sigma = max(scale * point.hop_multiplier, 0.05)
    temperature = max(scale * scale * 1e-5, 1e-8)
    best_params = None
    best_energy = math.inf
    accepted_hops = 0

    for seed_params in seeds:
        current_params, current_energy = large_swap.local_minimize(
            seed_params,
            parameterization,
            processed,
            {},
            MAX_ITERATIONS,
        )
        if current_energy < best_energy:
            best_params = current_params
            best_energy = current_energy

        for _ in range(max(point.basin_hops, 0)):
            candidate_start = proposed_params(
                current_params,
                parameterization,
                rng,
                hop_sigma,
                allowed_swaps,
                point.swap_probability,
            )
            candidate_params, candidate_energy = large_swap.local_minimize(
                candidate_start,
                parameterization,
                processed,
                {},
                MAX_ITERATIONS,
            )
            accept = candidate_energy <= current_energy
            if not accept:
                probability = math.exp(max(min((current_energy - candidate_energy) / temperature, 0.0), -60.0))
                accept = rng.random() < probability
            if accept:
                current_params = candidate_params
                current_energy = candidate_energy
                accepted_hops += 1
            if candidate_energy < best_energy:
                best_params = candidate_params
                best_energy = candidate_energy

    if best_params is None:
        raise ValueError("No solution")

    positions = parameterization.to_positions(best_params)
    residuals = pilot.pair_residuals(positions, processed)
    pair_rmse = math.sqrt(sum(value * value for value in residuals.values()) / len(residuals))
    max_pair = max(abs(value) for value in residuals.values())
    max_offset, median_offset, p95_offset = pilot.position_error_summary(case.positions, positions)
    return {
        "measurement": MEASUREMENT,
        "seed_count": point.seed_count,
        "basin_hops": point.basin_hops,
        "hop_multiplier": point.hop_multiplier,
        "swap_probability": point.swap_probability,
        "accepted_hops": accepted_hops,
        "anchor_count": len(case.positions),
        "pair_count": len(pairs),
        "pair_rmse_m": pair_rmse,
        "max_pair_residual_m": max_pair,
        "max_coord_offset_m": max_offset,
        "median_coord_offset_m": median_offset,
        "p95_coord_offset_m": p95_offset,
        "energy": best_energy,
    }


def all_points() -> list[GridPoint]:
    return [
        GridPoint(seed_count, basin_hops, hop_multiplier, swap_probability)
        for seed_count in SEED_COUNTS
        for basin_hops in BASIN_HOPS
        for hop_multiplier in HOP_MULTIPLIERS
        for swap_probability in SWAP_PROBABILITIES
    ]


def write_csv(rows: list[dict[str, float | int | str]], path: Path) -> None:
    fields = [
        "measurement",
        "seed_count",
        "basin_hops",
        "hop_multiplier",
        "swap_probability",
        "accepted_hops",
        "anchor_count",
        "pair_count",
        "pair_rmse_m",
        "max_pair_residual_m",
        "max_coord_offset_m",
        "median_coord_offset_m",
        "p95_coord_offset_m",
        "energy",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fields})


def write_summary(rows: list[dict[str, float | int | str]], path: Path) -> None:
    ordered = sorted(rows, key=lambda row: (float(row["max_coord_offset_m"]), float(row["pair_rmse_m"])))
    best = ordered[0]
    lines = [
        "# Corner-Frame Swap Parameter Grid Search",
        "",
        "Mode: corner-framed seed/order, swap proposals enabled, no coordinate/corner prior, no spacing prior, no degree prior.",
        f"Measurement: {MEASUREMENT}. Case: deterministic irregular 4x4 grid from `anchor_solver_large_hop_swap_sweep.py`.",
        f"Grid size: {len(rows)} parameter combinations.",
        "",
        "## Best Combination",
        "",
        f"- seed_count: {best['seed_count']}",
        f"- basin_hops: {best['basin_hops']}",
        f"- hop_multiplier: {best['hop_multiplier']}",
        f"- swap_probability: {best['swap_probability']}",
        f"- max coordinate offset: {float(best['max_coord_offset_m']):.3f} m",
        f"- pair RMSE: {float(best['pair_rmse_m']):.6f} m",
        f"- max pair residual: {float(best['max_pair_residual_m']):.6f} m",
        "",
        "## Top 12",
        "",
        "| rank | seeds | hops | hop multiplier | swap probability | max offset | pair RMSE |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for rank, row in enumerate(ordered[:12], start=1):
        lines.append(
            f"| {rank} | {row['seed_count']} | {row['basin_hops']} | {float(row['hop_multiplier']):.2f} | "
            f"{float(row['swap_probability']):.2f} | {float(row['max_coord_offset_m']):.3f} m | "
            f"{float(row['pair_rmse_m']):.6f} m |"
        )
    lines.extend(
        [
            "",
            "Interpretation: if pair RMSE is near zero but max coordinate offset is large, the search found a distance-valid but incorrectly labeled grid arrangement.",
        ]
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
    fig, axes = plt.subplots(len(SEED_COUNTS), len(SWAP_PROBABILITIES), figsize=(13.5, 10.5), dpi=180)
    fig.patch.set_facecolor(TOKENS["surface"])
    fig.text(
        0.035,
        0.972,
        "Corner-frame swaps without priors: parameter grid search",
        ha="left",
        va="top",
        fontsize=20,
        fontweight="bold",
        color=TOKENS["ink"],
    )
    fig.text(
        0.035,
        0.936,
        "Cell values are worst labeled-anchor coordinate offset in meters after no-scale alignment. Objective is distance-only; corner roles only set the frame/seed and protect those anchors from swaps.",
        ha="left",
        va="top",
        fontsize=9.5,
        color=TOKENS["muted"],
    )

    value_lookup = {
        (
            int(row["seed_count"]),
            float(row["swap_probability"]),
            int(row["basin_hops"]),
            float(row["hop_multiplier"]),
        ): float(row["max_coord_offset_m"])
        for row in rows
    }
    all_values = np.array([float(row["max_coord_offset_m"]) for row in rows], dtype=float)
    vmax = max(float(np.quantile(all_values, 0.90)), 1.0)
    for row_index, seed_count in enumerate(SEED_COUNTS):
        for col_index, swap_probability in enumerate(SWAP_PROBABILITIES):
            ax = axes[row_index, col_index]
            matrix = np.array(
                [
                    [
                        value_lookup[(seed_count, swap_probability, basin_hops, hop_multiplier)]
                        for hop_multiplier in HOP_MULTIPLIERS
                    ]
                    for basin_hops in BASIN_HOPS
                ],
                dtype=float,
            )
            image = ax.imshow(matrix, cmap="YlOrRd", vmin=0.0, vmax=vmax, aspect="auto")
            ax.set_title(
                f"{seed_count} seeds, swap p={swap_probability:.2f}",
                fontsize=10,
                fontweight="semibold",
                color=TOKENS["ink"],
            )
            ax.set_xticks(range(len(HOP_MULTIPLIERS)), [f"{value:.2g}" for value in HOP_MULTIPLIERS])
            ax.set_yticks(range(len(BASIN_HOPS)), [str(value) for value in BASIN_HOPS])
            ax.set_xlabel("hop multiplier", fontsize=8, color=TOKENS["muted"])
            ax.set_ylabel("basin hops", fontsize=8, color=TOKENS["muted"])
            ax.tick_params(labelsize=8, colors=TOKENS["muted"], length=0)
            for y in range(matrix.shape[0]):
                for x in range(matrix.shape[1]):
                    value = matrix[y, x]
                    color = TOKENS["panel"] if value > vmax * 0.55 else TOKENS["ink"]
                    ax.text(x, y, f"{value:.2f}", ha="center", va="center", fontsize=7.5, color=color)
            for spine in ax.spines.values():
                spine.set_color(TOKENS["axis"])
    cbar = fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.78, pad=0.015)
    cbar.ax.tick_params(labelsize=8, colors=TOKENS["muted"])
    cbar.set_label("max coordinate offset (m)", color=TOKENS["muted"], fontsize=8)
    fig.subplots_adjust(left=0.07, right=0.91, top=0.88, bottom=0.08, hspace=0.42, wspace=0.24)
    fig.savefig(path, bbox_inches="tight", dpi=180)
    plt.close(fig)


def main() -> None:
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    points = all_points()
    print(f"Running {len(points)} grid points with {MAX_WORKERS} workers", flush=True)
    rows: list[dict[str, float | int | str]] = []
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_map = {executor.submit(solve_grid_point, point): point for point in points}
        for future in as_completed(future_map):
            point = future_map[future]
            row = future.result()
            rows.append(row)
            print(
                f"seeds={point.seed_count} hops={point.basin_hops} "
                f"mult={point.hop_multiplier:.2f} swap={point.swap_probability:.2f}: "
                f"offset={float(row['max_coord_offset_m']):.3f} rmse={float(row['pair_rmse_m']):.6f}",
                flush=True,
            )

    rows.sort(
        key=lambda row: (
            int(row["seed_count"]),
            float(row["swap_probability"]),
            int(row["basin_hops"]),
            float(row["hop_multiplier"]),
        )
    )
    csv_path = OUTPUTS / "anchor_solver_corner_frame_no_prior_grid_search.csv"
    summary_path = OUTPUTS / "anchor_solver_corner_frame_no_prior_grid_search.md"
    png_path = OUTPUTS / "anchor_solver_corner_frame_no_prior_grid_search.png"
    write_csv(rows, csv_path)
    write_summary(rows, summary_path)
    render_heatmap(rows, png_path)
    best = min(rows, key=lambda row: (float(row["max_coord_offset_m"]), float(row["pair_rmse_m"])))
    print(
        "BEST "
        f"seeds={best['seed_count']} hops={best['basin_hops']} "
        f"mult={float(best['hop_multiplier']):.2f} swap={float(best['swap_probability']):.2f} "
        f"offset={float(best['max_coord_offset_m']):.3f} rmse={float(best['pair_rmse_m']):.6f}",
        flush=True,
    )
    print(f"Wrote {csv_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {png_path}")


if __name__ == "__main__":
    main()
