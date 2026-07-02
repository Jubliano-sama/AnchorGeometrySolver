from __future__ import annotations

import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import math
import os
from pathlib import Path
import random
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT / "work" / "SmartClicker-GUI"
OUTPUTS = ROOT / "outputs"
sys.path.insert(0, str(REPO))

from uwb_capture.anchor_geometry import (  # noqa: E402
    ANCHOR_LAYOUT_ALGORITHM,
    AnchorLayoutResult,
    AnchorPairDistance,
    _Parameterization,
    _anchor_ids,
    _circle_seed,
    _layout_scale,
    _local_minimize,
    _preprocess_pairs,
    _random_seed,
    _spring_energy,
    _triangulated_seed,
    _validate_connected,
    pair_residuals,
    rotate_layout_to_level,
    solve_anchor_layout,
)


TOKENS = {
    "surface": "#FCFCFD",
    "panel": "#FFFFFF",
    "ink": "#1F2430",
    "muted": "#6F768A",
    "grid": "#E6E8F0",
    "axis": "#D7DBE7",
}
BLUE = {"xlight": "#EAF1FE", "light": "#CEDFFE", "base": "#A3BEFA", "mid": "#5477C4", "dark": "#2E4780"}
GOLD = {"xlight": "#FFF4C2", "light": "#FFEA8F", "base": "#FFE15B", "mid": "#B8A037", "dark": "#736422"}
ORANGE = {"xlight": "#FFEDDE", "light": "#FFBDA1", "base": "#F0986E", "mid": "#CC6F47", "dark": "#804126"}
OLIVE = {"xlight": "#D8ECBD", "light": "#BEEB96", "base": "#A3D576", "mid": "#71B436", "dark": "#386411"}
PINK = {"xlight": "#FCDAD6", "light": "#F5BACC", "base": "#F390CA", "mid": "#BD569B", "dark": "#8A3A6F"}
NEUTRAL = {"xlight": "#F4F5F7", "light": "#E2E5EA", "base": "#C5CAD3", "mid": "#7A828F", "dark": "#464C55"}


RNG_SEED = 20260626
CASE_COUNT_PER_FAMILY = 3
SEED_COUNT = 8
BASIN_HOPS = 2
MAX_ITERATIONS = 60
MAX_WORKERS = max(1, min(CASE_COUNT_PER_FAMILY * 2, os.cpu_count() or 4, 4))
EDGE_RADIUS_M = 8.0
PAIR_SIGMA_M = 0.05
NOISE_SIGMA_M = 0.03
NLOS_PROBABILITY = 1.0 / 3.0
NLOS_MAX_OFFSET_M = 0.20


@dataclass(frozen=True)
class LayoutCase:
    family: str
    trial_id: int
    true_positions: dict[str, tuple[float, float]]
    pairs: list[AnchorPairDistance]
    nlos_count: int
    discarded_before: int


@dataclass(frozen=True)
class SolvedCase:
    case: LayoutCase
    solver: str
    positions: dict[str, tuple[float, float]]
    pair_rmse_m: float
    max_pair_residual_m: float
    max_coord_offset_m: float
    median_coord_offset_m: float
    p95_coord_offset_m: float
    energy: float
    elapsed_s: float


def generate_random_positions(rng: random.Random, n: int) -> dict[str, tuple[float, float]]:
    points: list[tuple[float, float]] = []
    attempts = 0
    while len(points) < n and attempts < 50_000:
        attempts += 1
        candidate = (rng.uniform(0.0, 25.0), rng.uniform(0.0, 25.0))
        if all(math.hypot(candidate[0] - x, candidate[1] - y) >= 2.0 for x, y in points):
            points.append(candidate)
    if len(points) != n:
        raise RuntimeError("Could not place random anchors with the minimum spacing rule.")
    return {f"A{index:02d}": point for index, point in enumerate(points)}


def generate_grid_positions(rng: random.Random) -> dict[str, tuple[float, float]]:
    choices = [(4, 4), (4, 5), (5, 4), (4, 6), (6, 4), (5, 5), (5, 6), (6, 5)]
    rows, cols = rng.choice(choices)
    x_gaps = [rng.uniform(4.0, 8.0) for _ in range(cols - 1)]
    y_gaps = [rng.uniform(4.0, 8.0) for _ in range(rows - 1)]
    x = [0.0]
    y = [0.0]
    for gap in x_gaps:
        x.append(x[-1] + gap)
    for gap in y_gaps:
        y.append(y[-1] + gap)
    positions: dict[str, tuple[float, float]] = {}
    index = 0
    for row in range(rows):
        for col in range(cols):
            positions[f"A{index:02d}"] = (x[col], y[row])
            index += 1
    return positions


def generate_pairs(
    positions: dict[str, tuple[float, float]],
    rng: random.Random,
) -> tuple[list[AnchorPairDistance], int]:
    pairs: list[AnchorPairDistance] = []
    nlos_count = 0
    ids = sorted(positions)
    for i, anchor_a in enumerate(ids):
        ax, ay = positions[anchor_a]
        for anchor_b in ids[i + 1 :]:
            bx, by = positions[anchor_b]
            true_distance = math.hypot(ax - bx, ay - by)
            if true_distance > EDGE_RADIUS_M:
                continue
            measured = true_distance + rng.gauss(0.0, NOISE_SIGMA_M)
            source = "los"
            if rng.random() < NLOS_PROBABILITY:
                measured += rng.uniform(0.0, NLOS_MAX_OFFSET_M)
                nlos_count += 1
                source = "nlos"
            pairs.append(
                AnchorPairDistance(
                    anchor_a,
                    anchor_b,
                    max(measured, 0.05),
                    sigma_m=PAIR_SIGMA_M,
                    source=source,
                )
            )
    return pairs, nlos_count


def is_connected(anchor_ids: list[str], pairs: list[AnchorPairDistance]) -> bool:
    if not anchor_ids:
        return False
    neighbors = {anchor_id: set() for anchor_id in anchor_ids}
    for pair in pairs:
        neighbors[pair.anchor_a_id].add(pair.anchor_b_id)
        neighbors[pair.anchor_b_id].add(pair.anchor_a_id)
    seen = {anchor_ids[0]}
    queue = [anchor_ids[0]]
    while queue:
        current = queue.pop(0)
        for neighbor in neighbors[current]:
            if neighbor not in seen:
                seen.add(neighbor)
                queue.append(neighbor)
    return len(seen) == len(anchor_ids)


def rigidity_rank(
    positions: dict[str, tuple[float, float]],
    pairs: list[AnchorPairDistance],
) -> int:
    anchor_ids = sorted(positions)
    index = {anchor_id: offset for offset, anchor_id in enumerate(anchor_ids)}
    matrix = np.zeros((len(pairs), 2 * len(anchor_ids)), dtype=float)
    for row, pair in enumerate(pairs):
        a = index[pair.anchor_a_id]
        b = index[pair.anchor_b_id]
        ax, ay = positions[pair.anchor_a_id]
        bx, by = positions[pair.anchor_b_id]
        dx = ax - bx
        dy = ay - by
        matrix[row, 2 * a] = dx
        matrix[row, 2 * a + 1] = dy
        matrix[row, 2 * b] = -dx
        matrix[row, 2 * b + 1] = -dy
    return int(np.linalg.matrix_rank(matrix, tol=1e-7))


def is_three_vertex_connected(anchor_ids: list[str], pairs: list[AnchorPairDistance]) -> bool:
    if len(anchor_ids) <= 3:
        return True
    for first_index, first in enumerate(anchor_ids):
        for second in anchor_ids[first_index + 1 :]:
            kept = [anchor_id for anchor_id in anchor_ids if anchor_id not in {first, second}]
            kept_set = set(kept)
            kept_pairs = [
                pair
                for pair in pairs
                if pair.anchor_a_id in kept_set and pair.anchor_b_id in kept_set
            ]
            if kept and not is_connected(kept, kept_pairs):
                return False
    return True


def accepted_graph(positions: dict[str, tuple[float, float]], pairs: list[AnchorPairDistance]) -> bool:
    anchor_ids = sorted(positions)
    target_rank = 2 * len(anchor_ids) - 3
    if len(pairs) < target_rank or not is_connected(anchor_ids, pairs):
        return False
    return rigidity_rank(positions, pairs) >= target_rank


def make_case(family: str, trial_id: int, rng: random.Random) -> LayoutCase:
    discarded = 0
    while True:
        positions = (
            generate_random_positions(rng, rng.randint(16, 32))
            if family == "random"
            else generate_grid_positions(rng)
        )
        pairs, nlos_count = generate_pairs(positions, rng)
        if accepted_graph(positions, pairs):
            return LayoutCase(family, trial_id, positions, pairs, nlos_count, discarded)
        discarded += 1


def old_initial_parameters(
    parameterization: _Parameterization,
    pairs,
    scale: float,
    seed_count: int,
    rng: random.Random,
) -> list[list[float]]:
    seeds = [
        _triangulated_seed(parameterization, pairs, scale),
        _circle_seed(parameterization, scale, flip_y=False),
        _circle_seed(parameterization, scale, flip_y=True),
    ]
    while len(seeds) < seed_count:
        seeds.append(_random_seed(parameterization, scale, rng))
    return seeds[:seed_count]


def solve_old(case: LayoutCase, random_seed: int) -> tuple[dict[str, tuple[float, float]], float, float, float]:
    processed = _preprocess_pairs(case.pairs, min_sigma_m=0.02, min_distance_m=0.05)
    anchor_ids = _anchor_ids(processed)
    _validate_connected(anchor_ids, processed)
    scale = _layout_scale(processed)
    parameterization = _Parameterization(anchor_ids)
    rng = random.Random(random_seed)
    seeds = old_initial_parameters(parameterization, processed, scale, SEED_COUNT, rng)
    best_params: list[float] | None = None
    best_energy = math.inf
    temperature = max(scale * scale * 1e-5, 1e-8)
    hop_scale = max(scale * 0.35, 0.05)
    for seed_params in seeds:
        current_params, current_energy = _local_minimize(
            seed_params,
            parameterization,
            processed,
            max_iterations=MAX_ITERATIONS,
        )
        if current_energy < best_energy:
            best_params = current_params
            best_energy = current_energy
        for _ in range(max(BASIN_HOPS, 0)):
            hopped = [value + rng.gauss(0.0, hop_scale) for value in current_params]
            candidate_params, candidate_energy = _local_minimize(
                hopped,
                parameterization,
                processed,
                max_iterations=MAX_ITERATIONS,
            )
            accept = candidate_energy <= current_energy
            if not accept:
                probability = math.exp(max(min((current_energy - candidate_energy) / temperature, 0.0), -60.0))
                accept = rng.random() < probability
            if accept:
                current_params = candidate_params
                current_energy = candidate_energy
            if candidate_energy < best_energy:
                best_params = candidate_params
                best_energy = candidate_energy
    if best_params is None:
        raise ValueError("Old solver produced no parameters.")
    positions = parameterization.to_positions(best_params)
    positions = rotate_layout_to_level(positions, anchor_ids[0], anchor_ids[1])
    residuals = pair_residuals(positions, processed)
    pair_rmse = math.sqrt(sum(value * value for value in residuals.values()) / len(residuals))
    max_pair = max(abs(value) for value in residuals.values())
    energy = _spring_energy(best_params, parameterization, processed)
    return positions, energy, pair_rmse, max_pair


def align_positions(
    truth: dict[str, tuple[float, float]],
    estimate: dict[str, tuple[float, float]],
) -> dict[str, tuple[float, float]]:
    ids = sorted(set(truth) & set(estimate))
    target = np.array([truth[anchor_id] for anchor_id in ids], dtype=float)
    source = np.array([estimate[anchor_id] for anchor_id in ids], dtype=float)
    target_center = target.mean(axis=0)
    source_center = source.mean(axis=0)
    target_centered = target - target_center
    source_centered = source - source_center
    best_aligned = None
    best_max = math.inf
    for reflect_y in (1.0, -1.0):
        reflected = source_centered.copy()
        reflected[:, 1] *= reflect_y
        u, _s, vt = np.linalg.svd(reflected.T @ target_centered)
        transform = u @ vt
        aligned = reflected @ transform + target_center
        max_offset = float(np.max(np.linalg.norm(aligned - target, axis=1)))
        if max_offset < best_max:
            best_max = max_offset
            best_aligned = aligned
    assert best_aligned is not None
    return {anchor_id: tuple(point) for anchor_id, point in zip(ids, best_aligned)}


def offset_summary(
    truth: dict[str, tuple[float, float]],
    estimate: dict[str, tuple[float, float]],
) -> tuple[float, float, float]:
    aligned = align_positions(truth, estimate)
    offsets = np.array(
        [
            math.hypot(aligned[anchor_id][0] - truth[anchor_id][0], aligned[anchor_id][1] - truth[anchor_id][1])
            for anchor_id in sorted(aligned)
        ],
        dtype=float,
    )
    return float(offsets.max()), float(np.median(offsets)), float(np.quantile(offsets, 0.95))


def solve_case(case: LayoutCase, solver: str, solver_seed: int) -> SolvedCase:
    started = time.time()
    if solver == "old":
        positions, energy, pair_rmse, max_pair = solve_old(case, solver_seed)
    elif solver == "new":
        result: AnchorLayoutResult = solve_anchor_layout(
            case.pairs,
            seed_count=SEED_COUNT,
            basin_hops=BASIN_HOPS,
            max_iterations=MAX_ITERATIONS,
            random_seed=solver_seed,
        )
        positions = result.positions_m
        energy = result.energy
        pair_rmse = result.rmse_m
        max_pair = result.max_residual_m
    else:
        raise ValueError(f"Unknown solver: {solver}")
    elapsed = time.time() - started
    max_offset, median_offset, p95_offset = offset_summary(case.true_positions, positions)
    return SolvedCase(
        case=case,
        solver=solver,
        positions=positions,
        pair_rmse_m=pair_rmse,
        max_pair_residual_m=max_pair,
        max_coord_offset_m=max_offset,
        median_coord_offset_m=median_offset,
        p95_coord_offset_m=p95_offset,
        energy=energy,
        elapsed_s=elapsed,
    )


def solve_candidate(family: str, trial_id: int) -> tuple[LayoutCase, SolvedCase, SolvedCase]:
    rng = random.Random(RNG_SEED + (0 if family == "random" else 50_000) + trial_id * 997)
    case = make_case(family, trial_id, rng)
    seed = RNG_SEED + (0 if family == "random" else 10_000) + trial_id
    old_result = solve_case(case, "old", seed)
    new_result = solve_case(case, "new", seed)
    return case, old_result, new_result


def layout_limits(case: LayoutCase, solved: list[SolvedCase]) -> tuple[float, float, float, float]:
    xs = [x for x, _y in case.true_positions.values()]
    ys = [y for _x, y in case.true_positions.values()]
    for result in solved:
        aligned = align_positions(case.true_positions, result.positions)
        xs.extend(x for x, _y in aligned.values())
        ys.extend(y for _x, y in aligned.values())
    span = max(max(xs) - min(xs), max(ys) - min(ys), 1.0)
    pad = 0.10 * span
    return min(xs) - pad, max(xs) + pad, min(ys) - pad, max(ys) + pad


def draw_panel(ax, result: SolvedCase, limits, color, edge_color) -> None:
    case = result.case
    truth = case.true_positions
    solved = align_positions(truth, result.positions)
    ax.set_facecolor(TOKENS["panel"])
    for pair in case.pairs:
        ax.plot(
            [truth[pair.anchor_a_id][0], truth[pair.anchor_b_id][0]],
            [truth[pair.anchor_a_id][1], truth[pair.anchor_b_id][1]],
            color=NEUTRAL["light"],
            linewidth=0.55,
            alpha=0.55,
            zorder=1,
        )
    worst_anchor = max(
        solved,
        key=lambda anchor_id: math.hypot(solved[anchor_id][0] - truth[anchor_id][0], solved[anchor_id][1] - truth[anchor_id][1]),
    )
    for anchor_id, true_point in truth.items():
        solved_point = solved[anchor_id]
        is_worst = anchor_id == worst_anchor
        ax.plot(
            [true_point[0], solved_point[0]],
            [true_point[1], solved_point[1]],
            color=ORANGE["mid"] if not is_worst else PINK["dark"],
            linewidth=0.75 if not is_worst else 1.6,
            alpha=0.40 if not is_worst else 0.88,
            zorder=2,
        )
    ax.scatter(
        [x for x, _y in truth.values()],
        [y for _x, y in truth.values()],
        s=26,
        facecolors=TOKENS["panel"],
        edgecolors=NEUTRAL["dark"],
        linewidths=0.9,
        zorder=3,
        label="truth",
    )
    ax.scatter(
        [x for x, _y in solved.values()],
        [y for _x, y in solved.values()],
        s=24,
        facecolors=color,
        edgecolors=edge_color,
        linewidths=0.9,
        zorder=4,
        label="solved",
    )
    wx, wy = solved[worst_anchor]
    ax.text(
        wx,
        wy,
        f" {worst_anchor}",
        fontsize=6.8,
        color=PINK["dark"],
        va="center",
        ha="left",
        zorder=5,
    )
    ax.set_xlim(limits[0], limits[1])
    ax.set_ylim(limits[2], limits[3])
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, color=TOKENS["grid"], linewidth=0.6)
    ax.tick_params(labelsize=7, colors=TOKENS["muted"], length=0)
    for spine in ax.spines.values():
        spine.set_color(TOKENS["axis"])
    title = "Old solver" if result.solver == "old" else "New solver"
    detail = (
        f"worst offset {result.max_coord_offset_m:.2f} m | "
        f"RMSE {result.pair_rmse_m:.3f} m | max residual {result.max_pair_residual_m:.3f} m"
    )
    ax.set_title(f"{title}\n{detail}", loc="left", fontsize=9.4, fontweight="semibold", color=TOKENS["ink"])


def write_csv(rows: list[SolvedCase], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "family",
        "trial_id",
        "selected_as_median_of_old_solver",
        "solver",
        "anchors",
        "pairs",
        "nlos_count",
        "nlos_share",
        "discarded_before",
        "pair_rmse_m",
        "max_pair_residual_m",
        "max_coord_offset_m",
        "median_coord_offset_m",
        "p95_coord_offset_m",
        "energy",
        "elapsed_s",
    ]
    selected_keys = {(row.case.family, row.case.trial_id) for row in rows}
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "family": row.case.family,
                    "trial_id": row.case.trial_id,
                    "selected_as_median_of_old_solver": (row.case.family, row.case.trial_id) in selected_keys,
                    "solver": row.solver,
                    "anchors": len(row.case.true_positions),
                    "pairs": len(row.case.pairs),
                    "nlos_count": row.case.nlos_count,
                    "nlos_share": row.case.nlos_count / max(len(row.case.pairs), 1),
                    "discarded_before": row.case.discarded_before,
                    "pair_rmse_m": row.pair_rmse_m,
                    "max_pair_residual_m": row.max_pair_residual_m,
                    "max_coord_offset_m": row.max_coord_offset_m,
                    "median_coord_offset_m": row.median_coord_offset_m,
                    "p95_coord_offset_m": row.p95_coord_offset_m,
                    "energy": row.energy,
                    "elapsed_s": row.elapsed_s,
                }
            )


def render_figure(selected: dict[str, tuple[LayoutCase, SolvedCase, SolvedCase]], path: Path) -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": TOKENS["surface"],
            "savefig.facecolor": TOKENS["surface"],
            "font.family": ["Segoe UI", "DejaVu Sans", "Arial", "sans-serif"],
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(16.5, 11.0), dpi=190)
    fig.patch.set_facecolor(TOKENS["surface"])
    fig.text(
        0.035,
        0.974,
        "Old vs new anchor geometry solver on matched median cases",
        ha="left",
        va="top",
        fontsize=23,
        fontweight="bold",
        color=TOKENS["ink"],
    )
    fig.text(
        0.035,
        0.935,
        (
            f"Each row is the same synthetic layout and same noisy measurements. Median case is selected by old-solver "
            f"worst-anchor offset from {CASE_COUNT_PER_FAMILY} accepted candidates per family. "
            "Gray rings are ground truth, colored dots are solved after no-scale translation/rotation/mirror alignment."
        ),
        ha="left",
        va="top",
        fontsize=10.2,
        color=TOKENS["muted"],
    )
    fig.text(
        0.035,
        0.905,
        (
            f"Noise: Gaussian sigma {NOISE_SIGMA_M * 100:.0f} cm plus +uniform(0,{NLOS_MAX_OFFSET_M * 100:.0f} cm) "
            f"NLOS on {NLOS_PROBABILITY:.0%} of true <= {EDGE_RADIUS_M:.0f} m links. "
            f"Solver compute: {SEED_COUNT} seeds, {BASIN_HOPS} basin hops, {MAX_ITERATIONS} LM iterations."
        ),
        ha="left",
        va="top",
        fontsize=9.3,
        color=TOKENS["muted"],
    )
    for row_index, family in enumerate(["random", "grid"]):
        case, old_result, new_result = selected[family]
        limits = layout_limits(case, [old_result, new_result])
        draw_panel(axes[row_index, 0], old_result, limits, NEUTRAL["base"], NEUTRAL["dark"])
        draw_panel(axes[row_index, 1], new_result, limits, BLUE["base"], BLUE["dark"])
        axes[row_index, 0].set_ylabel("y (m)", fontsize=8.5, color=TOKENS["muted"])
        for col in range(2):
            axes[row_index, col].set_xlabel("x (m)", fontsize=8.5, color=TOKENS["muted"])
        label = (
            f"{family.title()} median case: {len(case.true_positions)} anchors, {len(case.pairs)} measured links, "
            f"NLOS {case.nlos_count}/{len(case.pairs)} ({case.nlos_count / max(len(case.pairs), 1):.0%})"
        )
        fig.text(
            0.035,
            0.824 - row_index * 0.386,
            label,
            ha="left",
            va="center",
            fontsize=11,
            fontweight="bold",
            color=TOKENS["ink"],
        )
        improvement = old_result.max_coord_offset_m - new_result.max_coord_offset_m
        fig.text(
            0.62,
            0.824 - row_index * 0.386,
            f"new - old worst-anchor offset: {-improvement:+.2f} m",
            ha="left",
            va="center",
            fontsize=10,
            color=OLIVE["dark"] if improvement >= 0 else ORANGE["dark"],
        )
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower left", bbox_to_anchor=(0.035, 0.018), frameon=False, ncol=2, fontsize=9)
    fig.text(
        0.22,
        0.036,
        (
            "Orange spokes are labeled-anchor coordinate error after alignment; the pink spoke labels the worst anchor. "
            f"New solver objective: {ANCHOR_LAYOUT_ALGORITHM} plus soft spacing and missing-pair separation priors."
        ),
        ha="left",
        va="center",
        fontsize=8.8,
        color=TOKENS["muted"],
    )
    fig.subplots_adjust(left=0.065, right=0.985, top=0.86, bottom=0.08, wspace=0.12, hspace=0.34)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", dpi=190)
    plt.close(fig)


def main() -> None:
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    selected: dict[str, tuple[LayoutCase, SolvedCase, SolvedCase]] = {}
    selected_rows: list[SolvedCase] = []
    print(
        f"Solving {CASE_COUNT_PER_FAMILY * 2} candidates with {MAX_WORKERS} worker processes "
        f"({SEED_COUNT} seeds, {BASIN_HOPS} hops, {MAX_ITERATIONS} LM iterations per solver).",
        flush=True,
    )
    solved_by_family: dict[str, list[tuple[LayoutCase, SolvedCase, SolvedCase]]] = {
        "random": [],
        "grid": [],
    }
    futures = {}
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for family in ["random", "grid"]:
            for trial_id in range(CASE_COUNT_PER_FAMILY):
                futures[executor.submit(solve_candidate, family, trial_id)] = (family, trial_id)

        for future in as_completed(futures):
            family, trial_id = futures[future]
            case, old_result, new_result = future.result()
            solved_by_family[family].append((case, old_result, new_result))
            print(
                f"{family} {trial_id + 1}/{CASE_COUNT_PER_FAMILY}: "
                f"n={len(case.true_positions)} links={len(case.pairs)} "
                f"old={old_result.max_coord_offset_m:.3f}m new={new_result.max_coord_offset_m:.3f}m "
                f"oldRMSE={old_result.pair_rmse_m:.3f} newRMSE={new_result.pair_rmse_m:.3f}",
                flush=True,
            )

    for family in ["random", "grid"]:
        solved_candidates = solved_by_family[family]
        ordered = sorted(solved_candidates, key=lambda item: item[1].max_coord_offset_m)
        selected_case, selected_old, selected_new = ordered[len(ordered) // 2]
        selected[family] = (selected_case, selected_old, selected_new)
        selected_rows.extend([selected_old, selected_new])
        print(
            f"selected {family} trial {selected_case.trial_id}: "
            f"old={selected_old.max_coord_offset_m:.3f}m new={selected_new.max_coord_offset_m:.3f}m",
            flush=True,
        )
    png_path = OUTPUTS / "anchor_solver_old_vs_new_median_cases.png"
    csv_path = OUTPUTS / "anchor_solver_old_vs_new_median_cases.csv"
    render_figure(selected, png_path)
    write_csv(selected_rows, csv_path)
    print(f"Wrote {png_path}")
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
