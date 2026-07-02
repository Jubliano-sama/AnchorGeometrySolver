from __future__ import annotations

import csv
from dataclasses import dataclass
import math
from pathlib import Path
import random
import sys
import textwrap
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT / "work" / "SmartClicker-GUI"
OUTPUTS = ROOT / "outputs"
sys.path.insert(0, str(REPO))

from uwb_capture.anchor_geometry import (  # noqa: E402
    AnchorPairDistance,
    _Parameterization,
    _anchor_ids,
    _initial_parameters,
    _layout_scale,
    _local_minimize,
    _preprocess_pairs,
    _spring_energy,
    _validate_connected,
    pair_residuals,
    rotate_layout_to_level,
)


PILOT_RANDOM_TRIALS = 8
PILOT_GRID_TRIALS = 8
SOLVER_SEED_COUNT = 8
SOLVER_BASIN_HOPS = 3
SOLVER_MAX_ITERATIONS = 60
NOISE_SIGMA_M = 0.03
NLOS_PROBABILITY = 1.0 / 3.0
NLOS_MAX_OFFSET_M = 0.20
EDGE_RADIUS_M = 8.0
PAIR_SIGMA_M = 0.05
RNG_SEED = 20260626

THRESHOLDS_M = [0.05, 0.10, 0.20, 0.50, 1.00, 2.00, 5.00, 10.00, 15.00, 20.00]


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


@dataclass(frozen=True)
class LayoutCase:
    family: str
    trial_id: int
    true_positions: dict[str, tuple[float, float]]
    pairs: list[AnchorPairDistance]
    noisy_edge_count: int
    nlos_edge_count: int
    discarded_before: int


@dataclass(frozen=True)
class TrialResult:
    family: str
    trial_id: int
    anchor_count: int
    pair_count: int
    nlos_pair_share: float
    discarded_before: int
    final_winner_group: str
    final_energy: float
    final_pair_rmse_m: float
    final_max_pair_residual_m: float
    raw_triangulated_max_offset_m: float
    optimized_triangulated_max_offset_m: float
    final_max_offset_m: float
    final_median_offset_m: float
    final_p95_offset_m: float


def pair_key(a: str, b: str) -> tuple[str, str]:
    return tuple(sorted((a, b)))


def is_connected(anchor_ids: list[str], pairs: Iterable[AnchorPairDistance]) -> bool:
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


def is_generically_globally_rigid(
    positions: dict[str, tuple[float, float]],
    pairs: list[AnchorPairDistance],
) -> bool:
    anchor_ids = sorted(positions)
    target_rank = 2 * len(anchor_ids) - 3
    if not is_connected(anchor_ids, pairs):
        return False
    if len(pairs) < target_rank:
        return False
    if rigidity_rank(positions, pairs) < target_rank:
        return False
    if not is_three_vertex_connected(anchor_ids, pairs):
        return False
    for remove_index in range(len(pairs)):
        remaining = pairs[:remove_index] + pairs[remove_index + 1 :]
        if rigidity_rank(positions, remaining) < target_rank:
            return False
    return True


def accepted_graph(positions: dict[str, tuple[float, float]], pairs: list[AnchorPairDistance]) -> bool:
    return is_generically_globally_rigid(positions, pairs)


def generate_random_positions(rng: random.Random, n: int = 30) -> dict[str, tuple[float, float]]:
    points: list[tuple[float, float]] = []
    attempts = 0
    while len(points) < n and attempts < 20_000:
        attempts += 1
        candidate = (rng.uniform(0.0, 25.0), rng.uniform(0.0, 25.0))
        if all(math.hypot(candidate[0] - x, candidate[1] - y) >= 2.0 for x, y in points):
            points.append(candidate)
    if len(points) != n:
        raise RuntimeError("Could not place random anchors with the minimum spacing rule.")
    return {f"A{index:02d}": point for index, point in enumerate(points)}


def grid_dimensions(rng: random.Random) -> tuple[int, int]:
    choices = [
        (4, 4),
        (4, 5),
        (5, 4),
        (4, 6),
        (6, 4),
        (5, 5),
        (5, 6),
        (6, 5),
        (4, 7),
        (7, 4),
        (4, 8),
        (8, 4),
    ]
    return rng.choice(choices)


def generate_grid_positions(rng: random.Random) -> dict[str, tuple[float, float]]:
    rows, cols = grid_dimensions(rng)
    x_spacing = rng.uniform(4.0, 8.0)
    y_spacing = rng.uniform(4.0, 8.0)
    positions = {}
    index = 0
    for row in range(rows):
        for col in range(cols):
            positions[f"A{index:02d}"] = (col * x_spacing, row * y_spacing)
            index += 1
    return positions


def generate_noisy_pairs(
    positions: dict[str, tuple[float, float]],
    rng: random.Random,
) -> tuple[list[AnchorPairDistance], int, int]:
    pairs: list[AnchorPairDistance] = []
    ids = sorted(positions)
    nlos_count = 0
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
                    anchor_a_id=anchor_a,
                    anchor_b_id=anchor_b,
                    distance_m=max(measured, 0.05),
                    sigma_m=PAIR_SIGMA_M,
                    source=source,
                )
            )
    return pairs, len(pairs), nlos_count


def make_case(
    family: str,
    trial_id: int,
    rng: random.Random,
    discarded_before: int,
) -> LayoutCase:
    if family == "random":
        positions = generate_random_positions(rng, n=30)
    elif family == "grid":
        positions = generate_grid_positions(rng)
    else:
        raise ValueError(f"Unknown family: {family}")
    pairs, noisy_edge_count, nlos_edge_count = generate_noisy_pairs(positions, rng)
    return LayoutCase(
        family=family,
        trial_id=trial_id,
        true_positions=positions,
        pairs=pairs,
        noisy_edge_count=noisy_edge_count,
        nlos_edge_count=nlos_edge_count,
        discarded_before=discarded_before,
    )


def make_accepted_case(family: str, trial_id: int, rng: random.Random) -> LayoutCase:
    discarded = 0
    while True:
        case = make_case(family, trial_id, rng, discarded)
        if accepted_graph(case.true_positions, case.pairs):
            return case
        discarded += 1


def align_offsets_no_scale(
    truth: dict[str, tuple[float, float]],
    estimate: dict[str, tuple[float, float]],
) -> np.ndarray:
    ids = sorted(set(truth) & set(estimate))
    if not ids:
        return np.array([math.inf])
    target = np.array([truth[anchor_id] for anchor_id in ids], dtype=float)
    source = np.array([estimate[anchor_id] for anchor_id in ids], dtype=float)
    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    source_centered = source - source_center
    target_centered = target - target_center
    u, _s, vt = np.linalg.svd(source_centered.T @ target_centered)
    rotation_or_reflection = u @ vt
    aligned = source_centered @ rotation_or_reflection + target_center
    return np.linalg.norm(aligned - target, axis=1)


def position_error_summary(
    truth: dict[str, tuple[float, float]],
    estimate: dict[str, tuple[float, float]],
) -> tuple[float, float, float]:
    offsets = align_offsets_no_scale(truth, estimate)
    return (
        float(np.max(offsets)),
        float(np.median(offsets)),
        float(np.quantile(offsets, 0.95)),
    )


def seed_group(index: int) -> str:
    if index == 0:
        return "triangulated"
    if index == 1:
        return "circle"
    if index == 2:
        return "flipped circle"
    return "random"


def solve_with_diagnostics(case: LayoutCase, solver_random_seed: int) -> TrialResult:
    processed = _preprocess_pairs(case.pairs, min_sigma_m=0.02, min_distance_m=0.05)
    anchor_ids = _anchor_ids(processed)
    _validate_connected(anchor_ids, processed)
    scale = _layout_scale(processed)
    parameterization = _Parameterization(anchor_ids)
    rng = random.Random(solver_random_seed)
    initial_seeds = _initial_parameters(
        parameterization,
        processed,
        seed_count=max(SOLVER_SEED_COUNT, 1),
        scale=scale,
        rng=rng,
    )

    raw_triangulated_positions = parameterization.to_positions(initial_seeds[0])
    raw_triangulated_positions = rotate_layout_to_level(
        raw_triangulated_positions,
        anchor_ids[0],
        anchor_ids[1],
    )
    raw_max, _raw_median, _raw_p95 = position_error_summary(case.true_positions, raw_triangulated_positions)

    optimized_triangulated_params, _optimized_triangulated_energy = _local_minimize(
        initial_seeds[0],
        parameterization,
        processed,
        max_iterations=SOLVER_MAX_ITERATIONS,
    )
    optimized_triangulated_positions = parameterization.to_positions(optimized_triangulated_params)
    optimized_triangulated_positions = rotate_layout_to_level(
        optimized_triangulated_positions,
        anchor_ids[0],
        anchor_ids[1],
    )
    optimized_max, _optimized_median, _optimized_p95 = position_error_summary(
        case.true_positions,
        optimized_triangulated_positions,
    )

    best_params: list[float] | None = None
    best_energy = math.inf
    best_group = "none"
    temperature = max(scale * scale * 1e-5, 1e-8)
    hop_scale = max(scale * 0.35, 0.05)

    for seed_index, seed_params in enumerate(initial_seeds):
        current_params, current_energy = _local_minimize(
            seed_params,
            parameterization,
            processed,
            max_iterations=SOLVER_MAX_ITERATIONS,
        )
        seed_best_params = current_params
        seed_best_energy = current_energy

        for _hop_index in range(max(SOLVER_BASIN_HOPS, 0)):
            hopped = [value + rng.gauss(0.0, hop_scale) for value in current_params]
            candidate_params, candidate_energy = _local_minimize(
                hopped,
                parameterization,
                processed,
                max_iterations=SOLVER_MAX_ITERATIONS,
            )
            accept = candidate_energy <= current_energy
            if not accept:
                probability = math.exp(
                    max(min((current_energy - candidate_energy) / temperature, 0.0), -60.0)
                )
                accept = rng.random() < probability
            if accept:
                current_params = candidate_params
                current_energy = candidate_energy
            if candidate_energy < seed_best_energy:
                seed_best_params = candidate_params
                seed_best_energy = candidate_energy

        if seed_best_energy < best_energy:
            best_params = seed_best_params
            best_energy = seed_best_energy
            best_group = seed_group(seed_index)

    if best_params is None:
        raise ValueError("Solver produced no best parameters.")
    final_positions = parameterization.to_positions(best_params)
    final_positions = rotate_layout_to_level(final_positions, anchor_ids[0], anchor_ids[1])
    residuals = pair_residuals(final_positions, processed)
    final_pair_rmse = math.sqrt(sum(value * value for value in residuals.values()) / len(residuals))
    final_max_pair_residual = max(abs(value) for value in residuals.values())
    final_max, final_median, final_p95 = position_error_summary(case.true_positions, final_positions)
    return TrialResult(
        family=case.family,
        trial_id=case.trial_id,
        anchor_count=len(case.true_positions),
        pair_count=len(case.pairs),
        nlos_pair_share=case.nlos_edge_count / max(case.noisy_edge_count, 1),
        discarded_before=case.discarded_before,
        final_winner_group=best_group,
        final_energy=float(best_energy),
        final_pair_rmse_m=final_pair_rmse,
        final_max_pair_residual_m=final_max_pair_residual,
        raw_triangulated_max_offset_m=raw_max,
        optimized_triangulated_max_offset_m=optimized_max,
        final_max_offset_m=final_max,
        final_median_offset_m=final_median,
        final_p95_offset_m=final_p95,
    )


def run_pilot() -> list[TrialResult]:
    rng = random.Random(RNG_SEED)
    results: list[TrialResult] = []
    for family, target in (("random", PILOT_RANDOM_TRIALS), ("grid", PILOT_GRID_TRIALS)):
        for trial_id in range(target):
            case = make_accepted_case(family, trial_id, rng)
            solver_seed = RNG_SEED + 1000 * (0 if family == "random" else 1) + trial_id
            results.append(solve_with_diagnostics(case, solver_seed))
            print(
                f"{family} {trial_id + 1}/{target}: "
                f"n={len(case.true_positions)} pairs={len(case.pairs)} "
                f"winner={results[-1].final_winner_group} "
                f"max_offset={results[-1].final_max_offset_m:.3f}m"
            )
    return results


def by_family(results: list[TrialResult]) -> dict[str, list[TrialResult]]:
    families = {}
    for result in results:
        families.setdefault(result.family, []).append(result)
    return families


def percentile(values: list[float], q: float) -> float:
    return float(np.quantile(np.array(values, dtype=float), q))


def threshold_table(results: list[TrialResult], field: str) -> dict[str, dict[float, float]]:
    table: dict[str, dict[float, float]] = {}
    for family, family_results in by_family(results).items():
        values = [getattr(result, field) for result in family_results]
        table[family] = {
            threshold: sum(value <= threshold for value in values) / len(values)
            for threshold in THRESHOLDS_M
        }
    return table


def winner_share_table(results: list[TrialResult]) -> dict[str, dict[str, float]]:
    groups = ["triangulated", "circle", "flipped circle", "random"]
    table = {}
    for family, family_results in by_family(results).items():
        table[family] = {
            group: sum(result.final_winner_group == group for result in family_results) / len(family_results)
            for group in groups
        }
    return table


def write_csv(results: list[TrialResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(TrialResult.__dataclass_fields__)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for result in results:
            writer.writerow({field: getattr(result, field) for field in fields})


def write_summary(results: list[TrialResult], path: Path) -> None:
    lines = [
        "# Anchor Solver Pilot Summary",
        "",
        f"Trials: {len(results)} accepted ({PILOT_RANDOM_TRIALS} random, {PILOT_GRID_TRIALS} grid).",
        f"Solver pilot settings: seed_count={SOLVER_SEED_COUNT}, basin_hops={SOLVER_BASIN_HOPS}, max_iterations={SOLVER_MAX_ITERATIONS}.",
        f"Edges: true anchor distance <= {EDGE_RADIUS_M:.1f} m. Noise: Gaussian sigma={NOISE_SIGMA_M:.2f} m; NLOS: {NLOS_PROBABILITY:.0%} of edges get +uniform(0,{NLOS_MAX_OFFSET_M:.2f}) m.",
        f"Scoring: rigid alignment with optional reflection, no scale change; metric is max Euclidean anchor coordinate offset.",
        "",
        "## Family Summary",
        "",
        "| family | trials | anchors median | pairs median | final max offset median | final max offset p90 | winner mode | discarded before accepted |",
        "|---|---:|---:|---:|---:|---:|---|---:|",
    ]
    for family, family_results in sorted(by_family(results).items()):
        final_values = [result.final_max_offset_m for result in family_results]
        winner_counts: dict[str, int] = {}
        for result in family_results:
            winner_counts[result.final_winner_group] = winner_counts.get(result.final_winner_group, 0) + 1
        winner_mode = max(winner_counts.items(), key=lambda item: item[1])[0]
        lines.append(
            "| {family} | {trials} | {anchors:.0f} | {pairs:.0f} | {median:.3f} m | {p90:.3f} m | {winner} | {discarded} |".format(
                family=family,
                trials=len(family_results),
                anchors=percentile([result.anchor_count for result in family_results], 0.5),
                pairs=percentile([result.pair_count for result in family_results], 0.5),
                median=percentile(final_values, 0.5),
                p90=percentile(final_values, 0.9),
                winner=winner_mode,
                discarded=sum(result.discarded_before for result in family_results),
            )
        )
    lines.extend(["", "## Threshold Hit Rates", ""])
    for field, label in [
        ("raw_triangulated_max_offset_m", "raw triangulated"),
        ("optimized_triangulated_max_offset_m", "optimized triangulated"),
        ("final_max_offset_m", "final solver"),
    ]:
        lines.append(f"### {label}")
        lines.append("")
        lines.append("| family | " + " | ".join(f"<={threshold:.2f} m" for threshold in THRESHOLDS_M) + " |")
        lines.append("|---|" + "|".join("---:" for _ in THRESHOLDS_M) + "|")
        table = threshold_table(results, field)
        for family in sorted(table):
            lines.append(
                "| {family} | {values} |".format(
                    family=family,
                    values=" | ".join(f"{share:.0%}" for share in table[family].values()),
                )
            )
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def style_axis(ax) -> None:
    ax.set_facecolor(TOKENS["panel"])
    ax.grid(True, color=TOKENS["grid"], linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(TOKENS["axis"])
    ax.spines["bottom"].set_color(TOKENS["axis"])
    ax.tick_params(colors=TOKENS["muted"], labelsize=9)
    ax.xaxis.label.set_color(TOKENS["ink"])
    ax.yaxis.label.set_color(TOKENS["ink"])


def add_card(fig, x: float, y: float, w: float, h: float, title: str, value: str, note: str, color: str) -> None:
    ax = fig.add_axes([x, y, w, h])
    ax.set_facecolor(TOKENS["panel"])
    for spine in ax.spines.values():
        spine.set_edgecolor(TOKENS["grid"])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.text(0.06, 0.78, title, ha="left", va="top", fontsize=10, color=TOKENS["muted"], transform=ax.transAxes)
    ax.text(0.06, 0.48, value, ha="left", va="center", fontsize=18, fontweight="bold", color=color, transform=ax.transAxes)
    ax.text(0.06, 0.15, note, ha="left", va="bottom", fontsize=8.5, color=TOKENS["muted"], transform=ax.transAxes)


def render_infographic(results: list[TrialResult], path: Path) -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": TOKENS["surface"],
            "savefig.facecolor": TOKENS["surface"],
            "axes.edgecolor": TOKENS["axis"],
            "axes.labelcolor": TOKENS["ink"],
            "font.family": ["Segoe UI", "DejaVu Sans", "Arial", "sans-serif"],
        }
    )
    fig = plt.figure(figsize=(15, 10), dpi=180)
    fig.patch.set_facecolor(TOKENS["surface"])
    fig.text(
        0.045,
        0.965,
        "Anchor Geometry Solver Pilot",
        fontsize=24,
        fontweight="bold",
        color=TOKENS["ink"],
        ha="left",
        va="top",
    )
    fig.text(
        0.045,
        0.925,
        "Shape recovery after rigid alignment with optional mirror; edge noise sigma 3 cm plus NLOS positive bias on one-third of links.",
        fontsize=11,
        color=TOKENS["muted"],
        ha="left",
        va="top",
    )

    families = by_family(results)
    total_discarded = sum(result.discarded_before for result in results)
    final_values = [result.final_max_offset_m for result in results]
    add_card(
        fig,
        0.045,
        0.805,
        0.20,
        0.085,
        "Accepted pilot trials",
        str(len(results)),
        f"{PILOT_RANDOM_TRIALS} random + {PILOT_GRID_TRIALS} grid",
        BLUE["dark"],
    )
    add_card(
        fig,
        0.270,
        0.805,
        0.20,
        0.085,
        "Median final max offset",
        f"{percentile(final_values, 0.5) * 100:.0f} cm",
        f"p90 {percentile(final_values, 0.9) * 100:.0f} cm",
        ORANGE["dark"],
    )
    add_card(
        fig,
        0.495,
        0.805,
        0.20,
        0.085,
        "Solver settings",
        f"{SOLVER_SEED_COUNT} x {SOLVER_BASIN_HOPS}",
        "seeds x basin hops, pilot",
        OLIVE["dark"],
    )
    add_card(
        fig,
        0.720,
        0.805,
        0.235,
        0.085,
        "Discarded candidates",
        str(total_discarded),
        "disconnected or < 2n - 3 edges",
        PINK["dark"],
    )

    ax1 = fig.add_axes([0.06, 0.505, 0.39, 0.235])
    style_axis(ax1)
    width = 0.36
    x = np.arange(len(THRESHOLDS_M))
    threshold_hits = threshold_table(results, "final_max_offset_m")
    for offset, family, color, edge in [
        (-width / 2, "grid", BLUE["base"], BLUE["dark"]),
        (width / 2, "random", ORANGE["base"], ORANGE["dark"]),
    ]:
        if family not in threshold_hits:
            continue
        values = [threshold_hits[family][threshold] * 100 for threshold in THRESHOLDS_M]
        bars = ax1.bar(x + offset, values, width=width, color=color, edgecolor=edge, linewidth=1.0, label=family)
        for bar, value in zip(bars, values):
            ax1.text(bar.get_x() + bar.get_width() / 2, value + 2, f"{value:.0f}%", ha="center", fontsize=8, color=TOKENS["ink"])
    ax1.set_ylim(0, 108)
    ax1.set_xticks(x, [f"{threshold * 100:.0f}cm" if threshold < 1 else f"{threshold:.0f}m" for threshold in THRESHOLDS_M])
    ax1.set_ylabel("Share of trials")
    ax1.yaxis.set_major_formatter(mticker.PercentFormatter())
    ax1.set_title("How often the final shape lands under max-offset thresholds", loc="left", fontsize=12, fontweight="bold", color=TOKENS["ink"])
    ax1.legend(frameon=False, loc="upper left", bbox_to_anchor=(0, 1.02), ncol=2)

    ax2 = fig.add_axes([0.53, 0.505, 0.39, 0.235])
    style_axis(ax2)
    winner_table = winner_share_table(results)
    groups = ["triangulated", "circle", "flipped circle", "random"]
    colors = [BLUE["base"], GOLD["base"], PINK["base"], OLIVE["base"]]
    edges = [BLUE["dark"], GOLD["dark"], PINK["dark"], OLIVE["dark"]]
    family_order = [family for family in ["grid", "random"] if family in winner_table]
    bottom = np.zeros(len(family_order))
    for group, color, edge in zip(groups, colors, edges):
        values = np.array([winner_table[family][group] * 100 for family in family_order])
        ax2.barh(family_order, values, left=bottom, color=color, edgecolor=edge, linewidth=1.0, label=group)
        for row_index, value in enumerate(values):
            if value >= 12:
                ax2.text(bottom[row_index] + value / 2, row_index, f"{value:.0f}%", ha="center", va="center", fontsize=8, color=TOKENS["ink"])
        bottom += values
    ax2.set_xlim(0, 100)
    ax2.xaxis.set_major_formatter(mticker.PercentFormatter())
    ax2.set_xlabel("Share of trials")
    ax2.set_title("Which seed group produced the final lowest-energy layout", loc="left", fontsize=12, fontweight="bold", color=TOKENS["ink"])
    ax2.legend(frameon=False, loc="upper left", bbox_to_anchor=(0, 1.14), ncol=4, fontsize=8)

    ax3 = fig.add_axes([0.06, 0.185, 0.39, 0.235])
    style_axis(ax3)
    phase_fields = [
        ("Raw triangulated", "raw_triangulated_max_offset_m", NEUTRAL["light"], NEUTRAL["dark"]),
        ("Optimized triangulated", "optimized_triangulated_max_offset_m", GOLD["base"], GOLD["dark"]),
        ("Final solver", "final_max_offset_m", BLUE["base"], BLUE["dark"]),
    ]
    positions = []
    values = []
    box_colors = []
    labels = []
    pos = 1
    for family in ["grid", "random"]:
        if family not in families:
            continue
        for label, field, fill, edge in phase_fields:
            positions.append(pos)
            values.append([getattr(result, field) * 100 for result in families[family]])
            box_colors.append((fill, edge))
            labels.append(f"{family}\n{label}")
            pos += 1
        pos += 0.6
    box = ax3.boxplot(values, positions=positions, widths=0.56, patch_artist=True, showfliers=True)
    for patch, (fill, edge) in zip(box["boxes"], box_colors):
        patch.set_facecolor(fill)
        patch.set_edgecolor(edge)
        patch.set_linewidth(1.0)
    for median in box["medians"]:
        median.set_color(TOKENS["ink"])
        median.set_linewidth(1.0)
    for whisker in box["whiskers"]:
        whisker.set_color(NEUTRAL["mid"])
    for cap in box["caps"]:
        cap.set_color(NEUTRAL["mid"])
    ax3.set_xticks(positions, labels, rotation=25, ha="right")
    ax3.set_ylabel("Worst anchor offset (cm)")
    ax3.set_title("Triangulated baseline versus final basin-hopped result", loc="left", fontsize=12, fontweight="bold", color=TOKENS["ink"])

    ax4 = fig.add_axes([0.53, 0.185, 0.39, 0.235])
    style_axis(ax4)
    for family, color, edge, marker in [
        ("grid", BLUE["base"], BLUE["dark"], "o"),
        ("random", ORANGE["base"], ORANGE["dark"], "s"),
    ]:
        if family not in families:
            continue
        xvals = [result.pair_count / result.anchor_count for result in families[family]]
        yvals = [result.final_max_offset_m * 100 for result in families[family]]
        sizes = [60 + result.nlos_pair_share * 180 for result in families[family]]
        ax4.scatter(xvals, yvals, s=sizes, c=color, edgecolors=edge, linewidths=1.0, alpha=0.75, marker=marker, label=family)
    ax4.set_xlabel("Edges per anchor")
    ax4.set_ylabel("Final worst anchor offset (cm)")
    ax4.set_title("Connectivity and biased links drive the hard cases", loc="left", fontsize=12, fontweight="bold", color=TOKENS["ink"])
    ax4.legend(frameon=False, loc="upper right")
    note = "\n".join(
        textwrap.wrap(
            "Bubble size reflects the observed NLOS share. Pilot stats are intentionally small; use them to judge the visual and experiment shape before a larger run.",
            width=92,
        )
    )
    fig.text(0.06, 0.085, note, ha="left", va="bottom", fontsize=9, color=TOKENS["muted"])
    fig.text(
        0.06,
        0.055,
        "Source: synthetic layouts using SmartClicker-GUI's dependency-free anchor_geometry solver. Score allows translation, rotation, and mirror; scale is fixed.",
        ha="left",
        va="bottom",
        fontsize=8,
        color=TOKENS["muted"],
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", dpi=180)
    plt.close(fig)


def main() -> None:
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    results = run_pilot()
    write_csv(results, OUTPUTS / "anchor_solver_pilot_trials.csv")
    write_summary(results, OUTPUTS / "anchor_solver_pilot_summary.md")
    render_infographic(results, OUTPUTS / "anchor_solver_pilot_infographic.png")
    print(f"Wrote {OUTPUTS / 'anchor_solver_pilot_infographic.png'}")
    print(f"Wrote {OUTPUTS / 'anchor_solver_pilot_trials.csv'}")
    print(f"Wrote {OUTPUTS / 'anchor_solver_pilot_summary.md'}")


if __name__ == "__main__":
    main()
