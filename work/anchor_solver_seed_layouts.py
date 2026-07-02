
from __future__ import annotations

import csv
import importlib.util
import math
from pathlib import Path
import random
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PILOT_PATH = ROOT / "work" / "anchor_solver_pilot.py"
OUTPUTS = ROOT / "outputs"

spec = importlib.util.spec_from_file_location("anchor_solver_pilot", PILOT_PATH)
pilot = importlib.util.module_from_spec(spec)
sys.modules["anchor_solver_pilot"] = pilot
assert spec.loader is not None
spec.loader.exec_module(pilot)

TOKENS = pilot.TOKENS
BLUE = pilot.BLUE
GOLD = pilot.GOLD
ORANGE = pilot.ORANGE
OLIVE = pilot.OLIVE
PINK = pilot.PINK
NEUTRAL = pilot.NEUTRAL


def load_results() -> list[pilot.TrialResult]:
    path = OUTPUTS / "anchor_solver_pilot_trials.csv"
    results: list[pilot.TrialResult] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            converted = {}
            for field in pilot.TrialResult.__dataclass_fields__:
                value = row[field]
                if field in {"family", "final_winner_group"}:
                    converted[field] = value
                elif field in {"trial_id", "anchor_count", "pair_count", "discarded_before"}:
                    converted[field] = int(value)
                else:
                    converted[field] = float(value)
            results.append(pilot.TrialResult(**converted))
    return results


def median_cases(results: list[pilot.TrialResult]) -> dict[str, pilot.TrialResult]:
    selected = {}
    for family in sorted({result.family for result in results}):
        family_results = [result for result in results if result.family == family]
        median = float(np.quantile([result.final_max_offset_m for result in family_results], 0.5))
        selected[family] = min(
            family_results,
            key=lambda result: abs(result.final_max_offset_m - median),
        )
    return selected


def regenerate_selected_cases(selected: dict[str, pilot.TrialResult]) -> dict[str, pilot.LayoutCase]:
    rng = random.Random(pilot.RNG_SEED)
    cases = {}
    for family, target in (("random", pilot.PILOT_RANDOM_TRIALS), ("grid", pilot.PILOT_GRID_TRIALS)):
        for trial_id in range(target):
            case = pilot.make_accepted_case(family, trial_id, rng)
            if family in selected and selected[family].trial_id == trial_id:
                cases[family] = case
    return cases


def align_positions_no_scale(
    truth: dict[str, tuple[float, float]],
    estimate: dict[str, tuple[float, float]],
) -> dict[str, tuple[float, float]]:
    ids = sorted(set(truth) & set(estimate))
    target = np.array([truth[anchor_id] for anchor_id in ids], dtype=float)
    source = np.array([estimate[anchor_id] for anchor_id in ids], dtype=float)
    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    source_centered = source - source_center
    target_centered = target - target_center
    u, _s, vt = np.linalg.svd(source_centered.T @ target_centered)
    transform = u @ vt
    aligned = source_centered @ transform + target_center
    return {anchor_id: tuple(point) for anchor_id, point in zip(ids, aligned)}


def seed_group(index: int) -> str:
    if index == 0:
        return "triangulated + hops"
    if index == 1:
        return "circle + hops"
    if index == 2:
        return "flipped circle + hops"
    return "best random + hops"


def seed_layouts(case: pilot.LayoutCase, solver_random_seed: int) -> dict[str, tuple[dict[str, tuple[float, float]], float]]:
    processed = pilot._preprocess_pairs(case.pairs, min_sigma_m=0.02, min_distance_m=0.05)
    anchor_ids = pilot._anchor_ids(processed)
    pilot._validate_connected(anchor_ids, processed)
    scale = pilot._layout_scale(processed)
    parameterization = pilot._Parameterization(anchor_ids)
    rng = random.Random(solver_random_seed)
    initial_seeds = pilot._initial_parameters(
        parameterization,
        processed,
        seed_count=max(pilot.SOLVER_SEED_COUNT, 1),
        scale=scale,
        rng=rng,
    )

    layouts: dict[str, tuple[dict[str, tuple[float, float]], float]] = {}
    raw_positions = parameterization.to_positions(initial_seeds[0])
    raw_energy = pilot._spring_energy(initial_seeds[0], parameterization, processed)
    layouts["raw triangulated"] = (raw_positions, raw_energy)

    optimized_params, optimized_energy = pilot._local_minimize(
        initial_seeds[0],
        parameterization,
        processed,
        max_iterations=pilot.SOLVER_MAX_ITERATIONS,
    )
    layouts["optimized triangulated"] = (parameterization.to_positions(optimized_params), optimized_energy)

    temperature = max(scale * scale * 1e-5, 1e-8)
    hop_scale = max(scale * 0.35, 0.05)
    grouped: dict[str, tuple[dict[str, tuple[float, float]], float]] = {}

    for seed_index, seed_params in enumerate(initial_seeds):
        current_params, current_energy = pilot._local_minimize(
            seed_params,
            parameterization,
            processed,
            max_iterations=pilot.SOLVER_MAX_ITERATIONS,
        )
        seed_best_params = current_params
        seed_best_energy = current_energy
        for _hop_index in range(max(pilot.SOLVER_BASIN_HOPS, 0)):
            hopped = [value + rng.gauss(0.0, hop_scale) for value in current_params]
            candidate_params, candidate_energy = pilot._local_minimize(
                hopped,
                parameterization,
                processed,
                max_iterations=pilot.SOLVER_MAX_ITERATIONS,
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

        group = seed_group(seed_index)
        positions = parameterization.to_positions(seed_best_params)
        if group not in grouped or seed_best_energy < grouped[group][1]:
            grouped[group] = (positions, seed_best_energy)

    layouts.update(grouped)
    return layouts


def max_offset(truth: dict[str, tuple[float, float]], estimate: dict[str, tuple[float, float]]) -> float:
    aligned = align_positions_no_scale(truth, estimate)
    return max(math.hypot(aligned[anchor_id][0] - truth[anchor_id][0], aligned[anchor_id][1] - truth[anchor_id][1]) for anchor_id in aligned)


def family_extents(
    truth: dict[str, tuple[float, float]],
    aligned_layouts: list[dict[str, tuple[float, float]]],
) -> tuple[float, float, float, float]:
    xs = [point[0] for point in truth.values()]
    ys = [point[1] for point in truth.values()]
    for layout in aligned_layouts:
        xs.extend(point[0] for point in layout.values())
        ys.extend(point[1] for point in layout.values())
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    span = max(max_x - min_x, max_y - min_y, 1.0)
    pad = 0.08 * span
    return min_x - pad, max_x + pad, min_y - pad, max_y + pad


def draw_panel(
    ax,
    case: pilot.LayoutCase,
    title: str,
    positions: dict[str, tuple[float, float]],
    color: str,
    edge: str,
    limits: tuple[float, float, float, float],
    is_winner: bool,
) -> None:
    truth = case.true_positions
    aligned = align_positions_no_scale(truth, positions)
    for pair in case.pairs:
        ax.plot(
            [truth[pair.anchor_a_id][0], truth[pair.anchor_b_id][0]],
            [truth[pair.anchor_a_id][1], truth[pair.anchor_b_id][1]],
            color=NEUTRAL["light"],
            linewidth=0.55,
            alpha=0.55,
            zorder=1,
        )
    for anchor_id, true_point in truth.items():
        solved_point = aligned[anchor_id]
        ax.plot(
            [true_point[0], solved_point[0]],
            [true_point[1], solved_point[1]],
            color=ORANGE["mid"],
            linewidth=0.6,
            alpha=0.38,
            zorder=2,
        )
    ax.scatter(
        [point[0] for point in truth.values()],
        [point[1] for point in truth.values()],
        s=24,
        facecolors=TOKENS["panel"],
        edgecolors=NEUTRAL["dark"],
        linewidths=0.8,
        label="truth",
        zorder=3,
    )
    ax.scatter(
        [point[0] for point in aligned.values()],
        [point[1] for point in aligned.values()],
        s=22,
        color=color,
        edgecolors=edge,
        linewidths=0.8,
        label="solved",
        zorder=4,
    )
    err = max_offset(truth, positions)
    winner = " (winner)" if is_winner else ""
    ax.set_title(f"{title}{winner}\nmax offset {err:.2f} m", loc="left", fontsize=9, color=TOKENS["ink"], fontweight="semibold")
    min_x, max_x, min_y, max_y = limits
    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, color=TOKENS["grid"], linewidth=0.55)
    ax.tick_params(labelsize=7, colors=TOKENS["muted"], length=0)
    for spine in ax.spines.values():
        spine.set_color(TOKENS["axis"])


def render_seed_layouts(results: list[pilot.TrialResult]) -> Path:
    selected = median_cases(results)
    cases = regenerate_selected_cases(selected)
    columns = [
        "raw triangulated",
        "optimized triangulated",
        "triangulated + hops",
        "circle + hops",
        "flipped circle + hops",
        "best random + hops",
    ]
    palette = {
        "raw triangulated": (NEUTRAL["light"], NEUTRAL["dark"]),
        "optimized triangulated": (GOLD["base"], GOLD["dark"]),
        "triangulated + hops": (BLUE["base"], BLUE["dark"]),
        "circle + hops": (PINK["base"], PINK["dark"]),
        "flipped circle + hops": (ORANGE["base"], ORANGE["dark"]),
        "best random + hops": (OLIVE["base"], OLIVE["dark"]),
    }

    plt.rcParams.update(
        {
            "figure.facecolor": TOKENS["surface"],
            "savefig.facecolor": TOKENS["surface"],
            "font.family": ["Segoe UI", "DejaVu Sans", "Arial", "sans-serif"],
        }
    )
    fig, axes = plt.subplots(2, len(columns), figsize=(20, 7.7), dpi=180)
    fig.patch.set_facecolor(TOKENS["surface"])
    fig.text(0.02, 0.98, "Median-Case Seed Layout Overlays", ha="left", va="top", fontsize=22, fontweight="bold", color=TOKENS["ink"])
    fig.text(
        0.02,
        0.935,
        "Ground truth edges and anchors are gray; solved anchors are colored after translation, rotation, and optional mirror alignment with no scaling.",
        ha="left",
        va="top",
        fontsize=10.5,
        color=TOKENS["muted"],
    )

    for row_index, family in enumerate(["grid", "random"]):
        case = cases[family]
        result = selected[family]
        solver_seed = pilot.RNG_SEED + 1000 * (0 if family == "random" else 1) + result.trial_id
        layouts = seed_layouts(case, solver_seed)
        aligned_layouts = [align_positions_no_scale(case.true_positions, layouts[column][0]) for column in columns]
        limits = family_extents(case.true_positions, aligned_layouts)
        fig.text(
            0.02,
            0.795 - row_index * 0.405,
            f"{family.title()} median trial: id {result.trial_id}, {result.anchor_count} anchors, {result.pair_count} links, final max offset {result.final_max_offset_m:.2f} m",
            ha="left",
            va="center",
            fontsize=11,
            fontweight="semibold",
            color=TOKENS["ink"],
        )
        winner_label = {
            "triangulated": "triangulated + hops",
            "circle": "circle + hops",
            "flipped circle": "flipped circle + hops",
            "random": "best random + hops",
        }.get(result.final_winner_group, "")
        for col_index, column in enumerate(columns):
            color, edge = palette[column]
            draw_panel(
                axes[row_index, col_index],
                case,
                column,
                layouts[column][0],
                color,
                edge,
                limits,
                column == winner_label,
            )
            if row_index == 1:
                axes[row_index, col_index].set_xlabel("x (m)", fontsize=8, color=TOKENS["muted"])
            if col_index == 0:
                axes[row_index, col_index].set_ylabel("y (m)", fontsize=8, color=TOKENS["muted"])
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower left", bbox_to_anchor=(0.02, 0.025), frameon=False, ncol=2, fontsize=9)
    fig.text(
        0.19,
        0.038,
        "Orange spokes show per-anchor label error after alignment; large spokes mean the distances fit a different labeled arrangement.",
        ha="left",
        va="center",
        fontsize=9,
        color=TOKENS["muted"],
    )
    fig.subplots_adjust(left=0.055, right=0.99, top=0.865, bottom=0.09, wspace=0.18, hspace=0.42)
    path = OUTPUTS / "anchor_solver_median_seed_layouts.png"
    fig.savefig(path, bbox_inches="tight", dpi=180)
    plt.close(fig)
    return path


def main() -> None:
    results = load_results()
    path = render_seed_layouts(results)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
