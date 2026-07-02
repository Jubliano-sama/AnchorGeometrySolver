from __future__ import annotations

import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
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
SWEEP_PATH = ROOT / "work" / "anchor_solver_large_hop_swap_sweep.py"
OUTPUTS = ROOT / "outputs"

spec = importlib.util.spec_from_file_location("large_swap", SWEEP_PATH)
large_swap = importlib.util.module_from_spec(spec)
sys.modules["large_swap"] = large_swap
assert spec.loader is not None
spec.loader.exec_module(large_swap)

from uwb_capture.anchor_geometry import _circle_seed, _random_seed, _triangulated_seed  # noqa: E402

pilot = large_swap.pilot
TOKENS = large_swap.TOKENS
BLUE = large_swap.BLUE
GOLD = large_swap.GOLD
ORANGE = large_swap.ORANGE
OLIVE = large_swap.OLIVE
PINK = large_swap.PINK
NEUTRAL = large_swap.NEUTRAL

METHODS = [
    ("old stock", 24, 10, 0.35, False, False, False),
    ("corner-frame swaps", 24, 10, 1.50, True, True, False),
    ("corner-frame swaps + prior", 24, 10, 1.50, True, True, True),
]


def old_initial_parameters(parameterization, pairs, seed_count: int, scale: float, rng: random.Random) -> list[list[float]]:
    seeds = [
        _triangulated_seed(parameterization, pairs, scale),
        _circle_seed(parameterization, scale, flip_y=False),
        _circle_seed(parameterization, scale, flip_y=True),
    ]
    while len(seeds) < seed_count:
        seeds.append(_random_seed(parameterization, scale, rng))
    return seeds[:seed_count]


def stable_label_seed(label: str) -> int:
    return sum((index + 1) * ord(char) for index, char in enumerate(label))


def solve_search(case, pairs, method):
    label, seed_count, hops, hop_multiplier, use_swaps, use_corner_seed, use_corner_prior = method
    processed = pilot._preprocess_pairs(pairs, min_sigma_m=0.02, min_distance_m=0.05)
    order = large_swap.anchor_order(case, processed, use_corner_seed)
    pilot._validate_connected(pilot._anchor_ids(processed), processed)
    parameterization = pilot._Parameterization(order)
    scale = pilot._layout_scale(processed)
    rng = random.Random(large_swap.RNG_SEED + stable_label_seed(label))

    seeds: list[list[float]] = []
    if use_corner_seed:
        seeds.append(large_swap.corner_seed_params(case, parameterization))
    seeds.extend(
        old_initial_parameters(
            parameterization,
            processed,
            seed_count=max(seed_count - len(seeds), 1),
            scale=scale,
            rng=rng,
        )
    )
    seeds = seeds[:seed_count]

    priors = large_swap.prior_map(case) if use_corner_prior else {}
    allowed_swaps = large_swap.swappable_anchor_ids(case, parameterization, use_corner_seed)
    hop_sigma = max(scale * hop_multiplier, 0.05)
    temperature = max(scale * scale * 1e-5, 1e-8)

    best_params = None
    best_energy = math.inf
    for seed_params in seeds:
        current_params, current_energy = large_swap.local_minimize(
            seed_params,
            parameterization,
            processed,
            priors,
            pilot.SOLVER_MAX_ITERATIONS,
        )
        if current_energy < best_energy:
            best_params = current_params
            best_energy = current_energy

        for _ in range(max(hops, 0)):
            candidate_start = large_swap.proposed_params(
                current_params,
                parameterization,
                rng,
                hop_sigma,
                use_swaps,
                allowed_swaps,
            )
            candidate_params, candidate_energy = large_swap.local_minimize(
                candidate_start,
                parameterization,
                processed,
                priors,
                pilot.SOLVER_MAX_ITERATIONS,
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
        raise ValueError("no solution")

    positions = parameterization.to_positions(best_params)
    residuals = pilot.pair_residuals(positions, processed)
    pair_rmse = math.sqrt(sum(value * value for value in residuals.values()) / len(residuals))
    max_pair = max(abs(value) for value in residuals.values())
    return positions, best_energy, pair_rmse, max_pair


def solve_one(measurement: str, method_index: int):
    case = large_swap.make_case()
    pairs = case.exact_pairs if measurement == "exact" else case.noisy_pairs
    method = METHODS[method_index]
    positions, energy, pair_rmse, max_pair = solve_search(case, pairs, method)
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
        "anchor_count": len(case.positions),
        "pair_count": len(pairs),
        "nlos_count": case.nlos_count if measurement == "noisy+nlos" else 0,
        "nlos_share": case.nlos_count / max(len(case.noisy_pairs), 1) if measurement == "noisy+nlos" else 0.0,
        "corner_prior": method[-1],
        "corner_seed": method[-2],
        "swaps": method[-3],
        "hop_multiplier": method[3],
    }


def align_positions(truth, estimate):
    ids = sorted(set(truth) & set(estimate))
    target = np.array([truth[a] for a in ids], dtype=float)
    source = np.array([estimate[a] for a in ids], dtype=float)
    sc = source.mean(axis=0)
    tc = target.mean(axis=0)
    source_centered = source - sc
    target_centered = target - tc
    best = None
    best_max = math.inf
    for reflect in (1.0, -1.0):
        reflected = source_centered.copy()
        reflected[:, 1] *= reflect
        u, _s, vt = np.linalg.svd(reflected.T @ target_centered)
        aligned = reflected @ (u @ vt) + tc
        max_offset = float(np.linalg.norm(aligned - target, axis=1).max())
        if max_offset < best_max:
            best = aligned
            best_max = max_offset
    assert best is not None
    return {anchor_id: tuple(point) for anchor_id, point in zip(ids, best)}


def layout_limits(case, results):
    xs = [p[0] for p in case.positions.values()]
    ys = [p[1] for p in case.positions.values()]
    for result in results:
        aligned = align_positions(case.positions, result["positions"])
        xs.extend(p[0] for p in aligned.values())
        ys.extend(p[1] for p in aligned.values())
    span = max(max(xs) - min(xs), max(ys) - min(ys), 1.0)
    pad = 0.10 * span
    return min(xs) - pad, max(xs) + pad, min(ys) - pad, max(ys) + pad


def draw_panel(ax, case, result, limits, fill, edge):
    truth = case.positions
    solved = align_positions(truth, result["positions"])
    ax.set_facecolor(TOKENS["panel"])
    for pair in case.exact_pairs:
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
        key=lambda anchor_id: math.hypot(
            solved[anchor_id][0] - truth[anchor_id][0],
            solved[anchor_id][1] - truth[anchor_id][1],
        ),
    )
    for anchor_id, true_point in truth.items():
        solved_point = solved[anchor_id]
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
        [p[0] for p in truth.values()],
        [p[1] for p in truth.values()],
        s=30,
        facecolors=TOKENS["panel"],
        edgecolors=NEUTRAL["dark"],
        linewidths=0.9,
        zorder=3,
        label="truth",
    )
    ax.scatter(
        [p[0] for p in solved.values()],
        [p[1] for p in solved.values()],
        s=25,
        facecolors=fill,
        edgecolors=edge,
        linewidths=0.9,
        zorder=4,
        label="solved",
    )
    ax.set_xlim(limits[0], limits[1])
    ax.set_ylim(limits[2], limits[3])
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, color=TOKENS["grid"], linewidth=0.6)
    ax.tick_params(labelsize=7, colors=TOKENS["muted"], length=0)
    for spine in ax.spines.values():
        spine.set_color(TOKENS["axis"])
    ax.set_title(
        f"{result['method']}\nRMSE {result['pair_rmse_m']:.4f} m | max offset {result['max_coord_offset_m']:.3f} m",
        loc="left",
        fontsize=9.2,
        fontweight="semibold",
        color=TOKENS["ink"],
    )


def write_csv(results, path: Path) -> None:
    fields = [
        "measurement",
        "method",
        "anchor_count",
        "pair_count",
        "nlos_count",
        "nlos_share",
        "pair_rmse_m",
        "max_pair_residual_m",
        "max_coord_offset_m",
        "median_coord_offset_m",
        "p95_coord_offset_m",
        "energy",
        "hop_multiplier",
        "swaps",
        "corner_seed",
        "corner_prior",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fields)
        writer.writeheader()
        for result in results:
            writer.writerow({field: result[field] for field in fields})


def render(results, path: Path) -> None:
    case = large_swap.make_case()
    color_map = {
        "old stock": (NEUTRAL["base"], NEUTRAL["dark"]),
        "corner-frame swaps": (GOLD["base"], GOLD["dark"]),
        "corner-frame swaps + prior": (OLIVE["base"], OLIVE["dark"]),
    }
    plt.rcParams.update(
        {
            "figure.facecolor": TOKENS["surface"],
            "savefig.facecolor": TOKENS["surface"],
            "font.family": ["Segoe UI", "DejaVu Sans", "Arial", "sans-serif"],
        }
    )
    fig, axes = plt.subplots(2, 3, figsize=(17.5, 9.4), dpi=190)
    fig.patch.set_facecolor(TOKENS["surface"])
    fig.text(0.035, 0.975, "Old solver vs corner-framed swap search on the same grid layout", ha="left", va="top", fontsize=22, fontweight="bold", color=TOKENS["ink"])
    fig.text(
        0.035,
        0.935,
        "Same irregular 4x4 grid-like anchor layout. Corner modes know only the roles top-left=A00, top-right=A03, bottom-left=A12; the prior uses nominal 6 m grid spacing with sigma 3 m.",
        ha="left",
        va="top",
        fontsize=10,
        color=TOKENS["muted"],
    )
    fig.text(
        0.035,
        0.905,
        f"Edges are true distance <= {large_swap.EDGE_RADIUS_M:.1f} m. Noisy row adds Gaussian sigma {large_swap.NOISE_SIGMA_M * 100:.0f} cm and +uniform(0,{large_swap.NLOS_MAX_OFFSET_M * 100:.0f} cm) NLOS on {large_swap.NLOS_PROBABILITY:.0%} of links. All modes use distance-only objective except the explicit three-corner prior column.",
        ha="left",
        va="top",
        fontsize=9.2,
        color=TOKENS["muted"],
    )
    limits = layout_limits(case, results)
    for row, measurement in enumerate(["exact", "noisy+nlos"]):
        row_results = [next(r for r in results if r["measurement"] == measurement and r["method"] == method[0]) for method in METHODS]
        for col, result in enumerate(row_results):
            fill, edge = color_map[result["method"]]
            draw_panel(axes[row, col], case, result, limits, fill, edge)
            axes[row, col].set_xlabel("x (m)", fontsize=8, color=TOKENS["muted"])
            if col == 0:
                axes[row, col].set_ylabel("y (m)", fontsize=8, color=TOKENS["muted"])
        label = "Exact measured edges" if measurement == "exact" else f"Noisy + NLOS edges ({case.nlos_count}/{len(case.noisy_pairs)} NLOS)"
        fig.text(0.035, 0.794 - row * 0.387, label, ha="left", va="center", fontsize=11, fontweight="bold", color=TOKENS["ink"])
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower left", bbox_to_anchor=(0.035, 0.025), frameon=False, ncol=2, fontsize=9)
    fig.text(
        0.20,
        0.04,
        "Gray rings are ground truth; colored dots are solved positions after no-scale rotation/translation/mirror alignment. Orange/pink spokes are labeled-anchor coordinate error.",
        ha="left",
        va="center",
        fontsize=8.8,
        color=TOKENS["muted"],
    )
    fig.subplots_adjust(left=0.06, right=0.99, top=0.855, bottom=0.09, wspace=0.16, hspace=0.39)
    fig.savefig(path, bbox_inches="tight", dpi=190)
    plt.close(fig)


def main() -> None:
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    jobs = [(measurement, index) for measurement in ("exact", "noisy+nlos") for index in range(len(METHODS))]
    results = []
    with ProcessPoolExecutor(max_workers=3) as executor:
        future_map = {executor.submit(solve_one, measurement, index): (measurement, index) for measurement, index in jobs}
        for future in as_completed(future_map):
            measurement, index = future_map[future]
            result = future.result()
            results.append(result)
            print(
                f"{measurement} | {METHODS[index][0]}: RMSE={result['pair_rmse_m']:.4f} maxOffset={result['max_coord_offset_m']:.3f}",
                flush=True,
            )
    order = {method[0]: i for i, method in enumerate(METHODS)}
    results.sort(key=lambda row: (row["measurement"] != "exact", order[row["method"]]))
    png_path = OUTPUTS / "anchor_solver_corner_comparison.png"
    csv_path = OUTPUTS / "anchor_solver_corner_comparison.csv"
    render(results, png_path)
    write_csv(results, csv_path)
    print(f"Wrote {png_path}")
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
