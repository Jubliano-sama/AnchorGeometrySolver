
from __future__ import annotations

import csv
from dataclasses import dataclass
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

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT / "work" / "SmartClicker-GUI"
OUTPUTS = ROOT / "outputs"
PILOT_PATH = ROOT / "work" / "anchor_solver_pilot.py"
sys.path.insert(0, str(REPO))

from uwb_capture.anchor_geometry import (  # noqa: E402
    AnchorPairDistance,
    _normal_equations,
    _positions_to_params,
    _solve_linear_system,
    _vector_norm,
)

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

EDGE_RADIUS_M = 8.0
NOISE_SIGMA_M = 0.03
NLOS_PROBABILITY = 1.0 / 3.0
NLOS_MAX_OFFSET_M = 0.20
PAIR_SIGMA_M = 0.05
RNG_SEED = 20260626
NOMINAL_GRID_SPACING_M = 6.0
CORNER_PRIOR_SIGMA_M = 3.0

METHODS = [
    # label, seeds, hops, hop multiplier, swaps, corner seed, corner prior
    ("stock hops", 24, 10, 0.35, False, False, False),
    ("larger hops", 24, 10, 1.50, False, False, False),
    ("swap hops", 24, 10, 0.35, True, False, False),
    ("large + swap", 24, 10, 1.50, True, False, False),
    ("corner large + swap", 24, 10, 1.50, True, True, False),
    ("corner prior + swap", 24, 10, 1.50, True, True, True),
]


@dataclass(frozen=True)
class GridCase:
    positions: dict[str, tuple[float, float]]
    rows: int
    cols: int
    exact_pairs: list[AnchorPairDistance]
    noisy_pairs: list[AnchorPairDistance]
    nlos_count: int
    discarded_before: int

    @property
    def top_left(self) -> str:
        return "A00"

    @property
    def top_right(self) -> str:
        return f"A{self.cols - 1:02d}"

    @property
    def bottom_left(self) -> str:
        return f"A{(self.rows - 1) * self.cols:02d}"


@dataclass(frozen=True)
class SearchResult:
    measurement: str
    method: str
    seed_count: int
    basin_hops: int
    hop_multiplier: float
    swaps: bool
    corner_seed: bool
    corner_prior: bool
    elapsed_s: float
    anchor_count: int
    pair_count: int
    nlos_share: float
    pair_rmse_m: float
    max_pair_residual_m: float
    max_coord_offset_m: float
    median_coord_offset_m: float
    p95_coord_offset_m: float
    energy: float


def irregular_grid_positions(rng: random.Random, rows: int = 4, cols: int = 4) -> dict[str, tuple[float, float]]:
    # Independent row/column gaps: grid-like, but not a perfect repeated-distance lattice.
    x = [0.0]
    y = [0.0]
    for _ in range(cols - 1):
        x.append(x[-1] + rng.uniform(4.05, 5.25))
    for _ in range(rows - 1):
        y.append(y[-1] + rng.uniform(4.05, 5.25))
    positions: dict[str, tuple[float, float]] = {}
    index = 0
    for row in range(rows):
        for col in range(cols):
            positions[f"A{index:02d}"] = (x[col], y[row])
            index += 1
    return positions


def exact_pairs_from_positions(positions: dict[str, tuple[float, float]]) -> list[AnchorPairDistance]:
    pairs: list[AnchorPairDistance] = []
    ids = sorted(positions)
    for i, a in enumerate(ids):
        ax, ay = positions[a]
        for b in ids[i + 1 :]:
            bx, by = positions[b]
            distance = math.hypot(ax - bx, ay - by)
            if distance <= EDGE_RADIUS_M:
                pairs.append(AnchorPairDistance(a, b, distance, sigma_m=PAIR_SIGMA_M, source="exact"))
    return pairs


def noisy_pairs_from_exact(exact_pairs: list[AnchorPairDistance], rng: random.Random) -> tuple[list[AnchorPairDistance], int]:
    pairs: list[AnchorPairDistance] = []
    nlos_count = 0
    for pair in exact_pairs:
        measurement = pair.distance_m + rng.gauss(0.0, NOISE_SIGMA_M)
        source = "los"
        if rng.random() < NLOS_PROBABILITY:
            measurement += rng.uniform(0.0, NLOS_MAX_OFFSET_M)
            nlos_count += 1
            source = "nlos"
        pairs.append(
            AnchorPairDistance(
                pair.anchor_a_id,
                pair.anchor_b_id,
                max(measurement, 0.05),
                sigma_m=PAIR_SIGMA_M,
                source=source,
            )
        )
    return pairs, nlos_count


def make_case() -> GridCase:
    rng = random.Random(RNG_SEED + 701)
    discarded = 0
    while True:
        positions = irregular_grid_positions(rng)
        exact_pairs = exact_pairs_from_positions(positions)
        if pilot.accepted_graph(positions, exact_pairs):
            noisy_pairs, nlos_count = noisy_pairs_from_exact(exact_pairs, random.Random(RNG_SEED + 702))
            return GridCase(
                positions=positions,
                rows=4,
                cols=4,
                exact_pairs=exact_pairs,
                noisy_pairs=noisy_pairs,
                nlos_count=nlos_count,
                discarded_before=discarded,
            )
        discarded += 1


def anchor_order(case: GridCase, processed, corner_seed: bool) -> list[str]:
    ids = pilot._anchor_ids(processed)
    if not corner_seed:
        return ids
    corners = [case.top_left, case.top_right, case.bottom_left]
    return corners + [anchor_id for anchor_id in ids if anchor_id not in set(corners)]


def nominal_corner_positions(case: GridCase) -> dict[str, tuple[float, float]]:
    width = NOMINAL_GRID_SPACING_M * (case.cols - 1)
    height = NOMINAL_GRID_SPACING_M * (case.rows - 1)
    return {
        case.top_left: (0.0, 0.0),
        case.top_right: (width, 0.0),
        case.bottom_left: (0.0, height),
    }


def corner_seed_params(case: GridCase, parameterization) -> list[float]:
    width = NOMINAL_GRID_SPACING_M * (case.cols - 1)
    height = NOMINAL_GRID_SPACING_M * (case.rows - 1)
    positions: dict[str, tuple[float, float]] = {}
    for row in range(case.rows):
        for col in range(case.cols):
            anchor_id = f"A{row * case.cols + col:02d}"
            positions[anchor_id] = (
                width * col / max(case.cols - 1, 1),
                height * row / max(case.rows - 1, 1),
            )
    return _positions_to_params(parameterization, positions)


def prior_map(case: GridCase) -> dict[str, tuple[float, float, float]]:
    return {
        anchor_id: (xy[0], xy[1], CORNER_PRIOR_SIGMA_M)
        for anchor_id, xy in nominal_corner_positions(case).items()
    }


def energy_with_priors(params, parameterization, pairs, priors) -> float:
    energy = pilot._spring_energy(params, parameterization, pairs)
    positions = parameterization.to_positions(params)
    for anchor_id, (prior_x, prior_y, sigma) in priors.items():
        if anchor_id not in positions:
            continue
        x, y = positions[anchor_id]
        weight = 1.0 / (sigma * sigma)
        energy += 0.5 * weight * ((x - prior_x) ** 2 + (y - prior_y) ** 2)
    return energy


def normal_with_priors(params, parameterization, pairs, priors):
    normal, rhs = _normal_equations(params, parameterization, pairs)
    positions = parameterization.to_positions(params)
    for anchor_id, (prior_x, prior_y, sigma) in priors.items():
        if anchor_id not in positions:
            continue
        x, y = positions[anchor_id]
        weight = 1.0 / (sigma * sigma)
        for axis, current, prior in (("x", x, prior_x), ("y", y, prior_y)):
            idx = parameterization.derivative_index(anchor_id, axis)
            if idx is None:
                continue
            normal[idx][idx] += weight
            rhs[idx] -= weight * (current - prior)
    return normal, rhs


def local_minimize(params, parameterization, pairs, priors, iterations):
    params = list(params)
    energy = energy_with_priors(params, parameterization, pairs, priors)
    damping = 1e-3
    for _ in range(max(iterations, 1)):
        normal, rhs = normal_with_priors(params, parameterization, pairs, priors)
        damped = [row[:] for row in normal]
        for index in range(len(damped)):
            damped[index][index] += damping * max(normal[index][index], 1.0)
        try:
            delta = _solve_linear_system(damped, rhs)
        except ValueError:
            damping *= 10.0
            if damping > 1e12:
                break
            continue
        if _vector_norm(delta) <= 1e-10:
            break
        candidate = [value + step for value, step in zip(params, delta)]
        candidate_energy = energy_with_priors(candidate, parameterization, pairs, priors)
        if candidate_energy <= energy:
            params = candidate
            if abs(energy - candidate_energy) <= 1e-14:
                energy = candidate_energy
                break
            energy = candidate_energy
            damping = max(damping * 0.35, 1e-12)
        else:
            damping *= 4.0
            if damping > 1e12:
                break
    return params, energy


def swappable_anchor_ids(case: GridCase, parameterization, corner_seed: bool) -> list[str]:
    protected = {case.top_left, case.top_right, case.bottom_left} if corner_seed else set()
    ids = []
    for anchor_id in parameterization.anchor_ids:
        if anchor_id in protected:
            continue
        if parameterization.derivative_index(anchor_id, "x") is not None and parameterization.derivative_index(anchor_id, "y") is not None:
            ids.append(anchor_id)
    return ids


def swapped_params(params, parameterization, rng: random.Random, allowed_ids: list[str]) -> list[float]:
    if len(allowed_ids) < 2:
        return list(params)
    a, b = rng.sample(allowed_ids, 2)
    positions = parameterization.to_positions(params)
    positions[a], positions[b] = positions[b], positions[a]
    return _positions_to_params(parameterization, positions)


def proposed_params(params, parameterization, rng: random.Random, hop_sigma: float, use_swaps: bool, allowed_ids: list[str]) -> list[float]:
    proposal = [value + rng.gauss(0.0, hop_sigma) for value in params]
    if use_swaps and rng.random() < 0.85:
        proposal = swapped_params(proposal, parameterization, rng, allowed_ids)
    return proposal


def solve_search(case: GridCase, pairs: list[AnchorPairDistance], method) -> tuple[dict[str, tuple[float, float]], float, float, float]:
    label, seed_count, hops, hop_multiplier, use_swaps, use_corner_seed, use_corner_prior = method
    processed = pilot._preprocess_pairs(pairs, min_sigma_m=0.02, min_distance_m=0.05)
    order = anchor_order(case, processed, use_corner_seed)
    pilot._validate_connected(pilot._anchor_ids(processed), processed)
    parameterization = pilot._Parameterization(order)
    scale = pilot._layout_scale(processed)
    rng = random.Random(RNG_SEED + abs(hash(label)) % 100000)
    seeds: list[list[float]] = []
    if use_corner_seed:
        seeds.append(corner_seed_params(case, parameterization))
    seeds.extend(
        pilot._initial_parameters(
            parameterization,
            processed,
            seed_count=max(seed_count - len(seeds), 1),
            scale=scale,
            rng=rng,
        )
    )
    seeds = seeds[:seed_count]
    priors = prior_map(case) if use_corner_prior else {}
    allowed_swaps = swappable_anchor_ids(case, parameterization, use_corner_seed)
    hop_sigma = max(scale * hop_multiplier, 0.05)
    temperature = max(scale * scale * 1e-5, 1e-8)

    best_params = None
    best_energy = math.inf
    for seed_params in seeds:
        current_params, current_energy = local_minimize(seed_params, parameterization, processed, priors, pilot.SOLVER_MAX_ITERATIONS)
        if current_energy < best_energy:
            best_params = current_params
            best_energy = current_energy
        for _ in range(max(hops, 0)):
            candidate_start = proposed_params(current_params, parameterization, rng, hop_sigma, use_swaps, allowed_swaps)
            candidate_params, candidate_energy = local_minimize(candidate_start, parameterization, processed, priors, pilot.SOLVER_MAX_ITERATIONS)
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


def run_method(case: GridCase, measurement: str, pairs: list[AnchorPairDistance], method) -> SearchResult:
    label, seed_count, hops, hop_multiplier, swaps, corner_seed, corner_prior = method
    t0 = time.time()
    positions, energy, pair_rmse, max_pair = solve_search(case, pairs, method)
    elapsed = time.time() - t0
    max_offset, median_offset, p95_offset = pilot.position_error_summary(case.positions, positions)
    print(
        f"{measurement} | {label}: RMSE={pair_rmse:.4f} maxPair={max_pair:.4f} maxCoord={max_offset:.3f} elapsed={elapsed:.1f}s",
        flush=True,
    )
    return SearchResult(
        measurement=measurement,
        method=label,
        seed_count=seed_count,
        basin_hops=hops,
        hop_multiplier=hop_multiplier,
        swaps=swaps,
        corner_seed=corner_seed,
        corner_prior=corner_prior,
        elapsed_s=elapsed,
        anchor_count=len(case.positions),
        pair_count=len(pairs),
        nlos_share=case.nlos_count / max(len(case.noisy_pairs), 1) if measurement == "noisy+nlos" else 0.0,
        pair_rmse_m=pair_rmse,
        max_pair_residual_m=max_pair,
        max_coord_offset_m=max_offset,
        median_coord_offset_m=median_offset,
        p95_coord_offset_m=p95_offset,
        energy=energy,
    )


def write_csv(results: list[SearchResult], path: Path) -> None:
    fields = list(SearchResult.__dataclass_fields__)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fields)
        writer.writeheader()
        for result in results:
            writer.writerow({field: getattr(result, field) for field in fields})


def write_summary(case: GridCase, results: list[SearchResult], path: Path) -> None:
    lines = [
        "# Large-Hop And Swap Search Diagnostic",
        "",
        "This run tests whether the bad grid-like solves are local-minimum/search failures by adding larger basin proposals and explicit anchor-position swaps. It also tests a three-corner frame seed/prior.",
        "",
        "## Case Audit",
        "",
        f"- Layout: irregular 4x4 grid-like case, {len(case.positions)} anchors",
        f"- Edge rule: true distance <= {EDGE_RADIUS_M:.1f} m",
        f"- True edges: {len(case.exact_pairs)}",
        f"- NLOS links in noisy case: {case.nlos_count}/{len(case.noisy_pairs)} ({case.nlos_count / len(case.noisy_pairs):.0%})",
        f"- Noise: Gaussian sigma {NOISE_SIGMA_M:.2f} m; NLOS adds +uniform(0,{NLOS_MAX_OFFSET_M:.2f}) m",
        f"- Known corner roles: top-left={case.top_left}, top-right={case.top_right}, bottom-left={case.bottom_left}",
        f"- Corner prior: nominal {NOMINAL_GRID_SPACING_M:.1f} m spacing, sigma {CORNER_PRIOR_SIGMA_M:.1f} m",
        "",
        "## Results",
        "",
        "| measurement | method | hop sigma multiplier | swaps | corner seed | corner prior | pair RMSE | max pair residual | worst anchor offset | elapsed |",
        "|---|---|---:|---|---|---|---:|---:|---:|---:|",
    ]
    for result in results:
        lines.append(
            f"| {result.measurement} | {result.method} | {result.hop_multiplier:.2f} | {result.swaps} | {result.corner_seed} | {result.corner_prior} | {result.pair_rmse_m:.4f} m | {result.max_pair_residual_m:.4f} m | {result.max_coord_offset_m:.3f} m | {result.elapsed_s:.1f}s |"
        )
    lines.extend([
        "",
        "Interpretation cue: high RMSE means the optimizer did not fit the measured distances. Low RMSE plus high coordinate offset means the distances were technically fit, but the labeled layout is still wrong or ambiguous under the available constraints.",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def render_chart(results: list[SearchResult], path: Path) -> None:
    plt.rcParams.update({"figure.facecolor": TOKENS["surface"], "savefig.facecolor": TOKENS["surface"], "font.family": ["Segoe UI", "DejaVu Sans", "Arial", "sans-serif"]})
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.7), dpi=180)
    fig.patch.set_facecolor(TOKENS["surface"])
    fig.text(0.04, 0.965, "Large Hops, Swaps, And Three-Corner Knowledge", ha="left", va="top", fontsize=20, fontweight="bold", color=TOKENS["ink"])
    fig.text(0.04, 0.91, "One irregular grid-like 4x4 case. Bars compare labeled-coordinate recovery and pair-distance fit for exact and noisy/NLOS measurements.", ha="left", va="top", fontsize=10, color=TOKENS["muted"])
    methods = [method[0] for method in METHODS]
    x = np.arange(len(methods))
    colors = {"exact": BLUE["base"], "noisy+nlos": ORANGE["base"]}
    edges = {"exact": BLUE["dark"], "noisy+nlos": ORANGE["dark"]}
    for ax in axes:
        ax.set_facecolor(TOKENS["panel"])
        ax.grid(True, axis="y", color=TOKENS["grid"], linewidth=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color(TOKENS["axis"])
        ax.spines["bottom"].set_color(TOKENS["axis"])
        ax.tick_params(colors=TOKENS["muted"], labelsize=8)
    for measurement, offset in [("exact", -0.18), ("noisy+nlos", 0.18)]:
        part = [next(result for result in results if result.measurement == measurement and result.method == method) for method in methods]
        coord_values = [result.max_coord_offset_m for result in part]
        rmse_values = [result.pair_rmse_m for result in part]
        axes[0].bar(x + offset, coord_values, width=0.34, color=colors[measurement], edgecolor=edges[measurement], linewidth=1.0, label=measurement)
        axes[1].bar(x + offset, rmse_values, width=0.34, color=colors[measurement], edgecolor=edges[measurement], linewidth=1.0, label=measurement)
        for ax, values in ((axes[0], coord_values), (axes[1], rmse_values)):
            for xx, value in zip(x + offset, values):
                ax.text(xx, value, f"{value:.2f}", ha="center", va="bottom", fontsize=7, color=TOKENS["ink"])
    axes[0].set_title("Worst labeled-anchor coordinate offset", loc="left", fontsize=12, fontweight="bold", color=TOKENS["ink"])
    axes[1].set_title("Final pair RMSE", loc="left", fontsize=12, fontweight="bold", color=TOKENS["ink"])
    for ax in axes:
        ax.set_ylabel("meters")
        ax.set_xticks(x, methods, rotation=20, ha="right")
        ax.legend(frameon=False, fontsize=8)
    fig.text(0.04, 0.055, "Swap proposals exchange two free anchor positions before local minimization. Corner methods protect the three named corners and use them as an approximate frame.", ha="left", va="bottom", fontsize=8.5, color=TOKENS["muted"])
    fig.subplots_adjust(left=0.07, right=0.985, top=0.80, bottom=0.27, wspace=0.24)
    fig.savefig(path, bbox_inches="tight", dpi=180)
    plt.close(fig)


def main() -> None:
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    case = make_case()
    print(
        f"case: anchors={len(case.positions)} edges={len(case.exact_pairs)} nlos={case.nlos_count}/{len(case.noisy_pairs)} discarded={case.discarded_before} corners={case.top_left},{case.top_right},{case.bottom_left}",
        flush=True,
    )
    results: list[SearchResult] = []
    for measurement, pairs in (("exact", case.exact_pairs), ("noisy+nlos", case.noisy_pairs)):
        for method in METHODS:
            results.append(run_method(case, measurement, pairs, method))
    write_csv(results, OUTPUTS / "anchor_solver_large_hop_swap_sweep.csv")
    write_summary(case, results, OUTPUTS / "anchor_solver_large_hop_swap_summary.md")
    render_chart(results, OUTPUTS / "anchor_solver_large_hop_swap_sweep.png")
    print(f"Wrote {OUTPUTS / 'anchor_solver_large_hop_swap_sweep.png'}")
    print(f"Wrote {OUTPUTS / 'anchor_solver_large_hop_swap_summary.md'}")
    print(f"Wrote {OUTPUTS / 'anchor_solver_large_hop_swap_sweep.csv'}")


if __name__ == "__main__":
    main()
