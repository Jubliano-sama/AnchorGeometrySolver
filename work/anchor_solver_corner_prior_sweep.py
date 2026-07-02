
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
import matplotlib.ticker as mticker
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT / "work" / "SmartClicker-GUI"
OUTPUTS = ROOT / "outputs"
PILOT_PATH = ROOT / "work" / "anchor_solver_pilot.py"
sys.path.insert(0, str(REPO))

from uwb_capture.anchor_geometry import AnchorPairDistance, solve_anchor_layout  # noqa: E402

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
    ("default 24x10", "default", 24, 10, 80, False),
    ("default 24x50", "default", 24, 50, 80, False),
    ("corner seed 24x50", "corner", 24, 50, 80, False),
    ("corner prior 24x50", "corner", 24, 50, 80, True),
]


@dataclass(frozen=True)
class GridCase:
    positions: dict[str, tuple[float, float]]
    rows: int
    cols: int
    exact_pairs: list[AnchorPairDistance]
    noisy_pairs: list[AnchorPairDistance]
    nlos_count: int
    los_noise: tuple[float, ...]
    nlos_offsets: tuple[float, ...]
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
class MethodResult:
    measurement: str
    method: str
    seed_count: int
    basin_hops: int
    prior: bool
    elapsed_s: float
    pair_rmse_m: float
    max_pair_residual_m: float
    max_coord_offset_m: float
    median_coord_offset_m: float
    p95_coord_offset_m: float
    warnings: str


def irregular_grid_positions(rng: random.Random, rows: int = 4, cols: int = 4) -> dict[str, tuple[float, float]]:
    # Independent row/column gaps keep the layout grid-like without perfect-lattice duplicate distances.
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


def exact_pairs(positions: dict[str, tuple[float, float]]) -> list[AnchorPairDistance]:
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


def noisy_pairs(exact: list[AnchorPairDistance], rng: random.Random) -> tuple[list[AnchorPairDistance], int, list[float], list[float]]:
    noisy: list[AnchorPairDistance] = []
    nlos_count = 0
    los_noise: list[float] = []
    nlos_offsets: list[float] = []
    for pair in exact:
        normal_noise = rng.gauss(0.0, NOISE_SIGMA_M)
        positive_bias = 0.0
        source = "los"
        if rng.random() < NLOS_PROBABILITY:
            positive_bias = rng.uniform(0.0, NLOS_MAX_OFFSET_M)
            nlos_offsets.append(positive_bias)
            nlos_count += 1
            source = "nlos"
        else:
            los_noise.append(normal_noise)
        noisy.append(
            AnchorPairDistance(
                pair.anchor_a_id,
                pair.anchor_b_id,
                max(pair.distance_m + normal_noise + positive_bias, 0.05),
                sigma_m=PAIR_SIGMA_M,
                source=source,
            )
        )
    return noisy, nlos_count, los_noise, nlos_offsets


def make_case() -> GridCase:
    rng = random.Random(RNG_SEED + 401)
    discarded = 0
    while True:
        positions = irregular_grid_positions(rng)
        exact = exact_pairs(positions)
        if pilot.accepted_graph(positions, exact):
            noise_rng = random.Random(RNG_SEED + 402)
            noisy, nlos_count, los_noise, nlos_offsets = noisy_pairs(exact, noise_rng)
            return GridCase(
                positions=positions,
                rows=4,
                cols=4,
                exact_pairs=exact,
                noisy_pairs=noisy,
                nlos_count=nlos_count,
                los_noise=tuple(los_noise),
                nlos_offsets=tuple(nlos_offsets),
                discarded_before=discarded,
            )
        discarded += 1


def custom_anchor_order(case: GridCase, processed) -> list[str]:
    corner_ids = [case.top_left, case.top_right, case.bottom_left]
    all_ids = pilot._anchor_ids(processed)
    return corner_ids + [anchor_id for anchor_id in all_ids if anchor_id not in set(corner_ids)]


def corner_seed(case: GridCase, parameterization) -> list[float]:
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
    return pilot._positions_to_params(parameterization, positions)


def corner_priors(case: GridCase) -> dict[str, tuple[float, float, float]]:
    width = NOMINAL_GRID_SPACING_M * (case.cols - 1)
    height = NOMINAL_GRID_SPACING_M * (case.rows - 1)
    return {
        case.top_left: (0.0, 0.0, CORNER_PRIOR_SIGMA_M),
        case.top_right: (width, 0.0, CORNER_PRIOR_SIGMA_M),
        case.bottom_left: (0.0, height, CORNER_PRIOR_SIGMA_M),
    }


def normal_equations_with_priors(params, parameterization, pairs, priors):
    normal, rhs = pilot._normal_equations(params, parameterization, pairs)
    positions = parameterization.to_positions(params)
    for anchor_id, (prior_x, prior_y, sigma) in priors.items():
        if anchor_id not in positions:
            continue
        weight = 1.0 / (sigma * sigma)
        current_x, current_y = positions[anchor_id]
        for axis, current, prior in (("x", current_x, prior_x), ("y", current_y, prior_y)):
            idx = parameterization.derivative_index(anchor_id, axis)
            if idx is None:
                continue
            normal[idx][idx] += weight
            rhs[idx] -= weight * (current - prior)
    return normal, rhs


def spring_energy_with_priors(params, parameterization, pairs, priors) -> float:
    energy = pilot._spring_energy(params, parameterization, pairs)
    positions = parameterization.to_positions(params)
    for anchor_id, (prior_x, prior_y, sigma) in priors.items():
        if anchor_id not in positions:
            continue
        x, y = positions[anchor_id]
        weight = 1.0 / (sigma * sigma)
        energy += 0.5 * weight * ((x - prior_x) ** 2 + (y - prior_y) ** 2)
    return energy


def local_minimize_with_priors(initial_params, parameterization, pairs, priors, max_iterations):
    params = list(initial_params)
    energy = spring_energy_with_priors(params, parameterization, pairs, priors)
    damping = 1e-3
    for _iteration in range(max(max_iterations, 1)):
        normal, rhs = normal_equations_with_priors(params, parameterization, pairs, priors)
        damped = [row[:] for row in normal]
        for index in range(len(damped)):
            damped[index][index] += damping * max(normal[index][index], 1.0)
        try:
            delta = pilot._solve_linear_system(damped, rhs)
        except ValueError:
            damping *= 10.0
            if damping > 1e12:
                break
            continue
        if pilot._vector_norm(delta) <= 1e-10:
            break
        candidate = [value + step for value, step in zip(params, delta)]
        candidate_energy = spring_energy_with_priors(candidate, parameterization, pairs, priors)
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


def solve_corner_mode(case: GridCase, pairs: list[AnchorPairDistance], seed_count: int, hops: int, iterations: int, use_prior: bool):
    processed = pilot._preprocess_pairs(pairs, min_sigma_m=0.02, min_distance_m=0.05)
    anchor_ids = custom_anchor_order(case, processed)
    parameterization = pilot._Parameterization(anchor_ids)
    scale = pilot._layout_scale(processed)
    rng = random.Random(RNG_SEED + 500 + seed_count * 13 + hops)
    seeds = [corner_seed(case, parameterization)]
    generated = pilot._initial_parameters(parameterization, processed, seed_count=max(seed_count - 1, 1), scale=scale, rng=rng)
    seeds.extend(generated)
    seeds = seeds[:seed_count]
    priors = corner_priors(case) if use_prior else {}

    best_params = None
    best_energy = math.inf
    temperature = max(scale * scale * 1e-5, 1e-8)
    hop_scale = max(scale * 0.35, 0.05)
    for seed_params in seeds:
        if use_prior:
            current_params, current_energy = local_minimize_with_priors(seed_params, parameterization, processed, priors, iterations)
        else:
            current_params, current_energy = pilot._local_minimize(seed_params, parameterization, processed, max_iterations=iterations)
        if current_energy < best_energy:
            best_params = current_params
            best_energy = current_energy
        for _ in range(max(hops, 0)):
            hopped = [value + rng.gauss(0.0, hop_scale) for value in current_params]
            if use_prior:
                candidate_params, candidate_energy = local_minimize_with_priors(hopped, parameterization, processed, priors, iterations)
            else:
                candidate_params, candidate_energy = pilot._local_minimize(hopped, parameterization, processed, max_iterations=iterations)
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
        raise ValueError("corner mode produced no solution")
    positions = parameterization.to_positions(best_params)
    residuals = pilot.pair_residuals(positions, processed)
    pair_rmse = math.sqrt(sum(value * value for value in residuals.values()) / len(residuals))
    max_pair = max(abs(value) for value in residuals.values())
    max_offset, median_offset, p95_offset = pilot.position_error_summary(case.positions, positions)
    return pair_rmse, max_pair, max_offset, median_offset, p95_offset, ()


def solve_default(case: GridCase, pairs: list[AnchorPairDistance], seed_count: int, hops: int, iterations: int):
    result = solve_anchor_layout(
        pairs,
        seed_count=seed_count,
        basin_hops=hops,
        max_iterations=iterations,
        random_seed=RNG_SEED + 600 + seed_count * 13 + hops,
    )
    max_offset, median_offset, p95_offset = pilot.position_error_summary(case.positions, result.positions_m)
    return result.rmse_m, result.max_residual_m, max_offset, median_offset, p95_offset, result.warnings


def run_method(case: GridCase, measurement: str, pairs: list[AnchorPairDistance], method) -> MethodResult:
    label, mode, seeds, hops, iterations, use_prior = method
    t0 = time.time()
    if mode == "default":
        pair_rmse, max_pair, max_offset, median_offset, p95_offset, warnings = solve_default(case, pairs, seeds, hops, iterations)
    else:
        pair_rmse, max_pair, max_offset, median_offset, p95_offset, warnings = solve_corner_mode(case, pairs, seeds, hops, iterations, use_prior)
    elapsed = time.time() - t0
    print(f"{measurement} | {label}: RMSE={pair_rmse:.4f} maxPair={max_pair:.4f} maxCoord={max_offset:.3f} elapsed={elapsed:.1f}s", flush=True)
    return MethodResult(
        measurement=measurement,
        method=label,
        seed_count=seeds,
        basin_hops=hops,
        prior=use_prior,
        elapsed_s=elapsed,
        pair_rmse_m=pair_rmse,
        max_pair_residual_m=max_pair,
        max_coord_offset_m=max_offset,
        median_coord_offset_m=median_offset,
        p95_coord_offset_m=p95_offset,
        warnings="; ".join(warnings),
    )


def write_csv(results: list[MethodResult], path: Path) -> None:
    fields = list(MethodResult.__dataclass_fields__)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fields)
        writer.writeheader()
        for result in results:
            writer.writerow({field: getattr(result, field) for field in fields})


def write_summary(case: GridCase, results: list[MethodResult], path: Path) -> None:
    lines = [
        "# Corner Prior Solver Diagnostic",
        "",
        "This run uses one accepted 4x4 irregular grid-like case. It compares the stock solver with larger hop counts against a mode that uses the named top-left, top-right, and bottom-left anchors as a weak frame.",
        "",
        "## Case Audit",
        "",
        f"- Anchors: {len(case.positions)} ({case.rows}x{case.cols})",
        f"- True pair edges: {len(case.exact_pairs)} using true distance <= {EDGE_RADIUS_M:.1f} m",
        f"- NLOS links: {case.nlos_count}/{len(case.noisy_pairs)} ({case.nlos_count / len(case.noisy_pairs):.0%})",
        f"- LOS Gaussian noise sigma: {NOISE_SIGMA_M:.2f} m",
        f"- NLOS positive offset: uniform(0, {NLOS_MAX_OFFSET_M:.2f}) m",
        f"- Known corner roles: top-left={case.top_left}, top-right={case.top_right}, bottom-left={case.bottom_left}",
        f"- Corner prior coordinates use nominal {NOMINAL_GRID_SPACING_M:.1f} m spacing with sigma {CORNER_PRIOR_SIGMA_M:.1f} m, so they are intentionally approximate.",
        "",
        "## Results",
        "",
        "| measurement | method | pair RMSE | max pair residual | worst anchor offset | p95 anchor offset | elapsed |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        lines.append(
            f"| {result.measurement} | {result.method} | {result.pair_rmse_m:.4f} m | {result.max_pair_residual_m:.4f} m | {result.max_coord_offset_m:.3f} m | {result.p95_coord_offset_m:.3f} m | {result.elapsed_s:.1f}s |"
        )
    lines.extend([
        "",
        "Interpretation cue: low pair RMSE with high coordinate error means the distance constraints were technically fit but did not recover the intended labeled layout. If higher hops reduce RMSE but not coordinate error, the graph/corner information is the issue. If higher hops reduce both, it was mostly a search/local-minimum issue.",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def render_chart(results: list[MethodResult], path: Path) -> None:
    plt.rcParams.update({"figure.facecolor": TOKENS["surface"], "savefig.facecolor": TOKENS["surface"], "font.family": ["Segoe UI", "DejaVu Sans", "Arial", "sans-serif"]})
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.4), dpi=180)
    fig.patch.set_facecolor(TOKENS["surface"])
    fig.text(0.04, 0.96, "Corner Knowledge and Bigger Hops", ha="left", va="top", fontsize=20, fontweight="bold", color=TOKENS["ink"])
    fig.text(0.04, 0.905, "One irregular 4x4 grid-like case. Exact distances test information/search; noisy+NLOS adds the requested measurement model.", ha="left", va="top", fontsize=10, color=TOKENS["muted"])
    method_order = [method[0] for method in METHODS]
    x = np.arange(len(method_order))
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
    for measurement, offset in [("exact", -0.17), ("noisy+nlos", 0.17)]:
        part = [next(r for r in results if r.measurement == measurement and r.method == method) for method in method_order]
        axes[0].bar(x + offset, [r.max_coord_offset_m for r in part], width=0.32, color=colors[measurement], edgecolor=edges[measurement], linewidth=1.0, label=measurement)
        axes[1].bar(x + offset, [r.pair_rmse_m for r in part], width=0.32, color=colors[measurement], edgecolor=edges[measurement], linewidth=1.0, label=measurement)
        for ax, values in [(axes[0], [r.max_coord_offset_m for r in part]), (axes[1], [r.pair_rmse_m for r in part])]:
            for xx, value in zip(x + offset, values):
                ax.text(xx, value, f"{value:.2f}", ha="center", va="bottom", fontsize=7, color=TOKENS["ink"])
    axes[0].set_title("Worst labeled-anchor offset", loc="left", fontsize=12, fontweight="bold", color=TOKENS["ink"])
    axes[1].set_title("Final pair RMSE", loc="left", fontsize=12, fontweight="bold", color=TOKENS["ink"])
    for ax in axes:
        ax.set_xticks(x, method_order, rotation=18, ha="right")
        ax.set_ylabel("meters")
        ax.legend(frameon=False, fontsize=8)
    fig.text(0.04, 0.045, "Corner seed uses top-left/top-right/bottom-left to choose a starting basin. Corner prior adds weak 3 m-sigma coordinate penalties to those same three named anchors.", fontsize=8.5, color=TOKENS["muted"], ha="left")
    fig.subplots_adjust(left=0.07, right=0.98, top=0.78, bottom=0.25, wspace=0.22)
    fig.savefig(path, bbox_inches="tight", dpi=180)
    plt.close(fig)


def main() -> None:
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    case = make_case()
    print(
        f"case: anchors={len(case.positions)} pairs={len(case.exact_pairs)} nlos={case.nlos_count}/{len(case.noisy_pairs)} corners={case.top_left},{case.top_right},{case.bottom_left} discarded={case.discarded_before}",
        flush=True,
    )
    results: list[MethodResult] = []
    for measurement, pairs in (("exact", case.exact_pairs), ("noisy+nlos", case.noisy_pairs)):
        for method in METHODS:
            results.append(run_method(case, measurement, pairs, method))
    write_csv(results, OUTPUTS / "anchor_solver_corner_prior_sweep.csv")
    write_summary(case, results, OUTPUTS / "anchor_solver_corner_prior_summary.md")
    render_chart(results, OUTPUTS / "anchor_solver_corner_prior_sweep.png")
    print(f"Wrote {OUTPUTS / 'anchor_solver_corner_prior_sweep.png'}")
    print(f"Wrote {OUTPUTS / 'anchor_solver_corner_prior_summary.md'}")
    print(f"Wrote {OUTPUTS / 'anchor_solver_corner_prior_sweep.csv'}")


if __name__ == "__main__":
    main()
