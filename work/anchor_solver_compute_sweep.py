
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

COMPUTE_SETTINGS = [
    ("pilot 8x3", 8, 3, 60),
    ("gui default 24x10", 24, 10, 80),
    ("heavy 48x12", 48, 12, 100),
]
EXACT_SETTINGS = [
    ("gui default 24x10", 24, 10, 80),
    ("heavy 48x12", 48, 12, 100),
]


@dataclass(frozen=True)
class CaseData:
    family: str
    case_id: str
    positions: dict[str, tuple[float, float]]
    exact_pairs: list[AnchorPairDistance]
    noisy_pairs: list[AnchorPairDistance]
    nlos_count: int
    discarded_before: int


@dataclass(frozen=True)
class SweepResult:
    family: str
    case_id: str
    measurement: str
    setting: str
    seed_count: int
    basin_hops: int
    max_iterations: int
    anchor_count: int
    pair_count: int
    nlos_share: float
    elapsed_s: float
    pair_rmse_m: float
    max_pair_residual_m: float
    max_coord_offset_m: float
    median_coord_offset_m: float
    p95_coord_offset_m: float
    warnings: str


def grid_dimensions(rng: random.Random) -> tuple[int, int]:
    choices = [(4, 4), (4, 5), (5, 4), (4, 6), (6, 4), (5, 5), (5, 6), (6, 5), (4, 7), (7, 4), (4, 8), (8, 4)]
    return rng.choice(choices)


def irregular_grid_positions(rng: random.Random) -> dict[str, tuple[float, float]]:
    rows, cols = grid_dimensions(rng)
    x = [0.0]
    y = [0.0]
    for _ in range(cols - 1):
        x.append(x[-1] + rng.uniform(4.0, 8.0))
    for _ in range(rows - 1):
        y.append(y[-1] + rng.uniform(4.0, 8.0))
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
            d = math.hypot(ax - bx, ay - by)
            if d <= EDGE_RADIUS_M:
                pairs.append(AnchorPairDistance(a, b, d, sigma_m=PAIR_SIGMA_M, source="exact"))
    return pairs


def noisy_pairs_from_exact(exact_pairs: list[AnchorPairDistance], rng: random.Random) -> tuple[list[AnchorPairDistance], int, list[float], list[float]]:
    pairs: list[AnchorPairDistance] = []
    nlos_count = 0
    los_errors: list[float] = []
    nlos_offsets: list[float] = []
    for pair in exact_pairs:
        los_noise = rng.gauss(0.0, NOISE_SIGMA_M)
        nlos_offset = 0.0
        source = "los"
        if rng.random() < NLOS_PROBABILITY:
            nlos_offset = rng.uniform(0.0, NLOS_MAX_OFFSET_M)
            nlos_count += 1
            source = "nlos"
            nlos_offsets.append(nlos_offset)
        else:
            los_errors.append(los_noise)
        pairs.append(
            AnchorPairDistance(
                pair.anchor_a_id,
                pair.anchor_b_id,
                max(pair.distance_m + los_noise + nlos_offset, 0.05),
                sigma_m=PAIR_SIGMA_M,
                source=source,
            )
        )
    return pairs, nlos_count, los_errors, nlos_offsets


def make_random_case() -> CaseData:
    rng = random.Random(RNG_SEED + 17)
    discarded = 0
    while True:
        positions = pilot.generate_random_positions(rng, n=30)
        exact_pairs = exact_pairs_from_positions(positions)
        if pilot.accepted_graph(positions, exact_pairs):
            noise_rng = random.Random(RNG_SEED + 18)
            noisy_pairs, nlos_count, _los_errors, _nlos_offsets = noisy_pairs_from_exact(exact_pairs, noise_rng)
            return CaseData("random", "accepted random 30", positions, exact_pairs, noisy_pairs, nlos_count, discarded)
        discarded += 1


def make_grid_case() -> CaseData:
    rng = random.Random(RNG_SEED + 29)
    discarded = 0
    while True:
        positions = irregular_grid_positions(rng)
        exact_pairs = exact_pairs_from_positions(positions)
        if pilot.accepted_graph(positions, exact_pairs):
            noise_rng = random.Random(RNG_SEED + 30)
            noisy_pairs, nlos_count, _los_errors, _nlos_offsets = noisy_pairs_from_exact(exact_pairs, noise_rng)
            return CaseData("irregular grid", "accepted grid-like", positions, exact_pairs, noisy_pairs, nlos_count, discarded)
        discarded += 1


def solve_case(case: CaseData, measurement: str, pairs: list[AnchorPairDistance], setting: tuple[str, int, int, int]) -> SweepResult:
    label, seeds, hops, iterations = setting
    t0 = time.time()
    result = solve_anchor_layout(
        pairs,
        seed_count=seeds,
        basin_hops=hops,
        max_iterations=iterations,
        random_seed=RNG_SEED + hash((case.family, measurement, label)) % 100000,
    )
    elapsed = time.time() - t0
    max_offset, median_offset, p95_offset = pilot.position_error_summary(case.positions, result.positions_m)
    print(
        f"{case.family} | {measurement} | {label}: "
        f"rmse={result.rmse_m:.4f}m max_pair={result.max_residual_m:.4f}m "
        f"max_coord={max_offset:.3f}m elapsed={elapsed:.1f}s",
        flush=True,
    )
    return SweepResult(
        family=case.family,
        case_id=case.case_id,
        measurement=measurement,
        setting=label,
        seed_count=seeds,
        basin_hops=hops,
        max_iterations=iterations,
        anchor_count=len(case.positions),
        pair_count=len(pairs),
        nlos_share=case.nlos_count / max(len(case.noisy_pairs), 1) if measurement == "noisy+nlos" else 0.0,
        elapsed_s=elapsed,
        pair_rmse_m=result.rmse_m,
        max_pair_residual_m=result.max_residual_m,
        max_coord_offset_m=max_offset,
        median_coord_offset_m=median_offset,
        p95_coord_offset_m=p95_offset,
        warnings="; ".join(result.warnings),
    )


def write_csv(results: list[SweepResult], path: Path) -> None:
    fields = list(SweepResult.__dataclass_fields__)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fields)
        writer.writeheader()
        for result in results:
            writer.writerow({field: getattr(result, field) for field in fields})


def write_summary(cases: list[CaseData], results: list[SweepResult], path: Path) -> None:
    lines = [
        "# Anchor Solver Compute Sweep",
        "",
        "This is a diagnostic rerun, not a large simulation. It checks whether higher seed/hop counts reduce pair RMSE and coordinate error on representative accepted cases.",
        "",
        "## Case Audit",
        "",
        "| family | anchors | true edges | NLOS share | discarded candidates | edge rule | noise rule |",
        "|---|---:|---:|---:|---:|---|---|",
    ]
    for case in cases:
        lines.append(
            f"| {case.family} | {len(case.positions)} | {len(case.exact_pairs)} | {case.nlos_count / max(len(case.noisy_pairs), 1):.0%} | {case.discarded_before} | true distance <= {EDGE_RADIUS_M:.1f} m | Gaussian sigma {NOISE_SIGMA_M:.2f} m plus +uniform(0,{NLOS_MAX_OFFSET_M:.2f}) m on one-third links |"
        )
    lines.extend([
        "",
        "## Results",
        "",
        "| family | measurement | setting | pair RMSE | max pair residual | worst anchor offset | elapsed |",
        "|---|---|---|---:|---:|---:|---:|",
    ])
    order = {name: i for i, (name, _s, _h, _it) in enumerate(COMPUTE_SETTINGS + EXACT_SETTINGS)}
    for result in sorted(results, key=lambda r: (r.family, r.measurement, order.get(r.setting, 99))):
        lines.append(
            f"| {result.family} | {result.measurement} | {result.setting} | {result.pair_rmse_m:.4f} m | {result.max_pair_residual_m:.4f} m | {result.max_coord_offset_m:.3f} m | {result.elapsed_s:.1f}s |"
        )
    lines.extend([
        "",
        "Interpretation cue: low pair RMSE with high coordinate error means a technically distance-fitting but label-wise wrong/ambiguous embedding. High pair RMSE means the optimizer did not find a good distance fit at that compute budget.",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def render_chart(results: list[SweepResult], path: Path) -> None:
    plt.rcParams.update({"figure.facecolor": TOKENS["surface"], "savefig.facecolor": TOKENS["surface"], "font.family": ["Segoe UI", "DejaVu Sans", "Arial", "sans-serif"]})
    fig, axes = plt.subplots(2, 2, figsize=(14, 8.5), dpi=180)
    fig.patch.set_facecolor(TOKENS["surface"])
    fig.text(0.04, 0.975, "Compute Sweep: RMSE Versus Shape Error", ha="left", va="top", fontsize=20, fontweight="bold", color=TOKENS["ink"])
    fig.text(0.04, 0.935, "Exact and noisy/NLOS distances on representative accepted random and irregular-grid cases; higher budgets test local-minimum sensitivity.", ha="left", va="top", fontsize=10, color=TOKENS["muted"])

    families = ["irregular grid", "random"]
    colors = {"exact": BLUE["base"], "noisy+nlos": ORANGE["base"]}
    edges = {"exact": BLUE["dark"], "noisy+nlos": ORANGE["dark"]}
    for row, family in enumerate(families):
        family_results = [r for r in results if r.family == family]
        settings = []
        for r in family_results:
            if r.setting not in settings:
                settings.append(r.setting)
        x = np.arange(len(settings))
        ax_err = axes[row, 0]
        ax_rmse = axes[row, 1]
        for ax in (ax_err, ax_rmse):
            ax.set_facecolor(TOKENS["panel"])
            ax.grid(True, color=TOKENS["grid"], linewidth=0.8)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_color(TOKENS["axis"])
            ax.spines["bottom"].set_color(TOKENS["axis"])
            ax.tick_params(colors=TOKENS["muted"], labelsize=8)
        for measurement, offset in [("exact", -0.16), ("noisy+nlos", 0.16)]:
            part = [next((r for r in family_results if r.measurement == measurement and r.setting == setting), None) for setting in settings]
            xs = [i + offset for i, r in enumerate(part) if r is not None]
            err_vals = [r.max_coord_offset_m for r in part if r is not None]
            rmse_vals = [r.pair_rmse_m for r in part if r is not None]
            ax_err.plot(xs, err_vals, marker="o", color=colors[measurement], markeredgecolor=edges[measurement], linewidth=1.0, label=measurement)
            ax_rmse.plot(xs, rmse_vals, marker="o", color=colors[measurement], markeredgecolor=edges[measurement], linewidth=1.0, label=measurement)
            for xx, yy in zip(xs, err_vals):
                ax_err.text(xx, yy, f"{yy:.1f}", fontsize=7, ha="center", va="bottom", color=TOKENS["ink"])
            for xx, yy in zip(xs, rmse_vals):
                ax_rmse.text(xx, yy, f"{yy:.2f}", fontsize=7, ha="center", va="bottom", color=TOKENS["ink"])
        ax_err.set_title(f"{family}: worst anchor offset", loc="left", fontsize=11, fontweight="bold", color=TOKENS["ink"])
        ax_rmse.set_title(f"{family}: final pair RMSE", loc="left", fontsize=11, fontweight="bold", color=TOKENS["ink"])
        ax_err.set_ylabel("meters")
        ax_rmse.set_ylabel("meters")
        ax_err.set_xticks(x, settings, rotation=15, ha="right")
        ax_rmse.set_xticks(x, settings, rotation=15, ha="right")
        ax_err.legend(frameon=False, fontsize=8)
        ax_rmse.legend(frameon=False, fontsize=8)
    fig.text(0.04, 0.035, "Low RMSE + high coordinate error points to a distance-equivalent or label-ambiguous embedding; high RMSE points to optimizer/search failure.", fontsize=8.5, color=TOKENS["muted"], ha="left")
    fig.subplots_adjust(left=0.07, right=0.98, top=0.86, bottom=0.13, wspace=0.22, hspace=0.42)
    fig.savefig(path, bbox_inches="tight", dpi=180)
    plt.close(fig)


def main() -> None:
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    cases = [make_grid_case(), make_random_case()]
    for case in cases:
        print(
            f"case {case.family}: anchors={len(case.positions)} pairs={len(case.exact_pairs)} nlos={case.nlos_count}/{len(case.noisy_pairs)} discarded={case.discarded_before}",
            flush=True,
        )
    results: list[SweepResult] = []
    for case in cases:
        for setting in EXACT_SETTINGS:
            results.append(solve_case(case, "exact", case.exact_pairs, setting))
        for setting in COMPUTE_SETTINGS:
            results.append(solve_case(case, "noisy+nlos", case.noisy_pairs, setting))
    write_csv(results, OUTPUTS / "anchor_solver_compute_sweep.csv")
    write_summary(cases, results, OUTPUTS / "anchor_solver_compute_sweep_summary.md")
    render_chart(results, OUTPUTS / "anchor_solver_compute_sweep.png")
    print(f"Wrote {OUTPUTS / 'anchor_solver_compute_sweep.png'}")
    print(f"Wrote {OUTPUTS / 'anchor_solver_compute_sweep_summary.md'}")
    print(f"Wrote {OUTPUTS / 'anchor_solver_compute_sweep.csv'}")


if __name__ == "__main__":
    main()
