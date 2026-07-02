from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import math
from pathlib import Path
import random
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
sys.path.insert(0, str(ROOT / "work"))

import anchor_solver_ml_distance_completion as dc  # noqa: E402


OFFICE_BUCKET_SHAPES = ("corridor", "l_shape", "t_shape", "u_shape", "cross", "hollow_square", "rooms")


TOKENS = {
    "surface": "#FCFCFD",
    "panel": "#FFFFFF",
    "ink": "#1F2430",
    "muted": "#6F768A",
    "grid": "#E6E8F0",
    "axis": "#D7DBE7",
}
BLUE = {"base": "#A3BEFA", "dark": "#2E4780"}
PINK = {"base": "#F390CA", "dark": "#8A3A6F"}
GOLD = {"base": "#FFE15B", "dark": "#736422"}
ORANGE = {"base": "#F0986E", "dark": "#804126"}
NEUTRAL = {"light": "#E2E5EA", "base": "#C5CAD3", "dark": "#464C55"}


@dataclass(frozen=True)
class CaseSpec:
    bucket: str
    family: str
    shape: str
    points: torch.Tensor


@dataclass(frozen=True)
class SolvedCase:
    bucket: str
    method: str
    case_index: int
    family: str
    shape: str
    anchors: int
    known_pairs: int
    full_pairs: int
    missing_mae_m: float
    known_rmse_m: float
    known_max_residual_m: float
    max_offset_m: float
    median_offset_m: float
    p95_offset_m: float
    truth: dict[str, tuple[float, float]]
    known_pairs_list: list[dc.AnchorPairDistance]
    estimate: dict[str, tuple[float, float]]


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested, but torch.cuda.is_available() is false.")
    return device


def fair_points(
    points: torch.Tensor,
    *,
    min_nodes: int,
    max_nodes: int | None = None,
    min_vertex_connectivity: int = 1,
) -> torch.Tensor | None:
    points = dc.random_rigid_transform(points)
    points = dc.prune_to_fair_measured_graph(points, min_vertex_connectivity=min_vertex_connectivity)
    if points is None or points.shape[0] < min_nodes:
        return None
    if max_nodes is not None and points.shape[0] > max_nodes:
        return None
    return points


def generate_random_rectangle(device: torch.device) -> CaseSpec:
    for _attempt in range(5000):
        target_n = random.randint(16, 32)
        width = random.uniform(13.0, 25.0)
        height = random.uniform(13.0, 25.0)
        params = {"width": width, "height": height}
        points = dc.random_points_in_shape("rectangle", params, target_n, device)
        if points is None:
            continue
        points = fair_points(points, min_nodes=16, max_nodes=32)
        if points is not None:
            return CaseSpec("Random 16-32", "random", "rectangle", points)
    raise RuntimeError("Could not generate fair random rectangle case.")


def generate_grid_rectangle(device: torch.device) -> CaseSpec:
    for _attempt in range(5000):
        width = random.uniform(16.0, 38.0)
        height = random.uniform(16.0, 38.0)
        target_n = random.randint(16, 50)
        params = {"width": width, "height": height}
        points = dc.grid_points_in_shape("rectangle", params, target_n, device)
        if points is None:
            continue
        points = fair_points(points, min_nodes=16)
        if points is not None:
            return CaseSpec("Grid >=16", "grid", "rectangle", points)
    raise RuntimeError("Could not generate fair grid rectangle case.")


def generate_office_shape(device: torch.device) -> CaseSpec:
    for _attempt in range(7000):
        shape, params = dc.sample_office_shape()
        if shape not in OFFICE_BUCKET_SHAPES:
            continue
        target_n = max(16, min(dc.MAX_NODES, dc.target_anchor_count(shape, params) + random.randint(0, 8)))
        if random.random() < 0.65:
            points = dc.grid_points_in_shape(shape, params, target_n, device)
            family = "grid-office"
        else:
            points = dc.random_points_in_shape(shape, params, target_n, device)
            family = "random-office"
        if points is None:
            continue
        points = fair_points(points, min_nodes=16, min_vertex_connectivity=dc.MIN_VERTEX_CONNECTIVITY)
        if points is not None:
            return CaseSpec("Office >=16", family, shape, points)
    raise RuntimeError("Could not generate fair office-shaped case.")


def graph_batch_from_cases(
    cases: list[CaseSpec],
    device: torch.device,
    *,
    graph_solution_features: bool = False,
) -> dc.GraphBatch:
    batch_size = len(cases)
    positions = torch.zeros((batch_size, dc.MAX_NODES, 2), dtype=torch.float32, device=device)
    mask = torch.zeros((batch_size, dc.MAX_NODES), dtype=torch.bool, device=device)
    families: list[str] = []
    shapes: list[str] = []
    node_counts: list[int] = []
    for batch_index, case in enumerate(cases):
        points = case.points.to(device=device, dtype=torch.float32)
        n = int(points.shape[0])
        positions[batch_index, :n] = points
        mask[batch_index, :n] = True
        families.append(case.family)
        shapes.append(case.shape)
        node_counts.append(n)

    true_dist = torch.cdist(positions, positions)
    eye = torch.eye(dc.MAX_NODES, dtype=torch.bool, device=device).unsqueeze(0)
    pair_mask = mask.unsqueeze(2) & mask.unsqueeze(1) & ~eye
    measured_mask = (true_dist <= dc.EDGE_RADIUS_M) & pair_mask
    upper = torch.triu(torch.ones((dc.MAX_NODES, dc.MAX_NODES), dtype=torch.bool, device=device), diagonal=1).unsqueeze(0)
    noise = torch.randn_like(true_dist) * dc.NOISE_SIGMA_M
    nlos = (torch.rand_like(true_dist) < dc.NLOS_PROBABILITY).float()
    nlos_offset = torch.rand_like(true_dist) * dc.NLOS_MAX_OFFSET_M * nlos
    perturbation = torch.where(upper, noise + nlos_offset, torch.zeros_like(true_dist))
    perturbation = perturbation + perturbation.transpose(1, 2)
    measured_dist = (true_dist + perturbation).clamp_min(0.05) * measured_mask.float()

    edge_count = measured_mask.float().sum(dim=(1, 2)).clamp_min(1.0)
    scale = (measured_dist.sum(dim=(1, 2)) / edge_count).clamp_min(1.0)
    shortest, hops = dc.shortest_paths(measured_dist, measured_mask, pair_mask)
    node_features = dc.build_node_features(measured_dist, measured_mask, mask, pair_mask, shortest, hops, scale)
    edge_features = dc.build_edge_features(measured_dist, measured_mask, pair_mask, shortest, hops, scale)
    batch = dc.GraphBatch(
        positions_m=positions,
        mask=mask,
        true_dist_m=true_dist,
        measured_dist_m=measured_dist,
        measured_mask=measured_mask,
        pair_mask=pair_mask,
        shortest_m=shortest,
        hop_count=hops,
        scale_m=scale,
        node_features=node_features,
        edge_features=edge_features,
        family=families,
        shape=shapes,
        node_counts=node_counts,
    )
    if graph_solution_features:
        batch = dc.append_graph_solution_edge_features(batch)
    return batch


def load_model(checkpoint_path: Path, device: torch.device) -> dc.DistanceCompletionNet:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    saved_args = checkpoint.get("args", {})
    probe = dc.make_graph_batch(2, device=device, random_fraction=0.5)
    model = dc.DistanceCompletionNet(
        probe.node_features.shape[-1],
        probe.edge_features.shape[-1],
        hidden=int(saved_args.get("hidden", 144)),
        layers=int(saved_args.get("layers", 5)),
        dropout=float(saved_args.get("dropout", 0.0)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def generate_cases(bucket: str, count: int, device: torch.device) -> list[CaseSpec]:
    generator = {
        "random": generate_random_rectangle,
        "grid": generate_grid_rectangle,
        "office": generate_office_shape,
    }[bucket]
    cases: list[CaseSpec] = []
    while len(cases) < count:
        cases.append(generator(device))
    return cases


@torch.no_grad()
def predict_batches(
    model: dc.DistanceCompletionNet,
    cases: list[CaseSpec],
    *,
    device: torch.device,
    batch_size: int,
) -> list[tuple[dc.GraphBatch, torch.Tensor, list[CaseSpec]]]:
    chunks: list[tuple[dc.GraphBatch, torch.Tensor, list[CaseSpec]]] = []
    for start in range(0, len(cases), batch_size):
        chunk_cases = cases[start : start + batch_size]
        batch = graph_batch_from_cases(chunk_cases, device)
        pred = model(batch.node_features, batch.edge_features, batch.mask, batch.measured_mask, batch.pair_mask)
        chunks.append((batch, pred, chunk_cases))
    return chunks


def solve_bucket(
    bucket: str,
    chunks: list[tuple[dc.GraphBatch, torch.Tensor, list[CaseSpec]]],
    *,
    predicted_sigma: float,
    predicted_sigma_slope: float,
    closest_predicted_pairs_per_anchor: float,
    solver_iterations: int,
    polish_iterations: int,
    weak_polish_iterations: int,
    weak_completion_max_distance: float,
    weak_completion_sigma_multiplier: float,
) -> list[SolvedCase]:
    rows: list[SolvedCase] = []
    global_case_index = 0
    for batch, pred, _chunk_cases in chunks:
        for case_index in range(len(batch.node_counts)):
            truth = dc.graph_to_truth(batch, case_index)
            known_pairs = dc.known_pairs_from_batch(batch, case_index)
            completed_pairs, _pred_matrix, _scale = dc.completed_pairs_from_prediction(
                batch,
                pred,
                case_index,
                predicted_sigma_m=predicted_sigma,
                predicted_sigma_slope=predicted_sigma_slope,
                closest_predicted_pairs_per_anchor=closest_predicted_pairs_per_anchor,
            )
            oracle_pairs = dc.oracle_full_pairs(batch, case_index)
            mae = dc.missing_mae(batch, pred, case_index)
            solutions = {
                "ML completed": dc.completion_solution(
                    completed_pairs,
                    known_pairs,
                    max_iterations=solver_iterations,
                    polish_known_iterations=polish_iterations,
                ),
                "ML weak polish": dc.completion_solution_weak_polish(
                    completed_pairs,
                    known_pairs,
                    max_iterations=solver_iterations,
                    weak_polish_iterations=weak_polish_iterations,
                    max_predicted_distance_m=weak_completion_max_distance,
                    sigma_multiplier=weak_completion_sigma_multiplier,
                ),
            }
            for method, estimate in solutions.items():
                known_rmse, known_max = dc.pair_metrics(estimate, known_pairs)
                max_offset, median_offset, p95_offset = dc.offset_summary(truth, estimate)
                rows.append(
                    SolvedCase(
                        bucket=bucket,
                        method=method,
                        case_index=global_case_index,
                        family=batch.family[case_index],
                        shape=batch.shape[case_index],
                        anchors=batch.node_counts[case_index],
                        known_pairs=len(known_pairs),
                        full_pairs=len(oracle_pairs),
                        missing_mae_m=mae,
                        known_rmse_m=known_rmse,
                        known_max_residual_m=known_max,
                        max_offset_m=max_offset,
                        median_offset_m=median_offset,
                        p95_offset_m=p95_offset,
                        truth=truth,
                        known_pairs_list=known_pairs,
                        estimate=estimate,
                    )
                )
            global_case_index += 1
            if global_case_index % 20 == 0:
                print(f"solved bucket={bucket} cases={global_case_index}", flush=True)
    return rows


def p95_case(rows: list[SolvedCase], bucket: str, method: str) -> SolvedCase:
    part = [row for row in rows if row.bucket == bucket and row.method == method]
    if not part:
        raise ValueError(f"No rows for {bucket} {method}")
    part = sorted(part, key=lambda row: row.max_offset_m)
    index = min(len(part) - 1, int(math.ceil(0.95 * len(part))) - 1)
    return part[index]


def write_metrics(path: Path, rows: list[SolvedCase]) -> None:
    fields = [
        "bucket",
        "method",
        "case_index",
        "family",
        "shape",
        "anchors",
        "known_pairs",
        "full_pairs",
        "missing_mae_m",
        "known_rmse_m",
        "known_max_residual_m",
        "max_offset_m",
        "median_offset_m",
        "p95_offset_m",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: getattr(row, field) for field in fields})


def aligned_estimate(
    truth: dict[str, tuple[float, float]],
    estimate: dict[str, tuple[float, float]],
) -> dict[str, tuple[float, float]]:
    return dc.aligned_estimate(truth, estimate)


def draw_case(ax, row: SolvedCase, *, color: str, edge_color: str) -> None:
    truth = row.truth
    estimate = aligned_estimate(truth, row.estimate)
    ax.set_facecolor(TOKENS["panel"])
    for pair in row.known_pairs_list:
        ax.plot(
            [truth[pair.anchor_a_id][0], truth[pair.anchor_b_id][0]],
            [truth[pair.anchor_a_id][1], truth[pair.anchor_b_id][1]],
            color=NEUTRAL["light"],
            linewidth=0.55,
            alpha=0.55,
            zorder=1,
        )
    for anchor_id, true_point in truth.items():
        solved = estimate[anchor_id]
        ax.plot(
            [true_point[0], solved[0]],
            [true_point[1], solved[1]],
            color=ORANGE["dark"],
            linewidth=0.7,
            alpha=0.42,
            zorder=2,
        )
    ax.scatter(
        [point[0] for point in truth.values()],
        [point[1] for point in truth.values()],
        s=20,
        facecolors=TOKENS["panel"],
        edgecolors=NEUTRAL["dark"],
        linewidths=0.8,
        label="truth",
        zorder=3,
    )
    ax.scatter(
        [point[0] for point in estimate.values()],
        [point[1] for point in estimate.values()],
        s=26,
        color=color,
        edgecolors=edge_color,
        linewidths=0.65,
        label="solved",
        zorder=4,
    )
    xs = [x for x, _ in truth.values()] + [x for x, _ in estimate.values()]
    ys = [y for _, y in truth.values()] + [y for _, y in estimate.values()]
    span = max(max(xs) - min(xs), max(ys) - min(ys), 1.0)
    pad = span * 0.14
    ax.set_xlim(min(xs) - pad, max(xs) + pad)
    ax.set_ylim(min(ys) - pad, max(ys) + pad)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, color=TOKENS["grid"], linewidth=0.55)
    ax.tick_params(labelsize=7, colors=TOKENS["muted"], length=0)
    for spine in ax.spines.values():
        spine.set_color(TOKENS["axis"])
    ax.set_title(
        (
            f"{row.bucket} | {row.method}\n"
            f"p95 max offset {row.max_offset_m:.2f} m, RMSE {row.known_rmse_m:.3f} m, "
            f"{row.anchors} anchors, {row.known_pairs} measured"
        ),
        loc="left",
        fontsize=9.0,
        fontweight="semibold",
        color=TOKENS["ink"],
    )
    ax.text(
        0.012,
        0.02,
        f"shape={row.shape}, missing MAE={row.missing_mae_m:.2f} m",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=7.2,
        color=TOKENS["muted"],
        bbox={"facecolor": TOKENS["panel"], "edgecolor": "none", "alpha": 0.82, "pad": 1.6},
    )


def make_figure(path: Path, rows: list[SolvedCase], *, cases_per_bucket: int) -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": TOKENS["surface"],
            "savefig.facecolor": TOKENS["surface"],
            "font.family": ["Segoe UI", "DejaVu Sans", "Arial", "sans-serif"],
        }
    )
    buckets = ["Random 16-32", "Grid >=16", "Office >=16"]
    methods = ["ML completed", "ML weak polish"]
    fig, axes = plt.subplots(len(buckets), len(methods), figsize=(15.8, 16.5), dpi=170)
    fig.text(
        0.035,
        0.985,
        "P95 ML solve cases on fair anchor graphs",
        ha="left",
        va="top",
        fontsize=20,
        fontweight="bold",
        color=TOKENS["ink"],
    )
    fig.text(
        0.035,
        0.958,
        (
            f"Each panel is the case at the 95th percentile of max labeled-anchor coordinate offset "
            f"within {cases_per_bucket} generated layouts. All layouts are connected and every anchor "
            f"has at least {dc.MIN_MEASURED_DEGREE} measured <=8 m links before noise; rotation and mirror are aligned away."
        ),
        ha="left",
        va="top",
        fontsize=9.2,
        color=TOKENS["muted"],
    )
    palette = {
        "ML completed": (BLUE["base"], BLUE["dark"]),
        "ML weak polish": (PINK["base"], PINK["dark"]),
    }
    for row_index, bucket in enumerate(buckets):
        for col_index, method in enumerate(methods):
            selected = p95_case(rows, bucket, method)
            color, edge_color = palette[method]
            draw_case(axes[row_index, col_index], selected, color=color, edge_color=edge_color)
    handles = [
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=TOKENS["panel"], markeredgecolor=NEUTRAL["dark"], label="Ground truth", markersize=6),
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=BLUE["base"], markeredgecolor=BLUE["dark"], label="ML completed solved", markersize=6),
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=PINK["base"], markeredgecolor=PINK["dark"], label="ML weak-polish solved", markersize=6),
        plt.Line2D([0], [0], color=ORANGE["dark"], linewidth=1.0, label="Per-anchor offset"),
    ]
    fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.975, 0.987), frameon=False, ncol=2, fontsize=8.5)
    fig.subplots_adjust(left=0.035, right=0.98, top=0.91, bottom=0.04, hspace=0.28, wspace=0.16)
    fig.savefig(path, bbox_inches="tight", dpi=170)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render p95 ML anchor solve cases for fair layout buckets.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--checkpoint", type=Path, default=OUTPUTS / "anchor_solver_ml_distance_completion_fair_diag_cuda.pt")
    parser.add_argument("--cases-per-bucket", type=int, default=96)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260627)
    parser.add_argument("--solver-iterations", type=int, default=80)
    parser.add_argument("--polish-iterations", type=int, default=80)
    parser.add_argument("--weak-polish-iterations", type=int, default=80)
    parser.add_argument("--predicted-sigma", type=float, default=0.55)
    parser.add_argument("--predicted-sigma-slope", type=float, default=0.65)
    parser.add_argument("--closest-predicted-pairs-per-anchor", type=float, default=4.0)
    parser.add_argument("--weak-completion-max-distance", type=float, default=18.0)
    parser.add_argument("--weak-completion-sigma-multiplier", type=float, default=2.5)
    parser.add_argument("--prefix", default="anchor_solver_ml_p95_fair_cases")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed % (2**32 - 1))
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = choose_device(args.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
    print(f"device={device} checkpoint={args.checkpoint}", flush=True)
    model = load_model(args.checkpoint, device)
    all_rows: list[SolvedCase] = []
    for bucket_key in ("random", "grid", "office"):
        print(f"generating bucket={bucket_key} cases={args.cases_per_bucket}", flush=True)
        cases = generate_cases(bucket_key, args.cases_per_bucket, device)
        node_counts = [int(case.points.shape[0]) for case in cases]
        print(
            f"generated bucket={bucket_key} min_nodes={min(node_counts)} "
            f"median_nodes={np.median(node_counts):.0f} max_nodes={max(node_counts)}",
            flush=True,
        )
        chunks = predict_batches(model, cases, device=device, batch_size=args.batch_size)
        rows = solve_bucket(
            cases[0].bucket,
            chunks,
            predicted_sigma=args.predicted_sigma,
            predicted_sigma_slope=args.predicted_sigma_slope,
            closest_predicted_pairs_per_anchor=args.closest_predicted_pairs_per_anchor,
            solver_iterations=args.solver_iterations,
            polish_iterations=args.polish_iterations,
            weak_polish_iterations=args.weak_polish_iterations,
            weak_completion_max_distance=args.weak_completion_max_distance,
            weak_completion_sigma_multiplier=args.weak_completion_sigma_multiplier,
        )
        all_rows.extend(rows)
    metrics_path = OUTPUTS / f"{args.prefix}_metrics.csv"
    figure_path = OUTPUTS / f"{args.prefix}.png"
    write_metrics(metrics_path, all_rows)
    make_figure(figure_path, all_rows, cases_per_bucket=args.cases_per_bucket)
    for bucket in ("Random 16-32", "Grid >=16", "Office >=16"):
        for method in ("ML completed", "ML weak polish"):
            selected = p95_case(all_rows, bucket, method)
            print(
                f"p95 bucket={bucket} method={method} max_offset={selected.max_offset_m:.3f}m "
                f"known_rmse={selected.known_rmse_m:.4f}m anchors={selected.anchors} "
                f"shape={selected.shape}",
                flush=True,
            )
    print(f"Wrote {metrics_path}", flush=True)
    print(f"Wrote {figure_path}", flush=True)


if __name__ == "__main__":
    main()

