from __future__ import annotations

import argparse
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
import torch


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT / "work" / "SmartClicker-GUI"
OUTPUTS = ROOT / "outputs"
MODEL_PATH = ROOT / "work" / "anchor_solver_ml_distance_completion.py"
sys.path.insert(0, str(REPO))

spec = importlib.util.spec_from_file_location("distance_completion", MODEL_PATH)
dc = importlib.util.module_from_spec(spec)
sys.modules["distance_completion"] = dc
assert spec.loader is not None
spec.loader.exec_module(dc)

from uwb_capture.anchor_geometry import solve_anchor_layout  # noqa: E402


TOKENS = dc.TOKENS
BLUE = dc.BLUE
GOLD = dc.GOLD
ORANGE = dc.ORANGE
OLIVE = dc.OLIVE
PINK = dc.PINK
NEUTRAL = dc.NEUTRAL


def make_irregular_5x5(seed: int) -> dict[str, tuple[float, float]]:
    rng = random.Random(seed)
    x_gaps = [rng.uniform(4.85, 5.35) for _ in range(4)]
    y_gaps = [rng.uniform(4.85, 5.35) for _ in range(4)]
    xs = [0.0]
    ys = [0.0]
    for gap in x_gaps:
        xs.append(xs[-1] + gap)
    for gap in y_gaps:
        ys.append(ys[-1] + gap)
    positions: dict[str, tuple[float, float]] = {}
    index = 0
    for row, y_m in enumerate(ys):
        for col, x_m in enumerate(xs):
            jitter_x = rng.uniform(-0.08, 0.08)
            jitter_y = rng.uniform(-0.08, 0.08)
            if row in (0, 4):
                jitter_y *= 0.35
            if col in (0, 4):
                jitter_x *= 0.35
            positions[f"A{index:02d}"] = (x_m + jitter_x, y_m + jitter_y)
            index += 1
    min_x = min(x for x, _ in positions.values())
    min_y = min(y for _, y in positions.values())
    return {anchor_id: (x - min_x, y - min_y) for anchor_id, (x, y) in positions.items()}


def graph_batch_from_positions(
    positions: dict[str, tuple[float, float]],
    *,
    device: torch.device,
    seed: int,
) -> dc.GraphBatch:
    torch.manual_seed(seed)
    random.seed(seed)
    n = len(positions)
    tensor = torch.zeros((1, dc.MAX_NODES, 2), dtype=torch.float32, device=device)
    mask = torch.zeros((1, dc.MAX_NODES), dtype=torch.bool, device=device)
    for index, anchor_id in enumerate(sorted(positions)):
        tensor[0, index, 0] = positions[anchor_id][0]
        tensor[0, index, 1] = positions[anchor_id][1]
    mask[0, :n] = True
    true_dist = torch.cdist(tensor, tensor)
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
    scale = measured_dist.sum(dim=(1, 2)) / edge_count
    shortest, hops = dc.shortest_paths(measured_dist, measured_mask, pair_mask)
    node_features = dc.build_node_features(measured_dist, measured_mask, mask, scale)
    edge_features = dc.build_edge_features(measured_dist, measured_mask, pair_mask, shortest, hops, scale)
    return dc.GraphBatch(
        positions_m=tensor,
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
        family=["grid"],
        shape=["normal_5x5_irregular_grid"],
        node_counts=[n],
    )


def load_model(checkpoint_path: Path, batch: dc.GraphBatch, device: torch.device) -> dc.DistanceCompletionNet:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    args = checkpoint["args"]
    model = dc.DistanceCompletionNet(
        batch.node_features.shape[-1],
        batch.edge_features.shape[-1],
        hidden=int(args["hidden"]),
        layers=int(args["layers"]),
        dropout=float(args.get("dropout", 0.0)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def solve_current_production(known_pairs):
    result = solve_anchor_layout(
        known_pairs,
        seed_count=32,
        basin_hops=12,
        max_iterations=100,
        random_seed=20260626,
    )
    return result.positions_m, result.rmse_m, result.max_residual_m


def metrics(
    truth: dict[str, tuple[float, float]],
    positions: dict[str, tuple[float, float]],
    known_pairs,
) -> dict[str, float]:
    max_offset, median_offset, p95_offset = dc.offset_summary(truth, positions)
    rmse, max_residual = dc.pair_metrics(positions, known_pairs)
    return {
        "max_offset_m": max_offset,
        "median_offset_m": median_offset,
        "p95_offset_m": p95_offset,
        "known_rmse_m": rmse,
        "known_max_residual_m": max_residual,
    }


def aligned_positions(truth, estimate):
    return dc.aligned_estimate(truth, estimate)


def draw_truth(ax, truth, known_pairs):
    ax.set_facecolor(TOKENS["panel"])
    for pair in known_pairs:
        ax.plot(
            [truth[pair.anchor_a_id][0], truth[pair.anchor_b_id][0]],
            [truth[pair.anchor_a_id][1], truth[pair.anchor_b_id][1]],
            color=NEUTRAL["base"],
            linewidth=0.8,
            alpha=0.7,
            zorder=1,
        )
    degrees = {anchor_id: 0 for anchor_id in truth}
    for pair in known_pairs:
        degrees[pair.anchor_a_id] += 1
        degrees[pair.anchor_b_id] += 1
    ax.scatter(
        [x for x, _ in truth.values()],
        [y for _, y in truth.values()],
        s=44,
        color=GOLD["base"],
        edgecolors=GOLD["dark"],
        linewidths=0.9,
        zorder=3,
    )
    for anchor_id, (x_m, y_m) in truth.items():
        ax.text(x_m + 0.08, y_m + 0.08, f"{anchor_id}\nd{degrees[anchor_id]}", fontsize=6.2, color=TOKENS["ink"])
    format_axis(ax, truth, truth)
    ax.set_title("Ground truth known graph\n<=8 m links, noisy ranges used by solvers", loc="left", fontsize=10, fontweight="bold", color=TOKENS["ink"])


def draw_solution(ax, truth, estimate, known_pairs, title, color):
    aligned = aligned_positions(truth, estimate)
    m = metrics(truth, estimate, known_pairs)
    ax.set_facecolor(TOKENS["panel"])
    for pair in known_pairs:
        ax.plot(
            [truth[pair.anchor_a_id][0], truth[pair.anchor_b_id][0]],
            [truth[pair.anchor_a_id][1], truth[pair.anchor_b_id][1]],
            color=NEUTRAL["light"],
            linewidth=0.45,
            alpha=0.45,
            zorder=1,
        )
    for anchor_id, true_point in truth.items():
        solved = aligned[anchor_id]
        ax.plot(
            [true_point[0], solved[0]],
            [true_point[1], solved[1]],
            color=PINK["dark"] if math.hypot(true_point[0] - solved[0], true_point[1] - solved[1]) == m["max_offset_m"] else ORANGE["dark"],
            linewidth=0.8,
            alpha=0.50,
            zorder=2,
        )
    ax.scatter(
        [x for x, _ in truth.values()],
        [y for _, y in truth.values()],
        s=24,
        facecolors=TOKENS["panel"],
        edgecolors=NEUTRAL["dark"],
        linewidths=0.75,
        zorder=3,
    )
    ax.scatter(
        [x for x, _ in aligned.values()],
        [y for _, y in aligned.values()],
        s=30,
        color=color,
        edgecolors=TOKENS["ink"],
        linewidths=0.65,
        zorder=4,
    )
    format_axis(ax, truth, aligned)
    ax.set_title(
        f"{title}\nmax {m['max_offset_m']:.3f} m | med {m['median_offset_m']:.3f} m | RMSE {m['known_rmse_m']:.4f} m",
        loc="left",
        fontsize=10,
        fontweight="bold",
        color=TOKENS["ink"],
    )


def draw_distance_scatter(ax, batch, pred):
    n = batch.node_counts[0]
    missing = (~batch.measured_mask[0, :n, :n]) & batch.pair_mask[0, :n, :n]
    upper = torch.triu(torch.ones((n, n), dtype=torch.bool, device=pred.device), diagonal=1)
    missing = missing & upper
    scale = batch.scale_m[0]
    true_m = batch.true_dist_m[0, :n, :n][missing].detach().cpu().numpy()
    pred_m = (pred[0, :n, :n][missing] * scale).detach().cpu().numpy()
    ax.set_facecolor(TOKENS["panel"])
    ax.scatter(true_m, pred_m, s=13, color=BLUE["base"], edgecolors=BLUE["dark"], linewidths=0.35, alpha=0.78)
    limit = max(float(true_m.max()), float(pred_m.max())) * 1.05
    ax.plot([0, limit], [0, limit], color=NEUTRAL["dark"], linewidth=1.0, alpha=0.75)
    mae = float(np.mean(np.abs(pred_m - true_m)))
    ax.set_xlim(0, limit)
    ax.set_ylim(0, limit)
    ax.grid(True, color=TOKENS["grid"], linewidth=0.6)
    ax.tick_params(labelsize=8, colors=TOKENS["muted"], length=0)
    ax.set_xlabel("true missing distance (m)", fontsize=8, color=TOKENS["muted"])
    ax.set_ylabel("predicted missing distance (m)", fontsize=8, color=TOKENS["muted"])
    ax.set_title(f"Missing distance predictions\nMAE {mae:.3f} m over {len(true_m)} missing pairs", loc="left", fontsize=10, fontweight="bold", color=TOKENS["ink"])
    for spine in ax.spines.values():
        spine.set_color(TOKENS["axis"])


def format_axis(ax, truth, estimate):
    xs = [x for x, _ in truth.values()] + [x for x, _ in estimate.values()]
    ys = [y for _, y in truth.values()] + [y for _, y in estimate.values()]
    span = max(max(xs) - min(xs), max(ys) - min(ys), 1.0)
    pad = span * 0.10
    ax.set_xlim(min(xs) - pad, max(xs) + pad)
    ax.set_ylim(min(ys) - pad, max(ys) + pad)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, color=TOKENS["grid"], linewidth=0.55)
    ax.tick_params(labelsize=7, colors=TOKENS["muted"], length=0)
    for spine in ax.spines.values():
        spine.set_color(TOKENS["axis"])


def write_metrics_csv(path: Path, rows: list[dict[str, str | float | int]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=str(OUTPUTS / "anchor_solver_ml_distance_completion_office_cuda.pt"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=20260626)
    parser.add_argument("--prefix", default="anchor_solver_5x5_grid_completion_example")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    device = dc.choose_device(args.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
    truth = make_irregular_5x5(args.seed)
    batch = graph_batch_from_positions(truth, device=device, seed=args.seed + 17)
    known_pairs = dc.known_pairs_from_batch(batch, 0)
    checkpoint_path = Path(args.checkpoint)
    model = load_model(checkpoint_path, batch, device)
    with torch.no_grad():
        pred = model(batch.node_features, batch.edge_features, batch.mask, batch.measured_mask, batch.pair_mask)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    ckpt_args = checkpoint["args"]
    completed_pairs, _pred_matrix, _scale = dc.completed_pairs_from_prediction(
        batch,
        pred,
        0,
        predicted_sigma_m=float(ckpt_args.get("predicted_sigma", 0.55)),
        predicted_sigma_slope=float(ckpt_args.get("predicted_sigma_slope", 0.65)),
    )
    oracle_pairs = dc.oracle_full_pairs(batch, 0)

    known_tri = dc.known_only_solution(known_pairs, max_iterations=100)
    production, production_rmse, production_max = solve_current_production(known_pairs)
    ml_completed = dc.completion_solution(completed_pairs, known_pairs, max_iterations=100, polish_known_iterations=100)
    oracle = dc.completion_solution(oracle_pairs, known_pairs, max_iterations=100, polish_known_iterations=0)

    solutions = [
        ("known_triangulated_lm", known_tri, NEUTRAL["base"]),
        ("current_solver_known_edges", production, GOLD["base"]),
        ("ml_completed_full_graph", ml_completed, BLUE["base"]),
        ("oracle_full_graph", oracle, OLIVE["base"]),
    ]

    rows: list[dict[str, str | float | int]] = []
    n = batch.node_counts[0]
    known_count = len(known_pairs)
    full_count = n * (n - 1) // 2
    missing_mae = dc.missing_mae(batch, pred, 0)
    for name, positions, _color in solutions:
        row = {
            "method": name,
            "anchors": n,
            "known_pairs": known_count,
            "full_pairs": full_count,
            "missing_mae_m": missing_mae,
            **metrics(truth, positions, known_pairs),
        }
        if name == "current_solver_known_edges":
            row["known_rmse_m"] = production_rmse
            row["known_max_residual_m"] = production_max
        rows.append(row)

    plt.rcParams.update(
        {
            "figure.facecolor": TOKENS["surface"],
            "savefig.facecolor": TOKENS["surface"],
            "font.family": ["Segoe UI", "DejaVu Sans", "Arial", "sans-serif"],
        }
    )
    fig = plt.figure(figsize=(18, 9.5), dpi=185)
    gs = fig.add_gridspec(2, 3, hspace=0.34, wspace=0.24)
    fig.text(0.035, 0.975, "Normal 5x5 Grid: ML Distance Completion", ha="left", va="top", fontsize=20, fontweight="bold", color=TOKENS["ink"])
    fig.text(
        0.035,
        0.94,
        f"Slightly irregular ~5 m grid, 25 anchors, {known_count}/{full_count} known noisy <=8 m links. This graph has healthy horizontal, vertical, and diagonal local constraints.",
        ha="left",
        va="top",
        fontsize=9.5,
        color=TOKENS["muted"],
    )
    draw_truth(fig.add_subplot(gs[0, 0]), truth, known_pairs)
    draw_solution(fig.add_subplot(gs[0, 1]), truth, known_tri, known_pairs, "Known-only triangulated LM", NEUTRAL["base"])
    draw_solution(fig.add_subplot(gs[0, 2]), truth, production, known_pairs, "Current solver on known edges", GOLD["base"])
    draw_solution(fig.add_subplot(gs[1, 0]), truth, ml_completed, known_pairs, "ML-completed distances", BLUE["base"])
    draw_solution(fig.add_subplot(gs[1, 1]), truth, oracle, known_pairs, "Oracle full distances", OLIVE["base"])
    draw_distance_scatter(fig.add_subplot(gs[1, 2]), batch, pred)
    fig.text(
        0.035,
        0.026,
        "Solved layouts are rigid-aligned to ground truth with optional mirror and no scaling. The ML model sees no anchor index/order hint.",
        ha="left",
        va="bottom",
        fontsize=8.4,
        color=TOKENS["muted"],
    )
    png_path = OUTPUTS / f"{args.prefix}.png"
    csv_path = OUTPUTS / f"{args.prefix}.csv"
    fig.savefig(png_path, bbox_inches="tight", dpi=185)
    plt.close(fig)
    write_metrics_csv(csv_path, rows)

    print(f"anchors={n} known_pairs={known_count} full_pairs={full_count} missing_mae={missing_mae:.3f}m")
    for row in rows:
        print(
            f"{row['method']}: max_offset={row['max_offset_m']:.4f}m "
            f"median_offset={row['median_offset_m']:.4f}m "
            f"known_rmse={row['known_rmse_m']:.5f}m"
        )
    print(f"Wrote {png_path}")
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
