from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import math
from pathlib import Path
import random
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT / "work" / "SmartClicker-GUI"
OUTPUTS = ROOT / "outputs"
sys.path.insert(0, str(REPO))

from uwb_capture.anchor_geometry import (  # noqa: E402
    AnchorPairDistance,
    _Parameterization,
    _anchor_ids,
    _degree_shell_seed,
    _layout_scale,
    _local_minimize,
    _positions_to_params,
    _preprocess_pairs,
    _triangulated_seed,
    pair_residuals,
    rotate_layout_to_level,
)


EDGE_RADIUS_M = 8.0
MIN_ANCHOR_SPACING_M = 2.0
AREA_WIDTH_M = 25.0
AREA_HEIGHT_M = 25.0
NOISE_SIGMA_M = 0.03
NLOS_PROBABILITY = 1.0 / 3.0
NLOS_MAX_OFFSET_M = 0.20
PAIR_SIGMA_M = 0.05
MAX_NODES = 32

GRID_SHAPES = (
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
)

TOKENS = {
    "surface": "#FCFCFD",
    "panel": "#FFFFFF",
    "ink": "#1F2430",
    "muted": "#6F768A",
    "grid": "#E6E8F0",
    "axis": "#D7DBE7",
}
BLUE = {"light": "#CEDFFE", "base": "#A3BEFA", "mid": "#5477C4", "dark": "#2E4780"}
GOLD = {"light": "#FFEA8F", "base": "#FFE15B", "mid": "#B8A037", "dark": "#736422"}
ORANGE = {"light": "#FFBDA1", "base": "#F0986E", "mid": "#CC6F47", "dark": "#804126"}
PINK = {"light": "#F5BACC", "base": "#F390CA", "mid": "#BD569B", "dark": "#8A3A6F"}
OLIVE = {"light": "#BEEB96", "base": "#A3D576", "mid": "#71B436", "dark": "#386411"}
NEUTRAL = {"light": "#E2E5EA", "base": "#C5CAD3", "mid": "#7A828F", "dark": "#464C55"}


@dataclass
class GraphBatch:
    positions_m: torch.Tensor
    mask: torch.Tensor
    true_dist_m: torch.Tensor
    measured_dist_m: torch.Tensor
    measured_mask: torch.Tensor
    pair_mask: torch.Tensor
    scale_m: torch.Tensor
    node_features: torch.Tensor
    edge_features: torch.Tensor
    family: list[str]
    node_counts: list[int]


@dataclass(frozen=True)
class EvalRow:
    family: str
    case_index: int
    method: str
    stage: str
    anchors: int
    pairs: int
    rmse_m: float
    max_residual_m: float
    max_offset_m: float
    median_offset_m: float
    p95_offset_m: float


class DenseGraphSeedNet(nn.Module):
    def __init__(
        self,
        node_features: int,
        edge_features: int,
        *,
        hidden: int,
        layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.node_projection = nn.Sequential(
            nn.Linear(node_features, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        self.message_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden * 2 + edge_features, hidden),
                    nn.SiLU(),
                    nn.Linear(hidden, hidden),
                    nn.SiLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden, hidden),
                )
                for _ in range(layers)
            ]
        )
        self.update_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden * 2, hidden),
                    nn.SiLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden, hidden),
                )
                for _ in range(layers)
            ]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(layers)])
        self.head = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden // 2),
            nn.SiLU(),
            nn.Linear(hidden // 2, 2),
        )

    def forward(
        self,
        node_features: torch.Tensor,
        edge_features: torch.Tensor,
        mask: torch.Tensor,
        pair_mask: torch.Tensor,
    ) -> torch.Tensor:
        h = self.node_projection(node_features)
        h = h * mask.unsqueeze(-1)
        pair_weight = pair_mask.float().unsqueeze(-1)
        normalizer = pair_weight.sum(dim=2).clamp_min(1.0)
        for message_layer, update_layer, norm in zip(
            self.message_layers,
            self.update_layers,
            self.norms,
        ):
            sender = h.unsqueeze(1).expand(-1, h.shape[1], -1, -1)
            receiver = h.unsqueeze(2).expand(-1, -1, h.shape[1], -1)
            message_input = torch.cat([receiver, sender, edge_features], dim=-1)
            messages = message_layer(message_input) * pair_weight
            aggregated = messages.sum(dim=2) / normalizer
            update = update_layer(torch.cat([h, aggregated], dim=-1))
            h = norm(h + update)
            h = h * mask.unsqueeze(-1)
        global_context = masked_mean(h, mask, dim=1)
        global_context = global_context.unsqueeze(1).expand(-1, h.shape[1], -1)
        coords = self.head(torch.cat([h, global_context], dim=-1))
        return center_by_mask(coords, mask)


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but this PyTorch install cannot see CUDA.")
    return device


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def masked_mean(values: torch.Tensor, mask: torch.Tensor, *, dim: int) -> torch.Tensor:
    weights = mask.float().unsqueeze(-1)
    return (values * weights).sum(dim=dim) / weights.sum(dim=dim).clamp_min(1.0)


def center_by_mask(coords: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    center = masked_mean(coords, mask, dim=1).unsqueeze(1)
    return (coords - center) * mask.unsqueeze(-1)


def random_positions(n: int, device: torch.device) -> torch.Tensor:
    accepted = torch.empty((0, 2), device=device)
    while accepted.shape[0] < n:
        remaining = n - accepted.shape[0]
        candidates = torch.rand((max(remaining * 24, 256), 2), device=device)
        candidates[:, 0] *= AREA_WIDTH_M
        candidates[:, 1] *= AREA_HEIGHT_M
        if accepted.numel():
            distances = torch.cdist(candidates, accepted)
            candidates = candidates[distances.min(dim=1).values >= MIN_ANCHOR_SPACING_M]
        if candidates.numel() == 0:
            continue
        chosen: list[torch.Tensor] = []
        for candidate in candidates:
            if accepted.shape[0] + len(chosen) >= n:
                break
            if chosen:
                chosen_tensor = torch.stack(chosen)
                if torch.linalg.norm(chosen_tensor - candidate, dim=1).min() < MIN_ANCHOR_SPACING_M:
                    continue
            chosen.append(candidate)
        if chosen:
            accepted = torch.cat([accepted, torch.stack(chosen)], dim=0)
    return accepted[:n]


def grid_positions(device: torch.device) -> torch.Tensor:
    rows, cols = random.choice(GRID_SHAPES)
    x_spacing = random.uniform(4.0, 8.0)
    y_spacing = random.uniform(4.0, 8.0)
    xs = torch.arange(cols, dtype=torch.float32, device=device) * x_spacing
    ys = torch.arange(rows, dtype=torch.float32, device=device) * y_spacing
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1)


def make_graph_batch(
    batch_size: int,
    *,
    device: torch.device,
    random_fraction: float,
    include_index_features: bool,
) -> GraphBatch:
    positions = torch.zeros((batch_size, MAX_NODES, 2), dtype=torch.float32, device=device)
    mask = torch.zeros((batch_size, MAX_NODES), dtype=torch.bool, device=device)
    families: list[str] = []
    node_counts: list[int] = []
    for batch_index in range(batch_size):
        if random.random() < random_fraction:
            n = random.randint(16, MAX_NODES)
            points = random_positions(n, device)
            family = "random"
        else:
            points = grid_positions(device)
            n = int(points.shape[0])
            family = "grid"
        positions[batch_index, :n] = points
        mask[batch_index, :n] = True
        families.append(family)
        node_counts.append(n)

    true_dist = torch.cdist(positions, positions)
    eye = torch.eye(MAX_NODES, dtype=torch.bool, device=device).unsqueeze(0)
    pair_mask = mask.unsqueeze(2) & mask.unsqueeze(1) & ~eye
    measured_mask = (true_dist <= EDGE_RADIUS_M) & pair_mask

    upper = torch.triu(torch.ones((MAX_NODES, MAX_NODES), dtype=torch.bool, device=device), diagonal=1)
    upper = upper.unsqueeze(0)
    noise = torch.randn_like(true_dist) * NOISE_SIGMA_M
    nlos = (torch.rand_like(true_dist) < NLOS_PROBABILITY).float()
    nlos_offset = torch.rand_like(true_dist) * NLOS_MAX_OFFSET_M * nlos
    perturbation = torch.where(upper, noise + nlos_offset, torch.zeros_like(true_dist))
    perturbation = perturbation + perturbation.transpose(1, 2)
    measured_dist = (true_dist + perturbation).clamp_min(0.05) * measured_mask.float()

    edge_count = measured_mask.float().sum(dim=(1, 2)).clamp_min(1.0)
    scale = measured_dist.sum(dim=(1, 2)) / edge_count
    scale = scale.clamp_min(1.0)

    node_features = build_node_features(
        measured_dist,
        measured_mask,
        mask,
        scale,
        include_index_features=include_index_features,
    )
    edge_features = build_edge_features(
        measured_dist,
        measured_mask,
        pair_mask,
        scale,
    )
    return GraphBatch(
        positions_m=positions,
        mask=mask,
        true_dist_m=true_dist,
        measured_dist_m=measured_dist,
        measured_mask=measured_mask,
        pair_mask=pair_mask,
        scale_m=scale,
        node_features=node_features,
        edge_features=edge_features,
        family=families,
        node_counts=node_counts,
    )


def build_node_features(
    measured_dist: torch.Tensor,
    measured_mask: torch.Tensor,
    mask: torch.Tensor,
    scale: torch.Tensor,
    *,
    include_index_features: bool,
) -> torch.Tensor:
    batch_size, nodes, _ = measured_dist.shape
    scale_view = scale.view(batch_size, 1, 1)
    dist_norm = measured_dist / scale_view
    degree = measured_mask.float().sum(dim=2)
    node_count = mask.float().sum(dim=1, keepdim=True)
    degree_norm = degree / (node_count - 1.0).clamp_min(1.0)
    degree_mean = (degree * mask.float()).sum(dim=1, keepdim=True) / node_count.clamp_min(1.0)
    degree_std = torch.sqrt(
        (((degree - degree_mean) * mask.float()) ** 2).sum(dim=1, keepdim=True)
        / node_count.clamp_min(1.0)
    ).clamp_min(1e-4)
    degree_z = (degree - degree_mean) / degree_std
    max_degree = degree.masked_fill(~mask, -1e9).max(dim=1, keepdim=True).values
    min_degree = degree.masked_fill(~mask, 1e9).min(dim=1, keepdim=True).values
    low_degree_score = (max_degree - degree) / (max_degree - min_degree).clamp_min(1.0)
    count = degree.clamp_min(1.0)
    mean_dist = dist_norm.sum(dim=2) / count
    min_dist = dist_norm.masked_fill(~measured_mask, 1e9).min(dim=2).values
    min_dist = torch.where(degree > 0.0, min_dist, torch.zeros_like(min_dist))
    max_dist = dist_norm.max(dim=2).values
    variance = (((dist_norm - mean_dist.unsqueeze(2)) * measured_mask.float()) ** 2).sum(dim=2) / count
    std_dist = torch.sqrt(variance.clamp_min(0.0))
    density = measured_mask.float().sum(dim=(1, 2), keepdim=False).view(batch_size, 1)
    density = density / (node_count * (node_count - 1.0)).clamp_min(1.0)
    density = density.expand(-1, nodes)
    features = [
        mask.float(),
        degree_norm,
        degree_z,
        low_degree_score,
        mean_dist,
        min_dist,
        max_dist,
        std_dist,
        density,
    ]
    if include_index_features:
        index = torch.arange(nodes, dtype=torch.float32, device=measured_dist.device).view(1, nodes)
        index_fraction = index / (node_count - 1.0).clamp_min(1.0)
        features.extend(
            [
                index_fraction,
                torch.sin(2.0 * math.pi * index_fraction),
                torch.cos(2.0 * math.pi * index_fraction),
            ]
        )
    return torch.stack(features, dim=-1) * mask.unsqueeze(-1)


def build_edge_features(
    measured_dist: torch.Tensor,
    measured_mask: torch.Tensor,
    pair_mask: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    scale_view = scale.view(-1, 1, 1)
    dist_norm = measured_dist / scale_view
    missing_mask = pair_mask & ~measured_mask
    clipped = torch.clamp(dist_norm, 0.0, 2.5)
    return torch.stack(
        [
            pair_mask.float(),
            measured_mask.float(),
            missing_mask.float(),
            clipped,
            clipped * clipped,
            torch.exp(-clipped) * measured_mask.float(),
        ],
        dim=-1,
    )


def distance_shape_loss(pred: torch.Tensor, batch: GraphBatch) -> tuple[torch.Tensor, dict[str, float]]:
    pred = center_by_mask(pred, batch.mask)
    pred_dist = torch.cdist(pred, pred)
    target_dist = batch.true_dist_m / batch.scale_m.view(-1, 1, 1)
    measured_target = batch.measured_dist_m / batch.scale_m.view(-1, 1, 1)
    upper = torch.triu(
        torch.ones((MAX_NODES, MAX_NODES), dtype=torch.bool, device=pred.device),
        diagonal=1,
    ).unsqueeze(0)
    all_pairs = batch.pair_mask & upper
    measured_pairs = batch.measured_mask & upper
    shape_loss = F.smooth_l1_loss(pred_dist[all_pairs], target_dist[all_pairs], beta=0.05)
    measured_loss = F.smooth_l1_loss(pred_dist[measured_pairs], measured_target[measured_pairs], beta=0.05)
    min_spacing = MIN_ANCHOR_SPACING_M / batch.scale_m.view(-1, 1, 1)
    spacing_hinge = F.relu(min_spacing - pred_dist)
    spacing_loss = (spacing_hinge[all_pairs] ** 2).mean()
    loss = shape_loss + 0.25 * measured_loss + 0.05 * spacing_loss
    return loss, {
        "shape": float(shape_loss.detach().cpu()),
        "measured": float(measured_loss.detach().cpu()),
        "spacing": float(spacing_loss.detach().cpu()),
    }


def graph_to_pairs(batch: GraphBatch, case_index: int) -> list[AnchorPairDistance]:
    n = batch.node_counts[case_index]
    measured = batch.measured_mask[case_index, :n, :n].detach().cpu().numpy()
    distances = batch.measured_dist_m[case_index, :n, :n].detach().cpu().numpy()
    pairs: list[AnchorPairDistance] = []
    for i in range(n):
        for j in range(i + 1, n):
            if measured[i, j]:
                pairs.append(
                    AnchorPairDistance(
                        f"A{i:02d}",
                        f"A{j:02d}",
                        float(distances[i, j]),
                        sigma_m=PAIR_SIGMA_M,
                        source="ml-sim",
                    )
                )
    return pairs


def graph_to_truth(batch: GraphBatch, case_index: int) -> dict[str, tuple[float, float]]:
    n = batch.node_counts[case_index]
    points = batch.positions_m[case_index, :n].detach().cpu().numpy()
    return {f"A{i:02d}": (float(points[i, 0]), float(points[i, 1])) for i in range(n)}


def tensor_to_positions(points: torch.Tensor, n: int, scale_m: float) -> dict[str, tuple[float, float]]:
    array = (points[:n].detach().cpu().numpy() * scale_m).astype(float)
    return {f"A{i:02d}": (float(array[i, 0]), float(array[i, 1])) for i in range(n)}


def align_offsets_no_scale(
    truth: dict[str, tuple[float, float]],
    estimate: dict[str, tuple[float, float]],
) -> np.ndarray:
    ids = sorted(set(truth) & set(estimate))
    if not ids:
        return np.array([math.inf])
    target = np.array([truth[anchor_id] for anchor_id in ids], dtype=float)
    source = np.array([estimate[anchor_id] for anchor_id in ids], dtype=float)
    target_center = target.mean(axis=0)
    source_center = source.mean(axis=0)
    target_centered = target - target_center
    source_centered = source - source_center
    best_offsets: np.ndarray | None = None
    best_max = math.inf
    for reflection in (1.0, -1.0):
        reflected = source_centered.copy()
        reflected[:, 1] *= reflection
        u, _s, vt = np.linalg.svd(reflected.T @ target_centered)
        aligned = reflected @ (u @ vt) + target_center
        offsets = np.linalg.norm(aligned - target, axis=1)
        max_offset = float(offsets.max())
        if max_offset < best_max:
            best_max = max_offset
            best_offsets = offsets
    assert best_offsets is not None
    return best_offsets


def offset_summary(
    truth: dict[str, tuple[float, float]],
    estimate: dict[str, tuple[float, float]],
) -> tuple[float, float, float]:
    offsets = align_offsets_no_scale(truth, estimate)
    return float(offsets.max()), float(np.median(offsets)), float(np.quantile(offsets, 0.95))


def pair_metrics(
    positions: dict[str, tuple[float, float]],
    pairs: list[AnchorPairDistance],
) -> tuple[float, float]:
    residuals = pair_residuals(positions, pairs)
    if not residuals:
        return math.inf, math.inf
    values = list(residuals.values())
    rmse = math.sqrt(sum(value * value for value in values) / len(values))
    return rmse, max(abs(value) for value in values)


def canonical_params_from_positions(
    positions: dict[str, tuple[float, float]],
    parameterization: _Parameterization,
) -> list[float]:
    anchor_ids = parameterization.anchor_ids
    try:
        positions = rotate_layout_to_level(positions, anchor_ids[0], anchor_ids[1])
    except ValueError:
        ax, ay = positions[anchor_ids[0]]
        positions = {anchor_id: (x - ax, y - ay) for anchor_id, (x, y) in positions.items()}
    return _positions_to_params(parameterization, positions)


def solve_from_seed(
    seed_positions: dict[str, tuple[float, float]],
    pairs: list[AnchorPairDistance],
    *,
    max_iterations: int,
) -> tuple[dict[str, tuple[float, float]], float, float]:
    processed = _preprocess_pairs(pairs, min_sigma_m=0.02, min_distance_m=0.05)
    anchor_ids = _anchor_ids(processed)
    parameterization = _Parameterization(anchor_ids)
    seed_params = canonical_params_from_positions(seed_positions, parameterization)
    solved_params, _energy = _local_minimize(
        seed_params,
        parameterization,
        processed,
        None,
        max_iterations=max_iterations,
    )
    solved = parameterization.to_positions(solved_params)
    solved = rotate_layout_to_level(solved, anchor_ids[0], anchor_ids[1])
    rmse, max_residual = pair_metrics(solved, pairs)
    return solved, rmse, max_residual


def baseline_seed_positions(
    method: str,
    pairs: list[AnchorPairDistance],
) -> dict[str, tuple[float, float]]:
    processed = _preprocess_pairs(pairs, min_sigma_m=0.02, min_distance_m=0.05)
    anchor_ids = _anchor_ids(processed)
    parameterization = _Parameterization(anchor_ids)
    scale = _layout_scale(processed)
    if method == "triangulated":
        params = _triangulated_seed(parameterization, processed, scale)
    elif method == "degree-shell":
        params = _degree_shell_seed(parameterization, processed, scale)
    else:
        raise ValueError(method)
    positions = parameterization.to_positions(params)
    return rotate_layout_to_level(positions, anchor_ids[0], anchor_ids[1])


def train_model(args: argparse.Namespace, device: torch.device) -> tuple[DenseGraphSeedNet, list[dict[str, float]]]:
    probe = make_graph_batch(
        2,
        device=device,
        random_fraction=args.random_fraction,
        include_index_features=args.index_features,
    )
    model = DenseGraphSeedNet(
        probe.node_features.shape[-1],
        probe.edge_features.shape[-1],
        hidden=args.hidden,
        layers=args.layers,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")
    history: list[dict[str, float]] = []
    step_index = 0
    model.train()
    for epoch in range(args.epochs):
        epoch_start = time.perf_counter()
        epoch_losses: list[float] = []
        for _ in range(args.steps_per_epoch):
            batch = make_graph_batch(
                args.batch_size,
                device=device,
                random_fraction=args.random_fraction,
                include_index_features=args.index_features,
            )
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=args.amp and device.type == "cuda",
            ):
                pred = model(batch.node_features, batch.edge_features, batch.mask, batch.pair_mask)
                loss, parts = distance_shape_loss(pred, batch)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            loss_value = float(loss.detach().cpu())
            epoch_losses.append(loss_value)
            history.append(
                {
                    "step": float(step_index),
                    "epoch": float(epoch + 1),
                    "loss": loss_value,
                    **parts,
                }
            )
            step_index += 1
        elapsed = time.perf_counter() - epoch_start
        print(
            f"epoch {epoch + 1}/{args.epochs}: loss={np.mean(epoch_losses):.5f} "
            f"last={epoch_losses[-1]:.5f} elapsed={elapsed:.1f}s"
        )
    return model, history


@torch.no_grad()
def evaluate_model(
    model: DenseGraphSeedNet,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[list[EvalRow], GraphBatch, torch.Tensor]:
    model.eval()
    rows: list[EvalRow] = []
    eval_batch = make_graph_batch(
        args.eval_cases,
        device=device,
        random_fraction=0.5,
        include_index_features=args.index_features,
    )
    pred = model(
        eval_batch.node_features,
        eval_batch.edge_features,
        eval_batch.mask,
        eval_batch.pair_mask,
    )
    for case_index in range(args.eval_cases):
        family = eval_batch.family[case_index]
        n = eval_batch.node_counts[case_index]
        truth = graph_to_truth(eval_batch, case_index)
        pairs = graph_to_pairs(eval_batch, case_index)
        scale = float(eval_batch.scale_m[case_index].detach().cpu())
        seeds = {
            "triangulated": baseline_seed_positions("triangulated", pairs),
            "degree-shell": baseline_seed_positions("degree-shell", pairs),
            "ml-graph": tensor_to_positions(pred[case_index], n, scale),
        }
        for method, seed_positions in seeds.items():
            rmse, max_residual = pair_metrics(seed_positions, pairs)
            max_offset, median_offset, p95_offset = offset_summary(truth, seed_positions)
            rows.append(
                EvalRow(
                    family=family,
                    case_index=case_index,
                    method=method,
                    stage="seed",
                    anchors=n,
                    pairs=len(pairs),
                    rmse_m=rmse,
                    max_residual_m=max_residual,
                    max_offset_m=max_offset,
                    median_offset_m=median_offset,
                    p95_offset_m=p95_offset,
                )
            )
            solved, solved_rmse, solved_max_residual = solve_from_seed(
                seed_positions,
                pairs,
                max_iterations=args.solver_iterations,
            )
            max_offset, median_offset, p95_offset = offset_summary(truth, solved)
            rows.append(
                EvalRow(
                    family=family,
                    case_index=case_index,
                    method=method,
                    stage="lm",
                    anchors=n,
                    pairs=len(pairs),
                    rmse_m=solved_rmse,
                    max_residual_m=solved_max_residual,
                    max_offset_m=max_offset,
                    median_offset_m=median_offset,
                    p95_offset_m=p95_offset,
                )
            )
    return rows, eval_batch, pred


def summarize(rows: list[EvalRow]) -> list[dict[str, float | str]]:
    summary: list[dict[str, float | str]] = []
    groups = sorted({(row.family, row.method, row.stage) for row in rows})
    for family, method, stage in groups:
        part = [row for row in rows if row.family == family and row.method == method and row.stage == stage]
        summary.append(
            {
                "family": family,
                "method": method,
                "stage": stage,
                "cases": float(len(part)),
                "median_max_offset_m": float(np.median([row.max_offset_m for row in part])),
                "p90_max_offset_m": float(np.quantile([row.max_offset_m for row in part], 0.90)),
                "median_rmse_m": float(np.median([row.rmse_m for row in part])),
                "under_20cm": float(np.mean([row.max_offset_m <= 0.20 for row in part])),
                "under_50cm": float(np.mean([row.max_offset_m <= 0.50 for row in part])),
                "under_1m": float(np.mean([row.max_offset_m <= 1.00 for row in part])),
            }
        )
    return summary


def write_rows(path: Path, rows: list[EvalRow]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, EvalRow.__dataclass_fields__.keys())
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


def write_history(path: Path, history: list[dict[str, float]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, history[0].keys())
        writer.writeheader()
        writer.writerows(history)


def draw_layout_panel(
    ax,
    truth: dict[str, tuple[float, float]],
    estimate: dict[str, tuple[float, float]],
    pairs: list[AnchorPairDistance],
    title: str,
    color: str,
) -> None:
    ax.set_facecolor(TOKENS["panel"])
    for pair in pairs:
        ax.plot(
            [truth[pair.anchor_a_id][0], truth[pair.anchor_b_id][0]],
            [truth[pair.anchor_a_id][1], truth[pair.anchor_b_id][1]],
            color=NEUTRAL["light"],
            linewidth=0.55,
            alpha=0.75,
            zorder=1,
        )
    offsets = align_offsets_no_scale(truth, estimate)
    max_offset, median_offset, _p95 = offset_summary(truth, estimate)
    aligned = aligned_estimate(truth, estimate)
    for anchor_id, true_point in truth.items():
        solved_point = aligned[anchor_id]
        ax.plot(
            [true_point[0], solved_point[0]],
            [true_point[1], solved_point[1]],
            color=ORANGE["mid"],
            linewidth=0.75,
            alpha=0.45,
            zorder=2,
        )
    ax.scatter(
        [point[0] for point in truth.values()],
        [point[1] for point in truth.values()],
        s=24,
        facecolors=TOKENS["panel"],
        edgecolors=NEUTRAL["dark"],
        linewidths=0.8,
        zorder=3,
    )
    ax.scatter(
        [point[0] for point in aligned.values()],
        [point[1] for point in aligned.values()],
        s=28,
        color=color,
        edgecolors=TOKENS["ink"],
        linewidths=0.6,
        zorder=4,
    )
    xs = [x for x, _ in truth.values()] + [x for x, _ in aligned.values()]
    ys = [y for _, y in truth.values()] + [y for _, y in aligned.values()]
    span = max(max(xs) - min(xs), max(ys) - min(ys), 1.0)
    pad = span * 0.12
    ax.set_xlim(min(xs) - pad, max(xs) + pad)
    ax.set_ylim(min(ys) - pad, max(ys) + pad)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, color=TOKENS["grid"], linewidth=0.55)
    ax.tick_params(labelsize=6.5, colors=TOKENS["muted"], length=0)
    for spine in ax.spines.values():
        spine.set_color(TOKENS["axis"])
    ax.set_title(
        f"{title}\nmax {max_offset:.2f} m | med {median_offset:.2f} m",
        loc="left",
        fontsize=9,
        color=TOKENS["ink"],
        fontweight="semibold",
    )


def aligned_estimate(
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
    best_aligned: np.ndarray | None = None
    best_max = math.inf
    for reflection in (1.0, -1.0):
        reflected = source_centered.copy()
        reflected[:, 1] *= reflection
        u, _s, vt = np.linalg.svd(reflected.T @ target_centered)
        aligned = reflected @ (u @ vt) + target_center
        max_offset = float(np.linalg.norm(aligned - target, axis=1).max())
        if max_offset < best_max:
            best_max = max_offset
            best_aligned = aligned
    assert best_aligned is not None
    return {anchor_id: (float(x), float(y)) for anchor_id, (x, y) in zip(ids, best_aligned)}


def make_infographic(
    path: Path,
    history: list[dict[str, float]],
    rows: list[EvalRow],
    batch: GraphBatch,
    pred: torch.Tensor,
) -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": TOKENS["surface"],
            "savefig.facecolor": TOKENS["surface"],
            "font.family": ["Segoe UI", "DejaVu Sans", "Arial", "sans-serif"],
        }
    )
    fig = plt.figure(figsize=(16, 10), dpi=180)
    gs = fig.add_gridspec(3, 3, height_ratios=[0.95, 1.15, 1.15], hspace=0.52, wspace=0.30)
    fig.text(
        0.035,
        0.975,
        "Learned graph seed for anchor geometry",
        ha="left",
        va="top",
        fontsize=20,
        fontweight="bold",
        color=TOKENS["ink"],
    )
    fig.text(
        0.035,
        0.94,
        "Dense message-passing model trained on mixed random and grid layouts. Edges use true <=8 m links with 3 cm noise and one-third NLOS positive bias.",
        ha="left",
        va="top",
        fontsize=9.5,
        color=TOKENS["muted"],
    )

    ax_loss = fig.add_subplot(gs[0, 0])
    ax_loss.set_facecolor(TOKENS["panel"])
    ax_loss.plot([row["step"] for row in history], [row["loss"] for row in history], color=BLUE["dark"], linewidth=1.5)
    ax_loss.set_title("Training loss", loc="left", fontsize=11, fontweight="bold", color=TOKENS["ink"])
    ax_loss.set_xlabel("step", fontsize=8, color=TOKENS["muted"])
    ax_loss.set_ylabel("shape loss", fontsize=8, color=TOKENS["muted"])
    ax_loss.grid(True, color=TOKENS["grid"], linewidth=0.6)

    ax_bar = fig.add_subplot(gs[0, 1:])
    ax_bar.set_facecolor(TOKENS["panel"])
    summary = summarize(rows)
    methods = ["triangulated", "degree-shell", "ml-graph"]
    labels = []
    values = []
    colors = []
    palette = {"triangulated": NEUTRAL["base"], "degree-shell": GOLD["base"], "ml-graph": BLUE["base"]}
    for family in ["grid", "random"]:
        for method in methods:
            part = [
                item
                for item in summary
                if item["family"] == family and item["method"] == method and item["stage"] == "lm"
            ]
            if not part:
                continue
            labels.append(f"{family}\n{method}")
            values.append(float(part[0]["median_max_offset_m"]))
            colors.append(palette[method])
    ax_bar.bar(range(len(values)), values, color=colors, edgecolor=TOKENS["ink"], linewidth=0.6)
    ax_bar.set_xticks(range(len(labels)), labels, fontsize=7.5)
    ax_bar.set_ylabel("median max offset after LM (m)", fontsize=8, color=TOKENS["muted"])
    ax_bar.set_title("Does the seed land in a better basin?", loc="left", fontsize=11, fontweight="bold", color=TOKENS["ink"])
    ax_bar.grid(True, axis="y", color=TOKENS["grid"], linewidth=0.6)

    example_indices = choose_examples(batch)
    for row_index, case_index in enumerate(example_indices):
        truth = graph_to_truth(batch, case_index)
        pairs = graph_to_pairs(batch, case_index)
        n = batch.node_counts[case_index]
        scale = float(batch.scale_m[case_index].detach().cpu())
        ml_seed = tensor_to_positions(pred[case_index], n, scale)
        tri_seed = baseline_seed_positions("triangulated", pairs)
        degree_seed = baseline_seed_positions("degree-shell", pairs)
        examples = [
            ("triangulated seed", tri_seed, NEUTRAL["base"]),
            ("degree-shell seed", degree_seed, GOLD["base"]),
            ("ml-graph seed", ml_seed, BLUE["base"]),
        ]
        for col_index, (title, estimate, color) in enumerate(examples):
            ax = fig.add_subplot(gs[row_index + 1, col_index])
            draw_layout_panel(
                ax,
                truth,
                estimate,
                pairs,
                f"{batch.family[case_index]} case: {title}",
                color,
            )

    for ax in fig.axes:
        ax.tick_params(colors=TOKENS["muted"])
        for spine in ax.spines.values():
            spine.set_color(TOKENS["axis"])
    fig.savefig(path, bbox_inches="tight", dpi=180)
    plt.close(fig)


def choose_examples(batch: GraphBatch) -> list[int]:
    grid_indices = [index for index, family in enumerate(batch.family) if family == "grid"]
    random_indices = [index for index, family in enumerate(batch.family) if family == "random"]
    indices = []
    indices.append(grid_indices[len(grid_indices) // 2] if grid_indices else 0)
    indices.append(random_indices[len(random_indices) // 2] if random_indices else min(1, len(batch.family) - 1))
    return indices


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a graph neural seed model for anchor geometry.")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:0")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--steps-per-epoch", type=int, default=45)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--hidden", type=int, default=96)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.02)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=2.0)
    parser.add_argument("--random-fraction", type=float, default=0.5)
    parser.add_argument("--eval-cases", type=int, default=40)
    parser.add_argument("--solver-iterations", type=int, default=80)
    parser.add_argument("--seed", type=int, default=20260626)
    parser.add_argument("--amp", action="store_true", help="Use CUDA autocast/GradScaler.")
    parser.add_argument("--no-index-features", dest="index_features", action="store_false")
    parser.set_defaults(index_features=True)
    parser.add_argument("--prefix", default="anchor_solver_ml_seed")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    device = choose_device(args.device)
    print(f"device={device} torch={torch.__version__} cuda_available={torch.cuda.is_available()}")
    if device.type == "cuda":
        print(f"cuda_device={torch.cuda.get_device_name(device)}")
        torch.backends.cuda.matmul.allow_tf32 = True
    started = time.perf_counter()
    model, history = train_model(args, device)
    rows, eval_batch, pred = evaluate_model(model, args, device)
    elapsed = time.perf_counter() - started

    checkpoint_path = OUTPUTS / f"{args.prefix}.pt"
    metrics_path = OUTPUTS / f"{args.prefix}_metrics.csv"
    history_path = OUTPUTS / f"{args.prefix}_history.csv"
    figure_path = OUTPUTS / f"{args.prefix}.png"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "args": vars(args),
            "edge_radius_m": EDGE_RADIUS_M,
            "min_anchor_spacing_m": MIN_ANCHOR_SPACING_M,
        },
        checkpoint_path,
    )
    write_rows(metrics_path, rows)
    write_history(history_path, history)
    make_infographic(figure_path, history, rows, eval_batch, pred)

    print(f"elapsed_s={elapsed:.1f}")
    for row in summarize(rows):
        if row["stage"] != "lm":
            continue
        print(
            f"{row['family']} {row['method']} lm: "
            f"median_max={row['median_max_offset_m']:.3f}m "
            f"p90={row['p90_max_offset_m']:.3f}m "
            f"median_rmse={row['median_rmse_m']:.4f}m "
            f"under1m={row['under_1m']:.0%}"
        )
    print(f"Wrote {checkpoint_path}")
    print(f"Wrote {metrics_path}")
    print(f"Wrote {history_path}")
    print(f"Wrote {figure_path}")


if __name__ == "__main__":
    main()
