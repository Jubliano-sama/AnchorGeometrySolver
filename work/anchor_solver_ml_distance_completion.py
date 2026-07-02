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
MIN_MEASURED_DEGREE = 3
MIN_VERTEX_CONNECTIVITY = 3
MIN_EXTENT_M = 8.0
MAX_EXTENT_M = 40.0
NOISE_SIGMA_M = 0.03
NLOS_PROBABILITY = 1.0 / 3.0
NLOS_MAX_OFFSET_M = 0.20
KNOWN_SIGMA_M = 0.05
MIN_NODES = 5
MAX_NODES = 50
INF_DISTANCE = 1e6

OFFICE_SHAPES = (
    "rectangle",
    "corridor",
    "l_shape",
    "t_shape",
    "u_shape",
    "cross",
    "hollow_square",
    "disc",
    "annulus",
    "rooms",
)

TOKENS = {
    "surface": "#FCFCFD",
    "panel": "#FFFFFF",
    "ink": "#1F2430",
    "muted": "#6F768A",
    "grid": "#E6E8F0",
    "axis": "#D7DBE7",
}
BLUE = {"base": "#A3BEFA", "dark": "#2E4780"}
GOLD = {"base": "#FFE15B", "dark": "#736422"}
ORANGE = {"base": "#F0986E", "dark": "#804126"}
PINK = {"base": "#F390CA", "dark": "#8A3A6F"}
OLIVE = {"base": "#A3D576", "dark": "#386411"}
NEUTRAL = {"light": "#E2E5EA", "base": "#C5CAD3", "dark": "#464C55"}


@dataclass
class GraphBatch:
    positions_m: torch.Tensor
    mask: torch.Tensor
    true_dist_m: torch.Tensor
    measured_dist_m: torch.Tensor
    measured_mask: torch.Tensor
    pair_mask: torch.Tensor
    shortest_m: torch.Tensor
    hop_count: torch.Tensor
    scale_m: torch.Tensor
    node_features: torch.Tensor
    edge_features: torch.Tensor
    family: list[str]
    shape: list[str]
    node_counts: list[int]


@dataclass(frozen=True)
class EvalRow:
    family: str
    shape: str
    case_index: int
    method: str
    anchors: int
    known_pairs: int
    full_pairs: int
    missing_mae_m: float
    known_rmse_m: float
    known_max_residual_m: float
    max_offset_m: float
    median_offset_m: float
    p95_offset_m: float


class DistanceCompletionNet(nn.Module):
    """Permutation-equivariant dense graph distance completer."""

    def __init__(
        self,
        node_feature_count: int,
        edge_feature_count: int,
        *,
        hidden: int,
        layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.node_projection = nn.Sequential(
            nn.Linear(node_feature_count, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        self.message_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden * 2 + edge_feature_count, hidden),
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
        self.graph_projection = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        self.pair_head = nn.Sequential(
            nn.Linear(hidden * 4 + edge_feature_count, hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.SiLU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(
        self,
        node_features: torch.Tensor,
        edge_features: torch.Tensor,
        mask: torch.Tensor,
        measured_mask: torch.Tensor,
        pair_mask: torch.Tensor,
    ) -> torch.Tensor:
        h = self.node_projection(node_features) * mask.unsqueeze(-1)
        hop_norm = edge_features[..., 7]
        measured_weight = measured_mask.float()
        short_path_weight = torch.exp(-3.0 * hop_norm) * (
            (hop_norm > 0.0) & (hop_norm <= 0.50) & pair_mask & ~measured_mask
        ).float()
        message_weight = (measured_weight + 0.35 * short_path_weight).unsqueeze(-1)
        normalizer = message_weight.sum(dim=2).clamp_min(1.0)
        for message_layer, update_layer, norm in zip(
            self.message_layers,
            self.update_layers,
            self.norms,
        ):
            source = h.unsqueeze(1).expand(-1, h.shape[1], -1, -1)
            target = h.unsqueeze(2).expand(-1, -1, h.shape[1], -1)
            message_input = torch.cat([target, source, edge_features], dim=-1)
            messages = message_layer(message_input) * message_weight
            aggregate = messages.sum(dim=2) / normalizer
            update = update_layer(torch.cat([h, aggregate], dim=-1))
            h = norm(h + update) * mask.unsqueeze(-1)

        weights = mask.float().unsqueeze(-1)
        graph_context = (h * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        graph_context = self.graph_projection(graph_context)
        graph_pair = graph_context.unsqueeze(1).unsqueeze(2).expand(-1, h.shape[1], h.shape[1], -1)
        hi = h.unsqueeze(2).expand(-1, -1, h.shape[1], -1)
        hj = h.unsqueeze(1).expand(-1, h.shape[1], -1, -1)
        pair_input = torch.cat([hi + hj, torch.abs(hi - hj), hi * hj, graph_pair, edge_features], dim=-1)
        raw = self.pair_head(pair_input).squeeze(-1)
        positive = F.softplus(raw) + 1e-4
        shortest_norm = edge_features[..., 5]
        radio_lower_norm = edge_features[..., 9]
        missing = edge_features[..., 2] > 0.5
        upper = torch.maximum(shortest_norm, radio_lower_norm + 0.05)
        bounded_missing = radio_lower_norm + torch.sigmoid(raw) * (upper - radio_lower_norm)
        pred = torch.where(missing, bounded_missing, positive)
        pred = 0.5 * (pred + pred.transpose(1, 2))
        eye = torch.eye(pred.shape[1], dtype=torch.bool, device=pred.device).unsqueeze(0)
        return pred.masked_fill(~pair_mask | eye, 0.0)


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested, but torch.cuda.is_available() is false.")
    return device


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_scaler(device: torch.device, enabled: bool):
    try:
        return torch.amp.GradScaler(device.type, enabled=enabled and device.type == "cuda")
    except TypeError:
        return torch.cuda.amp.GradScaler(enabled=enabled and device.type == "cuda")


def autocast_context(device: torch.device, enabled: bool):
    return torch.autocast(
        device_type=device.type,
        dtype=torch.float16,
        enabled=enabled and device.type == "cuda",
    )


def sample_extent(*, elongated: bool = False, squareish: bool = False) -> tuple[float, float]:
    if squareish:
        base = random.uniform(MIN_EXTENT_M, MAX_EXTENT_M)
        return base, random.uniform(max(MIN_EXTENT_M, base * 0.82), min(MAX_EXTENT_M, base * 1.18))
    if elongated:
        long_side = random.uniform(18.0, MAX_EXTENT_M)
        short_side = random.uniform(MIN_EXTENT_M, min(16.0, long_side * 0.55))
        return (long_side, short_side) if random.random() < 0.5 else (short_side, long_side)
    return random.uniform(MIN_EXTENT_M, MAX_EXTENT_M), random.uniform(MIN_EXTENT_M, MAX_EXTENT_M)


def sample_office_shape() -> tuple[str, dict[str, float]]:
    shape = random.choice(OFFICE_SHAPES)
    if shape == "corridor":
        width, height = sample_extent(elongated=True)
    elif shape in {"disc", "annulus", "hollow_square"}:
        width, height = sample_extent(squareish=True)
    else:
        width, height = sample_extent()
    params: dict[str, float] = {"width": width, "height": height}
    if shape == "l_shape":
        params["leg_x"] = width * random.uniform(0.34, 0.58)
        params["leg_y"] = height * random.uniform(0.34, 0.58)
    elif shape == "t_shape":
        params["bar_w"] = width * random.uniform(0.24, 0.42)
        params["top_h"] = height * random.uniform(0.26, 0.42)
    elif shape == "u_shape":
        params["leg_w"] = width * random.uniform(0.22, 0.35)
        params["bottom_h"] = height * random.uniform(0.24, 0.42)
    elif shape == "cross":
        params["bar_w"] = width * random.uniform(0.22, 0.38)
        params["bar_h"] = height * random.uniform(0.22, 0.38)
    elif shape == "hollow_square":
        params["hole_w"] = width * random.uniform(0.26, 0.52)
        params["hole_h"] = height * random.uniform(0.26, 0.52)
    elif shape == "disc":
        side = min(width, height)
        params["width"] = side
        params["height"] = side
        params["radius"] = side * 0.50
    elif shape == "annulus":
        side = min(width, height)
        params["width"] = side
        params["height"] = side
        params["outer_radius"] = side * 0.50
        params["inner_radius"] = side * random.uniform(0.18, 0.34)
    elif shape == "rooms":
        params["corridor_w"] = width * random.uniform(0.16, 0.28)
        params["corridor_h"] = height * random.uniform(0.16, 0.28)
    return shape, params


def shape_area_estimate(shape: str, params: dict[str, float]) -> float:
    width = params["width"]
    height = params["height"]
    if shape == "rectangle" or shape == "corridor":
        return width * height
    if shape == "l_shape":
        return params["leg_x"] * height + width * params["leg_y"] - params["leg_x"] * params["leg_y"]
    if shape == "t_shape":
        return params["bar_w"] * height + width * params["top_h"] - params["bar_w"] * params["top_h"]
    if shape == "u_shape":
        return 2.0 * params["leg_w"] * height + width * params["bottom_h"] - 2.0 * params["leg_w"] * params["bottom_h"]
    if shape == "cross":
        return params["bar_w"] * height + width * params["bar_h"] - params["bar_w"] * params["bar_h"]
    if shape == "hollow_square":
        return width * height - params["hole_w"] * params["hole_h"]
    if shape == "disc":
        return math.pi * params["radius"] ** 2
    if shape == "annulus":
        return math.pi * (params["outer_radius"] ** 2 - params["inner_radius"] ** 2)
    if shape == "rooms":
        corridor = params["corridor_w"] * height + width * params["corridor_h"] - params["corridor_w"] * params["corridor_h"]
        rooms = 0.72 * width * height
        return min(width * height, max(corridor, rooms))
    return width * height


def target_anchor_count(shape: str, params: dict[str, float]) -> int:
    area = shape_area_estimate(shape, params)
    density_area = random.uniform(18.0, 42.0)
    estimate = int(round(area / density_area))
    if random.random() < 0.18:
        estimate = random.randint(MIN_NODES, min(MAX_NODES, 12))
    jitter = random.randint(-4, 6)
    return max(MIN_NODES, min(MAX_NODES, estimate + jitter))


def shape_contains(points: torch.Tensor, shape: str, params: dict[str, float]) -> torch.Tensor:
    x = points[:, 0]
    y = points[:, 1]
    width = params["width"]
    height = params["height"]
    inside = (x >= 0.0) & (x <= width) & (y >= 0.0) & (y <= height)
    if shape in {"rectangle", "corridor"}:
        return inside
    if shape == "l_shape":
        return inside & ((x <= params["leg_x"]) | (y <= params["leg_y"]))
    if shape == "t_shape":
        centered_bar = torch.abs(x - width * 0.5) <= params["bar_w"] * 0.5
        top_bar = y >= height - params["top_h"]
        return inside & (centered_bar | top_bar)
    if shape == "u_shape":
        return inside & ((x <= params["leg_w"]) | (x >= width - params["leg_w"]) | (y <= params["bottom_h"]))
    if shape == "cross":
        vertical = torch.abs(x - width * 0.5) <= params["bar_w"] * 0.5
        horizontal = torch.abs(y - height * 0.5) <= params["bar_h"] * 0.5
        return inside & (vertical | horizontal)
    if shape == "hollow_square":
        hole_x0 = (width - params["hole_w"]) * 0.5
        hole_x1 = hole_x0 + params["hole_w"]
        hole_y0 = (height - params["hole_h"]) * 0.5
        hole_y1 = hole_y0 + params["hole_h"]
        hole = (x >= hole_x0) & (x <= hole_x1) & (y >= hole_y0) & (y <= hole_y1)
        return inside & ~hole
    if shape == "disc":
        dx = x - width * 0.5
        dy = y - height * 0.5
        return dx * dx + dy * dy <= params["radius"] ** 2
    if shape == "annulus":
        dx = x - width * 0.5
        dy = y - height * 0.5
        radius_sq = dx * dx + dy * dy
        return (radius_sq <= params["outer_radius"] ** 2) & (radius_sq >= params["inner_radius"] ** 2)
    if shape == "rooms":
        vertical = torch.abs(x - width * 0.5) <= params["corridor_w"] * 0.5
        horizontal = torch.abs(y - height * 0.5) <= params["corridor_h"] * 0.5
        room_margin_x = width * 0.07
        room_margin_y = height * 0.07
        rooms = (x >= room_margin_x) & (x <= width - room_margin_x) & (y >= room_margin_y) & (y <= height - room_margin_y)
        return inside & (vertical | horizontal | rooms)
    return inside


def choose_spaced_subset(candidates: torch.Tensor, target_n: int) -> torch.Tensor | None:
    if candidates.shape[0] == 0:
        return None
    order = torch.randperm(candidates.shape[0], device=candidates.device)
    candidates = candidates[order]
    chosen: list[torch.Tensor] = []
    for candidate in candidates:
        if len(chosen) >= target_n:
            break
        if chosen:
            previous = torch.stack(chosen)
            if torch.linalg.norm(previous - candidate, dim=1).min() < MIN_ANCHOR_SPACING_M:
                continue
        chosen.append(candidate)
    if len(chosen) < MIN_NODES:
        return None
    return torch.stack(chosen[: min(len(chosen), target_n)])


def random_points_in_shape(shape: str, params: dict[str, float], target_n: int, device: torch.device) -> torch.Tensor | None:
    width = params["width"]
    height = params["height"]
    for _attempt in range(80):
        candidate_count = max(target_n * 96, 1024)
        candidates = torch.rand((candidate_count, 2), device=device)
        candidates[:, 0] *= width
        candidates[:, 1] *= height
        candidates = candidates[shape_contains(candidates, shape, params)]
        points = choose_spaced_subset(candidates, target_n)
        if points is not None and points.shape[0] >= min(target_n, MIN_NODES):
            return points
    return None


def grid_points_in_shape(shape: str, params: dict[str, float], target_n: int, device: torch.device) -> torch.Tensor | None:
    width = params["width"]
    height = params["height"]
    for _attempt in range(80):
        spacing_x = random.uniform(3.4, 7.8)
        spacing_y = random.uniform(3.4, 7.8)
        offset_x = random.uniform(0.0, spacing_x)
        offset_y = random.uniform(0.0, spacing_y)
        xs = torch.arange(offset_x, width + spacing_x * 0.35, spacing_x, dtype=torch.float32, device=device)
        ys = torch.arange(offset_y, height + spacing_y * 0.35, spacing_y, dtype=torch.float32, device=device)
        if xs.numel() == 0 or ys.numel() == 0:
            continue
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        candidates = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1)
        candidates = candidates[shape_contains(candidates, shape, params)]
        if candidates.shape[0] < MIN_NODES:
            continue
        jitter = torch.randn_like(candidates) * random.uniform(0.0, 0.18)
        candidates = candidates + jitter
        candidates[:, 0].clamp_(0.0, width)
        candidates[:, 1].clamp_(0.0, height)
        candidates = candidates[shape_contains(candidates, shape, params)]
        if candidates.shape[0] < MIN_NODES:
            continue
        count = min(target_n, MAX_NODES, int(candidates.shape[0]))
        if count < MIN_NODES:
            continue
        points = choose_spaced_subset(candidates, count)
        if points is not None:
            return points
    return random_points_in_shape(shape, params, target_n, device)


def random_rigid_transform(points: torch.Tensor) -> torch.Tensor:
    center = points.mean(dim=0, keepdim=True)
    shifted = points - center
    angle = random.uniform(-math.pi, math.pi)
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    rotation = torch.tensor([[cos_a, -sin_a], [sin_a, cos_a]], dtype=points.dtype, device=points.device)
    transformed = shifted @ rotation.T
    if random.random() < 0.5:
        transformed[:, 0] *= -1.0
    transformed = transformed - transformed.min(dim=0).values
    return transformed


def measured_adjacency(points: torch.Tensor) -> torch.Tensor:
    distances = torch.cdist(points.unsqueeze(0), points.unsqueeze(0))[0]
    eye = torch.eye(points.shape[0], dtype=torch.bool, device=points.device)
    return (distances <= EDGE_RADIUS_M) & ~eye


def is_connected(adjacency: torch.Tensor, n: int) -> bool:
    if n <= 1:
        return True
    neighbors = adjacency[:n, :n].detach().cpu().numpy()
    seen = {0}
    queue = [0]
    while queue:
        current = queue.pop()
        for neighbor in np.nonzero(neighbors[current])[0]:
            if int(neighbor) not in seen:
                seen.add(int(neighbor))
                queue.append(int(neighbor))
    return len(seen) == n


def _numpy_connected_after_removal(neighbors: np.ndarray, removed: set[int]) -> bool:
    active = [index for index in range(neighbors.shape[0]) if index not in removed]
    if len(active) <= 1:
        return True
    start = active[0]
    seen = {start}
    queue = [start]
    while queue:
        current = queue.pop()
        for neighbor in np.nonzero(neighbors[current])[0]:
            neighbor_i = int(neighbor)
            if neighbor_i in removed or neighbor_i in seen:
                continue
            seen.add(neighbor_i)
            queue.append(neighbor_i)
    return len(seen) == len(active)


def measured_graph_vertex_connectivity(points: torch.Tensor, *, cap: int = MIN_VERTEX_CONNECTIVITY) -> int:
    """Return a capped vertex-connectivity diagnostic for the measured graph."""
    n = int(points.shape[0])
    if n <= 1:
        return n
    adjacency = measured_adjacency(points)
    degree = adjacency.float().sum(dim=1)
    if not is_connected(adjacency, n):
        return 0
    upper = max(1, min(int(cap), int(degree.min().detach().cpu())))
    neighbors = adjacency.detach().cpu().numpy()
    if upper <= 1:
        return 1
    for removed in range(n):
        if not _numpy_connected_after_removal(neighbors, {removed}):
            return 1
    if upper <= 2:
        return 2
    for first in range(n):
        for second in range(first + 1, n):
            if not _numpy_connected_after_removal(neighbors, {first, second}):
                return 2
    return upper


def measured_graph_diagnostics(points: torch.Tensor) -> dict[str, float]:
    adjacency = measured_adjacency(points)
    degree = adjacency.float().sum(dim=1)
    n = int(points.shape[0])
    edge_count = float(adjacency.float().sum().detach().cpu()) * 0.5
    required_edges = max(1.0, 2.0 * n - 3.0)
    return {
        "min_degree": float(degree.min().detach().cpu()) if n else 0.0,
        "mean_degree": float(degree.mean().detach().cpu()) if n else 0.0,
        "measured_edges": edge_count,
        "rigidity_surplus": edge_count - required_edges,
        "vertex_connectivity_capped3": float(measured_graph_vertex_connectivity(points, cap=3)),
    }


def is_fair_measured_graph(points: torch.Tensor, *, min_vertex_connectivity: int = 1) -> bool:
    if points.shape[0] < MIN_NODES or points.shape[0] > MAX_NODES:
        return False
    adjacency = measured_adjacency(points)
    degree = adjacency.float().sum(dim=1)
    if not bool((degree >= MIN_MEASURED_DEGREE).all().detach().cpu()) or not is_connected(adjacency, int(points.shape[0])):
        return False
    if min_vertex_connectivity > 1:
        return measured_graph_vertex_connectivity(points, cap=min_vertex_connectivity) >= min_vertex_connectivity
    return True


def prune_to_fair_measured_graph(points: torch.Tensor, *, min_vertex_connectivity: int = 1) -> torch.Tensor | None:
    if points.shape[0] < MIN_NODES:
        return None
    kept = torch.ones(points.shape[0], dtype=torch.bool, device=points.device)
    for _iteration in range(points.shape[0]):
        active = points[kept]
        if active.shape[0] < MIN_NODES:
            return None
        adjacency = measured_adjacency(active)
        degree = adjacency.float().sum(dim=1)
        if bool((degree >= MIN_MEASURED_DEGREE).all().detach().cpu()) and is_connected(adjacency, int(active.shape[0])):
            if min_vertex_connectivity > 1 and measured_graph_vertex_connectivity(active, cap=min_vertex_connectivity) < min_vertex_connectivity:
                return None
            return active
        remove_local = int(torch.argmin(degree).detach().cpu())
        active_indices = torch.nonzero(kept, as_tuple=False).flatten()
        kept[active_indices[remove_local]] = False
    return None


def make_positions_case(
    device: torch.device,
    random_fraction: float,
    *,
    min_vertex_connectivity: int = 1,
) -> tuple[torch.Tensor, str, str]:
    for _attempt in range(700):
        shape, params = sample_office_shape()
        target_n = target_anchor_count(shape, params)
        if random.random() < random_fraction:
            points = random_points_in_shape(shape, params, target_n, device)
            family = "random"
        else:
            points = grid_points_in_shape(shape, params, target_n, device)
            family = "grid"
        if points is None or points.shape[0] < MIN_NODES or points.shape[0] > MAX_NODES:
            continue
        points = random_rigid_transform(points)
        points = prune_to_fair_measured_graph(points, min_vertex_connectivity=min_vertex_connectivity)
        if points is not None:
            return points, family, shape

    for _attempt in range(200):
        shape = "rectangle"
        params = {"width": random.uniform(MIN_EXTENT_M, 14.0), "height": random.uniform(MIN_EXTENT_M, 14.0)}
        target_n = random.randint(max(MIN_NODES, 8), 16)
        points = grid_points_in_shape(shape, params, target_n, device)
        if points is None:
            points = random_points_in_shape(shape, params, target_n, device)
        if points is None:
            continue
        points = random_rigid_transform(points)
        points = prune_to_fair_measured_graph(points, min_vertex_connectivity=min_vertex_connectivity)
        if points is not None:
            return points, "grid", "fallback_rectangle"
    raise RuntimeError("Could not generate a connected fair graph with minimum measured degree 3.")

def make_graph_batch(
    batch_size: int,
    *,
    device: torch.device,
    random_fraction: float,
    min_vertex_connectivity: int = 1,
    graph_solution_features: bool = False,
) -> GraphBatch:
    positions = torch.zeros((batch_size, MAX_NODES, 2), dtype=torch.float32, device=device)
    mask = torch.zeros((batch_size, MAX_NODES), dtype=torch.bool, device=device)
    families: list[str] = []
    shapes: list[str] = []
    node_counts: list[int] = []
    for batch_index in range(batch_size):
        points, family, shape = make_positions_case(device, random_fraction, min_vertex_connectivity=min_vertex_connectivity)
        n = int(points.shape[0])
        positions[batch_index, :n] = points
        mask[batch_index, :n] = True
        families.append(family)
        shapes.append(shape)
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
    shortest, hops = shortest_paths(measured_dist, measured_mask, pair_mask)
    node_features = build_node_features(measured_dist, measured_mask, mask, pair_mask, shortest, hops, scale)
    edge_features = build_edge_features(measured_dist, measured_mask, pair_mask, shortest, hops, scale)
    batch = GraphBatch(
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
        batch = append_graph_solution_edge_features(batch)
    return batch


def shortest_paths(
    measured_dist: torch.Tensor,
    measured_mask: torch.Tensor,
    pair_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size, nodes, _ = measured_dist.shape
    eye = torch.eye(nodes, dtype=torch.bool, device=measured_dist.device).unsqueeze(0)
    metric = torch.where(measured_mask, measured_dist, torch.full_like(measured_dist, INF_DISTANCE))
    metric = metric.masked_fill(eye, 0.0)
    hop = torch.where(measured_mask, torch.ones_like(measured_dist), torch.full_like(measured_dist, INF_DISTANCE))
    hop = hop.masked_fill(eye, 0.0)
    for k in range(nodes):
        metric = torch.minimum(metric, metric[:, :, k : k + 1] + metric[:, k : k + 1, :])
        hop = torch.minimum(hop, hop[:, :, k : k + 1] + hop[:, k : k + 1, :])
    metric = torch.where(pair_mask | eye, metric, torch.zeros_like(metric))
    hop = torch.where(pair_mask | eye, hop, torch.zeros_like(hop))
    return metric, hop


def build_node_features(
    measured_dist: torch.Tensor,
    measured_mask: torch.Tensor,
    mask: torch.Tensor,
    pair_mask: torch.Tensor,
    shortest: torch.Tensor,
    hops: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    batch_size, nodes, _ = measured_dist.shape
    scale_view = scale.view(batch_size, 1, 1)
    adj = measured_mask.float()
    dist_norm = measured_dist / scale_view
    degree = adj.sum(dim=2)
    node_count = mask.float().sum(dim=1, keepdim=True)
    degree_norm = degree / (node_count - 1.0).clamp_min(1.0)
    degree_count_norm = torch.clamp(degree / 12.0, 0.0, 2.0)
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
    max_dist = dist_norm.masked_fill(~measured_mask, 0.0).max(dim=2).values
    variance = (((dist_norm - mean_dist.unsqueeze(2)) * adj) ** 2).sum(dim=2) / count
    std_dist = torch.sqrt(variance.clamp_min(0.0))

    edge_count = adj.sum(dim=(1, 2)).view(batch_size, 1) * 0.5
    density = edge_count / (node_count * (node_count - 1.0) * 0.5).clamp_min(1.0)
    density = density.expand(-1, nodes)
    node_count_norm = (node_count / MAX_NODES).expand(-1, nodes)
    required_edges = (2.0 * node_count - 3.0).clamp_min(1.0)
    rigidity_surplus = torch.clamp((edge_count - required_edges) / node_count.clamp_min(1.0), -2.0, 2.0) / 2.0
    rigidity_surplus = rigidity_surplus.expand(-1, nodes)
    min_degree_norm = (min_degree / (node_count - 1.0).clamp_min(1.0)).expand(-1, nodes)

    valid_path = pair_mask & (hops < INF_DISTANCE * 0.5)
    path_count = valid_path.float().sum(dim=2).clamp_min(1.0)
    hop_values = torch.where(valid_path, hops, torch.zeros_like(hops))
    mean_hop_raw = hop_values.sum(dim=2) / path_count
    mean_hop = torch.clamp(mean_hop_raw / 8.0, 0.0, 3.0)
    max_hop = torch.clamp(hop_values.masked_fill(~valid_path, 0.0).max(dim=2).values / 8.0, 0.0, 3.0)
    closeness = 1.0 / (1.0 + mean_hop_raw)
    shortest_norm = torch.clamp(shortest / scale_view, 0.0, 8.0)
    shortest_values = torch.where(valid_path, shortest_norm, torch.zeros_like(shortest_norm))
    mean_shortest = shortest_values.sum(dim=2) / path_count
    max_shortest = shortest_values.masked_fill(~valid_path, 0.0).max(dim=2).values

    common_neighbors_raw = adj @ adj
    triangle_twice = (common_neighbors_raw * adj).sum(dim=2)
    clustering = triangle_twice / (degree * (degree - 1.0)).clamp_min(1.0)
    bridge_like = (adj * torch.exp(-common_neighbors_raw)).sum(dim=2) / degree.clamp_min(1.0)

    features = torch.stack(
        [
            mask.float(),
            degree_norm,
            degree_count_norm,
            degree_z,
            low_degree_score,
            mean_dist,
            min_dist,
            max_dist,
            std_dist,
            density,
            node_count_norm,
            min_degree_norm,
            rigidity_surplus,
            closeness,
            mean_hop,
            max_hop,
            mean_shortest,
            max_shortest,
            clustering,
            bridge_like,
        ],
        dim=-1,
    )
    return features * mask.unsqueeze(-1)

def build_edge_features(
    measured_dist: torch.Tensor,
    measured_mask: torch.Tensor,
    pair_mask: torch.Tensor,
    shortest: torch.Tensor,
    hops: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    scale_view = scale.view(-1, 1, 1)
    measured_norm = measured_dist / scale_view
    shortest_norm = torch.clamp(shortest / scale_view, 0.0, 8.0)
    hop_norm = torch.clamp(hops / 8.0, 0.0, 8.0)
    missing_mask = pair_mask & ~measured_mask
    adj = measured_mask.float()
    degree = adj.sum(dim=2)
    active = pair_mask.any(dim=2)
    node_count = active.float().sum(dim=1, keepdim=True).clamp_min(1.0)
    edge_count = adj.sum(dim=(1, 2)).view(-1, 1, 1) * 0.5
    density = edge_count / (node_count.view(-1, 1, 1) * (node_count.view(-1, 1, 1) - 1.0) * 0.5).clamp_min(1.0)
    required_edges = (2.0 * node_count.view(-1, 1, 1) - 3.0).clamp_min(1.0)
    rigidity_surplus = torch.clamp((edge_count - required_edges) / node_count.view(-1, 1, 1), -2.0, 2.0) / 2.0
    degree_i = degree.unsqueeze(2).expand_as(measured_norm)
    degree_j = degree.unsqueeze(1).expand_as(measured_norm)
    degree_den = (node_count.view(-1, 1, 1) - 1.0).clamp_min(1.0)
    degree_i_norm = degree_i / degree_den
    degree_j_norm = degree_j / degree_den
    endpoint_min_degree = torch.minimum(degree_i_norm, degree_j_norm)
    endpoint_degree_delta = torch.abs(degree_i_norm - degree_j_norm)
    common_neighbors_raw = adj @ adj
    common_neighbors = common_neighbors_raw.clamp_max(12.0) / 12.0
    jaccard = common_neighbors_raw / (degree_i + degree_j - common_neighbors_raw).clamp_min(1.0)
    bridge_like = torch.exp(-common_neighbors_raw)
    measured_clipped = torch.clamp(measured_norm, 0.0, 2.5)
    edge_radius_norm = (EDGE_RADIUS_M / scale).view(-1, 1, 1).expand_as(measured_norm)
    min_spacing_norm = (MIN_ANCHOR_SPACING_M / scale).view(-1, 1, 1).expand_as(measured_norm)
    return torch.stack(
        [
            pair_mask.float(),
            measured_mask.float(),
            missing_mask.float(),
            measured_clipped,
            measured_clipped * measured_clipped,
            shortest_norm,
            torch.exp(-shortest_norm),
            hop_norm,
            common_neighbors,
            edge_radius_norm,
            min_spacing_norm,
            degree_i_norm,
            degree_j_norm,
            endpoint_min_degree,
            endpoint_degree_delta,
            jaccard,
            bridge_like,
            rigidity_surplus.expand_as(measured_norm),
            density.expand_as(measured_norm),
        ],
        dim=-1,
    ) * pair_mask.unsqueeze(-1)

def masked_smooth_l1(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    beta: float,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    if not bool(mask.any()):
        return pred.sum() * 0.0
    values = F.smooth_l1_loss(pred[mask], target[mask], beta=beta, reduction="none")
    if weights is None:
        return values.mean()
    selected_weights = weights[mask]
    return (values * selected_weights).sum() / selected_weights.sum().clamp_min(1e-6)


def triangle_metric_loss(pred_norm: torch.Tensor, pair_mask: torch.Tensor) -> torch.Tensor:
    nodes = pred_norm.shape[1]
    upper = torch.triu(torch.ones((nodes, nodes), dtype=torch.bool, device=pred_norm.device), diagonal=1).unsqueeze(0)
    valid_pairs = pair_mask & upper
    if not bool(valid_pairs.any()):
        return pred_norm.sum() * 0.0
    stride = 2 if nodes > 36 else 1
    losses: list[torch.Tensor] = []
    for k in range(0, nodes, stride):
        via_k = pred_norm[:, :, k : k + 1] + pred_norm[:, k : k + 1, :]
        violation = F.relu(pred_norm - via_k)
        losses.append(violation[valid_pairs].mean())
    return torch.stack(losses).mean() if losses else pred_norm.sum() * 0.0


def edm_rank2_loss(pred_norm: torch.Tensor, mask: torch.Tensor, pair_mask: torch.Tensor) -> torch.Tensor:
    pred_float = pred_norm.float()
    nodes = pred_norm.shape[1]
    eye = torch.eye(nodes, dtype=torch.float32, device=pred_norm.device).unsqueeze(0)
    valid = mask.float()
    counts = valid.sum(dim=1).clamp_min(1.0)
    projector = eye - valid.unsqueeze(2) * valid.unsqueeze(1) / counts.view(-1, 1, 1)
    squared = (pred_float * pred_float) * pair_mask.float()
    gram = -0.5 * (projector @ squared @ projector)
    gram = 0.5 * (gram + gram.transpose(1, 2))
    gram = torch.nan_to_num(gram.float(), nan=0.0, posinf=1e6, neginf=-1e6)
    jitter_eye = torch.eye(nodes, dtype=torch.float32, device=pred_norm.device).unsqueeze(0)
    eigenvalues = None
    for jitter in (0.0, 1e-6, 1e-4, 1e-3):
        try:
            eigenvalues = torch.linalg.eigvalsh(gram + jitter * jitter_eye)
            break
        except RuntimeError:
            continue
    if eigenvalues is None:
        return pred_norm.sum() * 0.0
    negative_loss = F.relu(-eigenvalues).pow(2).sum(dim=1) / counts
    high_rank = F.relu(eigenvalues[:, :-2]).pow(2).sum(dim=1) / counts
    scale = F.relu(eigenvalues[:, -2:]).pow(2).sum(dim=1).clamp_min(1.0)
    return ((negative_loss + high_rank) / scale.sqrt()).mean()



def distance_completion_loss(pred_norm: torch.Tensor, batch: GraphBatch, *, edm_weight: float = 0.02, radio_lower_weight: float = 0.0) -> tuple[torch.Tensor, dict[str, float]]:
    upper = torch.triu(
        torch.ones((MAX_NODES, MAX_NODES), dtype=torch.bool, device=pred_norm.device),
        diagonal=1,
    ).unsqueeze(0)
    pairs = batch.pair_mask & upper
    known = batch.measured_mask & upper
    missing = pairs & ~batch.measured_mask
    target_norm = batch.true_dist_m / batch.scale_m.view(-1, 1, 1)
    measured_norm = batch.measured_dist_m / batch.scale_m.view(-1, 1, 1)
    missing_weights = 0.30 + torch.exp(-target_norm / 3.0)
    missing_loss = masked_smooth_l1(pred_norm, target_norm, missing, beta=0.04, weights=missing_weights)
    log_missing_loss = masked_smooth_l1(
        torch.log(pred_norm.clamp_min(1e-4)),
        torch.log(target_norm.clamp_min(1e-4)),
        missing,
        beta=0.035,
        weights=missing_weights,
    )
    known_denoise_loss = masked_smooth_l1(pred_norm, target_norm, known, beta=0.04)
    known_input_loss = masked_smooth_l1(pred_norm, measured_norm, known, beta=0.04)
    all_loss = masked_smooth_l1(pred_norm, target_norm, pairs, beta=0.06)
    radio_lower_norm = (EDGE_RADIUS_M / batch.scale_m).view(-1, 1, 1)
    radio_hinge_values = F.relu(radio_lower_norm - pred_norm)
    radio_lower_loss = radio_hinge_values[missing].pow(2).mean() if bool(missing.any()) else pred_norm.sum() * 0.0
    metric_loss = triangle_metric_loss(pred_norm, batch.pair_mask)
    edm_loss = edm_rank2_loss(pred_norm, batch.mask, batch.pair_mask)
    loss = (
        missing_loss
        + 0.25 * log_missing_loss
        + 0.18 * known_denoise_loss
        + 0.06 * known_input_loss
        + 0.08 * all_loss
        + radio_lower_weight * radio_lower_loss
        + 0.025 * metric_loss
        + edm_weight * edm_loss
    )
    return loss, {
        "missing": float(missing_loss.detach().cpu()),
        "known_true": float(known_denoise_loss.detach().cpu()),
        "known_input": float(known_input_loss.detach().cpu()),
        "all": float(all_loss.detach().cpu()),
        "metric": float(metric_loss.detach().cpu()),
        "edm": float(edm_loss.detach().cpu()),
        "radio_lower": float(radio_lower_loss.detach().cpu()),
    }


def graph_to_truth(batch: GraphBatch, case_index: int) -> dict[str, tuple[float, float]]:
    n = batch.node_counts[case_index]
    points = batch.positions_m[case_index, :n].detach().cpu().numpy()
    return {f"A{i:02d}": (float(points[i, 0]), float(points[i, 1])) for i in range(n)}


def known_pairs_from_batch(batch: GraphBatch, case_index: int) -> list[AnchorPairDistance]:
    n = batch.node_counts[case_index]
    measured = batch.measured_mask[case_index, :n, :n].detach().cpu().numpy()
    distances = batch.measured_dist_m[case_index, :n, :n].detach().cpu().numpy()
    pairs: list[AnchorPairDistance] = []
    for i in range(n):
        for j in range(i + 1, n):
            if measured[i, j]:
                pairs.append(AnchorPairDistance(f"A{i:02d}", f"A{j:02d}", float(distances[i, j]), KNOWN_SIGMA_M, True, "known"))
    return pairs


def graph_shortest_mds_distances(batch: GraphBatch) -> torch.Tensor:
    """Approximate graph-shortest solve: MDS on noisy all-pairs shortest-path distances."""
    out = torch.zeros_like(batch.true_dist_m)
    for case_index, node_count in enumerate(batch.node_counts):
        n = int(node_count)
        if n <= 1:
            continue
        distances = batch.shortest_m[case_index, :n, :n].clone()
        finite = distances < INF_DISTANCE * 0.5
        finite_offdiag = finite & ~torch.eye(n, dtype=torch.bool, device=distances.device)
        if not bool(finite_offdiag.any()):
            continue
        fill = distances[finite_offdiag].max() * 1.25
        distances = torch.where(finite, distances, fill)
        distances = 0.5 * (distances + distances.T)
        distances.fill_diagonal_(0.0)
        centering = torch.eye(n, dtype=distances.dtype, device=distances.device) - torch.full(
            (n, n), 1.0 / float(n), dtype=distances.dtype, device=distances.device
        )
        gram = -0.5 * centering @ (distances * distances) @ centering
        try:
            values, vectors = torch.linalg.eigh(gram)
        except RuntimeError:
            out[case_index, :n, :n] = distances
            continue
        order = torch.argsort(values, descending=True)[:2]
        top_values = torch.clamp(values[order], min=0.0)
        coords = vectors[:, order] * torch.sqrt(top_values).unsqueeze(0)
        solved = torch.cdist(coords.unsqueeze(0), coords.unsqueeze(0))[0]
        out[case_index, :n, :n] = solved
    return out


def append_graph_solution_edge_features(batch: GraphBatch) -> GraphBatch:
    graph_dist = graph_shortest_mds_distances(batch)
    scale = batch.scale_m.view(-1, 1, 1)
    solved_norm = torch.clamp(graph_dist / scale, 0.0, 8.0)
    delta_norm = torch.clamp((graph_dist - batch.shortest_m) / scale, -4.0, 4.0)
    ratio = torch.where(batch.shortest_m > 0.05, graph_dist / batch.shortest_m.clamp_min(0.05), torch.zeros_like(graph_dist))
    ratio = torch.clamp(ratio, 0.0, 3.0)
    valid = batch.pair_mask.float()
    extra = torch.stack([solved_norm, delta_norm, ratio, valid], dim=-1) * batch.pair_mask.unsqueeze(-1)
    return GraphBatch(
        positions_m=batch.positions_m,
        mask=batch.mask,
        true_dist_m=batch.true_dist_m,
        measured_dist_m=batch.measured_dist_m,
        measured_mask=batch.measured_mask,
        pair_mask=batch.pair_mask,
        shortest_m=batch.shortest_m,
        hop_count=batch.hop_count,
        scale_m=batch.scale_m,
        node_features=batch.node_features,
        edge_features=torch.cat([batch.edge_features, extra], dim=-1),
        family=batch.family,
        shape=batch.shape,
        node_counts=batch.node_counts,
    )

def completed_pairs_from_prediction(
    batch: GraphBatch,
    pred_norm: torch.Tensor,
    case_index: int,
    *,
    predicted_sigma_m: float,
    predicted_sigma_slope: float = 0.0,
    closest_predicted_pairs: int = 0,
    closest_predicted_pairs_per_anchor: float = 0.0,
) -> tuple[list[AnchorPairDistance], np.ndarray, float]:
    n = batch.node_counts[case_index]
    scale = float(batch.scale_m[case_index].detach().cpu())
    measured = batch.measured_mask[case_index, :n, :n].detach().cpu().numpy()
    measured_dist = batch.measured_dist_m[case_index, :n, :n].detach().cpu().numpy()
    pred_dist = (pred_norm[case_index, :n, :n].detach().cpu().numpy() * scale).astype(float)
    pairs: list[AnchorPairDistance] = []
    predicted_candidates: list[tuple[float, int, int, float]] = []
    for i in range(n):
        for j in range(i + 1, n):
            if measured[i, j]:
                pairs.append(AnchorPairDistance(f"A{i:02d}", f"A{j:02d}", float(measured_dist[i, j]), KNOWN_SIGMA_M, True, "known"))
            else:
                predicted = max(float(pred_dist[i, j]), 0.05)
                sigma = predicted_sigma_m * (1.0 + predicted_sigma_slope * max(predicted - EDGE_RADIUS_M, 0.0) / EDGE_RADIUS_M)
                predicted_candidates.append((predicted, i, j, sigma))

    limit = len(predicted_candidates)
    if closest_predicted_pairs > 0:
        limit = min(limit, closest_predicted_pairs)
    if closest_predicted_pairs_per_anchor > 0.0:
        scaled_limit = int(math.ceil(float(closest_predicted_pairs_per_anchor) * n))
        limit = min(limit, max(0, scaled_limit))
    predicted_candidates.sort(key=lambda item: (item[0], item[1], item[2]))
    for predicted, i, j, sigma in predicted_candidates[:limit]:
        pairs.append(AnchorPairDistance(f"A{i:02d}", f"A{j:02d}", predicted, sigma, True, "predicted"))
    return pairs, pred_dist, scale

def oracle_full_pairs(batch: GraphBatch, case_index: int) -> list[AnchorPairDistance]:
    n = batch.node_counts[case_index]
    true_dist = batch.true_dist_m[case_index, :n, :n].detach().cpu().numpy()
    return [
        AnchorPairDistance(f"A{i:02d}", f"A{j:02d}", float(true_dist[i, j]), KNOWN_SIGMA_M, True, "oracle")
        for i in range(n)
        for j in range(i + 1, n)
    ]


def classical_mds_seed(pairs: list[AnchorPairDistance]) -> dict[str, tuple[float, float]]:
    processed = _preprocess_pairs(pairs, min_sigma_m=0.02, min_distance_m=0.05)
    anchor_ids = _anchor_ids(processed)
    index = {anchor_id: i for i, anchor_id in enumerate(anchor_ids)}
    n = len(anchor_ids)
    matrix = np.full((n, n), np.inf, dtype=float)
    np.fill_diagonal(matrix, 0.0)
    for pair in processed:
        i = index[pair.anchor_a_id]
        j = index[pair.anchor_b_id]
        distance = float(pair.distance_m)
        if distance < matrix[i, j]:
            matrix[i, j] = matrix[j, i] = distance
    for k in range(n):
        matrix = np.minimum(matrix, matrix[:, k : k + 1] + matrix[k : k + 1, :])
    finite = np.isfinite(matrix)
    if not finite.all():
        finite_offdiag = matrix[finite & ~np.eye(n, dtype=bool)]
        fill = float(finite_offdiag.max() * 1.25) if finite_offdiag.size else 1.0
        matrix[~finite] = fill
    squared = matrix * matrix
    centering = np.eye(n) - np.full((n, n), 1.0 / n)
    gram = -0.5 * centering @ squared @ centering
    values, vectors = np.linalg.eigh(gram)
    order = np.argsort(values)[::-1][:2]
    values = np.maximum(values[order], 0.0)
    coords = vectors[:, order] * np.sqrt(values)
    positions = {anchor_id: (float(coords[i, 0]), float(coords[i, 1])) for anchor_id, i in index.items()}
    return rotate_layout_to_level(positions, anchor_ids[0], anchor_ids[1])

def solve_from_seed(
    seed_positions: dict[str, tuple[float, float]],
    solve_pairs: list[AnchorPairDistance],
    *,
    max_iterations: int,
) -> dict[str, tuple[float, float]]:
    processed = _preprocess_pairs(solve_pairs, min_sigma_m=0.02, min_distance_m=0.05)
    anchor_ids = _anchor_ids(processed)
    parameterization = _Parameterization(anchor_ids)
    try:
        seed_positions = rotate_layout_to_level(seed_positions, anchor_ids[0], anchor_ids[1])
    except ValueError:
        ax, ay = seed_positions[anchor_ids[0]]
        seed_positions = {anchor_id: (x - ax, y - ay) for anchor_id, (x, y) in seed_positions.items()}
    seed_params = _positions_to_params(parameterization, seed_positions)
    params, _energy = _local_minimize(seed_params, parameterization, processed, None, max_iterations=max_iterations)
    positions = parameterization.to_positions(params)
    return rotate_layout_to_level(positions, anchor_ids[0], anchor_ids[1])


def known_only_solution(known_pairs: list[AnchorPairDistance], *, max_iterations: int) -> dict[str, tuple[float, float]]:
    processed = _preprocess_pairs(known_pairs, min_sigma_m=0.02, min_distance_m=0.05)
    anchor_ids = _anchor_ids(processed)
    parameterization = _Parameterization(anchor_ids)
    scale = _layout_scale(processed)
    params = _triangulated_seed(parameterization, processed, scale)
    params, _energy = _local_minimize(params, parameterization, processed, None, max_iterations=max_iterations)
    positions = parameterization.to_positions(params)
    return rotate_layout_to_level(positions, anchor_ids[0], anchor_ids[1])


def weak_completed_pairs(
    completed_pairs: list[AnchorPairDistance],
    *,
    max_predicted_distance_m: float,
    sigma_multiplier: float,
) -> list[AnchorPairDistance]:
    pairs: list[AnchorPairDistance] = []
    for pair in completed_pairs:
        if pair.source == "known":
            pairs.append(pair)
            continue
        if pair.source == "predicted" and pair.distance_m <= max_predicted_distance_m:
            pairs.append(
                AnchorPairDistance(
                    pair.anchor_a_id,
                    pair.anchor_b_id,
                    pair.distance_m,
                    sigma_m=pair.sigma_m * sigma_multiplier,
                    enabled=pair.enabled,
                    source="weak-predicted",
                )
            )
    return pairs


def completion_solution_weak_polish(
    completed_pairs: list[AnchorPairDistance],
    known_pairs: list[AnchorPairDistance],
    *,
    max_iterations: int,
    weak_polish_iterations: int,
    max_predicted_distance_m: float,
    sigma_multiplier: float,
) -> dict[str, tuple[float, float]]:
    seed = classical_mds_seed(completed_pairs)
    full_solution = solve_from_seed(seed, completed_pairs, max_iterations=max_iterations)
    weak_pairs = weak_completed_pairs(
        completed_pairs,
        max_predicted_distance_m=max_predicted_distance_m,
        sigma_multiplier=sigma_multiplier,
    )
    if weak_polish_iterations <= 0 or len(weak_pairs) <= len(known_pairs):
        return full_solution
    return solve_from_seed(full_solution, weak_pairs, max_iterations=weak_polish_iterations)



def completion_solution(
    completed_pairs: list[AnchorPairDistance],
    known_pairs: list[AnchorPairDistance],
    *,
    max_iterations: int,
    polish_known_iterations: int,
) -> dict[str, tuple[float, float]]:
    seed = classical_mds_seed(completed_pairs)
    full_solution = solve_from_seed(seed, completed_pairs, max_iterations=max_iterations)
    if polish_known_iterations <= 0:
        return full_solution
    return solve_from_seed(full_solution, known_pairs, max_iterations=polish_known_iterations)


def pair_metrics(
    positions: dict[str, tuple[float, float]],
    pairs: list[AnchorPairDistance],
) -> tuple[float, float]:
    residuals = pair_residuals(positions, pairs)
    values = list(residuals.values())
    if not values:
        return math.inf, math.inf
    return math.sqrt(sum(value * value for value in values) / len(values)), max(abs(value) for value in values)


def align_offsets_no_scale(
    truth: dict[str, tuple[float, float]],
    estimate: dict[str, tuple[float, float]],
) -> np.ndarray:
    ids = sorted(set(truth) & set(estimate))
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


def missing_mae(
    batch: GraphBatch,
    pred_norm: torch.Tensor,
    case_index: int,
) -> float:
    n = batch.node_counts[case_index]
    missing = (~batch.measured_mask[case_index, :n, :n]) & batch.pair_mask[case_index, :n, :n]
    upper = torch.triu(torch.ones((n, n), dtype=torch.bool, device=pred_norm.device), diagonal=1)
    missing = missing & upper
    if not bool(missing.any()):
        return 0.0
    scale = batch.scale_m[case_index]
    pred_m = pred_norm[case_index, :n, :n] * scale
    true_m = batch.true_dist_m[case_index, :n, :n]
    return float(torch.abs(pred_m[missing] - true_m[missing]).mean().detach().cpu())


def save_training_checkpoint(
    model: DistanceCompletionNet,
    args: argparse.Namespace,
    path: Path,
    *,
    epoch: int,
    epoch_mean_loss: float,
    step: int,
) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "args": vars(args),
            "edge_radius_m": EDGE_RADIUS_M,
            "known_sigma_m": KNOWN_SIGMA_M,
            "predicted_sigma_m": args.predicted_sigma,
            "predicted_sigma_slope": args.predicted_sigma_slope,
            "closest_predicted_pairs": args.closest_predicted_pairs,
            "closest_predicted_pairs_per_anchor": args.closest_predicted_pairs_per_anchor,
            "min_measured_degree": MIN_MEASURED_DEGREE,
            "edm_weight": args.edm_weight,
            "radio_lower_weight": args.radio_lower_weight,
            "diagnostic_graph_features": True,
            "no_anchor_order_features": True,
            "partial_training_checkpoint": True,
            "epoch": epoch,
            "epoch_mean_loss": epoch_mean_loss,
            "step": step,
        },
        path,
    )

def train_model(args: argparse.Namespace, device: torch.device) -> tuple[DistanceCompletionNet, list[dict[str, float]]]:
    probe = make_graph_batch(2, device=device, random_fraction=args.random_fraction)
    model = DistanceCompletionNet(
        probe.node_features.shape[-1],
        probe.edge_features.shape[-1],
        hidden=args.hidden,
        layers=args.layers,
        dropout=args.dropout,
    ).to(device)
    if args.init_checkpoint:
        checkpoint = torch.load(args.init_checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"loaded_init_checkpoint={args.init_checkpoint}", flush=True)
    print(
        f"feature_dims node={probe.node_features.shape[-1]} edge={probe.edge_features.shape[-1]} "
        f"min_measured_degree={MIN_MEASURED_DEGREE} closest_predicted_pairs={args.closest_predicted_pairs} "
        f"closest_predicted_pairs_per_anchor={args.closest_predicted_pairs_per_anchor}",
        flush=True,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = make_scaler(device, args.amp)
    history: list[dict[str, float]] = []
    started = time.perf_counter()
    step_index = 0
    best_epoch_loss = math.inf
    model.train()
    for epoch in range(args.epochs):
        epoch_losses: list[float] = []
        for step_in_epoch in range(args.steps_per_epoch):
            batch = make_graph_batch(args.batch_size, device=device, random_fraction=args.random_fraction)
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device, args.amp):
                pred = model(batch.node_features, batch.edge_features, batch.mask, batch.measured_mask, batch.pair_mask)
                loss, parts = distance_completion_loss(pred, batch, edm_weight=args.edm_weight, radio_lower_weight=args.radio_lower_weight)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            loss_value = float(loss.detach().cpu())
            epoch_losses.append(loss_value)
            row = {
                "step": float(step_index),
                "epoch": float(epoch + 1),
                "loss": loss_value,
                **parts,
            }
            history.append(row)
            step_index += 1
            if step_index == 1 or step_index % args.log_every == 0:
                elapsed = time.perf_counter() - started
                cases_per_s = (step_index * args.batch_size) / max(elapsed, 1e-9)
                print(
                    f"progress step={step_index}/{args.epochs * args.steps_per_epoch} "
                    f"epoch={epoch + 1}/{args.epochs} inner={step_in_epoch + 1}/{args.steps_per_epoch} "
                    f"loss={loss_value:.5f} missing={parts['missing']:.5f} known={parts['known_true']:.5f} radio={parts['radio_lower']:.5f} "
                    f"cases_per_s={cases_per_s:.1f} elapsed_s={elapsed:.1f}",
                    flush=True,
                )
        epoch_mean_loss = float(np.mean(epoch_losses))
        print(
            f"epoch_done {epoch + 1}/{args.epochs}: mean_loss={epoch_mean_loss:.5f} "
            f"last_loss={epoch_losses[-1]:.5f}",
            flush=True,
        )
        if args.checkpoint_every_epochs > 0:
            if epoch_mean_loss < best_epoch_loss:
                best_epoch_loss = epoch_mean_loss
                best_path = OUTPUTS / f"{args.prefix}_best_loss.pt"
                save_training_checkpoint(model, args, best_path, epoch=epoch + 1, epoch_mean_loss=epoch_mean_loss, step=step_index)
                print(f"checkpoint_best_loss epoch={epoch + 1} path={best_path}", flush=True)
            if (epoch + 1) % args.checkpoint_every_epochs == 0:
                latest_path = OUTPUTS / f"{args.prefix}_latest.pt"
                save_training_checkpoint(model, args, latest_path, epoch=epoch + 1, epoch_mean_loss=epoch_mean_loss, step=step_index)
                print(f"checkpoint_latest epoch={epoch + 1} path={latest_path}", flush=True)
    return model, history


@torch.no_grad()
def evaluate_model(
    model: DistanceCompletionNet,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[list[EvalRow], GraphBatch, torch.Tensor]:
    model.eval()
    eval_batch = make_graph_batch(args.eval_cases, device=device, random_fraction=0.5)
    pred = model(
        eval_batch.node_features,
        eval_batch.edge_features,
        eval_batch.mask,
        eval_batch.measured_mask,
        eval_batch.pair_mask,
    )
    rows: list[EvalRow] = []
    for case_index in range(args.eval_cases):
        truth = graph_to_truth(eval_batch, case_index)
        known_pairs = known_pairs_from_batch(eval_batch, case_index)
        completed_pairs, _pred_matrix, _scale = completed_pairs_from_prediction(
            eval_batch,
            pred,
            case_index,
            predicted_sigma_m=args.predicted_sigma,
            predicted_sigma_slope=args.predicted_sigma_slope,
            closest_predicted_pairs=args.closest_predicted_pairs,
            closest_predicted_pairs_per_anchor=args.closest_predicted_pairs_per_anchor,
        )
        oracle_pairs = oracle_full_pairs(eval_batch, case_index)
        full_pairs = len(oracle_pairs)
        known_count = len(known_pairs)
        mae = missing_mae(eval_batch, pred, case_index)
        solutions = {
            "known_triangulated_lm": known_only_solution(known_pairs, max_iterations=args.solver_iterations),
            "ml_completed_mds_full_lm_polish": completion_solution(
                completed_pairs,
                known_pairs,
                max_iterations=args.solver_iterations,
                polish_known_iterations=args.polish_iterations,
            ),
            "ml_completed_weak_polish": completion_solution_weak_polish(
                completed_pairs,
                known_pairs,
                max_iterations=args.solver_iterations,
                weak_polish_iterations=args.weak_polish_iterations,
                max_predicted_distance_m=args.weak_completion_max_distance,
                sigma_multiplier=args.weak_completion_sigma_multiplier,
            ),
            "oracle_full_mds_lm": completion_solution(
                oracle_pairs,
                known_pairs,
                max_iterations=args.solver_iterations,
                polish_known_iterations=0,
            ),
        }
        for method, positions in solutions.items():
            known_rmse, known_max = pair_metrics(positions, known_pairs)
            max_offset, median_offset, p95_offset = offset_summary(truth, positions)
            rows.append(
                EvalRow(
                    family=eval_batch.family[case_index],
                    shape=eval_batch.shape[case_index],
                    case_index=case_index,
                    method=method,
                    anchors=eval_batch.node_counts[case_index],
                    known_pairs=known_count,
                    full_pairs=full_pairs,
                    missing_mae_m=mae,
                    known_rmse_m=known_rmse,
                    known_max_residual_m=known_max,
                    max_offset_m=max_offset,
                    median_offset_m=median_offset,
                    p95_offset_m=p95_offset,
                )
            )
    return rows, eval_batch, pred


def summarize(rows: list[EvalRow]) -> list[dict[str, float | str]]:
    summary: list[dict[str, float | str]] = []
    for family, method in sorted({(row.family, row.method) for row in rows}):
        part = [row for row in rows if row.family == family and row.method == method]
        summary.append(
            {
                "family": family,
                "method": method,
                "cases": float(len(part)),
                "median_missing_mae_m": float(np.median([row.missing_mae_m for row in part])),
                "median_max_offset_m": float(np.median([row.max_offset_m for row in part])),
                "p90_max_offset_m": float(np.quantile([row.max_offset_m for row in part], 0.90)),
                "median_known_rmse_m": float(np.median([row.known_rmse_m for row in part])),
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
    best_aligned = None
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


def draw_solution_panel(
    ax,
    truth: dict[str, tuple[float, float]],
    estimate: dict[str, tuple[float, float]],
    pairs: list[AnchorPairDistance],
    title: str,
    color: str,
) -> None:
    aligned = aligned_estimate(truth, estimate)
    max_offset, median_offset, _p95 = offset_summary(truth, estimate)
    known_rmse, _known_max = pair_metrics(estimate, pairs)
    ax.set_facecolor(TOKENS["panel"])
    for pair in pairs:
        ax.plot(
            [truth[pair.anchor_a_id][0], truth[pair.anchor_b_id][0]],
            [truth[pair.anchor_a_id][1], truth[pair.anchor_b_id][1]],
            color=NEUTRAL["light"],
            linewidth=0.45,
            alpha=0.45,
            zorder=1,
        )
    for anchor_id, true_point in truth.items():
        point = aligned[anchor_id]
        ax.plot(
            [true_point[0], point[0]],
            [true_point[1], point[1]],
            color=ORANGE["dark"],
            linewidth=0.65,
            alpha=0.45,
            zorder=2,
        )
    ax.scatter(
        [point[0] for point in truth.values()],
        [point[1] for point in truth.values()],
        s=22,
        facecolors=TOKENS["panel"],
        edgecolors=NEUTRAL["dark"],
        linewidths=0.75,
        zorder=3,
    )
    ax.scatter(
        [point[0] for point in aligned.values()],
        [point[1] for point in aligned.values()],
        s=26,
        color=color,
        edgecolors=TOKENS["ink"],
        linewidths=0.55,
        zorder=4,
    )
    xs = [x for x, _ in truth.values()] + [x for x, _ in aligned.values()]
    ys = [y for _, y in truth.values()] + [y for _, y in aligned.values()]
    span = max(max(xs) - min(xs), max(ys) - min(ys), 1.0)
    pad = span * 0.12
    ax.set_xlim(min(xs) - pad, max(xs) + pad)
    ax.set_ylim(min(ys) - pad, max(ys) + pad)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, color=TOKENS["grid"], linewidth=0.5)
    ax.tick_params(labelsize=6.5, colors=TOKENS["muted"], length=0)
    for spine in ax.spines.values():
        spine.set_color(TOKENS["axis"])
    ax.set_title(
        f"{title}\nmax {max_offset:.2f}m | med {median_offset:.2f}m | known RMSE {known_rmse:.3f}m",
        loc="left",
        fontsize=8.7,
        fontweight="semibold",
        color=TOKENS["ink"],
    )


def make_infographic(
    path: Path,
    rows: list[EvalRow],
    history: list[dict[str, float]],
    batch: GraphBatch,
    pred: torch.Tensor,
    args: argparse.Namespace,
) -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": TOKENS["surface"],
            "savefig.facecolor": TOKENS["surface"],
            "font.family": ["Segoe UI", "DejaVu Sans", "Arial", "sans-serif"],
        }
    )
    fig = plt.figure(figsize=(17, 10.5), dpi=180)
    gs = fig.add_gridspec(3, 3, height_ratios=[0.9, 1.12, 1.12], hspace=0.52, wspace=0.28)
    fig.text(
        0.035,
        0.975,
        "ML distance completion for anchor geometry",
        ha="left",
        va="top",
        fontsize=20,
        fontweight="bold",
        color=TOKENS["ink"],
    )
    fig.text(
        0.035,
        0.94,
        "The network predicts missing anchor-to-anchor distances from observed noisy <=8 m ranges. The completed graph is solved by classical MDS + LM, then polished on real known ranges.",
        ha="left",
        va="top",
        fontsize=9.3,
        color=TOKENS["muted"],
    )

    ax_loss = fig.add_subplot(gs[0, 0])
    ax_loss.set_facecolor(TOKENS["panel"])
    ax_loss.plot([row["step"] for row in history], [row["loss"] for row in history], color=BLUE["dark"], linewidth=1.3)
    ax_loss.plot([row["step"] for row in history], [row["missing"] for row in history], color=PINK["dark"], linewidth=1.0, alpha=0.85)
    ax_loss.set_title("Training loss", loc="left", fontsize=11, fontweight="bold", color=TOKENS["ink"])
    ax_loss.set_xlabel("step", fontsize=8, color=TOKENS["muted"])
    ax_loss.set_ylabel("normalized loss", fontsize=8, color=TOKENS["muted"])
    ax_loss.grid(True, color=TOKENS["grid"], linewidth=0.55)

    ax_bar = fig.add_subplot(gs[0, 1:])
    ax_bar.set_facecolor(TOKENS["panel"])
    summary = summarize(rows)
    labels: list[str] = []
    values: list[float] = []
    colors: list[str] = []
    palette = {
        "known_triangulated_lm": NEUTRAL["base"],
        "ml_completed_mds_full_lm_polish": BLUE["base"],
        "ml_completed_weak_polish": PINK["base"],
        "oracle_full_mds_lm": OLIVE["base"],
    }
    for family in ("grid", "random"):
        for method in ("known_triangulated_lm", "ml_completed_mds_full_lm_polish", "ml_completed_weak_polish", "oracle_full_mds_lm"):
            item = next((row for row in summary if row["family"] == family and row["method"] == method), None)
            if item is None:
                continue
            labels.append(f"{family}\n{short_method(method)}")
            values.append(float(item["median_max_offset_m"]))
            colors.append(palette[method])
    ax_bar.bar(range(len(values)), values, color=colors, edgecolor=TOKENS["ink"], linewidth=0.55)
    ax_bar.set_xticks(range(len(labels)), labels, fontsize=7)
    ax_bar.set_ylabel("median max offset (m)", fontsize=8, color=TOKENS["muted"])
    ax_bar.set_title("Completed distances vs known-only solve", loc="left", fontsize=11, fontweight="bold", color=TOKENS["ink"])
    ax_bar.grid(True, axis="y", color=TOKENS["grid"], linewidth=0.55)

    examples = choose_examples(batch)
    for row_index, case_index in enumerate(examples):
        truth = graph_to_truth(batch, case_index)
        known_pairs = known_pairs_from_batch(batch, case_index)
        completed_pairs, _pred_matrix, _scale = completed_pairs_from_prediction(
            batch,
            pred,
            case_index,
            predicted_sigma_m=args.predicted_sigma,
            predicted_sigma_slope=args.predicted_sigma_slope,
            closest_predicted_pairs=args.closest_predicted_pairs,
            closest_predicted_pairs_per_anchor=args.closest_predicted_pairs_per_anchor,
        )
        oracle_pairs = oracle_full_pairs(batch, case_index)
        solutions = [
            (
                "known-only",
                known_only_solution(known_pairs, max_iterations=args.solver_iterations),
                NEUTRAL["base"],
            ),
            (
                "ML completed",
                completion_solution(
                    completed_pairs,
                    known_pairs,
                    max_iterations=args.solver_iterations,
                    polish_known_iterations=args.polish_iterations,
                ),
                BLUE["base"],
            ),
            (
                "oracle full",
                completion_solution(oracle_pairs, known_pairs, max_iterations=args.solver_iterations, polish_known_iterations=0),
                OLIVE["base"],
            ),
        ]
        for col_index, (name, solution, color) in enumerate(solutions):
            ax = fig.add_subplot(gs[row_index + 1, col_index])
            draw_solution_panel(
                ax,
                truth,
                solution,
                known_pairs,
                f"{batch.family[case_index]} case: {name}",
                color,
            )

    fig.text(
        0.035,
        0.022,
        f"Training input has no index/sine/cosine anchor-order hint. Predicted missing pair base sigma={args.predicted_sigma:.2f} m with far-pair softening; known sigma={KNOWN_SIGMA_M:.2f} m.",
        ha="left",
        va="bottom",
        fontsize=8.3,
        color=TOKENS["muted"],
    )
    for ax in fig.axes:
        ax.tick_params(colors=TOKENS["muted"])
        for spine in ax.spines.values():
            spine.set_color(TOKENS["axis"])
    fig.savefig(path, bbox_inches="tight", dpi=180)
    plt.close(fig)


def short_method(method: str) -> str:
    return {
        "known_triangulated_lm": "known",
        "ml_completed_mds_full_lm_polish": "ML pure",
        "ml_completed_weak_polish": "ML weak",
        "oracle_full_mds_lm": "oracle",
    }[method]


def choose_examples(batch: GraphBatch) -> list[int]:
    grid = [index for index, family in enumerate(batch.family) if family == "grid"]
    random_cases = [index for index, family in enumerate(batch.family) if family == "random"]
    return [
        grid[len(grid) // 2] if grid else 0,
        random_cases[len(random_cases) // 2] if random_cases else min(1, len(batch.family) - 1),
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ML distance completion for anchor geometry.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--steps-per-epoch", type=int, default=140)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--hidden", type=int, default=144)
    parser.add_argument("--layers", type=int, default=5)
    parser.add_argument("--dropout", type=float, default=0.02)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=2.0)
    parser.add_argument("--random-fraction", type=float, default=0.5)
    parser.add_argument("--eval-cases", type=int, default=50)
    parser.add_argument("--solver-iterations", type=int, default=80)
    parser.add_argument("--polish-iterations", type=int, default=80)
    parser.add_argument("--predicted-sigma", type=float, default=0.55)
    parser.add_argument("--predicted-sigma-slope", type=float, default=0.65)
    parser.add_argument("--closest-predicted-pairs", type=int, default=0)
    parser.add_argument("--closest-predicted-pairs-per-anchor", type=float, default=0.0)
    parser.add_argument("--edm-weight", type=float, default=0.02)
    parser.add_argument("--radio-lower-weight", type=float, default=0.0)
    parser.add_argument("--weak-polish-iterations", type=int, default=80)
    parser.add_argument("--weak-completion-max-distance", type=float, default=18.0)
    parser.add_argument("--weak-completion-sigma-multiplier", type=float, default=2.5)
    parser.add_argument("--seed", type=int, default=20260626)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--checkpoint-every-epochs", type=int, default=0)
    parser.add_argument("--init-checkpoint", type=Path, default=None)
    parser.add_argument("--prefix", default="anchor_solver_ml_distance_completion")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    device = choose_device(args.device)
    print(f"device={device} torch={torch.__version__} cuda_available={torch.cuda.is_available()}", flush=True)
    if device.type == "cuda":
        print(f"cuda_device={torch.cuda.get_device_name(device)} cuda_version={torch.version.cuda}", flush=True)
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
            "known_sigma_m": KNOWN_SIGMA_M,
            "predicted_sigma_m": args.predicted_sigma,
            "predicted_sigma_slope": args.predicted_sigma_slope,
            "closest_predicted_pairs": args.closest_predicted_pairs,
            "closest_predicted_pairs_per_anchor": args.closest_predicted_pairs_per_anchor,
            "min_measured_degree": MIN_MEASURED_DEGREE,
            "edm_weight": args.edm_weight,
            "radio_lower_weight": args.radio_lower_weight,
            "diagnostic_graph_features": True,
            "no_anchor_order_features": True,
        },
        checkpoint_path,
    )
    write_rows(metrics_path, rows)
    write_history(history_path, history)
    make_infographic(figure_path, rows, history, eval_batch, pred, args)

    print(f"elapsed_s={elapsed:.1f}", flush=True)
    for row in summarize(rows):
        print(
            f"summary family={row['family']} method={row['method']} "
            f"missing_mae={row['median_missing_mae_m']:.3f}m "
            f"median_max={row['median_max_offset_m']:.3f}m "
            f"p90_max={row['p90_max_offset_m']:.3f}m "
            f"known_rmse={row['median_known_rmse_m']:.4f}m "
            f"under1m={row['under_1m']:.0%}",
            flush=True,
        )
    print(f"Wrote {checkpoint_path}", flush=True)
    print(f"Wrote {metrics_path}", flush=True)
    print(f"Wrote {history_path}", flush=True)
    print(f"Wrote {figure_path}", flush=True)


if __name__ == "__main__":
    main()





