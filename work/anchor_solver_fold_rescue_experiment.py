from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import math
from pathlib import Path
import random
import sys
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
sys.path.insert(0, str(ROOT / "work"))

import anchor_solver_ml_distance_completion as dc  # noqa: E402
import anchor_solver_p95_ml_cases as p95  # noqa: E402

from uwb_capture.anchor_geometry import (  # noqa: E402
    AnchorPairDistance,
    ProcessedAnchorPair,
    _LayoutPriors,
    _Parameterization,
    _anchor_ids,
    _initial_parameters,
    _layout_priors,
    _layout_scale,
    _local_minimize,
    _positions_to_params,
    _preprocess_pairs,
    pair_residuals,
    rotate_layout_to_level,
    solve_anchor_layout,
)

TOKENS = dc.TOKENS
BLUE = dc.BLUE
PINK = dc.PINK
GOLD = dc.GOLD
ORANGE = dc.ORANGE
OLIVE = dc.OLIVE
NEUTRAL = dc.NEUTRAL

BUCKET_KEYS = ("random", "grid", "office")
BUCKET_LABELS = {
    "random": "Random 16-32",
    "grid": "Grid >=16",
    "office": "Office >=16",
}


class _FixedFrameParameterization:
    def __init__(self, anchor_ids: list[str], fixed_positions: dict[str, tuple[float, float]]) -> None:
        self.anchor_ids = anchor_ids
        self.fixed_positions = dict(fixed_positions)
        self.variable_index: dict[tuple[str, str], int] = {}
        index = 0
        for anchor_id in anchor_ids:
            if anchor_id in self.fixed_positions:
                continue
            self.variable_index[(anchor_id, "x")] = index
            index += 1
            self.variable_index[(anchor_id, "y")] = index
            index += 1
        self.dimension = index

    def to_positions(self, params: list[float]) -> dict[str, tuple[float, float]]:
        positions: dict[str, tuple[float, float]] = {}
        for anchor_id in self.anchor_ids:
            if anchor_id in self.fixed_positions:
                positions[anchor_id] = self.fixed_positions[anchor_id]
                continue
            x_index = self.variable_index[(anchor_id, "x")]
            y_index = self.variable_index[(anchor_id, "y")]
            positions[anchor_id] = (params[x_index], params[y_index])
        return positions

    def derivative_index(self, anchor_id: str, axis: str) -> int | None:
        return self.variable_index.get((anchor_id, axis))


@dataclass(frozen=True)
class MethodResult:
    bucket: str
    case_index: int
    family: str
    shape: str
    method: str
    anchors: int
    known_pairs: int
    max_offset_m: float
    median_offset_m: float
    p95_offset_m: float
    known_rmse_m: float
    known_max_residual_m: float
    min_pair_distance_m: float
    close_pair_count: int
    fold_cluster_count: int
    kicked_anchor_count: int
    positions: dict[str, tuple[float, float]]
    truth: dict[str, tuple[float, float]]
    known_pair_list: list[AnchorPairDistance]


def center(positions: dict[str, tuple[float, float]]) -> tuple[float, float]:
    return (
        sum(x for x, _y in positions.values()) / max(len(positions), 1),
        sum(y for _x, y in positions.values()) / max(len(positions), 1),
    )


def pair_distance_values(positions: dict[str, tuple[float, float]]) -> list[tuple[float, str, str]]:
    rows: list[tuple[float, str, str]] = []
    ids = sorted(positions)
    for index, anchor_a in enumerate(ids):
        ax, ay = positions[anchor_a]
        for anchor_b in ids[index + 1 :]:
            bx, by = positions[anchor_b]
            rows.append((math.hypot(ax - bx, ay - by), anchor_a, anchor_b))
    return rows


def min_pair_distance(positions: dict[str, tuple[float, float]]) -> float:
    values = pair_distance_values(positions)
    return min((row[0] for row in values), default=math.inf)


def folded_clusters(positions: dict[str, tuple[float, float]], threshold_m: float) -> list[list[str]]:
    ids = sorted(positions)
    parent = {anchor_id: anchor_id for anchor_id in ids}

    def find(anchor_id: str) -> str:
        while parent[anchor_id] != anchor_id:
            parent[anchor_id] = parent[parent[anchor_id]]
            anchor_id = parent[anchor_id]
        return anchor_id

    def union(a: str, b: str) -> None:
        ra = find(a)
        rb = find(b)
        if ra != rb:
            parent[rb] = ra

    for distance, anchor_a, anchor_b in pair_distance_values(positions):
        if distance < threshold_m:
            union(anchor_a, anchor_b)
    groups: dict[str, list[str]] = {}
    for anchor_id in ids:
        groups.setdefault(find(anchor_id), []).append(anchor_id)
    return [group for group in groups.values() if len(group) > 1]


def close_pair_count(positions: dict[str, tuple[float, float]], threshold_m: float) -> int:
    return sum(1 for distance, _a, _b in pair_distance_values(positions) if distance < threshold_m)


def strong_spacing_priors(
    anchor_ids: list[str],
    processed: list[ProcessedAnchorPair],
    *,
    min_spacing_m: float,
    sigma_m: float,
) -> _LayoutPriors:
    measured_keys = frozenset(tuple(sorted((pair.anchor_a_id, pair.anchor_b_id))) for pair in processed)
    return _LayoutPriors(
        min_anchor_spacing_m=max(min_spacing_m, 0.0),
        spacing_weight=1.0 / max(sigma_m, 1e-6) ** 2,
        unmeasured_pair_min_distance_m=0.0,
        unmeasured_pair_weight=0.0,
        measured_pair_keys=measured_keys,
        boundary_radius_targets_m={},
        boundary_weight=0.0,
    )


def production_priors(anchor_ids: list[str], processed: list[ProcessedAnchorPair], scale: float) -> _LayoutPriors:
    return _layout_priors(
        anchor_ids,
        processed,
        scale=scale,
        min_anchor_spacing_m=None,
        anchor_spacing_sigma_m=0.35,
        unmeasured_pair_min_distance_m=None,
        unmeasured_pair_sigma_m=1.0,
        boundary_degree_prior_sigma_m=None,
    )


def score_positions(
    positions: dict[str, tuple[float, float]],
    known_pairs: list[AnchorPairDistance],
    *,
    fold_threshold_m: float,
) -> float:
    rmse, _max_residual = dc.pair_metrics(positions, known_pairs)
    close = close_pair_count(positions, fold_threshold_m)
    shortage = sum(max(0.0, fold_threshold_m - distance) ** 2 for distance, _a, _b in pair_distance_values(positions))
    return rmse + 0.35 * close + 0.20 * shortage


def topology_selection_score(
    positions: dict[str, tuple[float, float]],
    known_pairs: list[AnchorPairDistance],
    *,
    fold_threshold_m: float,
) -> float:
    score = score_positions(positions, known_pairs, fold_threshold_m=fold_threshold_m)
    anchor_ids = sorted(positions)
    if len(anchor_ids) < 3 or not known_pairs:
        return score

    measured_keys = {tuple(sorted((pair.anchor_a_id, pair.anchor_b_id))) for pair in known_pairs}
    max_measured = max(pair.distance_m for pair in known_pairs)
    unmeasured_min = max(2.0, max_measured * 0.90)
    unmeasured_shortage_sq: list[float] = []
    spacing_shortage_sq: list[float] = []
    for index, anchor_a in enumerate(anchor_ids):
        ax, ay = positions[anchor_a]
        for anchor_b in anchor_ids[index + 1 :]:
            bx, by = positions[anchor_b]
            distance = math.hypot(ax - bx, ay - by)
            spacing_shortage_sq.append(max(0.0, 2.0 - distance) ** 2)
            if tuple(sorted((anchor_a, anchor_b))) not in measured_keys:
                unmeasured_shortage_sq.append(max(0.0, unmeasured_min - distance) ** 2)

    if unmeasured_shortage_sq:
        score += 0.22 * math.sqrt(sum(unmeasured_shortage_sq) / len(unmeasured_shortage_sq))
    if spacing_shortage_sq:
        score += 0.35 * math.sqrt(sum(spacing_shortage_sq) / len(spacing_shortage_sq))

    degrees = {anchor_id: 0 for anchor_id in anchor_ids}
    for pair in known_pairs:
        degrees[pair.anchor_a_id] = degrees.get(pair.anchor_a_id, 0) + 1
        degrees[pair.anchor_b_id] = degrees.get(pair.anchor_b_id, 0) + 1
    min_degree = min(degrees.values())
    max_degree = max(degrees.values())
    if max_degree > min_degree:
        cx, cy = center(positions)
        radii = {anchor_id: math.hypot(positions[anchor_id][0] - cx, positions[anchor_id][1] - cy) for anchor_id in anchor_ids}
        inversion_shortage_sq: list[float] = []
        for anchor_a in anchor_ids:
            for anchor_b in anchor_ids:
                if degrees[anchor_a] >= degrees[anchor_b]:
                    continue
                # Lower-degree nodes are more likely to sit on edges/corners, so prefer them
                # not to be closer to the center than higher-degree nodes.
                inversion_shortage_sq.append(max(0.0, radii[anchor_b] - radii[anchor_a]) ** 2)
        if inversion_shortage_sq:
            score += 0.035 * math.sqrt(sum(inversion_shortage_sq) / len(inversion_shortage_sq))
    return score


def internal_solve_from_positions(
    positions: dict[str, tuple[float, float]],
    processed: list[ProcessedAnchorPair],
    parameterization: _Parameterization,
    *,
    priors: _LayoutPriors | None,
    iterations: int,
) -> dict[str, tuple[float, float]]:
    params = _positions_to_params(parameterization, positions)
    params, _energy = _local_minimize(params, parameterization, processed, priors, max_iterations=iterations)
    result = parameterization.to_positions(params)
    ids = parameterization.anchor_ids
    return rotate_layout_to_level(result, ids[0], ids[1])


def best_distance_only_solve(
    known_pairs: list[AnchorPairDistance],
    *,
    seed_count: int,
    iterations: int,
    rng: random.Random,
) -> dict[str, tuple[float, float]]:
    processed = _preprocess_pairs(known_pairs, min_sigma_m=0.02, min_distance_m=0.05)
    anchor_ids = _anchor_ids(processed)
    parameterization = _Parameterization(anchor_ids)
    scale = _layout_scale(processed)
    seeds = _initial_parameters(parameterization, processed, seed_count=max(seed_count, 1), scale=scale, rng=rng)
    best_positions: dict[str, tuple[float, float]] | None = None
    best_score = math.inf
    for seed in seeds:
        params, _energy = _local_minimize(seed, parameterization, processed, None, max_iterations=iterations)
        positions = rotate_layout_to_level(parameterization.to_positions(params), anchor_ids[0], anchor_ids[1])
        score = score_positions(positions, known_pairs, fold_threshold_m=1.55)
        if score < best_score:
            best_score = score
            best_positions = positions
    assert best_positions is not None
    return best_positions


def kick_folded_positions(
    positions: dict[str, tuple[float, float]],
    clusters: list[list[str]],
    *,
    amplification: float,
    jitter_m: float,
    min_radius_m: float,
    rng: random.Random,
) -> tuple[dict[str, tuple[float, float]], int]:
    if not clusters:
        return dict(positions), 0
    cx, cy = center(positions)
    result = dict(positions)
    kicked = 0
    for cluster_index, cluster in enumerate(clusters):
        ordered = sorted(cluster)
        if len(ordered) <= 1:
            continue
        pinned = rng.choice(ordered)
        movable = [anchor_id for anchor_id in ordered if anchor_id != pinned]
        for local_index, anchor_id in enumerate(movable):
            x, y = positions[anchor_id]
            vx = x - cx
            vy = y - cy
            radius = math.hypot(vx, vy)
            if radius < 1e-6:
                angle = 2.0 * math.pi * (cluster_index + local_index + rng.random()) / max(len(positions), 1)
                vx = math.cos(angle) * min_radius_m
                vy = math.sin(angle) * min_radius_m
                radius = min_radius_m
            ux = vx / radius
            uy = vy / radius
            tangent_x = -uy
            tangent_y = ux
            radial_scale = amplification * max(radius, min_radius_m) * rng.uniform(0.86, 1.18)
            tangent = rng.uniform(-jitter_m, jitter_m)
            radial_jitter = rng.uniform(-0.25 * jitter_m, 0.25 * jitter_m)
            result[anchor_id] = (
                cx - ux * (radial_scale + radial_jitter) + tangent_x * tangent,
                cy - uy * (radial_scale + radial_jitter) + tangent_y * tangent,
            )
            kicked += 1
    return result, kicked


def fixed_frame_solve_from_positions(
    positions: dict[str, tuple[float, float]],
    processed: list[ProcessedAnchorPair],
    anchor_ids: list[str],
    fixed_positions: dict[str, tuple[float, float]],
    *,
    priors: _LayoutPriors | None,
    iterations: int,
) -> dict[str, tuple[float, float]]:
    parameterization = _FixedFrameParameterization(anchor_ids, fixed_positions)
    params = _positions_to_params(parameterization, positions)
    params, _energy = _local_minimize(params, parameterization, processed, priors, max_iterations=iterations)
    return parameterization.to_positions(params)


def mirrored_from_center(
    positions: dict[str, tuple[float, float]],
    anchor_id: str,
    *,
    amplification: float,
    min_radius_m: float,
    jitter_m: float,
    rng: random.Random,
) -> tuple[float, float]:
    cx, cy = center(positions)
    x, y = positions[anchor_id]
    vx = x - cx
    vy = y - cy
    radius = math.hypot(vx, vy)
    if radius < 1e-6:
        angle = rng.uniform(-math.pi, math.pi)
        vx = math.cos(angle) * min_radius_m
        vy = math.sin(angle) * min_radius_m
        radius = min_radius_m
    ux = vx / radius
    uy = vy / radius
    tangent_x = -uy
    tangent_y = ux
    mirrored_radius = amplification * max(radius, min_radius_m) * rng.uniform(0.90, 1.18)
    tangent = rng.uniform(-jitter_m, jitter_m)
    radial_jitter = rng.uniform(-0.20 * jitter_m, 0.20 * jitter_m)
    return (
        cx - ux * (mirrored_radius + radial_jitter) + tangent_x * tangent,
        cy - uy * (mirrored_radius + radial_jitter) + tangent_y * tangent,
    )


def frame_anneal_seed(
    positions: dict[str, tuple[float, float]],
    clusters: list[list[str]],
    *,
    scale: float,
    fold_threshold_m: float,
    cycle: int,
    rng: random.Random,
) -> tuple[dict[str, tuple[float, float]], dict[str, tuple[float, float]], int]:
    seed_positions = dict(positions)
    fixed_positions: dict[str, tuple[float, float]] = {}
    moved = 0
    for cluster in clusters:
        ordered = sorted(cluster)
        if len(ordered) <= 1:
            continue
        survivor = rng.choice(ordered)
        peer_choices = [anchor_id for anchor_id in ordered if anchor_id != survivor]
        mirrored_peer = rng.choice(peer_choices)
        fixed_positions[survivor] = positions[survivor]
        mirrored_position = mirrored_from_center(
            positions,
            mirrored_peer,
            amplification=1.20 + 0.08 * cycle + rng.uniform(-0.06, 0.10),
            min_radius_m=max(scale * 0.78, fold_threshold_m * 1.55),
            jitter_m=max(scale * 0.22, 0.35),
            rng=rng,
        )
        fixed_positions[mirrored_peer] = mirrored_position
        seed_positions[mirrored_peer] = mirrored_position
        moved += 1
        for anchor_id in peer_choices:
            if anchor_id == mirrored_peer:
                continue
            seed_positions[anchor_id] = mirrored_from_center(
                positions,
                anchor_id,
                amplification=1.03 + 0.06 * cycle + rng.uniform(-0.05, 0.08),
                min_radius_m=max(scale * 0.62, fold_threshold_m * 1.25),
                jitter_m=max(scale * 0.30, 0.45),
                rng=rng,
            )
            moved += 1
    while len(fixed_positions) >= len(positions):
        fixed_positions.pop(rng.choice(sorted(fixed_positions)))
    return seed_positions, fixed_positions, moved

def fold_kick_solve(
    known_pairs: list[AnchorPairDistance],
    *,
    seed_count: int,
    iterations: int,
    cycles: int,
    trials_per_cycle: int,
    fold_threshold_m: float,
    rng_seed: int,
) -> tuple[dict[str, tuple[float, float]], int]:
    rng = random.Random(rng_seed)
    processed = _preprocess_pairs(known_pairs, min_sigma_m=0.02, min_distance_m=0.05)
    anchor_ids = _anchor_ids(processed)
    parameterization = _Parameterization(anchor_ids)
    scale = _layout_scale(processed)
    priors = production_priors(anchor_ids, processed, scale)
    positions = best_distance_only_solve(known_pairs, seed_count=seed_count, iterations=iterations, rng=rng)
    best_positions = positions
    best_clusters = folded_clusters(best_positions, fold_threshold_m)
    best_close_count = close_pair_count(best_positions, fold_threshold_m)
    best_score = (
        score_positions(best_positions, known_pairs, fold_threshold_m=fold_threshold_m)
        + 0.55 * best_close_count
        + 0.25 * len(best_clusters)
    )
    current_positions = best_positions
    total_kicked = 0
    attempt_budget = max(cycles, 1) * max(trials_per_cycle, 1)
    attempts_per_cycle = max(trials_per_cycle, 1)
    for attempt in range(attempt_budget):
        clusters = folded_clusters(current_positions, fold_threshold_m)
        if not clusters:
            return current_positions, total_kicked
        cycle = attempt // attempts_per_cycle
        kicked_seed, kicked = kick_folded_positions(
            current_positions,
            clusters,
            amplification=1.26 + 0.11 * cycle + rng.uniform(-0.08, 0.12),
            jitter_m=max(scale * 0.34, 0.55),
            min_radius_m=max(scale * 0.74, fold_threshold_m * 1.5),
            rng=rng,
        )
        rescued = internal_solve_from_positions(
            kicked_seed,
            processed,
            parameterization,
            priors=priors,
            iterations=iterations,
        )
        polished = internal_solve_from_positions(
            rescued,
            processed,
            parameterization,
            priors=None,
            iterations=max(iterations // 2, 20),
        )
        total_kicked += kicked
        candidate_clusters = folded_clusters(polished, fold_threshold_m)
        candidate_close_count = close_pair_count(polished, fold_threshold_m)
        candidate_score = (
            score_positions(polished, known_pairs, fold_threshold_m=fold_threshold_m)
            + 0.55 * candidate_close_count
            + 0.25 * len(candidate_clusters)
        )
        if not candidate_clusters:
            return polished, total_kicked
        if candidate_close_count < best_close_count or candidate_score < best_score:
            best_positions = polished
            best_score = candidate_score
            best_close_count = candidate_close_count
            current_positions = polished
        else:
            current_positions = best_positions
    return best_positions, total_kicked


def fold_kick_frame_anneal_solve(
    known_pairs: list[AnchorPairDistance],
    *,
    seed_count: int,
    iterations: int,
    cycles: int,
    trials_per_cycle: int,
    fold_threshold_m: float,
    rng_seed: int,
) -> tuple[dict[str, tuple[float, float]], int]:
    rng = random.Random(rng_seed)
    processed = _preprocess_pairs(known_pairs, min_sigma_m=0.02, min_distance_m=0.05)
    anchor_ids = _anchor_ids(processed)
    free_parameterization = _Parameterization(anchor_ids)
    scale = _layout_scale(processed)
    production = production_priors(anchor_ids, processed, scale)
    positions = best_distance_only_solve(known_pairs, seed_count=seed_count, iterations=iterations, rng=rng)
    best_positions = positions
    best_clusters = folded_clusters(best_positions, fold_threshold_m)
    best_close_count = close_pair_count(best_positions, fold_threshold_m)
    best_score = (
        score_positions(best_positions, known_pairs, fold_threshold_m=fold_threshold_m)
        + 0.60 * best_close_count
        + 0.30 * len(best_clusters)
    )
    current_positions = best_positions
    total_moved = 0
    attempt_budget = max(cycles, 1) * max(trials_per_cycle, 1)
    attempts_per_cycle = max(trials_per_cycle, 1)
    for attempt in range(attempt_budget):
        clusters = folded_clusters(current_positions, fold_threshold_m)
        if not clusters:
            return current_positions, total_moved
        cycle = attempt // attempts_per_cycle
        seed_positions, fixed_positions, moved = frame_anneal_seed(
            current_positions,
            clusters,
            scale=scale,
            fold_threshold_m=fold_threshold_m,
            cycle=cycle,
            rng=rng,
        )
        if not fixed_positions:
            continue
        framed_priors = strong_spacing_priors(
            anchor_ids,
            processed,
            min_spacing_m=max(fold_threshold_m * 1.18, min(scale * 0.55, 3.2)),
            sigma_m=0.12,
        )
        framed = fixed_frame_solve_from_positions(
            seed_positions,
            processed,
            anchor_ids,
            fixed_positions,
            priors=framed_priors,
            iterations=max(iterations // 2, 30),
        )
        released = internal_solve_from_positions(
            framed,
            processed,
            free_parameterization,
            priors=production,
            iterations=iterations,
        )
        polished = internal_solve_from_positions(
            released,
            processed,
            free_parameterization,
            priors=None,
            iterations=max(iterations // 2, 30),
        )
        total_moved += moved
        candidate_clusters = folded_clusters(polished, fold_threshold_m)
        candidate_close_count = close_pair_count(polished, fold_threshold_m)
        candidate_score = (
            score_positions(polished, known_pairs, fold_threshold_m=fold_threshold_m)
            + 0.60 * candidate_close_count
            + 0.30 * len(candidate_clusters)
        )
        if not candidate_clusters:
            return polished, total_moved
        if candidate_close_count < best_close_count or candidate_score < best_score:
            best_positions = polished
            best_score = candidate_score
            best_close_count = candidate_close_count
            current_positions = polished
        else:
            current_positions = best_positions
    return best_positions, total_moved

def repel_anneal_solve(
    known_pairs: list[AnchorPairDistance],
    *,
    seed_count: int,
    iterations: int,
    cycles: int,
    rng_seed: int,
    start_positions: dict[str, tuple[float, float]] | None = None,
) -> dict[str, tuple[float, float]]:
    rng = random.Random(rng_seed)
    processed = _preprocess_pairs(known_pairs, min_sigma_m=0.02, min_distance_m=0.05)
    anchor_ids = _anchor_ids(processed)
    parameterization = _Parameterization(anchor_ids)
    scale = _layout_scale(processed)
    positions = start_positions or best_distance_only_solve(known_pairs, seed_count=seed_count, iterations=iterations, rng=rng)
    min_spacing_base = max(2.0, min(scale * 0.42, 3.0))
    for cycle in range(max(cycles, 1)):
        schedule = [
            (min_spacing_base * 1.35, 0.075),
            (min_spacing_base * 1.18, 0.11),
            (min_spacing_base * 1.03, 0.17),
            (min_spacing_base * 0.94, 0.28),
        ]
        for spacing, sigma in schedule:
            priors = strong_spacing_priors(anchor_ids, processed, min_spacing_m=spacing, sigma_m=sigma)
            positions = internal_solve_from_positions(
                positions,
                processed,
                parameterization,
                priors=priors,
                iterations=max(iterations // 2, 30),
            )
        # Let real ranges pull the expanded shape back in, but keep a mild spacing guard.
        priors = strong_spacing_priors(anchor_ids, processed, min_spacing_m=min_spacing_base * 0.92, sigma_m=0.45)
        positions = internal_solve_from_positions(
            positions,
            processed,
            parameterization,
            priors=priors,
            iterations=iterations,
        )
    positions = internal_solve_from_positions(
        positions,
        processed,
        parameterization,
        priors=None,
        iterations=max(iterations // 2, 30),
    )
    return positions


def graph_shortest_scaffold_pairs(
    processed: list[ProcessedAnchorPair],
    anchor_ids: list[str],
    *,
    max_hops: int,
    relative_sigma: float,
    scaffold_weight: float = 1.0,
    hop_weight_base: float = 1.0,
) -> list[AnchorPairDistance]:
    n = len(anchor_ids)
    index = {anchor_id: offset for offset, anchor_id in enumerate(anchor_ids)}
    inf = 1e12
    shortest = [[inf for _ in range(n)] for _ in range(n)]
    hops = [[10**9 for _ in range(n)] for _ in range(n)]
    measured_keys = set()
    for i in range(n):
        shortest[i][i] = 0.0
        hops[i][i] = 0
    for pair in processed:
        i = index[pair.anchor_a_id]
        j = index[pair.anchor_b_id]
        measured_keys.add(tuple(sorted((pair.anchor_a_id, pair.anchor_b_id))))
        if pair.distance_m < shortest[i][j]:
            shortest[i][j] = shortest[j][i] = pair.distance_m
            hops[i][j] = hops[j][i] = 1
    for k in range(n):
        for i in range(n):
            via_distance = shortest[i][k]
            if via_distance >= inf:
                continue
            via_hops = hops[i][k]
            for j in range(n):
                candidate_distance = via_distance + shortest[k][j]
                candidate_hops = via_hops + hops[k][j]
                if candidate_distance < shortest[i][j] - 1e-9 or (
                    abs(candidate_distance - shortest[i][j]) <= 1e-9 and candidate_hops < hops[i][j]
                ):
                    shortest[i][j] = candidate_distance
                    hops[i][j] = candidate_hops
    scaffold: list[AnchorPairDistance] = []
    for i, anchor_a in enumerate(anchor_ids):
        for j in range(i + 1, n):
            anchor_b = anchor_ids[j]
            if tuple(sorted((anchor_a, anchor_b))) in measured_keys:
                continue
            hop_count = hops[i][j]
            if hop_count < 2 or hop_count > max_hops or shortest[i][j] >= inf:
                continue
            path_distance = shortest[i][j]
            base_sigma = max(0.35, path_distance * relative_sigma)
            spring_multiplier = max(float(scaffold_weight) * (float(hop_weight_base) ** float(hop_count)), 1e-6)
            sigma = base_sigma / math.sqrt(spring_multiplier)
            scaffold.append(
                AnchorPairDistance(
                    anchor_a,
                    anchor_b,
                    path_distance,
                    sigma_m=sigma,
                    enabled=True,
                    source=f"graph-shortest-h{hop_count}-w{spring_multiplier:.3g}",
                )
            )
    return scaffold


def graph_shortest_scaffold_solve(
    known_pairs: list[AnchorPairDistance],
    *,
    seed_count: int,
    iterations: int,
    rng_seed: int,
    max_hops: int = 3,
    relative_sigma: float = 0.30,
    scaffold_weight: float = 1.0,
    hop_weight_base: float = 1.0,
) -> dict[str, tuple[float, float]]:
    rng = random.Random(rng_seed)
    processed = _preprocess_pairs(known_pairs, min_sigma_m=0.02, min_distance_m=0.05)
    anchor_ids = _anchor_ids(processed)
    scaffold = graph_shortest_scaffold_pairs(
        processed,
        anchor_ids,
        max_hops=max_hops,
        relative_sigma=relative_sigma,
        scaffold_weight=scaffold_weight,
        hop_weight_base=hop_weight_base,
    )
    if not scaffold:
        return best_distance_only_solve(known_pairs, seed_count=seed_count, iterations=iterations, rng=rng)
    augmented = _preprocess_pairs([*known_pairs, *scaffold], min_sigma_m=0.02, min_distance_m=0.05)
    parameterization = _Parameterization(anchor_ids)
    scale = _layout_scale(augmented)
    seeds = _initial_parameters(parameterization, augmented, seed_count=max(seed_count, 1), scale=scale, rng=rng)
    try:
        mds_positions = dc.classical_mds_seed([*known_pairs, *scaffold])
        seeds.insert(0, _positions_to_params(parameterization, mds_positions))
    except Exception:
        pass
    known_priors = production_priors(anchor_ids, processed, _layout_scale(processed))
    scaffold_priors = production_priors(anchor_ids, processed, scale)
    best_positions: dict[str, tuple[float, float]] | None = None
    best_score = math.inf
    for seed in seeds:
        scaffold_params, _energy = _local_minimize(
            seed,
            parameterization,
            augmented,
            scaffold_priors,
            max_iterations=max(iterations // 2, 30),
        )
        scaffold_positions = parameterization.to_positions(scaffold_params)
        prior_positions = internal_solve_from_positions(
            scaffold_positions,
            processed,
            parameterization,
            priors=known_priors,
            iterations=max(iterations, 30),
        )
        polished = internal_solve_from_positions(
            prior_positions,
            processed,
            parameterization,
            priors=None,
            iterations=max(iterations // 2, 30),
        )
        for candidate_positions in (prior_positions, polished):
            score = topology_selection_score(candidate_positions, known_pairs, fold_threshold_m=1.65)
            if score < best_score:
                best_score = score
                best_positions = candidate_positions
    assert best_positions is not None
    return best_positions

def production_solve(known_pairs: list[AnchorPairDistance], *, seed_count: int, iterations: int, rng_seed: int) -> dict[str, tuple[float, float]]:
    result = solve_anchor_layout(
        known_pairs,
        seed_count=seed_count,
        basin_hops=8,
        max_iterations=iterations,
        random_seed=rng_seed,
        distance_polish_iterations=max(iterations // 2, 30),
    )
    return result.positions_m


def metrics_row(
    bucket: str,
    case_index: int,
    family: str,
    shape: str,
    method: str,
    truth: dict[str, tuple[float, float]],
    positions: dict[str, tuple[float, float]],
    known_pairs: list[AnchorPairDistance],
    *,
    fold_threshold_m: float,
    kicked: int = 0,
) -> MethodResult:
    max_offset, median_offset, p95_offset = dc.offset_summary(truth, positions)
    known_rmse, known_max = dc.pair_metrics(positions, known_pairs)
    clusters = folded_clusters(positions, fold_threshold_m)
    return MethodResult(
        bucket=bucket,
        case_index=case_index,
        family=family,
        shape=shape,
        method=method,
        anchors=len(truth),
        known_pairs=len(known_pairs),
        max_offset_m=max_offset,
        median_offset_m=median_offset,
        p95_offset_m=p95_offset,
        known_rmse_m=known_rmse,
        known_max_residual_m=known_max,
        min_pair_distance_m=min_pair_distance(positions),
        close_pair_count=close_pair_count(positions, fold_threshold_m),
        fold_cluster_count=len(clusters),
        kicked_anchor_count=kicked,
        positions=positions,
        truth=truth,
        known_pair_list=known_pairs,
    )


def graph_batch_from_case(case: p95.CaseSpec, device: torch.device) -> dc.GraphBatch:
    return p95.graph_batch_from_cases([case], device)


def run_case(
    case: p95.CaseSpec,
    *,
    case_index: int,
    device: torch.device,
    seed_count: int,
    iterations: int,
    fold_threshold_m: float,
    rng_seed: int,
) -> list[MethodResult]:
    batch = graph_batch_from_case(case, device)
    truth = dc.graph_to_truth(batch, 0)
    known_pairs = dc.known_pairs_from_batch(batch, 0)
    rows: list[MethodResult] = []
    distance_only = best_distance_only_solve(known_pairs, seed_count=seed_count, iterations=iterations, rng=random.Random(rng_seed + 11))
    production = production_solve(known_pairs, seed_count=seed_count, iterations=iterations, rng_seed=rng_seed + 17)
    graph_scaffold = graph_shortest_scaffold_solve(
        known_pairs,
        seed_count=seed_count,
        iterations=iterations,
        rng_seed=rng_seed + 19,
    )
    fold_kick, kicked = fold_kick_solve(
        known_pairs,
        seed_count=seed_count,
        iterations=iterations,
        cycles=4,
        trials_per_cycle=8,
        fold_threshold_m=fold_threshold_m,
        rng_seed=rng_seed + 23,
    )
    repel = repel_anneal_solve(
        known_pairs,
        seed_count=seed_count,
        iterations=iterations,
        cycles=3,
        rng_seed=rng_seed + 31,
        start_positions=distance_only,
    )
    frame_anneal, frame_moved = fold_kick_frame_anneal_solve(
        known_pairs,
        seed_count=seed_count,
        iterations=iterations,
        cycles=4,
        trials_per_cycle=8,
        fold_threshold_m=fold_threshold_m,
        rng_seed=rng_seed + 37,
    )
    kicked_seed, kicked2 = fold_kick_solve(
        known_pairs,
        seed_count=seed_count,
        iterations=iterations,
        cycles=2,
        trials_per_cycle=6,
        fold_threshold_m=fold_threshold_m,
        rng_seed=rng_seed + 41,
    )
    fold_repel = repel_anneal_solve(
        known_pairs,
        seed_count=seed_count,
        iterations=iterations,
        cycles=2,
        rng_seed=rng_seed + 43,
        start_positions=kicked_seed,
    )
    methods = [
        ("distance-only", distance_only, 0),
        ("production-priors", production, 0),
        ("graph-shortest-scaffold", graph_scaffold, 0),
        ("fold-kick", fold_kick, kicked),
        ("fold-kick-frame-anneal", frame_anneal, frame_moved),
        ("repel-anneal", repel, 0),
        ("fold-kick+repel", fold_repel, kicked2),
    ]
    for method, positions, kicked_count in methods:
        rows.append(
            metrics_row(
                case.bucket,
                case_index,
                case.family,
                case.shape,
                method,
                truth,
                positions,
                known_pairs,
                fold_threshold_m=fold_threshold_m,
                kicked=kicked_count,
            )
        )
    return rows


def result_to_dict(row: MethodResult) -> dict[str, str | int | float]:
    return {
        "bucket": row.bucket,
        "case_index": row.case_index,
        "family": row.family,
        "shape": row.shape,
        "method": row.method,
        "anchors": row.anchors,
        "known_pairs": row.known_pairs,
        "max_offset_m": row.max_offset_m,
        "median_offset_m": row.median_offset_m,
        "p95_offset_m": row.p95_offset_m,
        "known_rmse_m": row.known_rmse_m,
        "known_max_residual_m": row.known_max_residual_m,
        "min_pair_distance_m": row.min_pair_distance_m,
        "close_pair_count": row.close_pair_count,
        "fold_cluster_count": row.fold_cluster_count,
        "kicked_anchor_count": row.kicked_anchor_count,
    }


def write_csv(path: Path, rows: list[dict[str, str | int | float]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[MethodResult]) -> list[dict[str, str | int | float]]:
    summary: list[dict[str, str | int | float]] = []
    keys = sorted({(row.bucket, row.method) for row in rows})
    for bucket, method in keys:
        part = [row for row in rows if row.bucket == bucket and row.method == method]
        offsets = np.array([row.max_offset_m for row in part], dtype=float)
        rmse = np.array([row.known_rmse_m for row in part], dtype=float)
        min_dist = np.array([row.min_pair_distance_m for row in part], dtype=float)
        close_counts = np.array([row.close_pair_count for row in part], dtype=float)
        summary.append(
            {
                "bucket": bucket,
                "method": method,
                "cases": len(part),
                "median_max_offset_m": float(np.median(offsets)),
                "p90_max_offset_m": float(np.quantile(offsets, 0.90)),
                "p95_max_offset_m": float(np.quantile(offsets, 0.95)),
                "under_20cm": float(np.mean(offsets <= 0.20)),
                "under_50cm": float(np.mean(offsets <= 0.50)),
                "under_1m": float(np.mean(offsets <= 1.00)),
                "median_known_rmse_m": float(np.median(rmse)),
                "median_min_pair_distance_m": float(np.median(min_dist)),
                "folded_case_rate": float(np.mean(close_counts > 0)),
                "mean_close_pair_count": float(np.mean(close_counts)),
            }
        )
    return summary


def aligned(truth: dict[str, tuple[float, float]], positions: dict[str, tuple[float, float]]) -> dict[str, tuple[float, float]]:
    return dc.aligned_estimate(truth, positions)


def draw_layout(ax, row: MethodResult, *, color: str) -> None:
    truth = row.truth
    estimate = aligned(truth, row.positions)
    ax.set_facecolor(TOKENS["panel"])
    for pair in row.known_pair_list:
        ax.plot(
            [truth[pair.anchor_a_id][0], truth[pair.anchor_b_id][0]],
            [truth[pair.anchor_a_id][1], truth[pair.anchor_b_id][1]],
            color=NEUTRAL["light"],
            linewidth=0.45,
            alpha=0.42,
            zorder=1,
        )
    for anchor_id, true_point in truth.items():
        solved = estimate[anchor_id]
        ax.plot([true_point[0], solved[0]], [true_point[1], solved[1]], color=ORANGE["dark"], linewidth=0.75, alpha=0.45, zorder=2)
    ax.scatter([x for x, _ in truth.values()], [y for _, y in truth.values()], s=18, facecolors=TOKENS["panel"], edgecolors=NEUTRAL["dark"], linewidths=0.7, zorder=3)
    ax.scatter([x for x, _ in estimate.values()], [y for _, y in estimate.values()], s=24, color=color, edgecolors=TOKENS["ink"], linewidths=0.55, zorder=4)
    xs = [x for x, _ in truth.values()] + [x for x, _ in estimate.values()]
    ys = [y for _, y in truth.values()] + [y for _, y in estimate.values()]
    span = max(max(xs) - min(xs), max(ys) - min(ys), 1.0)
    pad = span * 0.13
    ax.set_xlim(min(xs) - pad, max(xs) + pad)
    ax.set_ylim(min(ys) - pad, max(ys) + pad)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, color=TOKENS["grid"], linewidth=0.5)
    ax.tick_params(labelsize=6.5, colors=TOKENS["muted"], length=0)
    for spine in ax.spines.values():
        spine.set_color(TOKENS["axis"])
    ax.set_title(
        f"{row.method}\nmax {row.max_offset_m:.2f}m | RMSE {row.known_rmse_m:.3f}m | close pairs {row.close_pair_count}",
        loc="left",
        fontsize=8.6,
        fontweight="semibold",
        color=TOKENS["ink"],
    )


def choose_example(rows: list[MethodResult], bucket: str) -> list[MethodResult]:
    production_by_case = {row.case_index: row for row in rows if row.bucket == bucket and row.method == "production-priors"}
    rescue_methods = {"graph-shortest-scaffold", "fold-kick", "fold-kick-frame-anneal", "repel-anneal", "fold-kick+repel"}
    best_case = None
    best_gain = -math.inf
    for case_index, base in production_by_case.items():
        rescues = [row for row in rows if row.bucket == bucket and row.case_index == case_index and row.method in rescue_methods]
        if not rescues:
            continue
        best_rescue = min(rescues, key=lambda row: row.max_offset_m)
        gain = base.max_offset_m - best_rescue.max_offset_m
        if gain > best_gain:
            best_gain = gain
            best_case = case_index
    if best_case is None:
        best_case = next(iter(production_by_case))
    ordered_methods = ["distance-only", "production-priors", "graph-shortest-scaffold", "fold-kick", "fold-kick-frame-anneal", "repel-anneal", "fold-kick+repel"]
    return [next(row for row in rows if row.bucket == bucket and row.case_index == best_case and row.method == method) for method in ordered_methods]


def make_figure(path: Path, rows: list[MethodResult], summary: list[dict[str, str | int | float]], *, cases_per_bucket: int) -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": TOKENS["surface"],
            "savefig.facecolor": TOKENS["surface"],
            "font.family": ["Segoe UI", "DejaVu Sans", "Arial", "sans-serif"],
        }
    )
    methods = ["distance-only", "production-priors", "graph-shortest-scaffold", "fold-kick", "fold-kick-frame-anneal", "repel-anneal", "fold-kick+repel"]
    colors = {
        "distance-only": NEUTRAL["base"],
        "production-priors": BLUE["base"],
        "graph-shortest-scaffold": "#2C9C8F",
        "fold-kick": PINK["base"],
        "fold-kick-frame-anneal": "#8C6FD1",
        "repel-anneal": GOLD["base"],
        "fold-kick+repel": OLIVE["base"],
    }
    bucket_order = [BUCKET_LABELS[key] for key in BUCKET_KEYS if any(row.bucket == BUCKET_LABELS[key] for row in rows)]
    fig = plt.figure(figsize=(20.2, 15.6), dpi=170)
    gs = fig.add_gridspec(4, 7, height_ratios=[0.90, 1.15, 1.15, 1.15], hspace=0.44, wspace=0.18)
    fig.text(0.035, 0.985, "Fold rescue solver experiment", ha="left", va="top", fontsize=20, fontweight="bold", color=TOKENS["ink"])
    fig.text(
        0.035,
        0.958,
        (
            f"{cases_per_bucket} fair cases per bucket. Fold-kick detects solved anchors closer than the fold threshold and kicks all collapsed anchors outward/opposite the center except one pinned anchor; "
            "repel-anneal repeatedly solves with a strong close-anchor repulsion, then relaxes back to range fitting."
        ),
        ha="left",
        va="top",
        fontsize=9.0,
        color=TOKENS["muted"],
    )
    metric_specs = [("p95_max_offset_m", "p95 max offset (m)"), ("under_1m", "share under 1 m"), ("folded_case_rate", "folded solved cases")]
    for col, (metric, ylabel) in enumerate(metric_specs):
        ax = fig.add_subplot(gs[0, col])
        ax.set_facecolor(TOKENS["panel"])
        labels: list[str] = []
        values: list[float] = []
        bar_colors: list[str] = []
        for bucket in bucket_order:
            for method in methods:
                item = next((row for row in summary if row["bucket"] == bucket and row["method"] == method), None)
                if item is None:
                    continue
                labels.append(f"{bucket.split()[0]}\n{method}")
                values.append(float(item[metric]))
                bar_colors.append(colors[method])
        x = np.arange(len(values))
        ax.bar(x, values, color=bar_colors, edgecolor=TOKENS["ink"], linewidth=0.35)
        ax.set_title(ylabel, loc="left", fontsize=10.2, fontweight="bold", color=TOKENS["ink"])
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=58, ha="right", fontsize=5.8)
        ax.tick_params(axis="y", labelsize=7, colors=TOKENS["muted"], length=0)
        ax.grid(True, axis="y", color=TOKENS["grid"], linewidth=0.5)
        for spine in ax.spines.values():
            spine.set_color(TOKENS["axis"])
    ax_note = fig.add_subplot(gs[0, 3:])
    ax_note.axis("off")
    best_lines = []
    for bucket in bucket_order:
        part = [row for row in summary if row["bucket"] == bucket]
        best = min(part, key=lambda row: float(row["p95_max_offset_m"]))
        best_lines.append(f"{bucket}: best p95={float(best['p95_max_offset_m']):.2f}m via {best['method']}")
    ax_note.text(0.0, 0.92, "Best p95 by bucket", fontsize=11, fontweight="bold", color=TOKENS["ink"], ha="left", va="top")
    ax_note.text(0.0, 0.72, "\n".join(best_lines), fontsize=9.0, color=TOKENS["muted"], ha="left", va="top", linespacing=1.55)

    for row_index, bucket in enumerate(bucket_order, start=1):
        example_rows = choose_example(rows, bucket)
        for col, example in enumerate(example_rows):
            ax = fig.add_subplot(gs[row_index, col])
            draw_layout(ax, example, color=colors[example.method])
            if col == 0:
                ax.text(-0.12, 0.5, bucket, transform=ax.transAxes, rotation=90, ha="center", va="center", fontsize=10.5, fontweight="bold", color=TOKENS["ink"])
    fig.subplots_adjust(left=0.055, right=0.985, top=0.91, bottom=0.055)
    fig.savefig(path, bbox_inches="tight", dpi=170)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Try fold-kick and repel-anneal anchor solver variants.")
    parser.add_argument("--cases-per-bucket", type=int, default=12)
    parser.add_argument("--seed", type=int, default=2026062705)
    parser.add_argument("--seed-count", type=int, default=14)
    parser.add_argument("--iterations", type=int, default=80)
    parser.add_argument("--buckets", default="random,grid,office", help="Comma-separated subset: random,grid,office")
    parser.add_argument("--fold-threshold", type=float, default=1.65)
    parser.add_argument("--prefix", default="anchor_solver_fold_rescue_experiment")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed % (2**32 - 1))
    torch.manual_seed(args.seed)
    device = torch.device("cpu")
    requested_buckets = tuple(part.strip().lower() for part in args.buckets.split(",") if part.strip())
    unknown_buckets = [bucket for bucket in requested_buckets if bucket not in BUCKET_KEYS]
    if unknown_buckets:
        raise ValueError(f"Unknown buckets: {unknown_buckets}")
    all_rows: list[MethodResult] = []
    case_counter = 0
    for bucket_key in requested_buckets:
        print(f"generating bucket={bucket_key} cases={args.cases_per_bucket}", flush=True)
        cases = p95.generate_cases(bucket_key, args.cases_per_bucket, device)
        for local_index, case in enumerate(cases):
            print(f"solving bucket={bucket_key} case={local_index + 1}/{len(cases)} anchors={case.points.shape[0]}", flush=True)
            rows = run_case(
                case,
                case_index=case_counter,
                device=device,
                seed_count=args.seed_count,
                iterations=args.iterations,
                fold_threshold_m=args.fold_threshold,
                rng_seed=args.seed + case_counter * 1009,
            )
            all_rows.extend(rows)
            case_counter += 1
    detail = [result_to_dict(row) for row in all_rows]
    summary = summarize(all_rows)
    detail_path = OUTPUTS / f"{args.prefix}_detail.csv"
    summary_path = OUTPUTS / f"{args.prefix}_summary.csv"
    figure_path = OUTPUTS / f"{args.prefix}.png"
    write_csv(detail_path, detail)
    write_csv(summary_path, summary)
    make_figure(figure_path, all_rows, summary, cases_per_bucket=args.cases_per_bucket)
    for row in summary:
        print(
            f"summary bucket={row['bucket']} method={row['method']} "
            f"p95={float(row['p95_max_offset_m']):.3f}m median={float(row['median_max_offset_m']):.3f}m "
            f"under1m={float(row['under_1m']):.3f} folded_rate={float(row['folded_case_rate']):.3f} "
            f"rmse={float(row['median_known_rmse_m']):.4f}m",
            flush=True,
        )
    print(f"Wrote {detail_path}", flush=True)
    print(f"Wrote {summary_path}", flush=True)
    print(f"Wrote {figure_path}", flush=True)


if __name__ == "__main__":
    main()












