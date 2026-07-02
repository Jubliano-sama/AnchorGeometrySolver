"""Anchor-to-anchor spring layout solver.

The clicker survey provides distances between anchors, not absolute anchor
coordinates. This module treats those distances as springs and finds the 2D
anchor layout with the lowest spring energy. The solver is dependency-free so
the GUI can run on the same lightweight install as the capture tools.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
from statistics import median
from typing import Iterable

try:
    import numpy as _np
except Exception:  # pragma: no cover - GUI can still run without NumPy.
    _np = None


ANCHOR_LAYOUT_ALGORITHM = "Spring energy basin hopping (multi-seed LM)"


@dataclass(frozen=True)
class AnchorPairDistance:
    """One enabled anchor-to-anchor distance measurement."""

    anchor_a_id: str
    anchor_b_id: str
    distance_m: float
    sigma_m: float = 0.05
    enabled: bool = True
    source: str = "survey"


@dataclass(frozen=True)
class ProcessedAnchorPair:
    """Validated and de-duplicated spring constraint."""

    anchor_a_id: str
    anchor_b_id: str
    distance_m: float
    sigma_m: float
    weight: float
    source: str


@dataclass(frozen=True)
class AnchorLayoutResult:
    """Solved anchor layout and residual diagnostics."""

    algorithm: str
    energy: float
    rmse_m: float
    max_residual_m: float
    positions_m: dict[str, tuple[float, float]]
    processed_pairs: tuple[ProcessedAnchorPair, ...]
    residuals_m: dict[str, float]
    warnings: tuple[str, ...]
    seed_count: int
    basin_hop_count: int


@dataclass(frozen=True)
class AnchorGraphDiagnostics:
    """Graph-only ambiguity diagnostics for sparse anchor range surveys."""

    anchor_count: int
    pair_count: int
    complete_pair_count: int
    missing_pair_count: int
    required_rigidity_rank: int
    rigidity_rank: int
    is_locally_rigid: bool
    is_redundantly_rigid: bool
    is_generically_globally_rigid_2d: bool
    articulation_points: tuple[str, ...]
    two_vertex_cuts: tuple[tuple[str, str], ...]
    bridge_edges: tuple[tuple[str, str], ...]
    near_automorphism_orbits: tuple[tuple[str, ...], ...]
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class _LayoutPriors:
    min_anchor_spacing_m: float
    spacing_weight: float
    unmeasured_pair_min_distance_m: float
    unmeasured_pair_weight: float
    measured_pair_keys: frozenset[tuple[str, str]]
    boundary_radius_targets_m: dict[str, float]
    boundary_weight: float


def solve_anchor_layout(
    pairs: Iterable[AnchorPairDistance],
    *,
    seed_count: int = 24,
    basin_hops: int = 10,
    max_iterations: int = 80,
    random_seed: int = 1337,
    min_sigma_m: float = 0.02,
    min_distance_m: float = 0.05,
    min_anchor_spacing_m: float | None = None,
    anchor_spacing_sigma_m: float = 0.35,
    unmeasured_pair_min_distance_m: float | None = None,
    unmeasured_pair_sigma_m: float = 1.0,
    boundary_degree_prior_sigma_m: float | None = None,
    distance_polish_iterations: int | None = None,
) -> AnchorLayoutResult:
    """Solve anchor coordinates from pair distances.

    Distances only constrain shape, so the result is arbitrary up to
    translation, rotation, and mirror reflection. The returned layout is
    canonicalized with the first two sorted anchor IDs on the same Y coordinate.

    The default search objective also uses soft survey priors: anchors resist
    being closer than the inferred minimum spacing, missing anchor-to-anchor
    readings resist becoming near-neighbor distances, and lower-degree anchors
    are pushed away from the layout center. After the prior-guided search, a
    distance-only polish step refits the selected shape for reported RMSE. Set
    the corresponding sigma/min-distance argument to 0.0 to disable a prior.
    """

    processed = _preprocess_pairs(
        pairs,
        min_sigma_m=min_sigma_m,
        min_distance_m=min_distance_m,
    )
    anchor_ids = _anchor_ids(processed)
    _validate_connected(anchor_ids, processed)

    scale = _layout_scale(processed)
    parameterization = _Parameterization(anchor_ids)
    priors = _layout_priors(
        anchor_ids,
        processed,
        scale=scale,
        min_anchor_spacing_m=min_anchor_spacing_m,
        anchor_spacing_sigma_m=anchor_spacing_sigma_m,
        unmeasured_pair_min_distance_m=unmeasured_pair_min_distance_m,
        unmeasured_pair_sigma_m=unmeasured_pair_sigma_m,
        boundary_degree_prior_sigma_m=boundary_degree_prior_sigma_m,
    )
    rng = random.Random(random_seed)
    initial_seeds = _initial_parameters(
        parameterization,
        processed,
        seed_count=max(seed_count, 1),
        scale=scale,
        rng=rng,
    )

    best_params: list[float] | None = None
    best_energy = math.inf
    accepted_hops = 0
    temperature = max(scale * scale * 1e-5, 1e-8)
    hop_scale = max(scale * 0.35, 0.05)

    for seed_params in initial_seeds:
        current_params, current_energy = _local_minimize(
            seed_params,
            parameterization,
            processed,
            priors,
            max_iterations=max_iterations,
        )
        if current_energy < best_energy:
            best_params = current_params
            best_energy = current_energy

        for _hop_index in range(max(basin_hops, 0)):
            hopped = [
                value + rng.gauss(0.0, hop_scale)
                for value in current_params
            ]
            candidate_params, candidate_energy = _local_minimize(
                hopped,
                parameterization,
                processed,
                priors,
                max_iterations=max_iterations,
            )
            accept = candidate_energy <= current_energy
            if not accept:
                probability = math.exp(
                    max(min((current_energy - candidate_energy) / temperature, 0.0), -60.0)
                )
                accept = rng.random() < probability
            if accept:
                accepted_hops += 1
                current_params = candidate_params
                current_energy = candidate_energy
            if candidate_energy < best_energy:
                best_params = candidate_params
                best_energy = candidate_energy

    if best_params is None:
        raise ValueError("Could not solve anchor layout.")

    polish_iterations = (
        max_iterations
        if distance_polish_iterations is None
        else max(distance_polish_iterations, 0)
    )
    if polish_iterations > 0:
        best_params, best_energy = _local_minimize(
            best_params,
            parameterization,
            processed,
            None,
            max_iterations=polish_iterations,
        )

    positions = parameterization.to_positions(best_params)
    positions = rotate_layout_to_level(positions, anchor_ids[0], anchor_ids[1])
    residuals = pair_residuals(positions, processed)
    rmse = _rmse(residuals.values())
    max_residual = max((abs(value) for value in residuals.values()), default=0.0)
    warnings = _layout_warnings(anchor_ids, processed, rmse, max_residual)
    return AnchorLayoutResult(
        algorithm=ANCHOR_LAYOUT_ALGORITHM,
        energy=best_energy,
        rmse_m=rmse,
        max_residual_m=max_residual,
        positions_m=positions,
        processed_pairs=tuple(processed),
        residuals_m=residuals,
        warnings=tuple(warnings),
        seed_count=len(initial_seeds),
        basin_hop_count=accepted_hops,
    )


def diagnose_anchor_graph(
    pairs: Iterable[AnchorPairDistance | ProcessedAnchorPair],
    *,
    min_sigma_m: float = 0.02,
    min_distance_m: float = 0.05,
    length_bin_m: float = 0.25,
) -> AnchorGraphDiagnostics:
    """Return graph-only rigidity and ambiguity diagnostics.

    These checks do not use solved coordinates. They are intended to flag cases
    where a low residual layout may still be non-unique, flippable, or label
    ambiguous because the measured range graph is sparse or symmetric.
    """

    collected = list(pairs)
    if collected and isinstance(collected[0], ProcessedAnchorPair):
        processed = [pair for pair in collected if isinstance(pair, ProcessedAnchorPair)]
    else:
        processed = _preprocess_pairs(
            collected,
            min_sigma_m=min_sigma_m,
            min_distance_m=min_distance_m,
        )
    anchor_ids = _anchor_ids(processed)
    return _graph_diagnostics_from_processed(
        anchor_ids,
        processed,
        length_bin_m=length_bin_m,
    )



def pair_residuals(
    positions_m: dict[str, tuple[float, float]],
    pairs: Iterable[ProcessedAnchorPair | AnchorPairDistance],
) -> dict[str, float]:
    """Return signed measured-minus-model residuals for each pair."""

    residuals: dict[str, float] = {}
    for pair in pairs:
        if pair.anchor_a_id not in positions_m or pair.anchor_b_id not in positions_m:
            continue
        ax, ay = positions_m[pair.anchor_a_id]
        bx, by = positions_m[pair.anchor_b_id]
        model_distance = math.hypot(ax - bx, ay - by)
        residuals[_pair_label(pair.anchor_a_id, pair.anchor_b_id)] = (
            model_distance - float(pair.distance_m)
        )
    return residuals


def rotate_layout_to_level(
    positions_m: dict[str, tuple[float, float]],
    anchor_a_id: str,
    anchor_b_id: str,
) -> dict[str, tuple[float, float]]:
    """Rotate and translate a layout so two anchors lie on the same Y.

    The first anchor becomes the origin. If the second anchor ends up to the
    left of it, the layout is rotated another 180 degrees so the pair reads as a
    left-to-right baseline.
    """

    if anchor_a_id not in positions_m or anchor_b_id not in positions_m:
        raise ValueError("Both selected anchors must exist in the layout.")
    ax, ay = positions_m[anchor_a_id]
    bx, by = positions_m[anchor_b_id]
    dx = bx - ax
    dy = by - ay
    if math.hypot(dx, dy) <= 1e-12:
        raise ValueError("Selected anchors are at the same position.")

    angle = -math.atan2(dy, dx)
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    rotated: dict[str, tuple[float, float]] = {}
    for anchor_id, (x_m, y_m) in positions_m.items():
        tx = x_m - ax
        ty = y_m - ay
        rotated[anchor_id] = (
            tx * cos_a - ty * sin_a,
            tx * sin_a + ty * cos_a,
        )
    if rotated[anchor_b_id][0] < 0:
        rotated = rotate_layout(rotated, 180.0, origin=(0.0, 0.0))
    return _clean_positions(rotated)


def rotate_layout(
    positions_m: dict[str, tuple[float, float]],
    angle_degrees: float,
    *,
    origin: tuple[float, float] | None = None,
) -> dict[str, tuple[float, float]]:
    """Rotate a layout around ``origin`` or its center."""

    ox, oy = origin if origin is not None else _layout_center(positions_m)
    radians = math.radians(angle_degrees)
    cos_a = math.cos(radians)
    sin_a = math.sin(radians)
    rotated = {}
    for anchor_id, (x_m, y_m) in positions_m.items():
        tx = x_m - ox
        ty = y_m - oy
        rotated[anchor_id] = (
            ox + tx * cos_a - ty * sin_a,
            oy + tx * sin_a + ty * cos_a,
        )
    return _clean_positions(rotated)


def mirror_layout(
    positions_m: dict[str, tuple[float, float]],
    axis: str,
    *,
    origin: tuple[float, float] | None = None,
) -> dict[str, tuple[float, float]]:
    """Mirror a layout across its center X or Y axis."""

    normalized_axis = axis.lower().strip()
    if normalized_axis not in {"x", "y"}:
        raise ValueError("Mirror axis must be 'x' or 'y'.")
    ox, oy = origin if origin is not None else _layout_center(positions_m)
    mirrored = {}
    for anchor_id, (x_m, y_m) in positions_m.items():
        if normalized_axis == "x":
            mirrored[anchor_id] = (2.0 * ox - x_m, y_m)
        else:
            mirrored[anchor_id] = (x_m, 2.0 * oy - y_m)
    return _clean_positions(mirrored)


def _preprocess_pairs(
    pairs: Iterable[AnchorPairDistance],
    *,
    min_sigma_m: float,
    min_distance_m: float,
) -> list[ProcessedAnchorPair]:
    aggregates: dict[tuple[str, str], dict[str, float | set[str]]] = {}
    for pair in pairs:
        if not pair.enabled:
            continue
        anchor_a = str(pair.anchor_a_id).strip()
        anchor_b = str(pair.anchor_b_id).strip()
        if not anchor_a or not anchor_b or anchor_a == anchor_b:
            continue
        distance = float(pair.distance_m)
        sigma = max(abs(float(pair.sigma_m)), min_sigma_m)
        if not math.isfinite(distance) or not math.isfinite(sigma):
            continue
        if distance <= min_distance_m:
            continue
        key = tuple(sorted((anchor_a, anchor_b)))
        weight = 1.0 / (sigma * sigma)
        aggregate = aggregates.setdefault(
            key,
            {"weighted_distance": 0.0, "weight": 0.0, "sources": set()},
        )
        aggregate["weighted_distance"] = float(aggregate["weighted_distance"]) + distance * weight
        aggregate["weight"] = float(aggregate["weight"]) + weight
        sources = aggregate["sources"]
        if isinstance(sources, set) and pair.source:
            sources.add(str(pair.source))

    processed: list[ProcessedAnchorPair] = []
    for (anchor_a, anchor_b), aggregate in sorted(aggregates.items()):
        weight = float(aggregate["weight"])
        if weight <= 0.0:
            continue
        sigma = max(math.sqrt(1.0 / weight), min_sigma_m)
        sources = aggregate["sources"]
        source_text = ", ".join(sorted(sources)) if isinstance(sources, set) and sources else "survey"
        processed.append(
            ProcessedAnchorPair(
                anchor_a_id=anchor_a,
                anchor_b_id=anchor_b,
                distance_m=float(aggregate["weighted_distance"]) / weight,
                sigma_m=sigma,
                weight=weight,
                source=source_text,
            )
        )

    if len(processed) < 1:
        raise ValueError("At least one valid anchor-to-anchor distance is required.")
    if len(_anchor_ids(processed)) < 2:
        raise ValueError("At least two anchors are required.")
    return processed


def _anchor_ids(pairs: Iterable[ProcessedAnchorPair]) -> list[str]:
    ids: set[str] = set()
    for pair in pairs:
        ids.add(pair.anchor_a_id)
        ids.add(pair.anchor_b_id)
    return sorted(ids)


def _validate_connected(anchor_ids: list[str], pairs: list[ProcessedAnchorPair]) -> None:
    neighbors: dict[str, set[str]] = {anchor_id: set() for anchor_id in anchor_ids}
    for pair in pairs:
        neighbors[pair.anchor_a_id].add(pair.anchor_b_id)
        neighbors[pair.anchor_b_id].add(pair.anchor_a_id)
    seen = {anchor_ids[0]}
    queue = [anchor_ids[0]]
    while queue:
        current = queue.pop(0)
        for neighbor in neighbors[current]:
            if neighbor not in seen:
                seen.add(neighbor)
                queue.append(neighbor)
    if len(seen) != len(anchor_ids):
        missing = ", ".join(anchor_id for anchor_id in anchor_ids if anchor_id not in seen)
        raise ValueError(f"Anchor distance graph is disconnected; missing {missing}.")


def _graph_diagnostics_from_processed(
    anchor_ids: list[str],
    pairs: list[ProcessedAnchorPair],
    *,
    length_bin_m: float,
) -> AnchorGraphDiagnostics:
    complete_pair_count = len(anchor_ids) * (len(anchor_ids) - 1) // 2
    required_rank = max(0, 2 * len(anchor_ids) - 3)
    rigidity_rank = _generic_rigidity_rank(anchor_ids, pairs)
    locally_rigid = len(anchor_ids) <= 2 or rigidity_rank >= required_rank
    redundantly_rigid = _is_redundantly_rigid(anchor_ids, pairs, required_rank)
    articulation_points = _articulation_points(anchor_ids, pairs)
    two_vertex_cuts = _two_vertex_cuts(anchor_ids, pairs)
    bridge_edges = _bridge_edges(anchor_ids, pairs)
    globally_rigid = (
        len(anchor_ids) <= 3 and len(pairs) == complete_pair_count
    ) or (
        locally_rigid
        and redundantly_rigid
        and not articulation_points
        and not two_vertex_cuts
    )
    near_orbits = _near_automorphism_orbits(
        anchor_ids,
        pairs,
        length_bin_m=max(length_bin_m, 1e-6),
    )
    warnings: list[str] = []
    if not locally_rigid:
        warnings.append(
            "Measured graph is not generically locally rigid in 2D; flexes or folds may fit the same ranges."
        )
    elif not globally_rigid:
        warnings.append(
            "Measured graph is locally rigid but not generically globally rigid; mirror/flip alternatives may exist."
        )
    if articulation_points:
        warnings.append(
            "Anchor graph has articulation vertices: " + ", ".join(articulation_points) + "."
        )
    if two_vertex_cuts:
        preview = ", ".join(f"{a}/{b}" for a, b in two_vertex_cuts[:4])
        if len(two_vertex_cuts) > 4:
            preview += ", ..."
        warnings.append(f"Anchor graph has 2-vertex separator candidates: {preview}.")
    if bridge_edges:
        preview = ", ".join(f"{a}-{b}" for a, b in bridge_edges[:4])
        if len(bridge_edges) > 4:
            preview += ", ..."
        warnings.append(f"Anchor graph has bridge edges: {preview}.")
    if near_orbits:
        largest = max(len(orbit) for orbit in near_orbits)
        warnings.append(
            f"Observed graph has near-indistinguishable label groups up to size {largest}; low RMSE may not choose a unique labeling."
        )
    return AnchorGraphDiagnostics(
        anchor_count=len(anchor_ids),
        pair_count=len(pairs),
        complete_pair_count=complete_pair_count,
        missing_pair_count=complete_pair_count - len(pairs),
        required_rigidity_rank=required_rank,
        rigidity_rank=rigidity_rank,
        is_locally_rigid=locally_rigid,
        is_redundantly_rigid=redundantly_rigid,
        is_generically_globally_rigid_2d=globally_rigid,
        articulation_points=tuple(articulation_points),
        two_vertex_cuts=tuple(two_vertex_cuts),
        bridge_edges=tuple(bridge_edges),
        near_automorphism_orbits=tuple(near_orbits),
        warnings=tuple(warnings),
    )


def _generic_rigidity_rank(
    anchor_ids: list[str],
    pairs: list[ProcessedAnchorPair],
    *,
    skip_edge_index: int | None = None,
) -> int:
    if len(anchor_ids) < 2 or not pairs:
        return 0
    rng = random.Random(7919 + len(anchor_ids) * 101 + len(pairs) * 17)
    positions = {
        anchor_id: (rng.uniform(-1.0, 1.0), rng.uniform(-1.0, 1.0))
        for anchor_id in anchor_ids
    }
    index = {anchor_id: offset for offset, anchor_id in enumerate(anchor_ids)}
    rows: list[list[float]] = []
    for edge_index, pair in enumerate(pairs):
        if skip_edge_index is not None and edge_index == skip_edge_index:
            continue
        row = [0.0] * (2 * len(anchor_ids))
        a = index[pair.anchor_a_id]
        b = index[pair.anchor_b_id]
        ax, ay = positions[pair.anchor_a_id]
        bx, by = positions[pair.anchor_b_id]
        dx = ax - bx
        dy = ay - by
        row[2 * a] = dx
        row[2 * a + 1] = dy
        row[2 * b] = -dx
        row[2 * b + 1] = -dy
        rows.append(row)
    return _matrix_rank(rows, tolerance=1e-9)


def _matrix_rank(matrix: list[list[float]], *, tolerance: float) -> int:
    if not matrix:
        return 0
    rows = [row[:] for row in matrix]
    row_count = len(rows)
    col_count = len(rows[0])
    rank = 0
    for col in range(col_count):
        pivot = max(range(rank, row_count), key=lambda row: abs(rows[row][col]))
        if abs(rows[pivot][col]) <= tolerance:
            continue
        rows[rank], rows[pivot] = rows[pivot], rows[rank]
        pivot_value = rows[rank][col]
        for entry in range(col, col_count):
            rows[rank][entry] /= pivot_value
        for row in range(row_count):
            if row == rank:
                continue
            factor = rows[row][col]
            if abs(factor) <= tolerance:
                continue
            for entry in range(col, col_count):
                rows[row][entry] -= factor * rows[rank][entry]
        rank += 1
        if rank == row_count:
            break
    return rank


def _is_redundantly_rigid(
    anchor_ids: list[str],
    pairs: list[ProcessedAnchorPair],
    required_rank: int,
) -> bool:
    if len(anchor_ids) <= 2:
        return True
    if len(anchor_ids) <= 3:
        return len(pairs) == len(anchor_ids) * (len(anchor_ids) - 1) // 2
    if len(pairs) <= required_rank:
        return False
    if _generic_rigidity_rank(anchor_ids, pairs) < required_rank:
        return False
    for edge_index in range(len(pairs)):
        if _generic_rigidity_rank(anchor_ids, pairs, skip_edge_index=edge_index) < required_rank:
            return False
    return True


def _neighbors_from_pairs(
    anchor_ids: list[str],
    pairs: list[ProcessedAnchorPair],
) -> dict[str, set[str]]:
    neighbors = {anchor_id: set() for anchor_id in anchor_ids}
    for pair in pairs:
        neighbors[pair.anchor_a_id].add(pair.anchor_b_id)
        neighbors[pair.anchor_b_id].add(pair.anchor_a_id)
    return neighbors


def _is_connected_after_removing(
    anchor_ids: list[str],
    neighbors: dict[str, set[str]],
    removed_vertices: set[str],
    removed_edge: tuple[str, str] | None = None,
) -> bool:
    remaining = [anchor_id for anchor_id in anchor_ids if anchor_id not in removed_vertices]
    if len(remaining) <= 1:
        return True
    removed_edge_key = tuple(sorted(removed_edge)) if removed_edge else None
    seen = {remaining[0]}
    queue = [remaining[0]]
    while queue:
        current = queue.pop(0)
        for neighbor in neighbors[current]:
            if neighbor in removed_vertices:
                continue
            if removed_edge_key and tuple(sorted((current, neighbor))) == removed_edge_key:
                continue
            if neighbor not in seen:
                seen.add(neighbor)
                queue.append(neighbor)
    return len(seen) == len(remaining)


def _articulation_points(
    anchor_ids: list[str],
    pairs: list[ProcessedAnchorPair],
) -> list[str]:
    neighbors = _neighbors_from_pairs(anchor_ids, pairs)
    return [
        anchor_id
        for anchor_id in anchor_ids
        if not _is_connected_after_removing(anchor_ids, neighbors, {anchor_id})
    ]


def _two_vertex_cuts(
    anchor_ids: list[str],
    pairs: list[ProcessedAnchorPair],
) -> list[tuple[str, str]]:
    if len(anchor_ids) <= 3:
        return []
    neighbors = _neighbors_from_pairs(anchor_ids, pairs)
    cuts: list[tuple[str, str]] = []
    for first_index, first in enumerate(anchor_ids):
        for second in anchor_ids[first_index + 1 :]:
            if not _is_connected_after_removing(anchor_ids, neighbors, {first, second}):
                cuts.append((first, second))
    return cuts


def _bridge_edges(
    anchor_ids: list[str],
    pairs: list[ProcessedAnchorPair],
) -> list[tuple[str, str]]:
    neighbors = _neighbors_from_pairs(anchor_ids, pairs)
    bridges: list[tuple[str, str]] = []
    for pair in pairs:
        edge = tuple(sorted((pair.anchor_a_id, pair.anchor_b_id)))
        if not _is_connected_after_removing(anchor_ids, neighbors, set(), edge):
            bridges.append(edge)
    return bridges


def _near_automorphism_orbits(
    anchor_ids: list[str],
    pairs: list[ProcessedAnchorPair],
    *,
    length_bin_m: float,
) -> tuple[tuple[str, ...], ...]:
    edge_bins: dict[tuple[str, str], int] = {}
    neighbors: dict[str, list[tuple[str, int]]] = {anchor_id: [] for anchor_id in anchor_ids}
    for pair in pairs:
        key = tuple(sorted((pair.anchor_a_id, pair.anchor_b_id)))
        length_bin = int(round(pair.distance_m / length_bin_m))
        edge_bins[key] = length_bin
        neighbors[pair.anchor_a_id].append((pair.anchor_b_id, length_bin))
        neighbors[pair.anchor_b_id].append((pair.anchor_a_id, length_bin))
    colors: dict[str, tuple] = {}
    for anchor_id in anchor_ids:
        incident = sorted(length_bin for _neighbor, length_bin in neighbors[anchor_id])
        colors[anchor_id] = (len(neighbors[anchor_id]), tuple(incident))
    for _iteration in range(4):
        compressed: dict[tuple, int] = {}
        next_color_id = 0
        new_colors: dict[str, tuple] = {}
        for anchor_id in anchor_ids:
            signature = (
                colors[anchor_id],
                tuple(sorted((length_bin, colors[neighbor]) for neighbor, length_bin in neighbors[anchor_id])),
            )
            if signature not in compressed:
                compressed[signature] = next_color_id
                next_color_id += 1
            new_colors[anchor_id] = (compressed[signature],)
        colors = new_colors
    groups: dict[tuple, list[str]] = {}
    for anchor_id, color in colors.items():
        groups.setdefault(color, []).append(anchor_id)
    return tuple(tuple(group) for group in groups.values() if len(group) > 1)



def _layout_scale(pairs: list[ProcessedAnchorPair]) -> float:
    distances = [pair.distance_m for pair in pairs if pair.distance_m > 0.0]
    if not distances:
        return 1.0
    return max(median(distances), 0.25)


class _Parameterization:
    def __init__(self, anchor_ids: list[str]) -> None:
        self.anchor_ids = anchor_ids
        self.variable_index: dict[tuple[str, str], int] = {}
        index = 0
        for anchor_position, anchor_id in enumerate(anchor_ids):
            if anchor_position == 0:
                continue
            self.variable_index[(anchor_id, "x")] = index
            index += 1
            if anchor_position > 1:
                self.variable_index[(anchor_id, "y")] = index
                index += 1
        self.dimension = index

    def to_positions(self, params: list[float]) -> dict[str, tuple[float, float]]:
        positions: dict[str, tuple[float, float]] = {}
        for anchor_position, anchor_id in enumerate(self.anchor_ids):
            if anchor_position == 0:
                positions[anchor_id] = (0.0, 0.0)
                continue
            x_index = self.variable_index[(anchor_id, "x")]
            x_m = params[x_index]
            y_index = self.variable_index.get((anchor_id, "y"))
            y_m = params[y_index] if y_index is not None else 0.0
            positions[anchor_id] = (x_m, y_m)
        return positions

    def derivative_index(self, anchor_id: str, axis: str) -> int | None:
        return self.variable_index.get((anchor_id, axis))


def _initial_parameters(
    parameterization: _Parameterization,
    pairs: list[ProcessedAnchorPair],
    *,
    seed_count: int,
    scale: float,
    rng: random.Random,
) -> list[list[float]]:
    seeds: list[list[float]] = []
    seeds.append(_triangulated_seed(parameterization, pairs, scale))
    seeds.append(_circle_seed(parameterization, scale, flip_y=False))
    seeds.append(_circle_seed(parameterization, scale, flip_y=True))
    seeds.append(_degree_shell_seed(parameterization, pairs, scale))

    while len(seeds) < seed_count:
        seeds.append(_random_seed(parameterization, scale, rng))
    return seeds[:seed_count]


def _triangulated_seed(
    parameterization: _Parameterization,
    pairs: list[ProcessedAnchorPair],
    scale: float,
) -> list[float]:
    anchor_ids = parameterization.anchor_ids
    positions: dict[str, tuple[float, float]] = {anchor_ids[0]: (0.0, 0.0)}
    base_distance = _pair_distance(anchor_ids[0], anchor_ids[1], pairs) or scale
    positions[anchor_ids[1]] = (base_distance, 0.0)

    for index, anchor_id in enumerate(anchor_ids[2:], start=2):
        d0 = _pair_distance(anchor_ids[0], anchor_id, pairs)
        d1 = _pair_distance(anchor_ids[1], anchor_id, pairs)
        if d0 is not None and d1 is not None and base_distance > 1e-9:
            x_m = (d0 * d0 + base_distance * base_distance - d1 * d1) / (2.0 * base_distance)
            y_sq = max(d0 * d0 - x_m * x_m, 0.0)
            y_m = math.sqrt(y_sq)
            if index % 2:
                y_m = -y_m
            positions[anchor_id] = (x_m, y_m)
        else:
            angle = 2.0 * math.pi * (index - 1) / max(len(anchor_ids) - 1, 1)
            positions[anchor_id] = (
                math.cos(angle) * scale,
                math.sin(angle) * scale,
            )
    return _positions_to_params(parameterization, positions)


def _circle_seed(
    parameterization: _Parameterization,
    scale: float,
    *,
    flip_y: bool,
) -> list[float]:
    anchor_ids = parameterization.anchor_ids
    positions = {anchor_ids[0]: (0.0, 0.0), anchor_ids[1]: (scale, 0.0)}
    radius = max(scale, 0.25)
    for index, anchor_id in enumerate(anchor_ids[2:], start=2):
        angle = 2.0 * math.pi * (index - 1) / max(len(anchor_ids) - 1, 1)
        y_sign = -1.0 if flip_y else 1.0
        positions[anchor_id] = (
            radius * math.cos(angle),
            y_sign * radius * math.sin(angle),
        )
    return _positions_to_params(parameterization, positions)


def _degree_shell_seed(
    parameterization: _Parameterization,
    pairs: list[ProcessedAnchorPair],
    scale: float,
) -> list[float]:
    anchor_ids = parameterization.anchor_ids
    if len(anchor_ids) < 4:
        return _circle_seed(parameterization, scale, flip_y=False)

    degrees = _anchor_degrees(anchor_ids, pairs)
    outer_count = _inferred_outer_shell_count(len(anchor_ids))
    if outer_count <= 0 or outer_count >= len(anchor_ids):
        return _circle_seed(parameterization, scale, flip_y=False)

    by_degree = sorted(anchor_ids, key=lambda anchor_id: (degrees.get(anchor_id, 0), anchor_id))
    outer_ids = by_degree[:outer_count]
    inner_ids = by_degree[outer_count:]
    outer_radius = max(scale * math.sqrt(len(anchor_ids)) * 0.5, scale)
    inner_radius = max(outer_radius * 0.42, scale * 0.45, 0.25)
    positions: dict[str, tuple[float, float]] = {}

    for index, anchor_id in enumerate(outer_ids):
        angle = 2.0 * math.pi * index / max(len(outer_ids), 1)
        positions[anchor_id] = (
            outer_radius * math.cos(angle),
            outer_radius * math.sin(angle),
        )
    for index, anchor_id in enumerate(inner_ids):
        angle = 2.0 * math.pi * (index + 0.5) / max(len(inner_ids), 1)
        positions[anchor_id] = (
            inner_radius * math.cos(angle),
            inner_radius * math.sin(angle),
        )

    try:
        positions = rotate_layout_to_level(positions, anchor_ids[0], anchor_ids[1])
    except ValueError:
        ax, ay = positions[anchor_ids[0]]
        positions = {
            anchor_id: (x_m - ax, y_m - ay)
            for anchor_id, (x_m, y_m) in positions.items()
        }
    return _positions_to_params(parameterization, positions)


def _inferred_outer_shell_count(anchor_count: int) -> int:
    if anchor_count < 4:
        return 0
    perimeter_estimate = 4.0 * math.sqrt(anchor_count) - 4.0
    return min(anchor_count - 1, max(1, int(round(perimeter_estimate))))


def _random_seed(
    parameterization: _Parameterization,
    scale: float,
    rng: random.Random,
) -> list[float]:
    anchor_ids = parameterization.anchor_ids
    positions = {
        anchor_ids[0]: (0.0, 0.0),
        anchor_ids[1]: (max(scale + rng.gauss(0.0, scale * 0.25), 0.1), 0.0),
    }
    radius = max(scale, 0.25)
    for anchor_id in anchor_ids[2:]:
        angle = rng.uniform(-math.pi, math.pi)
        distance = rng.uniform(0.35 * radius, 1.75 * radius)
        positions[anchor_id] = (
            distance * math.cos(angle),
            distance * math.sin(angle),
        )
    return _positions_to_params(parameterization, positions)


def _positions_to_params(
    parameterization: _Parameterization,
    positions: dict[str, tuple[float, float]],
) -> list[float]:
    params = [0.0] * parameterization.dimension
    for anchor_id, (x_m, y_m) in positions.items():
        x_index = parameterization.derivative_index(anchor_id, "x")
        if x_index is not None:
            params[x_index] = x_m
        y_index = parameterization.derivative_index(anchor_id, "y")
        if y_index is not None:
            params[y_index] = y_m
    return params


def _pair_distance(
    anchor_a_id: str,
    anchor_b_id: str,
    pairs: list[ProcessedAnchorPair],
) -> float | None:
    wanted = set((anchor_a_id, anchor_b_id))
    for pair in pairs:
        if {pair.anchor_a_id, pair.anchor_b_id} == wanted:
            return pair.distance_m
    return None


def _local_minimize(
    initial_params: list[float],
    parameterization: _Parameterization,
    pairs: list[ProcessedAnchorPair],
    priors: _LayoutPriors | None = None,
    *,
    max_iterations: int,
) -> tuple[list[float], float]:
    if _np is not None and parameterization.dimension > 0:
        result = _local_minimize_numpy(
            initial_params,
            parameterization,
            pairs,
            priors,
            max_iterations=max_iterations,
        )
        if result is not None:
            return result
    return _local_minimize_python(
        initial_params,
        parameterization,
        pairs,
        priors,
        max_iterations=max_iterations,
    )


def _local_minimize_python(
    initial_params: list[float],
    parameterization: _Parameterization,
    pairs: list[ProcessedAnchorPair],
    priors: _LayoutPriors | None = None,
    *,
    max_iterations: int,
) -> tuple[list[float], float]:
    params = list(initial_params)
    energy = _spring_energy(params, parameterization, pairs, priors)
    damping = 1e-3

    for _iteration in range(max(max_iterations, 1)):
        normal, rhs = _normal_equations(params, parameterization, pairs, priors)
        if not normal:
            break
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
        candidate_energy = _spring_energy(candidate, parameterization, pairs, priors)
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


def _local_minimize_numpy(
    initial_params: list[float],
    parameterization: _Parameterization,
    pairs: list[ProcessedAnchorPair],
    priors: _LayoutPriors | None = None,
    *,
    max_iterations: int,
) -> tuple[list[float], float] | None:
    assert _np is not None
    try:
        context = _numpy_solver_context(parameterization, pairs, priors)
        params = _np.asarray(initial_params, dtype=float).copy()
        energy = _numpy_energy(params, context)
        damping = 1e-3

        for _iteration in range(max(max_iterations, 1)):
            energy, normal, rhs = _numpy_energy_normal_rhs(params, context)
            if normal.size == 0:
                break
            damped = normal.copy()
            diagonal = _np.diag(normal)
            diag_indices = _np.diag_indices_from(damped)
            damped[diag_indices] += damping * _np.maximum(diagonal, 1.0)
            try:
                delta = _np.linalg.solve(damped, rhs)
            except _np.linalg.LinAlgError:
                damping *= 10.0
                if damping > 1e12:
                    break
                continue
            if not _np.all(_np.isfinite(delta)):
                damping *= 10.0
                if damping > 1e12:
                    break
                continue
            if float(_np.linalg.norm(delta)) <= 1e-10:
                break
            candidate = params + delta
            candidate_energy = _numpy_energy(candidate, context)
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
        return [float(value) for value in params.tolist()], float(energy)
    except Exception:
        return None


def _numpy_solver_context(
    parameterization: _Parameterization,
    pairs: list[ProcessedAnchorPair],
    priors: _LayoutPriors | None,
) -> dict[str, object]:
    assert _np is not None
    anchor_ids = parameterization.anchor_ids
    anchor_index = {anchor_id: index for index, anchor_id in enumerate(anchor_ids)}
    anchor_count = len(anchor_ids)
    var_x = _np.full(anchor_count, -1, dtype=int)
    var_y = _np.full(anchor_count, -1, dtype=int)
    for anchor_id, index in anchor_index.items():
        x_index = parameterization.derivative_index(anchor_id, "x")
        y_index = parameterization.derivative_index(anchor_id, "y")
        if x_index is not None:
            var_x[index] = x_index
        if y_index is not None:
            var_y[index] = y_index

    priors = priors or _empty_layout_priors()
    pair_a = _np.asarray([anchor_index[pair.anchor_a_id] for pair in pairs], dtype=int)
    pair_b = _np.asarray([anchor_index[pair.anchor_b_id] for pair in pairs], dtype=int)
    pair_distance = _np.asarray([pair.distance_m for pair in pairs], dtype=float)
    pair_sqrt_weight = _np.sqrt(_np.asarray([pair.weight for pair in pairs], dtype=float))

    all_a: list[int] = []
    all_b: list[int] = []
    unmeasured_a: list[int] = []
    unmeasured_b: list[int] = []
    fallback_angles: list[float] = []
    unmeasured_angles: list[float] = []
    for index_a, anchor_a in enumerate(anchor_ids):
        for index_b in range(index_a + 1, anchor_count):
            anchor_b = anchor_ids[index_b]
            all_a.append(index_a)
            all_b.append(index_b)
            fallback_angles.append(2.0 * math.pi * index_b / max(anchor_count, 1))
            if _pair_key(anchor_a, anchor_b) not in priors.measured_pair_keys:
                unmeasured_a.append(index_a)
                unmeasured_b.append(index_b)
                unmeasured_angles.append(2.0 * math.pi * index_b / max(anchor_count, 1))

    boundary_indices: list[int] = []
    boundary_targets: list[float] = []
    for anchor_id, target_radius in priors.boundary_radius_targets_m.items():
        if anchor_id in anchor_index:
            boundary_indices.append(anchor_index[anchor_id])
            boundary_targets.append(float(target_radius))

    return {
        "anchor_count": anchor_count,
        "dimension": parameterization.dimension,
        "var_x": var_x,
        "var_y": var_y,
        "pair_a": pair_a,
        "pair_b": pair_b,
        "pair_distance": pair_distance,
        "pair_sqrt_weight": pair_sqrt_weight,
        "spacing_a": _np.asarray(all_a, dtype=int),
        "spacing_b": _np.asarray(all_b, dtype=int),
        "spacing_angles": _np.asarray(fallback_angles, dtype=float),
        "spacing_target": float(priors.min_anchor_spacing_m),
        "spacing_sqrt_weight": math.sqrt(priors.spacing_weight) if priors.spacing_weight > 0.0 else 0.0,
        "unmeasured_a": _np.asarray(unmeasured_a, dtype=int),
        "unmeasured_b": _np.asarray(unmeasured_b, dtype=int),
        "unmeasured_angles": _np.asarray(unmeasured_angles, dtype=float),
        "unmeasured_target": float(priors.unmeasured_pair_min_distance_m),
        "unmeasured_sqrt_weight": math.sqrt(priors.unmeasured_pair_weight) if priors.unmeasured_pair_weight > 0.0 else 0.0,
        "boundary_indices": _np.asarray(boundary_indices, dtype=int),
        "boundary_targets": _np.asarray(boundary_targets, dtype=float),
        "boundary_sqrt_weight": math.sqrt(priors.boundary_weight) if priors.boundary_weight > 0.0 else 0.0,
    }


def _numpy_coords(params, context: dict[str, object]):
    assert _np is not None
    anchor_count = int(context["anchor_count"])
    coords = _np.zeros((anchor_count, 2), dtype=float)
    var_x = context["var_x"]
    var_y = context["var_y"]
    x_mask = var_x >= 0
    y_mask = var_y >= 0
    coords[x_mask, 0] = params[var_x[x_mask]]
    coords[y_mask, 1] = params[var_y[y_mask]]
    return coords


def _numpy_add_pair_derivatives(jacobian, row_start: int, context: dict[str, object], anchor_a, anchor_b, deriv_x, deriv_y, sign_a: float) -> None:
    assert _np is not None
    if len(anchor_a) == 0:
        return
    rows = _np.arange(row_start, row_start + len(anchor_a), dtype=int)
    var_x = context["var_x"]
    var_y = context["var_y"]

    idx = var_x[anchor_a]
    mask = idx >= 0
    jacobian[rows[mask], idx[mask]] += sign_a * deriv_x[mask]
    idx = var_y[anchor_a]
    mask = idx >= 0
    jacobian[rows[mask], idx[mask]] += sign_a * deriv_y[mask]

    idx = var_x[anchor_b]
    mask = idx >= 0
    jacobian[rows[mask], idx[mask]] -= sign_a * deriv_x[mask]
    idx = var_y[anchor_b]
    mask = idx >= 0
    jacobian[rows[mask], idx[mask]] -= sign_a * deriv_y[mask]


def _numpy_fill_distance_rows(residuals, jacobian, row: int, context: dict[str, object], coords, *, jacobian_enabled: bool) -> int:
    assert _np is not None
    pair_a = context["pair_a"]
    pair_b = context["pair_b"]
    if len(pair_a) == 0:
        return row
    dx = coords[pair_a, 0] - coords[pair_b, 0]
    dy = coords[pair_a, 1] - coords[pair_b, 1]
    lengths = _np.maximum(_np.hypot(dx, dy), 1e-9)
    sqrt_weight = context["pair_sqrt_weight"]
    weighted = sqrt_weight * (lengths - context["pair_distance"])
    count = len(pair_a)
    residuals[row : row + count] = weighted
    if jacobian_enabled:
        _numpy_add_pair_derivatives(
            jacobian,
            row,
            context,
            pair_a,
            pair_b,
            sqrt_weight * dx / lengths,
            sqrt_weight * dy / lengths,
            1.0,
        )
    return row + count


def _numpy_fill_repulsion_rows(residuals, jacobian, row: int, context: dict[str, object], coords, *, prefix: str, jacobian_enabled: bool) -> int:
    assert _np is not None
    sqrt_weight = float(context[f"{prefix}_sqrt_weight"])
    target = float(context[f"{prefix}_target"])
    if target <= 0.0 or sqrt_weight <= 0.0:
        return row
    anchor_a = context[f"{prefix}_a"]
    anchor_b = context[f"{prefix}_b"]
    if len(anchor_a) == 0:
        return row
    dx = coords[anchor_a, 0] - coords[anchor_b, 0]
    dy = coords[anchor_a, 1] - coords[anchor_b, 1]
    raw_lengths = _np.hypot(dx, dy)
    residual = target - raw_lengths
    active = residual > 0.0
    if not bool(active.any()):
        return row
    anchor_a = anchor_a[active]
    anchor_b = anchor_b[active]
    dx = dx[active]
    dy = dy[active]
    raw_lengths = raw_lengths[active]
    residual = residual[active]
    lengths = _np.maximum(raw_lengths, 1e-9)
    ux = dx / lengths
    uy = dy / lengths
    zero = raw_lengths <= 1e-9
    if bool(zero.any()):
        angles = context[f"{prefix}_angles"][active][zero]
        ux[zero] = _np.cos(angles)
        uy[zero] = _np.sin(angles)
    count = len(anchor_a)
    residuals[row : row + count] = sqrt_weight * residual
    if jacobian_enabled:
        _numpy_add_pair_derivatives(
            jacobian,
            row,
            context,
            anchor_a,
            anchor_b,
            sqrt_weight * ux,
            sqrt_weight * uy,
            -1.0,
        )
    return row + count


def _numpy_fill_boundary_rows(residuals, jacobian, row: int, context: dict[str, object], coords, *, jacobian_enabled: bool) -> int:
    assert _np is not None
    sqrt_weight = float(context["boundary_sqrt_weight"])
    boundary_indices = context["boundary_indices"]
    if sqrt_weight <= 0.0 or len(boundary_indices) == 0:
        return row
    center = coords.mean(axis=0)
    dx = coords[boundary_indices, 0] - center[0]
    dy = coords[boundary_indices, 1] - center[1]
    radii = _np.hypot(dx, dy)
    residual = context["boundary_targets"] - radii
    active_positions = _np.flatnonzero(residual > 0.0)
    if len(active_positions) == 0:
        return row
    anchor_count = int(context["anchor_count"])
    var_x = context["var_x"]
    var_y = context["var_y"]
    x_mask = var_x >= 0
    y_mask = var_y >= 0
    for active_position in active_positions:
        anchor_index = int(boundary_indices[active_position])
        residuals[row] = sqrt_weight * residual[active_position]
        if jacobian_enabled:
            radius = float(radii[active_position])
            if radius <= 1e-9:
                angle = 2.0 * math.pi * anchor_index / max(anchor_count, 1)
                ux = math.cos(angle)
                uy = math.sin(angle)
            else:
                ux = float(dx[active_position] / radius)
                uy = float(dy[active_position] / radius)
            coefficients = _np.full(anchor_count, -1.0 / max(anchor_count, 1), dtype=float)
            coefficients[anchor_index] = 1.0 - 1.0 / max(anchor_count, 1)
            derivative_x = -sqrt_weight * coefficients * ux
            derivative_y = -sqrt_weight * coefficients * uy
            jacobian[row, var_x[x_mask]] += derivative_x[x_mask]
            jacobian[row, var_y[y_mask]] += derivative_y[y_mask]
        row += 1
    return row


def _numpy_weighted_residuals(params, context: dict[str, object], *, jacobian_enabled: bool):
    assert _np is not None
    dimension = int(context["dimension"])
    coords = _numpy_coords(params, context)
    row_capacity = (
        len(context["pair_a"])
        + len(context["spacing_a"])
        + len(context["unmeasured_a"])
        + len(context["boundary_indices"])
    )
    residuals = _np.zeros(row_capacity, dtype=float)
    jacobian = _np.zeros((row_capacity, dimension), dtype=float) if jacobian_enabled else None
    row = 0
    row = _numpy_fill_distance_rows(residuals, jacobian, row, context, coords, jacobian_enabled=jacobian_enabled)
    row = _numpy_fill_repulsion_rows(residuals, jacobian, row, context, coords, prefix="spacing", jacobian_enabled=jacobian_enabled)
    row = _numpy_fill_repulsion_rows(residuals, jacobian, row, context, coords, prefix="unmeasured", jacobian_enabled=jacobian_enabled)
    row = _numpy_fill_boundary_rows(residuals, jacobian, row, context, coords, jacobian_enabled=jacobian_enabled)
    if jacobian_enabled:
        return residuals[:row], jacobian[:row, :]
    return residuals[:row], None


def _numpy_energy(params, context: dict[str, object]) -> float:
    residuals, _jacobian = _numpy_weighted_residuals(params, context, jacobian_enabled=False)
    return 0.5 * float(residuals @ residuals)


def _numpy_energy_normal_rhs(params, context: dict[str, object]):
    assert _np is not None
    residuals, jacobian = _numpy_weighted_residuals(params, context, jacobian_enabled=True)
    if jacobian is None or jacobian.size == 0:
        dimension = int(context["dimension"])
        return 0.5 * float(residuals @ residuals), _np.zeros((dimension, dimension), dtype=float), _np.zeros(dimension, dtype=float)
    normal = jacobian.T @ jacobian
    rhs = -(jacobian.T @ residuals)
    return 0.5 * float(residuals @ residuals), normal, rhs


def _spring_energy(
    params: list[float],
    parameterization: _Parameterization,
    pairs: list[ProcessedAnchorPair],
    priors: _LayoutPriors | None = None,
) -> float:
    positions = parameterization.to_positions(params)
    energy = 0.0
    for pair in pairs:
        ax, ay = positions[pair.anchor_a_id]
        bx, by = positions[pair.anchor_b_id]
        residual = math.hypot(ax - bx, ay - by) - pair.distance_m
        energy += 0.5 * pair.weight * residual * residual
    priors = priors or _empty_layout_priors()
    energy += _spacing_prior_energy(positions, priors)
    energy += _unmeasured_pair_prior_energy(positions, priors)
    energy += _boundary_prior_energy(positions, priors)
    return energy


def _normal_equations(
    params: list[float],
    parameterization: _Parameterization,
    pairs: list[ProcessedAnchorPair],
    priors: _LayoutPriors | None = None,
) -> tuple[list[list[float]], list[float]]:
    dimension = parameterization.dimension
    normal = [[0.0 for _col in range(dimension)] for _row in range(dimension)]
    rhs = [0.0 for _row in range(dimension)]
    positions = parameterization.to_positions(params)

    for pair in pairs:
        ax, ay = positions[pair.anchor_a_id]
        bx, by = positions[pair.anchor_b_id]
        dx = ax - bx
        dy = ay - by
        length = max(math.hypot(dx, dy), 1e-9)
        residual = length - pair.distance_m
        sqrt_weight = math.sqrt(pair.weight)
        weighted_residual = sqrt_weight * residual
        derivatives: dict[int, float] = {}

        for anchor_id, sign in ((pair.anchor_a_id, 1.0), (pair.anchor_b_id, -1.0)):
            x_index = parameterization.derivative_index(anchor_id, "x")
            y_index = parameterization.derivative_index(anchor_id, "y")
            if x_index is not None:
                derivatives[x_index] = derivatives.get(x_index, 0.0) + sign * sqrt_weight * dx / length
            if y_index is not None:
                derivatives[y_index] = derivatives.get(y_index, 0.0) + sign * sqrt_weight * dy / length

        for row_index, row_value in derivatives.items():
            rhs[row_index] -= row_value * weighted_residual
            for col_index, col_value in derivatives.items():
                normal[row_index][col_index] += row_value * col_value
    _append_prior_normal_equations(normal, rhs, positions, parameterization, priors)
    return normal, rhs


def _empty_layout_priors() -> _LayoutPriors:
    return _LayoutPriors(
        min_anchor_spacing_m=0.0,
        spacing_weight=0.0,
        unmeasured_pair_min_distance_m=0.0,
        unmeasured_pair_weight=0.0,
        measured_pair_keys=frozenset(),
        boundary_radius_targets_m={},
        boundary_weight=0.0,
    )


def _layout_priors(
    anchor_ids: list[str],
    pairs: list[ProcessedAnchorPair],
    *,
    scale: float,
    min_anchor_spacing_m: float | None,
    anchor_spacing_sigma_m: float,
    unmeasured_pair_min_distance_m: float | None,
    unmeasured_pair_sigma_m: float,
    boundary_degree_prior_sigma_m: float | None,
) -> _LayoutPriors:
    if min_anchor_spacing_m is None:
        spacing = _inferred_min_anchor_spacing(scale)
    else:
        spacing = max(float(min_anchor_spacing_m), 0.0)
    spacing_sigma = max(abs(float(anchor_spacing_sigma_m)), 1e-6)
    spacing_weight = 1.0 / (spacing_sigma * spacing_sigma) if spacing > 0.0 else 0.0

    if unmeasured_pair_min_distance_m is None:
        unmeasured_spacing = _inferred_unmeasured_pair_spacing(pairs)
    else:
        unmeasured_spacing = max(float(unmeasured_pair_min_distance_m), 0.0)
    unmeasured_sigma = max(abs(float(unmeasured_pair_sigma_m)), 1e-6)
    unmeasured_weight = (
        1.0 / (unmeasured_sigma * unmeasured_sigma)
        if unmeasured_spacing > 0.0
        else 0.0
    )
    measured_keys = frozenset(
        _pair_key(pair.anchor_a_id, pair.anchor_b_id)
        for pair in pairs
    )

    boundary_targets = _boundary_radius_targets(anchor_ids, pairs, scale=scale)
    if boundary_degree_prior_sigma_m == 0.0:
        boundary_targets = {}
    boundary_sigma = (
        max(scale * 2.0, 12.0)
        if boundary_degree_prior_sigma_m is None
        else max(abs(float(boundary_degree_prior_sigma_m)), 1e-6)
    )
    boundary_weight = 1.0 / (boundary_sigma * boundary_sigma) if boundary_targets else 0.0
    return _LayoutPriors(
        min_anchor_spacing_m=spacing,
        spacing_weight=spacing_weight,
        unmeasured_pair_min_distance_m=unmeasured_spacing,
        unmeasured_pair_weight=unmeasured_weight,
        measured_pair_keys=measured_keys,
        boundary_radius_targets_m=boundary_targets,
        boundary_weight=boundary_weight,
    )


def _inferred_min_anchor_spacing(scale: float) -> float:
    return min(2.0, max(0.75, scale * 0.35))


def _inferred_unmeasured_pair_spacing(pairs: list[ProcessedAnchorPair]) -> float:
    if not pairs:
        return 0.0
    max_measured = max(pair.distance_m for pair in pairs)
    return max(max_measured * 0.90, 0.0)


def _pair_key(anchor_a_id: str, anchor_b_id: str) -> tuple[str, str]:
    return tuple(sorted((anchor_a_id, anchor_b_id)))


def _anchor_degrees(
    anchor_ids: list[str],
    pairs: Iterable[ProcessedAnchorPair],
) -> dict[str, int]:
    degrees = {anchor_id: 0 for anchor_id in anchor_ids}
    for pair in pairs:
        degrees[pair.anchor_a_id] = degrees.get(pair.anchor_a_id, 0) + 1
        degrees[pair.anchor_b_id] = degrees.get(pair.anchor_b_id, 0) + 1
    return degrees


def _boundary_radius_targets(
    anchor_ids: list[str],
    pairs: list[ProcessedAnchorPair],
    *,
    scale: float,
) -> dict[str, float]:
    if len(anchor_ids) < 4:
        return {}
    degrees = _anchor_degrees(anchor_ids, pairs)
    min_degree = min(degrees.values(), default=0)
    max_degree = max(degrees.values(), default=0)
    if max_degree <= min_degree:
        return {}

    outer_radius = max(scale * math.sqrt(len(anchor_ids)) * 0.5, scale)
    inner_radius = outer_radius * 0.25
    span = max_degree - min_degree
    return {
        anchor_id: inner_radius
        + (outer_radius - inner_radius) * ((max_degree - degree) / span)
        for anchor_id, degree in degrees.items()
    }


def _spacing_prior_energy(
    positions: dict[str, tuple[float, float]],
    priors: _LayoutPriors,
) -> float:
    if priors.min_anchor_spacing_m <= 0.0 or priors.spacing_weight <= 0.0:
        return 0.0
    energy = 0.0
    anchor_ids = sorted(positions)
    for index, anchor_a in enumerate(anchor_ids):
        ax, ay = positions[anchor_a]
        for anchor_b in anchor_ids[index + 1 :]:
            bx, by = positions[anchor_b]
            residual = priors.min_anchor_spacing_m - math.hypot(ax - bx, ay - by)
            if residual > 0.0:
                energy += 0.5 * priors.spacing_weight * residual * residual
    return energy


def _unmeasured_pair_prior_energy(
    positions: dict[str, tuple[float, float]],
    priors: _LayoutPriors,
) -> float:
    if priors.unmeasured_pair_min_distance_m <= 0.0 or priors.unmeasured_pair_weight <= 0.0:
        return 0.0
    energy = 0.0
    anchor_ids = sorted(positions)
    for index, anchor_a in enumerate(anchor_ids):
        ax, ay = positions[anchor_a]
        for anchor_b in anchor_ids[index + 1 :]:
            if _pair_key(anchor_a, anchor_b) in priors.measured_pair_keys:
                continue
            bx, by = positions[anchor_b]
            residual = priors.unmeasured_pair_min_distance_m - math.hypot(ax - bx, ay - by)
            if residual > 0.0:
                energy += 0.5 * priors.unmeasured_pair_weight * residual * residual
    return energy


def _boundary_prior_energy(
    positions: dict[str, tuple[float, float]],
    priors: _LayoutPriors,
) -> float:
    if not priors.boundary_radius_targets_m or priors.boundary_weight <= 0.0:
        return 0.0
    center_x, center_y = _layout_center(positions)
    energy = 0.0
    for anchor_id, target_radius in priors.boundary_radius_targets_m.items():
        if anchor_id not in positions:
            continue
        x_m, y_m = positions[anchor_id]
        residual = target_radius - math.hypot(x_m - center_x, y_m - center_y)
        if residual > 0.0:
            energy += 0.5 * priors.boundary_weight * residual * residual
    return energy


def _append_prior_normal_equations(
    normal: list[list[float]],
    rhs: list[float],
    positions: dict[str, tuple[float, float]],
    parameterization: _Parameterization,
    priors: _LayoutPriors | None,
) -> None:
    priors = priors or _empty_layout_priors()
    _append_spacing_prior_normal_equations(normal, rhs, positions, parameterization, priors)
    _append_unmeasured_pair_prior_normal_equations(normal, rhs, positions, parameterization, priors)
    _append_boundary_prior_normal_equations(normal, rhs, positions, parameterization, priors)


def _append_spacing_prior_normal_equations(
    normal: list[list[float]],
    rhs: list[float],
    positions: dict[str, tuple[float, float]],
    parameterization: _Parameterization,
    priors: _LayoutPriors,
) -> None:
    if priors.min_anchor_spacing_m <= 0.0 or priors.spacing_weight <= 0.0:
        return
    sqrt_weight = math.sqrt(priors.spacing_weight)
    anchor_ids = sorted(positions)
    for index, anchor_a in enumerate(anchor_ids):
        ax, ay = positions[anchor_a]
        for peer_offset, anchor_b in enumerate(anchor_ids[index + 1 :], start=1):
            bx, by = positions[anchor_b]
            dx = ax - bx
            dy = ay - by
            raw_length = math.hypot(dx, dy)
            residual = priors.min_anchor_spacing_m - raw_length
            if residual <= 0.0:
                continue
            if raw_length <= 1e-9:
                angle = 2.0 * math.pi * (index + peer_offset) / max(len(anchor_ids), 1)
                ux = math.cos(angle)
                uy = math.sin(angle)
            else:
                ux = dx / raw_length
                uy = dy / raw_length
            derivatives: dict[int, float] = {}
            for anchor_id, sign in ((anchor_a, -1.0), (anchor_b, 1.0)):
                x_index = parameterization.derivative_index(anchor_id, "x")
                y_index = parameterization.derivative_index(anchor_id, "y")
                if x_index is not None:
                    derivatives[x_index] = derivatives.get(x_index, 0.0) + sign * sqrt_weight * ux
                if y_index is not None:
                    derivatives[y_index] = derivatives.get(y_index, 0.0) + sign * sqrt_weight * uy
            _accumulate_normal_equation(normal, rhs, derivatives, sqrt_weight * residual)


def _append_unmeasured_pair_prior_normal_equations(
    normal: list[list[float]],
    rhs: list[float],
    positions: dict[str, tuple[float, float]],
    parameterization: _Parameterization,
    priors: _LayoutPriors,
) -> None:
    if priors.unmeasured_pair_min_distance_m <= 0.0 or priors.unmeasured_pair_weight <= 0.0:
        return
    sqrt_weight = math.sqrt(priors.unmeasured_pair_weight)
    anchor_ids = sorted(positions)
    for index, anchor_a in enumerate(anchor_ids):
        ax, ay = positions[anchor_a]
        for peer_offset, anchor_b in enumerate(anchor_ids[index + 1 :], start=1):
            if _pair_key(anchor_a, anchor_b) in priors.measured_pair_keys:
                continue
            bx, by = positions[anchor_b]
            dx = ax - bx
            dy = ay - by
            raw_length = math.hypot(dx, dy)
            residual = priors.unmeasured_pair_min_distance_m - raw_length
            if residual <= 0.0:
                continue
            if raw_length <= 1e-9:
                angle = 2.0 * math.pi * (index + peer_offset) / max(len(anchor_ids), 1)
                ux = math.cos(angle)
                uy = math.sin(angle)
            else:
                ux = dx / raw_length
                uy = dy / raw_length
            derivatives: dict[int, float] = {}
            for anchor_id, sign in ((anchor_a, -1.0), (anchor_b, 1.0)):
                x_index = parameterization.derivative_index(anchor_id, "x")
                y_index = parameterization.derivative_index(anchor_id, "y")
                if x_index is not None:
                    derivatives[x_index] = derivatives.get(x_index, 0.0) + sign * sqrt_weight * ux
                if y_index is not None:
                    derivatives[y_index] = derivatives.get(y_index, 0.0) + sign * sqrt_weight * uy
            _accumulate_normal_equation(normal, rhs, derivatives, sqrt_weight * residual)


def _append_boundary_prior_normal_equations(
    normal: list[list[float]],
    rhs: list[float],
    positions: dict[str, tuple[float, float]],
    parameterization: _Parameterization,
    priors: _LayoutPriors,
) -> None:
    if not priors.boundary_radius_targets_m or priors.boundary_weight <= 0.0:
        return
    center_x, center_y = _layout_center(positions)
    sqrt_weight = math.sqrt(priors.boundary_weight)
    anchor_count = max(len(positions), 1)
    for anchor_id, target_radius in priors.boundary_radius_targets_m.items():
        if anchor_id not in positions:
            continue
        x_m, y_m = positions[anchor_id]
        dx = x_m - center_x
        dy = y_m - center_y
        radius = math.hypot(dx, dy)
        residual = target_radius - radius
        if residual <= 0.0:
            continue
        if radius <= 1e-9:
            anchor_index = parameterization.anchor_ids.index(anchor_id)
            angle = 2.0 * math.pi * anchor_index / max(len(parameterization.anchor_ids), 1)
            ux = math.cos(angle)
            uy = math.sin(angle)
        else:
            ux = dx / radius
            uy = dy / radius
        derivatives: dict[int, float] = {}
        for peer_id in positions:
            center_derivative = -1.0 / anchor_count
            coefficient = 1.0 + center_derivative if peer_id == anchor_id else center_derivative
            x_index = parameterization.derivative_index(peer_id, "x")
            y_index = parameterization.derivative_index(peer_id, "y")
            if x_index is not None:
                derivatives[x_index] = derivatives.get(x_index, 0.0) - sqrt_weight * coefficient * ux
            if y_index is not None:
                derivatives[y_index] = derivatives.get(y_index, 0.0) - sqrt_weight * coefficient * uy
        _accumulate_normal_equation(normal, rhs, derivatives, sqrt_weight * residual)


def _accumulate_normal_equation(
    normal: list[list[float]],
    rhs: list[float],
    derivatives: dict[int, float],
    weighted_residual: float,
) -> None:
    for row_index, row_value in derivatives.items():
        rhs[row_index] -= row_value * weighted_residual
        for col_index, col_value in derivatives.items():
            normal[row_index][col_index] += row_value * col_value


def _solve_linear_system(matrix: list[list[float]], rhs: list[float]) -> list[float]:
    n = len(rhs)
    a = [row[:] + [rhs[index]] for index, row in enumerate(matrix)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda row: abs(a[row][col]))
        if abs(a[pivot][col]) <= 1e-14:
            raise ValueError("Singular normal equation matrix.")
        if pivot != col:
            a[col], a[pivot] = a[pivot], a[col]
        pivot_value = a[col][col]
        for entry in range(col, n + 1):
            a[col][entry] /= pivot_value
        for row in range(n):
            if row == col:
                continue
            factor = a[row][col]
            if abs(factor) <= 1e-18:
                continue
            for entry in range(col, n + 1):
                a[row][entry] -= factor * a[col][entry]
    return [a[row][n] for row in range(n)]


def _vector_norm(values: list[float]) -> float:
    return math.sqrt(sum(value * value for value in values))


def _rmse(values: Iterable[float]) -> float:
    collected = list(values)
    if not collected:
        return 0.0
    return math.sqrt(sum(value * value for value in collected) / len(collected))


def _layout_warnings(
    anchor_ids: list[str],
    pairs: list[ProcessedAnchorPair],
    rmse_m: float,
    max_residual_m: float,
) -> list[str]:
    warnings: list[str] = []
    diagnostics = _graph_diagnostics_from_processed(
        anchor_ids,
        pairs,
        length_bin_m=0.25,
    )
    warnings.extend(diagnostics.warnings)
    if rmse_m > 0.10:
        warnings.append("Spring RMSE is high; check bad pair ranges or NLOS measurements.")
    if max_residual_m > 0.25:
        warnings.append("At least one anchor pair residual exceeds 0.25 m.")
    return warnings


def _layout_center(positions_m: dict[str, tuple[float, float]]) -> tuple[float, float]:
    if not positions_m:
        return (0.0, 0.0)
    return (
        sum(x for x, _y in positions_m.values()) / len(positions_m),
        sum(y for _x, y in positions_m.values()) / len(positions_m),
    )


def _clean_positions(
    positions_m: dict[str, tuple[float, float]]
) -> dict[str, tuple[float, float]]:
    cleaned = {}
    for anchor_id, (x_m, y_m) in positions_m.items():
        cleaned[anchor_id] = (
            0.0 if abs(x_m) < 1e-12 else x_m,
            0.0 if abs(y_m) < 1e-12 else y_m,
        )
    return cleaned


def _pair_label(anchor_a_id: str, anchor_b_id: str) -> str:
    return f"{anchor_a_id}-{anchor_b_id}"
