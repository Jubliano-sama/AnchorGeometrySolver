from __future__ import annotations

from dataclasses import dataclass
import itertools
import math
from typing import Any

import numpy as np

from anchor_geometry_solver.compat import AnchorPairDistance, ensure_legacy_paths

ensure_legacy_paths()
import anchor_solver_ml_distance_completion as dc  # noqa: E402
from uwb_capture.anchor_geometry import _Parameterization, _anchor_ids, _positions_to_params, _preprocess_pairs, rotate_layout_to_level  # noqa: E402


INF = 1e12


@dataclass(frozen=True)
class RangeGraph:
    anchor_ids: tuple[str, ...]
    distances: dict[tuple[str, str], float]
    sigmas: dict[tuple[str, str], float]
    degree: dict[str, int]
    shortest_m: np.ndarray
    hops: np.ndarray
    index: dict[str, int]


@dataclass(frozen=True)
class PartialLayout:
    positions: dict[str, tuple[float, float]]
    score: float


def pair_key(anchor_a: str, anchor_b: str) -> tuple[str, str]:
    return tuple(sorted((anchor_a, anchor_b)))


def range_graph(known_pairs: list[AnchorPairDistance]) -> RangeGraph:
    processed = _preprocess_pairs(known_pairs, min_sigma_m=0.02, min_distance_m=0.05)
    anchor_ids = tuple(_anchor_ids(processed))
    index = {anchor_id: i for i, anchor_id in enumerate(anchor_ids)}
    n = len(anchor_ids)
    shortest = np.full((n, n), INF, dtype=float)
    hops = np.full((n, n), 10**9, dtype=float)
    np.fill_diagonal(shortest, 0.0)
    np.fill_diagonal(hops, 0.0)
    distances: dict[tuple[str, str], float] = {}
    sigmas: dict[tuple[str, str], float] = {}
    degree = {anchor_id: 0 for anchor_id in anchor_ids}
    for pair in processed:
        key = pair_key(pair.anchor_a_id, pair.anchor_b_id)
        distances[key] = float(pair.distance_m)
        sigmas[key] = float(pair.sigma_m)
        degree[pair.anchor_a_id] += 1
        degree[pair.anchor_b_id] += 1
        i = index[pair.anchor_a_id]
        j = index[pair.anchor_b_id]
        if pair.distance_m < shortest[i, j]:
            shortest[i, j] = shortest[j, i] = float(pair.distance_m)
            hops[i, j] = hops[j, i] = 1.0
    for k in range(n):
        for i in range(n):
            via = shortest[i, k]
            if via >= INF:
                continue
            for j in range(n):
                candidate = via + shortest[k, j]
                candidate_hops = hops[i, k] + hops[k, j]
                if candidate < shortest[i, j] - 1e-9 or (
                    abs(candidate - shortest[i, j]) <= 1e-9 and candidate_hops < hops[i, j]
                ):
                    shortest[i, j] = candidate
                    hops[i, j] = candidate_hops
    return RangeGraph(anchor_ids, distances, sigmas, degree, shortest, hops, index)


def circle_intersections(
    center_a: tuple[float, float],
    radius_a: float,
    center_b: tuple[float, float],
    radius_b: float,
) -> list[tuple[float, float]]:
    ax, ay = center_a
    bx, by = center_b
    dx = bx - ax
    dy = by - ay
    d = math.hypot(dx, dy)
    if d < 1e-9:
        return []
    along = (radius_a * radius_a - radius_b * radius_b + d * d) / (2.0 * d)
    h2 = radius_a * radius_a - along * along
    ux = dx / d
    uy = dy / d
    px = ax + along * ux
    py = ay + along * uy
    if h2 <= 1e-10:
        return [(px, py)]
    h = math.sqrt(h2)
    ox = -uy * h
    oy = ux * h
    return [(px + ox, py + oy), (px - ox, py - oy)]


def _triangle_area(side_ab: float, side_ac: float, side_bc: float) -> float:
    semiperimeter = 0.5 * (side_ab + side_ac + side_bc)
    return math.sqrt(max(semiperimeter * (semiperimeter - side_ab) * (semiperimeter - side_ac) * (semiperimeter - side_bc), 0.0))


def _starting_triangle(graph: RangeGraph) -> tuple[str, str, str] | None:
    if len(graph.anchor_ids) < 3:
        return None
    first = min(
        graph.anchor_ids,
        key=lambda anchor_id: (
            -graph.degree[anchor_id],
            -sum(graph.degree[other] for other in graph.anchor_ids if pair_key(anchor_id, other) in graph.distances),
            anchor_id,
        ),
    )
    candidates: list[tuple[float, tuple[str, str, str]]] = []
    for a, b, c in itertools.combinations(graph.anchor_ids, 3):
        keys = (pair_key(a, b), pair_key(a, c), pair_key(b, c))
        if any(key not in graph.distances for key in keys):
            continue
        if first not in {a, b, c}:
            continue
        side_ab = graph.distances[keys[0]]
        side_ac = graph.distances[keys[1]]
        side_bc = graph.distances[keys[2]]
        sides = (side_ab, side_ac, side_bc)
        area = _triangle_area(side_ab, side_ac, side_bc)
        if area <= 1e-8:
            continue
        skinny = min(sides) / max(sides)
        degree_bonus = 1.0 + 0.03 * (graph.degree[a] + graph.degree[b] + graph.degree[c])
        candidates.append((area * skinny * degree_bonus, (a, b, c)))
    if not candidates:
        return None
    _score, triangle = max(candidates, key=lambda item: item[0])
    if triangle[0] == first:
        return triangle
    ordered = [first, *[anchor_id for anchor_id in triangle if anchor_id != first]]
    return ordered[0], ordered[1], ordered[2]


def _triangle_positions(triangle: tuple[str, str, str], graph: RangeGraph) -> dict[str, tuple[float, float]]:
    anchor_a, anchor_b, anchor_c = triangle
    side_ab = graph.distances[pair_key(anchor_a, anchor_b)]
    side_ac = graph.distances[pair_key(anchor_a, anchor_c)]
    side_bc = graph.distances[pair_key(anchor_b, anchor_c)]
    x_c = (side_ac * side_ac + side_ab * side_ab - side_bc * side_bc) / max(2.0 * side_ab, 1e-9)
    y2 = max(side_ac * side_ac - x_c * x_c, 0.0)
    return {
        anchor_a: (0.0, 0.0),
        anchor_b: (side_ab, 0.0),
        anchor_c: (x_c, math.sqrt(y2)),
    }


def _known_refs(anchor_id: str, positions: dict[str, tuple[float, float]], graph: RangeGraph) -> list[tuple[str, tuple[float, float], float, float]]:
    refs: list[tuple[str, tuple[float, float], float, float]] = []
    for other, point in positions.items():
        key = pair_key(anchor_id, other)
        if key in graph.distances:
            refs.append((other, point, graph.distances[key], graph.sigmas.get(key, 0.05)))
    return refs


def _refine_point(
    start: tuple[float, float],
    refs: list[tuple[str, tuple[float, float], float, float]],
    *,
    iterations: int,
) -> tuple[float, float]:
    point = np.array(start, dtype=float)
    for _ in range(max(iterations, 0)):
        jacobian_rows: list[np.ndarray] = []
        residuals: list[float] = []
        for _anchor_id, ref_point, target, sigma in refs:
            ref = np.array(ref_point, dtype=float)
            diff = point - ref
            distance = max(float(np.linalg.norm(diff)), 1e-9)
            sigma = max(float(sigma), 0.02)
            residuals.append((distance - target) / sigma)
            jacobian_rows.append(diff / distance / sigma)
        if len(residuals) < 2:
            break
        jacobian = np.vstack(jacobian_rows)
        residual = np.array(residuals, dtype=float)
        lhs = jacobian.T @ jacobian + np.eye(2) * 1e-4
        rhs = -(jacobian.T @ residual)
        try:
            step = np.linalg.solve(lhs, rhs)
        except np.linalg.LinAlgError:
            break
        if float(np.linalg.norm(step)) > 2.0:
            step = step / max(float(np.linalg.norm(step)), 1e-9) * 2.0
        point = point + step
        if float(np.linalg.norm(step)) < 1e-5:
            break
    return float(point[0]), float(point[1])


def _unique_points(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    seen: set[tuple[int, int]] = set()
    out: list[tuple[float, float]] = []
    for x, y in points:
        key = (round(x * 10000), round(y * 10000))
        if key in seen:
            continue
        seen.add(key)
        out.append((float(x), float(y)))
    return out


def _position_candidates(
    anchor_id: str,
    positions: dict[str, tuple[float, float]],
    graph: RangeGraph,
    *,
    one_link_angles: int,
    point_refine_iterations: int,
) -> list[tuple[float, float]]:
    refs = _known_refs(anchor_id, positions, graph)
    if len(refs) >= 2:
        seeds: list[tuple[float, float]] = []
        for left, right in itertools.combinations(refs, 2):
            _left_id, left_point, left_distance, _left_sigma = left
            _right_id, right_point, right_distance, _right_sigma = right
            seeds.extend(circle_intersections(left_point, left_distance, right_point, right_distance))
        if len(refs) >= 3:
            centroid = np.mean(np.array([ref[1] for ref in refs], dtype=float), axis=0)
            seeds.append((float(centroid[0]), float(centroid[1])))
            seeds = [_refine_point(seed, refs, iterations=point_refine_iterations) for seed in seeds]
        return _unique_points(seeds)
    if len(refs) == 1:
        _other_id, center, radius, _sigma = refs[0]
        count = max(one_link_angles, 4)
        return [
            (center[0] + math.cos(2.0 * math.pi * step / count) * radius, center[1] + math.sin(2.0 * math.pi * step / count) * radius)
            for step in range(count)
        ]
    return []


def _visibility_score(
    anchor_a: str,
    point_a: tuple[float, float],
    anchor_b: str,
    point_b: tuple[float, float],
    graph: RangeGraph,
    *,
    radio_radius_m: float,
    missing_margin_m: float,
    missing_sigma_m: float,
    missing_weight: float,
    graph_upper_factor: float,
    graph_upper_slack_m: float,
    graph_upper_sigma_m: float,
    graph_upper_weight: float,
) -> float:
    distance = math.dist(point_a, point_b)
    key = pair_key(anchor_a, anchor_b)
    score = 0.0
    if key in graph.distances:
        sigma = max(graph.sigmas.get(key, 0.05), 0.02)
        residual = (distance - graph.distances[key]) / sigma
        score += residual * residual
    else:
        lower_bound = radio_radius_m + missing_margin_m
        if distance < lower_bound:
            residual = (lower_bound - distance) / max(missing_sigma_m, 1e-6)
            score += missing_weight * residual * residual
    i = graph.index[anchor_a]
    j = graph.index[anchor_b]
    shortest = graph.shortest_m[i, j]
    if shortest < INF * 0.5 and graph.hops[i, j] >= 2:
        upper_bound = shortest * graph_upper_factor + graph_upper_slack_m
        if distance > upper_bound:
            residual = (distance - upper_bound) / max(graph_upper_sigma_m, 1e-6)
            score += graph_upper_weight * residual * residual
    return score


def _candidate_score(anchor_id: str, point: tuple[float, float], positions: dict[str, tuple[float, float]], graph: RangeGraph, params: dict[str, float]) -> float:
    return sum(
        _visibility_score(anchor_id, point, other_id, other_point, graph, **params)
        for other_id, other_point in positions.items()
    )


def _next_anchor(unplaced: set[str], placed: set[str], graph: RangeGraph) -> str:
    def key(anchor_id: str) -> tuple[int, int, str]:
        links_to_placed = sum(1 for other in placed if pair_key(anchor_id, other) in graph.distances)
        tier = links_to_placed if links_to_placed >= 2 else links_to_placed - 10
        return -tier, -graph.degree[anchor_id], anchor_id

    return min(unplaced, key=key)


def visibility_branching_seed_layouts(
    known_pairs: list[AnchorPairDistance],
    *,
    beam_width: int = 32,
    radio_radius_m: float = 8.0,
    missing_margin_m: float = 0.0,
    missing_sigma_m: float = 0.75,
    missing_weight: float = 1.0,
    graph_upper_factor: float = 1.0,
    graph_upper_slack_m: float = 0.75,
    graph_upper_sigma_m: float = 1.0,
    graph_upper_weight: float = 0.35,
    one_link_angles: int = 16,
    point_refine_iterations: int = 12,
) -> list[dict[str, tuple[float, float]]]:
    graph = range_graph(known_pairs)
    triangle = _starting_triangle(graph)
    if triangle is None:
        return [dc.classical_mds_seed(known_pairs)]
    partials = [PartialLayout(_triangle_positions(triangle, graph), 0.0)]
    score_params = {
        "radio_radius_m": radio_radius_m,
        "missing_margin_m": missing_margin_m,
        "missing_sigma_m": missing_sigma_m,
        "missing_weight": missing_weight,
        "graph_upper_factor": graph_upper_factor,
        "graph_upper_slack_m": graph_upper_slack_m,
        "graph_upper_sigma_m": graph_upper_sigma_m,
        "graph_upper_weight": graph_upper_weight,
    }
    while len(partials[0].positions) < len(graph.anchor_ids):
        placed = set(partials[0].positions)
        unplaced = set(graph.anchor_ids) - placed
        anchor_id = _next_anchor(unplaced, placed, graph)
        expanded: list[PartialLayout] = []
        for partial in partials:
            candidates = _position_candidates(
                anchor_id,
                partial.positions,
                graph,
                one_link_angles=one_link_angles,
                point_refine_iterations=point_refine_iterations,
            )
            for point in candidates:
                score = partial.score + _candidate_score(anchor_id, point, partial.positions, graph, score_params)
                positions = dict(partial.positions)
                positions[anchor_id] = point
                expanded.append(PartialLayout(positions, score))
        if not expanded:
            return [dc.classical_mds_seed(known_pairs)]
        expanded.sort(key=lambda partial: partial.score)
        partials = expanded[: max(1, beam_width)]
    return [partial.positions for partial in partials]


def visibility_score_all(
    positions: dict[str, tuple[float, float]],
    known_pairs: list[AnchorPairDistance],
    *,
    radio_radius_m: float = 8.0,
    missing_margin_m: float = 0.0,
    missing_sigma_m: float = 0.75,
    missing_weight: float = 1.0,
    graph_upper_factor: float = 1.0,
    graph_upper_slack_m: float = 0.75,
    graph_upper_sigma_m: float = 1.0,
    graph_upper_weight: float = 0.35,
) -> float:
    graph = range_graph(known_pairs)
    score_params = {
        "radio_radius_m": radio_radius_m,
        "missing_margin_m": missing_margin_m,
        "missing_sigma_m": missing_sigma_m,
        "missing_weight": missing_weight,
        "graph_upper_factor": graph_upper_factor,
        "graph_upper_slack_m": graph_upper_slack_m,
        "graph_upper_sigma_m": graph_upper_sigma_m,
        "graph_upper_weight": graph_upper_weight,
    }
    score = 0.0
    count = 0
    for index, anchor_a in enumerate(graph.anchor_ids):
        if anchor_a not in positions:
            continue
        for anchor_b in graph.anchor_ids[index + 1 :]:
            if anchor_b not in positions:
                continue
            score += _visibility_score(anchor_a, positions[anchor_a], anchor_b, positions[anchor_b], graph, **score_params)
            count += 1
    return score / max(count, 1)


def visibility_branching_solve(
    known_pairs: list[AnchorPairDistance],
    *,
    beam_width: int = 32,
    optimizer_seeds: int = 32,
    iterations: int = 45,
    radio_radius_m: float = 8.0,
    final_visibility_weight: float = 1.0,
    constrained_polish: bool = False,
    constrained_iterations: int | None = None,
    constrained_known_weight: float = 1.0,
    **visibility_params: Any,
) -> dict[str, tuple[float, float]]:
    seed_params = {"radio_radius_m": radio_radius_m, **visibility_params}
    seeds = visibility_branching_seed_layouts(known_pairs, beam_width=beam_width, **seed_params)
    score_keys = {
        "missing_margin_m",
        "missing_sigma_m",
        "missing_weight",
        "graph_upper_factor",
        "graph_upper_slack_m",
        "graph_upper_sigma_m",
        "graph_upper_weight",
    }
    score_params = {key: value for key, value in visibility_params.items() if key in score_keys}
    best_positions: dict[str, tuple[float, float]] | None = None
    best_score = math.inf
    for seed in seeds[: max(1, optimizer_seeds)]:
        try:
            if constrained_polish:
                positions = visibility_constrained_solve_from_seed(
                    seed,
                    known_pairs,
                    max_iterations=constrained_iterations or iterations,
                    radio_radius_m=radio_radius_m,
                    known_weight=constrained_known_weight,
                    **score_params,
                )
            else:
                positions = dc.solve_from_seed(seed, known_pairs, max_iterations=iterations)
        except Exception:
            positions = seed
        known_rmse, _known_max = dc.pair_metrics(positions, known_pairs)
        visibility = visibility_score_all(positions, known_pairs, radio_radius_m=radio_radius_m, **score_params)
        score = known_rmse * known_rmse + final_visibility_weight * visibility
        if score < best_score:
            best_score = score
            best_positions = positions
    if best_positions is None:
        return dc.known_only_solution(known_pairs, max_iterations=iterations)
    return best_positions


def _relaxed_visibility_seed(
    known_pairs: list[AnchorPairDistance],
    *,
    iterations: int,
    step_size: float,
    radio_radius_m: float,
    known_weight: float,
    missing_weight: float,
    graph_upper_weight: float,
    graph_upper_factor: float,
    graph_upper_slack_m: float,
) -> dict[str, tuple[float, float]]:
    graph = range_graph(known_pairs)
    seed = dc.classical_mds_seed(known_pairs)
    coords = np.array([seed[anchor_id] for anchor_id in graph.anchor_ids], dtype=float)
    coords -= coords.mean(axis=0, keepdims=True)
    for iteration in range(max(iterations, 0)):
        grad = np.zeros_like(coords)
        for i, anchor_a in enumerate(graph.anchor_ids):
            for j in range(i + 1, len(graph.anchor_ids)):
                anchor_b = graph.anchor_ids[j]
                diff = coords[i] - coords[j]
                distance = max(float(np.linalg.norm(diff)), 1e-9)
                direction = diff / distance
                key = pair_key(anchor_a, anchor_b)
                if key in graph.distances:
                    sigma = max(graph.sigmas.get(key, 0.05), 0.02)
                    residual = distance - graph.distances[key]
                    weight = known_weight / (sigma * sigma)
                    force = weight * residual * direction
                    grad[i] += force
                    grad[j] -= force
                else:
                    if distance < radio_radius_m:
                        residual = distance - radio_radius_m
                        force = missing_weight * residual * direction
                        grad[i] += force
                        grad[j] -= force
                shortest = graph.shortest_m[i, j]
                if shortest < INF * 0.5 and graph.hops[i, j] >= 2:
                    upper_bound = shortest * graph_upper_factor + graph_upper_slack_m
                    if distance > upper_bound:
                        residual = distance - upper_bound
                        force = graph_upper_weight * residual * direction
                        grad[i] += force
                        grad[j] -= force
        grad -= grad.mean(axis=0, keepdims=True)
        norm = max(float(np.linalg.norm(grad) / max(len(graph.anchor_ids), 1)), 1e-9)
        lr = step_size / (1.0 + 0.015 * iteration)
        coords -= lr * grad / max(norm, 1.0)
    return {anchor_id: (float(coords[i, 0]), float(coords[i, 1])) for i, anchor_id in enumerate(graph.anchor_ids)}


def visibility_relaxed_solve(
    known_pairs: list[AnchorPairDistance],
    *,
    relax_iterations: int = 600,
    relax_step_size: float = 0.025,
    iterations: int = 45,
    radio_radius_m: float = 8.0,
    known_weight: float = 1.0,
    missing_weight: float = 0.3,
    graph_upper_weight: float = 0.1,
    graph_upper_factor: float = 1.0,
    graph_upper_slack_m: float = 0.75,
    missing_sigma_m: float = 0.75,
    graph_upper_sigma_m: float = 1.0,
    constrained_polish: bool = False,
    constrained_iterations: int | None = None,
) -> dict[str, tuple[float, float]]:
    seed = _relaxed_visibility_seed(
        known_pairs,
        iterations=relax_iterations,
        step_size=relax_step_size,
        radio_radius_m=radio_radius_m,
        known_weight=known_weight,
        missing_weight=missing_weight,
        graph_upper_weight=graph_upper_weight,
        graph_upper_factor=graph_upper_factor,
        graph_upper_slack_m=graph_upper_slack_m,
    )
    if constrained_polish:
        return visibility_constrained_solve_from_seed(
            seed,
            known_pairs,
            max_iterations=constrained_iterations or iterations,
            radio_radius_m=radio_radius_m,
            missing_sigma_m=missing_sigma_m,
            missing_weight=missing_weight,
            graph_upper_factor=graph_upper_factor,
            graph_upper_slack_m=graph_upper_slack_m,
            graph_upper_sigma_m=graph_upper_sigma_m,
            graph_upper_weight=graph_upper_weight,
            known_weight=known_weight,
        )
    return dc.solve_from_seed(seed, known_pairs, max_iterations=iterations)

def _sdp_distance_sq(g_var: Any, i: int, j: int) -> Any:
    return g_var[i, i] + g_var[j, j] - 2.0 * g_var[i, j]


def visibility_sdp_seed(
    known_pairs: list[AnchorPairDistance],
    *,
    radio_radius_m: float = 8.0,
    missing_margin_m: float = 0.0,
    known_weight: float = 1.0,
    missing_weight: float = 0.35,
    graph_upper_weight: float = 0.15,
    graph_upper_factor: float = 1.0,
    graph_upper_slack_m: float = 0.75,
    solver: str = "SCS",
    max_iters: int = 4000,
    eps: float = 1e-4,
) -> dict[str, tuple[float, float]]:
    try:
        import cvxpy as cp  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on optional dependency.
        raise RuntimeError("visibility_sdp requires the optional cvxpy dependency. Install requirements.txt first.") from exc

    graph = range_graph(known_pairs)
    n = len(graph.anchor_ids)
    gram = cp.Variable((n, n), PSD=True)
    constraints = [cp.sum(gram, axis=0) == 0.0]
    objective_terms: list[Any] = []

    for i, anchor_a in enumerate(graph.anchor_ids):
        for j in range(i + 1, n):
            anchor_b = graph.anchor_ids[j]
            key = pair_key(anchor_a, anchor_b)
            distance_sq = _sdp_distance_sq(gram, i, j)
            if key in graph.distances:
                measured = graph.distances[key]
                sigma = max(graph.sigmas.get(key, 0.05), 0.02)
                squared_sigma = max(2.0 * measured * sigma, 0.05)
                objective_terms.append(known_weight * cp.square((distance_sq - measured * measured) / squared_sigma))
            else:
                lower = radio_radius_m + missing_margin_m
                slack = cp.Variable(nonneg=True)
                constraints.append(distance_sq + slack >= lower * lower)
                objective_terms.append(missing_weight * cp.square(slack / max(lower * lower, 1.0)))
            shortest = graph.shortest_m[i, j]
            if shortest < INF * 0.5 and graph.hops[i, j] >= 2:
                upper = shortest * graph_upper_factor + graph_upper_slack_m
                slack = cp.Variable(nonneg=True)
                constraints.append(distance_sq <= upper * upper + slack)
                objective_terms.append(graph_upper_weight * cp.square(slack / max(upper * upper, 1.0)))

    problem = cp.Problem(cp.Minimize(cp.sum(objective_terms)), constraints)
    solve_kwargs: dict[str, Any] = {"solver": solver}
    if solver.upper() == "SCS":
        solve_kwargs.update({"max_iters": max_iters, "eps": eps, "verbose": False})
    problem.solve(**solve_kwargs)
    if problem.status not in {"optimal", "optimal_inaccurate"} or gram.value is None:
        raise RuntimeError(f"visibility_sdp failed with status={problem.status!r}")

    gram_value = np.asarray(gram.value, dtype=float)
    gram_value = 0.5 * (gram_value + gram_value.T)
    values, vectors = np.linalg.eigh(gram_value)
    order = np.argsort(values)[::-1][:2]
    values = np.maximum(values[order], 0.0)
    coords = vectors[:, order] * np.sqrt(values).reshape(1, -1)
    coords -= coords.mean(axis=0, keepdims=True)
    return {anchor_id: (float(coords[i, 0]), float(coords[i, 1])) for i, anchor_id in enumerate(graph.anchor_ids)}


def visibility_sdp_solve(
    known_pairs: list[AnchorPairDistance],
    *,
    iterations: int = 45,
    radio_radius_m: float = 8.0,
    missing_margin_m: float = 0.0,
    known_weight: float = 1.0,
    missing_weight: float = 0.35,
    graph_upper_weight: float = 0.15,
    graph_upper_factor: float = 1.0,
    graph_upper_slack_m: float = 0.75,
    missing_sigma_m: float = 0.75,
    graph_upper_sigma_m: float = 1.0,
    constrained_polish: bool = False,
    constrained_iterations: int | None = None,
    sdp_solver: str = "SCS",
    sdp_max_iters: int = 4000,
    sdp_eps: float = 1e-4,
) -> dict[str, tuple[float, float]]:
    seed = visibility_sdp_seed(
        known_pairs,
        radio_radius_m=radio_radius_m,
        missing_margin_m=missing_margin_m,
        known_weight=known_weight,
        missing_weight=missing_weight,
        graph_upper_weight=graph_upper_weight,
        graph_upper_factor=graph_upper_factor,
        graph_upper_slack_m=graph_upper_slack_m,
        solver=sdp_solver,
        max_iters=sdp_max_iters,
        eps=sdp_eps,
    )
    if constrained_polish:
        return visibility_constrained_solve_from_seed(
            seed,
            known_pairs,
            max_iterations=constrained_iterations or iterations,
            radio_radius_m=radio_radius_m,
            missing_margin_m=missing_margin_m,
            missing_sigma_m=missing_sigma_m,
            missing_weight=missing_weight,
            graph_upper_factor=graph_upper_factor,
            graph_upper_slack_m=graph_upper_slack_m,
            graph_upper_sigma_m=graph_upper_sigma_m,
            graph_upper_weight=graph_upper_weight,
            known_weight=known_weight,
        )
    return dc.solve_from_seed(seed, known_pairs, max_iterations=iterations)

def _canonical_seed_positions(
    seed_positions: dict[str, tuple[float, float]],
    anchor_ids: tuple[str, ...],
    known_pairs: list[AnchorPairDistance],
) -> dict[str, tuple[float, float]]:
    if any(anchor_id not in seed_positions for anchor_id in anchor_ids):
        seed_positions = dc.classical_mds_seed(known_pairs)
    try:
        return rotate_layout_to_level(seed_positions, anchor_ids[0], anchor_ids[1])
    except Exception:
        ax, ay = seed_positions.get(anchor_ids[0], (0.0, 0.0))
        return {anchor_id: (seed_positions[anchor_id][0] - ax, seed_positions[anchor_id][1] - ay) for anchor_id in anchor_ids}


def visibility_constrained_solve_from_seed(
    seed_positions: dict[str, tuple[float, float]],
    known_pairs: list[AnchorPairDistance],
    *,
    max_iterations: int,
    radio_radius_m: float = 8.0,
    missing_margin_m: float = 0.0,
    missing_sigma_m: float = 0.75,
    missing_weight: float = 1.0,
    graph_upper_factor: float = 1.0,
    graph_upper_slack_m: float = 0.75,
    graph_upper_sigma_m: float = 1.0,
    graph_upper_weight: float = 0.35,
    known_weight: float = 1.0,
) -> dict[str, tuple[float, float]]:
    try:
        from scipy.optimize import least_squares
    except Exception as exc:  # pragma: no cover - scipy is in requirements.
        raise RuntimeError("visibility-constrained polish requires scipy") from exc

    graph = range_graph(known_pairs)
    processed = _preprocess_pairs(known_pairs, min_sigma_m=0.02, min_distance_m=0.05)
    anchor_ids = graph.anchor_ids
    parameterization = _Parameterization(list(anchor_ids))
    seed_positions = _canonical_seed_positions(seed_positions, anchor_ids, known_pairs)
    x0 = np.asarray(_positions_to_params(parameterization, seed_positions), dtype=float)
    if x0.size == 0:
        return seed_positions
    sqrt_known = math.sqrt(max(known_weight, 0.0))
    sqrt_missing = math.sqrt(max(missing_weight, 0.0))
    sqrt_graph = math.sqrt(max(graph_upper_weight, 0.0))
    lower_bound = radio_radius_m + missing_margin_m

    def residuals(params: np.ndarray) -> np.ndarray:
        positions = parameterization.to_positions(params.tolist())
        values: list[float] = []
        if sqrt_known > 0.0:
            for pair in processed:
                ax, ay = positions[pair.anchor_a_id]
                bx, by = positions[pair.anchor_b_id]
                distance = math.hypot(ax - bx, ay - by)
                sigma = max(pair.sigma_m, 0.02)
                values.append(sqrt_known * (distance - pair.distance_m) / sigma)
        for i, anchor_a in enumerate(anchor_ids):
            ax, ay = positions[anchor_a]
            for j in range(i + 1, len(anchor_ids)):
                anchor_b = anchor_ids[j]
                bx, by = positions[anchor_b]
                distance = math.hypot(ax - bx, ay - by)
                key = pair_key(anchor_a, anchor_b)
                if key not in graph.distances and sqrt_missing > 0.0:
                    violation = lower_bound - distance
                    if violation > 0.0:
                        values.append(sqrt_missing * violation / max(missing_sigma_m, 1e-6))
                    else:
                        values.append(0.0)
                shortest = graph.shortest_m[i, j]
                if shortest < INF * 0.5 and graph.hops[i, j] >= 2 and sqrt_graph > 0.0:
                    upper_bound = shortest * graph_upper_factor + graph_upper_slack_m
                    violation = distance - upper_bound
                    if violation > 0.0:
                        values.append(sqrt_graph * violation / max(graph_upper_sigma_m, 1e-6))
                    else:
                        values.append(0.0)
        return np.asarray(values, dtype=float)

    result = least_squares(
        residuals,
        x0,
        max_nfev=max(1, int(max_iterations)),
        method="trf",
        x_scale="jac",
    )
    params = result.x if result.x is not None else x0
    positions = parameterization.to_positions(params.tolist())
    try:
        return rotate_layout_to_level(positions, anchor_ids[0], anchor_ids[1])
    except Exception:
        return positions
