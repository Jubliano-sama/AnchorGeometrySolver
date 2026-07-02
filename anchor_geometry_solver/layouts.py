from __future__ import annotations

import random
from typing import Iterable

import numpy as np
import torch

from anchor_geometry_solver.compat import ensure_legacy_paths
from anchor_geometry_solver.types import CaseContext, LayoutSpec

ensure_legacy_paths()
import anchor_solver_ml_distance_completion as dc  # noqa: E402
import anchor_solver_p95_ml_cases as p95  # noqa: E402


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)


def _case_context(case: p95.CaseSpec, case_index: int, device: torch.device) -> CaseContext:
    batch = p95.graph_batch_from_cases([case], device)
    truth = dc.graph_to_truth(batch, 0)
    known_pairs = tuple(dc.known_pairs_from_batch(batch, 0))
    diagnostics = dc.measured_graph_diagnostics(case.points)
    return CaseContext(
        bucket=case.bucket,
        case_index=case_index,
        family=case.family,
        shape=case.shape,
        truth=truth,
        known_pairs=known_pairs,
        diagnostics=diagnostics,
    )


def _within_node_limits(points: torch.Tensor, spec: LayoutSpec) -> bool:
    n = int(points.shape[0])
    if spec.min_nodes is not None and n < spec.min_nodes:
        return False
    if spec.max_nodes is not None and n > spec.max_nodes:
        return False
    return spec.count_range[0] <= n <= max(spec.count_range[1], spec.count_range[0])


def _sample_params(shape: str, spec: LayoutSpec) -> dict[str, float]:
    width = random.uniform(*spec.width_range_m)
    height = random.uniform(*spec.height_range_m)
    params: dict[str, float] = {"width": width, "height": height}
    if shape in {"rectangle", "corridor"}:
        return params
    if shape in {"disc", "annulus", "hollow_square"}:
        side = min(width, height)
        params = {"width": side, "height": side}
    if shape == "l_shape":
        params["leg_x"] = params["width"] * random.uniform(0.34, 0.58)
        params["leg_y"] = params["height"] * random.uniform(0.34, 0.58)
    elif shape == "t_shape":
        params["bar_w"] = params["width"] * random.uniform(0.24, 0.42)
        params["top_h"] = params["height"] * random.uniform(0.26, 0.42)
    elif shape == "u_shape":
        params["leg_w"] = params["width"] * random.uniform(0.22, 0.35)
        params["bottom_h"] = params["height"] * random.uniform(0.24, 0.42)
    elif shape == "cross":
        params["bar_w"] = params["width"] * random.uniform(0.22, 0.38)
        params["bar_h"] = params["height"] * random.uniform(0.22, 0.38)
    elif shape == "hollow_square":
        params["hole_w"] = params["width"] * random.uniform(0.26, 0.52)
        params["hole_h"] = params["height"] * random.uniform(0.26, 0.52)
    elif shape == "disc":
        params["radius"] = params["width"] * 0.50
    elif shape == "annulus":
        params["outer_radius"] = params["width"] * 0.50
        params["inner_radius"] = params["width"] * random.uniform(0.18, 0.34)
    elif shape == "rooms":
        params["corridor_w"] = params["width"] * random.uniform(0.16, 0.28)
        params["corridor_h"] = params["height"] * random.uniform(0.16, 0.28)
    else:
        raise ValueError(f"Unsupported generated shape {shape!r}")
    return params


def _generate_custom_case(spec: LayoutSpec, device: torch.device) -> p95.CaseSpec:
    shapes = spec.shapes or ("rectangle",)
    family = spec.family or ("grid" if spec.bucket == "grid" else "random")
    for _attempt in range(5000):
        shape = random.choice(shapes)
        params = _sample_params(shape, spec)
        target_n = random.randint(*spec.count_range)
        if family == "grid":
            points = dc.grid_points_in_shape(shape, params, target_n, device)
        elif family == "random":
            points = dc.random_points_in_shape(shape, params, target_n, device)
        else:
            raise ValueError(f"Unsupported layout family {family!r}; use 'random' or 'grid'.")
        if points is None:
            continue
        points = p95.fair_points(
            points,
            min_nodes=spec.min_nodes or spec.count_range[0],
            max_nodes=spec.max_nodes or spec.count_range[1],
            min_vertex_connectivity=spec.min_vertex_connectivity,
        )
        if points is not None and _within_node_limits(points, spec):
            return p95.CaseSpec(spec.name, family, shape, points)
    raise RuntimeError(f"Could not generate a fair case for layout {spec.name!r}")


def _generate_bucket_case(spec: LayoutSpec, device: torch.device) -> p95.CaseSpec:
    if spec.shapes or spec.family or spec.min_nodes is not None or spec.max_nodes is not None:
        return _generate_custom_case(spec, device)
    generator = {
        "random": p95.generate_random_rectangle,
        "grid": p95.generate_grid_rectangle,
        "office": p95.generate_office_shape,
    }.get(spec.bucket)
    if generator is None:
        return _generate_custom_case(spec, device)
    return generator(device)


def generate_case_contexts(layouts: Iterable[LayoutSpec], *, seed: int, device_name: str = "cpu") -> list[CaseContext]:
    seed_everything(seed)
    device = torch.device(device_name)
    contexts: list[CaseContext] = []
    for layout in layouts:
        for _ in range(layout.cases):
            case = _generate_bucket_case(layout, device)
            contexts.append(_case_context(case, len(contexts), device))
    return contexts
