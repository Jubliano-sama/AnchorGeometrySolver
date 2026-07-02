from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Any

from anchor_geometry_solver.types import BenchmarkConfig, LayoutSpec, MethodSpec, RunnerSpec


def _tuple(value: Any, default: tuple[Any, ...]) -> tuple[Any, ...]:
    if value is None:
        return default
    if isinstance(value, tuple):
        return value
    if isinstance(value, list):
        return tuple(value)
    return (value,)


def _range_tuple(value: Any, default: tuple[float, float]) -> tuple[float, float]:
    if value is None:
        return default
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"Expected a two-value range, got {value!r}")
    return float(value[0]), float(value[1])


def _int_range_tuple(value: Any, default: tuple[int, int]) -> tuple[int, int]:
    low, high = _range_tuple(value, (float(default[0]), float(default[1])))
    return int(low), int(high)


def load_mapping(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8-sig")
    if path.suffix.lower() == ".json":
        return json.loads(text)
    try:
        import yaml  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on optional dependency.
        raise RuntimeError("YAML configs require PyYAML; install pyyaml or use JSON.") from exc
    loaded = yaml.safe_load(text)
    if not isinstance(loaded, dict):
        raise ValueError(f"Config {path} did not contain a mapping.")
    return loaded


def parse_layout(raw: dict[str, Any], *, default_cases: int) -> LayoutSpec:
    name = str(raw.get("name") or raw.get("bucket") or "layout")
    return LayoutSpec(
        name=name,
        bucket=str(raw.get("bucket", name)),
        cases=int(raw.get("cases", default_cases)),
        family=raw.get("family"),
        shapes=tuple(str(item) for item in _tuple(raw.get("shapes") or raw.get("shape"), ())),
        count_range=_int_range_tuple(raw.get("count_range"), (16, 32)),
        width_range_m=_range_tuple(raw.get("width_range_m") or raw.get("width_m"), (13.0, 25.0)),
        height_range_m=_range_tuple(raw.get("height_range_m") or raw.get("height_m"), (13.0, 25.0)),
        min_nodes=int(raw["min_nodes"]) if "min_nodes" in raw else None,
        max_nodes=int(raw["max_nodes"]) if "max_nodes" in raw else None,
        min_vertex_connectivity=int(raw.get("min_vertex_connectivity", 1)),
    )


def _expanded_method_name(base: str, params: dict[str, Any]) -> str:
    if not params:
        return base
    suffix = ",".join(f"{key}={params[key]}" for key in sorted(params))
    return f"{base}[{suffix}]"


def parse_methods(raw_methods: list[dict[str, Any]]) -> tuple[MethodSpec, ...]:
    methods: list[MethodSpec] = []
    for raw in raw_methods:
        base_name = str(raw["name"])
        solver = str(raw.get("solver", base_name))
        base_params = dict(raw.get("params", {}))
        grid = raw.get("grid", {})
        if not grid:
            methods.append(MethodSpec(base_name, solver, base_params))
            continue
        keys = list(grid)
        values = [list(grid[key]) for key in keys]
        for combo in itertools.product(*values):
            params = {**base_params, **dict(zip(keys, combo))}
            methods.append(MethodSpec(_expanded_method_name(base_name, params), solver, params))
    return tuple(methods)


def parse_config(path: Path) -> BenchmarkConfig:
    raw = load_mapping(path)
    benchmark = raw.get("benchmark", {})
    default_cases = int(benchmark.get("cases_per_bucket", raw.get("cases_per_bucket", 1)))
    runner_raw = raw.get("runner", {})
    output_dir = Path(runner_raw.get("output_dir", "outputs"))
    if not output_dir.is_absolute():
        output_dir = Path.cwd() / output_dir
    return BenchmarkConfig(
        name=str(benchmark.get("name", path.stem)),
        seed=int(benchmark.get("seed", raw.get("seed", 1337))),
        device=str(benchmark.get("device", raw.get("device", "cpu"))),
        layouts=tuple(parse_layout(item, default_cases=default_cases) for item in raw.get("layouts", [])),
        methods=parse_methods(list(raw.get("methods", []))),
        runner=RunnerSpec(
            backend=str(runner_raw.get("backend", "serial")),
            workers=int(runner_raw.get("workers", 1)),
            fold_threshold_m=float(runner_raw.get("fold_threshold_m", 1.65)),
            output_dir=output_dir,
            write_outputs=bool(runner_raw.get("write_outputs", True)),
        ),
    )
