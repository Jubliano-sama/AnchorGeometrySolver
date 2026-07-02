from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from anchor_geometry_solver.compat import AnchorPairDistance


PositionMap = dict[str, tuple[float, float]]


@dataclass(frozen=True)
class LayoutSpec:
    """A generated layout bucket or a concrete generated-shape request."""

    name: str
    bucket: str = "random"
    cases: int = 1
    family: str | None = None
    shapes: tuple[str, ...] = ()
    count_range: tuple[int, int] = (16, 32)
    width_range_m: tuple[float, float] = (13.0, 25.0)
    height_range_m: tuple[float, float] = (13.0, 25.0)
    min_nodes: int | None = None
    max_nodes: int | None = None
    min_vertex_connectivity: int = 1


@dataclass(frozen=True)
class MethodSpec:
    """One solver/seed pipeline configuration."""

    name: str
    solver: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RunnerSpec:
    backend: str = "serial"
    workers: int = 1
    fold_threshold_m: float = 1.65
    output_dir: Path = Path("outputs")
    write_outputs: bool = True


@dataclass(frozen=True)
class BenchmarkConfig:
    name: str
    seed: int
    layouts: tuple[LayoutSpec, ...]
    methods: tuple[MethodSpec, ...]
    runner: RunnerSpec = RunnerSpec()
    device: str = "cpu"


@dataclass(frozen=True)
class CaseContext:
    bucket: str
    case_index: int
    family: str
    shape: str
    truth: PositionMap
    known_pairs: tuple[AnchorPairDistance, ...]
    diagnostics: dict[str, float]

    @property
    def anchor_count(self) -> int:
        return len(self.truth)

    @property
    def known_pair_count(self) -> int:
        return len(self.known_pairs)


@dataclass(frozen=True)
class BenchmarkRow:
    benchmark: str
    bucket: str
    case_index: int
    family: str
    shape: str
    method: str
    solver: str
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
    runtime_s: float
    status: str
    error: str = ""
    params: dict[str, Any] = field(default_factory=dict)
    diagnostics: dict[str, float] = field(default_factory=dict)

    def to_flat_dict(self) -> dict[str, Any]:
        row: dict[str, Any] = {
            "benchmark": self.benchmark,
            "bucket": self.bucket,
            "case_index": self.case_index,
            "family": self.family,
            "shape": self.shape,
            "method": self.method,
            "solver": self.solver,
            "anchors": self.anchors,
            "known_pairs": self.known_pairs,
            "max_offset_m": self.max_offset_m,
            "median_offset_m": self.median_offset_m,
            "p95_offset_m": self.p95_offset_m,
            "known_rmse_m": self.known_rmse_m,
            "known_max_residual_m": self.known_max_residual_m,
            "min_pair_distance_m": self.min_pair_distance_m,
            "close_pair_count": self.close_pair_count,
            "fold_cluster_count": self.fold_cluster_count,
            "runtime_s": self.runtime_s,
            "status": self.status,
            "error": self.error,
        }
        for key, value in self.params.items():
            row[f"param_{key}"] = value
        for key, value in self.diagnostics.items():
            row[f"diag_{key}"] = value
        return row
