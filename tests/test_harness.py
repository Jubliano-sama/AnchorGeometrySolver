from __future__ import annotations

import json
import math
import unittest
from pathlib import Path

from anchor_geometry_solver.benchmark import make_jobs, run_benchmark, run_jobs, summarize_rows
from anchor_geometry_solver.compat import AnchorPairDistance
from anchor_geometry_solver.config import parse_config
from anchor_geometry_solver.layouts import generate_case_contexts
from anchor_geometry_solver.types import BenchmarkConfig, CaseContext, LayoutSpec, MethodSpec, RunnerSpec


ROOT = Path(__file__).resolve().parents[1]
TMP_ROOT = ROOT / "outputs" / "test_runs"


def output_dir(name: str) -> Path:
    path = TMP_ROOT / name
    path.mkdir(parents=True, exist_ok=True)
    return path


class HarnessTest(unittest.TestCase):
    def test_config_expands_method_grid(self) -> None:
        path = output_dir("config_expand") / "config.json"
        path.write_text(
            json.dumps(
                {
                    "benchmark": {"name": "grid_expand", "seed": 12, "cases_per_bucket": 1},
                    "layouts": [{"name": "grid", "bucket": "grid"}],
                    "methods": [
                        {
                            "name": "graph",
                            "solver": "graph_shortest",
                            "params": {"iterations": 5},
                            "grid": {"max_hops": [2, 3], "scaffold_weight": [1.0, 2.0]},
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )

        config = parse_config(path)

        self.assertEqual(config.name, "grid_expand")
        self.assertEqual(len(config.layouts), 1)
        self.assertEqual(len(config.methods), 4)
        self.assertTrue(all(method.params["iterations"] == 5 for method in config.methods))
        self.assertEqual({method.params["max_hops"] for method in config.methods}, {2, 3})

    def test_method_registry_solves_triangle_context(self) -> None:
        truth = {"A00": (0.0, 0.0), "A01": (4.0, 0.0), "A02": (0.0, 3.0)}
        pairs = (
            AnchorPairDistance("A00", "A01", 4.0, sigma_m=0.03, source="known"),
            AnchorPairDistance("A00", "A02", 3.0, sigma_m=0.03, source="known"),
            AnchorPairDistance("A01", "A02", 5.0, sigma_m=0.03, source="known"),
        )
        context = CaseContext("triangle", 0, "manual", "triangle", truth, pairs, {"min_degree": 2.0})
        methods = (
            MethodSpec("triangulated", "known_triangulated", {"iterations": 20}),
            MethodSpec("graph", "graph_shortest", {"iterations": 20, "seed_count": 1, "max_hops": 2}),
        )
        config = BenchmarkConfig(
            "triangle_smoke",
            seed=123,
            layouts=(),
            methods=methods,
            runner=RunnerSpec(write_outputs=False),
        )

        rows = run_jobs(make_jobs(config, [context]), backend="serial", workers=1)
        summary = summarize_rows(rows)

        self.assertEqual(len(rows), 2)
        self.assertTrue(all(row.status == "ok" for row in rows))
        self.assertTrue(all(math.isfinite(row.max_offset_m) for row in rows))
        self.assertEqual(len(summary), 2)

    def test_generated_contexts_keep_fair_grid_constraints(self) -> None:
        contexts = generate_case_contexts(
            [LayoutSpec(name="grid", bucket="grid", cases=1)],
            seed=2026070202,
            device_name="cpu",
        )

        self.assertEqual(len(contexts), 1)
        context = contexts[0]
        self.assertGreaterEqual(context.anchor_count, 16)
        self.assertGreaterEqual(context.diagnostics["min_degree"], 3.0)
        self.assertGreaterEqual(context.known_pair_count, context.anchor_count)

    def test_run_benchmark_writes_detail_and_summary(self) -> None:
        tmp = output_dir("benchmark_write")
        config = BenchmarkConfig(
            name="tiny_harness",
            seed=2026070203,
            layouts=(LayoutSpec(name="grid", bucket="grid", cases=1),),
            methods=(MethodSpec("triangulated", "known_triangulated", {"iterations": 5}),),
            runner=RunnerSpec(output_dir=tmp, write_outputs=True),
        )

        rows, summary = run_benchmark(config)

        self.assertEqual(len(rows), 1)
        self.assertEqual(len(summary), 1)
        self.assertEqual(rows[0].status, "ok")
        self.assertTrue((tmp / "tiny_harness" / "detail.csv").exists())
        self.assertTrue((tmp / "tiny_harness" / "summary.csv").exists())


if __name__ == "__main__":
    unittest.main()