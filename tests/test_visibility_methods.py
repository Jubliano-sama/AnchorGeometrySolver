from __future__ import annotations

import importlib.util
import math
import unittest

from anchor_geometry_solver.benchmark import make_jobs, run_jobs
from anchor_geometry_solver.compat import AnchorPairDistance, ensure_legacy_paths
from anchor_geometry_solver.types import BenchmarkConfig, CaseContext, MethodSpec, RunnerSpec
from anchor_geometry_solver.visibility import visibility_branching_seed_layouts, visibility_score_all

ensure_legacy_paths()
import anchor_solver_ml_distance_completion as dc  # noqa: E402


def folded_intersection_fixture() -> tuple[dict[str, tuple[float, float]], tuple[AnchorPairDistance, ...]]:
    height = math.sqrt(27.0)
    truth = {
        "A00": (0.0, 0.0),
        "A01": (6.0, 0.0),
        "A02": (3.0, height),
        "A03": (9.0, height),
    }
    pairs = (
        AnchorPairDistance("A00", "A01", 6.0, sigma_m=0.03, source="known"),
        AnchorPairDistance("A00", "A02", 6.0, sigma_m=0.03, source="known"),
        AnchorPairDistance("A01", "A02", 6.0, sigma_m=0.03, source="known"),
        AnchorPairDistance("A01", "A03", 6.0, sigma_m=0.03, source="known"),
        AnchorPairDistance("A02", "A03", 6.0, sigma_m=0.03, source="known"),
    )
    return truth, pairs


class VisibilityMethodsTest(unittest.TestCase):
    def test_branching_seed_keeps_visible_consistent_circle_intersection(self) -> None:
        truth, pairs = folded_intersection_fixture()
        seeds = visibility_branching_seed_layouts(
            list(pairs),
            beam_width=1,
            radio_radius_m=8.0,
            missing_weight=8.0,
            graph_upper_weight=0.0,
        )

        self.assertEqual(len(seeds), 1)
        max_offset, _median_offset, _p95_offset = dc.offset_summary(truth, seeds[0])
        self.assertLess(max_offset, 0.35)

    def test_missing_visibility_penalizes_folded_layout(self) -> None:
        truth, pairs = folded_intersection_fixture()
        folded = dict(truth)
        folded["A03"] = truth["A00"]

        good_score = visibility_score_all(truth, list(pairs), radio_radius_m=8.0, missing_weight=8.0)
        folded_score = visibility_score_all(folded, list(pairs), radio_radius_m=8.0, missing_weight=8.0)

        self.assertGreater(folded_score, good_score + 1.0)

    def test_harness_visibility_branching_solver_with_both_polishes(self) -> None:
        truth, pairs = folded_intersection_fixture()
        context = CaseContext("manual", 0, "manual", "folded_intersection", truth, pairs, {"min_degree": 2.0})
        methods = (
            MethodSpec(
                "visibility_branching",
                "visibility_branching",
                {"iterations": 25, "beam_width": 4, "optimizer_seeds": 4, "missing_weight": 8.0},
            ),
            MethodSpec(
                "visibility_branching_constrained",
                "visibility_branching",
                {
                    "iterations": 25,
                    "beam_width": 4,
                    "optimizer_seeds": 4,
                    "missing_weight": 8.0,
                    "constrained_polish": True,
                    "constrained_iterations": 30,
                },
            ),
        )
        config = BenchmarkConfig("visibility_branching_smoke", 2026070205, (), methods, RunnerSpec(write_outputs=False))

        rows = run_jobs(make_jobs(config, [context]), backend="serial", workers=1)

        self.assertEqual(len(rows), 2)
        self.assertTrue(all(row.status == "ok" for row in rows), [row.error for row in rows])
        self.assertTrue(all(row.max_offset_m < 0.35 for row in rows), [row.max_offset_m for row in rows])

    @unittest.skipUnless(importlib.util.find_spec("cvxpy") is not None, "cvxpy is optional")
    def test_visibility_sdp_solver_when_cvxpy_available(self) -> None:
        truth, pairs = folded_intersection_fixture()
        context = CaseContext("manual", 0, "manual", "folded_intersection", truth, pairs, {"min_degree": 2.0})
        method = MethodSpec("visibility_sdp", "visibility_sdp", {"iterations": 20, "sdp_max_iters": 1000})
        config = BenchmarkConfig("visibility_sdp_smoke", 2026070206, (), (method,), RunnerSpec(write_outputs=False))

        rows = run_jobs(make_jobs(config, [context]), backend="serial", workers=1)

        self.assertEqual(rows[0].status, "ok")
        self.assertLess(rows[0].max_offset_m, 0.5)


if __name__ == "__main__":
    unittest.main()
