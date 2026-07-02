from __future__ import annotations

import math
import random
import sys
import unittest
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "work"))
sys.path.insert(0, str(ROOT / "work" / "SmartClicker-GUI"))

import anchor_solver_fold_rescue_experiment as exp  # noqa: E402
import anchor_solver_ml_distance_completion as dc  # noqa: E402
import anchor_solver_p95_ml_cases as p95  # noqa: E402
from uwb_capture.anchor_geometry import AnchorPairDistance, _anchor_ids, _preprocess_pairs  # noqa: E402


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)


class LegacyRegressionTest(unittest.TestCase):
    def test_graph_shortest_scaffold_pair_uses_hop_weight_formula(self) -> None:
        known = [
            AnchorPairDistance("A00", "A01", 3.0, sigma_m=0.05, source="known"),
            AnchorPairDistance("A01", "A02", 4.0, sigma_m=0.05, source="known"),
        ]
        processed = _preprocess_pairs(known, min_sigma_m=0.02, min_distance_m=0.05)
        scaffold = exp.graph_shortest_scaffold_pairs(
            processed,
            _anchor_ids(processed),
            max_hops=2,
            relative_sigma=0.25,
            scaffold_weight=4.0,
            hop_weight_base=2.0,
        )

        self.assertEqual(len(scaffold), 1)
        pair = scaffold[0]
        self.assertEqual((pair.anchor_a_id, pair.anchor_b_id), ("A00", "A02"))
        self.assertAlmostEqual(pair.distance_m, 7.0)
        self.assertIn("h2", pair.source)
        spring_multiplier = 4.0 * (2.0**2)
        expected_sigma = max(0.35, 7.0 * 0.25) / math.sqrt(spring_multiplier)
        self.assertAlmostEqual(pair.sigma_m, expected_sigma)

    def test_grid_case_generation_is_fair(self) -> None:
        seed_all(2026070201)
        case = p95.generate_cases("grid", 1, torch.device("cpu"))[0]
        diagnostics = dc.measured_graph_diagnostics(case.points)

        self.assertGreaterEqual(case.points.shape[0], 16)
        self.assertGreaterEqual(diagnostics["min_degree"], 3.0)
        self.assertGreaterEqual(diagnostics["vertex_connectivity_capped3"], 1.0)

    def test_offset_summary_ignores_translation_rotation_and_mirror(self) -> None:
        truth = {
            "A00": (0.0, 0.0),
            "A01": (4.0, 0.0),
            "A02": (4.0, 3.0),
            "A03": (0.0, 3.0),
        }
        angle = math.radians(37.0)
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        estimate: dict[str, tuple[float, float]] = {}
        for anchor_id, (x, y) in truth.items():
            mirrored_y = -y
            estimate[anchor_id] = (
                cos_a * x - sin_a * mirrored_y + 12.5,
                sin_a * x + cos_a * mirrored_y - 8.0,
            )

        max_offset, median_offset, p95_offset = dc.offset_summary(truth, estimate)
        self.assertLess(max_offset, 1e-8)
        self.assertLess(median_offset, 1e-8)
        self.assertLess(p95_offset, 1e-8)


if __name__ == "__main__":
    unittest.main()
