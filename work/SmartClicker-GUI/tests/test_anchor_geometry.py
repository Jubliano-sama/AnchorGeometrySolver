import math
import unittest

from uwb_capture.anchor_geometry import (
    ANCHOR_LAYOUT_ALGORITHM,
    AnchorPairDistance,
    diagnose_anchor_graph,
    mirror_layout,
    pair_residuals,
    rotate_layout,
    rotate_layout_to_level,
    solve_anchor_layout,
)


def pair(anchor_a: str, anchor_b: str, distance: float) -> AnchorPairDistance:
    return AnchorPairDistance(anchor_a, anchor_b, distance, sigma_m=0.03)


def max_aligned_offset(
    truth: dict[str, tuple[float, float]],
    estimate: dict[str, tuple[float, float]],
) -> float:
    anchor_ids = sorted(truth)
    target = [truth[anchor_id] for anchor_id in anchor_ids]
    source = [estimate[anchor_id] for anchor_id in anchor_ids]
    target_center = (
        sum(x for x, _y in target) / len(target),
        sum(y for _x, y in target) / len(target),
    )

    best = math.inf
    for reflection in (1.0, -1.0):
        reflected = [(x * reflection, y) for x, y in source]
        source_center = (
            sum(x for x, _y in reflected) / len(reflected),
            sum(y for _x, y in reflected) / len(reflected),
        )
        centered_source = [
            (x - source_center[0], y - source_center[1])
            for x, y in reflected
        ]
        centered_target = [
            (x - target_center[0], y - target_center[1])
            for x, y in target
        ]
        dot = sum(
            sx * tx + sy * ty
            for (sx, sy), (tx, ty) in zip(centered_source, centered_target)
        )
        cross = sum(
            sx * ty - sy * tx
            for (sx, sy), (tx, ty) in zip(centered_source, centered_target)
        )
        angle = math.atan2(cross, dot)
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        offsets = []
        for (sx, sy), (tx, ty) in zip(centered_source, target):
            aligned_x = sx * cos_a - sy * sin_a + target_center[0]
            aligned_y = sx * sin_a + sy * cos_a + target_center[1]
            offsets.append(math.hypot(aligned_x - tx, aligned_y - ty))
        best = min(best, max(offsets))
    return best


def local_grid_pairs(
    rows: int,
    columns: int,
    spacing_m: float,
    edge_radius_m: float,
) -> tuple[dict[str, tuple[float, float]], list[AnchorPairDistance]]:
    positions = {
        f"A{index:02d}": (
            float(index % columns) * spacing_m,
            float(index // columns) * spacing_m,
        )
        for index in range(rows * columns)
    }
    readings = []
    anchor_ids = sorted(positions)
    for index, anchor_a in enumerate(anchor_ids):
        ax, ay = positions[anchor_a]
        for anchor_b in anchor_ids[index + 1 :]:
            bx, by = positions[anchor_b]
            distance = math.hypot(ax - bx, ay - by)
            if distance <= edge_radius_m:
                readings.append(AnchorPairDistance(anchor_a, anchor_b, distance, sigma_m=0.05))
    return positions, readings


class AnchorGeometrySolverTests(unittest.TestCase):
    def test_exact_square_anchor_distances_solve_low_energy_layout(self) -> None:
        width = 4.0
        height = 3.0
        diagonal = math.hypot(width, height)
        result = solve_anchor_layout(
            [
                pair("A1", "A2", width),
                pair("A2", "A3", height),
                pair("A3", "A4", width),
                pair("A4", "A1", height),
                pair("A1", "A3", diagonal),
                pair("A2", "A4", diagonal),
            ],
            seed_count=8,
            basin_hops=4,
        )

        self.assertIn("basin", ANCHOR_LAYOUT_ALGORITHM.lower())
        self.assertLess(result.rmse_m, 1e-5)
        self.assertLess(result.max_residual_m, 1e-5)
        self.assertAlmostEqual(result.positions_m["A1"][1], result.positions_m["A2"][1], places=6)
        residuals = pair_residuals(result.positions_m, result.processed_pairs)
        self.assertTrue(all(abs(value) < 1e-5 for value in residuals.values()))

    def test_noisy_anchor_distances_remain_close(self) -> None:
        positions = {
            "A1": (0.0, 0.0),
            "A2": (5.0, 0.0),
            "A3": (5.5, 3.0),
            "A4": (2.0, 4.5),
            "A5": (-1.0, 2.0),
        }
        readings = []
        noise_by_pair = {
            ("A1", "A3"): 0.03,
            ("A2", "A4"): -0.02,
            ("A3", "A5"): 0.04,
        }
        anchor_ids = list(positions)
        for index, anchor_a in enumerate(anchor_ids):
            ax, ay = positions[anchor_a]
            for anchor_b in anchor_ids[index + 1 :]:
                bx, by = positions[anchor_b]
                noise = noise_by_pair.get((anchor_a, anchor_b), 0.0)
                readings.append(pair(anchor_a, anchor_b, math.hypot(ax - bx, ay - by) + noise))

        result = solve_anchor_layout(readings, seed_count=12, basin_hops=6)

        self.assertLess(result.rmse_m, 0.05)
        self.assertLess(result.max_residual_m, 0.08)

    def test_disconnected_anchor_graph_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "disconnected"):
            solve_anchor_layout(
                [
                    pair("A1", "A2", 1.0),
                    pair("A3", "A4", 1.0),
                ]
            )

    def test_complete_four_anchor_graph_is_diagnosed_globally_rigid(self) -> None:
        diagnostics = diagnose_anchor_graph(
            [
                pair("A1", "A2", 4.0),
                pair("A2", "A3", 3.0),
                pair("A3", "A4", 4.0),
                pair("A4", "A1", 3.0),
                pair("A1", "A3", 5.0),
                pair("A2", "A4", 5.0),
            ]
        )

        self.assertTrue(diagnostics.is_locally_rigid)
        self.assertTrue(diagnostics.is_redundantly_rigid)
        self.assertTrue(diagnostics.is_generically_globally_rigid_2d)
        self.assertEqual(diagnostics.missing_pair_count, 0)

    def test_four_cycle_graph_is_diagnosed_ambiguous(self) -> None:
        diagnostics = diagnose_anchor_graph(
            [
                pair("A1", "A2", 4.0),
                pair("A2", "A3", 3.0),
                pair("A3", "A4", 4.0),
                pair("A4", "A1", 3.0),
            ]
        )

        self.assertFalse(diagnostics.is_locally_rigid)
        self.assertFalse(diagnostics.is_generically_globally_rigid_2d)
        self.assertTrue(diagnostics.warnings)


    def test_min_anchor_spacing_prior_pushes_too_close_anchors_apart(self) -> None:
        result = solve_anchor_layout(
            [AnchorPairDistance("A1", "A2", 0.30, sigma_m=0.05)],
            seed_count=4,
            basin_hops=0,
            max_iterations=80,
            min_anchor_spacing_m=2.0,
            anchor_spacing_sigma_m=0.02,
            unmeasured_pair_min_distance_m=0.0,
            boundary_degree_prior_sigma_m=0.0,
            distance_polish_iterations=0,
        )

        ax, ay = result.positions_m["A1"]
        bx, by = result.positions_m["A2"]
        self.assertGreater(math.hypot(ax - bx, ay - by), 1.0)


    def test_local_grid_uses_missing_pairs_to_avoid_collapsed_alias(self) -> None:
        positions, readings = local_grid_pairs(
            rows=4,
            columns=4,
            spacing_m=5.0,
            edge_radius_m=8.0,
        )

        result = solve_anchor_layout(
            readings,
            seed_count=24,
            basin_hops=10,
            max_iterations=100,
            random_seed=1337,
        )

        self.assertLess(result.rmse_m, 1e-5)
        self.assertLess(result.max_residual_m, 1e-5)
        self.assertLess(max_aligned_offset(positions, result.positions_m), 1e-5)

    def test_rotate_layout_to_level_puts_selected_pair_on_same_y(self) -> None:
        positions = {
            "A1": (1.0, 2.0),
            "A2": (3.0, 4.0),
            "A3": (1.0, 4.0),
        }

        rotated = rotate_layout_to_level(positions, "A1", "A2")

        self.assertAlmostEqual(rotated["A1"][0], 0.0, places=6)
        self.assertAlmostEqual(rotated["A1"][1], 0.0, places=6)
        self.assertGreater(rotated["A2"][0], 0.0)
        self.assertAlmostEqual(rotated["A2"][1], 0.0, places=6)

    def test_rotate_and_mirror_preserve_pair_distances(self) -> None:
        positions = {
            "A1": (0.0, 0.0),
            "A2": (4.0, 0.0),
            "A3": (4.0, 3.0),
        }
        pairs = [
            pair("A1", "A2", 4.0),
            pair("A2", "A3", 3.0),
            pair("A1", "A3", 5.0),
        ]

        transformed = mirror_layout(rotate_layout(positions, 37.0), "x")
        residuals = pair_residuals(transformed, pairs)

        self.assertTrue(all(abs(value) < 1e-9 for value in residuals.values()))


if __name__ == "__main__":
    unittest.main()
