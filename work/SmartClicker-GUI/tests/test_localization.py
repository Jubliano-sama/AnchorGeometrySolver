import math
import unittest

from uwb_capture.localization import (
    LOCALIZATION_ALGORITHM,
    LocalizationReading,
    build_square_simulation,
    solve_position,
)


class LocalizationSolverTests(unittest.TestCase):
    def test_exact_square_anchor_ranges_solve_known_position(self) -> None:
        true_x = 3.1
        true_y = 4.2
        anchors = [
            ("A1", 0.0, 0.0),
            ("A2", 7.0, 0.0),
            ("A3", 7.0, 7.0),
            ("A4", 0.0, 7.0),
        ]
        readings = [
            LocalizationReading(
                anchor_id=anchor_id,
                x_m=x_m,
                y_m=y_m,
                range_m=math.hypot(true_x - x_m, true_y - y_m),
            )
            for anchor_id, x_m, y_m in anchors
        ]

        result = solve_position(readings)

        self.assertAlmostEqual(result.x_m, true_x, places=4)
        self.assertAlmostEqual(result.y_m, true_y, places=4)
        self.assertLess(result.rmse_m, 1e-4)
        self.assertEqual(result.confidence, "High")

    def test_requires_at_least_three_valid_anchors(self) -> None:
        readings = [
            LocalizationReading("A1", 0.0, 0.0, 1.0),
            LocalizationReading("A2", 1.0, 0.0, 1.0),
        ]

        with self.assertRaisesRegex(ValueError, "At least three"):
            solve_position(readings)

    def test_offset_correction_is_applied_before_solving(self) -> None:
        true_x = 2.0
        true_y = 2.0
        bias = 0.25
        anchors = [
            ("A1", 0.0, 0.0),
            ("A2", 5.0, 0.0),
            ("A3", 5.0, 5.0),
            ("A4", 0.0, 5.0),
        ]
        readings = [
            LocalizationReading(
                anchor_id=anchor_id,
                x_m=x_m,
                y_m=y_m,
                range_m=math.hypot(true_x - x_m, true_y - y_m) + bias,
                offset_m=bias,
            )
            for anchor_id, x_m, y_m in anchors
        ]

        result = solve_position(readings)

        self.assertAlmostEqual(result.x_m, true_x, places=4)
        self.assertAlmostEqual(result.y_m, true_y, places=4)

    def test_common_height_component_cancels_in_radical_axis_solve(self) -> None:
        true_x = 2.6
        true_y = 1.8
        height_delta = 1.4
        anchors = [
            ("A1", 0.0, 0.0),
            ("A2", 5.0, 0.0),
            ("A3", 5.0, 4.0),
            ("A4", 0.0, 4.0),
        ]
        readings = []
        for anchor_id, x_m, y_m in anchors:
            horizontal_distance = math.hypot(true_x - x_m, true_y - y_m)
            readings.append(
                LocalizationReading(
                    anchor_id=anchor_id,
                    x_m=x_m,
                    y_m=y_m,
                    range_m=math.hypot(horizontal_distance, height_delta),
                )
            )

        result = solve_position(readings)

        self.assertIn("radical-axis", LOCALIZATION_ALGORITHM.lower())
        self.assertIn("least squares", LOCALIZATION_ALGORITHM.lower())
        self.assertAlmostEqual(result.x_m, true_x, places=4)
        self.assertAlmostEqual(result.y_m, true_y, places=4)
        self.assertAlmostEqual(result.common_height_m or 0.0, height_delta, places=4)

    def test_physically_inconsistent_ranges_still_return_low_confidence_solution(self) -> None:
        readings = [
            LocalizationReading("A1", 0.0, 0.0, 2.121),
            LocalizationReading("A2", 0.3, 0.8, 0.852),
            LocalizationReading("A3", 1.5, 1.0, 2.314),
        ]

        radical_only = solve_position(readings, range_ls_trigger_rmse_m=999.0)
        result = solve_position(readings)

        self.assertEqual(result.confidence, "Low")
        self.assertGreater(radical_only.rmse_m, 1.0)
        self.assertLess(result.rmse_m, radical_only.rmse_m - 0.5)
        self.assertNotAlmostEqual(result.x_m, result.seed_x_m)
        self.assertTrue(any("least-squares refinement improved" in warning for warning in result.warnings))

    def test_square_simulation_solves_inside_floor_plan(self) -> None:
        scenario = build_square_simulation(
            width_m=7.0,
            height_m=7.0,
        )

        result = solve_position(list(scenario.readings))

        self.assertIn("radical-axis", LOCALIZATION_ALGORITHM.lower())
        self.assertIn("least squares", LOCALIZATION_ALGORITHM.lower())
        self.assertGreaterEqual(result.x_m, 0.0)
        self.assertLessEqual(result.x_m, 7.0)
        self.assertGreaterEqual(result.y_m, 0.0)
        self.assertLessEqual(result.y_m, 7.0)
        self.assertLess(result.rmse_m, 1e-4)

    def test_noisy_square_simulation_still_solves_inside_floor_plan(self) -> None:
        scenario = build_square_simulation(
            width_m=8.0,
            height_m=5.0,
            noise_m=0.08,
        )

        result = solve_position(list(scenario.readings))

        self.assertGreaterEqual(result.x_m, 0.0)
        self.assertLessEqual(result.x_m, 8.0)
        self.assertGreaterEqual(result.y_m, 0.0)
        self.assertLessEqual(result.y_m, 5.0)
        self.assertLess(result.rmse_m, 0.08)


if __name__ == "__main__":
    unittest.main()
