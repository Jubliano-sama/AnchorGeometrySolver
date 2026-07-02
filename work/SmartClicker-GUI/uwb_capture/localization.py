"""Small 2D UWB localization solver used by the capture GUI.

The GUI already owns the live measured ranges. This module keeps the position
solver dependency-free so the capture app does not need to launch the separate
simulation/localization project or install plotting libraries.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

LOCALIZATION_ALGORITHM = "Radical-axis line least squares with range-LS fallback"
RANGE_LS_TRIGGER_RMSE_M = 0.5
RANGE_LS_MIN_IMPROVEMENT_M = 0.01


@dataclass(frozen=True)
class LocalizationReading:
    """One enabled range measurement from the tag to a fixed anchor."""

    anchor_id: str
    x_m: float
    y_m: float
    range_m: float
    sigma_m: float = 0.05
    offset_m: float = 0.0
    enabled: bool = True


@dataclass(frozen=True)
class ProcessedReading:
    """Validated range measurement prepared for radical-axis solving."""

    anchor_id: str
    x_m: float
    y_m: float
    measured_range_m: float
    corrected_range_m: float
    sigma_m: float
    weight: float


@dataclass(frozen=True)
class LocalizationResult:
    """Final position estimate and diagnostics."""

    seed_x_m: float
    seed_y_m: float
    x_m: float
    y_m: float
    radical_axis_rmse_m: float
    rmse_m: float
    confidence: str
    processed_readings: tuple[ProcessedReading, ...]
    residuals_m: dict[str, float]
    range_residuals_m: dict[str, float]
    common_height_m: float | None
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class SimulatedLocalizationScenario:
    """Deterministic fake anchor ranges for GUI and regression testing."""

    readings: tuple[LocalizationReading, ...]


def solve_position(
    readings: list[LocalizationReading],
    *,
    min_sigma_m: float = 0.01,
    min_range_m: float = 0.05,
    range_ls_trigger_rmse_m: float = RANGE_LS_TRIGGER_RMSE_M,
    range_ls_min_improvement_m: float = RANGE_LS_MIN_IMPROVEMENT_M,
) -> LocalizationResult:
    """Solve a 2D position from radical-axis line least squares.

    Subtracting squared range equations turns each anchor pair into a line. A
    common vertical height term appears in every squared range, so it cancels
    before the initial least-squares solve. If the resulting range fit is poor,
    a bounded range least-squares refinement is tried and used only when it
    improves the range RMSE.
    """

    processed = _preprocess_readings(
        readings,
        min_sigma_m=min_sigma_m,
        min_range_m=min_range_m,
    )
    x, y = _solve_radical_axis_lines(processed)
    seed_x, seed_y = x, y
    line_residuals, line_rmse, line_warnings = _analyze_radical_axis_residuals(processed, x, y)
    range_residuals, range_rmse, common_height_m, range_warnings = _analyze_range_fit(processed, x, y)
    refinement_warnings: list[str] = []
    if range_rmse >= range_ls_trigger_rmse_m:
        original_range_rmse = range_rmse
        refined_x, refined_y = _solve_range_least_squares(processed, x, y)
        (
            refined_range_residuals,
            refined_range_rmse,
            refined_common_height_m,
            refined_range_warnings,
        ) = _analyze_range_fit(processed, refined_x, refined_y)
        if refined_range_rmse <= original_range_rmse - range_ls_min_improvement_m:
            x = refined_x
            y = refined_y
            line_residuals, line_rmse, line_warnings = _analyze_radical_axis_residuals(processed, x, y)
            range_residuals = refined_range_residuals
            range_rmse = refined_range_rmse
            common_height_m = refined_common_height_m
            range_warnings = refined_range_warnings
            refinement_warnings.append(
                "Range least-squares refinement improved RMSE "
                f"from {original_range_rmse:.3f} m to {range_rmse:.3f} m."
            )
        else:
            refinement_warnings.append(
                "Range least-squares refinement did not improve the range fit enough to use."
            )
    confidence, confidence_warnings = _localization_confidence(
        line_rmse=line_rmse,
        range_rmse=range_rmse,
        common_height_m=common_height_m,
    )
    return LocalizationResult(
        seed_x_m=seed_x,
        seed_y_m=seed_y,
        x_m=x,
        y_m=y,
        radical_axis_rmse_m=line_rmse,
        rmse_m=range_rmse,
        confidence=confidence,
        processed_readings=tuple(processed),
        residuals_m=line_residuals,
        range_residuals_m=range_residuals,
        common_height_m=common_height_m,
        warnings=tuple(line_warnings + range_warnings + refinement_warnings + confidence_warnings),
    )


def build_square_simulation(
    *,
    width_m: float = 7.0,
    height_m: float = 7.0,
    sigma_m: float = 0.05,
    noise_m: float = 0.0,
) -> SimulatedLocalizationScenario:
    """Build a simple four-anchor floor-plan simulation in metric units.

    The fake ranges come from a deterministic hidden tag point so the GUI can
    exercise the solver without asking the operator for the clicker position.
    The optional noise is deterministic so test output and GUI demos are
    repeatable. Distances remain physically plausible for small noise values.
    """

    if width_m <= 0 or height_m <= 0:
        raise ValueError("Simulation width and height must be greater than 0.")
    if sigma_m <= 0:
        raise ValueError("Simulation sigma must be greater than 0.")
    if noise_m < 0:
        raise ValueError("Simulation noise must be 0 or greater.")

    reference_x_m = width_m * 0.443
    reference_y_m = height_m * 0.600
    anchors = (
        ("A1", 0.0, 0.0, 0.0),
        ("A2", width_m, 0.0, 0.65),
        ("A3", width_m, height_m, -0.45),
        ("A4", 0.0, height_m, 0.25),
    )
    readings = []
    for anchor_id, x_m, y_m, noise_scale in anchors:
        true_range = math.hypot(reference_x_m - x_m, reference_y_m - y_m)
        measured_range = max(true_range + noise_m * noise_scale, 0.05)
        readings.append(
            LocalizationReading(
                anchor_id=anchor_id,
                x_m=x_m,
                y_m=y_m,
                range_m=measured_range,
                sigma_m=sigma_m,
            )
        )
    return SimulatedLocalizationScenario(
        readings=tuple(readings),
    )


def _preprocess_readings(
    readings: list[LocalizationReading],
    *,
    min_sigma_m: float,
    min_range_m: float,
) -> list[ProcessedReading]:
    processed: list[ProcessedReading] = []
    for reading in readings:
        if not reading.enabled:
            continue
        values = (reading.x_m, reading.y_m, reading.range_m, reading.sigma_m, reading.offset_m)
        if not all(math.isfinite(float(value)) for value in values):
            continue
        if reading.range_m <= min_range_m:
            continue
        sigma = max(abs(float(reading.sigma_m)), min_sigma_m)
        corrected_range = max(float(reading.range_m) - float(reading.offset_m), min_range_m)
        processed.append(
            ProcessedReading(
                anchor_id=str(reading.anchor_id),
                x_m=float(reading.x_m),
                y_m=float(reading.y_m),
                measured_range_m=float(reading.range_m),
                corrected_range_m=corrected_range,
                sigma_m=sigma,
                weight=1.0 / (sigma * sigma),
            )
        )
    if len(processed) < 3:
        raise ValueError("At least three valid enabled anchor readings are required.")
    return processed


def _solve_radical_axis_lines(readings: list[ProcessedReading]) -> tuple[float, float]:
    a11 = a12 = a22 = 0.0
    b1 = b2 = 0.0
    row_count = 0
    for first, second, rx, ry, rhs, weight in _radical_axis_rows(readings):
        a11 += weight * rx * rx
        a12 += weight * rx * ry
        a22 += weight * ry * ry
        b1 += weight * rx * rhs
        b2 += weight * ry * rhs
        row_count += 1
    if row_count < 2:
        raise ValueError("At least two independent anchor pairs are required.")
    return _solve_2x2(a11, a12, a22, b1, b2, "anchor geometry is singular")


def _radical_axis_rows(
    readings: list[ProcessedReading],
):
    for i, first in enumerate(readings):
        for second in readings[i + 1 :]:
            rx = 2.0 * (second.x_m - first.x_m)
            ry = 2.0 * (second.y_m - first.y_m)
            rhs = (
                first.corrected_range_m**2
                - second.corrected_range_m**2
                + second.x_m**2
                + second.y_m**2
                - first.x_m**2
                - first.y_m**2
            )
            if math.hypot(rx, ry) <= 1e-12:
                continue
            yield first, second, rx, ry, rhs, min(first.weight, second.weight)


def _analyze_radical_axis_residuals(
    readings: list[ProcessedReading],
    x_m: float,
    y_m: float,
    *,
    weak_rmse_m: float = 0.35,
) -> tuple[dict[str, float], float, list[str]]:
    residuals: dict[str, float] = {}
    weighted_residual_sum = 0.0
    weight_total = 0.0
    warnings: list[str] = []
    for first, second, rx, ry, rhs, weight in _radical_axis_rows(readings):
        line_length = math.hypot(rx, ry)
        residual = (rx * x_m + ry * y_m - rhs) / line_length
        pair_id = f"{first.anchor_id}-{second.anchor_id}"
        residuals[pair_id] = residual
        weighted_residual_sum += weight * residual * residual
        weight_total += weight
        if abs(residual) > weak_rmse_m:
            warnings.append(f"{pair_id} radical-axis residual is {residual:.3f} m.")

    if weight_total <= 0.0:
        raise ValueError("At least two independent anchor pairs are required.")
    rmse = math.sqrt(weighted_residual_sum / weight_total)
    if rmse > weak_rmse_m:
        warnings.append("Radical-axis RMSE is high; check anchor positions, NLOS, or bad ranges.")
    return residuals, rmse, warnings


def _analyze_range_fit(
    readings: list[ProcessedReading],
    x_m: float,
    y_m: float,
) -> tuple[dict[str, float], float, float | None, list[str]]:
    height_squared_sum = 0.0
    weight_total = 0.0
    warnings: list[str] = []
    horizontal_distances: dict[str, float] = {}

    for reading in readings:
        horizontal = math.hypot(x_m - reading.x_m, y_m - reading.y_m)
        horizontal_distances[reading.anchor_id] = horizontal
        height_squared = reading.corrected_range_m**2 - horizontal**2
        height_squared_sum += reading.weight * height_squared
        weight_total += reading.weight

    if weight_total <= 0.0:
        raise ValueError("At least three valid enabled anchor readings are required.")
    common_height_squared = height_squared_sum / weight_total

    modeled_height_squared = max(common_height_squared, 0.0)
    common_height_m = math.sqrt(common_height_squared) if common_height_squared >= 0.0 else None
    if common_height_squared < 0.0:
        warnings.append(
            "Ranges are shorter than the solved XY distances; check anchor coordinates, offsets, NLOS, or bad ranges."
        )

    residuals: dict[str, float] = {}
    weighted_residual_sum = 0.0
    for reading in readings:
        modeled_range = math.hypot(horizontal_distances[reading.anchor_id], math.sqrt(modeled_height_squared))
        residual = modeled_range - reading.corrected_range_m
        residuals[reading.anchor_id] = residual
        weighted_residual_sum += reading.weight * residual * residual
    rmse = math.sqrt(weighted_residual_sum / weight_total)
    return residuals, rmse, common_height_m, warnings


def _solve_range_least_squares(
    readings: list[ProcessedReading],
    seed_x_m: float,
    seed_y_m: float,
    *,
    max_iterations: int = 60,
    initial_damping: float = 1e-3,
) -> tuple[float, float]:
    x_m = seed_x_m
    y_m = seed_y_m
    best_x_m = x_m
    best_y_m = y_m
    best_cost = _range_least_squares_cost(readings, x_m, y_m)
    damping = initial_damping

    for _iteration in range(max_iterations):
        a11 = a12 = a22 = 0.0
        g1 = g2 = 0.0
        for reading in readings:
            dx = x_m - reading.x_m
            dy = y_m - reading.y_m
            modeled_range = max(math.hypot(dx, dy), 1e-9)
            residual = modeled_range - reading.corrected_range_m
            jx = dx / modeled_range
            jy = dy / modeled_range
            weight = reading.weight
            a11 += weight * jx * jx
            a12 += weight * jx * jy
            a22 += weight * jy * jy
            g1 += weight * jx * residual
            g2 += weight * jy * residual

        try:
            step_x, step_y = _solve_2x2(
                a11 + damping,
                a12,
                a22 + damping,
                -g1,
                -g2,
                "range least-squares matrix is singular",
            )
        except ValueError:
            break
        if not (math.isfinite(step_x) and math.isfinite(step_y)):
            break

        candidate_x = x_m + step_x
        candidate_y = y_m + step_y
        candidate_cost = _range_least_squares_cost(readings, candidate_x, candidate_y)
        if candidate_cost < best_cost:
            x_m = candidate_x
            y_m = candidate_y
            best_x_m = candidate_x
            best_y_m = candidate_y
            best_cost = candidate_cost
            damping = max(damping * 0.3, 1e-9)
            if math.hypot(step_x, step_y) < 1e-6:
                break
        else:
            damping *= 10.0
            if damping > 1e12:
                break
    return best_x_m, best_y_m


def _range_least_squares_cost(
    readings: list[ProcessedReading],
    x_m: float,
    y_m: float,
) -> float:
    cost = 0.0
    for reading in readings:
        modeled_range = math.hypot(x_m - reading.x_m, y_m - reading.y_m)
        residual = modeled_range - reading.corrected_range_m
        cost += reading.weight * residual * residual
    return cost


def _localization_confidence(
    *,
    line_rmse: float,
    range_rmse: float,
    common_height_m: float | None,
    good_rmse_m: float = 0.15,
    weak_rmse_m: float = 0.35,
) -> tuple[str, list[str]]:
    warnings: list[str] = []
    combined_rmse = max(line_rmse, range_rmse)
    if combined_rmse <= good_rmse_m:
        confidence = "High"
    elif combined_rmse <= weak_rmse_m:
        confidence = "Medium"
    else:
        confidence = "Low"
        warnings.append("Range fit RMSE is high; check anchor positions, NLOS, offsets, or bad ranges.")
    if common_height_m is not None and common_height_m > 3.0:
        warnings.append(
            f"Estimated common height component is {common_height_m:.2f} m; check whether ranges are 3D distances."
        )
    return confidence, warnings


def _solve_2x2(
    a11: float,
    a12: float,
    a22: float,
    b1: float,
    b2: float,
    error_message: str,
) -> tuple[float, float]:
    determinant = a11 * a22 - a12 * a12
    if abs(determinant) < 1e-12:
        raise ValueError(error_message)
    return (
        (b1 * a22 - b2 * a12) / determinant,
        (a11 * b2 - a12 * b1) / determinant,
    )
