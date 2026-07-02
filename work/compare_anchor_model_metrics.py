from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"


METRICS = (
    ("median_missing_mae_m", "lower"),
    ("median_max_offset_m", "lower"),
    ("p90_max_offset_m", "lower"),
    ("median_known_rmse_m", "lower"),
    ("under_20cm", "higher"),
    ("under_50cm", "higher"),
    ("under_1m", "higher"),
)


def read_detail(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def summarize_detail(rows: list[dict[str, str]]) -> list[dict[str, str | float]]:
    summary: list[dict[str, str | float]] = []
    keys = sorted({(row["family"], row["method"]) for row in rows})
    for family, method in keys:
        part = [row for row in rows if row["family"] == family and row["method"] == method]
        max_offsets = np.array([float(row["max_offset_m"]) for row in part], dtype=float)
        missing_mae = np.array([float(row["missing_mae_m"]) for row in part], dtype=float)
        known_rmse = np.array([float(row["known_rmse_m"]) for row in part], dtype=float)
        summary.append(
            {
                "family": family,
                "method": method,
                "cases": float(len(part)),
                "median_missing_mae_m": float(np.median(missing_mae)),
                "median_max_offset_m": float(np.median(max_offsets)),
                "p90_max_offset_m": float(np.quantile(max_offsets, 0.90)),
                "median_known_rmse_m": float(np.median(known_rmse)),
                "under_20cm": float(np.mean(max_offsets <= 0.20)),
                "under_50cm": float(np.mean(max_offsets <= 0.50)),
                "under_1m": float(np.mean(max_offsets <= 1.00)),
            }
        )
    return summary


def index_summary(rows: list[dict[str, str | float]]) -> dict[tuple[str, str], dict[str, str | float]]:
    return {(str(row["family"]), str(row["method"])): row for row in rows}


def better(candidate: float, baseline: float, direction: str, tolerance: float) -> bool:
    if direction == "lower":
        return candidate <= baseline - tolerance
    return candidate >= baseline + tolerance


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare anchor ML model metrics against a baseline detail CSV.")
    parser.add_argument(
        "--baseline",
        type=Path,
        default=OUTPUTS / "anchor_solver_ml_distance_completion_fair_diag_cuda_metrics.csv",
    )
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--tolerance", type=float, default=1e-9)
    parser.add_argument("--allow-missing-methods", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    baseline = index_summary(summarize_detail(read_detail(args.baseline)))
    candidate = index_summary(summarize_detail(read_detail(args.candidate)))
    failures: list[str] = []
    print(f"baseline={args.baseline}")
    print(f"candidate={args.candidate}")
    for key in sorted(baseline):
        if key not in candidate:
            message = f"missing candidate group family={key[0]} method={key[1]}"
            if args.allow_missing_methods:
                print(f"SKIP {message}")
                continue
            failures.append(message)
            continue
        base_row = baseline[key]
        cand_row = candidate[key]
        print(f"\n[{key[0]} | {key[1]}]")
        for metric, direction in METRICS:
            base_value = float(base_row[metric])
            cand_value = float(cand_row[metric])
            passed = better(cand_value, base_value, direction, args.tolerance)
            status = "PASS" if passed else "FAIL"
            print(f"{status} {metric}: baseline={base_value:.6g} candidate={cand_value:.6g} direction={direction}")
            if not passed:
                failures.append(f"{key[0]} {key[1]} {metric}: baseline={base_value:.6g}, candidate={cand_value:.6g}")
    if failures:
        print("\nMetric gate failed:")
        for failure in failures:
            print(f"- {failure}")
        sys.exit(1)
    print("\nMetric gate passed: candidate is better on every tracked metric.")


if __name__ == "__main__":
    main()
