from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
import random
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
sys.path.insert(0, str(ROOT / "work"))

import anchor_solver_ml_distance_completion as dc
import anchor_solver_p95_ml_cases as p95

BUCKET_KEYS = ("random", "grid", "office")
BUCKET_LABELS = {"random": "Random 16-32", "grid": "Grid >=16", "office": "Office >=16"}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return device


def completed_and_weak_pairs(
    batch: dc.GraphBatch,
    pred_norm: torch.Tensor,
    case_index: int,
    args: argparse.Namespace,
):
    completed_pairs, pred_matrix, scale = dc.completed_pairs_from_prediction(
        batch,
        pred_norm,
        case_index,
        predicted_sigma_m=args.predicted_sigma,
        predicted_sigma_slope=args.predicted_sigma_slope,
        closest_predicted_pairs_per_anchor=args.closest_predicted_pairs_per_anchor,
    )
    known_pairs = dc.known_pairs_from_batch(batch, case_index)
    weak_pairs = dc.weak_completed_pairs(
        completed_pairs,
        max_predicted_distance_m=args.weak_completion_max_distance,
        sigma_multiplier=args.weak_completion_sigma_multiplier,
    )
    return known_pairs, completed_pairs, weak_pairs, pred_matrix, scale


def solve_branch(
    branch: str,
    pairs: list[dc.AnchorPairDistance],
    known_pairs: list[dc.AnchorPairDistance],
    args: argparse.Namespace,
) -> dict[str, tuple[float, float]]:
    if branch == "completed":
        return dc.completion_solution(
            pairs,
            known_pairs,
            max_iterations=args.solver_iterations,
            polish_known_iterations=args.polish_iterations,
        )
    if branch == "weak":
        return dc.completion_solution_weak_polish(
            pairs,
            known_pairs,
            max_iterations=args.solver_iterations,
            weak_polish_iterations=args.weak_polish_iterations,
            max_predicted_distance_m=args.weak_completion_max_distance,
            sigma_multiplier=args.weak_completion_sigma_multiplier,
        )
    raise ValueError(branch)


def branch_pair_set(branch: str, completed_pairs: list[dc.AnchorPairDistance], weak_pairs: list[dc.AnchorPairDistance]) -> list[dc.AnchorPairDistance]:
    if branch == "completed":
        return completed_pairs
    if branch == "weak":
        return weak_pairs
    raise ValueError(branch)


def spring_weight_rows(
    *,
    label: str,
    bucket: str,
    bucket_key: str,
    case_index: int,
    family: str,
    shape: str,
    anchors: int,
    branch: str,
    pairs: list[dc.AnchorPairDistance],
) -> list[dict[str, float | int | str]]:
    processed = dc._preprocess_pairs(pairs, min_sigma_m=0.02, min_distance_m=0.05)
    rows: list[dict[str, float | int | str]] = []
    for pair in processed:
        rows.append(
            {
                "label": label,
                "bucket": bucket,
                "bucket_key": bucket_key,
                "case_index": case_index,
                "family": family,
                "shape": shape,
                "anchors": anchors,
                "branch": branch,
                "anchor_a_id": pair.anchor_a_id,
                "anchor_b_id": pair.anchor_b_id,
                "source": pair.source,
                "distance_m": pair.distance_m,
                "sigma_m": pair.sigma_m,
                "optimizer_weight": pair.weight,
                "sqrt_optimizer_weight": math.sqrt(pair.weight),
            }
        )
    return rows


def summarize(detail: list[dict[str, float | int | str]]) -> list[dict[str, float | int | str]]:
    groups = sorted({(row["branch"], row["bucket"]) for row in detail}, key=lambda item: (str(item[0]), str(item[1])))
    rows: list[dict[str, float | int | str]] = []
    for branch, bucket in groups:
        part = [row for row in detail if row["branch"] == branch and row["bucket"] == bucket]
        offsets = np.array([float(row["max_offset_m"]) for row in part], dtype=float)
        rmses = np.array([float(row["known_rmse_m"]) for row in part], dtype=float)
        missing = np.array([float(row["missing_mae_m"]) for row in part], dtype=float)
        synthetic = np.array([float(row["spring_count"]) for row in part], dtype=float)
        rows.append(
            {
                "branch": branch,
                "bucket": bucket,
                "cases": len(part),
                "median_max_offset_m": float(np.median(offsets)),
                "p90_max_offset_m": float(np.quantile(offsets, 0.90)),
                "p95_max_offset_m": float(np.quantile(offsets, 0.95)),
                "max_offset_m": float(np.max(offsets)),
                "under_0_20m": float(np.mean(offsets <= 0.20)),
                "under_0_50m": float(np.mean(offsets <= 0.50)),
                "under_1m": float(np.mean(offsets <= 1.0)),
                "mean_squared_max_offset": float(np.mean(offsets * offsets)),
                "median_known_rmse_m": float(np.median(rmses)),
                "median_missing_mae_m": float(np.median(missing)),
                "median_spring_count": float(np.median(synthetic)),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate weak vs completed ML solves and export optimizer spring weights.")
    parser.add_argument("--checkpoint", type=Path, default=OUTPUTS / "anchor_solver_ppo_gpu_full_continue_best.pt")
    parser.add_argument("--prefix", default="anchor_solver_branch_weight_compare")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=2026062717)
    parser.add_argument("--cases-per-bucket", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--solver-iterations", type=int, default=80)
    parser.add_argument("--polish-iterations", type=int, default=80)
    parser.add_argument("--weak-polish-iterations", type=int, default=80)
    parser.add_argument("--closest-predicted-pairs-per-anchor", type=float, default=5.0)
    parser.add_argument("--predicted-sigma", type=float, default=0.55)
    parser.add_argument("--predicted-sigma-slope", type=float, default=0.65)
    parser.add_argument("--weak-completion-max-distance", type=float, default=18.0)
    parser.add_argument("--weak-completion-sigma-multiplier", type=float, default=2.5)
    args = parser.parse_args()

    OUTPUTS.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    device = choose_device(args.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
    print(f"device={device} checkpoint={args.checkpoint}", flush=True)
    model = p95.load_model(args.checkpoint, device)
    model.eval()

    cases: list[p95.CaseSpec] = []
    for bucket_key in BUCKET_KEYS:
        generated = p95.generate_cases(bucket_key, args.cases_per_bucket, torch.device("cpu"))
        cases.extend(generated)
        print(f"generated bucket={bucket_key} cases={len(generated)}", flush=True)

    detail: list[dict[str, float | int | str]] = []
    spring_rows: list[dict[str, float | int | str]] = []
    global_case_index = 0
    with torch.no_grad():
        for start in range(0, len(cases), args.batch_size):
            chunk_cases = cases[start : start + args.batch_size]
            batch = p95.graph_batch_from_cases(chunk_cases, device)
            pred = model(batch.node_features, batch.edge_features, batch.mask, batch.measured_mask, batch.pair_mask)
            for local_index, case in enumerate(chunk_cases):
                truth = dc.graph_to_truth(batch, local_index)
                known_pairs, completed_pairs, weak_pairs, _pred_matrix, _scale = completed_and_weak_pairs(batch, pred, local_index, args)
                missing_mae = dc.missing_mae(batch, pred, local_index)
                for branch in ("weak", "completed"):
                    solver_pairs = branch_pair_set(branch, completed_pairs, weak_pairs)
                    positions = solve_branch(branch, solver_pairs, known_pairs, args)
                    max_offset, median_offset, p95_offset = dc.offset_summary(truth, positions)
                    known_rmse, known_max = dc.pair_metrics(positions, known_pairs)
                    processed = dc._preprocess_pairs(solver_pairs, min_sigma_m=0.02, min_distance_m=0.05)
                    detail.append(
                        {
                            "branch": branch,
                            "bucket": case.bucket,
                            "bucket_key": next((key for key, value in BUCKET_LABELS.items() if value == case.bucket), case.bucket),
                            "case_index": global_case_index,
                            "family": case.family,
                            "shape": case.shape,
                            "anchors": int(batch.node_counts[local_index]),
                            "known_pairs": len(known_pairs),
                            "spring_count": len(processed),
                            "missing_mae_m": missing_mae,
                            "known_rmse_m": known_rmse,
                            "known_max_residual_m": known_max,
                            "max_offset_m": max_offset,
                            "median_offset_m": median_offset,
                            "p95_offset_m": p95_offset,
                        }
                    )
                    spring_rows.extend(
                        spring_weight_rows(
                            label="eval",
                            bucket=case.bucket,
                            bucket_key=next((key for key, value in BUCKET_LABELS.items() if value == case.bucket), case.bucket),
                            case_index=global_case_index,
                            family=case.family,
                            shape=case.shape,
                            anchors=int(batch.node_counts[local_index]),
                            branch=branch,
                            pairs=solver_pairs,
                        )
                    )
                print(f"case={global_case_index} bucket={case.bucket} anchors={int(batch.node_counts[local_index])} done", flush=True)
                global_case_index += 1

    summary = summarize(detail)
    detail_path = OUTPUTS / f"{args.prefix}_detail.csv"
    summary_path = OUTPUTS / f"{args.prefix}_summary.csv"
    spring_path = OUTPUTS / f"{args.prefix}_spring_weights.csv"
    write_csv(detail_path, detail)
    write_csv(summary_path, summary)
    write_csv(spring_path, spring_rows)
    for row in summary:
        print(
            f"summary branch={row['branch']} bucket={row['bucket']} p95={float(row['p95_max_offset_m']):.3f}m "
            f"median={float(row['median_max_offset_m']):.3f}m under1={float(row['under_1m']):.2f} "
            f"mean_sq={float(row['mean_squared_max_offset']):.4f}",
            flush=True,
        )
    print(f"Wrote {detail_path}", flush=True)
    print(f"Wrote {summary_path}", flush=True)
    print(f"Wrote {spring_path}", flush=True)


if __name__ == "__main__":
    main()
