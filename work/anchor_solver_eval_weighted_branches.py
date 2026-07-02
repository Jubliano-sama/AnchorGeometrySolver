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
from anchor_solver_weighted_output import WeightedDistanceCompletionNet

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


def load_weighted_model(path: Path, device: torch.device) -> WeightedDistanceCompletionNet:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    saved_args = checkpoint.get("args", {})
    probe = dc.make_graph_batch(2, device=device, random_fraction=0.5)
    model = WeightedDistanceCompletionNet(
        probe.node_features.shape[-1],
        probe.edge_features.shape[-1],
        hidden=int(saved_args.get("hidden", 144)),
        layers=int(saved_args.get("layers", 5)),
        dropout=float(saved_args.get("dropout", 0.0)),
        max_abs_log_weight=float(saved_args.get("max_abs_log_weight", 2.0)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def weighted_completed_pairs_from_prediction(
    batch: dc.GraphBatch,
    pred_norm: torch.Tensor,
    weight_multiplier: torch.Tensor,
    case_index: int,
    *,
    predicted_sigma_m: float,
    predicted_sigma_slope: float,
    closest_predicted_pairs_per_anchor: float,
) -> tuple[list[dc.AnchorPairDistance], dict[tuple[int, int], dict[str, float]], np.ndarray, np.ndarray]:
    n = batch.node_counts[case_index]
    scale = float(batch.scale_m[case_index].detach().cpu())
    measured = batch.measured_mask[case_index, :n, :n].detach().cpu().numpy()
    measured_dist = batch.measured_dist_m[case_index, :n, :n].detach().cpu().numpy()
    pred_dist = (pred_norm[case_index, :n, :n].detach().cpu().numpy() * scale).astype(float)
    weight_np = weight_multiplier[case_index, :n, :n].detach().cpu().numpy().astype(float)
    pairs: list[dc.AnchorPairDistance] = []
    metadata: dict[tuple[int, int], dict[str, float]] = {}
    candidates: list[tuple[float, int, int, float, float, float]] = []
    for i in range(n):
        for j in range(i + 1, n):
            if measured[i, j]:
                pairs.append(dc.AnchorPairDistance(f"A{i:02d}", f"A{j:02d}", float(measured_dist[i, j]), dc.KNOWN_SIGMA_M, True, "known"))
                metadata[(i, j)] = {
                    "base_sigma_m": dc.KNOWN_SIGMA_M,
                    "model_weight_multiplier": 1.0,
                    "effective_sigma_m": dc.KNOWN_SIGMA_M,
                    "base_optimizer_weight": 1.0 / (dc.KNOWN_SIGMA_M * dc.KNOWN_SIGMA_M),
                    "effective_optimizer_weight": 1.0 / (dc.KNOWN_SIGMA_M * dc.KNOWN_SIGMA_M),
                }
                continue
            predicted = max(float(pred_dist[i, j]), 0.05)
            base_sigma = predicted_sigma_m * (1.0 + predicted_sigma_slope * max(predicted - dc.EDGE_RADIUS_M, 0.0) / dc.EDGE_RADIUS_M)
            multiplier = max(float(weight_np[i, j]), 1e-6)
            effective_sigma = base_sigma / math.sqrt(multiplier)
            candidates.append((predicted, i, j, effective_sigma, base_sigma, multiplier))

    limit = len(candidates)
    if closest_predicted_pairs_per_anchor > 0.0:
        limit = min(limit, max(0, int(math.ceil(float(closest_predicted_pairs_per_anchor) * n))))
    candidates.sort(key=lambda item: (item[0], item[1], item[2]))
    for predicted, i, j, effective_sigma, base_sigma, multiplier in candidates[:limit]:
        pairs.append(dc.AnchorPairDistance(f"A{i:02d}", f"A{j:02d}", predicted, effective_sigma, True, "predicted-weighted"))
        metadata[(i, j)] = {
            "base_sigma_m": base_sigma,
            "model_weight_multiplier": multiplier,
            "effective_sigma_m": effective_sigma,
            "base_optimizer_weight": 1.0 / (base_sigma * base_sigma),
            "effective_optimizer_weight": 1.0 / (effective_sigma * effective_sigma),
        }
    return pairs, metadata, pred_dist, weight_np


def branch_pair_set(branch: str, completed_pairs: list[dc.AnchorPairDistance], args: argparse.Namespace) -> list[dc.AnchorPairDistance]:
    if branch == "completed":
        return completed_pairs
    if branch == "weak":
        return dc.weak_completed_pairs(
            completed_pairs,
            max_predicted_distance_m=args.weak_completion_max_distance,
            sigma_multiplier=args.weak_completion_sigma_multiplier,
        )
    raise ValueError(branch)


def solve_branch(branch: str, pairs: list[dc.AnchorPairDistance], known_pairs: list[dc.AnchorPairDistance], args: argparse.Namespace) -> dict[str, tuple[float, float]]:
    if branch == "completed":
        return dc.completion_solution(pairs, known_pairs, max_iterations=args.solver_iterations, polish_known_iterations=args.polish_iterations)
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


def pair_key(pair) -> tuple[int, int]:
    a = int(str(pair.anchor_a_id).lstrip("A"))
    b = int(str(pair.anchor_b_id).lstrip("A"))
    return (min(a, b), max(a, b))


def spring_rows_for_branch(
    *,
    branch: str,
    pairs: list[dc.AnchorPairDistance],
    metadata: dict[tuple[int, int], dict[str, float]],
    case_row: dict[str, float | int | str],
) -> list[dict[str, float | int | str]]:
    processed = dc._preprocess_pairs(pairs, min_sigma_m=0.02, min_distance_m=0.05)
    rows: list[dict[str, float | int | str]] = []
    for pair in processed:
        key = pair_key(pair)
        meta = metadata.get(key, {})
        branch_sigma_multiplier = pair.sigma_m / float(meta.get("effective_sigma_m", pair.sigma_m)) if meta else 1.0
        rows.append(
            {
                **case_row,
                "branch": branch,
                "anchor_a_id": pair.anchor_a_id,
                "anchor_b_id": pair.anchor_b_id,
                "source": pair.source,
                "distance_m": pair.distance_m,
                "base_sigma_m": meta.get("base_sigma_m", pair.sigma_m),
                "model_weight_multiplier": meta.get("model_weight_multiplier", 1.0),
                "effective_sigma_before_branch_m": meta.get("effective_sigma_m", pair.sigma_m),
                "branch_sigma_multiplier": branch_sigma_multiplier,
                "sigma_m": pair.sigma_m,
                "base_optimizer_weight": meta.get("base_optimizer_weight", pair.weight),
                "optimizer_weight_before_branch": meta.get("effective_optimizer_weight", pair.weight),
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
        weight_medians = np.array([float(row["median_model_weight_multiplier"]) for row in part], dtype=float)
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
                "median_model_weight_multiplier": float(np.median(weight_medians)),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    if not rows:
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate weighted-output model with weak and completed branches.")
    parser.add_argument("--checkpoint", type=Path, default=OUTPUTS / "anchor_solver_weighted_distill_neutral_gpu_best_best.pt")
    parser.add_argument("--prefix", default="anchor_solver_weighted_branch_compare")
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
    model = load_weighted_model(args.checkpoint, device)

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
            pred, weight_multiplier = model(batch.node_features, batch.edge_features, batch.mask, batch.measured_mask, batch.pair_mask)
            for local_index, case in enumerate(chunk_cases):
                n = int(batch.node_counts[local_index])
                truth = dc.graph_to_truth(batch, local_index)
                known_pairs = dc.known_pairs_from_batch(batch, local_index)
                completed_pairs, metadata, _pred_matrix, weight_np = weighted_completed_pairs_from_prediction(
                    batch,
                    pred,
                    weight_multiplier,
                    local_index,
                    predicted_sigma_m=args.predicted_sigma,
                    predicted_sigma_slope=args.predicted_sigma_slope,
                    closest_predicted_pairs_per_anchor=args.closest_predicted_pairs_per_anchor,
                )
                missing = (~batch.measured_mask[local_index, :n, :n]) & batch.pair_mask[local_index, :n, :n]
                upper = np.triu(np.ones((n, n), dtype=bool), 1)
                model_weights = weight_np[missing.detach().cpu().numpy() & upper]
                median_weight = float(np.median(model_weights)) if model_weights.size else 1.0
                p05_weight = float(np.quantile(model_weights, 0.05)) if model_weights.size else 1.0
                p95_weight = float(np.quantile(model_weights, 0.95)) if model_weights.size else 1.0
                missing_mae = dc.missing_mae(batch, pred, local_index)
                for branch in ("weak", "completed"):
                    solver_pairs = branch_pair_set(branch, completed_pairs, args)
                    positions = solve_branch(branch, solver_pairs, known_pairs, args)
                    max_offset, median_offset, p95_offset = dc.offset_summary(truth, positions)
                    known_rmse, known_max = dc.pair_metrics(positions, known_pairs)
                    case_row = {
                        "bucket": case.bucket,
                        "bucket_key": next((key for key, value in BUCKET_LABELS.items() if value == case.bucket), case.bucket),
                        "case_index": global_case_index,
                        "family": case.family,
                        "shape": case.shape,
                        "anchors": n,
                    }
                    detail.append(
                        {
                            **case_row,
                            "branch": branch,
                            "known_pairs": len(known_pairs),
                            "spring_count": len(dc._preprocess_pairs(solver_pairs, min_sigma_m=0.02, min_distance_m=0.05)),
                            "missing_mae_m": missing_mae,
                            "median_model_weight_multiplier": median_weight,
                            "p05_model_weight_multiplier": p05_weight,
                            "p95_model_weight_multiplier": p95_weight,
                            "known_rmse_m": known_rmse,
                            "known_max_residual_m": known_max,
                            "max_offset_m": max_offset,
                            "median_offset_m": median_offset,
                            "p95_offset_m": p95_offset,
                        }
                    )
                    spring_rows.extend(spring_rows_for_branch(branch=branch, pairs=solver_pairs, metadata=metadata, case_row=case_row))
                print(f"case={global_case_index} bucket={case.bucket} anchors={n} weight_med={median_weight:.3f} done", flush=True)
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
            f"weight_med={float(row['median_model_weight_multiplier']):.3f}",
            flush=True,
        )
    print(f"Wrote {detail_path}", flush=True)
    print(f"Wrote {summary_path}", flush=True)
    print(f"Wrote {spring_path}", flush=True)


if __name__ == "__main__":
    main()
