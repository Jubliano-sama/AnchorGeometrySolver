from __future__ import annotations

import argparse
import math
import random
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
sys.path.insert(0, str(ROOT / "work"))

import anchor_solver_fixed_holdout_compare as fh  # noqa: E402
import anchor_solver_ml_distance_completion as dc  # noqa: E402
import anchor_solver_p95_ml_cases as p95  # noqa: E402
from uwb_capture.anchor_geometry import AnchorPairDistance  # noqa: E402


def hybrid_completed_pairs(batch: dc.GraphBatch, value_pred: torch.Tensor, rank_pred: torch.Tensor, case_index: int, args) -> list[AnchorPairDistance]:
    n = batch.node_counts[case_index]
    scale = float(batch.scale_m[case_index].detach().cpu())
    measured = batch.measured_mask[case_index, :n, :n].detach().cpu().numpy()
    measured_dist = batch.measured_dist_m[case_index, :n, :n].detach().cpu().numpy()
    value_dist = (value_pred[case_index, :n, :n].detach().cpu().numpy() * scale).astype(float)
    rank_dist = (rank_pred[case_index, :n, :n].detach().cpu().numpy() * scale).astype(float)
    pairs: list[AnchorPairDistance] = []
    candidates: list[tuple[float, int, int, float, float]] = []
    for i in range(n):
        for j in range(i + 1, n):
            if measured[i, j]:
                pairs.append(AnchorPairDistance(f"A{i:02d}", f"A{j:02d}", float(measured_dist[i, j]), dc.KNOWN_SIGMA_M, True, "known"))
            else:
                value = max(float(value_dist[i, j]), 0.05)
                rank = max(float(rank_dist[i, j]), 0.05)
                sigma = args.predicted_sigma * (1.0 + args.predicted_sigma_slope * max(value - dc.EDGE_RADIUS_M, 0.0) / dc.EDGE_RADIUS_M)
                candidates.append((rank, i, j, value, sigma))
    limit = len(candidates)
    if args.closest_predicted_pairs > 0:
        limit = min(limit, args.closest_predicted_pairs)
    if args.closest_predicted_pairs_per_anchor > 0.0:
        limit = min(limit, max(0, int(math.ceil(args.closest_predicted_pairs_per_anchor * n))))
    candidates.sort(key=lambda item: (item[0], item[1], item[2]))
    for _rank, i, j, value, sigma in candidates[:limit]:
        pairs.append(AnchorPairDistance(f"A{i:02d}", f"A{j:02d}", value, sigma, True, "hybrid"))
    return pairs


def solve_hybrid_bucket(bucket_label: str, value_chunks, rank_chunks, *, label: str, checkpoint_text: str, args) -> list[dict[str, str | int | float]]:
    rows: list[dict[str, str | int | float]] = []
    global_case_index = 0
    for (batch, value_pred, _cases), (_rank_batch, rank_pred, _rank_cases) in zip(value_chunks, rank_chunks):
        for case_index in range(len(batch.node_counts)):
            truth = dc.graph_to_truth(batch, case_index)
            known_pairs = dc.known_pairs_from_batch(batch, case_index)
            completed_pairs = hybrid_completed_pairs(batch, value_pred, rank_pred, case_index, args)
            oracle_pairs = dc.oracle_full_pairs(batch, case_index)
            mae = dc.missing_mae(batch, value_pred, case_index)
            solutions = {
                "ML completed": dc.completion_solution(completed_pairs, known_pairs, max_iterations=args.solver_iterations, polish_known_iterations=args.polish_iterations),
                "ML weak polish": dc.completion_solution_weak_polish(
                    completed_pairs,
                    known_pairs,
                    max_iterations=args.solver_iterations,
                    weak_polish_iterations=args.weak_polish_iterations,
                    max_predicted_distance_m=args.weak_completion_max_distance,
                    sigma_multiplier=args.weak_completion_sigma_multiplier,
                ),
            }
            for method, estimate in solutions.items():
                known_rmse, known_max = dc.pair_metrics(estimate, known_pairs)
                max_offset, median_offset, p95_offset = dc.offset_summary(truth, estimate)
                row = p95.SolvedCase(
                    bucket=bucket_label,
                    method=method,
                    case_index=global_case_index,
                    family=batch.family[case_index],
                    shape=batch.shape[case_index],
                    anchors=batch.node_counts[case_index],
                    known_pairs=len(known_pairs),
                    full_pairs=len(oracle_pairs),
                    missing_mae_m=mae,
                    known_rmse_m=known_rmse,
                    known_max_residual_m=known_max,
                    max_offset_m=max_offset,
                    median_offset_m=median_offset,
                    p95_offset_m=p95_offset,
                    truth=truth,
                    known_pairs_list=known_pairs,
                    estimate=estimate,
                )
                rows.append(fh.row_dict(label, Path(checkpoint_text), row))
            global_case_index += 1
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate hybrid ML completion: baseline values with candidate ranking.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--value", required=True, help="Value checkpoint label=path, usually baseline")
    parser.add_argument("--rank", action="append", required=True, help="Rank checkpoint label=path")
    parser.add_argument("--cases-per-bucket", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2026062703)
    parser.add_argument("--solver-iterations", type=int, default=80)
    parser.add_argument("--polish-iterations", type=int, default=80)
    parser.add_argument("--weak-polish-iterations", type=int, default=80)
    parser.add_argument("--predicted-sigma", type=float, default=0.55)
    parser.add_argument("--predicted-sigma-slope", type=float, default=0.65)
    parser.add_argument("--closest-predicted-pairs", type=int, default=0)
    parser.add_argument("--closest-predicted-pairs-per-anchor", type=float, default=5.0)
    parser.add_argument("--weak-completion-max-distance", type=float, default=18.0)
    parser.add_argument("--weak-completion-sigma-multiplier", type=float, default=2.5)
    parser.add_argument("--tolerance", type=float, default=1e-9)
    parser.add_argument("--prefix", default="anchor_solver_hybrid_rank_value_eval")
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed % (2**32 - 1))
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = p95.choose_device(args.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
    value_label, value_path = fh.parse_checkpoint_spec(args.value)
    rank_specs = [fh.parse_checkpoint_spec(item) for item in args.rank]
    print(f"device={device} value={value_label} rank_models={len(rank_specs)}", flush=True)
    frozen = {}
    for bucket_key in fh.BUCKET_KEYS:
        print(f"generating fixed bucket={bucket_key} cases={args.cases_per_bucket}", flush=True)
        cases = p95.generate_cases(bucket_key, args.cases_per_bucket, device)
        frozen[bucket_key] = fh.fixed_batches(cases, device=device, batch_size=args.batch_size)
    value_model = p95.load_model(value_path, device)
    value_preds = {bucket: fh.predict_fixed_batches(value_model, fixed) for bucket, fixed in frozen.items()}
    detail: list[dict[str, str | int | float]] = []
    for bucket_key, bucket_label in fh.BUCKET_LABELS.items():
        detail.extend(solve_hybrid_bucket(bucket_label, value_preds[bucket_key], value_preds[bucket_key], label=value_label, checkpoint_text=str(value_path), args=args))
    for rank_label, rank_path in rank_specs:
        rank_model = p95.load_model(rank_path, device)
        rank_preds = {bucket: fh.predict_fixed_batches(rank_model, fixed) for bucket, fixed in frozen.items()}
        label = f"hybrid_value_{value_label}_rank_{rank_label}"
        checkpoint_text = f"value={value_path};rank={rank_path}"
        for bucket_key, bucket_label in fh.BUCKET_LABELS.items():
            print(f"solving {label} bucket={bucket_key}", flush=True)
            detail.extend(solve_hybrid_bucket(bucket_label, value_preds[bucket_key], rank_preds[bucket_key], label=label, checkpoint_text=checkpoint_text, args=args))
    summary = fh.summarize(detail)
    gate = fh.compare_to_baseline(summary, value_label, args.tolerance)
    detail_path = OUTPUTS / f"{args.prefix}_detail.csv"
    summary_path = OUTPUTS / f"{args.prefix}_summary.csv"
    gate_path = OUTPUTS / f"{args.prefix}_gate.csv"
    fh.write_csv(detail_path, detail)
    fh.write_csv(summary_path, summary)
    if gate:
        fh.write_csv(gate_path, gate)
    for row in summary:
        print(
            f"summary model={row['model']} bucket={row['bucket']} method={row['method']} p95={float(row['p95_max_offset_m']):.3f}m "
            f"median={float(row['median_max_offset_m']):.3f}m under1m={float(row['under_1m']):.3f} missing_mae={float(row['median_missing_mae_m']):.3f}m",
            flush=True,
        )
    if gate:
        print(f"gate_checks={len(gate)} failed={sum(1 for row in gate if row['passed'] == 'no')}", flush=True)
    print(f"Wrote {detail_path}", flush=True)
    print(f"Wrote {summary_path}", flush=True)
    if gate:
        print(f"Wrote {gate_path}", flush=True)


if __name__ == "__main__":
    main()
