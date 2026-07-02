from __future__ import annotations

import argparse
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


def parse_alphas(text: str) -> list[float]:
    return [float(part.strip()) for part in text.split(",") if part.strip()]


def solve_chunks(label: str, checkpoint_text: str, bucket_label: str, chunks, args) -> list[dict[str, str | int | float]]:
    solved = p95.solve_bucket(
        bucket_label,
        chunks,
        predicted_sigma=args.predicted_sigma,
        predicted_sigma_slope=args.predicted_sigma_slope,
        closest_predicted_pairs_per_anchor=args.closest_predicted_pairs_per_anchor,
        solver_iterations=args.solver_iterations,
        polish_iterations=args.polish_iterations,
        weak_polish_iterations=args.weak_polish_iterations,
        weak_completion_max_distance=args.weak_completion_max_distance,
        weak_completion_sigma_multiplier=args.weak_completion_sigma_multiplier,
    )
    return [fh.row_dict(label, Path(checkpoint_text), row) for row in solved]


def blended_chunks(base_chunks, candidate_chunks, alpha: float):
    chunks = []
    for (base_batch, base_pred, base_cases), (cand_batch, cand_pred, cand_cases) in zip(base_chunks, candidate_chunks):
        if base_batch.node_counts != cand_batch.node_counts:
            raise ValueError("Mismatched fixed batches while blending predictions.")
        pred = base_pred * (1.0 - alpha) + cand_pred * alpha
        chunks.append((base_batch, pred, base_cases))
    return chunks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep blends between baseline and candidate distance-completion checkpoints.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--baseline", required=True, help="Baseline checkpoint spec label=path")
    parser.add_argument("--candidate", action="append", required=True, help="Candidate checkpoint spec label=path")
    parser.add_argument("--alphas", default="0,0.15,0.25,0.35,0.5,0.65,0.8,1.0")
    parser.add_argument("--cases-per-bucket", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2026062703)
    parser.add_argument("--solver-iterations", type=int, default=80)
    parser.add_argument("--polish-iterations", type=int, default=80)
    parser.add_argument("--weak-polish-iterations", type=int, default=80)
    parser.add_argument("--predicted-sigma", type=float, default=0.55)
    parser.add_argument("--predicted-sigma-slope", type=float, default=0.65)
    parser.add_argument("--closest-predicted-pairs-per-anchor", type=float, default=5.0)
    parser.add_argument("--weak-completion-max-distance", type=float, default=18.0)
    parser.add_argument("--weak-completion-sigma-multiplier", type=float, default=2.5)
    parser.add_argument("--tolerance", type=float, default=1e-9)
    parser.add_argument("--prefix", default="anchor_solver_checkpoint_blend_sweep")
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
    baseline_label, baseline_path = fh.parse_checkpoint_spec(args.baseline)
    candidate_specs = [fh.parse_checkpoint_spec(item) for item in args.candidate]
    alphas = parse_alphas(args.alphas)
    print(f"device={device} baseline={baseline_label} candidates={len(candidate_specs)} alphas={alphas}", flush=True)

    frozen_by_bucket = {}
    for bucket_key in fh.BUCKET_KEYS:
        print(f"generating fixed bucket={bucket_key} cases={args.cases_per_bucket}", flush=True)
        cases = p95.generate_cases(bucket_key, args.cases_per_bucket, device)
        frozen_by_bucket[bucket_key] = fh.fixed_batches(cases, device=device, batch_size=args.batch_size)

    baseline_model = p95.load_model(baseline_path, device)
    baseline_preds = {bucket_key: fh.predict_fixed_batches(baseline_model, fixed) for bucket_key, fixed in frozen_by_bucket.items()}
    detail_rows: list[dict[str, str | int | float]] = []
    for bucket_key, bucket_label in fh.BUCKET_LABELS.items():
        print(f"solving baseline bucket={bucket_key}", flush=True)
        detail_rows.extend(solve_chunks(baseline_label, str(baseline_path), bucket_label, baseline_preds[bucket_key], args))

    for candidate_label, candidate_path in candidate_specs:
        candidate_model = p95.load_model(candidate_path, device)
        candidate_preds = {bucket_key: fh.predict_fixed_batches(candidate_model, fixed) for bucket_key, fixed in frozen_by_bucket.items()}
        for alpha in alphas:
            if abs(alpha) <= 1e-12:
                continue
            label = f"blend_{candidate_label}_a{alpha:.2f}".replace(".", "p")
            checkpoint_text = f"{baseline_path}+{candidate_path}@alpha={alpha:.3f}"
            for bucket_key, bucket_label in fh.BUCKET_LABELS.items():
                print(f"solving {label} bucket={bucket_key}", flush=True)
                chunks = blended_chunks(baseline_preds[bucket_key], candidate_preds[bucket_key], alpha)
                detail_rows.extend(solve_chunks(label, checkpoint_text, bucket_label, chunks, args))

    summary_rows = fh.summarize(detail_rows)
    gate_rows = fh.compare_to_baseline(summary_rows, baseline_label, args.tolerance)
    detail_path = OUTPUTS / f"{args.prefix}_detail.csv"
    summary_path = OUTPUTS / f"{args.prefix}_summary.csv"
    gate_path = OUTPUTS / f"{args.prefix}_gate.csv"
    fh.write_csv(detail_path, detail_rows)
    fh.write_csv(summary_path, summary_rows)
    if gate_rows:
        fh.write_csv(gate_path, gate_rows)
    for row in summary_rows:
        print(
            f"summary model={row['model']} bucket={row['bucket']} method={row['method']} "
            f"p95={float(row['p95_max_offset_m']):.3f}m median={float(row['median_max_offset_m']):.3f}m "
            f"under1m={float(row['under_1m']):.3f} missing_mae={float(row['median_missing_mae_m']):.3f}m",
            flush=True,
        )
    if gate_rows:
        failed = sum(1 for row in gate_rows if row["passed"] == "no")
        print(f"gate_checks={len(gate_rows)} failed={failed}", flush=True)
    print(f"Wrote {detail_path}", flush=True)
    print(f"Wrote {summary_path}", flush=True)
    if gate_rows:
        print(f"Wrote {gate_path}", flush=True)


if __name__ == "__main__":
    main()
