from __future__ import annotations

import argparse
import csv
from pathlib import Path
import random
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
sys.path.insert(0, str(ROOT / "work"))

import anchor_solver_fold_rescue_experiment as exp
import anchor_solver_fixed_holdout_compare as fixed
import anchor_solver_ml_distance_completion as dc
import anchor_solver_p95_ml_cases as p95

BUCKET_KEYS = ("random", "grid", "office")
BUCKET_LABELS = {"random": "Random 16-32", "grid": "Grid >=16", "office": "Office >=16"}


def parse_checkpoint_spec(text: str) -> tuple[str, Path]:
    if "=" in text:
        label, path = text.split("=", 1)
        return label, Path(path)
    path = Path(text)
    return path.stem, path


def metric_row(bucket: str, case_index: int, family: str, shape: str, method: str, truth, positions, known_pairs, selected_from: str, score: float) -> dict[str, float | int | str]:
    result = exp.metrics_row(bucket, case_index, family, shape, method, truth, positions, known_pairs, fold_threshold_m=1.65)
    row = exp.result_to_dict(result)
    row["selected_from"] = selected_from
    row["selection_score"] = score
    return row


def summarize(rows: list[dict[str, float | int | str]]) -> list[dict[str, float | int | str]]:
    summary: list[dict[str, float | int | str]] = []
    for bucket, method in sorted({(str(r["bucket"]), str(r["method"])) for r in rows}):
        part = [r for r in rows if r["bucket"] == bucket and r["method"] == method]
        offsets = np.array([float(r["max_offset_m"]) for r in part], dtype=float)
        rmse = np.array([float(r["known_rmse_m"]) for r in part], dtype=float)
        close = np.array([float(r["close_pair_count"]) for r in part], dtype=float)
        summary.append(
            {
                "bucket": bucket,
                "method": method,
                "cases": len(part),
                "median_max_offset_m": float(np.median(offsets)),
                "p90_max_offset_m": float(np.quantile(offsets, 0.90)),
                "p95_max_offset_m": float(np.quantile(offsets, 0.95)),
                "max_offset_m": float(np.max(offsets)),
                "under_20cm": float(np.mean(offsets <= 0.20)),
                "under_50cm": float(np.mean(offsets <= 0.50)),
                "under_1m": float(np.mean(offsets <= 1.00)),
                "median_known_rmse_m": float(np.median(rmse)),
                "folded_case_rate": float(np.mean(close > 0)),
            }
        )
    return summary


def write_csv(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Select among ML and graph solver candidates using observable topology scores.")
    parser.add_argument("--checkpoint", action="append", required=True)
    parser.add_argument("--cases-per-bucket", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2026062703)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--predicted-sigma", type=float, default=0.55)
    parser.add_argument("--predicted-sigma-slope", type=float, default=0.65)
    parser.add_argument("--closest-predicted-pairs-per-anchor", type=float, default=5.0)
    parser.add_argument("--solver-iterations", type=int, default=80)
    parser.add_argument("--polish-iterations", type=int, default=80)
    parser.add_argument("--weak-polish-iterations", type=int, default=80)
    parser.add_argument("--prefix", default="anchor_solver_candidate_selector_eval")
    args = parser.parse_args()

    OUTPUTS.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed % (2**32 - 1))
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = p95.choose_device(args.device)
    checkpoint_specs = [parse_checkpoint_spec(item) for item in args.checkpoint]
    models = [(label, p95.load_model(path, device)) for label, path in checkpoint_specs]

    detail: list[dict[str, float | int | str]] = []
    for bucket_key in BUCKET_KEYS:
        print(f"generating bucket={bucket_key} cases={args.cases_per_bucket}", flush=True)
        cases = p95.generate_cases(bucket_key, args.cases_per_bucket, device)
        chunks_by_model: dict[str, list[tuple[dc.GraphBatch, torch.Tensor, list[p95.CaseSpec]]]] = {}
        for label, model in models:
            print(f"predicting model={label} bucket={bucket_key}", flush=True)
            chunks_by_model[label] = p95.predict_batches(model, cases, device=device, batch_size=args.batch_size)

        # Reuse batches from first model for known pairs/truth. Predictions do not alter the batch.
        first_label = models[0][0]
        global_index = 0
        for chunk_i, (batch, _first_pred, chunk_cases) in enumerate(chunks_by_model[first_label]):
            for case_index in range(len(batch.node_counts)):
                truth = dc.graph_to_truth(batch, case_index)
                known_pairs = dc.known_pairs_from_batch(batch, case_index)
                candidates: list[tuple[str, dict[str, tuple[float, float]], float]] = []
                for label, _model in models:
                    model_batch, pred, _cases = chunks_by_model[label][chunk_i]
                    completed_pairs, _pred_matrix, _scale = dc.completed_pairs_from_prediction(
                        model_batch,
                        pred,
                        case_index,
                        predicted_sigma_m=args.predicted_sigma,
                        predicted_sigma_slope=args.predicted_sigma_slope,
                        closest_predicted_pairs_per_anchor=args.closest_predicted_pairs_per_anchor,
                    )
                    ml_completed = dc.completion_solution(
                        completed_pairs,
                        known_pairs,
                        max_iterations=args.solver_iterations,
                        polish_known_iterations=args.polish_iterations,
                    )
                    ml_weak = dc.completion_solution_weak_polish(
                        completed_pairs,
                        known_pairs,
                        max_iterations=args.solver_iterations,
                        weak_polish_iterations=args.weak_polish_iterations,
                        max_predicted_distance_m=18.0,
                        sigma_multiplier=2.5,
                    )
                    candidates.append((f"{label}:ML completed", ml_completed, exp.topology_selection_score(ml_completed, known_pairs, fold_threshold_m=1.65)))
                    candidates.append((f"{label}:ML weak", ml_weak, exp.topology_selection_score(ml_weak, known_pairs, fold_threshold_m=1.65)))
                for name, hops, rs, hs in (("graph_h2", 2, 0.36, 0.65), ("graph_h4", 4, 0.30, 0.85)):
                    positions = exp.graph_shortest_scaffold_solve(
                        known_pairs,
                        seed_count=6,
                        iterations=45,
                        rng_seed=args.seed + global_index * 1009 + hops * 101,
                        max_hops=hops,
                        relative_sigma=rs,
                    )
                    candidates.append((name, positions, exp.topology_selection_score(positions, known_pairs, fold_threshold_m=1.65)))

                selected_name, selected_positions, selected_score = min(candidates, key=lambda item: item[2])
                oracle_name, oracle_positions, oracle_score = min(candidates, key=lambda item: dc.offset_summary(truth, item[1])[0])
                bucket = BUCKET_LABELS[bucket_key]
                family = batch.family[case_index]
                shape = batch.shape[case_index]
                detail.append(metric_row(bucket, global_index, family, shape, "selector_topology", truth, selected_positions, known_pairs, selected_name, selected_score))
                detail.append(metric_row(bucket, global_index, family, shape, "oracle_best_candidate", truth, oracle_positions, known_pairs, oracle_name, oracle_score))
                for name, positions, score in candidates:
                    if name.endswith(":ML completed") and (name.startswith("baseline:") or name.startswith("bigD:") or name.startswith("bigF:")):
                        detail.append(metric_row(bucket, global_index, family, shape, name, truth, positions, known_pairs, name, score))
                print(f"case bucket={bucket_key} index={global_index} selected={selected_name} oracle={oracle_name}", flush=True)
                global_index += 1

    summary = summarize(detail)
    detail_path = OUTPUTS / f"{args.prefix}_detail.csv"
    summary_path = OUTPUTS / f"{args.prefix}_summary.csv"
    write_csv(detail_path, detail)
    write_csv(summary_path, summary)
    for row in summary:
        print(
            f"summary bucket={row['bucket']} method={row['method']} p95={float(row['p95_max_offset_m']):.3f}m "
            f"median={float(row['median_max_offset_m']):.3f}m under1={float(row['under_1m']):.3f} rmse={float(row['median_known_rmse_m']):.4f}",
            flush=True,
        )
    print(f"Wrote {detail_path}", flush=True)
    print(f"Wrote {summary_path}", flush=True)


if __name__ == "__main__":
    main()
