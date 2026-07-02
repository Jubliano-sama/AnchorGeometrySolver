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
import anchor_solver_ml_distance_completion as dc
import anchor_solver_p95_ml_cases as p95

BUCKET_KEYS = ("random", "grid", "office")
BUCKET_LABELS = {"random": "Random 16-32", "grid": "Grid >=16", "office": "Office >=16"}


def parse_checkpoint_spec(text: str) -> tuple[str, Path]:
    if "=" in text:
        label, raw = text.split("=", 1)
        return label, Path(raw)
    path = Path(text)
    return path.stem, path


def write_csv(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict[str, float | int | str]]) -> list[dict[str, float | int | str]]:
    summary: list[dict[str, float | int | str]] = []
    for bucket, method in sorted({(str(r["bucket"]), str(r["method"])) for r in rows}):
        part = [r for r in rows if r["bucket"] == bucket and r["method"] == method]
        offsets = np.array([float(r["max_offset_m"]) for r in part], dtype=float)
        rmses = np.array([float(r["known_rmse_m"]) for r in part], dtype=float)
        missing = np.array([float(r["missing_mae_m"]) for r in part if str(r["missing_mae_m"]) != ""], dtype=float)
        close = np.array([float(r["close_pair_count"]) for r in part], dtype=float)
        summary.append({
            "bucket": bucket,
            "method": method,
            "cases": len(part),
            "median_missing_mae_m": float(np.median(missing)) if missing.size else "",
            "median_max_offset_m": float(np.median(offsets)),
            "p90_max_offset_m": float(np.quantile(offsets, 0.90)),
            "p95_max_offset_m": float(np.quantile(offsets, 0.95)),
            "max_offset_m": float(np.max(offsets)),
            "under_20cm": float(np.mean(offsets <= 0.20)),
            "under_50cm": float(np.mean(offsets <= 0.50)),
            "under_1m": float(np.mean(offsets <= 1.00)),
            "median_known_rmse_m": float(np.median(rmses)),
            "folded_case_rate": float(np.mean(close > 0)),
        })
    return summary


def metric_row(bucket: str, method: str, case_index: int, family: str, shape: str, truth, positions, known_pairs, missing_mae: float | str) -> dict[str, float | int | str]:
    result = exp.metrics_row(bucket, case_index, family, shape, method, truth, positions, known_pairs, fold_threshold_m=1.65)
    row = exp.result_to_dict(result)
    row["missing_mae_m"] = missing_mae
    return row


def solve_ml(batch: dc.GraphBatch, pred: torch.Tensor, case_index: int, *, branch: str) -> tuple[dict[str, tuple[float, float]], float]:
    known_pairs = dc.known_pairs_from_batch(batch, case_index)
    completed_pairs, _matrix, _scale = dc.completed_pairs_from_prediction(
        batch,
        pred,
        case_index,
        predicted_sigma_m=0.55,
        predicted_sigma_slope=0.65,
        closest_predicted_pairs_per_anchor=5.0,
    )
    if branch == "completed":
        positions = dc.completion_solution(completed_pairs, known_pairs, max_iterations=80, polish_known_iterations=80)
    elif branch == "weak":
        positions = dc.completion_solution_weak_polish(
            completed_pairs,
            known_pairs,
            max_iterations=80,
            weak_polish_iterations=80,
            max_predicted_distance_m=18.0,
            sigma_multiplier=2.5,
        )
    else:
        raise ValueError(branch)
    return positions, dc.missing_mae(batch, pred, case_index)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fair same-batch comparison of ML and graph-shortest scaffold.")
    parser.add_argument("--checkpoint", action="append", required=True)
    parser.add_argument("--cases-per-bucket", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2026062703)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--prefix", default="anchor_solver_fair_ml_graph_compare")
    args = parser.parse_args()

    OUTPUTS.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed % (2**32 - 1))
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    gen_device = torch.device("cpu")
    eval_device = p95.choose_device(args.device)
    models = [(label, p95.load_model(path, eval_device)) for label, path in (parse_checkpoint_spec(item) for item in args.checkpoint)]

    detail: list[dict[str, float | int | str]] = []
    for bucket_key in BUCKET_KEYS:
        print(f"generating bucket={bucket_key} cases={args.cases_per_bucket}", flush=True)
        cases = p95.generate_cases(bucket_key, args.cases_per_bucket, gen_device)
        global_case_index = 0
        for start in range(0, len(cases), args.batch_size):
            chunk_cases = cases[start : start + args.batch_size]
            batch = p95.graph_batch_from_cases(chunk_cases, eval_device)
            preds = [(label, model(batch.node_features, batch.edge_features, batch.mask, batch.measured_mask, batch.pair_mask)) for label, model in models]
            for case_index, case in enumerate(chunk_cases):
                truth = dc.graph_to_truth(batch, case_index)
                known_pairs = dc.known_pairs_from_batch(batch, case_index)
                bucket = BUCKET_LABELS[bucket_key]
                graph_positions = exp.graph_shortest_scaffold_solve(
                    known_pairs,
                    seed_count=6,
                    iterations=45,
                    rng_seed=args.seed + (start + case_index) * 1009 + 19,
                    max_hops=2,
                    relative_sigma=0.36,
                )
                detail.append(metric_row(bucket, "graph-shortest-scaffold", global_case_index, case.family, case.shape, truth, graph_positions, known_pairs, ""))
                for label, pred in preds:
                    for branch in ("completed", "weak"):
                        positions, mae = solve_ml(batch, pred, case_index, branch=branch)
                        detail.append(metric_row(bucket, f"{label}:ML {branch}", global_case_index, case.family, case.shape, truth, positions, known_pairs, mae))
                print(f"case bucket={bucket_key} index={global_case_index} anchors={len(truth)}", flush=True)
                global_case_index += 1

    summary = summarize(detail)
    detail_path = OUTPUTS / f"{args.prefix}_detail.csv"
    summary_path = OUTPUTS / f"{args.prefix}_summary.csv"
    write_csv(detail_path, detail)
    write_csv(summary_path, summary)
    for row in summary:
        print(
            f"summary bucket={row['bucket']} method={row['method']} p95={float(row['p95_max_offset_m']):.3f}m "
            f"median={float(row['median_max_offset_m']):.3f}m under1={float(row['under_1m']):.3f} "
            f"missing={row['median_missing_mae_m']}",
            flush=True,
        )
    print(f"Wrote {detail_path}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
