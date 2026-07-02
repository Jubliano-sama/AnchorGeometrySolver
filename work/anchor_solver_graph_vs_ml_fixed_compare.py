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
        rmse = np.array([float(r["known_rmse_m"]) for r in part], dtype=float)
        close = np.array([float(r["close_pair_count"]) for r in part], dtype=float)
        summary.append({
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
        })
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare graph-shortest scaffold on the same fixed heldout cases as ML.")
    parser.add_argument("--cases-per-bucket", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2026062703)
    parser.add_argument("--seed-count", type=int, default=6)
    parser.add_argument("--iterations", type=int, default=45)
    parser.add_argument("--max-hops", type=int, default=2)
    parser.add_argument("--relative-sigma", type=float, default=0.36)
    parser.add_argument("--scaffold-weight", type=float, default=1.0)
    parser.add_argument("--hop-weight-base", type=float, default=1.0)
    parser.add_argument("--prefix", default="anchor_solver_graph_vs_ml_fixed_compare")
    args = parser.parse_args()

    OUTPUTS.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed % (2**32 - 1))
    torch.manual_seed(args.seed)
    device = torch.device("cpu")
    detail: list[dict[str, float | int | str]] = []
    global_offset = 0
    for bucket_key in BUCKET_KEYS:
        print(f"generating bucket={bucket_key} cases={args.cases_per_bucket}", flush=True)
        cases = p95.generate_cases(bucket_key, args.cases_per_bucket, device)
        for local_index, case in enumerate(cases):
            batch = p95.graph_batch_from_cases([case], device)
            truth = dc.graph_to_truth(batch, 0)
            known_pairs = dc.known_pairs_from_batch(batch, 0)
            print(f"solving graph bucket={bucket_key} case={local_index + 1}/{len(cases)} anchors={len(truth)}", flush=True)
            positions = exp.graph_shortest_scaffold_solve(
                known_pairs,
                seed_count=args.seed_count,
                iterations=args.iterations,
                rng_seed=args.seed + (global_offset + local_index) * 1009 + 19,
                max_hops=args.max_hops,
                relative_sigma=args.relative_sigma,
                scaffold_weight=args.scaffold_weight,
                hop_weight_base=args.hop_weight_base,
            )
            row = exp.metrics_row(
                BUCKET_LABELS[bucket_key],
                local_index,
                case.family,
                case.shape,
                "graph-shortest-scaffold",
                truth,
                positions,
                known_pairs,
                fold_threshold_m=1.65,
            )
            detail.append(exp.result_to_dict(row))
        global_offset += args.cases_per_bucket

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
    print(f"Wrote {detail_path}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
