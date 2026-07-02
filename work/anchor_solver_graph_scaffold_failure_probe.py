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

import anchor_solver_graph_scaffold_eval as ge
import anchor_solver_fold_rescue_experiment as exp
import anchor_solver_p95_ml_cases as p95
import anchor_solver_ml_distance_completion as dc


def read_failed(detail_path: Path, threshold: float, cases_per_bucket: int) -> dict[str, set[int]]:
    failed: dict[str, set[int]] = {"random": set(), "grid": set(), "office": set()}
    with detail_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["method"] != "graph-shortest-scaffold":
                continue
            if float(row["max_offset_m"]) <= threshold:
                continue
            global_index = int(row["case_index"])
            if row["bucket"].startswith("Random"):
                failed["random"].add(global_index)
            elif row["bucket"].startswith("Grid"):
                failed["grid"].add(global_index - cases_per_bucket)
            elif row["bucket"].startswith("Office"):
                failed["office"].add(global_index - 2 * cases_per_bucket)
    return failed


def summarize(rows: list[dict[str, float | int | str]]) -> list[dict[str, float | int | str]]:
    out: list[dict[str, float | int | str]] = []
    keys = sorted({(str(r["config"]), str(r["bucket"])) for r in rows})
    for config, bucket in keys:
        part = [r for r in rows if r["config"] == config and r["bucket"] == bucket]
        offsets = np.array([float(r["max_offset_m"]) for r in part], dtype=float)
        rmse = np.array([float(r["known_rmse_m"]) for r in part], dtype=float)
        out.append(
            {
                "config": config,
                "bucket": bucket,
                "cases": len(part),
                "median_max_offset_m": float(np.median(offsets)),
                "p90_max_offset_m": float(np.quantile(offsets, 0.90)),
                "p95_max_offset_m": float(np.quantile(offsets, 0.95)),
                "max_offset_m": float(np.max(offsets)),
                "under_1m": float(np.mean(offsets <= 1.0)),
                "median_known_rmse_m": float(np.median(rmse)),
            }
        )
    return out


def write_csv(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe graph scaffold on previously failed cases.")
    parser.add_argument("--detail", type=Path, default=OUTPUTS / "anchor_solver_graph_scaffold_eval_rgo32_cap32_h2_detail.csv")
    parser.add_argument("--threshold", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=2026062707)
    parser.add_argument("--cases-per-bucket", type=int, default=32)
    parser.add_argument("--seed-count", type=int, default=6)
    parser.add_argument("--iterations", type=int, default=45)
    parser.add_argument("--prefix", default="anchor_solver_graph_scaffold_failure_probe")
    args = parser.parse_args()

    OUTPUTS.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed % (2**32 - 1))
    torch.manual_seed(args.seed)
    device = torch.device("cpu")
    failed = read_failed(args.detail, args.threshold, args.cases_per_bucket)
    configs = [
        ("h2_rs036", 2, 0.36),
        ("h3_rs030", 3, 0.30),
        ("h3_rs045", 3, 0.45),
        ("h4_rs030", 4, 0.30),
        ("h8_rs040", 8, 0.40),
    ]
    detail: list[dict[str, float | int | str]] = []
    case_counter = 0
    for bucket_key in ("random", "grid", "office"):
        cases = ge.generate_limited_cases(bucket_key, args.cases_per_bucket, device, max_nodes=32)
        for local_index, case in enumerate(cases):
            if local_index not in failed[bucket_key]:
                case_counter += 1
                continue
            batch = p95.graph_batch_from_cases([case], device)
            truth = dc.graph_to_truth(batch, 0)
            known_pairs = dc.known_pairs_from_batch(batch, 0)
            print(f"case bucket={bucket_key} local={local_index} global={case_counter} anchors={len(truth)} known={len(known_pairs)}", flush=True)
            for ci, (name, hops, relative_sigma) in enumerate(configs):
                positions = exp.graph_shortest_scaffold_solve(
                    known_pairs,
                    seed_count=args.seed_count,
                    iterations=args.iterations,
                    rng_seed=args.seed + case_counter * 1009 + 19 + ci * 100003,
                    max_hops=hops,
                    relative_sigma=relative_sigma,
                )
                row = exp.result_to_dict(exp.metrics_row(case.bucket, case_counter, case.family, case.shape, name, truth, positions, known_pairs, fold_threshold_m=1.65))
                row["config"] = name
                row["max_hops"] = hops
                row["relative_sigma"] = relative_sigma
                detail.append(row)
                print(f"  {name}: max={float(row['max_offset_m']):.3f}m median={float(row['median_offset_m']):.3f} rmse={float(row['known_rmse_m']):.4f}", flush=True)
            case_counter += 1
    summary = summarize(detail)
    detail_path = OUTPUTS / f"{args.prefix}_detail.csv"
    summary_path = OUTPUTS / f"{args.prefix}_summary.csv"
    write_csv(detail_path, detail)
    write_csv(summary_path, summary)
    for row in sorted(summary, key=lambda r: (str(r["bucket"]), float(r["p95_max_offset_m"]))):
        print(
            f"summary bucket={row['bucket']} config={row['config']} cases={row['cases']} "
            f"p95={float(row['p95_max_offset_m']):.3f} max={float(row['max_offset_m']):.3f} under1={float(row['under_1m']):.3f} rmse={float(row['median_known_rmse_m']):.4f}",
            flush=True,
        )
    print(f"Wrote {detail_path}", flush=True)
    print(f"Wrote {summary_path}", flush=True)


if __name__ == "__main__":
    main()

