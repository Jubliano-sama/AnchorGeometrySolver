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

import anchor_solver_fold_rescue_experiment as exp
import anchor_solver_ml_distance_completion as dc
import anchor_solver_p95_ml_cases as p95
from uwb_capture.anchor_geometry import AnchorPairDistance

BUCKET_KEYS = ("random", "grid", "office")
BUCKET_LABELS = {"random": "Random 16-32", "grid": "Grid >=16", "office": "Office >=16"}
INF = 1e12


def parse_floats(text: str) -> list[float]:
    return [float(part.strip()) for part in text.split(",") if part.strip()]


def transformed_scaffold_pairs(processed, anchor_ids, *, max_hops: int, relative_sigma: float, multiplier: float, power: float):
    # Start from the existing implementation so the graph path and sigma logic stays comparable.
    base = exp.graph_shortest_scaffold_pairs(
        processed,
        anchor_ids,
        max_hops=max_hops,
        relative_sigma=relative_sigma,
    )
    out: list[AnchorPairDistance] = []
    for pair in base:
        shortest = max(float(pair.distance_m), 0.05)
        transformed = dc.EDGE_RADIUS_M * multiplier * (shortest / dc.EDGE_RADIUS_M) ** power
        out.append(
            AnchorPairDistance(
                pair.anchor_a_id,
                pair.anchor_b_id,
                max(transformed, 0.05),
                sigma_m=pair.sigma_m,
                enabled=True,
                source=f"graph-transform-m{multiplier:g}-p{power:g}",
            )
        )
    return out


def solve_transformed_graph(known_pairs, *, seed_count: int, iterations: int, rng_seed: int, max_hops: int, relative_sigma: float, multiplier: float, power: float):
    rng = random.Random(rng_seed)
    processed = exp._preprocess_pairs(known_pairs, min_sigma_m=0.02, min_distance_m=0.05)
    anchor_ids = exp._anchor_ids(processed)
    scaffold = transformed_scaffold_pairs(
        processed,
        anchor_ids,
        max_hops=max_hops,
        relative_sigma=relative_sigma,
        multiplier=multiplier,
        power=power,
    )
    if not scaffold:
        return exp.best_distance_only_solve(known_pairs, seed_count=seed_count, iterations=iterations, rng=rng)
    augmented = exp._preprocess_pairs([*known_pairs, *scaffold], min_sigma_m=0.02, min_distance_m=0.05)
    parameterization = exp._Parameterization(anchor_ids)
    scale = exp._layout_scale(augmented)
    seeds = exp._initial_parameters(parameterization, augmented, seed_count=max(seed_count, 1), scale=scale, rng=rng)
    try:
        mds_positions = dc.classical_mds_seed([*known_pairs, *scaffold])
        seeds.insert(0, exp._positions_to_params(parameterization, mds_positions))
    except Exception:
        pass
    known_priors = exp.production_priors(anchor_ids, processed, exp._layout_scale(processed))
    scaffold_priors = exp.production_priors(anchor_ids, processed, scale)
    best_positions = None
    best_score = math.inf
    for seed in seeds:
        scaffold_params, _energy = exp._local_minimize(
            seed,
            parameterization,
            augmented,
            scaffold_priors,
            max_iterations=max(iterations // 2, 25),
        )
        scaffold_positions = parameterization.to_positions(scaffold_params)
        prior_positions = exp.internal_solve_from_positions(
            scaffold_positions,
            processed,
            parameterization,
            priors=known_priors,
            iterations=max(iterations, 25),
        )
        polished = exp.internal_solve_from_positions(
            prior_positions,
            processed,
            parameterization,
            priors=None,
            iterations=max(iterations // 2, 25),
        )
        for candidate in (prior_positions, polished):
            score = exp.topology_selection_score(candidate, known_pairs, fold_threshold_m=1.65)
            if score < best_score:
                best_score = score
                best_positions = candidate
    assert best_positions is not None
    return best_positions


def summarize(rows):
    out = []
    keys = sorted({(r["bucket"], r["config"], r["multiplier"], r["power"]) for r in rows})
    for bucket, config, multiplier, power in keys:
        part = [r for r in rows if r["bucket"] == bucket and r["config"] == config and r["multiplier"] == multiplier and r["power"] == power]
        offsets = np.array([float(r["max_offset_m"]) for r in part])
        rmses = np.array([float(r["known_rmse_m"]) for r in part])
        out.append({
            "bucket": bucket,
            "config": config,
            "multiplier": multiplier,
            "power": power,
            "cases": len(part),
            "median_max_offset_m": float(np.median(offsets)),
            "p90_max_offset_m": float(np.quantile(offsets, 0.90)),
            "p95_max_offset_m": float(np.quantile(offsets, 0.95)),
            "max_offset_m": float(np.max(offsets)),
            "under_50cm": float(np.mean(offsets <= 0.50)),
            "under_1m": float(np.mean(offsets <= 1.0)),
            "median_known_rmse_m": float(np.median(rmses)),
        })
    return out


def write_csv(path: Path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Sweep multiplier/power transforms of graph shortest path scaffold distances.")
    parser.add_argument("--cases-per-bucket", type=int, default=6)
    parser.add_argument("--seed", type=int, default=2026062711)
    parser.add_argument("--seed-count", type=int, default=4)
    parser.add_argument("--iterations", type=int, default=35)
    parser.add_argument("--multipliers", default="0.75,0.9,1.0,1.15,1.35")
    parser.add_argument("--powers", default="0.85,1.0,1.15")
    parser.add_argument("--max-hops", type=int, default=2)
    parser.add_argument("--relative-sigma", type=float, default=0.36)
    parser.add_argument("--prefix", default="anchor_solver_graph_power_sweep_quick")
    args = parser.parse_args()
    OUTPUTS.mkdir(exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed % (2**32 - 1))
    torch.manual_seed(args.seed)
    device = torch.device("cpu")
    multipliers = parse_floats(args.multipliers)
    powers = parse_floats(args.powers)
    detail = []
    case_counter = 0
    for bucket_key in BUCKET_KEYS:
        cases = p95.generate_cases(bucket_key, args.cases_per_bucket, device)
        print(f"bucket={bucket_key} cases={len(cases)}", flush=True)
        for local_index, case in enumerate(cases):
            batch = p95.graph_batch_from_cases([case], device)
            truth = dc.graph_to_truth(batch, 0)
            known_pairs = dc.known_pairs_from_batch(batch, 0)
            for multiplier in multipliers:
                for power in powers:
                    positions = solve_transformed_graph(
                        known_pairs,
                        seed_count=args.seed_count,
                        iterations=args.iterations,
                        rng_seed=args.seed + case_counter * 1009 + int(multiplier * 1000) + int(power * 10000),
                        max_hops=args.max_hops,
                        relative_sigma=args.relative_sigma,
                        multiplier=multiplier,
                        power=power,
                    )
                    result = exp.metrics_row(BUCKET_LABELS[bucket_key], case_counter, case.family, case.shape, "graph-power", truth, positions, known_pairs, fold_threshold_m=1.65)
                    row = exp.result_to_dict(result)
                    row["config"] = f"m{multiplier:g}_p{power:g}"
                    row["multiplier"] = multiplier
                    row["power"] = power
                    detail.append(row)
            print(f"case={case_counter} bucket={bucket_key} anchors={len(truth)} done", flush=True)
            case_counter += 1
    summary = summarize(detail)
    detail_path = OUTPUTS / f"{args.prefix}_detail.csv"
    summary_path = OUTPUTS / f"{args.prefix}_summary.csv"
    write_csv(detail_path, detail)
    write_csv(summary_path, summary)
    for bucket in [BUCKET_LABELS[k] for k in BUCKET_KEYS]:
        best = min([r for r in summary if r["bucket"] == bucket], key=lambda r: float(r["p95_max_offset_m"]))
        print(f"best bucket={bucket} config={best['config']} p95={float(best['p95_max_offset_m']):.3f} median={float(best['median_max_offset_m']):.3f} under1={float(best['under_1m']):.3f}", flush=True)
    print(f"Wrote {detail_path}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
