from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import math
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
sys.path.insert(0, str(ROOT / "work"))

import anchor_solver_fold_rescue_experiment as exp  # noqa: E402
import anchor_solver_ml_distance_completion as dc  # noqa: E402
import anchor_solver_p95_ml_cases as p95  # noqa: E402

BUCKET_KEYS = ("random", "grid", "office")
BUCKET_LABELS = {"random": "Random 16-32", "grid": "Grid >=16", "office": "Office >=16"}
FORMULA = "EDGE_RADIUS_M * multiplier * (shortest_path / EDGE_RADIUS_M) ** power"


@dataclass(frozen=True)
class PreparedCase:
    bucket_key: str
    bucket: str
    case_index: int
    family: str
    shape: str
    anchors: int
    known_pair_count: int
    truth: dict[str, tuple[float, float]]
    known_pairs: list[exp.AnchorPairDistance]


@dataclass(frozen=True)
class SweepConfig:
    max_hops: int
    relative_sigma: float
    hop_sigma_m: float
    distance_multiplier: float
    distance_power: float


def parse_ints(text: str) -> list[int]:
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def parse_floats(text: str) -> list[float]:
    return [float(part.strip()) for part in text.split(",") if part.strip()]


def config_key(config: SweepConfig) -> tuple[int, str, str, str, str]:
    return (
        config.max_hops,
        f"{config.relative_sigma:.8g}",
        f"{config.hop_sigma_m:.8g}",
        f"{config.distance_multiplier:.8g}",
        f"{config.distance_power:.8g}",
    )


def unique_configs(configs: list[SweepConfig]) -> list[SweepConfig]:
    seen: set[tuple[int, str, str, str, str]] = set()
    unique: list[SweepConfig] = []
    for config in configs:
        key = config_key(config)
        if key in seen:
            continue
        seen.add(key)
        unique.append(config)
    return unique


def transformed_path_distance(shortest_path_m: float, *, multiplier: float, power: float) -> float:
    edge_radius = float(dc.EDGE_RADIUS_M)
    return edge_radius * multiplier * (shortest_path_m / edge_radius) ** power


def transformed_graph_shortest_scaffold_pairs(
    processed: list[exp.ProcessedAnchorPair],
    anchor_ids: list[str],
    *,
    max_hops: int,
    relative_sigma: float,
    hop_sigma_m: float,
    distance_multiplier: float,
    distance_power: float,
) -> list[exp.AnchorPairDistance]:
    n = len(anchor_ids)
    index = {anchor_id: offset for offset, anchor_id in enumerate(anchor_ids)}
    inf = 1e12
    shortest = [[inf for _ in range(n)] for _ in range(n)]
    hops = [[10**9 for _ in range(n)] for _ in range(n)]
    measured_keys = set()
    for i in range(n):
        shortest[i][i] = 0.0
        hops[i][i] = 0
    for pair in processed:
        i = index[pair.anchor_a_id]
        j = index[pair.anchor_b_id]
        measured_keys.add(tuple(sorted((pair.anchor_a_id, pair.anchor_b_id))))
        if pair.distance_m < shortest[i][j]:
            shortest[i][j] = shortest[j][i] = pair.distance_m
            hops[i][j] = hops[j][i] = 1
    for k in range(n):
        for i in range(n):
            via_distance = shortest[i][k]
            if via_distance >= inf:
                continue
            via_hops = hops[i][k]
            for j in range(n):
                candidate_distance = via_distance + shortest[k][j]
                candidate_hops = via_hops + hops[k][j]
                if candidate_distance < shortest[i][j] - 1e-9 or (
                    abs(candidate_distance - shortest[i][j]) <= 1e-9 and candidate_hops < hops[i][j]
                ):
                    shortest[i][j] = candidate_distance
                    hops[i][j] = candidate_hops

    scaffold: list[exp.AnchorPairDistance] = []
    for i, anchor_a in enumerate(anchor_ids):
        for j in range(i + 1, n):
            anchor_b = anchor_ids[j]
            if tuple(sorted((anchor_a, anchor_b))) in measured_keys:
                continue
            hop_count = hops[i][j]
            if hop_count < 2 or hop_count > max_hops or shortest[i][j] >= inf:
                continue
            path_distance = shortest[i][j]
            transformed_distance = transformed_path_distance(
                path_distance,
                multiplier=distance_multiplier,
                power=distance_power,
            )
            sigma = max(0.35, transformed_distance * relative_sigma + hop_sigma_m * max(0, hop_count - 2))
            scaffold.append(
                exp.AnchorPairDistance(
                    anchor_a,
                    anchor_b,
                    transformed_distance,
                    sigma_m=sigma,
                    enabled=True,
                    source=f"graph-shortest-h{hop_count}-m{distance_multiplier:.2f}-p{distance_power:.2f}",
                )
            )
    return scaffold


def graph_shortest_transformed_scaffold_solve(
    known_pairs: list[exp.AnchorPairDistance],
    *,
    seed_count: int,
    iterations: int,
    rng_seed: int,
    max_hops: int,
    relative_sigma: float,
    hop_sigma_m: float,
    distance_multiplier: float,
    distance_power: float,
) -> dict[str, tuple[float, float]]:
    rng = random.Random(rng_seed)
    processed = exp._preprocess_pairs(known_pairs, min_sigma_m=0.02, min_distance_m=0.05)
    anchor_ids = exp._anchor_ids(processed)
    scaffold = transformed_graph_shortest_scaffold_pairs(
        processed,
        anchor_ids,
        max_hops=max_hops,
        relative_sigma=relative_sigma,
        hop_sigma_m=hop_sigma_m,
        distance_multiplier=distance_multiplier,
        distance_power=distance_power,
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
    best_positions: dict[str, tuple[float, float]] | None = None
    best_score = math.inf
    for seed in seeds:
        scaffold_params, _energy = exp._local_minimize(
            seed,
            parameterization,
            augmented,
            scaffold_priors,
            max_iterations=max(iterations // 2, 30),
        )
        scaffold_positions = parameterization.to_positions(scaffold_params)
        prior_positions = exp.internal_solve_from_positions(
            scaffold_positions,
            processed,
            parameterization,
            priors=known_priors,
            iterations=max(iterations, 30),
        )
        polished = exp.internal_solve_from_positions(
            prior_positions,
            processed,
            parameterization,
            priors=None,
            iterations=max(iterations // 2, 30),
        )
        for candidate_positions in (prior_positions, polished):
            score = exp.topology_selection_score(candidate_positions, known_pairs, fold_threshold_m=1.65)
            if score < best_score:
                best_score = score
                best_positions = candidate_positions
    assert best_positions is not None
    return best_positions


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def prepare_cases(*, cases_per_bucket: int, seed: int, device: torch.device) -> list[PreparedCase]:
    set_seeds(seed)
    prepared: list[PreparedCase] = []
    for bucket_key in BUCKET_KEYS:
        print(f"generating fixed fair validation set bucket={bucket_key} cases={cases_per_bucket}", flush=True)
        cases = p95.generate_cases(bucket_key, cases_per_bucket, device)
        node_counts = [int(case.points.shape[0]) for case in cases]
        print(
            f"generated bucket={bucket_key} min_nodes={min(node_counts)} "
            f"median_nodes={np.median(node_counts):.0f} max_nodes={max(node_counts)}",
            flush=True,
        )
        for case_index, case in enumerate(cases):
            batch = p95.graph_batch_from_cases([case], device)
            truth = dc.graph_to_truth(batch, 0)
            known_pairs = dc.known_pairs_from_batch(batch, 0)
            prepared.append(
                PreparedCase(
                    bucket_key=bucket_key,
                    bucket=BUCKET_LABELS[bucket_key],
                    case_index=case_index,
                    family=case.family,
                    shape=case.shape,
                    anchors=len(truth),
                    known_pair_count=len(known_pairs),
                    truth=truth,
                    known_pairs=known_pairs,
                )
            )
    return prepared


def write_csv(path: Path, rows: list[dict[str, str | int | float]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write for {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def case_rows(cases: list[PreparedCase]) -> list[dict[str, str | int | float]]:
    return [
        {
            "bucket_key": case.bucket_key,
            "bucket": case.bucket,
            "case_index": case.case_index,
            "family": case.family,
            "shape": case.shape,
            "anchors": case.anchors,
            "known_pairs": case.known_pair_count,
        }
        for case in cases
    ]


def row_from_result(
    result: exp.MethodResult,
    *,
    stage: str,
    config: SweepConfig,
    elapsed_s: float,
    seed_count: int,
    iterations: int,
) -> dict[str, str | int | float]:
    row = exp.result_to_dict(result)
    row.update(
        {
            "stage": stage,
            "max_hops": config.max_hops,
            "relative_sigma": config.relative_sigma,
            "hop_sigma_m": config.hop_sigma_m,
            "distance_multiplier": config.distance_multiplier,
            "distance_power": config.distance_power,
            "transformed_formula": FORMULA,
            "seed_count": seed_count,
            "iterations": iterations,
            "elapsed_s": elapsed_s,
        }
    )
    return row


def run_stage(
    *,
    stage: str,
    cases: list[PreparedCase],
    configs: list[SweepConfig],
    seed: int,
    seed_count: int,
    iterations: int,
    fold_threshold_m: float,
) -> list[dict[str, str | int | float]]:
    rows: list[dict[str, str | int | float]] = []
    total = len(cases) * len(configs)
    done = 0
    started = time.perf_counter()
    print(f"stage={stage} cases={len(cases)} configs={len(configs)} solves={total}", flush=True)
    for config_index, config in enumerate(configs):
        for case in cases:
            done += 1
            solve_seed = (
                seed
                + case.case_index * 1009
                + config_index * 7919
                + config.max_hops * 101
                + int(config.relative_sigma * 1000)
                + int(config.hop_sigma_m * 1000)
                + int(config.distance_multiplier * 1000) * 3
                + int(config.distance_power * 1000) * 7
            )
            case_started = time.perf_counter()
            positions = graph_shortest_transformed_scaffold_solve(
                case.known_pairs,
                seed_count=seed_count,
                iterations=iterations,
                rng_seed=solve_seed,
                max_hops=config.max_hops,
                relative_sigma=config.relative_sigma,
                hop_sigma_m=config.hop_sigma_m,
                distance_multiplier=config.distance_multiplier,
                distance_power=config.distance_power,
            )
            elapsed_s = time.perf_counter() - case_started
            result = exp.metrics_row(
                case.bucket,
                case.case_index,
                case.family,
                case.shape,
                "graph-shortest-transformed",
                case.truth,
                positions,
                case.known_pairs,
                fold_threshold_m=fold_threshold_m,
            )
            rows.append(
                row_from_result(
                    result,
                    stage=stage,
                    config=config,
                    elapsed_s=elapsed_s,
                    seed_count=seed_count,
                    iterations=iterations,
                )
            )
            if done == 1 or done % 50 == 0 or done == total:
                rate = done / max(time.perf_counter() - started, 1e-9)
                print(
                    f"stage={stage} progress={done}/{total} rate={rate:.2f}/s "
                    f"bucket={case.bucket_key} case={case.case_index} "
                    f"h={config.max_hops} rs={config.relative_sigma:.2f} hs={config.hop_sigma_m:.2f} "
                    f"mult={config.distance_multiplier:.2f} pow={config.distance_power:.2f}",
                    flush=True,
                )
    return rows


def summarize(rows: list[dict[str, str | int | float]]) -> list[dict[str, str | int | float]]:
    summary: list[dict[str, str | int | float]] = []
    keys: set[tuple[str, str, str, str, str, str, str]] = set()
    for row in rows:
        keys.add(
            (
                str(row["stage"]),
                str(row["bucket"]),
                str(row["max_hops"]),
                str(row["relative_sigma"]),
                str(row["hop_sigma_m"]),
                str(row["distance_multiplier"]),
                str(row["distance_power"]),
            )
        )
        keys.add(
            (
                str(row["stage"]),
                "ALL",
                str(row["max_hops"]),
                str(row["relative_sigma"]),
                str(row["hop_sigma_m"]),
                str(row["distance_multiplier"]),
                str(row["distance_power"]),
            )
        )
    for stage, bucket, max_hops, relative_sigma, hop_sigma_m, multiplier, power in sorted(keys):
        part = [
            row
            for row in rows
            if str(row["stage"]) == stage
            and (bucket == "ALL" or str(row["bucket"]) == bucket)
            and str(row["max_hops"]) == max_hops
            and str(row["relative_sigma"]) == relative_sigma
            and str(row["hop_sigma_m"]) == hop_sigma_m
            and str(row["distance_multiplier"]) == multiplier
            and str(row["distance_power"]) == power
        ]
        if not part:
            continue
        offsets = np.array([float(row["max_offset_m"]) for row in part], dtype=float)
        p95_offsets = np.array([float(row["p95_offset_m"]) for row in part], dtype=float)
        rmses = np.array([float(row["known_rmse_m"]) for row in part], dtype=float)
        min_dist = np.array([float(row["min_pair_distance_m"]) for row in part], dtype=float)
        close = np.array([float(row["close_pair_count"]) for row in part], dtype=float)
        elapsed = np.array([float(row["elapsed_s"]) for row in part], dtype=float)
        summary.append(
            {
                "stage": stage,
                "bucket": bucket,
                "method": "graph-shortest-transformed",
                "max_hops": int(max_hops),
                "relative_sigma": float(relative_sigma),
                "hop_sigma_m": float(hop_sigma_m),
                "distance_multiplier": float(multiplier),
                "distance_power": float(power),
                "transformed_formula": FORMULA,
                "cases": len(part),
                "median_max_offset_m": float(np.median(offsets)),
                "p90_max_offset_m": float(np.quantile(offsets, 0.90)),
                "p95_max_offset_m": float(np.quantile(offsets, 0.95)),
                "max_offset_m": float(np.max(offsets)),
                "median_anchor_p95_offset_m": float(np.median(p95_offsets)),
                "under_20cm": float(np.mean(offsets <= 0.20)),
                "under_50cm": float(np.mean(offsets <= 0.50)),
                "under_1m": float(np.mean(offsets <= 1.00)),
                "median_known_rmse_m": float(np.median(rmses)),
                "median_min_pair_distance_m": float(np.median(min_dist)),
                "folded_case_rate": float(np.mean(close > 0)),
                "mean_close_pair_count": float(np.mean(close)),
                "median_elapsed_s": float(np.median(elapsed)),
                "total_elapsed_s": float(np.sum(elapsed)),
            }
        )
    return summary


def rank_key(row: dict[str, str | int | float]) -> tuple[float, float, float, float, float]:
    return (
        float(row["p95_max_offset_m"]),
        -float(row["under_1m"]),
        float(row["median_max_offset_m"]),
        float(row["folded_case_rate"]),
        float(row["median_known_rmse_m"]),
    )


def config_from_summary(row: dict[str, str | int | float]) -> SweepConfig:
    return SweepConfig(
        max_hops=int(row["max_hops"]),
        relative_sigma=float(row["relative_sigma"]),
        hop_sigma_m=float(row["hop_sigma_m"]),
        distance_multiplier=float(row["distance_multiplier"]),
        distance_power=float(row["distance_power"]),
    )


def top_configs(
    summary: list[dict[str, str | int | float]],
    *,
    stage: str,
    bucket: str,
    limit: int,
) -> list[SweepConfig]:
    selected = [row for row in summary if row["stage"] == stage and row["bucket"] == bucket]
    return [config_from_summary(row) for row in sorted(selected, key=rank_key)[:limit]]


def top_transform_pairs(
    summary: list[dict[str, str | int | float]],
    *,
    stage: str,
    limit: int,
) -> list[tuple[float, float]]:
    rows = [row for row in summary if row["stage"] == stage and row["bucket"] == "ALL"]
    pairs: list[tuple[float, float]] = []
    seen: set[tuple[str, str]] = set()
    for row in sorted(rows, key=rank_key):
        key = (f"{float(row['distance_multiplier']):.8g}", f"{float(row['distance_power']):.8g}")
        if key in seen:
            continue
        seen.add(key)
        pairs.append((float(row["distance_multiplier"]), float(row["distance_power"])))
        if len(pairs) >= limit:
            break
    return pairs


def transform_pilot_configs(
    *,
    multipliers: list[float],
    powers: list[float],
    max_hops: list[int],
    relative_sigma: float,
    hop_sigma_m: float,
) -> list[SweepConfig]:
    return unique_configs(
        [
            SweepConfig(hop, relative_sigma, hop_sigma_m, multiplier, power)
            for hop in max_hops
            for multiplier in multipliers
            for power in powers
        ]
    )


def sigma_pilot_configs(
    *,
    transform_pairs: list[tuple[float, float]],
    max_hops: list[int],
    relative_sigmas: list[float],
    hop_sigmas: list[float],
    default_relative_sigma: float,
    default_hop_sigma_m: float,
) -> list[SweepConfig]:
    configs: list[SweepConfig] = []
    for multiplier, power in transform_pairs:
        for hop in max_hops:
            for relative_sigma in relative_sigmas:
                configs.append(SweepConfig(hop, relative_sigma, default_hop_sigma_m, multiplier, power))
            for hop_sigma_m in hop_sigmas:
                configs.append(SweepConfig(hop, default_relative_sigma, hop_sigma_m, multiplier, power))
    return unique_configs(configs)


def final_validation_configs(
    *,
    summary: list[dict[str, str | int | float]],
    final_config_count: int,
) -> list[SweepConfig]:
    configs: list[SweepConfig] = []
    for bucket in ("ALL", *BUCKET_LABELS.values()):
        configs.extend(top_configs(summary, stage="sigma_pilot", bucket=bucket, limit=2 if bucket != "ALL" else final_config_count))
    configs.extend(top_configs(summary, stage="transform_pilot", bucket="ALL", limit=3))
    configs.extend(
        [
            SweepConfig(2, 0.36, 0.65, 1.0, 1.0),
            SweepConfig(3, 0.30, 0.85, 1.0, 1.0),
            SweepConfig(4, 0.30, 0.85, 1.0, 1.0),
        ]
    )
    return unique_configs(configs)[:final_config_count]


def write_report(
    path: Path,
    *,
    args: argparse.Namespace,
    cases: list[PreparedCase],
    summary: list[dict[str, str | int | float]],
    detail_rows: list[dict[str, str | int | float]],
    transform_pairs: list[tuple[float, float]],
    final_configs: list[SweepConfig],
) -> None:
    final_rows = [row for row in summary if row["stage"] == "final_validation"]
    best_overall = sorted([row for row in final_rows if row["bucket"] == "ALL"], key=rank_key)[:1]
    best_by_bucket = []
    for bucket in BUCKET_LABELS.values():
        bucket_rows = [row for row in final_rows if row["bucket"] == bucket]
        if bucket_rows:
            best_by_bucket.append(sorted(bucket_rows, key=rank_key)[0])
    transform_stage_rows = [row for row in summary if row["stage"] == "transform_pilot" and row["bucket"] == "ALL"]
    best_transform_stage = sorted(transform_stage_rows, key=rank_key)[:5]
    full_cartesian_configs = (
        len(parse_ints(args.max_hops))
        * len(parse_floats(args.relative_sigma))
        * len(parse_floats(args.hop_sigma))
        * len(parse_floats(args.distance_multipliers))
        * len(parse_floats(args.distance_powers))
    )
    full_cartesian_solves = full_cartesian_configs * len(cases)
    lines = [
        "# Agent Graph Shortest Heavy Sweep",
        "",
        f"Transform formula: `{FORMULA}` with `EDGE_RADIUS_M={dc.EDGE_RADIUS_M:.1f}`.",
        "",
        "## Run Shape",
        "",
        f"- Fixed fair validation set: {args.cases_per_bucket} cases per bucket across random/grid/office ({len(cases)} cases total).",
        f"- Pilot subset: first {args.pilot_cases_per_bucket} cases per bucket for staged search.",
        f"- Solver: seed_count={args.seed_count}, iterations={args.iterations}; shared solver files were not modified.",
        f"- Required hop values swept: {args.max_hops}.",
        f"- Required relative_sigma values swept: {args.relative_sigma}.",
        f"- Required hop_sigma values swept: {args.hop_sigma}.",
        f"- Required transform multipliers swept: {args.distance_multipliers}.",
        f"- Required transform powers swept: {args.distance_powers}.",
        f"- Staging reduction: full Cartesian would be {full_cartesian_configs:,} configs x {len(cases)} cases = {full_cartesian_solves:,} solves; this run evaluated {len(detail_rows):,} staged solves.",
        "",
        "## Best Final Validation Results",
        "",
    ]
    if best_overall:
        row = best_overall[0]
        lines.extend(
            [
                (
                    f"Overall best: h={row['max_hops']}, rs={float(row['relative_sigma']):.2f}, "
                    f"hs={float(row['hop_sigma_m']):.2f}, multiplier={float(row['distance_multiplier']):.2f}, "
                    f"power={float(row['distance_power']):.2f}; p95={float(row['p95_max_offset_m']):.3f} m, "
                    f"under1m={float(row['under_1m']):.3f}, median={float(row['median_max_offset_m']):.3f} m."
                ),
                "",
            ]
        )
    lines.append("| Bucket | h | rs | hs | multiplier | power | p95 max offset m | under1m | median max offset m | folded rate |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in best_by_bucket:
        lines.append(
            f"| {row['bucket']} | {row['max_hops']} | {float(row['relative_sigma']):.2f} | "
            f"{float(row['hop_sigma_m']):.2f} | {float(row['distance_multiplier']):.2f} | "
            f"{float(row['distance_power']):.2f} | {float(row['p95_max_offset_m']):.3f} | "
            f"{float(row['under_1m']):.3f} | {float(row['median_max_offset_m']):.3f} | "
            f"{float(row['folded_case_rate']):.3f} |"
        )
    lines.extend(["", "## Transform Pilot", ""])
    lines.append(
        "Top multiplier/power pairs carried into the sigma pilot: "
        + ", ".join(f"m={multiplier:.2f}/p={power:.2f}" for multiplier, power in transform_pairs)
        + "."
    )
    lines.append("")
    lines.append("| Rank | h | multiplier | power | p95 max offset m | under1m | median max offset m |")
    lines.append("| ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for rank, row in enumerate(best_transform_stage, start=1):
        lines.append(
            f"| {rank} | {row['max_hops']} | {float(row['distance_multiplier']):.2f} | "
            f"{float(row['distance_power']):.2f} | {float(row['p95_max_offset_m']):.3f} | "
            f"{float(row['under_1m']):.3f} | {float(row['median_max_offset_m']):.3f} |"
        )
    lines.extend(["", "## Final Configs Evaluated", ""])
    for config in final_configs:
        lines.append(
            f"- h={config.max_hops}, rs={config.relative_sigma:.2f}, hs={config.hop_sigma_m:.2f}, "
            f"multiplier={config.distance_multiplier:.2f}, power={config.distance_power:.2f}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Staged heavy graph-shortest path transform sweep.")
    parser.add_argument("--cases-per-bucket", type=int, default=12)
    parser.add_argument("--pilot-cases-per-bucket", type=int, default=2)
    parser.add_argument("--seed", type=int, default=2026062709)
    parser.add_argument("--seed-count", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--fold-threshold", type=float, default=1.65)
    parser.add_argument("--max-hops", default="2,3,4,5,6,8")
    parser.add_argument("--relative-sigma", default="0.18,0.24,0.30,0.36,0.48,0.65")
    parser.add_argument("--hop-sigma", default="0.35,0.55,0.75,1.0,1.35,1.8")
    parser.add_argument("--distance-multipliers", default="0.70,0.85,1.0,1.15,1.35")
    parser.add_argument("--distance-powers", default="0.75,0.9,1.0,1.1,1.25")
    parser.add_argument("--transform-pilot-hops", default="2,4")
    parser.add_argument("--transform-default-relative-sigma", type=float, default=0.30)
    parser.add_argument("--transform-default-hop-sigma", type=float, default=0.85)
    parser.add_argument("--sigma-default-relative-sigma", type=float, default=0.30)
    parser.add_argument("--sigma-default-hop-sigma", type=float, default=0.75)
    parser.add_argument("--top-transforms", type=int, default=3)
    parser.add_argument("--final-configs", type=int, default=12)
    parser.add_argument("--prefix", default="agent_graph_shortest_heavy_sweep")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.pilot_cases_per_bucket > args.cases_per_bucket:
        raise ValueError("--pilot-cases-per-bucket cannot exceed --cases-per-bucket")

    OUTPUTS.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")
    max_hops = parse_ints(args.max_hops)
    relative_sigmas = parse_floats(args.relative_sigma)
    hop_sigmas = parse_floats(args.hop_sigma)
    multipliers = parse_floats(args.distance_multipliers)
    powers = parse_floats(args.distance_powers)

    cases = prepare_cases(cases_per_bucket=args.cases_per_bucket, seed=args.seed, device=device)
    pilot_cases = [case for case in cases if case.case_index < args.pilot_cases_per_bucket]

    detail: list[dict[str, str | int | float]] = []
    transform_configs = transform_pilot_configs(
        multipliers=multipliers,
        powers=powers,
        max_hops=parse_ints(args.transform_pilot_hops),
        relative_sigma=args.transform_default_relative_sigma,
        hop_sigma_m=args.transform_default_hop_sigma,
    )
    detail.extend(
        run_stage(
            stage="transform_pilot",
            cases=pilot_cases,
            configs=transform_configs,
            seed=args.seed + 10_000,
            seed_count=args.seed_count,
            iterations=args.iterations,
            fold_threshold_m=args.fold_threshold,
        )
    )
    summary = summarize(detail)
    transform_pairs = top_transform_pairs(summary, stage="transform_pilot", limit=args.top_transforms)
    sigma_configs = sigma_pilot_configs(
        transform_pairs=transform_pairs,
        max_hops=max_hops,
        relative_sigmas=relative_sigmas,
        hop_sigmas=hop_sigmas,
        default_relative_sigma=args.sigma_default_relative_sigma,
        default_hop_sigma_m=args.sigma_default_hop_sigma,
    )
    detail.extend(
        run_stage(
            stage="sigma_pilot",
            cases=pilot_cases,
            configs=sigma_configs,
            seed=args.seed + 20_000,
            seed_count=args.seed_count,
            iterations=args.iterations,
            fold_threshold_m=args.fold_threshold,
        )
    )
    summary = summarize(detail)
    final_configs = final_validation_configs(summary=summary, final_config_count=args.final_configs)
    detail.extend(
        run_stage(
            stage="final_validation",
            cases=cases,
            configs=final_configs,
            seed=args.seed + 30_000,
            seed_count=args.seed_count,
            iterations=args.iterations,
            fold_threshold_m=args.fold_threshold,
        )
    )
    summary = summarize(detail)

    case_path = OUTPUTS / f"{args.prefix}_cases.csv"
    detail_path = OUTPUTS / f"{args.prefix}_detail.csv"
    summary_path = OUTPUTS / f"{args.prefix}_summary.csv"
    report_path = OUTPUTS / f"{args.prefix}_report.md"
    write_csv(case_path, case_rows(cases))
    write_csv(detail_path, detail)
    write_csv(summary_path, summary)
    write_report(
        report_path,
        args=args,
        cases=cases,
        summary=summary,
        detail_rows=detail,
        transform_pairs=transform_pairs,
        final_configs=final_configs,
    )

    final_summary = [row for row in summary if row["stage"] == "final_validation"]
    for row in sorted([row for row in final_summary if row["bucket"] == "ALL"], key=rank_key)[:8]:
        print(
            f"final overall h={row['max_hops']} rs={float(row['relative_sigma']):.2f} "
            f"hs={float(row['hop_sigma_m']):.2f} mult={float(row['distance_multiplier']):.2f} "
            f"pow={float(row['distance_power']):.2f} p95={float(row['p95_max_offset_m']):.3f}m "
            f"under1m={float(row['under_1m']):.3f}",
            flush=True,
        )
    print(f"Wrote {case_path}", flush=True)
    print(f"Wrote {detail_path}", flush=True)
    print(f"Wrote {summary_path}", flush=True)
    print(f"Wrote {report_path}", flush=True)


if __name__ == "__main__":
    main()
