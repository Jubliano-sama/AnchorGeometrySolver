from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import math
from pathlib import Path
import random
import sys
from typing import Iterable

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
DEFAULT_ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)
DEFAULT_MULTIPLIERS = (0.70, 0.85, 1.0, 1.15, 1.35)
DEFAULT_POWERS = (0.75, 0.90, 1.0, 1.10, 1.25)


@dataclass(frozen=True)
class CheckpointSpec:
    label: str
    path: Path


@dataclass(frozen=True)
class GraphCandidate:
    distance_m: float
    raw_shortest_m: float
    hops: int
    sigma_m: float


@dataclass(frozen=True)
class MlCandidate:
    distance_m: float
    sigma_m: float


@dataclass
class CaseRecord:
    bucket_key: str
    bucket: str
    case_index: int
    family: str
    shape: str
    batch: dc.GraphBatch
    local_index: int
    truth: dict[str, tuple[float, float]]
    known_pairs: list[dc.AnchorPairDistance]
    true_dist: np.ndarray
    measured: np.ndarray
    node_count: int
    graph_raw: dict[tuple[int, int], tuple[float, int]]


@dataclass(frozen=True)
class VariantSpec:
    name: str
    ml_sigma_scale: float = 1.0
    graph_sigma_scale: float = 1.0


def parse_float_list(text: str) -> list[float]:
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def parse_checkpoint_spec(text: str) -> CheckpointSpec:
    if "=" in text:
        label, raw_path = text.split("=", 1)
        return CheckpointSpec(label.strip(), Path(raw_path.strip()))
    path = Path(text)
    return CheckpointSpec(path.stem, path)


def default_checkpoints() -> list[CheckpointSpec]:
    return [
        CheckpointSpec("baseline", OUTPUTS / "anchor_solver_ml_distance_completion_fair_diag_cuda.pt"),
        CheckpointSpec("BigF", OUTPUTS / "anchor_solver_ml_distance_completion_bigF_bigD_edm08_lr35e5_best_loss.pt"),
    ]


def write_csv(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write for {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def anchor_id(index: int) -> str:
    return f"A{index:02d}"


def pair_key_to_ids(key: tuple[int, int]) -> tuple[str, str]:
    return anchor_id(key[0]), anchor_id(key[1])


def load_model(checkpoint_path: Path, device: torch.device, node_features: int, edge_features: int) -> dc.DistanceCompletionNet:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    saved_args = checkpoint.get("args", {})
    model = dc.DistanceCompletionNet(
        node_features,
        edge_features,
        hidden=int(saved_args.get("hidden", 144)),
        layers=int(saved_args.get("layers", 5)),
        dropout=float(saved_args.get("dropout", 0.0)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_case_records(
    *,
    cases_per_bucket: int,
    batch_size: int,
    device: torch.device,
) -> tuple[list[CaseRecord], list[dc.GraphBatch]]:
    records: list[CaseRecord] = []
    batches: list[dc.GraphBatch] = []
    for bucket_key in BUCKET_KEYS:
        cases = p95.generate_cases(bucket_key, cases_per_bucket, torch.device("cpu"))
        node_counts = [int(case.points.shape[0]) for case in cases]
        print(
            f"generated bucket={bucket_key} cases={len(cases)} "
            f"min_nodes={min(node_counts)} median_nodes={np.median(node_counts):.0f} max_nodes={max(node_counts)}",
            flush=True,
        )
        global_case_index = 0
        for start in range(0, len(cases), batch_size):
            chunk_cases = cases[start : start + batch_size]
            batch = p95.graph_batch_from_cases(chunk_cases, device)
            batches.append(batch)
            for local_index, case in enumerate(chunk_cases):
                n = batch.node_counts[local_index]
                known_pairs = dc.known_pairs_from_batch(batch, local_index)
                records.append(
                    CaseRecord(
                        bucket_key=bucket_key,
                        bucket=BUCKET_LABELS[bucket_key],
                        case_index=global_case_index,
                        family=case.family,
                        shape=case.shape,
                        batch=batch,
                        local_index=local_index,
                        truth=dc.graph_to_truth(batch, local_index),
                        known_pairs=known_pairs,
                        true_dist=batch.true_dist_m[local_index, :n, :n].detach().cpu().numpy().astype(float),
                        measured=batch.measured_mask[local_index, :n, :n].detach().cpu().numpy().astype(bool),
                        node_count=n,
                        graph_raw=raw_graph_shortest_candidates(known_pairs, n, max_hops=3),
                    )
                )
                global_case_index += 1
    return records, batches


def raw_graph_shortest_candidates(
    known_pairs: list[dc.AnchorPairDistance],
    node_count: int,
    *,
    max_hops: int,
) -> dict[tuple[int, int], tuple[float, int]]:
    processed = dc._preprocess_pairs(known_pairs, min_sigma_m=0.02, min_distance_m=0.05)
    anchor_ids = [anchor_id(index) for index in range(node_count)]
    index = {name: offset for offset, name in enumerate(anchor_ids)}
    inf = 1e12
    shortest = [[inf for _ in range(node_count)] for _ in range(node_count)]
    hops = [[10**9 for _ in range(node_count)] for _ in range(node_count)]
    measured_keys: set[tuple[int, int]] = set()
    for i in range(node_count):
        shortest[i][i] = 0.0
        hops[i][i] = 0
    for pair in processed:
        i = index[pair.anchor_a_id]
        j = index[pair.anchor_b_id]
        measured_keys.add((min(i, j), max(i, j)))
        if pair.distance_m < shortest[i][j]:
            shortest[i][j] = shortest[j][i] = pair.distance_m
            hops[i][j] = hops[j][i] = 1
    for k in range(node_count):
        for i in range(node_count):
            if shortest[i][k] >= inf:
                continue
            for j in range(node_count):
                candidate_distance = shortest[i][k] + shortest[k][j]
                candidate_hops = hops[i][k] + hops[k][j]
                if candidate_distance < shortest[i][j] - 1e-9 or (
                    abs(candidate_distance - shortest[i][j]) <= 1e-9 and candidate_hops < hops[i][j]
                ):
                    shortest[i][j] = candidate_distance
                    hops[i][j] = candidate_hops
    candidates: dict[tuple[int, int], tuple[float, int]] = {}
    for i in range(node_count):
        for j in range(i + 1, node_count):
            if (i, j) in measured_keys:
                continue
            hop_count = hops[i][j]
            if 2 <= hop_count <= max_hops and shortest[i][j] < inf:
                candidates[(i, j)] = (float(shortest[i][j]), int(hop_count))
    return candidates


def transformed_graph_candidates(
    record: CaseRecord,
    *,
    max_hops: int,
    multiplier: float,
    power: float,
    relative_sigma: float,
    hop_sigma_m: float,
    sigma_scale: float,
) -> dict[tuple[int, int], GraphCandidate]:
    candidates: dict[tuple[int, int], GraphCandidate] = {}
    for key, (raw_shortest, hop_count) in record.graph_raw.items():
        if hop_count > max_hops:
            continue
        transformed = dc.EDGE_RADIUS_M * multiplier * ((raw_shortest / dc.EDGE_RADIUS_M) ** power)
        sigma = max(0.35, transformed * relative_sigma + hop_sigma_m * max(0, hop_count - 2))
        candidates[key] = GraphCandidate(
            distance_m=max(float(transformed), 0.05),
            raw_shortest_m=float(raw_shortest),
            hops=hop_count,
            sigma_m=float(sigma * sigma_scale),
        )
    return candidates


def all_ml_candidates(
    record: CaseRecord,
    pred_norm: torch.Tensor,
    *,
    predicted_sigma_m: float,
    predicted_sigma_slope: float,
    sigma_scale: float,
) -> dict[tuple[int, int], MlCandidate]:
    scale = float(record.batch.scale_m[record.local_index].detach().cpu())
    pred_dist = (pred_norm[record.local_index, : record.node_count, : record.node_count].detach().cpu().numpy() * scale).astype(float)
    candidates: dict[tuple[int, int], MlCandidate] = {}
    for i in range(record.node_count):
        for j in range(i + 1, record.node_count):
            if record.measured[i, j]:
                continue
            predicted = max(float(pred_dist[i, j]), 0.05)
            sigma = predicted_sigma_m * (1.0 + predicted_sigma_slope * max(predicted - dc.EDGE_RADIUS_M, 0.0) / dc.EDGE_RADIUS_M)
            candidates[(i, j)] = MlCandidate(predicted, float(sigma * sigma_scale))
    return candidates


def closest_ml_cap(candidates: dict[tuple[int, int], MlCandidate], node_count: int, cap_per_anchor: float) -> dict[tuple[int, int], MlCandidate]:
    limit = min(len(candidates), max(0, int(math.ceil(cap_per_anchor * node_count))))
    ordered = sorted(candidates.items(), key=lambda item: (item[1].distance_m, item[0][0], item[0][1]))
    return dict(ordered[:limit])


def known_pairs(record: CaseRecord) -> list[dc.AnchorPairDistance]:
    return list(record.known_pairs)


def pairs_from_synthetic(
    record: CaseRecord,
    synthetic: dict[tuple[int, int], tuple[float, float, str]],
) -> list[dc.AnchorPairDistance]:
    pairs = known_pairs(record)
    for key, (distance_m, sigma_m, source) in sorted(synthetic.items()):
        anchor_a, anchor_b = pair_key_to_ids(key)
        pairs.append(dc.AnchorPairDistance(anchor_a, anchor_b, max(float(distance_m), 0.05), sigma_m=max(float(sigma_m), 0.02), enabled=True, source=source))
    return pairs


def blend_sigma(alpha: float, ml_sigma: float, graph_sigma: float) -> float:
    return alpha * ml_sigma + (1.0 - alpha) * graph_sigma


def synthetic_for_variant(
    variant: VariantSpec,
    *,
    alpha: float,
    ml_cap: dict[tuple[int, int], MlCandidate],
    ml_all: dict[tuple[int, int], MlCandidate],
    graph: dict[tuple[int, int], GraphCandidate],
) -> dict[tuple[int, int], tuple[float, float, str]]:
    synthetic: dict[tuple[int, int], tuple[float, float, str]] = {}
    if variant.name == "ml_cap":
        for key, ml in ml_cap.items():
            synthetic[key] = (ml.distance_m, ml.sigma_m, "ml-cap")
        return synthetic
    if variant.name == "graph_only":
        for key, graph_candidate in graph.items():
            synthetic[key] = (graph_candidate.distance_m, graph_candidate.sigma_m, f"graph-h{graph_candidate.hops}")
        return synthetic
    if variant.name == "overlap_blend":
        for key, graph_candidate in graph.items():
            ml = ml_all.get(key)
            if ml is None:
                continue
            distance = alpha * ml.distance_m + (1.0 - alpha) * graph_candidate.distance_m
            sigma = blend_sigma(alpha, ml.sigma_m, graph_candidate.sigma_m)
            synthetic[key] = (distance, sigma, f"blend-a{alpha:.2f}-h{graph_candidate.hops}")
        return synthetic
    if variant.name.startswith("union_"):
        for key in set(ml_cap) | set(graph):
            ml = ml_cap.get(key)
            graph_candidate = graph.get(key)
            if ml is not None and graph_candidate is not None:
                distance = alpha * ml.distance_m + (1.0 - alpha) * graph_candidate.distance_m
                sigma = blend_sigma(alpha, ml.sigma_m, graph_candidate.sigma_m)
                source = f"union-blend-a{alpha:.2f}-h{graph_candidate.hops}"
            elif ml is not None:
                distance = ml.distance_m
                sigma = ml.sigma_m
                source = "union-ml"
            elif graph_candidate is not None:
                distance = graph_candidate.distance_m
                sigma = graph_candidate.sigma_m
                source = f"union-graph-h{graph_candidate.hops}"
            else:
                continue
            synthetic[key] = (distance, sigma, source)
        return synthetic
    raise ValueError(f"Unknown variant {variant.name}")


def synthetic_missing_mae(record: CaseRecord, synthetic: dict[tuple[int, int], tuple[float, float, str]]) -> float | str:
    errors = [abs(distance - float(record.true_dist[i, j])) for (i, j), (distance, _sigma, _source) in synthetic.items()]
    if not errors:
        return ""
    return float(np.mean(errors))


def solve_completed_pairs(
    completed_pairs: list[dc.AnchorPairDistance],
    known_pairs_for_case: list[dc.AnchorPairDistance],
    *,
    solver_iterations: int,
    polish_iterations: int,
) -> dict[str, tuple[float, float]]:
    if len(completed_pairs) <= len(known_pairs_for_case):
        return dc.known_only_solution(known_pairs_for_case, max_iterations=max(solver_iterations, 30))
    return dc.completion_solution(
        completed_pairs,
        known_pairs_for_case,
        max_iterations=max(solver_iterations, 30),
        polish_known_iterations=max(polish_iterations, 0),
    )


def evaluate_row(
    record: CaseRecord,
    positions: dict[str, tuple[float, float]],
    *,
    checkpoint: str,
    variant: str,
    alpha: float | str,
    multiplier: float | str,
    power: float | str,
    max_hops: int,
    synthetic: dict[tuple[int, int], tuple[float, float, str]],
    model_missing_mae: float | str,
    ml_pairs: int,
    graph_pairs: int,
    overlap_pairs: int,
    ml_sigma_scale: float,
    graph_sigma_scale: float,
) -> dict[str, float | int | str]:
    max_offset, median_offset, p95_offset = dc.offset_summary(record.truth, positions)
    known_rmse, known_max = dc.pair_metrics(positions, record.known_pairs)
    return {
        "bucket": record.bucket,
        "bucket_key": record.bucket_key,
        "case_index": record.case_index,
        "family": record.family,
        "shape": record.shape,
        "checkpoint": checkpoint,
        "variant": variant,
        "method": method_name(checkpoint, variant, alpha, multiplier, power, ml_sigma_scale, graph_sigma_scale),
        "alpha": alpha,
        "multiplier": multiplier,
        "power": power,
        "max_hops": max_hops,
        "ml_sigma_scale": ml_sigma_scale,
        "graph_sigma_scale": graph_sigma_scale,
        "anchors": record.node_count,
        "known_pairs": len(record.known_pairs),
        "synthetic_pairs": len(synthetic),
        "ml_pairs": ml_pairs,
        "graph_pairs": graph_pairs,
        "overlap_pairs": overlap_pairs,
        "model_missing_mae_m": model_missing_mae,
        "synthetic_missing_mae_m": synthetic_missing_mae(record, synthetic),
        "known_rmse_m": known_rmse,
        "known_max_residual_m": known_max,
        "max_offset_m": max_offset,
        "median_offset_m": median_offset,
        "p95_offset_m": p95_offset,
    }


def method_name(
    checkpoint: str,
    variant: str,
    alpha: float | str,
    multiplier: float | str,
    power: float | str,
    ml_sigma_scale: float,
    graph_sigma_scale: float,
) -> str:
    if variant == "graph_only":
        return f"graph_only m={multiplier} p={power}"
    if variant == "ml_cap":
        return f"{checkpoint} ml_cap"
    sigma_text = ""
    if variant.startswith("union_"):
        sigma_text = f" mlw={ml_sigma_scale:g} graphw={graph_sigma_scale:g}"
    return f"{checkpoint} {variant} a={alpha} m={multiplier} p={power}{sigma_text}"


def summarize(detail: list[dict[str, float | int | str]]) -> list[dict[str, float | int | str]]:
    keys = [
        "bucket",
        "bucket_key",
        "checkpoint",
        "variant",
        "method",
        "alpha",
        "multiplier",
        "power",
        "max_hops",
        "ml_sigma_scale",
        "graph_sigma_scale",
    ]
    grouped = sorted({tuple(row[key] for key in keys) for row in detail}, key=lambda item: tuple(str(value) for value in item))
    summary: list[dict[str, float | int | str]] = []
    for group in grouped:
        part = [row for row in detail if tuple(row[key] for key in keys) == group]
        offsets = np.array([float(row["max_offset_m"]) for row in part], dtype=float)
        rmses = np.array([float(row["known_rmse_m"]) for row in part], dtype=float)
        known_max = np.array([float(row["known_max_residual_m"]) for row in part], dtype=float)
        model_missing = np.array([float(row["model_missing_mae_m"]) for row in part if str(row["model_missing_mae_m"]) != ""], dtype=float)
        synthetic_missing = np.array([float(row["synthetic_missing_mae_m"]) for row in part if str(row["synthetic_missing_mae_m"]) != ""], dtype=float)
        synthetic_pairs = np.array([float(row["synthetic_pairs"]) for row in part], dtype=float)
        ml_pairs = np.array([float(row["ml_pairs"]) for row in part], dtype=float)
        graph_pairs = np.array([float(row["graph_pairs"]) for row in part], dtype=float)
        overlap_pairs = np.array([float(row["overlap_pairs"]) for row in part], dtype=float)
        row = {key: value for key, value in zip(keys, group)}
        row.update(
            {
                "cases": len(part),
                "median_model_missing_mae_m": float(np.median(model_missing)) if model_missing.size else "",
                "median_synthetic_missing_mae_m": float(np.median(synthetic_missing)) if synthetic_missing.size else "",
                "median_max_offset_m": float(np.median(offsets)),
                "p90_max_offset_m": float(np.quantile(offsets, 0.90)),
                "p95_max_offset_m": float(np.quantile(offsets, 0.95)),
                "max_offset_m": float(np.max(offsets)),
                "under_1m": float(np.mean(offsets <= 1.0)),
                "median_known_rmse_m": float(np.median(rmses)),
                "median_known_max_residual_m": float(np.median(known_max)),
                "median_synthetic_pairs": float(np.median(synthetic_pairs)),
                "median_ml_pairs": float(np.median(ml_pairs)),
                "median_graph_pairs": float(np.median(graph_pairs)),
                "median_overlap_pairs": float(np.median(overlap_pairs)),
            }
        )
        summary.append(row)
    return summary


def transform_key(row: dict[str, float | int | str]) -> tuple[float, float]:
    return float(row["multiplier"]), float(row["power"])


def choose_blend_transforms(
    graph_summary: list[dict[str, float | int | str]],
    *,
    limit: int,
) -> list[tuple[float, float]]:
    by_transform: dict[tuple[float, float], list[float]] = {}
    for row in graph_summary:
        if row["variant"] != "graph_only":
            continue
        by_transform.setdefault(transform_key(row), []).append(float(row["p95_max_offset_m"]))
    ranked = sorted(by_transform, key=lambda key: (float(np.mean(by_transform[key])), key[0], key[1]))
    selected: list[tuple[float, float]] = [(1.0, 1.0)]
    for key in ranked:
        if key not in selected:
            selected.append(key)
        if len(selected) >= limit:
            break
    return selected


def best_rows(
    summary: list[dict[str, float | int | str]],
    *,
    bucket: str | None = None,
    checkpoint: str | None = None,
    variants: Iterable[str] | None = None,
) -> list[dict[str, float | int | str]]:
    variant_set = set(variants) if variants is not None else None
    rows = [
        row
        for row in summary
        if (bucket is None or row["bucket"] == bucket)
        and (checkpoint is None or row["checkpoint"] == checkpoint)
        and (variant_set is None or row["variant"] in variant_set)
    ]
    return sorted(rows, key=lambda row: (float(row["p95_max_offset_m"]), -float(row["under_1m"]), float(row["median_known_rmse_m"])))


def row_by_identity(
    summary: list[dict[str, float | int | str]],
    *,
    bucket: str,
    checkpoint: str,
    variants: Iterable[str],
) -> dict[str, float | int | str] | None:
    rows = [
        row
        for row in summary
        if row["bucket"] == bucket
        and row["checkpoint"] == checkpoint
        and row["variant"] in set(variants)
        and str(row["multiplier"]) not in ("", "nan")
        and str(row["power"]) not in ("", "nan")
        and abs(float(row["multiplier"]) - 1.0) < 1e-9
        and abs(float(row["power"]) - 1.0) < 1e-9
    ]
    if not rows:
        return None
    return best_rows(rows)[0]


def write_report(
    path: Path,
    summary: list[dict[str, float | int | str]],
    *,
    cases_per_bucket: int,
    alphas: list[float],
    multipliers: list[float],
    powers: list[float],
    selected_transforms: list[tuple[float, float]],
) -> None:
    lines: list[str] = []
    lines.append("# ML + graph blend sweep report")
    lines.append("")
    lines.append(
        f"Fixed noisy evaluation: {cases_per_bucket} cases per bucket, alpha grid {alphas}, "
        f"graph transform multipliers {multipliers}, powers {powers}."
    )
    lines.append(
        "Graph-only was swept over the full transform grid. Blend and union variants were run on "
        f"the staged transform set {selected_transforms}."
    )
    lines.append("")
    lines.append("## Best settings by bucket")
    for bucket in [BUCKET_LABELS[key] for key in BUCKET_KEYS]:
        best = best_rows(summary, bucket=bucket)[0]
        lines.append(
            f"- {bucket}: {best['method']} | p95 max offset {float(best['p95_max_offset_m']):.3f} m, "
            f"under1m {float(best['under_1m']):.3f}, median RMSE {float(best['median_known_rmse_m']):.4f} m."
        )
    lines.append("")
    lines.append("## Transform effect")
    for bucket in [BUCKET_LABELS[key] for key in BUCKET_KEYS]:
        graph_identity = row_by_identity(summary, bucket=bucket, checkpoint="graph", variants=("graph_only",))
        graph_best = best_rows(summary, bucket=bucket, checkpoint="graph", variants=("graph_only",))[0]
        if graph_identity is not None:
            delta = float(graph_identity["p95_max_offset_m"]) - float(graph_best["p95_max_offset_m"])
            verdict = "improved" if delta > 1e-9 else "did not improve"
            lines.append(
                f"- {bucket} graph-only: best m={graph_best['multiplier']}, p={graph_best['power']} "
                f"{verdict} vs identity by {delta:.3f} m p95."
            )
    blend_variants = ("overlap_blend", "union_balanced", "union_graph_loose", "union_ml_loose")
    for bucket in [BUCKET_LABELS[key] for key in BUCKET_KEYS]:
        for checkpoint in sorted({str(row["checkpoint"]) for row in summary if row["checkpoint"] not in ("graph",)}):
            identity = row_by_identity(summary, bucket=bucket, checkpoint=checkpoint, variants=blend_variants)
            transformed = [
                row
                for row in best_rows(summary, bucket=bucket, checkpoint=checkpoint, variants=blend_variants)
                if not (
                    str(row["multiplier"]) not in ("", "nan")
                    and str(row["power"]) not in ("", "nan")
                    and abs(float(row["multiplier"]) - 1.0) < 1e-9
                    and abs(float(row["power"]) - 1.0) < 1e-9
                )
            ]
            if identity is None or not transformed:
                continue
            best_transformed = transformed[0]
            delta = float(identity["p95_max_offset_m"]) - float(best_transformed["p95_max_offset_m"])
            verdict = "improved" if delta > 1e-9 else "did not improve"
            lines.append(
                f"- {bucket} {checkpoint} blend/union: transformed graph distances {verdict} vs identity "
                f"by {delta:.3f} m p95; best transformed is {best_transformed['method']}."
            )
    lines.append("")
    lines.append("## Notes")
    lines.append("- `alpha=1` means the overlapping synthetic distance is ML-only; `alpha=0` means transformed graph-only.")
    lines.append("- The summary CSV contains p95 max offset, under1m, median known-pair RMSE, model missing MAE, and synthetic-pair missing MAE.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def predict_all(
    models: list[tuple[str, dc.DistanceCompletionNet]],
    batches: list[dc.GraphBatch],
) -> dict[str, dict[int, torch.Tensor]]:
    predictions: dict[str, dict[int, torch.Tensor]] = {}
    with torch.no_grad():
        for label, model in models:
            predictions[label] = {}
            for batch_index, batch in enumerate(batches):
                predictions[label][batch_index] = model(
                    batch.node_features,
                    batch.edge_features,
                    batch.mask,
                    batch.measured_mask,
                    batch.pair_mask,
                )
            print(f"predicted checkpoint={label} batches={len(batches)}", flush=True)
    return predictions


def batch_index_by_identity(batches: list[dc.GraphBatch]) -> dict[int, int]:
    return {id(batch): index for index, batch in enumerate(batches)}


def run_graph_only(
    records: list[CaseRecord],
    *,
    multipliers: list[float],
    powers: list[float],
    max_hops: int,
    graph_relative_sigma: float,
    graph_hop_sigma_m: float,
    solver_iterations: int,
    polish_iterations: int,
) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    variant = VariantSpec("graph_only")
    total = len(multipliers) * len(powers)
    done = 0
    for multiplier in multipliers:
        for power in powers:
            done += 1
            print(f"graph_only transform {done}/{total} multiplier={multiplier} power={power}", flush=True)
            for record in records:
                graph = transformed_graph_candidates(
                    record,
                    max_hops=max_hops,
                    multiplier=multiplier,
                    power=power,
                    relative_sigma=graph_relative_sigma,
                    hop_sigma_m=graph_hop_sigma_m,
                    sigma_scale=variant.graph_sigma_scale,
                )
                synthetic = synthetic_for_variant(variant, alpha=0.0, ml_cap={}, ml_all={}, graph=graph)
                completed = pairs_from_synthetic(record, synthetic)
                positions = solve_completed_pairs(
                    completed,
                    record.known_pairs,
                    solver_iterations=solver_iterations,
                    polish_iterations=polish_iterations,
                )
                rows.append(
                    evaluate_row(
                        record,
                        positions,
                        checkpoint="graph",
                        variant=variant.name,
                        alpha=0.0,
                        multiplier=multiplier,
                        power=power,
                        max_hops=max_hops,
                        synthetic=synthetic,
                        model_missing_mae="",
                        ml_pairs=0,
                        graph_pairs=len(graph),
                        overlap_pairs=0,
                        ml_sigma_scale=variant.ml_sigma_scale,
                        graph_sigma_scale=variant.graph_sigma_scale,
                    )
                )
    return rows


def run_model_variants(
    records: list[CaseRecord],
    batches: list[dc.GraphBatch],
    predictions: dict[str, dict[int, torch.Tensor]],
    *,
    selected_transforms: list[tuple[float, float]],
    alphas: list[float],
    max_hops: int,
    graph_relative_sigma: float,
    graph_hop_sigma_m: float,
    predicted_sigma_m: float,
    predicted_sigma_slope: float,
    closest_predicted_pairs_per_anchor: float,
    solver_iterations: int,
    polish_iterations: int,
) -> list[dict[str, float | int | str]]:
    batch_lookup = batch_index_by_identity(batches)
    rows: list[dict[str, float | int | str]] = []
    union_variants = [
        VariantSpec("union_balanced", ml_sigma_scale=1.0, graph_sigma_scale=1.0),
        VariantSpec("union_graph_loose", ml_sigma_scale=1.0, graph_sigma_scale=1.75),
        VariantSpec("union_ml_loose", ml_sigma_scale=1.75, graph_sigma_scale=1.0),
    ]
    for checkpoint, pred_by_batch in predictions.items():
        print(f"evaluating checkpoint={checkpoint} ml_cap", flush=True)
        for record in records:
            pred = pred_by_batch[batch_lookup[id(record.batch)]]
            ml_all = all_ml_candidates(
                record,
                pred,
                predicted_sigma_m=predicted_sigma_m,
                predicted_sigma_slope=predicted_sigma_slope,
                sigma_scale=1.0,
            )
            ml_cap = closest_ml_cap(ml_all, record.node_count, closest_predicted_pairs_per_anchor)
            synthetic = synthetic_for_variant(VariantSpec("ml_cap"), alpha=1.0, ml_cap=ml_cap, ml_all=ml_all, graph={})
            completed = pairs_from_synthetic(record, synthetic)
            positions = solve_completed_pairs(
                completed,
                record.known_pairs,
                solver_iterations=solver_iterations,
                polish_iterations=polish_iterations,
            )
            rows.append(
                evaluate_row(
                    record,
                    positions,
                    checkpoint=checkpoint,
                    variant="ml_cap",
                    alpha=1.0,
                    multiplier="",
                    power="",
                    max_hops=max_hops,
                    synthetic=synthetic,
                    model_missing_mae=dc.missing_mae(record.batch, pred, record.local_index),
                    ml_pairs=len(ml_cap),
                    graph_pairs=0,
                    overlap_pairs=0,
                    ml_sigma_scale=1.0,
                    graph_sigma_scale=1.0,
                )
            )
        for multiplier, power in selected_transforms:
            for alpha in alphas:
                print(f"evaluating checkpoint={checkpoint} transform=({multiplier},{power}) alpha={alpha}", flush=True)
                for variant in [VariantSpec("overlap_blend"), *union_variants]:
                    for record in records:
                        pred = pred_by_batch[batch_lookup[id(record.batch)]]
                        ml_all = all_ml_candidates(
                            record,
                            pred,
                            predicted_sigma_m=predicted_sigma_m,
                            predicted_sigma_slope=predicted_sigma_slope,
                            sigma_scale=variant.ml_sigma_scale,
                        )
                        ml_cap = closest_ml_cap(ml_all, record.node_count, closest_predicted_pairs_per_anchor)
                        graph = transformed_graph_candidates(
                            record,
                            max_hops=max_hops,
                            multiplier=multiplier,
                            power=power,
                            relative_sigma=graph_relative_sigma,
                            hop_sigma_m=graph_hop_sigma_m,
                            sigma_scale=variant.graph_sigma_scale,
                        )
                        synthetic = synthetic_for_variant(variant, alpha=alpha, ml_cap=ml_cap, ml_all=ml_all, graph=graph)
                        completed = pairs_from_synthetic(record, synthetic)
                        positions = solve_completed_pairs(
                            completed,
                            record.known_pairs,
                            solver_iterations=solver_iterations,
                            polish_iterations=polish_iterations,
                        )
                        rows.append(
                            evaluate_row(
                                record,
                                positions,
                                checkpoint=checkpoint,
                                variant=variant.name,
                                alpha=alpha,
                                multiplier=multiplier,
                                power=power,
                                max_hops=max_hops,
                                synthetic=synthetic,
                                model_missing_mae=dc.missing_mae(record.batch, pred, record.local_index),
                                ml_pairs=len(ml_cap),
                                graph_pairs=len(graph),
                                overlap_pairs=len(set(ml_cap) & set(graph)),
                                ml_sigma_scale=variant.ml_sigma_scale,
                                graph_sigma_scale=variant.graph_sigma_scale,
                            )
                        )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep ML + transformed graph-shortest distance blends on fixed noisy anchor cases.")
    parser.add_argument("--checkpoint", action="append", help="Checkpoint spec label=path. Defaults to baseline and BigF.")
    parser.add_argument("--cases-per-bucket", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2026062707)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--prefix", default="agent_ml_graph_blend_sweep")
    parser.add_argument("--alphas", default="0,0.25,0.5,0.75,1")
    parser.add_argument("--multipliers", default="0.70,0.85,1.0,1.15,1.35")
    parser.add_argument("--powers", default="0.75,0.9,1.0,1.1,1.25")
    parser.add_argument("--blend-transform-limit", type=int, default=6, help="Stage blend/union on identity plus best graph transforms.")
    parser.add_argument("--max-hops", type=int, default=2)
    parser.add_argument("--solver-iterations", type=int, default=45)
    parser.add_argument("--polish-iterations", type=int, default=45)
    parser.add_argument("--predicted-sigma", type=float, default=0.55)
    parser.add_argument("--predicted-sigma-slope", type=float, default=0.65)
    parser.add_argument("--closest-predicted-pairs-per-anchor", type=float, default=5.0)
    parser.add_argument("--graph-relative-sigma", type=float, default=0.36)
    parser.add_argument("--graph-hop-sigma", type=float, default=0.65)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    alphas = parse_float_list(args.alphas)
    multipliers = parse_float_list(args.multipliers)
    powers = parse_float_list(args.powers)
    checkpoints = [parse_checkpoint_spec(item) for item in args.checkpoint] if args.checkpoint else default_checkpoints()
    for checkpoint in checkpoints:
        if not checkpoint.path.exists():
            raise FileNotFoundError(checkpoint.path)

    set_seeds(args.seed)
    device = p95.choose_device(args.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
    print(f"device={device} seed={args.seed}", flush=True)
    records, batches = build_case_records(cases_per_bucket=args.cases_per_bucket, batch_size=args.batch_size, device=device)
    first_batch = batches[0]
    node_features = int(first_batch.node_features.shape[-1])
    edge_features = int(first_batch.edge_features.shape[-1])
    models = [(spec.label, load_model(spec.path, device, node_features, edge_features)) for spec in checkpoints]
    predictions = predict_all(models, batches)

    detail = run_graph_only(
        records,
        multipliers=multipliers,
        powers=powers,
        max_hops=args.max_hops,
        graph_relative_sigma=args.graph_relative_sigma,
        graph_hop_sigma_m=args.graph_hop_sigma,
        solver_iterations=args.solver_iterations,
        polish_iterations=args.polish_iterations,
    )
    graph_summary = summarize(detail)
    selected_transforms = choose_blend_transforms(graph_summary, limit=max(1, args.blend_transform_limit))
    print(f"selected_blend_transforms={selected_transforms}", flush=True)
    detail.extend(
        run_model_variants(
            records,
            batches,
            predictions,
            selected_transforms=selected_transforms,
            alphas=alphas,
            max_hops=args.max_hops,
            graph_relative_sigma=args.graph_relative_sigma,
            graph_hop_sigma_m=args.graph_hop_sigma,
            predicted_sigma_m=args.predicted_sigma,
            predicted_sigma_slope=args.predicted_sigma_slope,
            closest_predicted_pairs_per_anchor=args.closest_predicted_pairs_per_anchor,
            solver_iterations=args.solver_iterations,
            polish_iterations=args.polish_iterations,
        )
    )

    summary = summarize(detail)
    detail_path = OUTPUTS / f"{args.prefix}_detail.csv"
    summary_path = OUTPUTS / f"{args.prefix}_summary.csv"
    report_path = OUTPUTS / f"{args.prefix}_report.md"
    write_csv(detail_path, detail)
    write_csv(summary_path, summary)
    write_report(
        report_path,
        summary,
        cases_per_bucket=args.cases_per_bucket,
        alphas=alphas,
        multipliers=multipliers,
        powers=powers,
        selected_transforms=selected_transforms,
    )
    print(f"Wrote {detail_path}", flush=True)
    print(f"Wrote {summary_path}", flush=True)
    print(f"Wrote {report_path}", flush=True)


if __name__ == "__main__":
    main()
