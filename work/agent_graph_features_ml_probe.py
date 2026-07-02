from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import math
from pathlib import Path
import random
import sys
import time
from typing import Iterable

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "work"
OUTPUTS = ROOT / "outputs"
sys.path.insert(0, str(WORK))

import anchor_solver_ml_distance_completion as dc  # noqa: E402
import anchor_solver_p95_ml_cases as p95  # noqa: E402


@dataclass(frozen=True)
class ScaffoldSpec:
    max_hops: int
    relative_sigma: float
    hop_sigma_m: float

    @property
    def label(self) -> str:
        return f"h{self.max_hops}_rs{self.relative_sigma:g}_hs{self.hop_sigma_m:g}"


class ResidualDistanceCalibrator(nn.Module):
    def __init__(self, feature_count: int, *, hidden: int, layers: int, max_delta_norm: float) -> None:
        super().__init__()
        blocks: list[nn.Module] = []
        width = feature_count
        for _ in range(layers):
            blocks.extend([nn.Linear(width, hidden), nn.SiLU(), nn.LayerNorm(hidden)])
            width = hidden
        final = nn.Linear(width, 1)
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)
        blocks.append(final)
        self.net = nn.Sequential(*blocks)
        self.max_delta_norm = max_delta_norm

    def forward(self, features: torch.Tensor, base_pred_norm: torch.Tensor) -> torch.Tensor:
        delta = torch.tanh(self.net(features).squeeze(-1)) * self.max_delta_norm
        return (base_pred_norm + delta).clamp_min(0.02)


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested, but torch.cuda.is_available() is false.")
    return device


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_scaffold_specs(text: str) -> list[ScaffoldSpec]:
    specs: list[ScaffoldSpec] = []
    for raw_part in text.split(","):
        part = raw_part.strip()
        if not part:
            continue
        pieces = part.split(":")
        if len(pieces) != 3:
            raise ValueError(f"Bad scaffold spec {part!r}; expected max_hops:relative_sigma:hop_sigma_m")
        specs.append(ScaffoldSpec(int(pieces[0]), float(pieces[1]), float(pieces[2])))
    if not specs:
        raise ValueError("At least one scaffold spec is required.")
    return specs


def upper_missing_mask(batch: dc.GraphBatch) -> torch.Tensor:
    nodes = batch.pair_mask.shape[1]
    upper = torch.triu(torch.ones((nodes, nodes), dtype=torch.bool, device=batch.pair_mask.device), diagonal=1)
    return batch.pair_mask & ~batch.measured_mask & upper.unsqueeze(0)


def scaffold_feature_channels(
    batch: dc.GraphBatch,
    specs: list[ScaffoldSpec],
) -> tuple[torch.Tensor, list[str]]:
    scale = batch.scale_m.view(-1, 1, 1)
    path_norm = torch.clamp(batch.shortest_m / scale, 0.0, 8.0)
    hop_raw = batch.hop_count
    hop_norm = torch.clamp(hop_raw / 8.0, 0.0, 8.0)
    finite_path = hop_raw < dc.INF_DISTANCE * 0.5
    missing = batch.pair_mask & ~batch.measured_mask

    channels: list[torch.Tensor] = []
    names: list[str] = []
    for spec in specs:
        source = missing & finite_path & (hop_raw >= 2.0) & (hop_raw <= float(spec.max_hops))
        sigma_m = torch.maximum(
            torch.full_like(batch.shortest_m, 0.35),
            batch.shortest_m * spec.relative_sigma
            + torch.clamp(hop_raw - 2.0, min=0.0) * spec.hop_sigma_m,
        )
        sigma_norm = torch.clamp(sigma_m / scale, 0.0, 8.0)
        max_hops_channel = torch.full_like(path_norm, float(spec.max_hops) / 8.0)
        relative_sigma_channel = torch.full_like(path_norm, spec.relative_sigma)
        hop_sigma_channel = torch.full_like(path_norm, spec.hop_sigma_m) / scale
        mask_float = source.float()
        channels.extend(
            [
                mask_float,
                path_norm * mask_float,
                hop_norm * mask_float,
                sigma_norm * mask_float,
                max_hops_channel * mask_float,
                relative_sigma_channel * mask_float,
                hop_sigma_channel * mask_float,
            ]
        )
        prefix = f"scaffold_{spec.label}"
        names.extend(
            [
                f"{prefix}_source_mask",
                f"{prefix}_path_norm",
                f"{prefix}_hop_norm",
                f"{prefix}_sigma_norm",
                f"{prefix}_max_hops_norm",
                f"{prefix}_relative_sigma",
                f"{prefix}_hop_sigma_norm",
            ]
        )
    return torch.stack(channels, dim=-1), names


def feature_tensor(
    batch: dc.GraphBatch,
    base_pred_norm: torch.Tensor,
    *,
    mode: str,
    specs: list[ScaffoldSpec],
) -> tuple[torch.Tensor, list[str]]:
    scale = batch.scale_m.view(-1, 1, 1)
    measured_norm = batch.measured_dist_m / scale
    edge = batch.edge_features
    base_channels = [
        base_pred_norm,
        torch.log1p(base_pred_norm.clamp_min(0.0)),
        batch.measured_mask.float(),
        (batch.pair_mask & ~batch.measured_mask).float(),
        measured_norm,
        edge[..., 8],   # common neighbors
        edge[..., 9],   # edge radius lower bound
        edge[..., 10],  # minimum spacing lower bound
        edge[..., 11],  # endpoint degree i
        edge[..., 12],  # endpoint degree j
        edge[..., 14],  # degree imbalance
        edge[..., 15],  # local jaccard
        edge[..., 17],  # rigidity surplus
        edge[..., 18],  # density
    ]
    names = [
        "ml_pred_norm",
        "log1p_ml_pred_norm",
        "measured_mask",
        "missing_mask",
        "measured_norm",
        "common_neighbors",
        "edge_radius_norm",
        "min_spacing_norm",
        "degree_i_norm",
        "degree_j_norm",
        "endpoint_degree_delta",
        "jaccard",
        "rigidity_surplus",
        "density",
    ]
    if mode in {"existing_shortest", "scaffold"}:
        base_channels.extend([edge[..., 5], edge[..., 6], edge[..., 7]])
        names.extend(["existing_shortest_norm", "existing_exp_neg_shortest", "existing_hop_norm"])
    if mode == "scaffold":
        scaffold_channels, scaffold_names = scaffold_feature_channels(batch, specs)
        return torch.cat([torch.stack(base_channels, dim=-1), scaffold_channels], dim=-1), names + scaffold_names
    return torch.stack(base_channels, dim=-1), names


def pair_training_tensors(
    batch: dc.GraphBatch,
    base_pred_norm: torch.Tensor,
    *,
    mode: str,
    specs: list[ScaffoldSpec],
    max_pairs: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    features, _names = feature_tensor(batch, base_pred_norm, mode=mode, specs=specs)
    mask = upper_missing_mask(batch)
    selected = torch.nonzero(mask.reshape(-1), as_tuple=False).flatten()
    if max_pairs > 0 and selected.numel() > max_pairs:
        selected = selected[torch.randperm(selected.numel(), device=selected.device)[:max_pairs]]
    flat_features = features.reshape(-1, features.shape[-1])[selected]
    flat_base = base_pred_norm.reshape(-1)[selected]
    target = (batch.true_dist_m / batch.scale_m.view(-1, 1, 1)).reshape(-1)[selected]
    return flat_features, flat_base, target


def weighted_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    weights = 0.30 + torch.exp(-target / 3.0)
    loss = F.smooth_l1_loss(pred, target, beta=0.04, reduction="none")
    return (loss * weights).mean()


@torch.no_grad()
def corrected_prediction(
    batch: dc.GraphBatch,
    base_pred_norm: torch.Tensor,
    model: ResidualDistanceCalibrator,
    *,
    mode: str,
    specs: list[ScaffoldSpec],
    chunk_size: int = 65536,
) -> torch.Tensor:
    model.eval()
    corrected = base_pred_norm.clone()
    mask = upper_missing_mask(batch)
    selected = torch.nonzero(mask.reshape(-1), as_tuple=False).flatten()
    if selected.numel() == 0:
        return corrected
    features, _names = feature_tensor(batch, base_pred_norm, mode=mode, specs=specs)
    flat_features = features.reshape(-1, features.shape[-1])
    flat_base = base_pred_norm.reshape(-1)
    flat_corrected = corrected.reshape(-1)
    for start in range(0, selected.numel(), chunk_size):
        idx = selected[start : start + chunk_size]
        flat_corrected[idx] = model(flat_features[idx], flat_base[idx])
    nodes = corrected.shape[1]
    upper = torch.triu(torch.ones((nodes, nodes), dtype=torch.bool, device=corrected.device), diagonal=1).unsqueeze(0)
    corrected = torch.where(upper, corrected, corrected.transpose(1, 2))
    eye = torch.eye(nodes, dtype=torch.bool, device=corrected.device).unsqueeze(0)
    return corrected.masked_fill(~batch.pair_mask | eye, 0.0)


def distance_metrics(batch: dc.GraphBatch, pred_norm: torch.Tensor, case_index: int) -> dict[str, float]:
    n = batch.node_counts[case_index]
    mask = (~batch.measured_mask[case_index, :n, :n]) & batch.pair_mask[case_index, :n, :n]
    upper = torch.triu(torch.ones((n, n), dtype=torch.bool, device=pred_norm.device), diagonal=1)
    mask = mask & upper
    if not bool(mask.any()):
        return {
            "missing_mae_m": 0.0,
            "missing_median_abs_m": 0.0,
            "missing_p90_abs_m": 0.0,
            "missing_bias_m": 0.0,
        }
    scale = batch.scale_m[case_index]
    error = pred_norm[case_index, :n, :n][mask] * scale - batch.true_dist_m[case_index, :n, :n][mask]
    abs_error = error.abs()
    return {
        "missing_mae_m": float(abs_error.mean().detach().cpu()),
        "missing_median_abs_m": float(torch.quantile(abs_error, 0.50).detach().cpu()),
        "missing_p90_abs_m": float(torch.quantile(abs_error, 0.90).detach().cpu()),
        "missing_bias_m": float(error.mean().detach().cpu()),
    }


def solve_metrics(
    batch: dc.GraphBatch,
    pred_norm: torch.Tensor,
    case_index: int,
    args: argparse.Namespace,
) -> dict[str, float | bool]:
    try:
        truth = dc.graph_to_truth(batch, case_index)
        known_pairs = dc.known_pairs_from_batch(batch, case_index)
        completed_pairs, _pred_matrix, _scale = dc.completed_pairs_from_prediction(
            batch,
            pred_norm,
            case_index,
            predicted_sigma_m=args.predicted_sigma,
            predicted_sigma_slope=args.predicted_sigma_slope,
            closest_predicted_pairs_per_anchor=args.closest_predicted_pairs_per_anchor,
        )
        estimate = dc.completion_solution_weak_polish(
            completed_pairs,
            known_pairs,
            max_iterations=args.solver_iterations,
            weak_polish_iterations=args.weak_polish_iterations,
            max_predicted_distance_m=args.weak_completion_max_distance,
            sigma_multiplier=args.weak_completion_sigma_multiplier,
        )
        known_rmse, known_max = dc.pair_metrics(estimate, known_pairs)
        max_offset, median_offset, p95_offset = dc.offset_summary(truth, estimate)
        return {
            "solved": True,
            "solver_max_offset_m": max_offset,
            "solver_median_offset_m": median_offset,
            "solver_p95_offset_m": p95_offset,
            "solver_known_rmse_m": known_rmse,
            "solver_known_max_residual_m": known_max,
        }
    except Exception as exc:
        print(f"solve_failed case={case_index} error={type(exc).__name__}: {exc}", flush=True)
        return {
            "solved": False,
            "solver_max_offset_m": math.nan,
            "solver_median_offset_m": math.nan,
            "solver_p95_offset_m": math.nan,
            "solver_known_rmse_m": math.nan,
            "solver_known_max_residual_m": math.nan,
        }


def finite_median(values: Iterable[float]) -> float:
    finite = [value for value in values if math.isfinite(float(value))]
    return float(np.median(finite)) if finite else math.nan


def finite_quantile(values: Iterable[float], q: float) -> float:
    finite = [value for value in values if math.isfinite(float(value))]
    return float(np.quantile(finite, q)) if finite else math.nan


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write for {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def summarize(detail: list[dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    keys = sorted({(str(row["bucket"]), str(row["method"])) for row in detail})
    baseline_by_bucket: dict[str, dict[str, float]] = {}
    for bucket in sorted({str(row["bucket"]) for row in detail}):
        base = [row for row in detail if row["bucket"] == bucket and row["method"] == "ml_frozen"]
        baseline_by_bucket[bucket] = {
            "mae": finite_median(float(row["missing_mae_m"]) for row in base),
            "solver": finite_median(float(row["solver_max_offset_m"]) for row in base),
        }
    for bucket, method in keys:
        part = [row for row in detail if row["bucket"] == bucket and row["method"] == method]
        median_mae = finite_median(float(row["missing_mae_m"]) for row in part)
        median_solver = finite_median(float(row["solver_max_offset_m"]) for row in part)
        baseline = baseline_by_bucket[bucket]
        rows.append(
            {
                "bucket": bucket,
                "method": method,
                "cases": len(part),
                "median_missing_mae_m": median_mae,
                "mean_missing_mae_m": float(np.mean([float(row["missing_mae_m"]) for row in part])),
                "median_missing_p90_abs_m": finite_median(float(row["missing_p90_abs_m"]) for row in part),
                "median_missing_bias_m": finite_median(float(row["missing_bias_m"]) for row in part),
                "distance_delta_vs_ml_m": median_mae - baseline["mae"],
                "distance_pct_vs_ml": (median_mae / baseline["mae"] - 1.0) if baseline["mae"] > 0 else math.nan,
                "median_solver_max_offset_m": median_solver,
                "p90_solver_max_offset_m": finite_quantile((float(row["solver_max_offset_m"]) for row in part), 0.90),
                "solver_delta_vs_ml_m": median_solver - baseline["solver"] if math.isfinite(baseline["solver"]) else math.nan,
                "median_solver_known_rmse_m": finite_median(float(row["solver_known_rmse_m"]) for row in part),
                "under_1m_solver_rate": float(
                    np.mean(
                        [
                            math.isfinite(float(row["solver_max_offset_m"]))
                            and float(row["solver_max_offset_m"]) <= 1.0
                            for row in part
                        ]
                    )
                ),
            }
        )
    return rows


def generate_eval_cases(args: argparse.Namespace, device: torch.device) -> list[p95.CaseSpec]:
    all_cases: list[p95.CaseSpec] = []
    for bucket in ("random", "grid", "office"):
        print(f"generating_eval bucket={bucket} cases={args.cases_per_bucket}", flush=True)
        all_cases.extend(p95.generate_cases(bucket, args.cases_per_bucket, device))
    return all_cases


def method_predictions(
    batch: dc.GraphBatch,
    base_pred_norm: torch.Tensor,
    models: dict[str, ResidualDistanceCalibrator],
    specs: list[ScaffoldSpec],
) -> dict[str, torch.Tensor]:
    preds = {"ml_frozen": base_pred_norm}
    for method, mode in [
        ("residual_control_no_shortest", "control"),
        ("residual_existing_shortest", "existing_shortest"),
        ("residual_scaffold_channels", "scaffold"),
    ]:
        preds[method] = corrected_prediction(batch, base_pred_norm, models[method], mode=mode, specs=specs)
    return preds


def train_calibrators(
    args: argparse.Namespace,
    base_model: dc.DistanceCompletionNet,
    specs: list[ScaffoldSpec],
    device: torch.device,
) -> tuple[dict[str, ResidualDistanceCalibrator], list[dict[str, object]], dict[str, list[str]]]:
    probe = dc.make_graph_batch(2, device=device, random_fraction=args.random_fraction)
    with torch.no_grad():
        probe_pred = base_model(
            probe.node_features,
            probe.edge_features,
            probe.mask,
            probe.measured_mask,
            probe.pair_mask,
        )
    modes = {
        "residual_control_no_shortest": "control",
        "residual_existing_shortest": "existing_shortest",
        "residual_scaffold_channels": "scaffold",
    }
    feature_names: dict[str, list[str]] = {}
    models: dict[str, ResidualDistanceCalibrator] = {}
    optimizers: dict[str, torch.optim.Optimizer] = {}
    for method, mode in modes.items():
        features, names = feature_tensor(probe, probe_pred, mode=mode, specs=specs)
        feature_names[method] = names
        models[method] = ResidualDistanceCalibrator(
            features.shape[-1],
            hidden=args.hidden,
            layers=args.layers,
            max_delta_norm=args.max_delta_norm,
        ).to(device)
        optimizers[method] = torch.optim.AdamW(models[method].parameters(), lr=args.lr, weight_decay=args.weight_decay)

    history: list[dict[str, object]] = []
    started = time.perf_counter()
    base_model.eval()
    for model in models.values():
        model.train()
    for step in range(1, args.train_steps + 1):
        batch = dc.make_graph_batch(args.train_batch_size, device=device, random_fraction=args.random_fraction)
        with torch.no_grad():
            base_pred = base_model(
                batch.node_features,
                batch.edge_features,
                batch.mask,
                batch.measured_mask,
                batch.pair_mask,
            ).detach()
        for method, mode in modes.items():
            features, base_flat, target = pair_training_tensors(
                batch,
                base_pred,
                mode=mode,
                specs=specs,
                max_pairs=args.max_train_pairs_per_step,
            )
            optimizer = optimizers[method]
            optimizer.zero_grad(set_to_none=True)
            pred = models[method](features, base_flat)
            loss = weighted_loss(pred, target)
            loss.backward()
            nn.utils.clip_grad_norm_(models[method].parameters(), args.grad_clip)
            optimizer.step()
            with torch.no_grad():
                base_loss = weighted_loss(base_flat, target)
            history.append(
                {
                    "step": step,
                    "method": method,
                    "loss": float(loss.detach().cpu()),
                    "base_loss_same_pairs": float(base_loss.detach().cpu()),
                    "pairs": int(target.numel()),
                }
            )
        if step == 1 or step % args.log_every == 0 or step == args.train_steps:
            elapsed = time.perf_counter() - started
            recent = history[-len(modes) :]
            loss_text = " ".join(f"{row['method']}={float(row['loss']):.4f}" for row in recent)
            print(f"train step={step}/{args.train_steps} {loss_text} elapsed_s={elapsed:.1f}", flush=True)
    return models, history, feature_names


@torch.no_grad()
def evaluate(
    args: argparse.Namespace,
    base_model: dc.DistanceCompletionNet,
    models: dict[str, ResidualDistanceCalibrator],
    specs: list[ScaffoldSpec],
    device: torch.device,
) -> list[dict[str, object]]:
    set_seed(args.eval_seed)
    cases = generate_eval_cases(args, device)
    rows: list[dict[str, object]] = []
    base_model.eval()
    for start in range(0, len(cases), args.eval_batch_size):
        chunk = cases[start : start + args.eval_batch_size]
        batch = p95.graph_batch_from_cases(chunk, device)
        base_pred = base_model(
            batch.node_features,
            batch.edge_features,
            batch.mask,
            batch.measured_mask,
            batch.pair_mask,
        )
        preds = method_predictions(batch, base_pred, models, specs)
        for case_index, case in enumerate(chunk):
            known_pairs = int(batch.measured_mask[case_index].sum().detach().cpu().item() // 2)
            full_pairs = batch.node_counts[case_index] * (batch.node_counts[case_index] - 1) // 2
            for method, pred in preds.items():
                dmetrics = distance_metrics(batch, pred, case_index)
                if args.skip_solver:
                    smetrics: dict[str, float | bool] = {
                        "solved": False,
                        "solver_max_offset_m": math.nan,
                        "solver_median_offset_m": math.nan,
                        "solver_p95_offset_m": math.nan,
                        "solver_known_rmse_m": math.nan,
                        "solver_known_max_residual_m": math.nan,
                    }
                else:
                    smetrics = solve_metrics(batch, pred, case_index, args)
                row: dict[str, object] = {
                    "bucket": case.bucket,
                    "case_index": start + case_index,
                    "method": method,
                    "family": case.family,
                    "shape": case.shape,
                    "anchors": batch.node_counts[case_index],
                    "known_pairs": known_pairs,
                    "full_pairs": full_pairs,
                    **dmetrics,
                    **smetrics,
                }
                rows.append(row)
            print(
                f"evaluated case={start + case_index + 1}/{len(cases)} bucket={case.bucket} "
                f"anchors={batch.node_counts[case_index]}",
                flush=True,
            )
    return rows


def write_report(
    path: Path,
    args: argparse.Namespace,
    specs: list[ScaffoldSpec],
    summary: list[dict[str, object]],
    feature_names: dict[str, list[str]],
    *,
    elapsed_s: float,
    device: torch.device,
) -> None:
    scaffold_rows = [row for row in summary if row["method"] == "residual_scaffold_channels"]
    existing_rows = [row for row in summary if row["method"] == "residual_existing_shortest"]
    control_rows = [row for row in summary if row["method"] == "residual_control_no_shortest"]

    def mean_delta(rows: list[dict[str, object]], key: str) -> float:
        values = [float(row[key]) for row in rows if math.isfinite(float(row[key]))]
        return float(np.mean(values)) if values else math.nan

    scaffold_distance_delta = mean_delta(scaffold_rows, "distance_delta_vs_ml_m")
    existing_distance_delta = mean_delta(existing_rows, "distance_delta_vs_ml_m")
    control_distance_delta = mean_delta(control_rows, "distance_delta_vs_ml_m")
    scaffold_solver_delta = mean_delta(scaffold_rows, "solver_delta_vs_ml_m")
    material_threshold_m = 0.01
    material_distance = scaffold_distance_delta <= -material_threshold_m
    tiny_distance = scaffold_distance_delta < 0.0
    helped_solver = math.isfinite(scaffold_solver_delta) and scaffold_solver_delta <= -material_threshold_m
    if material_distance:
        verdict = "materially helped missing-distance calibration"
    elif tiny_distance:
        verdict = "made only a tiny sub-centimeter missing-distance improvement"
    else:
        verdict = "did not beat the frozen ML baseline on missing-distance MAE"
    if not args.skip_solver:
        verdict += "; solver offsets improved" if helped_solver else "; solver offsets did not show a material improvement"

    lines = [
        "# Agent Graph Features ML Probe",
        "",
        f"Verdict: **{verdict}.**",
        "",
        "Existing ML shortest-path coverage:",
        "",
        "- `GraphBatch` already carries `shortest_m` and `hop_count`.",
        "- Node features already include graph closeness, mean/max hop, and mean/max shortest path.",
        "- Edge features already include `shortest_norm` at slot 5, `exp(-shortest_norm)` at slot 6, and `hop_norm` at slot 7.",
        "- The explicit scaffold settings/source channels added here were not present: candidate `max_hops`, `relative_sigma`, `hop_sigma_m`, generated source mask, and scaffold sigma.",
        "",
        "Experiment:",
        "",
        f"- Checkpoint: `{args.checkpoint}`",
        f"- Device: `{device}`",
        f"- Train steps: {args.train_steps}, train batch size: {args.train_batch_size}",
        f"- Eval: {args.cases_per_bucket} fixed generated cases each for random/grid/office buckets",
        f"- Scaffold specs: {', '.join(spec.label for spec in specs)}",
        f"- Runtime: {elapsed_s:.1f}s",
        "",
        "Feature dimensions:",
        "",
    ]
    for method, names in feature_names.items():
        lines.append(f"- `{method}`: {len(names)} channels")
    lines.extend(["", "Summary:", "", "| bucket | method | median MAE m | delta vs ML m | median solver max m | solver delta m |"])
    lines.append("|---|---:|---:|---:|---:|---:|")
    for row in summary:
        lines.append(
            "| {bucket} | {method} | {mae:.3f} | {delta:.3f} | {solver:.3f} | {solver_delta:.3f} |".format(
                bucket=row["bucket"],
                method=row["method"],
                mae=float(row["median_missing_mae_m"]),
                delta=float(row["distance_delta_vs_ml_m"]),
                solver=float(row["median_solver_max_offset_m"]),
                solver_delta=float(row["solver_delta_vs_ml_m"]),
            )
        )
    lines.append("")
    lines.append(
        "Interpretation: because the frozen ML model already consumed shortest distance and hop count, the new scaffold channels mainly test whether exposing the solver-style settings and sigma/source mask adds incremental signal."
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe explicit graph-shortest scaffold channels as a post-ML residual model.")
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=OUTPUTS / "anchor_solver_ml_distance_completion_bigG_bigD_radio12_lr35e5.pt",
    )
    parser.add_argument("--prefix", default="agent_graph_features_ml")
    parser.add_argument("--seed", type=int, default=2026062701)
    parser.add_argument("--eval-seed", type=int, default=2026062702)
    parser.add_argument("--train-steps", type=int, default=80)
    parser.add_argument("--train-batch-size", type=int, default=20)
    parser.add_argument("--eval-batch-size", type=int, default=3)
    parser.add_argument("--cases-per-bucket", type=int, default=4)
    parser.add_argument("--random-fraction", type=float, default=0.5)
    parser.add_argument("--max-train-pairs-per-step", type=int, default=12000)
    parser.add_argument("--hidden", type=int, default=96)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--max-delta-norm", type=float, default=1.25)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--weight-decay", type=float, default=2e-4)
    parser.add_argument("--grad-clip", type=float, default=2.0)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--scaffold-specs", default="2:0.36:0.65,3:0.30:0.85,4:0.30:0.65")
    parser.add_argument("--skip-solver", action="store_true")
    parser.add_argument("--solver-iterations", type=int, default=45)
    parser.add_argument("--weak-polish-iterations", type=int, default=45)
    parser.add_argument("--predicted-sigma", type=float, default=0.55)
    parser.add_argument("--predicted-sigma-slope", type=float, default=0.65)
    parser.add_argument("--closest-predicted-pairs-per-anchor", type=float, default=4.0)
    parser.add_argument("--weak-completion-max-distance", type=float, default=18.0)
    parser.add_argument("--weak-completion-sigma-multiplier", type=float, default=2.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.prefix.startswith("agent_graph_features_ml"):
        raise ValueError("--prefix must start with agent_graph_features_ml")
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    device = choose_device(args.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
    specs = parse_scaffold_specs(args.scaffold_specs)
    print(
        f"device={device} torch={torch.__version__} cuda_available={torch.cuda.is_available()} "
        f"checkpoint={args.checkpoint}",
        flush=True,
    )
    base_model = p95.load_model(args.checkpoint, device)
    started = time.perf_counter()
    models, history, feature_names = train_calibrators(args, base_model, specs, device)
    detail = evaluate(args, base_model, models, specs, device)
    summary = summarize(detail)
    elapsed_s = time.perf_counter() - started

    detail_path = OUTPUTS / f"{args.prefix}_detail.csv"
    summary_path = OUTPUTS / f"{args.prefix}_summary.csv"
    history_path = OUTPUTS / f"{args.prefix}_history.csv"
    report_path = OUTPUTS / f"{args.prefix}_report.md"
    model_path = OUTPUTS / f"{args.prefix}_residual_models.pt"
    write_csv(detail_path, detail)
    write_csv(summary_path, summary)
    write_csv(history_path, history)
    torch.save(
        {
            "args": vars(args),
            "feature_names": feature_names,
            "scaffold_specs": [spec.__dict__ for spec in specs],
            "model_state_dicts": {name: model.state_dict() for name, model in models.items()},
        },
        model_path,
    )
    write_report(report_path, args, specs, summary, feature_names, elapsed_s=elapsed_s, device=device)
    for row in summary:
        print(
            f"summary bucket={row['bucket']} method={row['method']} "
            f"mae={float(row['median_missing_mae_m']):.3f}m "
            f"delta={float(row['distance_delta_vs_ml_m']):+.3f}m "
            f"solver={float(row['median_solver_max_offset_m']):.3f}m "
            f"solver_delta={float(row['solver_delta_vs_ml_m']):+.3f}m",
            flush=True,
        )
    print(f"elapsed_s={elapsed_s:.1f}", flush=True)
    print(f"Wrote {detail_path}", flush=True)
    print(f"Wrote {summary_path}", flush=True)
    print(f"Wrote {history_path}", flush=True)
    print(f"Wrote {report_path}", flush=True)
    print(f"Wrote {model_path}", flush=True)


if __name__ == "__main__":
    main()



