from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
import math
import os
from dataclasses import dataclass
from pathlib import Path
import random
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
sys.path.insert(0, str(ROOT / "work"))

import anchor_solver_ml_distance_completion as dc
import anchor_solver_p95_ml_cases as p95
from anchor_solver_weighted_output import WeightedDistanceCompletionNet

BUCKET_KEYS = ("random", "grid", "office")
BUCKET_LABELS = {"random": "Random 16-32", "grid": "Grid >=16", "office": "Office >=16"}


@dataclass
class RolloutItem:
    batch: dc.GraphBatch
    case_index: int
    action_log_dist_m: torch.Tensor
    action_log_weight: torch.Tensor
    distance_mask: torch.Tensor
    weight_mask: torch.Tensor
    old_logp: torch.Tensor
    advantage: float
    reward: float
    max_offset_m: float
    bucket: str


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return device


def load_weighted_model(path: Path, device: torch.device) -> tuple[WeightedDistanceCompletionNet, dict]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    saved_args = checkpoint.get("args", {})
    use_graph_features = bool(saved_args.get("graph_solution_features", saved_args.get("use_graph_solution_features", False)))
    probe = dc.make_graph_batch(2, device=device, random_fraction=0.5, graph_solution_features=use_graph_features)
    model = WeightedDistanceCompletionNet(
        probe.node_features.shape[-1],
        probe.edge_features.shape[-1],
        hidden=int(saved_args.get("hidden", 144)),
        layers=int(saved_args.get("layers", 5)),
        dropout=float(saved_args.get("dropout", 0.0)),
        max_abs_log_weight=float(saved_args.get("max_abs_log_weight", 2.0)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model, checkpoint


def make_cases(cases_per_bucket: int, device: torch.device) -> list[p95.CaseSpec]:
    cases: list[p95.CaseSpec] = []
    for bucket in BUCKET_KEYS:
        cases.extend(p95.generate_cases(bucket, cases_per_bucket, device))
    random.shuffle(cases)
    return cases


def graph_batch_from_cases(cases: list[p95.CaseSpec], device: torch.device, args: argparse.Namespace) -> dc.GraphBatch:
    return p95.graph_batch_from_cases(cases, device, graph_solution_features=args.graph_solution_features)


def upper_missing_action_mask(batch: dc.GraphBatch, pred_norm: torch.Tensor, closest_per_anchor: float) -> torch.Tensor:
    b, n, _ = pred_norm.shape
    upper = torch.triu(torch.ones((n, n), dtype=torch.bool, device=pred_norm.device), diagonal=1).unsqueeze(0)
    missing = batch.pair_mask & ~batch.measured_mask & upper
    if closest_per_anchor <= 0.0:
        return missing
    out = torch.zeros_like(missing)
    for case_index, node_count in enumerate(batch.node_counts):
        limit = min(int(math.ceil(float(node_count) * closest_per_anchor)), int(missing[case_index].sum().item()))
        if limit <= 0:
            continue
        coords = missing[case_index].nonzero(as_tuple=False)
        values = pred_norm[case_index, coords[:, 0], coords[:, 1]]
        selected = coords[torch.argsort(values)[:limit]]
        out[case_index, selected[:, 0], selected[:, 1]] = True
    return out


def upper_known_action_mask(batch: dc.GraphBatch, like: torch.Tensor) -> torch.Tensor:
    _b, n, _ = like.shape
    upper = torch.triu(torch.ones((n, n), dtype=torch.bool, device=like.device), diagonal=1).unsqueeze(0)
    return batch.pair_mask & batch.measured_mask & upper


def mean_logprob(action: torch.Tensor, mean: torch.Tensor, mask: torch.Tensor, log_std: float) -> torch.Tensor:
    std = math.exp(log_std)
    logp = -0.5 * (((action - mean) / std).pow(2) + 2.0 * log_std + math.log(2.0 * math.pi))
    denom = mask.float().sum(dim=(1, 2)).clamp_min(1.0)
    return (logp * mask.float()).sum(dim=(1, 2)) / denom


def sym_fill(base: torch.Tensor, action: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    out = torch.where(mask, action, base)
    out = torch.where(mask.transpose(1, 2), out.transpose(1, 2), out)
    return 0.5 * (out + out.transpose(1, 2))


def apply_actions(
    batch: dc.GraphBatch,
    pred_norm: torch.Tensor,
    weight: torch.Tensor,
    action_log_dist: torch.Tensor,
    action_log_weight: torch.Tensor,
    distance_mask: torch.Tensor,
    weight_mask: torch.Tensor,
    known_log_dist: torch.Tensor | None = None,
    known_distance_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    scale = batch.scale_m.view(-1, 1, 1)
    lower_m = batch.edge_features[..., 9] * scale
    upper_m = torch.maximum(batch.edge_features[..., 5], batch.edge_features[..., 9] + 0.05) * scale
    action_m = action_log_dist.exp().clamp_min(0.05)
    action_m = torch.minimum(torch.maximum(action_m, lower_m), upper_m.clamp_min(lower_m + 0.05))
    action_norm = action_m / scale
    action_weight = action_log_weight.exp().clamp(0.05, 20.0)
    pred_out = sym_fill(pred_norm, action_norm, distance_mask)
    weight_out = sym_fill(weight, action_weight, weight_mask)
    known_override_m = None
    if known_log_dist is not None and known_distance_mask is not None and bool(known_distance_mask.any()):
        known_m = known_log_dist.exp().clamp_min(0.05)
        known_override_m = sym_fill(batch.measured_dist_m.clamp_min(0.05), known_m, known_distance_mask)
    return pred_out, weight_out, known_override_m


def weighted_pairs(batch: dc.GraphBatch, pred_norm: torch.Tensor, weight: torch.Tensor, case_index: int, args: argparse.Namespace, known_override_m: torch.Tensor | None = None) -> list[dc.AnchorPairDistance]:
    n = batch.node_counts[case_index]
    scale = float(batch.scale_m[case_index].detach().cpu())
    measured = batch.measured_mask[case_index, :n, :n].detach().cpu().numpy()
    measured_dist = batch.measured_dist_m[case_index, :n, :n].detach().cpu().numpy()
    if known_override_m is not None:
        measured_dist = known_override_m[case_index, :n, :n].detach().cpu().numpy()
    pred_m = (pred_norm[case_index, :n, :n].detach().cpu().numpy() * scale).astype(float)
    w = weight[case_index, :n, :n].detach().cpu().numpy().astype(float)
    pairs: list[dc.AnchorPairDistance] = []
    candidates: list[tuple[float, int, int, float]] = []
    for i in range(n):
        for j in range(i + 1, n):
            if measured[i, j]:
                multiplier = max(float(w[i, j]), 1e-6) if args.weight_known_springs else 1.0
                sigma = dc.KNOWN_SIGMA_M / math.sqrt(multiplier)
                pairs.append(dc.AnchorPairDistance(f"A{i:02d}", f"A{j:02d}", float(measured_dist[i, j]), sigma, True, "known-weighted" if args.weight_known_springs else "known"))
            else:
                distance = max(float(pred_m[i, j]), 0.05)
                base_sigma = args.predicted_sigma * (1.0 + args.predicted_sigma_slope * max(distance - dc.EDGE_RADIUS_M, 0.0) / dc.EDGE_RADIUS_M)
                multiplier = max(float(w[i, j]), 1e-6)
                sigma = base_sigma / math.sqrt(multiplier)
                candidates.append((distance, i, j, sigma))
    limit = len(candidates)
    if args.closest_predicted_pairs_per_anchor > 0.0:
        limit = min(limit, max(0, int(math.ceil(args.closest_predicted_pairs_per_anchor * n))))
    candidates.sort(key=lambda item: (item[0], item[1], item[2]))
    for distance, i, j, sigma in candidates[:limit]:
        pairs.append(dc.AnchorPairDistance(f"A{i:02d}", f"A{j:02d}", distance, sigma, True, "predicted-weighted"))
    return pairs


def _iteration_budget(value: int, fallback: int) -> int:
    return fallback if value <= 0 else value


def solve_strong(
    batch: dc.GraphBatch,
    pred_norm: torch.Tensor,
    weight: torch.Tensor,
    case_index: int,
    args: argparse.Namespace,
    known_override_m: torch.Tensor | None = None,
    *,
    solver_iterations: int | None = None,
    polish_iterations: int | None = None,
) -> tuple[float, float]:
    truth = dc.graph_to_truth(batch, case_index)
    known = dc.known_pairs_from_batch(batch, case_index)
    pairs = weighted_pairs(batch, pred_norm, weight, case_index, args, known_override_m)
    solve_iters = args.solver_iterations if solver_iterations is None else solver_iterations
    polish_iters = args.polish_iterations if polish_iterations is None else polish_iterations
    pos = dc.completion_solution(pairs, known, max_iterations=solve_iters, polish_known_iterations=polish_iters)
    max_offset, _med, _p95 = dc.offset_summary(truth, pos)
    rmse, _known_max = dc.pair_metrics(pos, known)
    return max_offset, rmse


def solved_pair_distances(positions: dict[str, tuple[float, float]]) -> list[float]:
    ids = sorted(positions)
    values: list[float] = []
    for index, anchor_a in enumerate(ids):
        ax, ay = positions[anchor_a]
        for anchor_b in ids[index + 1:]:
            bx, by = positions[anchor_b]
            values.append(math.hypot(ax - bx, ay - by))
    return values


def solve_strong_detail(
    batch: dc.GraphBatch,
    pred_norm: torch.Tensor,
    weight: torch.Tensor,
    case_index: int,
    args: argparse.Namespace,
    *,
    solver_iterations: int | None = None,
    polish_iterations: int | None = None,
) -> dict[str, float]:
    truth = dc.graph_to_truth(batch, case_index)
    known = dc.known_pairs_from_batch(batch, case_index)
    pairs = weighted_pairs(batch, pred_norm, weight, case_index, args)
    solve_iters = args.solver_iterations if solver_iterations is None else solver_iterations
    polish_iters = args.polish_iterations if polish_iterations is None else polish_iterations
    pos = dc.completion_solution(pairs, known, max_iterations=solve_iters, polish_known_iterations=polish_iters)
    max_offset, median_offset, p95_offset = dc.offset_summary(truth, pos)
    rmse, known_max = dc.pair_metrics(pos, known)
    distances = solved_pair_distances(pos)
    min_pair = min(distances) if distances else math.inf
    close_count = sum(1 for value in distances if value < args.fold_threshold_m)
    return {
        "max_offset_m": max_offset,
        "median_offset_m": median_offset,
        "p95_offset_m": p95_offset,
        "known_rmse_m": rmse,
        "known_max_residual_m": known_max,
        "min_pair_distance_m": min_pair,
        "close_pair_count": float(close_count),
        "folded_case": float(close_count > 0),
    }


def solve_rollout_cases(
    batch: dc.GraphBatch,
    pred: torch.Tensor,
    weight: torch.Tensor,
    chunk: list[p95.CaseSpec],
    args: argparse.Namespace,
    known_override_m: torch.Tensor | None,
) -> list[tuple[int, float, float]]:
    rollout_solver_iterations = _iteration_budget(args.rollout_solver_iterations, args.solver_iterations)
    rollout_polish_iterations = _iteration_budget(args.rollout_polish_iterations, args.polish_iterations)

    def one(ci: int) -> tuple[int, float, float]:
        max_offset, rmse = solve_strong(
            batch,
            pred,
            weight,
            ci,
            args,
            known_override_m,
            solver_iterations=rollout_solver_iterations,
            polish_iterations=rollout_polish_iterations,
        )
        return ci, max_offset, rmse

    if args.solve_threads <= 1 or len(chunk) <= 1:
        return [one(ci) for ci in range(len(chunk))]
    max_workers = min(args.solve_threads, len(chunk))
    results: list[tuple[int, float, float]] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(one, ci) for ci in range(len(chunk))]
        for future in as_completed(futures):
            results.append(future.result())
    results.sort(key=lambda item: item[0])
    return results


def solve_eval_cases(
    batch: dc.GraphBatch,
    pred: torch.Tensor,
    weight: torch.Tensor,
    chunk: list[p95.CaseSpec],
    args: argparse.Namespace,
) -> list[tuple[int, dict[str, float]]]:
    eval_solver_iterations = _iteration_budget(args.eval_solver_iterations, args.solver_iterations)
    eval_polish_iterations = _iteration_budget(args.eval_polish_iterations, args.polish_iterations)

    def one(ci: int) -> tuple[int, dict[str, float]]:
        return (
            ci,
            solve_strong_detail(
                batch,
                pred,
                weight,
                ci,
                args,
                solver_iterations=eval_solver_iterations,
                polish_iterations=eval_polish_iterations,
            ),
        )

    if args.solve_threads <= 1 or len(chunk) <= 1:
        return [one(ci) for ci in range(len(chunk))]
    max_workers = min(args.solve_threads, len(chunk))
    results: list[tuple[int, dict[str, float]]] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(one, ci) for ci in range(len(chunk))]
        for future in as_completed(futures):
            results.append(future.result())
    results.sort(key=lambda item: item[0])
    return results


@torch.no_grad()
def collect_rollouts(model: WeightedDistanceCompletionNet, cases: list[p95.CaseSpec], args: argparse.Namespace, device: torch.device) -> list[RolloutItem]:
    model.eval()
    rollouts: list[RolloutItem] = []
    for start in range(0, len(cases), args.batch_size):
        chunk = cases[start : start + args.batch_size]
        batch = graph_batch_from_cases(chunk, device, args)
        pred, weight = model(batch.node_features, batch.edge_features, batch.mask, batch.measured_mask, batch.pair_mask)
        missing_mask = upper_missing_action_mask(batch, pred, args.closest_predicted_pairs_per_anchor)
        known_upper = upper_known_action_mask(batch, pred)
        known_distance_mask = known_upper if args.perturb_known_distances else torch.zeros_like(known_upper)
        weight_mask = missing_mask | (known_upper if args.weight_known_springs else torch.zeros_like(known_upper))
        mu_dist = (pred * batch.scale_m.view(-1, 1, 1)).clamp_min(0.05).log()
        mu_weight = weight.clamp_min(1e-6).log()
        known_base_m = batch.measured_dist_m.clamp_min(0.05)
        known_base_log = known_base_m.log()
        for sample in range(args.samples_per_case):
            a_dist = torch.where(missing_mask, mu_dist + torch.randn_like(mu_dist) * math.exp(args.distance_log_std), mu_dist)
            a_weight = torch.where(weight_mask, mu_weight + torch.randn_like(mu_weight) * math.exp(args.weight_log_std), mu_weight)
            known_log_dist = None
            if args.perturb_known_distances:
                known_sample_m = (known_base_m + torch.randn_like(known_base_m) * args.known_distance_std_m).clamp_min(0.05)
                known_log_dist = torch.where(known_distance_mask, known_sample_m.log(), known_base_log)
            old_logp = mean_logprob(a_dist, mu_dist, missing_mask, args.distance_log_std) + mean_logprob(a_weight, mu_weight, weight_mask, args.weight_log_std)
            pred_a, weight_a, known_override_m = apply_actions(batch, pred, weight, a_dist, a_weight, missing_mask, weight_mask, known_log_dist, known_distance_mask)
            offsets = []
            rewards = []
            for ci, max_offset, _rmse in solve_rollout_cases(batch, pred_a, weight_a, chunk, args, known_override_m):
                case = chunk[ci]
                reward = -(max_offset ** 2)
                offsets.append(max_offset)
                rewards.append(reward)
                rollouts.append(RolloutItem(batch, ci, a_dist[ci:ci+1].detach(), a_weight[ci:ci+1].detach(), missing_mask[ci:ci+1].detach(), weight_mask[ci:ci+1].detach(), old_logp[ci:ci+1].detach(), 0.0, reward, max_offset, case.bucket))
            w_values = a_weight[weight_mask].exp().detach()
            if int(w_values.numel()) > 0:
                w50 = float(torch.quantile(w_values, 0.50).detach().cpu())
                w95 = float(torch.quantile(w_values, 0.95).detach().cpu())
            else:
                w50 = 1.0; w95 = 1.0
            if known_override_m is not None and bool(known_distance_mask.any()):
                known_abs = (known_override_m - batch.measured_dist_m).abs()[known_distance_mask].detach()
                k50 = float(torch.quantile(known_abs, 0.50).detach().cpu()) if int(known_abs.numel()) > 0 else 0.0
                k95 = float(torch.quantile(known_abs, 0.95).detach().cpu()) if int(known_abs.numel()) > 0 else 0.0
            else:
                k50 = 0.0; k95 = 0.0
            if getattr(args, "rollout_log", True):
                print(f"rollout chunk={start // args.batch_size + 1} sample={sample + 1}/{args.samples_per_case} mean_max={np.mean(offsets):.3f} reward={np.mean(rewards):.4f} w50={w50:.3f} w95={w95:.3f} known_abs50={k50:.3f} known_abs95={k95:.3f}", flush=True)
    by_case: dict[tuple[int, int], list[RolloutItem]] = {}
    for item in rollouts:
        by_case.setdefault((id(item.batch), item.case_index), []).append(item)
    rewards = np.array([r.reward for r in rollouts], dtype=float)
    std = max(float(rewards.std()), 1e-4)
    for items in by_case.values():
        baseline = float(np.mean([x.reward for x in items]))
        for item in items:
            item.advantage = (item.reward - baseline) / std
    return rollouts
def batch_from_rollouts(items: list[RolloutItem]) -> dc.GraphBatch:
    return dc.GraphBatch(
        positions_m=torch.cat([x.batch.positions_m[x.case_index:x.case_index+1] for x in items], 0),
        mask=torch.cat([x.batch.mask[x.case_index:x.case_index+1] for x in items], 0),
        true_dist_m=torch.cat([x.batch.true_dist_m[x.case_index:x.case_index+1] for x in items], 0),
        measured_dist_m=torch.cat([x.batch.measured_dist_m[x.case_index:x.case_index+1] for x in items], 0),
        measured_mask=torch.cat([x.batch.measured_mask[x.case_index:x.case_index+1] for x in items], 0),
        pair_mask=torch.cat([x.batch.pair_mask[x.case_index:x.case_index+1] for x in items], 0),
        shortest_m=torch.cat([x.batch.shortest_m[x.case_index:x.case_index+1] for x in items], 0),
        hop_count=torch.cat([x.batch.hop_count[x.case_index:x.case_index+1] for x in items], 0),
        scale_m=torch.cat([x.batch.scale_m[x.case_index:x.case_index+1] for x in items], 0),
        node_features=torch.cat([x.batch.node_features[x.case_index:x.case_index+1] for x in items], 0),
        edge_features=torch.cat([x.batch.edge_features[x.case_index:x.case_index+1] for x in items], 0),
        family=[x.batch.family[x.case_index] for x in items],
        shape=[x.batch.shape[x.case_index] for x in items],
        node_counts=[x.batch.node_counts[x.case_index] for x in items],
    )


def ppo_update(model: WeightedDistanceCompletionNet, opt: torch.optim.Optimizer, rollouts: list[RolloutItem], args: argparse.Namespace) -> dict[str, float]:
    model.train()
    losses = []
    ratios = []
    random.shuffle(rollouts)
    for _epoch in range(args.ppo_epochs):
        for start in range(0, len(rollouts), args.ppo_minibatch):
            items = rollouts[start:start + args.ppo_minibatch]
            if not items:
                continue
            batch = batch_from_rollouts(items)
            pred, weight = model(batch.node_features, batch.edge_features, batch.mask, batch.measured_mask, batch.pair_mask)
            mu_dist = (pred * batch.scale_m.view(-1, 1, 1)).clamp_min(0.05).log()
            mu_weight = weight.clamp_min(1e-6).log()
            a_dist = torch.cat([x.action_log_dist_m for x in items], 0).to(mu_dist.device)
            a_weight = torch.cat([x.action_log_weight for x in items], 0).to(mu_dist.device)
            distance_mask = torch.cat([x.distance_mask for x in items], 0).to(mu_dist.device)
            weight_mask = torch.cat([x.weight_mask for x in items], 0).to(mu_dist.device)
            old = torch.cat([x.old_logp for x in items], 0).to(mu_dist.device)
            adv = torch.tensor([x.advantage for x in items], dtype=mu_dist.dtype, device=mu_dist.device)
            logp = mean_logprob(a_dist, mu_dist, distance_mask, args.distance_log_std) + mean_logprob(a_weight, mu_weight, weight_mask, args.weight_log_std)
            ratio = torch.exp(torch.clamp(logp - old, -5.0, 5.0))
            clipped = torch.clamp(ratio, 1.0 - args.clip_epsilon, 1.0 + args.clip_epsilon) * adv
            policy_loss = -torch.minimum(ratio * adv, clipped).mean()
            # Mild anchors: keep distances sane and discourage instant huge weight drift in this smoke run.
            sup_loss, _ = dc.distance_completion_loss(pred, batch, edm_weight=args.edm_weight, radio_lower_weight=0.0)
            reg_mask = batch.pair_mask if args.weight_known_springs else (batch.pair_mask & ~batch.measured_mask)
            weight_reg = torch.log(weight[reg_mask].clamp_min(1e-6)).pow(2).mean() if bool(reg_mask.any()) else policy_loss.detach() * 0.0
            loss = policy_loss + args.supervised_weight * sup_loss + args.weight_reg * weight_reg
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            losses.append(float(policy_loss.detach().cpu()))
            ratios.extend(float(v) for v in ratio.detach().cpu().tolist())
    return {"policy_loss": float(np.mean(losses)) if losses else 0.0, "ratio_mean": float(np.mean(ratios)) if ratios else 1.0}


@torch.no_grad()
def evaluate(model: WeightedDistanceCompletionNet, cases: list[p95.CaseSpec], args: argparse.Namespace, device: torch.device, label: str) -> list[dict[str, float | int | str]]:
    model.eval()
    rows = []
    idx = 0
    for start in range(0, len(cases), args.batch_size):
        chunk = cases[start:start + args.batch_size]
        batch = graph_batch_from_cases(chunk, device, args)
        pred, weight = model(batch.node_features, batch.edge_features, batch.mask, batch.measured_mask, batch.pair_mask)
        metrics_by_ci = dict(solve_eval_cases(batch, pred, weight, chunk, args))
        for ci, case in enumerate(chunk):
            metrics = metrics_by_ci[ci]
            n = batch.node_counts[ci]
            pair_mask_np = batch.pair_mask[ci, :n, :n].detach().cpu().numpy()
            measured_np = batch.measured_mask[ci, :n, :n].detach().cpu().numpy()
            w_np = weight[ci, :n, :n].detach().cpu().numpy()
            upper = np.triu(np.ones((n, n), dtype=bool), 1)
            known_w = w_np[upper & pair_mask_np & measured_np]
            pred_w = w_np[upper & pair_mask_np & ~measured_np]
            diag = dc.measured_graph_diagnostics(case.points)
            row = {
                "label": label,
                "bucket": case.bucket,
                "case_index": idx,
                "family": case.family,
                "shape": case.shape,
                "anchors": n,
                "known_pairs": int(np.sum(upper & pair_mask_np & measured_np)),
                "min_degree": diag["min_degree"],
                "mean_degree": diag["mean_degree"],
                "rigidity_surplus": diag["rigidity_surplus"],
                "vertex_connectivity_capped3": diag["vertex_connectivity_capped3"],
                **metrics,
                "known_weight_p05": float(np.quantile(known_w, 0.05)) if known_w.size else 1.0,
                "known_weight_p50": float(np.quantile(known_w, 0.50)) if known_w.size else 1.0,
                "known_weight_p95": float(np.quantile(known_w, 0.95)) if known_w.size else 1.0,
                "pred_weight_p05": float(np.quantile(pred_w, 0.05)) if pred_w.size else 1.0,
                "pred_weight_p50": float(np.quantile(pred_w, 0.50)) if pred_w.size else 1.0,
                "pred_weight_p95": float(np.quantile(pred_w, 0.95)) if pred_w.size else 1.0,
            }
            row["median_weight"] = row["pred_weight_p50"]
            rows.append(row)
            idx += 1
    return rows


def summarize(rows: list[dict[str, float | int | str]]) -> list[dict[str, float | int | str]]:
    out = []
    for label in sorted({r["label"] for r in rows}):
        for bucket in sorted({r["bucket"] for r in rows if r["label"] == label}):
            part = [r for r in rows if r["label"] == label and r["bucket"] == bucket]
            offsets = np.array([float(r["max_offset_m"]) for r in part])
            rmse = np.array([float(r["known_rmse_m"]) for r in part])
            close = np.array([float(r.get("close_pair_count", 0.0)) for r in part])
            pred_p05 = np.array([float(r.get("pred_weight_p05", 1.0)) for r in part])
            pred_p50 = np.array([float(r.get("pred_weight_p50", 1.0)) for r in part])
            pred_p95 = np.array([float(r.get("pred_weight_p95", 1.0)) for r in part])
            known_p50 = np.array([float(r.get("known_weight_p50", 1.0)) for r in part])
            out.append({
                "label": label,
                "bucket": bucket,
                "cases": len(part),
                "median_max_offset_m": float(np.median(offsets)),
                "p90_max_offset_m": float(np.quantile(offsets, 0.90)),
                "p95_max_offset_m": float(np.quantile(offsets, 0.95)),
                "max_offset_m": float(np.max(offsets)),
                "under_20cm": float(np.mean(offsets <= 0.20)),
                "under_50cm": float(np.mean(offsets <= 0.50)),
                "under_1m": float(np.mean(offsets <= 1.0)),
                "under_2m": float(np.mean(offsets <= 2.0)),
                "mean_squared_max_offset": float(np.mean(offsets * offsets)),
                "median_known_rmse_m": float(np.median(rmse)),
                "p95_known_rmse_m": float(np.quantile(rmse, 0.95)),
                "folded_case_rate": float(np.mean(close > 0.0)),
                "mean_close_pair_count": float(np.mean(close)),
                "known_weight_p50": float(np.median(known_p50)),
                "pred_weight_p05": float(np.median(pred_p05)),
                "pred_weight_p50": float(np.median(pred_p50)),
                "pred_weight_p95": float(np.median(pred_p95)),
                "median_weight": float(np.median(pred_p50)),
            })
    return out


def write_csv(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, list(rows[0].keys()))
        writer.writeheader(); writer.writerows(rows)


def save(path: Path, model: WeightedDistanceCompletionNet, args: argparse.Namespace, source: dict, history: list[dict[str, float | int | str]]) -> None:
    saved_args = dict(source.get("args", {})); saved_args.update(vars(args)); saved_args["weighted_ppo_strong_branch"] = True
    saved_args["graph_solution_features"] = bool(args.graph_solution_features)
    torch.save({"model_state_dict": model.state_dict(), "args": saved_args, "source_checkpoint": str(args.init_checkpoint), "history": history, "weighted_output": True, "spring_weight_is_multiplier": True, "graph_solution_features": bool(args.graph_solution_features)}, path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Small PPO run for weighted-output GNN on strong/completed branch.")
    parser.add_argument("--init-checkpoint", type=Path, default=OUTPUTS / "anchor_solver_weighted_distill_neutral_gpu_best_best.pt")
    parser.add_argument("--prefix", default="anchor_solver_weighted_ppo_strong_smoke")
    parser.add_argument("--max-runtime-minutes", type=float, default=0.0, help="Gracefully stop after this many minutes; 0 disables.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=2026062729)
    parser.add_argument("--updates", type=int, default=2)
    parser.add_argument("--train-cases-per-bucket", type=int, default=2)
    parser.add_argument("--eval-cases-per-bucket", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--samples-per-case", type=int, default=4)
    parser.add_argument("--ppo-epochs", type=int, default=3)
    parser.add_argument("--ppo-minibatch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1.5e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--clip-epsilon", type=float, default=0.2)
    parser.add_argument("--distance-log-std", type=float, default=-1.8)
    parser.add_argument("--weight-log-std", type=float, default=-1.4)
    parser.add_argument("--perturb-known-distances", action="store_true")
    parser.add_argument("--known-distance-std-m", type=float, default=0.08)
    parser.add_argument("--weight-known-springs", action="store_true")
    parser.add_argument("--supervised-weight", type=float, default=0.025)
    parser.add_argument("--weight-reg", type=float, default=0.01)
    parser.add_argument("--edm-weight", type=float, default=0.02)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--solver-iterations", type=int, default=60)
    parser.add_argument("--polish-iterations", type=int, default=60)
    parser.add_argument("--rollout-solver-iterations", type=int, default=35, help="Solver iterations used for PPO sampled actions; <=0 uses --solver-iterations.")
    parser.add_argument("--rollout-polish-iterations", type=int, default=20, help="Known-range polish iterations used for PPO sampled actions; <=0 uses --polish-iterations.")
    parser.add_argument("--eval-solver-iterations", type=int, default=0, help="Evaluation solver iterations; <=0 uses --solver-iterations.")
    parser.add_argument("--eval-polish-iterations", type=int, default=0, help="Evaluation polish iterations; <=0 uses --polish-iterations.")
    parser.add_argument("--solve-threads", type=int, default=max(1, min(4, (os.cpu_count() or 2) - 1)), help="CPU threads for independent per-case strong solves.")
    parser.add_argument("--closest-predicted-pairs-per-anchor", type=float, default=5.0)
    parser.add_argument("--predicted-sigma", type=float, default=0.55)
    parser.add_argument("--predicted-sigma-slope", type=float, default=0.65)
    parser.add_argument("--graph-solution-features", action="store_true")
    parser.add_argument("--fold-threshold-m", type=float, default=1.65)
    args = parser.parse_args()
    OUTPUTS.mkdir(exist_ok=True)
    set_seed(args.seed)
    device = choose_device(args.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.benchmark = True; torch.set_float32_matmul_precision("high")
    print(f"device={device} init={args.init_checkpoint} branch=completed", flush=True)
    model, source = load_weighted_model(args.init_checkpoint, device)
    source_args = source.get("args", {})
    if source_args.get("graph_solution_features", source_args.get("use_graph_solution_features", False)):
        args.graph_solution_features = True
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    gen_device = torch.device("cpu")
    eval_cases = make_cases(args.eval_cases_per_bucket, gen_device)
    eval_rows = evaluate(model, eval_cases, args, device, "before")
    history = []
    best_score = float(np.mean([float(r["max_offset_m"]) ** 2 for r in eval_rows]))
    best_path = OUTPUTS / f"{args.prefix}_best.pt"
    save(best_path, model, args, source, history)
    for row in summarize(eval_rows):
        print(f"before bucket={row['bucket']} p95={float(row['p95_max_offset_m']):.3f}m median={float(row['median_max_offset_m']):.3f}m under1={float(row['under_1m']):.2f} w={float(row['median_weight']):.3f} fold={float(row.get('folded_case_rate', 0.0)):.2f} rmse={float(row.get('median_known_rmse_m', 0.0)):.4f}", flush=True)
    started = time.perf_counter()
    for update in range(1, args.updates + 1):
        train_cases = make_cases(args.train_cases_per_bucket, gen_device)
        rollouts = collect_rollouts(model, train_cases, args, device)
        stats = ppo_update(model, opt, rollouts, args)
        upd_rows = evaluate(model, eval_cases, args, device, f"update_{update}")
        eval_rows.extend(upd_rows)
        score = float(np.mean([float(r["max_offset_m"]) ** 2 for r in upd_rows]))
        rec = {"update": update, "rollouts": len(rollouts), "mean_reward": float(np.mean([r.reward for r in rollouts])), "eval_mean_squared_max_offset": score, "policy_loss": stats["policy_loss"], "ratio_mean": stats["ratio_mean"], "elapsed_s": time.perf_counter() - started, "is_best": 0}
        if score < best_score:
            best_score = score; rec["is_best"] = 1; save(best_path, model, args, source, history + [rec]); print(f"checkpoint_best update={update} mean_sq={score:.4f} path={best_path}", flush=True)
        history.append(rec)
        for row in summarize(upd_rows):
            print(f"eval update={update} bucket={row['bucket']} p95={float(row['p95_max_offset_m']):.3f}m median={float(row['median_max_offset_m']):.3f}m under1={float(row['under_1m']):.2f} w={float(row['median_weight']):.3f} fold={float(row.get('folded_case_rate', 0.0)):.2f} rmse={float(row.get('median_known_rmse_m', 0.0)):.4f}", flush=True)
        print(f"ppo_update={update}/{args.updates} rollouts={len(rollouts)} reward={rec['mean_reward']:.4f} policy_loss={rec['policy_loss']:.5f} ratio={rec['ratio_mean']:.3f}", flush=True)
        save(OUTPUTS / f"{args.prefix}_latest.pt", model, args, source, history)
        write_csv(OUTPUTS / f"{args.prefix}_eval_detail.csv", eval_rows)
        write_csv(OUTPUTS / f"{args.prefix}_eval_summary.csv", summarize(eval_rows))
        write_csv(OUTPUTS / f"{args.prefix}_history.csv", history)
        if args.max_runtime_minutes > 0.0 and (time.perf_counter() - started) >= args.max_runtime_minutes * 60.0:
            print(f"max_runtime_reached minutes={args.max_runtime_minutes:.2f} update={update}", flush=True)
            break
    after_rows = evaluate(model, eval_cases, args, device, "after")
    eval_rows.extend(after_rows)
    save(OUTPUTS / f"{args.prefix}.pt", model, args, source, history)
    write_csv(OUTPUTS / f"{args.prefix}_eval_detail.csv", eval_rows)
    write_csv(OUTPUTS / f"{args.prefix}_eval_summary.csv", summarize(eval_rows))
    write_csv(OUTPUTS / f"{args.prefix}_history.csv", history)
    print(f"Wrote {OUTPUTS / (args.prefix + '.pt')}", flush=True)
    print(f"Wrote {best_path}", flush=True)


if __name__ == "__main__":
    main()





