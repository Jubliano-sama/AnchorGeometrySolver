from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
import random
import sys
import time
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
sys.path.insert(0, str(ROOT / "work"))

import anchor_solver_ml_distance_completion as dc
import anchor_solver_p95_ml_cases as p95


BUCKET_KEYS = ("random", "grid", "office")
BUCKET_LABELS = {"random": "Random 16-32", "grid": "Grid >=16", "office": "Office >=16"}
BUCKET_KEYS_BY_LABEL = {label: key for key, label in BUCKET_LABELS.items()}


@dataclass
class RolloutItem:
    batch: dc.GraphBatch
    case_index: int
    action_log_m: torch.Tensor
    action_mask: torch.Tensor
    old_logp: torch.Tensor
    advantage: float
    reward: float
    max_offset_m: float
    bucket: str
    shape: str
    anchors: int


def parse_checkpoint(path: Path, device: torch.device) -> tuple[dc.DistanceCompletionNet, dict]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    saved_args = checkpoint.get("args", {})
    probe = dc.make_graph_batch(2, device=device, random_fraction=0.5)
    model = dc.DistanceCompletionNet(
        probe.node_features.shape[-1],
        probe.edge_features.shape[-1],
        hidden=int(saved_args.get("hidden", 144)),
        layers=int(saved_args.get("layers", 5)),
        dropout=float(saved_args.get("dropout", 0.0)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model, saved_args


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested, but torch.cuda.is_available() is false")
    return device


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def generate_mixed_cases(cases_per_bucket: int, device: torch.device) -> list[p95.CaseSpec]:
    cases: list[p95.CaseSpec] = []
    for bucket in BUCKET_KEYS:
        cases.extend(p95.generate_cases(bucket, cases_per_bucket, device))
    random.shuffle(cases)
    return cases


def upper_missing_action_mask(batch: dc.GraphBatch, pred_norm: torch.Tensor, *, closest_per_anchor: float) -> torch.Tensor:
    batch_size, nodes, _ = pred_norm.shape
    upper = torch.triu(torch.ones((nodes, nodes), dtype=torch.bool, device=pred_norm.device), diagonal=1).unsqueeze(0)
    missing = batch.pair_mask & ~batch.measured_mask & upper
    if closest_per_anchor <= 0.0:
        return missing
    out = torch.zeros_like(missing)
    for case_index, n in enumerate(batch.node_counts):
        limit = min(int(math.ceil(float(n) * closest_per_anchor)), int(missing[case_index].sum().item()))
        if limit <= 0:
            continue
        coords = missing[case_index].nonzero(as_tuple=False)
        values = pred_norm[case_index, coords[:, 0], coords[:, 1]]
        order = torch.argsort(values)[:limit]
        selected = coords[order]
        out[case_index, selected[:, 0], selected[:, 1]] = True
    return out


def mean_logprob(log_action_m: torch.Tensor, mu_log_m: torch.Tensor, mask: torch.Tensor, log_std: float) -> torch.Tensor:
    std = math.exp(log_std)
    variance = std * std
    logp = -0.5 * ((log_action_m - mu_log_m).pow(2) / variance + 2.0 * log_std + math.log(2.0 * math.pi))
    denom = mask.float().sum(dim=(1, 2)).clamp_min(1.0)
    return (logp * mask.float()).sum(dim=(1, 2)) / denom


def apply_action_to_prediction(batch: dc.GraphBatch, base_pred_norm: torch.Tensor, action_log_m: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    scale = batch.scale_m.view(-1, 1, 1)
    lower = batch.edge_features[..., 9] * scale
    upper = torch.maximum(batch.edge_features[..., 5], batch.edge_features[..., 9] + 0.05) * scale
    action_m = action_log_m.exp().clamp_min(0.05)
    action_m = torch.minimum(torch.maximum(action_m, lower), upper.clamp_min(lower + 0.05))
    action_norm = action_m / scale
    pred = torch.where(mask, action_norm, base_pred_norm)
    pred = torch.where(mask.transpose(1, 2), pred.transpose(1, 2), pred)
    pred = 0.5 * (pred + pred.transpose(1, 2))
    return pred


def solve_case_max_offset(
    batch: dc.GraphBatch,
    pred_norm: torch.Tensor,
    case_index: int,
    *,
    branch: str,
    solver_iterations: int,
    weak_polish_iterations: int,
    closest_per_anchor: float,
    predicted_sigma: float,
    predicted_sigma_slope: float,
    weak_completion_max_distance: float,
    weak_completion_sigma_multiplier: float,
) -> tuple[float, float]:
    truth = dc.graph_to_truth(batch, case_index)
    known_pairs = dc.known_pairs_from_batch(batch, case_index)
    completed_pairs, _matrix, _scale = dc.completed_pairs_from_prediction(
        batch,
        pred_norm,
        case_index,
        predicted_sigma_m=predicted_sigma,
        predicted_sigma_slope=predicted_sigma_slope,
        closest_predicted_pairs_per_anchor=closest_per_anchor,
    )
    if branch == "completed":
        positions = dc.completion_solution(
            completed_pairs,
            known_pairs,
            max_iterations=solver_iterations,
            polish_known_iterations=solver_iterations,
        )
    elif branch == "weak":
        positions = dc.completion_solution_weak_polish(
            completed_pairs,
            known_pairs,
            max_iterations=solver_iterations,
            weak_polish_iterations=weak_polish_iterations,
            max_predicted_distance_m=weak_completion_max_distance,
            sigma_multiplier=weak_completion_sigma_multiplier,
        )
    else:
        raise ValueError(branch)
    max_offset, _median_offset, _p95_offset = dc.offset_summary(truth, positions)
    known_rmse, _known_max = dc.pair_metrics(positions, known_pairs)
    return max_offset, known_rmse


@torch.no_grad()
def collect_rollouts(
    model: dc.DistanceCompletionNet,
    cases: list[p95.CaseSpec],
    args: argparse.Namespace,
    device: torch.device,
) -> list[RolloutItem]:
    model.eval()
    rollouts: list[RolloutItem] = []
    for start in range(0, len(cases), args.batch_size):
        chunk_cases = cases[start : start + args.batch_size]
        batch = p95.graph_batch_from_cases(chunk_cases, device)
        base_pred = model(batch.node_features, batch.edge_features, batch.mask, batch.measured_mask, batch.pair_mask)
        action_mask = upper_missing_action_mask(batch, base_pred, closest_per_anchor=args.closest_predicted_pairs_per_anchor)
        scale = batch.scale_m.view(-1, 1, 1)
        mu_log_m = (base_pred * scale).clamp_min(0.05).log()
        for sample_index in range(args.samples_per_case):
            noise = torch.randn_like(mu_log_m) * math.exp(args.policy_log_std)
            action_log_m = torch.where(action_mask, mu_log_m + noise, mu_log_m)
            old_logp = mean_logprob(action_log_m, mu_log_m, action_mask, args.policy_log_std)
            action_pred = apply_action_to_prediction(batch, base_pred, action_log_m, action_mask)
            rewards_for_chunk: list[float] = []
            offsets_for_chunk: list[float] = []
            for case_index, case in enumerate(chunk_cases):
                max_offset, known_rmse = solve_case_max_offset(
                    batch,
                    action_pred,
                    case_index,
                    branch=args.branch,
                    solver_iterations=args.solver_iterations,
                    weak_polish_iterations=args.weak_polish_iterations,
                    closest_per_anchor=args.closest_predicted_pairs_per_anchor,
                    predicted_sigma=args.predicted_sigma,
                    predicted_sigma_slope=args.predicted_sigma_slope,
                    weak_completion_max_distance=args.weak_completion_max_distance,
                    weak_completion_sigma_multiplier=args.weak_completion_sigma_multiplier,
                )
                reward = -(max_offset ** 2)
                rewards_for_chunk.append(reward)
                offsets_for_chunk.append(max_offset)
                rollouts.append(
                    RolloutItem(
                        batch=batch,
                        case_index=case_index,
                        action_log_m=action_log_m[case_index : case_index + 1].detach(),
                        action_mask=action_mask[case_index : case_index + 1].detach(),
                        old_logp=old_logp[case_index : case_index + 1].detach(),
                        advantage=0.0,
                        reward=reward,
                        max_offset_m=max_offset,
                        bucket=case.bucket,
                        shape=case.shape,
                        anchors=int(batch.node_counts[case_index]),
                    )
                )
            print(
                f"rollout chunk={start // args.batch_size + 1} sample={sample_index + 1}/{args.samples_per_case} "
                f"mean_max={float(np.mean(offsets_for_chunk)):.3f}m mean_reward={float(np.mean(rewards_for_chunk)):.4f}",
                flush=True,
            )
    by_case: dict[tuple[int, int], list[RolloutItem]] = {}
    for item in rollouts:
        by_case.setdefault((id(item.batch), item.case_index), []).append(item)
    all_rewards = np.array([item.reward for item in rollouts], dtype=float)
    reward_std = float(all_rewards.std()) if all_rewards.size else 1.0
    reward_std = max(reward_std, 1e-4)
    for items in by_case.values():
        baseline = float(np.mean([item.reward for item in items]))
        for item in items:
            item.advantage = (item.reward - baseline) / reward_std
    return rollouts


def rollout_minibatch_to_graph_batch(minibatch: list[RolloutItem]) -> dc.GraphBatch:
    positions = torch.cat([item.batch.positions_m[item.case_index : item.case_index + 1] for item in minibatch], dim=0)
    mask = torch.cat([item.batch.mask[item.case_index : item.case_index + 1] for item in minibatch], dim=0)
    true_dist = torch.cat([item.batch.true_dist_m[item.case_index : item.case_index + 1] for item in minibatch], dim=0)
    measured_dist = torch.cat([item.batch.measured_dist_m[item.case_index : item.case_index + 1] for item in minibatch], dim=0)
    measured_mask = torch.cat([item.batch.measured_mask[item.case_index : item.case_index + 1] for item in minibatch], dim=0)
    pair_mask = torch.cat([item.batch.pair_mask[item.case_index : item.case_index + 1] for item in minibatch], dim=0)
    shortest = torch.cat([item.batch.shortest_m[item.case_index : item.case_index + 1] for item in minibatch], dim=0)
    hops = torch.cat([item.batch.hop_count[item.case_index : item.case_index + 1] for item in minibatch], dim=0)
    scale = torch.cat([item.batch.scale_m[item.case_index : item.case_index + 1] for item in minibatch], dim=0)
    node_features = torch.cat([item.batch.node_features[item.case_index : item.case_index + 1] for item in minibatch], dim=0)
    edge_features = torch.cat([item.batch.edge_features[item.case_index : item.case_index + 1] for item in minibatch], dim=0)
    return dc.GraphBatch(
        positions_m=positions,
        mask=mask,
        true_dist_m=true_dist,
        measured_dist_m=measured_dist,
        measured_mask=measured_mask,
        pair_mask=pair_mask,
        shortest_m=shortest,
        hop_count=hops,
        scale_m=scale,
        node_features=node_features,
        edge_features=edge_features,
        family=[item.batch.family[item.case_index] for item in minibatch],
        shape=[item.batch.shape[item.case_index] for item in minibatch],
        node_counts=[item.batch.node_counts[item.case_index] for item in minibatch],
    )


def ppo_update(
    model: dc.DistanceCompletionNet,
    optimizer: torch.optim.Optimizer,
    rollouts: list[RolloutItem],
    args: argparse.Namespace,
) -> dict[str, float]:
    model.train()
    random.shuffle(rollouts)
    policy_losses: list[float] = []
    supervised_losses: list[float] = []
    ratios: list[float] = []
    for epoch in range(args.ppo_epochs):
        for start in range(0, len(rollouts), args.ppo_minibatch):
            minibatch = rollouts[start : start + args.ppo_minibatch]
            if not minibatch:
                continue
            pseudo_batch = rollout_minibatch_to_graph_batch(minibatch)
            pred = model(
                pseudo_batch.node_features,
                pseudo_batch.edge_features,
                pseudo_batch.mask,
                pseudo_batch.measured_mask,
                pseudo_batch.pair_mask,
            )
            scale = pseudo_batch.scale_m.view(-1, 1, 1)
            mu_log_m = (pred * scale).clamp_min(0.05).log()
            action_log_m = torch.cat([item.action_log_m for item in minibatch], dim=0).to(mu_log_m.device)
            action_mask = torch.cat([item.action_mask for item in minibatch], dim=0).to(mu_log_m.device)
            old_logp = torch.cat([item.old_logp for item in minibatch], dim=0).to(mu_log_m.device)
            advantage = torch.tensor([item.advantage for item in minibatch], dtype=mu_log_m.dtype, device=mu_log_m.device)
            logp = mean_logprob(action_log_m, mu_log_m, action_mask, args.policy_log_std)
            ratio = torch.exp(torch.clamp(logp - old_logp, -5.0, 5.0))
            unclipped = ratio * advantage
            clipped = torch.clamp(ratio, 1.0 - args.clip_epsilon, 1.0 + args.clip_epsilon) * advantage
            policy_loss = -torch.minimum(unclipped, clipped).mean()
            if args.supervised_weight > 0.0:
                supervised_loss, _parts = dc.distance_completion_loss(
                    pred,
                    pseudo_batch,
                    edm_weight=args.edm_weight,
                    radio_lower_weight=0.0,
                )
            else:
                supervised_loss = policy_loss.detach() * 0.0
            loss = policy_loss + args.supervised_weight * supervised_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            policy_losses.append(float(policy_loss.detach().cpu()))
            supervised_losses.append(float(supervised_loss.detach().cpu()))
            ratios.extend(float(value) for value in ratio.detach().cpu().tolist())
    return {
        "policy_loss": float(np.mean(policy_losses)) if policy_losses else 0.0,
        "supervised_loss": float(np.mean(supervised_losses)) if supervised_losses else 0.0,
        "ratio_mean": float(np.mean(ratios)) if ratios else 1.0,
        "ratio_max": float(np.max(ratios)) if ratios else 1.0,
    }

@torch.no_grad()
def build_case_batches(
    cases: list[p95.CaseSpec],
    batch_size: int,
    device: torch.device,
) -> list[tuple[dc.GraphBatch, list[p95.CaseSpec]]]:
    batches: list[tuple[dc.GraphBatch, list[p95.CaseSpec]]] = []
    for start in range(0, len(cases), batch_size):
        chunk_cases = cases[start : start + batch_size]
        batches.append((p95.graph_batch_from_cases(chunk_cases, device), chunk_cases))
    return batches


@torch.no_grad()
def evaluate_model(
    label: str,
    model: dc.DistanceCompletionNet,
    case_batches: list[tuple[dc.GraphBatch, list[p95.CaseSpec]]],
    args: argparse.Namespace,
) -> list[dict[str, float | int | str]]:
    model.eval()
    rows: list[dict[str, float | int | str]] = []
    global_case_index = 0
    for batch, chunk_cases in case_batches:
        pred = model(batch.node_features, batch.edge_features, batch.mask, batch.measured_mask, batch.pair_mask)
        for case_index, case in enumerate(chunk_cases):
            max_offset, known_rmse = solve_case_max_offset(
                batch,
                pred,
                case_index,
                branch=args.branch,
                solver_iterations=args.eval_solver_iterations,
                weak_polish_iterations=args.eval_weak_polish_iterations,
                closest_per_anchor=args.closest_predicted_pairs_per_anchor,
                predicted_sigma=args.predicted_sigma,
                predicted_sigma_slope=args.predicted_sigma_slope,
                weak_completion_max_distance=args.weak_completion_max_distance,
                weak_completion_sigma_multiplier=args.weak_completion_sigma_multiplier,
            )
            rows.append(
                {
                    "label": label,
                    "bucket": case.bucket,
                    "bucket_key": BUCKET_KEYS_BY_LABEL.get(case.bucket, case.bucket),
                    "case_index": global_case_index,
                    "family": case.family,
                    "shape": case.shape,
                    "anchors": int(batch.node_counts[case_index]),
                    "missing_mae_m": dc.missing_mae(batch, pred, case_index),
                    "known_rmse_m": known_rmse,
                    "max_offset_m": max_offset,
                }
            )
            global_case_index += 1
    return rows


def summarize_eval(rows: list[dict[str, float | int | str]]) -> list[dict[str, float | int | str]]:
    summary: list[dict[str, float | int | str]] = []
    groups = sorted({(row["label"], row["bucket"]) for row in rows}, key=lambda item: (str(item[0]), str(item[1])))
    for label, bucket in groups:
        part = [row for row in rows if row["label"] == label and row["bucket"] == bucket]
        offsets = np.array([float(row["max_offset_m"]) for row in part], dtype=float)
        rmses = np.array([float(row["known_rmse_m"]) for row in part], dtype=float)
        maes = np.array([float(row["missing_mae_m"]) for row in part], dtype=float)
        summary.append(
            {
                "label": label,
                "bucket": bucket,
                "cases": len(part),
                "median_max_offset_m": float(np.median(offsets)),
                "p90_max_offset_m": float(np.quantile(offsets, 0.90)),
                "p95_max_offset_m": float(np.quantile(offsets, 0.95)),
                "max_offset_m": float(np.max(offsets)),
                "under_0_20m": float(np.mean(offsets <= 0.20)),
                "under_0_50m": float(np.mean(offsets <= 0.50)),
                "under_1m": float(np.mean(offsets <= 1.0)),
                "median_known_rmse_m": float(np.median(rmses)),
                "median_missing_mae_m": float(np.median(maes)),
            }
        )
    return summary


def eval_mean_squared_max_offset(rows: list[dict[str, float | int | str]]) -> float:
    offsets = np.array([float(row["max_offset_m"]) for row in rows], dtype=float)
    return float(np.mean(offsets * offsets)) if offsets.size else math.inf


def eval_bucket_p95(rows: list[dict[str, float | int | str]], bucket: str) -> float:
    offsets = np.array([float(row["max_offset_m"]) for row in rows if row["bucket"] == bucket], dtype=float)
    return float(np.quantile(offsets, 0.95)) if offsets.size else math.inf


def eval_worst_bucket_p95(rows: list[dict[str, float | int | str]]) -> float:
    buckets = sorted({str(row["bucket"]) for row in rows})
    return max((eval_bucket_p95(rows, bucket) for bucket in buckets), default=math.inf)


def write_csv(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    if not rows:
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def save_checkpoint(path: Path, model: dc.DistanceCompletionNet, args: argparse.Namespace, saved_args: dict, history: list[dict[str, float]]) -> None:
    merged_args = dict(saved_args)
    merged_args.update(vars(args))
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "args": merged_args,
            "ppo_history": history,
            "ppo_objective": "negative squared max aligned anchor offset",
            "source_checkpoint": str(args.init_checkpoint),
        },
        path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="PPO-style fine-tune of distance completion model for lowest max coordinate offset.")
    parser.add_argument("--init-checkpoint", type=Path, default=OUTPUTS / "anchor_solver_ml_distance_completion_bigF_bigD_edm08_lr35e5_best_loss.pt")
    parser.add_argument("--prefix", default="anchor_solver_ppo_squared_max_offset")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=2026062717)
    parser.add_argument("--updates", type=int, default=4)
    parser.add_argument("--train-cases-per-bucket", type=int, default=1)
    parser.add_argument("--eval-cases-per-bucket", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=3)
    parser.add_argument("--samples-per-case", type=int, default=4)
    parser.add_argument("--ppo-epochs", type=int, default=2)
    parser.add_argument("--ppo-minibatch", type=int, default=6)
    parser.add_argument("--lr", type=float, default=2.0e-5)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--clip-epsilon", type=float, default=0.20)
    parser.add_argument("--policy-log-std", type=float, default=-2.3)
    parser.add_argument("--supervised-weight", type=float, default=0.04)
    parser.add_argument("--edm-weight", type=float, default=0.02)
    parser.add_argument("--known-rmse-penalty", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--branch", choices=("weak", "completed"), default="weak")
    parser.add_argument("--solver-iterations", type=int, default=45)
    parser.add_argument("--weak-polish-iterations", type=int, default=45)
    parser.add_argument("--eval-solver-iterations", type=int, default=80)
    parser.add_argument("--eval-weak-polish-iterations", type=int, default=80)
    parser.add_argument("--closest-predicted-pairs-per-anchor", type=float, default=5.0)
    parser.add_argument("--predicted-sigma", type=float, default=0.55)
    parser.add_argument("--predicted-sigma-slope", type=float, default=0.65)
    parser.add_argument("--weak-completion-max-distance", type=float, default=18.0)
    parser.add_argument("--weak-completion-sigma-multiplier", type=float, default=2.5)
    parser.add_argument("--eval-every-updates", type=int, default=0)
    args = parser.parse_args()

    OUTPUTS.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    device = choose_device(args.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
    gen_device = torch.device("cpu")
    print(f"device={device} init={args.init_checkpoint}", flush=True)
    model, saved_args = parse_checkpoint(args.init_checkpoint, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    eval_cases = generate_mixed_cases(args.eval_cases_per_bucket, gen_device)
    eval_batches = build_case_batches(eval_cases, args.batch_size, device)
    print(f"eval_cases={len(eval_cases)} train_cases_per_update={args.train_cases_per_bucket * len(BUCKET_KEYS)}", flush=True)
    eval_rows: list[dict[str, float | int | str]] = []
    before_rows = evaluate_model("before", model, eval_batches, args)
    eval_rows.extend(before_rows)
    before_summary = summarize_eval(eval_rows)
    best_checkpoint_path = OUTPUTS / f"{args.prefix}_best.pt"
    best_eval_score = eval_mean_squared_max_offset(before_rows)
    best_update = 0
    save_checkpoint(best_checkpoint_path, model, args, saved_args, [])
    print(f"best_update=0 eval_mean_squared_max_offset={best_eval_score:.4f} path={best_checkpoint_path}", flush=True)
    for row in before_summary:
        print(
            f"before bucket={row['bucket']} p95={float(row['p95_max_offset_m']):.3f}m "
            f"median={float(row['median_max_offset_m']):.3f}m under1={float(row['under_1m']):.2f}",
            flush=True,
        )

    history: list[dict[str, float]] = []
    started = time.perf_counter()
    for update in range(1, args.updates + 1):
        train_cases = generate_mixed_cases(args.train_cases_per_bucket, gen_device)
        rollouts = collect_rollouts(model, train_cases, args, device)
        update_stats = ppo_update(model, optimizer, rollouts, args)
        rewards = np.array([item.reward for item in rollouts], dtype=float)
        offsets = np.array([item.max_offset_m for item in rollouts], dtype=float)
        row = {
            "update": update,
            "rollouts": len(rollouts),
            "mean_reward": float(rewards.mean()),
            "best_reward": float(rewards.max()),
            "median_max_offset_m": float(np.median(offsets)),
            "p90_max_offset_m": float(np.quantile(offsets, 0.90)),
            "best_max_offset_m": float(offsets.min()),
            "policy_loss": update_stats["policy_loss"],
            "supervised_loss": update_stats["supervised_loss"],
            "ratio_mean": update_stats["ratio_mean"],
            "ratio_max": update_stats["ratio_max"],
            "eval_mean_squared_max_offset": "",
            "eval_office_p95_max_offset_m": "",
            "eval_worst_bucket_p95_max_offset_m": "",
            "is_best_eval": 0,
            "elapsed_s": time.perf_counter() - started,
        }
        if args.eval_every_updates > 0 and update % args.eval_every_updates == 0:
            update_label = f"update_{update}"
            update_eval_rows = evaluate_model(update_label, model, eval_batches, args)
            eval_rows.extend(update_eval_rows)
            update_score = eval_mean_squared_max_offset(update_eval_rows)
            office_p95 = eval_bucket_p95(update_eval_rows, "Office >=16")
            worst_p95 = eval_worst_bucket_p95(update_eval_rows)
            row["eval_mean_squared_max_offset"] = update_score
            row["eval_office_p95_max_offset_m"] = office_p95
            row["eval_worst_bucket_p95_max_offset_m"] = worst_p95
            if update_score < best_eval_score:
                best_eval_score = update_score
                best_update = update
                row["is_best_eval"] = 1
                save_checkpoint(best_checkpoint_path, model, args, saved_args, history + [row])
                print(f"checkpoint_best update={update} eval_mean_squared_max_offset={best_eval_score:.4f} path={best_checkpoint_path}", flush=True)
            print(
                f"eval_update={update} mean_sq={update_score:.4f} office_p95={office_p95:.3f}m worst_p95={worst_p95:.3f}m best_update={best_update}",
                flush=True,
            )
        history.append(row)
        print(
            f"ppo_update={update}/{args.updates} rollouts={len(rollouts)} "
            f"median_max={row['median_max_offset_m']:.3f}m best={row['best_max_offset_m']:.3f}m "
            f"reward={row['mean_reward']:.4f} policy_loss={row['policy_loss']:.5f} "
            f"ratio={row['ratio_mean']:.3f}",
            flush=True,
        )

    eval_rows.extend(evaluate_model("after", model, eval_batches, args))
    summary = summarize_eval(eval_rows)
    for row in summary:
        print(
            f"eval label={row['label']} bucket={row['bucket']} p95={float(row['p95_max_offset_m']):.3f}m "
            f"median={float(row['median_max_offset_m']):.3f}m under1={float(row['under_1m']):.2f}",
            flush=True,
        )

    checkpoint_path = OUTPUTS / f"{args.prefix}.pt"
    history_path = OUTPUTS / f"{args.prefix}_history.csv"
    detail_path = OUTPUTS / f"{args.prefix}_eval_detail.csv"
    summary_path = OUTPUTS / f"{args.prefix}_eval_summary.csv"
    save_checkpoint(checkpoint_path, model, args, saved_args, history)
    write_csv(history_path, history)
    write_csv(detail_path, eval_rows)
    write_csv(summary_path, summary)
    print(f"Wrote {checkpoint_path}", flush=True)
    print(f"Wrote {history_path}", flush=True)
    print(f"Wrote {detail_path}", flush=True)
    print(f"Wrote {summary_path}", flush=True)


if __name__ == "__main__":
    main()











