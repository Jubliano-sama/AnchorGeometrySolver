from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
sys.path.insert(0, str(ROOT / "work"))

import anchor_solver_ml_distance_completion as dc
from anchor_solver_weighted_output import WeightedDistanceCompletionNet, load_expanded_weighted_state_dict


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


def build_model_from_checkpoint(path: Path, device: torch.device, *, graph_features: bool | None = None) -> tuple[WeightedDistanceCompletionNet, dict, bool]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    saved_args = checkpoint.get("args", {})
    checkpoint_graph_features = bool(saved_args.get("graph_solution_features", saved_args.get("use_graph_solution_features", False)))
    use_graph_features = checkpoint_graph_features if graph_features is None else graph_features
    probe = dc.make_graph_batch(2, device=device, random_fraction=0.5, min_vertex_connectivity=1, graph_solution_features=use_graph_features)
    model = WeightedDistanceCompletionNet(
        probe.node_features.shape[-1],
        probe.edge_features.shape[-1],
        hidden=int(saved_args.get("hidden", 144)),
        layers=int(saved_args.get("layers", 5)),
        dropout=float(saved_args.get("dropout", 0.0)),
        max_abs_log_weight=float(saved_args.get("max_abs_log_weight", 2.0)),
    ).to(device)
    return model, checkpoint, use_graph_features


def student_batch_from_base(base: dc.GraphBatch, use_graph_features: bool) -> dc.GraphBatch:
    return dc.append_graph_solution_edge_features(base) if use_graph_features else base


def distill_loss(
    student_pred: torch.Tensor,
    student_weight: torch.Tensor,
    teacher_pred: torch.Tensor,
    teacher_weight: torch.Tensor,
    base_batch: dc.GraphBatch,
    *,
    distance_weight: float,
    missing_distance_weight: float,
    weight_weight: float,
    true_distance_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    pair_mask = base_batch.pair_mask
    missing = pair_mask & ~base_batch.measured_mask
    target_true = base_batch.true_dist_m / base_batch.scale_m.view(-1, 1, 1)
    distance_all = F.smooth_l1_loss(student_pred[pair_mask], teacher_pred[pair_mask], beta=0.02)
    distance_missing = F.smooth_l1_loss(student_pred[missing], teacher_pred[missing], beta=0.02) if bool(missing.any()) else distance_all * 0.0
    true_missing = F.smooth_l1_loss(student_pred[missing], target_true[missing], beta=0.04) if bool(missing.any()) else distance_all * 0.0
    log_student_weight = torch.log(student_weight[pair_mask].clamp_min(1e-6))
    log_teacher_weight = torch.log(teacher_weight[pair_mask].clamp_min(1e-6))
    weight_loss = F.smooth_l1_loss(log_student_weight, log_teacher_weight, beta=0.02)
    loss = distance_weight * distance_all + missing_distance_weight * distance_missing + weight_weight * weight_loss + true_distance_weight * true_missing
    with torch.no_grad():
        weights = student_weight[pair_mask].detach().float()
        teacher_weights = teacher_weight[pair_mask].detach().float()
        missing_abs = (student_pred - teacher_pred).abs()[missing].detach().float() if bool(missing.any()) else (student_pred - teacher_pred).abs()[pair_mask].detach().float()
        parts = {
            "loss": float(loss.detach().cpu()),
            "distance_all": float(distance_all.detach().cpu()),
            "distance_missing": float(distance_missing.detach().cpu()),
            "true_missing": float(true_missing.detach().cpu()),
            "weight_loss": float(weight_loss.detach().cpu()),
            "missing_mean_abs_norm": float(missing_abs.mean().cpu()) if missing_abs.numel() else 0.0,
            "missing_p95_abs_norm": float(torch.quantile(missing_abs, 0.95).cpu()) if missing_abs.numel() else 0.0,
            "weight_p05": float(torch.quantile(weights, 0.05).cpu()) if weights.numel() else 1.0,
            "weight_p50": float(torch.quantile(weights, 0.50).cpu()) if weights.numel() else 1.0,
            "weight_p95": float(torch.quantile(weights, 0.95).cpu()) if weights.numel() else 1.0,
            "teacher_weight_p05": float(torch.quantile(teacher_weights, 0.05).cpu()) if teacher_weights.numel() else 1.0,
            "teacher_weight_p50": float(torch.quantile(teacher_weights, 0.50).cpu()) if teacher_weights.numel() else 1.0,
            "teacher_weight_p95": float(torch.quantile(teacher_weights, 0.95).cpu()) if teacher_weights.numel() else 1.0,
        }
    return loss, parts


def make_base_batch(args: argparse.Namespace, device: torch.device, batch_size: int) -> dc.GraphBatch:
    return dc.make_graph_batch(batch_size, device=device, random_fraction=args.random_fraction, min_vertex_connectivity=args.min_vertex_connectivity, graph_solution_features=False)


@torch.no_grad()
def evaluate(
    teacher: WeightedDistanceCompletionNet,
    student: WeightedDistanceCompletionNet,
    *,
    teacher_graph_features: bool,
    student_graph_features: bool,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, float]:
    teacher.eval()
    student.eval()
    accum: dict[str, list[float]] = {}
    for _ in range(args.eval_batches):
        base = make_base_batch(args, device, max(4, min(args.batch_size, 48)))
        teacher_batch = student_batch_from_base(base, teacher_graph_features)
        student_batch = student_batch_from_base(base, student_graph_features)
        teacher_pred, teacher_weight = teacher(teacher_batch.node_features, teacher_batch.edge_features, teacher_batch.mask, teacher_batch.measured_mask, teacher_batch.pair_mask)
        student_pred, student_weight = student(student_batch.node_features, student_batch.edge_features, student_batch.mask, student_batch.measured_mask, student_batch.pair_mask)
        _loss, parts = distill_loss(
            student_pred,
            student_weight,
            teacher_pred,
            teacher_weight,
            base,
            distance_weight=args.distance_weight,
            missing_distance_weight=args.missing_distance_weight,
            weight_weight=args.weight_weight,
            true_distance_weight=args.true_distance_weight,
        )
        for key, value in parts.items():
            accum.setdefault(key, []).append(value)
    return {key: float(np.mean(values)) for key, values in accum.items()}


def save_checkpoint(path: Path, student: WeightedDistanceCompletionNet, args: argparse.Namespace, teacher_checkpoint: dict, history: list[dict[str, float]], *, step: int, val_loss: float) -> None:
    saved_args = dict(teacher_checkpoint.get("args", {}))
    saved_args.update(vars(args))
    saved_args["weighted_output"] = True
    saved_args["spring_weight_is_multiplier"] = True
    saved_args["distilled_weighted_graph_features"] = True
    saved_args["graph_solution_features"] = bool(args.student_graph_solution_features)
    saved_args["use_graph_solution_features"] = bool(args.student_graph_solution_features)
    torch.save(
        {
            "model_state_dict": student.state_dict(),
            "args": saved_args,
            "teacher_checkpoint": str(args.teacher_checkpoint),
            "weighted_output": True,
            "spring_weight_is_multiplier": True,
            "distilled_weighted_graph_features": True,
            "graph_solution_features": bool(args.student_graph_solution_features),
            "history": history,
            "step": step,
            "val_loss": val_loss,
        },
        path,
    )


def write_history(path: Path, history: list[dict[str, float]]) -> None:
    if not history:
        return
    fields = sorted({key for row in history for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fields)
        writer.writeheader()
        writer.writerows(history)


def main() -> None:
    parser = argparse.ArgumentParser(description="Distill a weighted PPO checkpoint into graph-feature weighted model.")
    parser.add_argument("--teacher-checkpoint", type=Path, default=OUTPUTS / "anchor_solver_weighted_ppo_strong_known_weight_long_20260627_1930_best.pt")
    parser.add_argument("--prefix", default="anchor_solver_weighted_graph_feature_distill")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=2026062741)
    parser.add_argument("--steps", type=int, default=160)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--random-fraction", type=float, default=0.45)
    parser.add_argument("--min-vertex-connectivity", type=int, default=3)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--distance-weight", type=float, default=0.35)
    parser.add_argument("--missing-distance-weight", type=float, default=1.0)
    parser.add_argument("--weight-weight", type=float, default=0.75)
    parser.add_argument("--true-distance-weight", type=float, default=0.0)
    parser.add_argument("--max-abs-log-weight", type=float, default=2.0)
    parser.add_argument("--eval-every", type=int, default=20)
    parser.add_argument("--eval-batches", type=int, default=2)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--student-graph-solution-features", action="store_true", default=True)
    args = parser.parse_args()

    OUTPUTS.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    device = choose_device(args.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
    print(f"device={device} teacher={args.teacher_checkpoint} graph_features={args.student_graph_solution_features}", flush=True)

    teacher, teacher_checkpoint, teacher_graph_features = build_model_from_checkpoint(args.teacher_checkpoint, device, graph_features=None)
    teacher.load_state_dict(teacher_checkpoint["model_state_dict"])
    teacher.eval()

    student, _unused, student_graph_features = build_model_from_checkpoint(args.teacher_checkpoint, device, graph_features=args.student_graph_solution_features)
    student.max_abs_log_weight = float(args.max_abs_log_weight)
    load_info = load_expanded_weighted_state_dict(student, teacher_checkpoint["model_state_dict"])
    print(f"expanded_load copied={load_info['copied']} expanded={len(load_info['expanded'])} missing={len(load_info['missing'])} skipped={len(load_info['skipped'])}", flush=True)

    optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler(device.type, enabled=args.amp and device.type == "cuda")
    history: list[dict[str, float]] = []
    best_loss = math.inf
    best_path = OUTPUTS / f"{args.prefix}_best.pt"
    latest_path = OUTPUTS / f"{args.prefix}.pt"
    history_path = OUTPUTS / f"{args.prefix}_history.csv"
    started = time.perf_counter()

    initial = evaluate(teacher, student, teacher_graph_features=teacher_graph_features, student_graph_features=student_graph_features, args=args, device=device)
    best_loss = initial["loss"]
    history.append({"step": 0, "elapsed_s": 0.0, **{f"val_{key}": value for key, value in initial.items()}})
    save_checkpoint(best_path, student, args, teacher_checkpoint, history, step=0, val_loss=best_loss)
    print(
        f"initial val_loss={initial['loss']:.7f} miss_abs={initial['missing_mean_abs_norm']:.7f} "
        f"w_p05={initial['weight_p05']:.3f} w_p50={initial['weight_p50']:.3f} w_p95={initial['weight_p95']:.3f}",
        flush=True,
    )

    teacher.eval()
    student.train()
    for step in range(1, args.steps + 1):
        base = make_base_batch(args, device, args.batch_size)
        teacher_batch = student_batch_from_base(base, teacher_graph_features)
        student_batch = student_batch_from_base(base, student_graph_features)
        with torch.no_grad():
            teacher_pred, teacher_weight = teacher(teacher_batch.node_features, teacher_batch.edge_features, teacher_batch.mask, teacher_batch.measured_mask, teacher_batch.pair_mask)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=args.amp and device.type == "cuda"):
            student_pred, student_weight = student(student_batch.node_features, student_batch.edge_features, student_batch.mask, student_batch.measured_mask, student_batch.pair_mask)
            loss, parts = distill_loss(
                student_pred,
                student_weight,
                teacher_pred,
                teacher_weight,
                base,
                distance_weight=args.distance_weight,
                missing_distance_weight=args.missing_distance_weight,
                weight_weight=args.weight_weight,
                true_distance_weight=args.true_distance_weight,
            )
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(student.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        if step == 1 or step % args.eval_every == 0 or step == args.steps:
            val = evaluate(teacher, student, teacher_graph_features=teacher_graph_features, student_graph_features=student_graph_features, args=args, device=device)
            row = {
                "step": step,
                "elapsed_s": time.perf_counter() - started,
                **{f"train_{key}": value for key, value in parts.items()},
                **{f"val_{key}": value for key, value in val.items()},
            }
            history.append(row)
            if val["loss"] <= best_loss:
                best_loss = val["loss"]
                save_checkpoint(best_path, student, args, teacher_checkpoint, history, step=step, val_loss=best_loss)
                best_text = " best"
            else:
                best_text = ""
            print(
                f"step={step}/{args.steps} train_loss={parts['loss']:.7f} val_loss={val['loss']:.7f} "
                f"miss_abs={val['missing_mean_abs_norm']:.7f} w_p05={val['weight_p05']:.3f} "
                f"w_p50={val['weight_p50']:.3f} w_p95={val['weight_p95']:.3f}{best_text}",
                flush=True,
            )

    save_checkpoint(latest_path, student, args, teacher_checkpoint, history, step=args.steps, val_loss=history[-1].get("val_loss", best_loss))
    write_history(history_path, history)
    print(f"Wrote {latest_path}", flush=True)
    print(f"Wrote {best_path}", flush=True)
    print(f"Wrote {history_path}", flush=True)


if __name__ == "__main__":
    main()