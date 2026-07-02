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
import anchor_solver_p95_ml_cases as p95
from anchor_solver_weighted_output import WeightedDistanceCompletionNet, weighted_model_from_teacher_checkpoint


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


def load_teacher(path: Path, device: torch.device) -> tuple[dc.DistanceCompletionNet, dict]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    probe = dc.make_graph_batch(2, device=device, random_fraction=0.5)
    saved_args = checkpoint.get("args", {})
    teacher = dc.DistanceCompletionNet(
        probe.node_features.shape[-1],
        probe.edge_features.shape[-1],
        hidden=int(saved_args.get("hidden", 144)),
        layers=int(saved_args.get("layers", 5)),
        dropout=float(saved_args.get("dropout", 0.0)),
    ).to(device)
    teacher.load_state_dict(checkpoint["model_state_dict"])
    teacher.eval()
    return teacher, checkpoint


def distill_loss(
    student_pred: torch.Tensor,
    student_weight: torch.Tensor,
    teacher_pred: torch.Tensor,
    batch: dc.GraphBatch,
    *,
    distance_weight: float,
    missing_distance_weight: float,
    weight_loss_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    pair_mask = batch.pair_mask
    missing = pair_mask & ~batch.measured_mask
    distance_all = F.smooth_l1_loss(student_pred[pair_mask], teacher_pred[pair_mask], beta=0.02)
    if bool(missing.any()):
        distance_missing = F.smooth_l1_loss(student_pred[missing], teacher_pred[missing], beta=0.02)
        log_weight = torch.log(student_weight[missing].clamp_min(1e-6))
        weight_loss = F.smooth_l1_loss(log_weight, torch.zeros_like(log_weight), beta=0.02)
        weight_abs = log_weight.abs().mean()
    else:
        distance_missing = distance_all * 0.0
        weight_loss = distance_all * 0.0
        weight_abs = distance_all.detach() * 0.0
    loss = distance_weight * distance_all + missing_distance_weight * distance_missing + weight_loss_weight * weight_loss
    with torch.no_grad():
        diff = (student_pred - teacher_pred).abs()
        missing_diff = diff[missing] if bool(missing.any()) else diff[pair_mask]
        weights = student_weight[missing] if bool(missing.any()) else student_weight[pair_mask]
        parts = {
            "loss": float(loss.detach().cpu()),
            "distance_all": float(distance_all.detach().cpu()),
            "distance_missing": float(distance_missing.detach().cpu()),
            "weight_loss": float(weight_loss.detach().cpu()),
            "mean_abs_log_weight": float(weight_abs.detach().cpu()),
            "missing_mean_abs_norm": float(missing_diff.mean().detach().cpu()) if missing_diff.numel() else 0.0,
            "missing_max_abs_norm": float(missing_diff.max().detach().cpu()) if missing_diff.numel() else 0.0,
            "weight_median": float(weights.median().detach().cpu()) if weights.numel() else 1.0,
            "weight_p05": float(torch.quantile(weights.detach().float(), 0.05).cpu()) if weights.numel() else 1.0,
            "weight_p95": float(torch.quantile(weights.detach().float(), 0.95).cpu()) if weights.numel() else 1.0,
        }
    return loss, parts


@torch.no_grad()
def evaluate_distill(
    teacher: dc.DistanceCompletionNet,
    student: WeightedDistanceCompletionNet,
    *,
    batches: int,
    batch_size: int,
    random_fraction: float,
    device: torch.device,
    args: argparse.Namespace,
) -> dict[str, float]:
    teacher.eval()
    student.eval()
    accum: dict[str, list[float]] = {}
    for _ in range(batches):
        batch = dc.make_graph_batch(batch_size, device=device, random_fraction=random_fraction)
        teacher_pred = teacher(batch.node_features, batch.edge_features, batch.mask, batch.measured_mask, batch.pair_mask)
        student_pred, student_weight = student(batch.node_features, batch.edge_features, batch.mask, batch.measured_mask, batch.pair_mask)
        _loss, parts = distill_loss(
            student_pred,
            student_weight,
            teacher_pred,
            batch,
            distance_weight=args.distance_weight,
            missing_distance_weight=args.missing_distance_weight,
            weight_loss_weight=args.weight_loss_weight,
        )
        for key, value in parts.items():
            accum.setdefault(key, []).append(value)
    return {key: float(np.mean(values)) for key, values in accum.items()}


def save_checkpoint(
    path: Path,
    student: WeightedDistanceCompletionNet,
    args: argparse.Namespace,
    teacher_checkpoint: dict,
    history: list[dict[str, float]],
    *,
    step: int,
    val_loss: float,
) -> None:
    saved_args = dict(teacher_checkpoint.get("args", {}))
    saved_args.update(vars(args))
    saved_args["weighted_output"] = True
    saved_args["spring_weight_is_multiplier"] = True
    saved_args["distilled_from_teacher_distance_output"] = True
    torch.save(
        {
            "model_state_dict": student.state_dict(),
            "args": saved_args,
            "teacher_checkpoint": str(args.teacher_checkpoint),
            "weighted_output": True,
            "spring_weight_is_multiplier": True,
            "spring_weight_target": 1.0,
            "history": history,
            "step": step,
            "val_loss": val_loss,
        },
        path,
    )


def write_history(path: Path, history: list[dict[str, float]]) -> None:
    if not history:
        return
    fields = sorted({key for row in history for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(history)


def main() -> None:
    parser = argparse.ArgumentParser(description="Distill a weighted-output GNN from an existing distance-only GNN.")
    parser.add_argument("--teacher-checkpoint", type=Path, default=OUTPUTS / "anchor_solver_ppo_gpu_full_continue_best.pt")
    parser.add_argument("--prefix", default="anchor_solver_weighted_distill_neutral")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=2026062721)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--random-fraction", type=float, default=0.45)
    parser.add_argument("--lr", type=float, default=3.0e-5)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--distance-weight", type=float, default=0.35)
    parser.add_argument("--missing-distance-weight", type=float, default=1.0)
    parser.add_argument("--weight-loss-weight", type=float, default=0.50)
    parser.add_argument("--max-abs-log-weight", type=float, default=2.0)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--eval-batches", type=int, default=3)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--train-all", action="store_true", help="Also train copied backbone/distance parameters. Default freezes them to preserve teacher outputs exactly.")
    args = parser.parse_args()

    OUTPUTS.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    device = choose_device(args.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
    print(f"device={device} teacher={args.teacher_checkpoint}", flush=True)
    teacher, teacher_checkpoint = load_teacher(args.teacher_checkpoint, device)
    probe = dc.make_graph_batch(2, device=device, random_fraction=args.random_fraction)
    student = weighted_model_from_teacher_checkpoint(
        teacher_checkpoint,
        node_feature_count=probe.node_features.shape[-1],
        edge_feature_count=probe.edge_features.shape[-1],
        device=device,
        max_abs_log_weight=args.max_abs_log_weight,
    )
    if not args.train_all:
        for name, parameter in student.named_parameters():
            parameter.requires_grad_(name.startswith("weight_head."))
        print("trainable=weight_head_only copied_distance_path=frozen", flush=True)
    else:
        print("trainable=all_parameters", flush=True)
    optimizer = torch.optim.AdamW([p for p in student.parameters() if p.requires_grad], lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler(device.type, enabled=args.amp and device.type == "cuda")
    history: list[dict[str, float]] = []
    best_loss = math.inf
    best_path = OUTPUTS / f"{args.prefix}_best.pt"
    latest_path = OUTPUTS / f"{args.prefix}.pt"
    history_path = OUTPUTS / f"{args.prefix}_history.csv"
    started = time.perf_counter()

    initial = evaluate_distill(
        teacher,
        student,
        batches=args.eval_batches,
        batch_size=max(8, min(args.batch_size, 64)),
        random_fraction=args.random_fraction,
        device=device,
        args=args,
    )
    initial_row = {"step": 0, "elapsed_s": 0.0, **{f"val_{k}": v for k, v in initial.items()}}
    history.append(initial_row)
    best_loss = initial["loss"]
    save_checkpoint(best_path, student, args, teacher_checkpoint, history, step=0, val_loss=best_loss)
    print(
        f"initial val_loss={initial['loss']:.7f} missing_abs_norm={initial['missing_mean_abs_norm']:.7f} "
        f"weight_median={initial['weight_median']:.4f} p05={initial['weight_p05']:.4f} p95={initial['weight_p95']:.4f}",
        flush=True,
    )

    teacher.eval()
    student.train()
    for step in range(1, args.steps + 1):
        batch = dc.make_graph_batch(args.batch_size, device=device, random_fraction=args.random_fraction)
        with torch.no_grad():
            teacher_pred = teacher(batch.node_features, batch.edge_features, batch.mask, batch.measured_mask, batch.pair_mask)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=args.amp and device.type == "cuda"):
            student_pred, student_weight = student(batch.node_features, batch.edge_features, batch.mask, batch.measured_mask, batch.pair_mask)
            loss, parts = distill_loss(
                student_pred,
                student_weight,
                teacher_pred,
                batch,
                distance_weight=args.distance_weight,
                missing_distance_weight=args.missing_distance_weight,
                weight_loss_weight=args.weight_loss_weight,
            )
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(student.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        if step == 1 or step % args.eval_every == 0 or step == args.steps:
            val = evaluate_distill(
                teacher,
                student,
                batches=args.eval_batches,
                batch_size=max(8, min(args.batch_size, 64)),
                random_fraction=args.random_fraction,
                device=device,
                args=args,
            )
            row = {
                "step": step,
                "elapsed_s": time.perf_counter() - started,
                **{f"train_{k}": v for k, v in parts.items()},
                **{f"val_{k}": v for k, v in val.items()},
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
                f"val_missing_abs_norm={val['missing_mean_abs_norm']:.7f} val_weight_med={val['weight_median']:.4f} "
                f"val_w_p05={val['weight_p05']:.4f} val_w_p95={val['weight_p95']:.4f}{best_text}",
                flush=True,
            )

    save_checkpoint(latest_path, student, args, teacher_checkpoint, history, step=args.steps, val_loss=history[-1].get("val_loss", best_loss))
    write_history(history_path, history)
    print(f"Wrote {latest_path}", flush=True)
    print(f"Wrote {best_path}", flush=True)
    print(f"Wrote {history_path}", flush=True)


if __name__ == "__main__":
    main()

