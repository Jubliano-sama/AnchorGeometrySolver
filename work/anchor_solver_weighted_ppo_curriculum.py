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

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
sys.path.insert(0, str(ROOT / "work"))
sys.path.insert(0, str(ROOT / "work" / "SmartClicker-GUI"))

import anchor_solver_ml_distance_completion as dc
import anchor_solver_p95_ml_cases as p95
import anchor_solver_weighted_ppo_strong as ppo
from anchor_solver_weighted_output import WeightedDistanceCompletionNet

BUCKET_KEYS = ("random", "grid", "office")
OFFICE_SHAPES = ("corridor", "l_shape", "t_shape", "u_shape", "cross", "hollow_square", "rooms", "disc", "annulus")


def stage_min_nodes(max_nodes: int, frontier_window: int = 3) -> int:
    return max(dc.MIN_NODES, min(max_nodes, max_nodes - max(frontier_window, 0)))


def parse_stage_list(text: str) -> list[int]:
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def node_range(max_nodes: int, frontier_window: int = 3) -> tuple[int, int]:
    return stage_min_nodes(max_nodes, frontier_window), max_nodes


def fair_case_points(
    points: torch.Tensor | None,
    *,
    min_nodes: int,
    max_nodes: int,
    min_vertex_connectivity: int,
) -> torch.Tensor | None:
    if points is None:
        return None
    points = dc.random_rigid_transform(points)
    points = dc.prune_to_fair_measured_graph(points, min_vertex_connectivity=min_vertex_connectivity)
    if points is None:
        return None
    n = int(points.shape[0])
    if n < min_nodes or n > max_nodes:
        return None
    return points


def compact_rectangle_params(max_nodes: int) -> dict[str, float]:
    extent = max(8.0, 3.1 * math.sqrt(max_nodes) + 4.0)
    width = random.uniform(8.0, min(dc.MAX_EXTENT_M, extent * random.uniform(1.00, 1.45)))
    height = random.uniform(8.0, min(dc.MAX_EXTENT_M, extent * random.uniform(0.90, 1.35)))
    return {"width": width, "height": height}


def compact_office_shape(max_nodes: int) -> tuple[str, dict[str, float]]:
    shape = random.choice(OFFICE_SHAPES)
    base = max(8.0, 3.4 * math.sqrt(max_nodes) + 4.0)
    if shape == "corridor":
        long_side = random.uniform(max(10.0, base * 1.1), min(dc.MAX_EXTENT_M, base * 1.8))
        short_side = random.uniform(8.0, max(8.1, min(16.0, base * 0.75)))
        width, height = (long_side, short_side) if random.random() < 0.5 else (short_side, long_side)
    elif shape in {"disc", "annulus", "hollow_square"}:
        side = random.uniform(8.0, min(dc.MAX_EXTENT_M, base * 1.35))
        width, height = side, random.uniform(max(8.0, side * 0.86), min(dc.MAX_EXTENT_M, side * 1.12))
    else:
        width = random.uniform(8.0, min(dc.MAX_EXTENT_M, base * 1.45))
        height = random.uniform(8.0, min(dc.MAX_EXTENT_M, base * 1.45))
    params: dict[str, float] = {"width": width, "height": height}
    if shape == "l_shape":
        params["leg_x"] = width * random.uniform(0.38, 0.62)
        params["leg_y"] = height * random.uniform(0.38, 0.62)
    elif shape == "t_shape":
        params["bar_w"] = width * random.uniform(0.28, 0.48)
        params["top_h"] = height * random.uniform(0.30, 0.48)
    elif shape == "u_shape":
        params["leg_w"] = width * random.uniform(0.26, 0.40)
        params["bottom_h"] = height * random.uniform(0.28, 0.46)
    elif shape == "cross":
        params["bar_w"] = width * random.uniform(0.26, 0.44)
        params["bar_h"] = height * random.uniform(0.26, 0.44)
    elif shape == "hollow_square":
        params["hole_w"] = width * random.uniform(0.24, 0.45)
        params["hole_h"] = height * random.uniform(0.24, 0.45)
    elif shape == "disc":
        side = min(width, height)
        params["width"] = side
        params["height"] = side
        params["radius"] = side * 0.50
    elif shape == "annulus":
        side = min(width, height)
        params["width"] = side
        params["height"] = side
        params["outer_radius"] = side * 0.50
        params["inner_radius"] = side * random.uniform(0.16, 0.30)
    elif shape == "rooms":
        params["corridor_w"] = width * random.uniform(0.18, 0.32)
        params["corridor_h"] = height * random.uniform(0.18, 0.32)
    return shape, params


def generate_curriculum_case(bucket: str, max_nodes: int, device: torch.device, *, min_vertex_connectivity: int, frontier_window: int) -> p95.CaseSpec:
    min_nodes, _ = node_range(max_nodes, frontier_window)
    label = {"random": "Random", "grid": "Grid", "office": "Office"}[bucket] + f" <= {max_nodes}"
    for attempt in range(30000):
        target_n = random.randint(min_nodes, max_nodes)
        easy_fallback = attempt >= 18000
        relaxed_min_nodes = dc.MIN_NODES if attempt >= 24000 else min_nodes
        relaxed_vertex = 1 if attempt >= 24000 else min_vertex_connectivity
        if bucket == "random":
            shape = "rectangle"
            params = compact_rectangle_params(max_nodes)
            points = dc.grid_points_in_shape(shape, params, target_n, device) if easy_fallback else dc.random_points_in_shape(shape, params, target_n, device)
            family = "grid-fallback" if easy_fallback else "random"
        elif bucket == "grid":
            shape = "rectangle"
            params = compact_rectangle_params(max_nodes)
            points = dc.grid_points_in_shape(shape, params, target_n, device)
            family = "grid"
        else:
            if easy_fallback:
                shape = random.choice(("corridor", "rectangle"))
                if shape == "rectangle":
                    params = compact_rectangle_params(max_nodes)
                else:
                    long_side = random.uniform(max(12.0, 3.7 * math.sqrt(max_nodes) + 6.0), min(dc.MAX_EXTENT_M, 5.2 * math.sqrt(max_nodes) + 12.0))
                    short_side = random.uniform(8.0, min(16.0, max(8.1, 2.7 * math.sqrt(max_nodes) + 4.0)))
                    params = {"width": long_side, "height": short_side} if random.random() < 0.5 else {"width": short_side, "height": long_side}
                points = dc.grid_points_in_shape(shape, params, target_n, device)
                family = "grid-office-fallback"
            else:
                shape, params = compact_office_shape(max_nodes)
                if random.random() < 0.68:
                    points = dc.grid_points_in_shape(shape, params, target_n, device)
                    family = "grid-office"
                else:
                    points = dc.random_points_in_shape(shape, params, target_n, device)
                    family = "random-office"
        points = fair_case_points(
            points,
            min_nodes=relaxed_min_nodes,
            max_nodes=max_nodes,
            min_vertex_connectivity=relaxed_vertex,
        )
        if points is not None:
            return p95.CaseSpec(label, family, shape, points)
    raise RuntimeError(f"Could not generate fair {bucket} curriculum case <= {max_nodes} anchors after fallback attempts.")


def make_curriculum_cases(cases_per_bucket: int, max_nodes: int, device: torch.device, args: argparse.Namespace) -> list[p95.CaseSpec]:
    cases: list[p95.CaseSpec] = []
    for bucket in BUCKET_KEYS:
        while len([case for case in cases if case.bucket.startswith(bucket.capitalize())]) < cases_per_bucket:
            cases.append(
                generate_curriculum_case(
                    bucket,
                    max_nodes,
                    device,
                    min_vertex_connectivity=args.curriculum_min_vertex_connectivity,
                    frontier_window=args.curriculum_frontier_window,
                )
            )
    random.shuffle(cases)
    return cases


def write_csv(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    if not rows:
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fields)
        writer.writeheader()
        writer.writerows(rows)


def add_stage_fields(rows: list[dict[str, float | int | str]], *, stage_index: int, stage_max: int, global_update: int, label: str) -> None:
    for row in rows:
        row["stage_index"] = stage_index
        row["stage_max_anchors"] = stage_max
        row["global_update"] = global_update
        row["label"] = label


def score_rows(rows: list[dict[str, float | int | str]]) -> float:
    return float(np.mean([float(row["max_offset_m"]) ** 2 for row in rows]))


def summary_text(rows: list[dict[str, float | int | str]]) -> str:
    parts = []
    for row in ppo.summarize(rows):
        parts.append(
            f"{row['bucket']}:p95={float(row['p95_max_offset_m']):.3f} med={float(row['median_max_offset_m']):.3f} under1={float(row['under_1m']):.2f}"
        )
    return " | ".join(parts)


def save_checkpoint(path: Path, model: WeightedDistanceCompletionNet, args: argparse.Namespace, source: dict, history: list[dict[str, float | int | str]], *, stage_index: int, stage_max: int, best_score: float) -> None:
    ppo.save(path, model, args, source, history)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    checkpoint["curriculum"] = {"stage_index": stage_index, "stage_max_anchors": stage_max, "best_score": best_score}
    torch.save(checkpoint, path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Curriculum PPO for weighted-output distance-completion policy.")
    parser.add_argument("--init-checkpoint", type=Path, default=OUTPUTS / "anchor_solver_weighted_graph_feature_distill_smoke_best.pt")
    parser.add_argument("--prefix", default="anchor_solver_weighted_ppo_curriculum")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=2026062801)
    parser.add_argument("--max-runtime-minutes", type=float, default=600.0)
    parser.add_argument("--train-cases-per-bucket", type=int, default=8)
    parser.add_argument("--eval-cases-per-bucket", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--samples-per-case", type=int, default=4)
    parser.add_argument("--ppo-epochs", type=int, default=2)
    parser.add_argument("--ppo-minibatch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1.5e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--clip-epsilon", type=float, default=0.2)
    parser.add_argument("--distance-log-std", type=float, default=-1.7)
    parser.add_argument("--weight-log-std", type=float, default=-0.5)
    parser.add_argument("--perturb-known-distances", action="store_true")
    parser.add_argument("--known-distance-std-m", type=float, default=0.08)
    parser.add_argument("--weight-known-springs", action="store_true")
    parser.add_argument("--supervised-weight", type=float, default=0.025)
    parser.add_argument("--weight-reg", type=float, default=0.01)
    parser.add_argument("--edm-weight", type=float, default=0.02)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--solver-iterations", type=int, default=50)
    parser.add_argument("--polish-iterations", type=int, default=50)
    parser.add_argument("--rollout-solver-iterations", type=int, default=28)
    parser.add_argument("--rollout-polish-iterations", type=int, default=14)
    parser.add_argument("--eval-solver-iterations", type=int, default=50)
    parser.add_argument("--eval-polish-iterations", type=int, default=50)
    parser.add_argument("--solve-threads", type=int, default=6)
    parser.add_argument("--closest-predicted-pairs-per-anchor", type=float, default=5.0)
    parser.add_argument("--predicted-sigma", type=float, default=0.55)
    parser.add_argument("--predicted-sigma-slope", type=float, default=0.65)
    parser.add_argument("--graph-solution-features", action="store_true")
    parser.add_argument("--fold-threshold-m", type=float, default=1.65)
    parser.add_argument("--curriculum-stages", default="12,16,20,24,28,32,36,40,44,48,50")
    parser.add_argument("--curriculum-min-updates", type=int, default=80)
    parser.add_argument("--curriculum-patience-evals", type=int, default=10)
    parser.add_argument("--curriculum-min-delta", type=float, default=0.001)
    parser.add_argument("--curriculum-min-relative-delta", type=float, default=0.02)
    parser.add_argument("--curriculum-min-stage-minutes", type=float, default=45.0)
    parser.add_argument("--curriculum-frontier-window", type=int, default=3)
    parser.add_argument("--curriculum-min-vertex-connectivity", type=int, default=1)
    parser.add_argument("--eval-every-updates", type=int, default=10)
    parser.add_argument("--verbose-first-minutes", type=float, default=10.0)
    parser.add_argument("--verbose-eval-every-updates", type=int, default=2)
    parser.add_argument("--progress-every-minutes-late", type=float, default=30.0)
    parser.add_argument("--progress-every-minutes", type=float, default=2.0)
    args = parser.parse_args()
    args.rollout_log = False

    OUTPUTS.mkdir(exist_ok=True)
    ppo.set_seed(args.seed)
    device = ppo.choose_device(args.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
    print(f"device={device} init={args.init_checkpoint} curriculum_stages={args.curriculum_stages}", flush=True)

    model, source = ppo.load_weighted_model(args.init_checkpoint, device)
    source_args = source.get("args", {})
    if source_args.get("graph_solution_features", source_args.get("use_graph_solution_features", False)):
        args.graph_solution_features = True
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    stages = parse_stage_list(args.curriculum_stages)
    gen_device = torch.device("cpu")
    history: list[dict[str, float | int | str]] = []
    eval_rows: list[dict[str, float | int | str]] = []
    best_score = math.inf
    best_path = OUTPUTS / f"{args.prefix}_best.pt"
    latest_path = OUTPUTS / f"{args.prefix}_latest.pt"
    history_path = OUTPUTS / f"{args.prefix}_history.csv"
    detail_path = OUTPUTS / f"{args.prefix}_eval_detail.csv"
    summary_path = OUTPUTS / f"{args.prefix}_eval_summary.csv"
    stage_path = OUTPUTS / f"{args.prefix}_stage_summary.csv"
    stage_rows: list[dict[str, float | int | str]] = []

    started = time.perf_counter()
    last_progress = started
    global_update = 0
    stop = False

    for stage_index, stage_max in enumerate(stages):
        stage_started = time.perf_counter()
        stage_update = 0
        stage_eval_count = 0
        stage_best = math.inf
        last_best_eval = 0
        last_eval_score = math.inf
        min_nodes, max_nodes = node_range(stage_max, args.curriculum_frontier_window)
        print(f"stage_start index={stage_index} max_anchors={stage_max} node_range={min_nodes}-{max_nodes}", flush=True)
        eval_cases = make_curriculum_cases(args.eval_cases_per_bucket, stage_max, gen_device, args)
        node_counts = [int(case.points.shape[0]) for case in eval_cases]
        print(f"stage_eval_cases max_anchors={stage_max} cases={len(eval_cases)} min_n={min(node_counts)} median_n={np.median(node_counts):.0f} max_n={max(node_counts)}", flush=True)

        while True:
            if args.max_runtime_minutes > 0 and (time.perf_counter() - started) >= args.max_runtime_minutes * 60.0:
                stop = True
                break
            global_update += 1
            stage_update += 1
            train_cases = make_curriculum_cases(args.train_cases_per_bucket, stage_max, gen_device, args)
            rollouts = ppo.collect_rollouts(model, train_cases, args, device)
            stats = ppo.ppo_update(model, opt, rollouts, args)
            reward = float(np.mean([item.reward for item in rollouts])) if rollouts else 0.0
            rec: dict[str, float | int | str] = {
                "update": global_update,
                "stage_index": stage_index,
                "stage_max_anchors": stage_max,
                "stage_update": stage_update,
                "rollouts": len(rollouts),
                "mean_reward": reward,
                "policy_loss": stats["policy_loss"],
                "ratio_mean": stats["ratio_mean"],
                "elapsed_s": time.perf_counter() - started,
                "stage_elapsed_s": time.perf_counter() - stage_started,
                "is_best": 0,
                "advanced_stage": 0,
            }

            elapsed_s = time.perf_counter() - started
            in_verbose_window = elapsed_s < args.verbose_first_minutes * 60.0
            eval_every_updates = args.verbose_eval_every_updates if in_verbose_window else args.eval_every_updates
            should_eval = stage_update == 1 or stage_update % max(eval_every_updates, 1) == 0
            if should_eval:
                label = f"stage{stage_max}_update{global_update}"
                rows = ppo.evaluate(model, eval_cases, args, device, label)
                add_stage_fields(rows, stage_index=stage_index, stage_max=stage_max, global_update=global_update, label=label)
                eval_rows.extend(rows)
                score = score_rows(rows)
                stage_eval_count += 1
                rec["eval_label"] = label
                rec["eval_mean_squared_max_offset"] = score
                rec["stage_best_score_before"] = stage_best
                rec["previous_eval_score"] = last_eval_score
                rec["eval_delta_vs_previous"] = (last_eval_score - score) if math.isfinite(last_eval_score) else ""
                improvement_threshold = args.curriculum_min_delta
                if math.isfinite(stage_best):
                    improvement_threshold = max(improvement_threshold, abs(stage_best) * args.curriculum_min_relative_delta)
                stage_best_before = stage_best
                improved_stage = (not math.isfinite(stage_best)) or score < stage_best - improvement_threshold
                if improved_stage:
                    stage_best = score
                    last_best_eval = stage_eval_count
                if score < best_score:
                    best_score = score
                    rec["is_best"] = 1
                    save_checkpoint(best_path, model, args, source, history + [rec], stage_index=stage_index, stage_max=stage_max, best_score=best_score)
                    print(f"checkpoint_best update={global_update} stage_max={stage_max} mean_sq={score:.5f} path={best_path}", flush=True)
                rec["stage_best_score"] = stage_best
                rec["global_best_score"] = best_score
                rec["stage_eval_count"] = stage_eval_count
                rec["evals_since_stage_best"] = stage_eval_count - last_best_eval
                rec["stage_elapsed_min"] = (time.perf_counter() - stage_started) / 60.0
                rec["improvement_threshold"] = improvement_threshold
                rec["eval_delta_vs_stage_best_before"] = (stage_best_before - score) if math.isfinite(stage_best_before) else ""
                rec["improved_stage"] = int(improved_stage)
                rec["saturation_time_gate"] = int(rec["stage_elapsed_min"] >= args.curriculum_min_stage_minutes)
                print(
                    f"eval update={global_update} stage_max={stage_max} stage_update={stage_update} "
                    f"score={score:.5f} stage_best={stage_best:.5f} improved={int(improved_stage)} "
                    f"delta_prev={rec['eval_delta_vs_previous']} delta_best={rec['eval_delta_vs_stage_best_before']} "
                    f"evals_since_best={rec['evals_since_stage_best']}/{args.curriculum_patience_evals} "
                    f"stage_min={rec['stage_elapsed_min']:.1f}/{args.curriculum_min_stage_minutes:.1f} "
                    f"min_updates={stage_update}/{args.curriculum_min_updates} reward={reward:.4f} {summary_text(rows)}",
                    flush=True,
                )
                last_eval_score = score
                saturated = (
                    stage_update >= args.curriculum_min_updates
                    and stage_eval_count - last_best_eval >= args.curriculum_patience_evals
                    and rec["stage_elapsed_min"] >= args.curriculum_min_stage_minutes
                )
                if saturated:
                    if stage_index < len(stages) - 1:
                        rec["advanced_stage"] = 1
                        stage_rows.append(
                            {
                                "stage_index": stage_index,
                                "stage_max_anchors": stage_max,
                                "stage_updates": stage_update,
                                "stage_evals": stage_eval_count,
                                "stage_best_score": stage_best,
                                "elapsed_s": time.perf_counter() - started,
                                "reason": "saturated",
                            }
                        )
                        history.append(rec)
                        print(
                            f"stage_saturated index={stage_index} max_anchors={stage_max} updates={stage_update} evals={stage_eval_count} best={stage_best:.5f}",
                            flush=True,
                        )
                        break
                    last_best_eval = stage_eval_count
                    print(
                        f"final_stage_saturated_continue index={stage_index} max_anchors={stage_max} updates={stage_update} evals={stage_eval_count} best={stage_best:.5f}",
                        flush=True,
                    )
            else:
                rec["stage_best_score"] = stage_best
                rec["global_best_score"] = best_score

            history.append(rec)
            now = time.perf_counter()
            progress_interval = args.progress_every_minutes if (now - started) < args.verbose_first_minutes * 60.0 else args.progress_every_minutes_late
            if now - last_progress >= progress_interval * 60.0:
                print(
                    f"progress elapsed_min={(now-started)/60.0:.1f} update={global_update} stage_max={stage_max} "
                    f"stage_update={stage_update} stage_min={(now-stage_started)/60.0:.1f} reward={reward:.4f} "
                    f"stage_best={stage_best:.5f} global_best={best_score:.5f} evals={stage_eval_count}",
                    flush=True,
                )
                last_progress = now
            if global_update == 1 or should_eval:
                save_checkpoint(latest_path, model, args, source, history, stage_index=stage_index, stage_max=stage_max, best_score=best_score)
                write_csv(history_path, history)
                write_csv(detail_path, eval_rows)
                write_csv(summary_path, ppo.summarize(eval_rows))
                write_csv(stage_path, stage_rows)

        if stop:
            stage_rows.append(
                {
                    "stage_index": stage_index,
                    "stage_max_anchors": stage_max,
                    "stage_updates": stage_update,
                    "stage_evals": stage_eval_count,
                    "stage_best_score": stage_best,
                    "elapsed_s": time.perf_counter() - started,
                    "reason": "runtime",
                }
            )
            break

    final_rows = []
    if 'eval_cases' in locals():
        label = "after"
        final_rows = ppo.evaluate(model, eval_cases, args, device, label)
        add_stage_fields(final_rows, stage_index=stage_index if 'stage_index' in locals() else -1, stage_max=stage_max if 'stage_max' in locals() else -1, global_update=global_update, label=label)
        eval_rows.extend(final_rows)
    save_checkpoint(OUTPUTS / f"{args.prefix}.pt", model, args, source, history, stage_index=stage_index if 'stage_index' in locals() else -1, stage_max=stage_max if 'stage_max' in locals() else -1, best_score=best_score)
    save_checkpoint(latest_path, model, args, source, history, stage_index=stage_index if 'stage_index' in locals() else -1, stage_max=stage_max if 'stage_max' in locals() else -1, best_score=best_score)
    write_csv(history_path, history)
    write_csv(detail_path, eval_rows)
    write_csv(summary_path, ppo.summarize(eval_rows))
    write_csv(stage_path, stage_rows)
    print(f"curriculum_done updates={global_update} elapsed_min={(time.perf_counter()-started)/60.0:.1f} best={best_score:.5f}", flush=True)
    print(f"Wrote {OUTPUTS / (args.prefix + '.pt')}", flush=True)
    print(f"Wrote {best_path}", flush=True)


if __name__ == "__main__":
    main()