from __future__ import annotations

import argparse
import csv
from pathlib import Path
import random
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
sys.path.insert(0, str(ROOT / "work"))

import anchor_solver_ml_distance_completion as dc  # noqa: E402
import anchor_solver_p95_ml_cases as p95  # noqa: E402


BUCKET_KEYS = ("random", "grid", "office")
BUCKET_LABELS = {
    "random": "Random 16-32",
    "grid": "Grid >=16",
    "office": "Office >=16",
}
METHODS = ("ML completed", "ML weak polish")
LOWER_BETTER = ("median_missing_mae_m", "median_max_offset_m", "p90_max_offset_m", "p95_max_offset_m", "median_known_rmse_m")
HIGHER_BETTER = ("under_20cm", "under_50cm", "under_1m")

TOKENS = {
    "surface": "#FCFCFD",
    "panel": "#FFFFFF",
    "ink": "#1F2430",
    "muted": "#6F768A",
    "grid": "#E6E8F0",
    "axis": "#D7DBE7",
}
COLORS = ["#5477C4", "#BD569B", "#CC6F47", "#71B436", "#8C6FD1", "#2C9C8F"]


def parse_checkpoint_spec(text: str) -> tuple[str, Path]:
    if "=" in text:
        label, value = text.split("=", 1)
        return label.strip(), Path(value.strip())
    path = Path(text.strip())
    return path.stem, path


def fixed_batches(cases: list[p95.CaseSpec], *, device: torch.device, batch_size: int) -> list[tuple[dc.GraphBatch, list[p95.CaseSpec]]]:
    batches: list[tuple[dc.GraphBatch, list[p95.CaseSpec]]] = []
    for start in range(0, len(cases), batch_size):
        chunk_cases = cases[start : start + batch_size]
        batch = p95.graph_batch_from_cases(chunk_cases, device)
        batches.append((batch, chunk_cases))
    return batches


@torch.no_grad()
def predict_fixed_batches(
    model: dc.DistanceCompletionNet,
    fixed: list[tuple[dc.GraphBatch, list[p95.CaseSpec]]],
) -> list[tuple[dc.GraphBatch, torch.Tensor, list[p95.CaseSpec]]]:
    chunks: list[tuple[dc.GraphBatch, torch.Tensor, list[p95.CaseSpec]]] = []
    model.eval()
    for batch, chunk_cases in fixed:
        pred = model(batch.node_features, batch.edge_features, batch.mask, batch.measured_mask, batch.pair_mask)
        chunks.append((batch, pred, chunk_cases))
    return chunks


def row_dict(label: str, checkpoint: Path, row: p95.SolvedCase) -> dict[str, str | int | float]:
    return {
        "model": label,
        "checkpoint": str(checkpoint),
        "bucket": row.bucket,
        "method": row.method,
        "case_index": row.case_index,
        "family": row.family,
        "shape": row.shape,
        "anchors": row.anchors,
        "known_pairs": row.known_pairs,
        "full_pairs": row.full_pairs,
        "missing_mae_m": row.missing_mae_m,
        "known_rmse_m": row.known_rmse_m,
        "known_max_residual_m": row.known_max_residual_m,
        "max_offset_m": row.max_offset_m,
        "median_offset_m": row.median_offset_m,
        "p95_offset_m": row.p95_offset_m,
    }


def summarize(rows: list[dict[str, str | int | float]]) -> list[dict[str, str | int | float]]:
    summary: list[dict[str, str | int | float]] = []
    keys = sorted({(str(row["model"]), str(row["bucket"]), str(row["method"])) for row in rows})
    for model, bucket, method in keys:
        part = [row for row in rows if row["model"] == model and row["bucket"] == bucket and row["method"] == method]
        max_offsets = np.array([float(row["max_offset_m"]) for row in part], dtype=float)
        missing_mae = np.array([float(row["missing_mae_m"]) for row in part], dtype=float)
        known_rmse = np.array([float(row["known_rmse_m"]) for row in part], dtype=float)
        summary.append(
            {
                "model": model,
                "bucket": bucket,
                "method": method,
                "cases": len(part),
                "median_missing_mae_m": float(np.median(missing_mae)),
                "median_max_offset_m": float(np.median(max_offsets)),
                "p90_max_offset_m": float(np.quantile(max_offsets, 0.90)),
                "p95_max_offset_m": float(np.quantile(max_offsets, 0.95)),
                "max_offset_m": float(np.max(max_offsets)),
                "median_known_rmse_m": float(np.median(known_rmse)),
                "under_20cm": float(np.mean(max_offsets <= 0.20)),
                "under_50cm": float(np.mean(max_offsets <= 0.50)),
                "under_1m": float(np.mean(max_offsets <= 1.00)),
            }
        )
    return summary


def write_csv(path: Path, rows: list[dict[str, str | int | float]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write for {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def compare_to_baseline(summary: list[dict[str, str | int | float]], baseline_label: str, tolerance: float) -> list[dict[str, str | float]]:
    by_key = {(str(row["model"]), str(row["bucket"]), str(row["method"])): row for row in summary}
    gates: list[dict[str, str | float]] = []
    candidates = sorted({str(row["model"]) for row in summary if row["model"] != baseline_label})
    for candidate in candidates:
        for bucket in BUCKET_LABELS.values():
            for method in METHODS:
                base = by_key.get((baseline_label, bucket, method))
                cand = by_key.get((candidate, bucket, method))
                if base is None or cand is None:
                    continue
                for metric in LOWER_BETTER:
                    base_value = float(base[metric])
                    cand_value = float(cand[metric])
                    gates.append(
                        {
                            "candidate": candidate,
                            "bucket": bucket,
                            "method": method,
                            "metric": metric,
                            "direction": "lower",
                            "baseline": base_value,
                            "candidate_value": cand_value,
                            "delta": cand_value - base_value,
                            "passed": "yes" if cand_value <= base_value - tolerance else "no",
                        }
                    )
                for metric in HIGHER_BETTER:
                    base_value = float(base[metric])
                    cand_value = float(cand[metric])
                    gates.append(
                        {
                            "candidate": candidate,
                            "bucket": bucket,
                            "method": method,
                            "metric": metric,
                            "direction": "higher",
                            "baseline": base_value,
                            "candidate_value": cand_value,
                            "delta": cand_value - base_value,
                            "passed": "yes" if cand_value >= base_value + tolerance else "no",
                        }
                    )
    return gates


def make_figure(path: Path, summary: list[dict[str, str | int | float]], *, cases_per_bucket: int) -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": TOKENS["surface"],
            "savefig.facecolor": TOKENS["surface"],
            "font.family": ["Segoe UI", "DejaVu Sans", "Arial", "sans-serif"],
        }
    )
    models = sorted({str(row["model"]) for row in summary})
    model_colors = {model: COLORS[i % len(COLORS)] for i, model in enumerate(models)}
    fig, axes = plt.subplots(3, 2, figsize=(16.2, 13.4), dpi=170, sharex=False)
    fig.text(
        0.035,
        0.985,
        "Fixed-heldout checkpoint comparison",
        ha="left",
        va="top",
        fontsize=20,
        fontweight="bold",
        color=TOKENS["ink"],
    )
    fig.text(
        0.035,
        0.958,
        (
            f"Every model is evaluated on the same {cases_per_bucket} layouts per bucket and the same generated noisy measured ranges. "
            "Lower p95 max offset is better; higher share under 1 m is better."
        ),
        ha="left",
        va="top",
        fontsize=9.2,
        color=TOKENS["muted"],
    )
    metric_specs = (("p95_max_offset_m", "p95 max offset (m)"), ("under_1m", "share under 1 m"))
    bucket_order = [BUCKET_LABELS[key] for key in BUCKET_KEYS]
    for row_index, bucket in enumerate(bucket_order):
        for col_index, (metric, ylabel) in enumerate(metric_specs):
            ax = axes[row_index, col_index]
            ax.set_facecolor(TOKENS["panel"])
            labels: list[str] = []
            values: list[float] = []
            colors: list[str] = []
            for model in models:
                for method in METHODS:
                    part = [row for row in summary if row["model"] == model and row["bucket"] == bucket and row["method"] == method]
                    if not part:
                        continue
                    labels.append(f"{model}\n{method.replace('ML ', '')}")
                    values.append(float(part[0][metric]))
                    colors.append(model_colors[model])
            x = np.arange(len(values))
            ax.bar(x, values, color=colors, edgecolor=TOKENS["ink"], linewidth=0.35, alpha=0.86)
            ax.set_title(bucket, loc="left", fontsize=10.5, fontweight="semibold", color=TOKENS["ink"])
            ax.set_ylabel(ylabel, fontsize=8.5, color=TOKENS["muted"])
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=32, ha="right", fontsize=6.6)
            ax.grid(True, axis="y", color=TOKENS["grid"], linewidth=0.55)
            ax.tick_params(axis="y", labelsize=7.5, colors=TOKENS["muted"], length=0)
            ax.tick_params(axis="x", colors=TOKENS["muted"], length=0)
            for spine in ax.spines.values():
                spine.set_color(TOKENS["axis"])
    fig.subplots_adjust(left=0.055, right=0.985, top=0.90, bottom=0.08, hspace=0.44, wspace=0.18)
    fig.savefig(path, bbox_inches="tight", dpi=170)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate checkpoints on one fixed noisy heldout anchor set.")
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--checkpoint",
        action="append",
        required=True,
        help="Checkpoint spec as label=path or plain path. Pass once per model.",
    )
    parser.add_argument("--baseline-label", default="baseline")
    parser.add_argument("--cases-per-bucket", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=2026062703)
    parser.add_argument("--solver-iterations", type=int, default=80)
    parser.add_argument("--polish-iterations", type=int, default=80)
    parser.add_argument("--weak-polish-iterations", type=int, default=80)
    parser.add_argument("--predicted-sigma", type=float, default=0.55)
    parser.add_argument("--predicted-sigma-slope", type=float, default=0.65)
    parser.add_argument("--closest-predicted-pairs-per-anchor", type=float, default=5.0)
    parser.add_argument("--weak-completion-max-distance", type=float, default=18.0)
    parser.add_argument("--weak-completion-sigma-multiplier", type=float, default=2.5)
    parser.add_argument("--tolerance", type=float, default=1e-9)
    parser.add_argument("--prefix", default="anchor_solver_fixed_holdout_compare")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed % (2**32 - 1))
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = p95.choose_device(args.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
    checkpoint_specs = [parse_checkpoint_spec(item) for item in args.checkpoint]
    print(f"device={device} checkpoints={len(checkpoint_specs)}", flush=True)

    frozen_by_bucket: dict[str, list[tuple[dc.GraphBatch, list[p95.CaseSpec]]]] = {}
    for bucket_key in BUCKET_KEYS:
        print(f"generating fixed bucket={bucket_key} cases={args.cases_per_bucket}", flush=True)
        cases = p95.generate_cases(bucket_key, args.cases_per_bucket, device)
        node_counts = [int(case.points.shape[0]) for case in cases]
        fixed = fixed_batches(cases, device=device, batch_size=args.batch_size)
        frozen_by_bucket[bucket_key] = fixed
        measured_pairs = [int(batch.measured_mask.sum().detach().cpu().item() // 2) for batch, _chunk in fixed]
        print(
            f"fixed bucket={bucket_key} min_nodes={min(node_counts)} median_nodes={np.median(node_counts):.0f} "
            f"max_nodes={max(node_counts)} measured_pairs={sum(measured_pairs)}",
            flush=True,
        )

    detail_rows: list[dict[str, str | int | float]] = []
    for label, checkpoint in checkpoint_specs:
        if not checkpoint.exists():
            raise FileNotFoundError(checkpoint)
        print(f"loading model={label} checkpoint={checkpoint}", flush=True)
        model = p95.load_model(checkpoint, device)
        for bucket_key in BUCKET_KEYS:
            print(f"solving model={label} bucket={bucket_key}", flush=True)
            chunks = predict_fixed_batches(model, frozen_by_bucket[bucket_key])
            rows = p95.solve_bucket(
                BUCKET_LABELS[bucket_key],
                chunks,
                predicted_sigma=args.predicted_sigma,
                predicted_sigma_slope=args.predicted_sigma_slope,
                closest_predicted_pairs_per_anchor=args.closest_predicted_pairs_per_anchor,
                solver_iterations=args.solver_iterations,
                polish_iterations=args.polish_iterations,
                weak_polish_iterations=args.weak_polish_iterations,
                weak_completion_max_distance=args.weak_completion_max_distance,
                weak_completion_sigma_multiplier=args.weak_completion_sigma_multiplier,
            )
            detail_rows.extend(row_dict(label, checkpoint, row) for row in rows)

    summary_rows = summarize(detail_rows)
    gate_rows = compare_to_baseline(summary_rows, args.baseline_label, args.tolerance)
    detail_path = OUTPUTS / f"{args.prefix}_detail.csv"
    summary_path = OUTPUTS / f"{args.prefix}_summary.csv"
    gate_path = OUTPUTS / f"{args.prefix}_gate.csv"
    figure_path = OUTPUTS / f"{args.prefix}.png"
    write_csv(detail_path, detail_rows)
    write_csv(summary_path, summary_rows)
    if gate_rows:
        write_csv(gate_path, gate_rows)
    make_figure(figure_path, summary_rows, cases_per_bucket=args.cases_per_bucket)

    for row in summary_rows:
        print(
            f"summary model={row['model']} bucket={row['bucket']} method={row['method']} "
            f"p95={float(row['p95_max_offset_m']):.3f}m median={float(row['median_max_offset_m']):.3f}m "
            f"under1m={float(row['under_1m']):.3f} missing_mae={float(row['median_missing_mae_m']):.3f}m",
            flush=True,
        )
    if gate_rows:
        failed = sum(1 for row in gate_rows if row["passed"] == "no")
        print(f"gate_checks={len(gate_rows)} failed={failed}", flush=True)
    print(f"Wrote {detail_path}", flush=True)
    print(f"Wrote {summary_path}", flush=True)
    if gate_rows:
        print(f"Wrote {gate_path}", flush=True)
    print(f"Wrote {figure_path}", flush=True)


if __name__ == "__main__":
    main()
