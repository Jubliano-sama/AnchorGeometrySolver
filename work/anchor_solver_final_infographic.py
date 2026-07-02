from __future__ import annotations

import csv
from pathlib import Path
import textwrap

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"

FONT_FAMILY = ["Segoe UI", "DejaVu Sans", "Arial", "sans-serif"]
MONO_FONT_FAMILY = ["Consolas", "DejaVu Sans Mono", "monospace"]
TOKENS = {"surface": "#FCFCFD", "panel": "#FFFFFF", "ink": "#1F2430", "muted": "#6F768A", "grid": "#E6E8F0", "axis": "#D7DBE7"}
BLUE = {"base": "#A3BEFA", "mid": "#5477C4", "dark": "#2E4780"}
GOLD = {"base": "#FFE15B", "mid": "#B8A037", "dark": "#736422"}
ORANGE = {"base": "#F0986E", "mid": "#CC6F47", "dark": "#804126"}
OLIVE = {"base": "#A3D576", "mid": "#71B436", "dark": "#386411"}
PINK = {"base": "#F390CA", "mid": "#BD569B", "dark": "#8A3A6F"}
NEUTRAL = {"light": "#E2E5EA", "base": "#C5CAD3", "mid": "#7A828F", "dark": "#464C55"}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def row(rows, **match):
    for item in rows:
        if all(item.get(k) == v for k, v in match.items()):
            return item
    raise KeyError(match)


def style_ax(ax):
    ax.set_facecolor(TOKENS["panel"])
    ax.grid(True, axis="x", color=TOKENS["grid"], linewidth=0.7)
    ax.tick_params(axis="both", labelsize=8, colors=TOKENS["muted"], length=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color(TOKENS["axis"])
    ax.spines["bottom"].set_color(TOKENS["axis"])


def hbar(ax, labels, values, colors, title, subtitle, xmax=None, suffix="m"):
    y = np.arange(len(labels))[::-1]
    ax.barh(y, values, color=colors, edgecolor=TOKENS["ink"], linewidth=0.35)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8, color=TOKENS["ink"])
    style_ax(ax)
    ax.set_title(title, loc="left", fontsize=11, fontweight="bold", color=TOKENS["ink"], pad=18)
    ax.text(0, 1.02, subtitle, transform=ax.transAxes, ha="left", va="bottom", fontsize=8, color=TOKENS["muted"])
    if xmax is not None:
        ax.set_xlim(0, xmax)
    limit = ax.get_xlim()[1]
    for yi, val in zip(y, values):
        ax.text(min(val + limit * 0.015, limit * 0.98), yi, f"{val:.2f}{suffix}", va="center", ha="left", fontsize=8, color=TOKENS["ink"], fontfamily=MONO_FONT_FAMILY)


def main() -> None:
    ml = read_rows(OUTPUTS / "anchor_solver_fixed_holdout_bigF_e19_med24_summary.csv")
    graph = read_rows(OUTPUTS / "anchor_solver_graph_scaffold_eval_keepprior_rgo16_cap32_h2_summary.csv")
    selector = read_rows(OUTPUTS / "anchor_solver_candidate_selector_eval_small_summary.csv")

    buckets = ["Random 16-32", "Grid >=16", "Office >=16"]
    short = {"Random 16-32": "Random", "Grid >=16": "Grid", "Office >=16": "Office"}

    ml_p95_labels = []
    ml_p95_values = []
    ml_p95_colors = []
    for bucket in buckets:
        for model, label, color in (("baseline", "Baseline", NEUTRAL["base"]), ("bigF_best_e19", "BigF", BLUE["base"])):
            r = row(ml, model=model, bucket=bucket, method="ML completed")
            ml_p95_labels.append(f"{short[bucket]} {label}")
            ml_p95_values.append(float(r["p95_max_offset_m"]))
            ml_p95_colors.append(color)

    mae_labels = []
    mae_values = []
    mae_colors = []
    for bucket in buckets:
        for model, label, color in (("baseline", "Baseline", NEUTRAL["base"]), ("bigF_best_e19", "BigF", OLIVE["base"])):
            r = row(ml, model=model, bucket=bucket, method="ML completed")
            mae_labels.append(f"{short[bucket]} {label}")
            mae_values.append(float(r["median_missing_mae_m"]))
            mae_colors.append(color)

    graph_labels = []
    graph_values = []
    graph_colors = []
    for bucket in buckets:
        for method, label, color in (("production-priors", "Prod", NEUTRAL["base"]), ("graph-shortest-scaffold", "Graph", GOLD["base"])):
            r = row(graph, bucket=bucket, method=method)
            graph_labels.append(f"{short[bucket]} {label}")
            graph_values.append(float(r["p95_max_offset_m"]))
            graph_colors.append(color)

    selector_labels = []
    selector_values = []
    selector_colors = []
    for bucket in buckets:
        for method, label, color in (("bigD:ML completed", "BigD", NEUTRAL["base"]), ("selector_topology", "Selector", PINK["base"]), ("oracle_best_candidate", "Oracle", ORANGE["base"])):
            r = row(selector, bucket=bucket, method=method)
            selector_labels.append(f"{short[bucket]} {label}")
            selector_values.append(float(r["p95_max_offset_m"]))
            selector_colors.append(color)

    plt.rcParams.update({"figure.facecolor": TOKENS["surface"], "savefig.facecolor": TOKENS["surface"], "font.family": FONT_FAMILY})
    fig = plt.figure(figsize=(16, 10.5), dpi=170)
    gs = fig.add_gridspec(2, 2, left=0.055, right=0.98, top=0.84, bottom=0.11, hspace=0.42, wspace=0.30)
    fig.text(0.055, 0.965, "Anchor geometry solver experiment summary", fontsize=22, fontweight="bold", color=TOKENS["ink"], ha="left", va="top")
    subtitle = "Fair generated layouts with noisy <=8 m edges. Rotations/mirrors aligned away; lower p95 max coordinate offset is better."
    fig.text(0.055, 0.928, subtitle, fontsize=9.5, color=TOKENS["muted"], ha="left", va="top")

    ax1 = fig.add_subplot(gs[0, 0])
    hbar(ax1, ml_p95_labels, ml_p95_values, ml_p95_colors, "ML solve p95 offset", "24 cases per bucket, completed branch", xmax=max(10, max(ml_p95_values) * 1.18))

    ax2 = fig.add_subplot(gs[0, 1])
    hbar(ax2, mae_labels, mae_values, mae_colors, "ML missing-distance MAE", "Same 24-case validation; BigF learns distances", xmax=max(mae_values) * 1.35)

    ax3 = fig.add_subplot(gs[1, 0])
    hbar(ax3, graph_labels, graph_values, graph_colors, "Graph-shortest scaffold p95", "16 cases per bucket; temporary graph-path springs", xmax=max(graph_values) * 1.18)

    ax4 = fig.add_subplot(gs[1, 1])
    hbar(ax4, selector_labels, selector_values, selector_colors, "Candidate selector headroom", "8-case diagnostic: observable selector vs oracle-best pool", xmax=max(selector_values) * 1.18)

    note = (
        "Takeaway: the ML distance completer reliably reduces missing-distance MAE, but the solved layout p95 worsens on larger grid/office validation. "
        "Graph-shortest scaffolding gives excellent medians and random-layout p95, yet rare office/grid ambiguities remain. "
        "The hard cases look like sparse global shape ambiguity, not simple local RMSE failure."
    )
    fig.text(0.055, 0.045, "\n".join(textwrap.wrap(note, width=185)), fontsize=9, color=TOKENS["ink"], ha="left", va="bottom")

    out = OUTPUTS / "anchor_solver_final_experiment_infographic.png"
    fig.savefig(out, bbox_inches="tight", dpi=170)
    print(out)


if __name__ == "__main__":
    main()
