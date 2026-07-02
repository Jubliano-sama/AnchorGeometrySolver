from __future__ import annotations

import math
from pathlib import Path
import random
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from anchor_geometry_solver.compat import ensure_legacy_paths  # noqa: E402

ensure_legacy_paths()
import anchor_solver_ml_distance_completion as dc  # noqa: E402


TOKENS = dc.TOKENS
BLUE = dc.BLUE
GOLD = dc.GOLD
NEUTRAL = dc.NEUTRAL

LAYOUTS = (
    ("corridor", "Corridor", {"width": 40.0, "height": 12.0}, 24),
    ("l_shape", "L shape", {"width": 30.0, "height": 24.0, "leg_x": 12.0, "leg_y": 9.5}, 22),
    ("t_shape", "T shape", {"width": 30.0, "height": 24.0, "bar_w": 10.5, "top_h": 8.0}, 22),
    ("u_shape", "U shape", {"width": 30.0, "height": 24.0, "leg_w": 7.5, "bottom_h": 8.0}, 24),
    ("hollow_square", "Hollow square", {"width": 27.0, "height": 27.0, "hole_w": 11.0, "hole_h": 11.0}, 24),
    ("rooms", "Rooms", {"width": 32.0, "height": 24.0, "corridor_w": 5.5, "corridor_h": 5.0}, 24),
)


def footprint_mask(shape: str, params: dict[str, float], *, resolution: int = 220) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs = torch.linspace(0.0, params["width"], resolution)
    ys = torch.linspace(0.0, params["height"], resolution)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    points = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1)
    mask = dc.shape_contains(points, shape, params).reshape(resolution, resolution).cpu().numpy()
    return xs.cpu().numpy(), ys.cpu().numpy(), mask


def grid_points(shape: str, params: dict[str, float], target_n: int, *, seed: int) -> np.ndarray:
    rng = random.Random(seed)
    torch.manual_seed(seed)
    spacing = 5.3
    offset_x = spacing * 0.48
    offset_y = spacing * 0.47
    xs = torch.arange(offset_x, params["width"], spacing, dtype=torch.float32)
    ys = torch.arange(offset_y, params["height"], spacing, dtype=torch.float32)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    candidates = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1)
    candidates = candidates[dc.shape_contains(candidates, shape, params)]
    jitter = torch.randn_like(candidates) * 0.12
    candidates = candidates + jitter
    candidates[:, 0].clamp_(0.0, params["width"])
    candidates[:, 1].clamp_(0.0, params["height"])
    candidates = candidates[dc.shape_contains(candidates, shape, params)]
    if candidates.shape[0] <= target_n:
        return candidates.cpu().numpy()
    order = list(range(int(candidates.shape[0])))
    rng.shuffle(order)
    chosen = candidates[order[:target_n]]
    return chosen.cpu().numpy()


def degree_values(points: np.ndarray) -> tuple[list[tuple[int, int]], np.ndarray]:
    edges: list[tuple[int, int]] = []
    degrees = np.zeros(points.shape[0], dtype=int)
    for i in range(points.shape[0]):
        for j in range(i + 1, points.shape[0]):
            if math.dist(points[i], points[j]) <= dc.EDGE_RADIUS_M:
                edges.append((i, j))
                degrees[i] += 1
                degrees[j] += 1
    return edges, degrees


def draw_layout(ax, shape: str, title: str, params: dict[str, float], points: np.ndarray) -> None:
    _xs, _ys, mask = footprint_mask(shape, params)
    ax.imshow(
        mask,
        origin="lower",
        extent=(0.0, params["width"], 0.0, params["height"]),
        cmap=matplotlib.colors.ListedColormap([TOKENS["surface"], "#EDF3FF"]),
        alpha=0.88,
        interpolation="nearest",
        zorder=0,
    )
    edges, degrees = degree_values(points)
    for i, j in edges:
        ax.plot(
            [points[i, 0], points[j, 0]],
            [points[i, 1], points[j, 1]],
            color=NEUTRAL["base"],
            linewidth=0.62,
            alpha=0.46,
            zorder=1,
        )
    ax.scatter(
        points[:, 0],
        points[:, 1],
        s=42,
        color=GOLD["base"],
        edgecolors=BLUE["dark"],
        linewidths=0.8,
        zorder=3,
    )
    for index, (x_m, y_m) in enumerate(points):
        ax.text(x_m + 0.15, y_m + 0.15, f"A{index:02d}", fontsize=5.6, color=TOKENS["ink"], alpha=0.80)
    ax.set_title(
        f"{title}\n{points.shape[0]} anchors | degree {int(degrees.min())}-{int(degrees.max())}",
        loc="left",
        fontsize=10,
        fontweight="bold",
        color=TOKENS["ink"],
    )
    span = max(params["width"], params["height"])
    pad = span * 0.035
    ax.set_xlim(-pad, params["width"] + pad)
    ax.set_ylim(-pad, params["height"] + pad)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, color=TOKENS["grid"], linewidth=0.55)
    ax.tick_params(labelsize=7, colors=TOKENS["muted"], length=0)
    for spine in ax.spines.values():
        spine.set_color(TOKENS["axis"])


def main() -> None:
    output = ROOT / "docs" / "results" / "office_layout_families_ground_truth.png"
    plt.rcParams.update(
        {
            "figure.facecolor": TOKENS["surface"],
            "savefig.facecolor": TOKENS["surface"],
            "font.family": ["Segoe UI", "DejaVu Sans", "Arial", "sans-serif"],
        }
    )
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 8.7), dpi=185)
    for index, (shape, title, params, target_n) in enumerate(LAYOUTS):
        row, col = divmod(index, 3)
        points = grid_points(shape, params, target_n, seed=2026070300 + index)
        draw_layout(axes[row, col], shape, title, params, points)
    fig.suptitle(
        "Office Layout Families, Ground Truth Only",
        x=0.055,
        y=0.99,
        ha="left",
        fontsize=17,
        fontweight="bold",
        color=TOKENS["ink"],
    )
    fig.text(
        0.055,
        0.955,
        "Yellow dots are true anchor coordinates; faint lines are true anchor pairs within the 8 m radio radius used to create measured ranges.",
        ha="left",
        va="top",
        fontsize=9.4,
        color=TOKENS["muted"],
    )
    fig.subplots_adjust(left=0.052, right=0.985, top=0.895, bottom=0.065, wspace=0.18, hspace=0.38)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
