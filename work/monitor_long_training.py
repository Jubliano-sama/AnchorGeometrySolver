from __future__ import annotations

import argparse
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"

PROGRESS_RE = re.compile(
    r"progress step=(?P<step>\d+)/(?:\d+) epoch=(?P<epoch>\d+)/(?:\d+) inner=(?P<inner>\d+)/(?:\d+) "
    r"loss=(?P<loss>[0-9.]+).*?cases_per_s=(?P<cps>[0-9.]+) elapsed_s=(?P<elapsed>[0-9.]+)"
)
EPOCH_RE = re.compile(r"epoch_done (?P<epoch>\d+)/(?:\d+): mean_loss=(?P<mean>[0-9.]+) last_loss=(?P<last>[0-9.]+)")


def parse_log(path: Path) -> dict[str, float | int | str | None]:
    latest = None
    epochs: list[tuple[int, float, float]] = []
    if path.exists():
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            match = PROGRESS_RE.search(line)
            if match:
                latest = {
                    "step": int(match.group("step")),
                    "epoch": int(match.group("epoch")),
                    "inner": int(match.group("inner")),
                    "loss": float(match.group("loss")),
                    "cases_per_s": float(match.group("cps")),
                    "elapsed_s": float(match.group("elapsed")),
                }
                continue
            match = EPOCH_RE.search(line)
            if match:
                epochs.append((int(match.group("epoch")), float(match.group("mean")), float(match.group("last"))))
    best = min(epochs, key=lambda item: item[1]) if epochs else None
    last_epoch = epochs[-1] if epochs else None
    return {
        "step": latest["step"] if latest else None,
        "epoch": latest["epoch"] if latest else None,
        "inner": latest["inner"] if latest else None,
        "loss": latest["loss"] if latest else None,
        "cases_per_s": latest["cases_per_s"] if latest else None,
        "elapsed_s": latest["elapsed_s"] if latest else None,
        "last_epoch": last_epoch[0] if last_epoch else None,
        "last_epoch_mean": last_epoch[1] if last_epoch else None,
        "best_epoch": best[0] if best else None,
        "best_epoch_mean": best[1] if best else None,
    }


def fmt(value: object, digits: int = 3) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize active long anchor training logs.")
    parser.add_argument("prefix", nargs="*", default=[path.name[:-10] for path in OUTPUTS.glob("anchor_solver_ml_distance_completion_*.train.log")])
    args = parser.parse_args()
    print("prefix step epoch inner loss cps last_epoch_mean best_epoch_mean stderr_bytes checkpoints")
    for prefix in sorted(args.prefix):
        log = OUTPUTS / f"{prefix}.train.log"
        err = OUTPUTS / f"{prefix}.train.err.log"
        stats = parse_log(log)
        checkpoints = sorted(path.name for path in OUTPUTS.glob(f"{prefix}*.pt"))
        print(
            f"{prefix} {fmt(stats['step'], 0)} {fmt(stats['epoch'], 0)} {fmt(stats['inner'], 0)} "
            f"{fmt(stats['loss'])} {fmt(stats['cases_per_s'], 1)} {fmt(stats['last_epoch_mean'])} "
            f"{fmt(stats['best_epoch_mean'])} {err.stat().st_size if err.exists() else 0} {','.join(checkpoints) if checkpoints else '-'}"
        )


if __name__ == "__main__":
    main()

