from __future__ import annotations

import argparse
from pathlib import Path

from anchor_geometry_solver.benchmark import run_benchmark
from anchor_geometry_solver.config import parse_config


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Anchor geometry solver benchmark harness.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    bench = subparsers.add_parser("bench", help="Run a configured benchmark or sweep.")
    bench.add_argument("config", type=Path)

    args = parser.parse_args(argv)
    if args.command == "bench":
        config = parse_config(args.config)
        _rows, summary = run_benchmark(config)
        print("summary")
        for row in summary:
            print(row)
