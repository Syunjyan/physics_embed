from __future__ import annotations

from pathlib import Path
import argparse
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments._empirical_ablation_common import run_empirical_ablation


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="navier_stokes empirical formula ablation.")
    parser.add_argument("--equation", default="navier_stokes")
    parser.add_argument("--dataset", default="data/navier_stokes.npz")
    parser.add_argument("--output-root", default="runs/ablation/navier_stokes_empirical")
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--epochs", type=int, default=3000)
    parser.add_argument("--samples", type=int, default=4096)
    parser.add_argument("--boundary-samples", type=int, default=1024)
    parser.add_argument("--hidden", type=int, nargs="+", default=[64, 64, 64, 64])
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--empirical-weight", type=float, default=0.01)
    parser.add_argument("--pde-weight", type=float, default=1.0)
    parser.add_argument("--data-weight", type=float, default=10.0)
    parser.add_argument("--boundary-weight", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-evaluate", action="store_true")
    return parser


def main() -> None:
    run_empirical_ablation(build_parser().parse_args())


if __name__ == "__main__":
    main()
