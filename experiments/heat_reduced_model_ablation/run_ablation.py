from __future__ import annotations

from pathlib import Path
import argparse
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments._reduced_model_ablation_common import run_reduced_model_ablation


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="heat reduced-model ablation.")
    parser.add_argument("--equation", default="heat")
    parser.add_argument("--dataset", default="data/heat.npz")
    parser.add_argument("--output-root", default="runs/ablation/heat_reduced_model")
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--reduced-model", default="heat_1d_steady_conduction")
    parser.add_argument("--reduced-model-kwargs", default='{"hot_value": 0.0, "cold_value": 0.0}')
    parser.add_argument("--epochs", type=int, default=3000)
    parser.add_argument("--samples", type=int, default=2048)
    parser.add_argument("--boundary-samples", type=int, default=512)
    parser.add_argument("--hidden", type=int, nargs="+", default=[64, 64, 64, 64])
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--reduced-model-weight", type=float, default=0.1)
    parser.add_argument("--pde-weight", type=float, default=1.0)
    parser.add_argument("--data-weight", type=float, default=5.0)
    parser.add_argument("--boundary-weight", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-evaluate", action="store_true")
    return parser


def main() -> None:
    run_reduced_model_ablation(build_parser().parse_args())


if __name__ == "__main__":
    main()
