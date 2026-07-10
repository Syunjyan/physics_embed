from __future__ import annotations

from pathlib import Path
import argparse
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments._reduced_model_ablation_common import run_reduced_model_ablation


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="linear_elasticity_uniaxial reduced-model ablation.")
    parser.add_argument("--equation", default="linear_elasticity_uniaxial")
    parser.add_argument("--dataset", default="data/linear_elasticity_uniaxial.npz")
    parser.add_argument("--output-root", default="runs/ablation/linear_elasticity_uniaxial_reduced_model")
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--reduced-model", default="linear_elasticity_uniaxial_stress")
    parser.add_argument("--reduced-model-kwargs", default='{"young_modulus": 1.0, "poisson_ratio": 0.3}')
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--samples", type=int, default=2048)
    parser.add_argument("--boundary-samples", type=int, default=1024)
    parser.add_argument("--hidden", type=int, nargs="+", default=[64, 64, 64, 64])
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--reduced-model-weight", type=float, default=3.0)
    parser.add_argument("--pde-weight", type=float, default=1.0)
    parser.add_argument("--data-weight", type=float, default=8.0)
    parser.add_argument("--boundary-weight", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-evaluate", action="store_true")
    return parser


def main() -> None:
    run_reduced_model_ablation(build_parser().parse_args())


if __name__ == "__main__":
    main()
