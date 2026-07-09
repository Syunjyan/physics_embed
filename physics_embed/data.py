from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from physics_embed.equations import EQUATIONS, get_equation


def generate_dataset(
    equation_name: str,
    out: Path,
    spatial_resolution: int,
    time_steps: int,
) -> None:
    equation = get_equation(equation_name)
    points = equation.make_points(
        spatial_resolution=spatial_resolution,
        time_steps=time_steps,
    )
    fields = equation.dataset_fields(points)
    out.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "points": points.detach().cpu().numpy(),
        "equation": np.array(equation.name),
    }
    payload.update({key: value.numpy() for key, value in fields.items()})
    np.savez_compressed(out, **payload)
    print(f"saved {equation.name} dataset: {out} ({points.shape[0]} points)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate 2D PDE datasets with manufactured solutions.")
    parser.add_argument("--equation", required=True, choices=sorted(EQUATIONS))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--spatial-resolution", type=int, default=64)
    parser.add_argument("--time-steps", type=int, default=21)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    generate_dataset(args.equation, args.out, args.spatial_resolution, args.time_steps)


if __name__ == "__main__":
    main()
