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
    viscosity: float | None = None,
    t_final: float = 1.0,
) -> None:
    if equation_name == "navier_stokes_taylor_green":
        if viscosity is None:
            raise ValueError("navier_stokes_taylor_green requires --viscosity")
        from physics_embed.ns_taylor_green import generate_taylor_green_dataset

        meta = generate_taylor_green_dataset(
            out,
            viscosity=viscosity,
            spatial_resolution=spatial_resolution,
            time_steps=time_steps,
            t_final=t_final,
        )
        print(
            f"saved structured {equation_name} dataset: {out} "
            f"(nu={viscosity}, points={meta['points']})"
        )
        return

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
    parser.add_argument("--viscosity", type=float, default=None, help="Required for navier_stokes_taylor_green.")
    parser.add_argument("--t-final", type=float, default=1.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    generate_dataset(
        args.equation,
        args.out,
        args.spatial_resolution,
        args.time_steps,
        viscosity=args.viscosity,
        t_final=args.t_final,
    )


if __name__ == "__main__":
    main()
