from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from physics_embed.ns_taylor_green import generate_taylor_green_dataset


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Generate structured analytical Taylor-Green NS dataset.")
    parser.add_argument("--out", type=Path, default=None, help="Output .npz path (default: data/ns_tg_nu_<viscosity>.npz).")
    parser.add_argument("--viscosity", type=float, required=True)
    parser.add_argument("--spatial-resolution", type=int, default=32)
    parser.add_argument("--time-steps", type=int, default=21)
    parser.add_argument("--t-final", type=float, default=1.0)
    args = parser.parse_args()
    out = args.out or Path("data") / f"ns_tg_nu_{args.viscosity:g}.npz"
    meta = generate_taylor_green_dataset(
        out,
        viscosity=args.viscosity,
        spatial_resolution=args.spatial_resolution,
        time_steps=args.time_steps,
        t_final=args.t_final,
    )
    print(meta)


if __name__ == "__main__":
    main()
