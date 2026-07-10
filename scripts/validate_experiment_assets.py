from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from physics_embed.equations import NavierStokesTaylorGreen2D, get_equation
from physics_embed.reduced_models import linear_elasticity_uniaxial_stress


REQUIRED_DATASETS = (
    "heat.npz",
    "burgers.npz",
    "navier_stokes.npz",
    "linear_elasticity.npz",
    "linear_elasticity_uniaxial.npz",
    "linear_elasticity_mirror.npz",
    "ns_tg_nu_0.005.npz",
    "ns_tg_nu_0.01.npz",
    "ns_tg_nu_0.02.npz",
)


def _max_abs(actual: np.ndarray, expected: np.ndarray) -> float:
    return float(np.max(np.abs(actual.astype(np.float64) - expected.astype(np.float64))))


def _validate_dataset(path: Path, tolerance: float) -> dict[str, object]:
    with np.load(path) as data:
        if "points" not in data or "equation" not in data:
            raise ValueError(f"{path}: missing points or equation")
        for key in data.files:
            if np.issubdtype(data[key].dtype, np.number) and not np.isfinite(data[key]).all():
                raise ValueError(f"{path}: non-finite values in {key}")

        equation_name = str(data["equation"])
        if equation_name == "navier_stokes_taylor_green":
            viscosity = float(data["viscosity_value"])
            equation = NavierStokesTaylorGreen2D(viscosity=viscosity)
        else:
            equation = get_equation(equation_name)

        points = torch.tensor(data["points"], dtype=torch.float32).requires_grad_(True)
        exact = equation.exact(points)
        field_errors = {}
        for key in equation.supervised_keys:
            if key not in data:
                raise ValueError(f"{path}: missing supervised field {key}")
            error = _max_abs(data[key], exact[key].detach().numpy())
            field_errors[key] = error
            if error > tolerance:
                raise ValueError(f"{path}: {key} differs from declared exact field by {error:.3e}")

        if equation_name == "navier_stokes_taylor_green":
            fields_st = data["fields_st"]
            expected_shape = (
                1,
                int(data["time_steps"]),
                int(data["spatial_resolution"]),
                int(data["spatial_resolution"]),
                3,
            )
            if fields_st.shape != expected_shape:
                raise ValueError(f"{path}: fields_st shape {fields_st.shape}, expected {expected_shape}")
            for channel, key in enumerate(("u", "v", "p")):
                error = _max_abs(fields_st[0, ..., channel].reshape(-1, 1), data[key])
                if error > tolerance:
                    raise ValueError(f"{path}: structured {key} and point cloud differ by {error:.3e}")

        return {
            "equation": equation_name,
            "points": int(data["points"].shape[0]),
            "max_field_error": max(field_errors.values(), default=0.0),
        }


def _validate_uniaxial_reduction(data_root: Path, tolerance: float) -> float:
    equation = get_equation("linear_elasticity_uniaxial")
    with np.load(data_root / "linear_elasticity_uniaxial.npz") as data:
        points = torch.tensor(data["points"], dtype=torch.float32).requires_grad_(True)
        fields = equation.exact(points)
        residuals = linear_elasticity_uniaxial_stress(
            fields,
            points,
            young_modulus=equation.young_modulus,
            poisson_ratio=equation.poisson_ratio,
        )
        max_residual = max(float(value.detach().abs().max()) for value in residuals.values())
    if max_residual > tolerance:
        raise ValueError(f"uniaxial reduced-model assumptions are not satisfied: {max_residual:.3e}")
    return max_residual


def _validate_mirror_symmetry(data_root: Path, tolerance: float) -> float:
    equation = get_equation("linear_elasticity_mirror")
    with np.load(data_root / "linear_elasticity_mirror.npz") as data:
        points = torch.tensor(data["points"], dtype=torch.float32).requires_grad_(True)
        mirrored = points.detach().clone()
        mirrored[:, 0] = 1.0 - mirrored[:, 0]
        mirrored.requires_grad_(True)
        fields = equation.exact(points)
        mirrored_fields = equation.exact(mirrored)
        parity = {"ux": -1.0, "uy": 1.0, "sigmaxx": 1.0, "sigmayy": 1.0, "sigmaxy": -1.0}
        max_error = max(
            float((fields[key] - sign * mirrored_fields[key]).detach().abs().max())
            for key, sign in parity.items()
        )
    if max_error > tolerance:
        raise ValueError(f"mirror dataset violates declared parity: {max_error:.3e}")
    return max_error


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate datasets and assumptions used by reported experiments.")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--tolerance", type=float, default=2e-5)
    args = parser.parse_args()

    missing = [name for name in REQUIRED_DATASETS if not (args.data_root / name).exists()]
    if missing:
        raise FileNotFoundError(f"missing required datasets: {', '.join(missing)}")

    report = {
        "datasets": {
            name: _validate_dataset(args.data_root / name, args.tolerance)
            for name in REQUIRED_DATASETS
        },
        "assumptions": {
            "uniaxial_reduced_model_max_residual": _validate_uniaxial_reduction(
                args.data_root, args.tolerance
            ),
            "mirror_parity_max_error": _validate_mirror_symmetry(args.data_root, args.tolerance),
        },
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
