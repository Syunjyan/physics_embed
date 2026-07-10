"""Taylor-Green vortex datasets with a PDEBench-compatible tensor layout.

PDEBench incompressible-flow datasets are produced by numerical solvers. This
module does not reproduce that solver pipeline. It uses the analytical
Taylor-Green solution and exports the same kind of structured spatiotemporal
tensor, ``[sample, time, x, y, variable]``, together with point-cloud arrays
used by the experiments.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class TaylorGreenConfig:
    viscosity: float
    spatial_resolution: int
    time_steps: int
    t_final: float = 1.0


def _taylor_green_exact(
    x: np.ndarray,
    y: np.ndarray,
    t: float,
    viscosity: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    k = 2.0 * np.pi
    velocity_decay = np.exp(-2.0 * viscosity * k**2 * t)
    pressure_decay = np.exp(-4.0 * viscosity * k**2 * t)
    u = -np.cos(k * x) * np.sin(k * y) * velocity_decay
    v = np.sin(k * x) * np.cos(k * y) * velocity_decay
    p = -0.25 * (np.cos(2.0 * k * x) + np.cos(2.0 * k * y)) * pressure_decay
    return u, v, p


def generate_structured_fields(config: TaylorGreenConfig) -> dict[str, np.ndarray]:
    nx = ny = config.spatial_resolution
    nt = config.time_steps
    x = np.linspace(0.0, 1.0, nx, endpoint=False)
    y = np.linspace(0.0, 1.0, ny, endpoint=False)
    times = np.linspace(0.0, config.t_final, nt)
    xx, yy = np.meshgrid(x, y, indexing="ij")

    fields = np.zeros((1, nt, nx, ny, 3), dtype=np.float32)
    for step, time in enumerate(times):
        u, v, p = _taylor_green_exact(xx, yy, float(time), config.viscosity)
        fields[0, step, ..., 0] = u
        fields[0, step, ..., 1] = v
        fields[0, step, ..., 2] = p

    output_dt = times[1] - times[0] if nt > 1 else config.t_final
    return {
        "fields_st": fields,
        "times": times.astype(np.float32),
        "x": x.astype(np.float32),
        "y": y.astype(np.float32),
        "viscosity_value": np.array(config.viscosity, dtype=np.float32),
        "spatial_resolution": np.array(nx, dtype=np.int32),
        "time_steps": np.array(nt, dtype=np.int32),
        "dt": np.array(output_dt, dtype=np.float32),
        "generation_method": np.array("taylor_green_analytical_structured_layout"),
    }


def structured_to_pointcloud(payload: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    fields_st = payload["fields_st"][0]
    times = payload["times"]
    x = payload["x"]
    y = payload["y"]
    tt, xx, yy = np.meshgrid(times, x, y, indexing="ij")

    points = np.stack([xx.reshape(-1), yy.reshape(-1), tt.reshape(-1)], axis=1).astype(np.float32)
    u = fields_st[..., 0].reshape(-1, 1).astype(np.float32)
    v = fields_st[..., 1].reshape(-1, 1).astype(np.float32)
    p = fields_st[..., 2].reshape(-1, 1).astype(np.float32)

    k = 2.0 * np.pi
    viscosity = float(payload["viscosity_value"])
    psi = np.cos(k * xx) * np.cos(k * yy) * np.exp(-2.0 * viscosity * k**2 * tt) / k
    return {
        "points": points,
        "u": u,
        "v": v,
        "p": p,
        "psi": psi.reshape(-1, 1).astype(np.float32),
        "fu": np.zeros_like(u),
        "fv": np.zeros_like(v),
    }


def generate_taylor_green_dataset(
    out_path: str | Path,
    viscosity: float,
    spatial_resolution: int,
    time_steps: int,
    t_final: float = 1.0,
) -> dict[str, float | int | str]:
    config = TaylorGreenConfig(
        viscosity=viscosity,
        spatial_resolution=spatial_resolution,
        time_steps=time_steps,
        t_final=t_final,
    )
    structured = generate_structured_fields(config)
    pointcloud = structured_to_pointcloud(structured)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        **structured,
        **pointcloud,
        equation=np.array("navier_stokes_taylor_green"),
    )
    return {
        "viscosity": viscosity,
        "points": int(pointcloud["points"].shape[0]),
        "generation_method": str(structured["generation_method"]),
    }
