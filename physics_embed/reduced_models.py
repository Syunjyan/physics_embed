from __future__ import annotations

from typing import Dict

import torch
from torch import Tensor

from physics_embed.autodiff import column_grad, laplacian

TensorDict = Dict[str, Tensor]


def heat_1d_steady_conduction(
    fields: TensorDict,
    points: Tensor,
    axis: int = 0,
    hot_value: float = 1.0,
    cold_value: float = 0.0,
    length: float = 1.0,
    temperature_key: str = "u",
) -> TensorDict:
    """Source-free 1D steady conduction reduced solution.

    Assumptions: source-free medium, constant conductivity, one-dimensional heat
    transport, fixed temperatures at the two ends. This is a reduced model of
    the heat equation, not Fourier's law itself.
    """
    coord = points[:, axis : axis + 1]
    linear_temperature = hot_value + (cold_value - hot_value) * coord / length
    return {"heat_1d_steady": fields[temperature_key] - linear_temperature}


def burgers_diffusion_dominant(
    fields: TensorDict,
    points: Tensor,
    viscosity: float = 0.01,
) -> TensorDict:
    """Diffusion-dominant Burgers approximation.

    Assumptions: nonlinear advection is small compared with viscosity. The
    Burgers system reduces locally to vector heat equations.
    """
    u, v = fields["u"], fields["v"]
    return {
        "burgers_diffusion_u": column_grad(u, points, 2) - viscosity * laplacian(u, points),
        "burgers_diffusion_v": column_grad(v, points, 2) - viscosity * laplacian(v, points),
    }


def burgers_linearized_advection_diffusion(
    fields: TensorDict,
    points: Tensor,
    base_u: float = 1.0,
    base_v: float = 0.0,
    viscosity: float = 0.01,
) -> TensorDict:
    """Linearized Burgers model around a constant base velocity."""
    u, v = fields["u"], fields["v"]
    return {
        "linear_burgers_u": column_grad(u, points, 2)
        + base_u * column_grad(u, points, 0)
        + base_v * column_grad(u, points, 1)
        - viscosity * laplacian(u, points),
        "linear_burgers_v": column_grad(v, points, 2)
        + base_u * column_grad(v, points, 0)
        + base_v * column_grad(v, points, 1)
        - viscosity * laplacian(v, points),
    }


def navier_stokes_stokes_limit(
    fields: TensorDict,
    points: Tensor,
    viscosity: float = 0.01,
) -> TensorDict:
    """Stokes-flow reduction of incompressible Navier-Stokes.

    Assumptions: low Reynolds number, negligible inertia, steady or slowly
    varying flow. Pressure gradient balances viscous diffusion.
    """
    u, v, p = fields["u"], fields["v"], fields["p"]
    return {
        "stokes_x": column_grad(p, points, 0) - viscosity * laplacian(u, points),
        "stokes_y": column_grad(p, points, 1) - viscosity * laplacian(v, points),
    }


def navier_stokes_poiseuille_channel(
    fields: TensorDict,
    points: Tensor,
    centerline_velocity: float = 1.0,
    channel_height: float = 1.0,
    y_min: float = 0.0,
) -> TensorDict:
    """Planar Poiseuille reduced solution.

    Assumptions: steady, fully developed, laminar channel flow between parallel
    plates. This should only be applied to channel-like regions.
    """
    y = points[:, 1:2] - y_min
    profile = 4.0 * centerline_velocity * y * (channel_height - y) / channel_height**2
    return {
        "poiseuille_u": fields["u"] - profile,
        "poiseuille_v": fields["v"],
    }


def navier_stokes_bernoulli_inviscid(
    fields: TensorDict,
    _points: Tensor,
    density: float = 1.0,
    constant: float | None = None,
) -> TensorDict:
    """Bernoulli reduction for steady inviscid incompressible flow."""
    head = fields["p"] + 0.5 * density * (fields["u"] ** 2 + fields["v"] ** 2)
    target = head.detach().mean() if constant is None else torch.as_tensor(constant, device=head.device, dtype=head.dtype)
    return {"bernoulli_inviscid": head - target}


def linear_elasticity_uniaxial_stress(
    fields: TensorDict,
    points: Tensor,
    young_modulus: float = 1.0,
    poisson_ratio: float = 0.3,
) -> TensorDict:
    """Uniaxial stress reduction of 2D linear elasticity.

    Assumptions: slender specimen, uniaxial tension/compression, far from grips
    and stress concentrations.
    """
    ux, uy = fields["ux"], fields["uy"]
    exx = column_grad(ux, points, 0)
    eyy = column_grad(uy, points, 1)
    return {
        "uniaxial_sigma_x": fields["sigmaxx"] - young_modulus * exx,
        "uniaxial_poisson": eyy + poisson_ratio * exx,
        "uniaxial_sigma_y": fields["sigmayy"],
        "uniaxial_shear": fields["sigmaxy"],
    }


REDUCED_MODELS = {
    "heat_1d_steady_conduction": heat_1d_steady_conduction,
    "burgers_diffusion_dominant": burgers_diffusion_dominant,
    "burgers_linearized_advection_diffusion": burgers_linearized_advection_diffusion,
    "navier_stokes_stokes_limit": navier_stokes_stokes_limit,
    "navier_stokes_poiseuille_channel": navier_stokes_poiseuille_channel,
    "navier_stokes_bernoulli_inviscid": navier_stokes_bernoulli_inviscid,
    "linear_elasticity_uniaxial_stress": linear_elasticity_uniaxial_stress,
}
