from __future__ import annotations

from typing import Dict

import torch
from torch import Tensor

from physics_embed.autodiff import column_grad, laplacian

TensorDict = Dict[str, Tensor]


def heat_fourier_law(
    fields: TensorDict,
    points: Tensor,
    conductivity: float = 1.0,
    temperature_key: str = "u",
    flux_x_key: str = "qx",
    flux_y_key: str = "qy",
) -> TensorDict:
    """Fourier law residual q + k*grad(T)=0 when heat flux is predicted."""
    if flux_x_key not in fields or flux_y_key not in fields:
        return {}
    temperature = fields[temperature_key]
    return {
        "fourier_qx": fields[flux_x_key] + conductivity * column_grad(temperature, points, 0),
        "fourier_qy": fields[flux_y_key] + conductivity * column_grad(temperature, points, 1),
    }


def heat_robin_boundary(
    fields: TensorDict,
    points: Tensor,
    normal: tuple[float, float],
    conductivity: float = 1.0,
    h: float = 1.0,
    ambient: float = 0.0,
    temperature_key: str = "u",
) -> TensorDict:
    """Convective heat-transfer boundary: -k*grad(T).n = h*(T-T_inf)."""
    temperature = fields[temperature_key]
    normal_x, normal_y = normal
    normal_flux = -conductivity * (
        normal_x * column_grad(temperature, points, 0)
        + normal_y * column_grad(temperature, points, 1)
    )
    return {"heat_robin": normal_flux - h * (temperature - ambient)}


def heat_thermal_resistance(
    fields: TensorDict,
    _points: Tensor,
    heat_rate: float,
    resistance: float,
    hot_temperature: float,
    cold_temperature: float,
) -> TensorDict:
    """Lumped heat-resistance relation Q = DeltaT / R_th."""
    residual = torch.as_tensor(
        heat_rate - (hot_temperature - cold_temperature) / resistance,
        dtype=next(iter(fields.values())).dtype,
        device=next(iter(fields.values())).device,
    )
    return {"thermal_resistance": residual.reshape(1, 1)}


def burgers_diffusion_dominant(
    fields: TensorDict,
    points: Tensor,
    viscosity: float = 0.01,
) -> TensorDict:
    """Low-Re Burgers approximation: time diffusion dominates nonlinear convection."""
    u, v = fields["u"], fields["v"]
    return {
        "burgers_diffusion_u": column_grad(u, points, 2) - viscosity * laplacian(u, points),
        "burgers_diffusion_v": column_grad(v, points, 2) - viscosity * laplacian(v, points),
    }


def burgers_energy_decay(
    fields: TensorDict,
    points: Tensor,
    tolerance: float = 0.0,
) -> TensorDict:
    """Pointwise proxy for kinetic-energy decay in viscous Burgers flows."""
    kinetic = 0.5 * (fields["u"] ** 2 + fields["v"] ** 2)
    return {"burgers_energy_growth": torch.relu(column_grad(kinetic, points, 2) - tolerance)}


def navier_stokes_poiseuille(
    fields: TensorDict,
    points: Tensor,
    centerline_velocity: float = 1.0,
    channel_height: float = 1.0,
    y_min: float = 0.0,
) -> TensorDict:
    """Planar Poiseuille profile for fully developed laminar channel flow."""
    y = points[:, 1:2] - y_min
    profile = 4.0 * centerline_velocity * y * (channel_height - y) / channel_height**2
    return {
        "poiseuille_u": fields["u"] - profile,
        "poiseuille_v": fields["v"],
    }


def navier_stokes_bernoulli(
    fields: TensorDict,
    _points: Tensor,
    density: float = 1.0,
    constant: float | None = None,
) -> TensorDict:
    """Bernoulli invariant for steady inviscid regions."""
    head = fields["p"] + 0.5 * density * (fields["u"] ** 2 + fields["v"] ** 2)
    target = head.detach().mean() if constant is None else torch.as_tensor(constant, device=head.device, dtype=head.dtype)
    return {"bernoulli": head - target}


def navier_stokes_stokes_limit(
    fields: TensorDict,
    points: Tensor,
    viscosity: float = 0.01,
) -> TensorDict:
    """Low-Re Stokes approximation: pressure gradient balances viscosity."""
    u, v, p = fields["u"], fields["v"], fields["p"]
    return {
        "stokes_x": column_grad(p, points, 0) - viscosity * laplacian(u, points),
        "stokes_y": column_grad(p, points, 1) - viscosity * laplacian(v, points),
    }


def linear_elasticity_hooke(
    fields: TensorDict,
    points: Tensor,
    lambda_: float = 1.0,
    mu: float = 0.5,
) -> TensorDict:
    """Isotropic small-strain Hooke law residual."""
    ux, uy = fields["ux"], fields["uy"]
    exx = column_grad(ux, points, 0)
    eyy = column_grad(uy, points, 1)
    exy = 0.5 * (column_grad(ux, points, 1) + column_grad(uy, points, 0))
    return {
        "hooke_xx": fields["sigmaxx"] - ((lambda_ + 2.0 * mu) * exx + lambda_ * eyy),
        "hooke_yy": fields["sigmayy"] - ((lambda_ + 2.0 * mu) * eyy + lambda_ * exx),
        "hooke_xy": fields["sigmaxy"] - 2.0 * mu * exy,
    }


def linear_elasticity_uniaxial_stress(
    fields: TensorDict,
    points: Tensor,
    young_modulus: float = 1.0,
    poisson_ratio: float = 0.3,
) -> TensorDict:
    """Uniaxial stress approximation: sigma_x=E*epsilon_x, epsilon_y=-nu*epsilon_x."""
    ux, uy = fields["ux"], fields["uy"]
    exx = column_grad(ux, points, 0)
    eyy = column_grad(uy, points, 1)
    return {
        "uniaxial_sigma": fields["sigmaxx"] - young_modulus * exx,
        "uniaxial_poisson": eyy + poisson_ratio * exx,
        "uniaxial_sigma_yy": fields["sigmayy"],
        "uniaxial_shear": fields["sigmaxy"],
    }


EMPIRICAL_FORMULAS = {
    "heat_fourier_law": heat_fourier_law,
    "heat_robin_boundary": heat_robin_boundary,
    "heat_thermal_resistance": heat_thermal_resistance,
    "burgers_diffusion_dominant": burgers_diffusion_dominant,
    "burgers_energy_decay": burgers_energy_decay,
    "navier_stokes_poiseuille": navier_stokes_poiseuille,
    "navier_stokes_bernoulli": navier_stokes_bernoulli,
    "navier_stokes_stokes_limit": navier_stokes_stokes_limit,
    "linear_elasticity_hooke": linear_elasticity_hooke,
    "linear_elasticity_uniaxial_stress": linear_elasticity_uniaxial_stress,
}
