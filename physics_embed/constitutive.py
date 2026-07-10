from __future__ import annotations

from typing import Dict

from torch import Tensor

from physics_embed.autodiff import column_grad

TensorDict = Dict[str, Tensor]


def linear_elasticity_hooke_residual(
    fields: TensorDict,
    points: Tensor,
    lambda_: float = 1.0,
    mu: float = 0.5,
) -> TensorDict:
    """Hooke constitutive residual for isotropic small-strain linear elasticity.

    This is not an empirical formula in the reduced-model sense. When stress is
    predicted independently from displacement, this residual is part of the full
    linear-elasticity PINN physics loss.
    """
    ux, uy = fields["ux"], fields["uy"]
    exx = column_grad(ux, points, 0)
    eyy = column_grad(uy, points, 1)
    exy = 0.5 * (column_grad(ux, points, 1) + column_grad(uy, points, 0))
    return {
        "hooke_xx": fields["sigmaxx"] - ((lambda_ + 2.0 * mu) * exx + lambda_ * eyy),
        "hooke_yy": fields["sigmayy"] - ((lambda_ + 2.0 * mu) * eyy + lambda_ * exx),
        "hooke_xy": fields["sigmaxy"] - 2.0 * mu * exy,
    }

