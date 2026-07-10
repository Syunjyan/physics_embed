from __future__ import annotations

from typing import Dict, Mapping

import torch
from torch import Tensor, nn

TensorDict = Dict[str, Tensor]


def mirror_points(points: Tensor, dim: int = 0, center: float = 0.5) -> Tensor:
    """Mirror points with respect to x_dim=center."""
    mirrored = points.clone()
    mirrored[:, dim : dim + 1] = 2.0 * center - points[:, dim : dim + 1]
    return mirrored


def mirror_symmetry_projection(
    model: nn.Module,
    points: Tensor,
    predictor,
    parity: Mapping[str, float],
    dim: int = 0,
    center: float = 0.5,
) -> TensorDict:
    """Project predictions onto a hard mirror-symmetric/antisymmetric space.

    For parity +1, q(x)=q(Tx). For parity -1, q(x)=-q(Tx). The projected value
    at x is 0.5 * (q(x) + parity * q(Tx)).
    """
    raw = predictor(model, points)
    mirrored = predictor(model, mirror_points(points, dim=dim, center=center))
    projected = {}
    for key, value in raw.items():
        if key in parity:
            projected[key] = 0.5 * (value + parity[key] * mirrored[key])
        else:
            projected[key] = value
    return projected

