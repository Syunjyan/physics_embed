from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Mapping

import torch
from torch import Tensor, nn

TensorDict = Dict[str, Tensor]
Predictor = Callable[[nn.Module, Tensor], TensorDict]


@dataclass
class SymmetryResidual:
    """Soft residual for known reflection, rotation, or periodic symmetries."""

    point_transform: Callable[[Tensor], Tensor]
    output_parity: Mapping[str, float]
    weight: float = 1.0

    def __call__(self, model: nn.Module, points: Tensor, predictor: Predictor) -> Tensor:
        original = predictor(model, points)
        transformed = predictor(model, self.point_transform(points))
        loss = torch.zeros((), device=points.device)
        for key, parity in self.output_parity.items():
            loss = loss + torch.mean((original[key] - parity * transformed[key]) ** 2)
        return self.weight * loss


@dataclass
class InfinitesimalSymmetryResidual:
    """Lie-symmetry residual from an infinitesimal generator."""

    generator: Callable[[TensorDict, Tensor], TensorDict]
    weight: float = 1.0

    def __call__(self, model: nn.Module, points: Tensor, predictor: Predictor) -> Tensor:
        predictions = predictor(model, points)
        residuals = self.generator(predictions, points)
        return self.weight * sum(torch.mean(value**2) for value in residuals.values())


@dataclass
class MirrorFeatureMap:
    """Hard even-symmetry feature map around a coordinate center."""

    dim: int
    center: float = 0.5

    def __call__(self, points: Tensor) -> Tensor:
        mapped = points.clone()
        mapped[:, self.dim : self.dim + 1] = (points[:, self.dim : self.dim + 1] - self.center) ** 2
        return mapped


@dataclass
class PeriodicFeatureMap:
    """Hard periodic feature map using sine/cosine coordinates."""

    dim: int
    period: float = 1.0

    def __call__(self, points: Tensor) -> Tensor:
        angle = 2.0 * torch.pi * points[:, self.dim : self.dim + 1] / self.period
        return torch.cat([points, torch.sin(angle), torch.cos(angle)], dim=1)


@dataclass
class EmpiricalFormulaResidual:
    """Soft residual for empirical or simplified engineering formulas."""

    formula: Callable[[TensorDict, Tensor], TensorDict]
    weight: float = 1.0

    def __call__(self, model: nn.Module, points: Tensor, predictor: Predictor) -> Tensor:
        predictions = predictor(model, points)
        residuals = self.formula(predictions, points)
        return self.weight * sum(torch.mean(value**2) for value in residuals.values())


@dataclass
class BoxDirichletTransform:
    """Hard Dirichlet output transform for rectangular [0, 1]^d domains."""

    boundary_extension: Callable[[Tensor], Tensor]
    spatial_dims: tuple[int, ...] = (0, 1)

    def bubble(self, points: Tensor) -> Tensor:
        value = torch.ones((points.shape[0], 1), dtype=points.dtype, device=points.device)
        for dim in self.spatial_dims:
            coord = points[:, dim : dim + 1]
            value = value * coord * (1.0 - coord)
        return value

    def __call__(self, raw_output: Tensor, points: Tensor) -> Tensor:
        return self.boundary_extension(points) + self.bubble(points) * raw_output
