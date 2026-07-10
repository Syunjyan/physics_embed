from __future__ import annotations

from dataclasses import dataclass
from math import pi
from typing import Callable, Dict, Iterable

import torch
from torch import Tensor, nn

from physics_embed.autodiff import column_grad, laplacian

TensorDict = Dict[str, Tensor]


@dataclass
class BaseEquation:
    name: str
    input_dim: int
    raw_output_dim: int
    supervised_keys: Iterable[str]
    has_time: bool = False

    def make_points(
        self,
        spatial_resolution: int,
        time_steps: int = 1,
        device: str | torch.device = "cpu",
    ) -> Tensor:
        line = torch.linspace(0.0, 1.0, spatial_resolution, device=device)
        if self.has_time:
            time = torch.linspace(0.0, 1.0, time_steps, device=device)
            xx, yy, tt = torch.meshgrid(line, line, time, indexing="ij")
            return torch.stack([xx.reshape(-1), yy.reshape(-1), tt.reshape(-1)], dim=1)
        xx, yy = torch.meshgrid(line, line, indexing="ij")
        return torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=1)

    def exact(self, points: Tensor) -> TensorDict:
        raise NotImplementedError

    def sources(self, points: Tensor) -> TensorDict:
        return {}

    def dataset_fields(self, points: Tensor) -> TensorDict:
        calc_points = points.detach().clone().requires_grad_(True)
        fields = {**self.exact(calc_points), **self.sources(calc_points)}
        return {key: value.detach().cpu() for key, value in fields.items()}

    def prediction(self, model: nn.Module, points: Tensor) -> TensorDict:
        raw = model(points)
        return {key: raw[:, idx : idx + 1] for idx, key in enumerate(self.supervised_keys)}

    def pde_residuals(self, model: nn.Module, points: Tensor) -> TensorDict:
        raise NotImplementedError

    def boundary_loss(self, model: nn.Module, count: int, device: torch.device) -> Tensor:
        return torch.zeros((), device=device)


class HeatConduction2D(BaseEquation):
    def __init__(self) -> None:
        super().__init__("heat", input_dim=2, raw_output_dim=1, supervised_keys=("u",))

    def exact(self, points: Tensor) -> TensorDict:
        x, y = points[:, 0:1], points[:, 1:2]
        return {"u": torch.sin(torch.pi * x) * torch.sin(torch.pi * y)}

    def sources(self, points: Tensor) -> TensorDict:
        return {"f": 2.0 * torch.pi**2 * self.exact(points)["u"]}

    def pde_residuals(self, model: nn.Module, points: Tensor) -> TensorDict:
        u = self.prediction(model, points)["u"]
        residual = -laplacian(u, points) - self.sources(points)["f"].detach()
        return {"heat_pde": residual}

    def boundary_loss(self, model: nn.Module, count: int, device: torch.device) -> Tensor:
        if count <= 0:
            return torch.zeros((), device=device)
        s = torch.rand((count, 1), device=device)
        zeros, ones = torch.zeros_like(s), torch.ones_like(s)
        dirichlet = torch.cat(
            [torch.cat([zeros, s], dim=1), torch.cat([ones, s], dim=1)],
            dim=0,
        )
        u_pred = self.prediction(model, dirichlet)["u"]
        u_true = self.exact(dirichlet)["u"].detach()

        bottom = torch.cat([s, zeros], dim=1).requires_grad_(True)
        top = torch.cat([s, ones], dim=1).requires_grad_(True)
        u_bottom = self.prediction(model, bottom)["u"]
        u_top = self.prediction(model, top)["u"]
        flux_bottom = column_grad(u_bottom, bottom, 1)
        flux_top = column_grad(u_top, top, 1)
        target_bottom = torch.pi * torch.sin(torch.pi * bottom[:, 0:1])
        target_top = -torch.pi * torch.sin(torch.pi * top[:, 0:1])
        return (
            torch.mean((u_pred - u_true) ** 2)
            + torch.mean((flux_bottom - target_bottom) ** 2)
            + torch.mean((flux_top - target_top) ** 2)
        )


class Burgers2D(BaseEquation):
    def __init__(self, viscosity: float = 0.01) -> None:
        super().__init__(
            "burgers",
            input_dim=3,
            raw_output_dim=2,
            supervised_keys=("u", "v"),
            has_time=True,
        )
        self.viscosity = viscosity

    def exact(self, points: Tensor) -> TensorDict:
        x, y, t = points[:, 0:1], points[:, 1:2], points[:, 2:3]
        decay = torch.exp(-t)
        return {
            "u": torch.sin(torch.pi * x) * torch.sin(torch.pi * y) * decay,
            "v": torch.cos(torch.pi * x) * torch.sin(torch.pi * y) * decay,
        }

    def _lhs(self, points: Tensor, fields: TensorDict) -> TensorDict:
        u, v = fields["u"], fields["v"]
        u_t = column_grad(u, points, 2)
        v_t = column_grad(v, points, 2)
        u_x = column_grad(u, points, 0)
        u_y = column_grad(u, points, 1)
        v_x = column_grad(v, points, 0)
        v_y = column_grad(v, points, 1)
        return {
            "fu": u_t + u * u_x + v * u_y - self.viscosity * laplacian(u, points),
            "fv": v_t + u * v_x + v * v_y - self.viscosity * laplacian(v, points),
        }

    def sources(self, points: Tensor) -> TensorDict:
        return self._lhs(points, self.exact(points))

    def pde_residuals(self, model: nn.Module, points: Tensor) -> TensorDict:
        lhs = self._lhs(points, self.prediction(model, points))
        source = self.sources(points)
        return {
            "burgers_u": lhs["fu"] - source["fu"].detach(),
            "burgers_v": lhs["fv"] - source["fv"].detach(),
        }

    def boundary_loss(self, model: nn.Module, count: int, device: torch.device) -> Tensor:
        if count <= 0:
            return torch.zeros((), device=device)
        xyt = torch.rand((count, 3), device=device)
        initial = xyt.clone()
        initial[:, 2] = 0.0
        side = xyt.clone()
        side[:, 0] = torch.round(side[:, 0])
        pts = torch.cat([initial, side], dim=0).requires_grad_(True)
        pred = self.prediction(model, pts)
        true = self.exact(pts)
        return sum(torch.mean((pred[key] - true[key].detach()) ** 2) for key in self.supervised_keys)


class NavierStokes2D(BaseEquation):
    def __init__(self, viscosity: float = 0.01) -> None:
        super().__init__(
            "navier_stokes",
            input_dim=3,
            raw_output_dim=2,
            supervised_keys=("u", "v", "p"),
            has_time=True,
        )
        self.viscosity = viscosity

    def exact_psi_p(self, points: Tensor) -> TensorDict:
        x, y, t = points[:, 0:1], points[:, 1:2], points[:, 2:3]
        decay = torch.exp(-t)
        return {
            "psi": torch.sin(torch.pi * x) * torch.sin(torch.pi * y) * decay,
            "p": torch.sin(torch.pi * x) * torch.cos(torch.pi * y) * decay,
        }

    def exact(self, points: Tensor) -> TensorDict:
        fields = self.exact_psi_p(points)
        psi = fields["psi"]
        fields["u"] = column_grad(psi, points, 1)
        fields["v"] = -column_grad(psi, points, 0)
        return fields

    def prediction(self, model: nn.Module, points: Tensor) -> TensorDict:
        raw = model(points)
        psi = raw[:, 0:1]
        return {
            "psi": psi,
            "p": raw[:, 1:2],
            "u": column_grad(psi, points, 1),
            "v": -column_grad(psi, points, 0),
        }

    def _momentum_lhs(self, points: Tensor, fields: TensorDict) -> TensorDict:
        u, v, p = fields["u"], fields["v"], fields["p"]
        u_t = column_grad(u, points, 2)
        v_t = column_grad(v, points, 2)
        u_x = column_grad(u, points, 0)
        u_y = column_grad(u, points, 1)
        v_x = column_grad(v, points, 0)
        v_y = column_grad(v, points, 1)
        p_x = column_grad(p, points, 0)
        p_y = column_grad(p, points, 1)
        return {
            "fu": u_t + u * u_x + v * u_y + p_x - self.viscosity * laplacian(u, points),
            "fv": v_t + u * v_x + v * v_y + p_y - self.viscosity * laplacian(v, points),
        }

    def sources(self, points: Tensor) -> TensorDict:
        return self._momentum_lhs(points, self.exact(points))

    def pde_residuals(self, model: nn.Module, points: Tensor) -> TensorDict:
        lhs = self._momentum_lhs(points, self.prediction(model, points))
        source = self.sources(points)
        return {
            "ns_u": lhs["fu"] - source["fu"].detach(),
            "ns_v": lhs["fv"] - source["fv"].detach(),
        }

    def boundary_loss(self, model: nn.Module, count: int, device: torch.device) -> Tensor:
        if count <= 0:
            return torch.zeros((), device=device)
        pts = torch.rand((count, 3), device=device)
        initial = pts.clone()
        initial[:, 2] = 0.0
        side = pts.clone()
        side[:, 0] = torch.round(side[:, 0])
        bc_points = torch.cat([initial, side], dim=0).requires_grad_(True)
        pred = self.prediction(model, bc_points)
        true = self.exact(bc_points)
        return sum(torch.mean((pred[key] - true[key].detach()) ** 2) for key in ("u", "v", "p"))


class NavierStokesTaylorGreen2D(BaseEquation):
    def __init__(self, viscosity: float = 0.01) -> None:
        super().__init__(
            "navier_stokes_taylor_green",
            input_dim=3,
            raw_output_dim=2,
            supervised_keys=("u", "v", "p"),
            has_time=True,
        )
        self.viscosity = viscosity
        self.k = 2.0 * torch.pi

    def exact_psi_p(self, points: Tensor) -> TensorDict:
        x, y, t = points[:, 0:1], points[:, 1:2], points[:, 2:3]
        k = self.k
        velocity_decay = torch.exp(-2.0 * self.viscosity * k**2 * t)
        pressure_decay = torch.exp(-4.0 * self.viscosity * k**2 * t)
        return {
            "psi": torch.cos(k * x) * torch.cos(k * y) * velocity_decay / k,
            "p": -0.25 * (torch.cos(2.0 * k * x) + torch.cos(2.0 * k * y)) * pressure_decay,
        }

    def exact(self, points: Tensor) -> TensorDict:
        fields = self.exact_psi_p(points)
        psi = fields["psi"]
        fields["u"] = column_grad(psi, points, 1)
        fields["v"] = -column_grad(psi, points, 0)
        return fields

    def prediction(self, model: nn.Module, points: Tensor) -> TensorDict:
        raw = model(points)
        psi = raw[:, 0:1]
        return {
            "psi": psi,
            "p": raw[:, 1:2],
            "u": column_grad(psi, points, 1),
            "v": -column_grad(psi, points, 0),
        }

    def pde_residuals(self, model: nn.Module, points: Tensor) -> TensorDict:
        fields = self.prediction(model, points)
        u, v, p = fields["u"], fields["v"], fields["p"]
        u_t = column_grad(u, points, 2)
        v_t = column_grad(v, points, 2)
        u_x = column_grad(u, points, 0)
        u_y = column_grad(u, points, 1)
        v_x = column_grad(v, points, 0)
        v_y = column_grad(v, points, 1)
        p_x = column_grad(p, points, 0)
        p_y = column_grad(p, points, 1)
        return {
            "ns_u": u_t + u * u_x + v * u_y + p_x - self.viscosity * laplacian(u, points),
            "ns_v": v_t + u * v_x + v * v_y + p_y - self.viscosity * laplacian(v, points),
        }

    def sources(self, points: Tensor) -> TensorDict:
        zeros = torch.zeros((points.shape[0], 1), dtype=points.dtype, device=points.device)
        return {"fu": zeros, "fv": zeros, "viscosity": torch.full_like(zeros, self.viscosity)}

    def boundary_loss(self, model: nn.Module, count: int, device: torch.device) -> Tensor:
        if count <= 0:
            return torch.zeros((), device=device)
        pts = torch.rand((count, 3), device=device)
        initial = pts.clone()
        initial[:, 2] = 0.0

        side = pts.clone()
        side[:, 0] = torch.round(side[:, 0])
        horizontal = pts.clone()
        horizontal[:, 1] = torch.round(horizontal[:, 1])
        bc_points = torch.cat([initial, side, horizontal], dim=0).requires_grad_(True)
        pred = self.prediction(model, bc_points)
        true = self.exact(bc_points)
        return sum(torch.mean((pred[key] - true[key].detach()) ** 2) for key in ("u", "v", "p"))


class LinearElasticity2D(BaseEquation):
    def __init__(self, lambda_: float = 1.0, mu: float = 0.5, q: float = 4.0) -> None:
        super().__init__(
            "linear_elasticity",
            input_dim=2,
            raw_output_dim=5,
            supervised_keys=("ux", "uy", "sigmaxx", "sigmayy", "sigmaxy"),
        )
        self.lambda_ = lambda_
        self.mu = mu
        self.q = q

    def body_force(self, points: Tensor) -> TensorDict:
        x, y = points[:, 0:1], points[:, 1:2]
        lam, mu, q = self.lambda_, self.mu, self.q
        force_x = lam * (
            4 * pi**2 * torch.cos(2 * pi * x) * torch.sin(pi * y)
            - pi * torch.cos(pi * x) * q * y**3
        ) + mu * (
            9 * pi**2 * torch.cos(2 * pi * x) * torch.sin(pi * y)
            - pi * torch.cos(pi * x) * q * y**3
        )
        force_y = lam * (
            -3 * torch.sin(pi * x) * q * y**2
            + 2 * pi**2 * torch.sin(2 * pi * x) * torch.cos(pi * y)
        ) + mu * (
            -6 * torch.sin(pi * x) * q * y**2
            + 2 * pi**2 * torch.sin(2 * pi * x) * torch.cos(pi * y)
            + pi**2 * torch.sin(pi * x) * q * y**4 / 4
        )
        return {"fx": force_x, "fy": force_y}

    def exact(self, points: Tensor) -> TensorDict:
        x, y = points[:, 0:1], points[:, 1:2]
        ux = torch.cos(2 * pi * x) * torch.sin(pi * y)
        uy = torch.sin(pi * x) * self.q * y**4 / 4
        exx = column_grad(ux, points, 0)
        eyy = column_grad(uy, points, 1)
        exy = (column_grad(ux, points, 1) + column_grad(uy, points, 0)) / 2
        sigmaxx = (self.lambda_ + 2 * self.mu) * exx + self.lambda_ * eyy
        sigmayy = (self.lambda_ + 2 * self.mu) * eyy + self.lambda_ * exx
        sigmaxy = 2 * self.mu * exy
        return {
            "ux": ux,
            "uy": uy,
            "sigmaxx": sigmaxx,
            "sigmayy": sigmayy,
            "sigmaxy": sigmaxy,
            "exx": exx,
            "eyy": eyy,
            "exy": exy,
        }

    def sources(self, points: Tensor) -> TensorDict:
        return self.body_force(points)

    def pde_residuals(self, model: nn.Module, points: Tensor) -> TensorDict:
        pred = self.prediction(model, points)
        ux, uy = pred["ux"], pred["uy"]
        sigmaxx, sigmayy, sigmaxy = pred["sigmaxx"], pred["sigmayy"], pred["sigmaxy"]
        exx = column_grad(ux, points, 0)
        eyy = column_grad(uy, points, 1)
        exy = (column_grad(ux, points, 1) + column_grad(uy, points, 0)) / 2
        force = self.body_force(points)
        return {
            "balance_x": column_grad(sigmaxx, points, 0)
            + column_grad(sigmaxy, points, 1)
            + force["fx"],
            "balance_y": column_grad(sigmayy, points, 1)
            + column_grad(sigmaxy, points, 0)
            + force["fy"],
            "constitutive_xx": (self.lambda_ + 2 * self.mu) * exx
            + self.lambda_ * eyy
            - sigmaxx,
            "constitutive_yy": (self.lambda_ + 2 * self.mu) * eyy
            + self.lambda_ * exx
            - sigmayy,
            "constitutive_xy": 2 * self.mu * exy - sigmaxy,
        }

    def boundary_loss(self, model: nn.Module, count: int, device: torch.device) -> Tensor:
        if count <= 0:
            return torch.zeros((), device=device)
        s = torch.rand((count, 1), device=device)
        zeros, ones = torch.zeros_like(s), torch.ones_like(s)
        down = torch.cat([s, zeros], dim=1)
        left = torch.cat([zeros, s], dim=1)
        right = torch.cat([ones, s], dim=1)
        up = torch.cat([s, ones], dim=1)
        loss = torch.zeros((), device=device)
        pred_down = self.prediction(model, down)
        loss = loss + torch.mean(pred_down["ux"] ** 2) + torch.mean(pred_down["uy"] ** 2)
        pred_left = self.prediction(model, left)
        pred_right = self.prediction(model, right)
        loss = loss + torch.mean(pred_left["uy"] ** 2) + torch.mean(pred_left["sigmaxx"] ** 2)
        loss = loss + torch.mean(pred_right["uy"] ** 2) + torch.mean(pred_right["sigmaxx"] ** 2)
        pred_up = self.prediction(model, up)
        sigma_yy_up = (self.lambda_ + 2 * self.mu) * self.q * torch.sin(torch.pi * up[:, 0:1])
        return loss + torch.mean(pred_up["ux"] ** 2) + torch.mean((pred_up["sigmayy"] - sigma_yy_up) ** 2)


class LinearElasticityUniaxial2D(BaseEquation):
    def __init__(self, young_modulus: float = 1.0, poisson_ratio: float = 0.3, strain: float = 0.02) -> None:
        super().__init__(
            "linear_elasticity_uniaxial",
            input_dim=2,
            raw_output_dim=5,
            supervised_keys=("ux", "uy", "sigmaxx", "sigmayy", "sigmaxy"),
        )
        self.young_modulus = young_modulus
        self.poisson_ratio = poisson_ratio
        self.strain = strain

    @property
    def shear_modulus(self) -> float:
        return self.young_modulus / (2.0 * (1.0 + self.poisson_ratio))

    def exact(self, points: Tensor) -> TensorDict:
        x, y = points[:, 0:1], points[:, 1:2]
        exx = torch.full_like(x, self.strain)
        eyy = torch.full_like(y, -self.poisson_ratio * self.strain)
        exy = torch.zeros_like(x)
        ux = self.strain * x
        uy = -self.poisson_ratio * self.strain * y
        sigmaxx = self.young_modulus * self.strain * torch.ones_like(x)
        sigmayy = torch.zeros_like(x)
        sigmaxy = torch.zeros_like(x)
        return {
            "ux": ux,
            "uy": uy,
            "sigmaxx": sigmaxx,
            "sigmayy": sigmayy,
            "sigmaxy": sigmaxy,
            "exx": exx,
            "eyy": eyy,
            "exy": exy,
        }

    def sources(self, points: Tensor) -> TensorDict:
        zeros = torch.zeros((points.shape[0], 1), dtype=points.dtype, device=points.device)
        return {"fx": zeros, "fy": zeros}

    def pde_residuals(self, model: nn.Module, points: Tensor) -> TensorDict:
        pred = self.prediction(model, points)
        ux, uy = pred["ux"], pred["uy"]
        sigmaxx, sigmayy, sigmaxy = pred["sigmaxx"], pred["sigmayy"], pred["sigmaxy"]
        exx = column_grad(ux, points, 0)
        eyy = column_grad(uy, points, 1)
        exy = (column_grad(ux, points, 1) + column_grad(uy, points, 0)) / 2
        factor = self.young_modulus / (1.0 - self.poisson_ratio**2)
        return {
            "balance_x": column_grad(sigmaxx, points, 0) + column_grad(sigmaxy, points, 1),
            "balance_y": column_grad(sigmayy, points, 1) + column_grad(sigmaxy, points, 0),
            "plane_stress_xx": factor * (exx + self.poisson_ratio * eyy) - sigmaxx,
            "plane_stress_yy": factor * (eyy + self.poisson_ratio * exx) - sigmayy,
            "plane_stress_xy": 2.0 * self.shear_modulus * exy - sigmaxy,
        }

    def boundary_loss(self, model: nn.Module, count: int, device: torch.device) -> Tensor:
        if count <= 0:
            return torch.zeros((), device=device)
        s = torch.rand((count, 1), device=device)
        zeros, ones = torch.zeros_like(s), torch.ones_like(s)
        left = torch.cat([zeros, s], dim=1)
        bottom = torch.cat([s, zeros], dim=1)
        right = torch.cat([ones, s], dim=1)
        top = torch.cat([s, ones], dim=1)
        pred_left = self.prediction(model, left)
        pred_bottom = self.prediction(model, bottom)
        pred_right = self.prediction(model, right)
        pred_top = self.prediction(model, top)
        target_sigma = self.young_modulus * self.strain
        return (
            torch.mean(pred_left["ux"] ** 2)
            + torch.mean(pred_bottom["uy"] ** 2)
            + torch.mean((pred_right["sigmaxx"] - target_sigma) ** 2)
            + torch.mean(pred_right["sigmaxy"] ** 2)
            + torch.mean(pred_top["sigmayy"] ** 2)
            + torch.mean(pred_top["sigmaxy"] ** 2)
        )


class LinearElasticityMirrorSymmetric2D(BaseEquation):
    def __init__(self, lambda_: float = 1.0, mu: float = 0.5, amplitude_x: float = 0.04, amplitude_y: float = 0.02) -> None:
        super().__init__(
            "linear_elasticity_mirror",
            input_dim=2,
            raw_output_dim=5,
            supervised_keys=("ux", "uy", "sigmaxx", "sigmayy", "sigmaxy"),
        )
        self.lambda_ = lambda_
        self.mu = mu
        self.amplitude_x = amplitude_x
        self.amplitude_y = amplitude_y

    def exact(self, points: Tensor) -> TensorDict:
        x, y = points[:, 0:1], points[:, 1:2]
        xc = x - 0.5
        ux = self.amplitude_x * xc * (1.0 - 4.0 * xc**2) * torch.sin(torch.pi * y)
        uy = self.amplitude_y * torch.cos(torch.pi * xc) * y * (1.0 - y)
        exx = column_grad(ux, points, 0)
        eyy = column_grad(uy, points, 1)
        exy = (column_grad(ux, points, 1) + column_grad(uy, points, 0)) / 2
        sigmaxx = (self.lambda_ + 2.0 * self.mu) * exx + self.lambda_ * eyy
        sigmayy = (self.lambda_ + 2.0 * self.mu) * eyy + self.lambda_ * exx
        sigmaxy = 2.0 * self.mu * exy
        return {
            "ux": ux,
            "uy": uy,
            "sigmaxx": sigmaxx,
            "sigmayy": sigmayy,
            "sigmaxy": sigmaxy,
            "exx": exx,
            "eyy": eyy,
            "exy": exy,
        }

    def body_force(self, points: Tensor, fields: TensorDict | None = None) -> TensorDict:
        fields = self.exact(points) if fields is None else fields
        force_x = -(column_grad(fields["sigmaxx"], points, 0) + column_grad(fields["sigmaxy"], points, 1))
        force_y = -(column_grad(fields["sigmayy"], points, 1) + column_grad(fields["sigmaxy"], points, 0))
        return {"fx": force_x, "fy": force_y}

    def sources(self, points: Tensor) -> TensorDict:
        return self.body_force(points)

    def pde_residuals(self, model: nn.Module, points: Tensor) -> TensorDict:
        pred = self.prediction(model, points)
        ux, uy = pred["ux"], pred["uy"]
        sigmaxx, sigmayy, sigmaxy = pred["sigmaxx"], pred["sigmayy"], pred["sigmaxy"]
        exx = column_grad(ux, points, 0)
        eyy = column_grad(uy, points, 1)
        exy = (column_grad(ux, points, 1) + column_grad(uy, points, 0)) / 2
        force = self.body_force(points)
        return {
            "balance_x": column_grad(sigmaxx, points, 0) + column_grad(sigmaxy, points, 1) + force["fx"],
            "balance_y": column_grad(sigmayy, points, 1) + column_grad(sigmaxy, points, 0) + force["fy"],
            "constitutive_xx": (self.lambda_ + 2.0 * self.mu) * exx + self.lambda_ * eyy - sigmaxx,
            "constitutive_yy": (self.lambda_ + 2.0 * self.mu) * eyy + self.lambda_ * exx - sigmayy,
            "constitutive_xy": 2.0 * self.mu * exy - sigmaxy,
        }

    def boundary_loss(self, model: nn.Module, count: int, device: torch.device) -> Tensor:
        if count <= 0:
            return torch.zeros((), device=device)
        s = torch.rand((count, 1), device=device)
        zeros, ones = torch.zeros_like(s), torch.ones_like(s)
        boundaries = torch.cat(
            [
                torch.cat([zeros, s], dim=1),
                torch.cat([ones, s], dim=1),
                torch.cat([s, zeros], dim=1),
                torch.cat([s, ones], dim=1),
            ],
            dim=0,
        ).requires_grad_(True)
        pred = self.prediction(model, boundaries)
        true = self.exact(boundaries)
        return sum(torch.mean((pred[key] - true[key].detach()) ** 2) for key in self.supervised_keys)


EQUATIONS: Dict[str, Callable[[], BaseEquation]] = {
    "burgers": Burgers2D,
    "heat": HeatConduction2D,
    "linear_elasticity": LinearElasticity2D,
    "linear_elasticity_mirror": LinearElasticityMirrorSymmetric2D,
    "linear_elasticity_uniaxial": LinearElasticityUniaxial2D,
    "navier_stokes": NavierStokes2D,
    "navier_stokes_taylor_green": NavierStokesTaylorGreen2D,
}


def get_equation(name: str) -> BaseEquation:
    try:
        return EQUATIONS[name]()
    except KeyError as exc:
        choices = ", ".join(sorted(EQUATIONS))
        raise ValueError(f"Unknown equation '{name}'. Choose from: {choices}") from exc
