from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np
import torch

from physics_embed.equations import EQUATIONS, get_equation
from physics_embed.models import MLP


def _load_targets(npz: np.lib.npyio.NpzFile, device: torch.device) -> Dict[str, torch.Tensor]:
    return {
        key: torch.tensor(npz[key], dtype=torch.float32, device=device)
        for key in npz.files
        if key not in {"points", "equation"}
    }


def train_from_dataset(
    equation_name: str,
    dataset_path: Path,
    output_dir: Path,
    epochs: int,
    samples: int,
    boundary_samples: int,
    lr: float,
    hidden: tuple[int, ...],
    seed: int,
    device_name: str | None = None,
    pde_weight: float = 1.0,
    data_weight: float = 1.0,
    boundary_weight: float = 1.0,
) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    equation = get_equation(equation_name)

    data = np.load(dataset_path)
    points_all = torch.tensor(data["points"], dtype=torch.float32, device=device)
    targets_all = _load_targets(data, device)

    model = MLP(equation.input_dim, equation.raw_output_dim, hidden_layers=hidden).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    output_dir.mkdir(parents=True, exist_ok=True)

    history = []
    n_points = points_all.shape[0]
    sample_count = min(samples, n_points)
    supervised_keys = tuple(equation.supervised_keys)

    for epoch in range(1, epochs + 1):
        idx = torch.randint(0, n_points, (sample_count,), device=device)
        points = points_all[idx].detach().clone().requires_grad_(True)
        targets = {
            key: targets_all[key][idx]
            for key in supervised_keys
            if key in targets_all
        }

        optimizer.zero_grad()
        residuals = equation.pde_residuals(model, points)
        pde_loss = sum(torch.mean(value**2) for value in residuals.values())

        predictions = equation.prediction(model, points)
        data_loss = sum(torch.mean((predictions[key] - targets[key]) ** 2) for key in targets)
        bc_loss = equation.boundary_loss(model, boundary_samples, device)
        loss = pde_weight * pde_loss + data_weight * data_loss + boundary_weight * bc_loss

        loss.backward()
        optimizer.step()

        row = {
            "epoch": epoch,
            "loss": float(loss.detach().cpu()),
            "pde_loss": float(pde_loss.detach().cpu()),
            "data_loss": float(data_loss.detach().cpu()),
            "boundary_loss": float(bc_loss.detach().cpu()),
            "weighted_loss": float(loss.detach().cpu()),
        }
        history.append(row)
        if epoch == 1 or epoch % max(1, epochs // 10) == 0:
            print(json.dumps(row, ensure_ascii=False))

    np.save(output_dir / "training_history.npy", np.array(history, dtype=object))
    torch.save(model.state_dict(), output_dir / "model.pt")
    with (output_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "equation": equation_name,
                "dataset": str(dataset_path),
                "epochs": epochs,
                "samples": sample_count,
                "boundary_samples": boundary_samples,
                "lr": lr,
                "hidden": hidden,
                "seed": seed,
                "device": str(device),
                "pde_weight": pde_weight,
                "data_weight": data_weight,
                "boundary_weight": boundary_weight,
            },
            f,
            indent=2,
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a baseline PINN on a generated PDE dataset.")
    parser.add_argument("--equation", required=True, choices=sorted(EQUATIONS))
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--samples", type=int, default=4096)
    parser.add_argument("--boundary-samples", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", default="64,64,64,64")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default=None, help="Torch device, e.g. cpu, cuda, cuda:1.")
    parser.add_argument("--pde-weight", type=float, default=1.0)
    parser.add_argument("--data-weight", type=float, default=1.0)
    parser.add_argument("--boundary-weight", type=float, default=1.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    hidden = tuple(int(item) for item in args.hidden.split(",") if item.strip())
    train_from_dataset(
        args.equation,
        args.dataset,
        args.output_dir,
        args.epochs,
        args.samples,
        args.boundary_samples,
        args.lr,
        hidden,
        args.seed,
        args.device,
        args.pde_weight,
        args.data_weight,
        args.boundary_weight,
    )


if __name__ == "__main__":
    main()
