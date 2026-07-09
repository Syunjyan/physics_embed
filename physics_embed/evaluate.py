from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import torch

from physics_embed.equations import EQUATIONS, get_equation
from physics_embed.models import MLP


def _relative_l2(pred: np.ndarray, truth: np.ndarray) -> float:
    denom = np.linalg.norm(truth.reshape(-1))
    if denom == 0.0:
        return float(np.linalg.norm((pred - truth).reshape(-1)))
    return float(np.linalg.norm((pred - truth).reshape(-1)) / denom)


def _load_history(path: Path) -> list[dict]:
    if not path.exists():
        return []
    values = np.load(path, allow_pickle=True)
    return [dict(item) for item in values.tolist()]


def _plot_loss(history: list[dict], output_dir: Path) -> None:
    if not history:
        return
    epochs = [row["epoch"] for row in history]
    keys = [key for key in ("loss", "pde_loss", "data_loss", "boundary_loss") if key in history[0]]
    plt.figure(figsize=(7, 5))
    for key in keys:
        plt.plot(epochs, [row[key] for row in history], label=key)
    plt.yscale("log")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("Training loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "loss.png", dpi=200)
    plt.close()


def _slice_points(points: np.ndarray, t_value: float | None) -> np.ndarray:
    if points.shape[1] == 2:
        return np.ones(points.shape[0], dtype=bool)
    unique_t = np.unique(points[:, 2])
    target = unique_t[len(unique_t) // 2] if t_value is None else unique_t[np.argmin(np.abs(unique_t - t_value))]
    return np.isclose(points[:, 2], target)


def _grid_field(points: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_unique = np.unique(points[:, 0])
    y_unique = np.unique(points[:, 1])
    grid = values.reshape(len(x_unique), len(y_unique))
    return x_unique, y_unique, grid.T


def _plot_fields(
    equation_name: str,
    points: np.ndarray,
    exact: Dict[str, np.ndarray],
    pred: Dict[str, np.ndarray],
    output_dir: Path,
    t_value: float | None,
) -> None:
    keys = [key for key in pred if key in exact]
    mask = _slice_points(points, t_value)
    sliced_points = points[mask]
    rows = len(keys)
    fig, axes = plt.subplots(rows, 3, figsize=(12, max(3, 2.8 * rows)), squeeze=False)
    for row, key in enumerate(keys):
        _, _, exact_grid = _grid_field(sliced_points[:, :2], exact[key][mask, 0])
        x_unique, y_unique, pred_grid = _grid_field(sliced_points[:, :2], pred[key][mask, 0])
        err_grid = np.abs(pred_grid - exact_grid)
        extent = [x_unique.min(), x_unique.max(), y_unique.min(), y_unique.max()]
        panels = ((exact_grid, f"exact {key}"), (pred_grid, f"pred {key}"), (err_grid, f"|err| {key}"))
        for col, (field, title) in enumerate(panels):
            im = axes[row, col].imshow(field, origin="lower", extent=extent, aspect="auto", cmap="viridis")
            axes[row, col].set_title(title)
            axes[row, col].set_xlabel("x")
            axes[row, col].set_ylabel("y")
            fig.colorbar(im, ax=axes[row, col], fraction=0.046, pad=0.04)
    time_title = "" if points.shape[1] == 2 else f", t slice={sliced_points[0, 2]:.3f}"
    fig.suptitle(f"{equation_name} PINN field comparison{time_title}")
    fig.tight_layout()
    fig.savefig(output_dir / "fields.png", dpi=200)
    plt.close(fig)


def evaluate_run(
    run_dir: Path,
    dataset_path: Path | None,
    t_value: float | None,
    device_name: str | None,
) -> None:
    with (run_dir / "config.json").open("r", encoding="utf-8") as f:
        config = json.load(f)
    equation_name = config["equation"]
    equation = get_equation(equation_name)
    dataset = Path(dataset_path or config["dataset"])
    data = np.load(dataset)

    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    hidden = tuple(int(value) for value in config["hidden"])
    model = MLP(equation.input_dim, equation.raw_output_dim, hidden_layers=hidden).to(device)
    model.load_state_dict(torch.load(run_dir / "model.pt", map_location=device))
    model.eval()

    points = torch.tensor(data["points"], dtype=torch.float32, device=device).requires_grad_(True)
    pred_tensors = equation.prediction(model, points)
    exact_tensors = equation.exact(points)

    pred = {
        key: value.detach().cpu().numpy()
        for key, value in pred_tensors.items()
        if key in tuple(equation.supervised_keys)
    }
    exact = {
        key: exact_tensors[key].detach().cpu().numpy()
        for key in pred
    }

    metrics = {
        key: {
            "relative_l2": _relative_l2(pred[key], exact[key]),
            "mse": float(np.mean((pred[key] - exact[key]) ** 2)),
        }
        for key in pred
    }

    with (run_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    np.savez_compressed(
        run_dir / "predictions.npz",
        points=data["points"],
        **{f"pred_{key}": value for key, value in pred.items()},
        **{f"exact_{key}": value for key, value in exact.items()},
    )

    _plot_loss(_load_history(run_dir / "training_history.npy"), run_dir)
    _plot_fields(equation_name, data["points"], exact, pred, run_dir, t_value)
    print(json.dumps(metrics, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate and visualize a trained PINN run.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--t", type=float, default=None)
    parser.add_argument("--device", default=None, help="Torch device, e.g. cpu, cuda, cuda:1.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    evaluate_run(args.run_dir, args.dataset, args.t, args.device)


if __name__ == "__main__":
    main()
