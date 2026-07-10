from __future__ import annotations

from pathlib import Path
import argparse
import json
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from physics_embed.equations import get_equation
from physics_embed.models import MLP
from physics_embed.postprocess import mirror_symmetry_projection
from physics_embed.train import train_from_dataset


PARITY = {
    "ux": -1.0,
    "uy": 1.0,
    "sigmaxx": 1.0,
    "sigmayy": 1.0,
    "sigmaxy": -1.0,
}


def _relative_l2(pred: np.ndarray, truth: np.ndarray) -> float:
    denom = np.linalg.norm(truth.reshape(-1))
    return float(np.linalg.norm((pred - truth).reshape(-1)) / denom) if denom else float(np.linalg.norm(pred - truth))


def _metrics(pred: dict[str, np.ndarray], truth: dict[str, np.ndarray]) -> dict:
    return {
        key: {
            "relative_l2": _relative_l2(pred[key], truth[key]),
            "mse": float(np.mean((pred[key] - truth[key]) ** 2)),
        }
        for key in pred
    }


def _grid(points: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_unique = np.unique(points[:, 0])
    y_unique = np.unique(points[:, 1])
    return x_unique, y_unique, values.reshape(len(x_unique), len(y_unique)).T


def _plot_comparison(points: np.ndarray, truth: dict[str, np.ndarray], raw: dict[str, np.ndarray], projected: dict[str, np.ndarray], out: Path) -> None:
    keys = list(raw)
    fig, axes = plt.subplots(len(keys), 4, figsize=(15, 2.8 * len(keys)), squeeze=False)
    for row, key in enumerate(keys):
        x_unique, y_unique, truth_grid = _grid(points, truth[key][:, 0])
        _, _, raw_grid = _grid(points, raw[key][:, 0])
        _, _, projected_grid = _grid(points, projected[key][:, 0])
        panels = (
            (truth_grid, f"exact {key}"),
            (raw_grid, f"raw {key}"),
            (projected_grid, f"sym {key}"),
            (np.abs(projected_grid - truth_grid), f"sym err {key}"),
        )
        extent = [x_unique.min(), x_unique.max(), y_unique.min(), y_unique.max()]
        for col, (field, title) in enumerate(panels):
            im = axes[row, col].imshow(field, origin="lower", extent=extent, aspect="auto", cmap="coolwarm")
            axes[row, col].set_title(title)
            axes[row, col].set_xlabel("x")
            axes[row, col].set_ylabel("y")
            fig.colorbar(im, ax=axes[row, col], fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def _markdown_table(raw_metrics: dict, projected_metrics: dict) -> str:
    lines = [
        "| Field | Metric | Raw prediction | Symmetry projected | Delta | Relative improvement |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for field in raw_metrics:
        for metric_name in ("mse", "relative_l2"):
            before = raw_metrics[field][metric_name]
            after = projected_metrics[field][metric_name]
            delta = after - before
            improvement = (before - after) / before * 100.0 if before else 0.0
            lines.append(f"| {field} | {metric_name} | {before:.6g} | {after:.6g} | {delta:.6g} | {improvement:.2f}% |")
    return "\n".join(lines)


def evaluate_symmetry(run_dir: Path, dataset: Path, device_name: str) -> None:
    equation = get_equation("linear_elasticity_mirror")
    data = np.load(dataset)
    device = torch.device(device_name)
    with (run_dir / "config.json").open("r", encoding="utf-8") as f:
        config = json.load(f)
    hidden = tuple(int(value) for value in config["hidden"])
    model = MLP(equation.input_dim, equation.raw_output_dim, hidden_layers=hidden).to(device)
    model.load_state_dict(torch.load(run_dir / "model.pt", map_location=device))
    model.eval()

    points = torch.tensor(data["points"], dtype=torch.float32, device=device).requires_grad_(True)
    raw_tensors = equation.prediction(model, points)
    projected_tensors = mirror_symmetry_projection(model, points, equation.prediction, PARITY)
    truth_tensors = equation.exact(points)
    keys = tuple(equation.supervised_keys)
    raw = {key: raw_tensors[key].detach().cpu().numpy() for key in keys}
    projected = {key: projected_tensors[key].detach().cpu().numpy() for key in keys}
    truth = {key: truth_tensors[key].detach().cpu().numpy() for key in keys}

    raw_metrics = _metrics(raw, truth)
    projected_metrics = _metrics(projected, truth)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "raw_metrics.json").write_text(json.dumps(raw_metrics, indent=2), encoding="utf-8")
    (run_dir / "symmetry_projected_metrics.json").write_text(json.dumps(projected_metrics, indent=2), encoding="utf-8")
    table = _markdown_table(raw_metrics, projected_metrics)
    (run_dir / "metrics_table.md").write_text(table + "\n", encoding="utf-8")
    np.savez_compressed(
        run_dir / "postprocess_predictions.npz",
        points=data["points"],
        **{f"raw_{key}": value for key, value in raw.items()},
        **{f"sym_{key}": value for key, value in projected.items()},
        **{f"exact_{key}": value for key, value in truth.items()},
    )
    _plot_comparison(data["points"], truth, raw, projected, run_dir / "symmetry_projection_fields.png")
    print(table)


def run(args: argparse.Namespace) -> None:
    dataset = Path(args.dataset)
    run_dir = Path(args.output_root)
    if not args.skip_train:
        train_from_dataset(
            equation_name="linear_elasticity_mirror",
            dataset_path=dataset,
            output_dir=run_dir,
            epochs=args.epochs,
            samples=args.samples,
            boundary_samples=args.boundary_samples,
            lr=args.lr,
            hidden=tuple(args.hidden),
            seed=args.seed,
            device_name=args.device,
            pde_weight=args.pde_weight,
            data_weight=args.data_weight,
            boundary_weight=args.boundary_weight,
        )
    evaluate_symmetry(run_dir, dataset, args.device)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Postprocess hard mirror-symmetry ablation for linear elasticity.")
    parser.add_argument("--dataset", default="data/linear_elasticity_mirror.npz")
    parser.add_argument("--output-root", default="runs/ablation/linear_elasticity_mirror_postprocess_symmetry")
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--epochs", type=int, default=1200)
    parser.add_argument("--samples", type=int, default=1024)
    parser.add_argument("--boundary-samples", type=int, default=512)
    parser.add_argument("--hidden", type=int, nargs="+", default=[64, 64, 64, 64])
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--pde-weight", type=float, default=1.0)
    parser.add_argument("--data-weight", type=float, default=5.0)
    parser.add_argument("--boundary-weight", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--skip-train", action="store_true")
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
