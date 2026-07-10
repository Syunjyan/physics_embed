from __future__ import annotations

from pathlib import Path
import argparse
import json
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from physics_embed.constitutive import linear_elasticity_hooke_residual
from physics_embed.equations import LinearElasticity2D
from physics_embed.evaluate import evaluate_run
from physics_embed.models import MLP


def _load_targets(npz: np.lib.npyio.NpzFile, device: torch.device) -> dict[str, torch.Tensor]:
    return {
        key: torch.tensor(npz[key], dtype=torch.float32, device=device)
        for key in npz.files
        if key not in {"points", "equation"}
    }


def _train_one(
    run_dir: Path,
    dataset: Path,
    device: torch.device,
    epochs: int,
    samples: int,
    boundary_samples: int,
    hidden: tuple[int, ...],
    lr: float,
    use_constitutive: bool,
    constitutive_weight: float,
    seed: int,
) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    equation = LinearElasticity2D()
    data = np.load(dataset)
    points_all = torch.tensor(data["points"], dtype=torch.float32, device=device)
    targets_all = _load_targets(data, device)
    model = MLP(equation.input_dim, equation.raw_output_dim, hidden_layers=hidden).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    run_dir.mkdir(parents=True, exist_ok=True)

    history = []
    sample_count = min(samples, points_all.shape[0])
    supervised_keys = tuple(equation.supervised_keys)

    for epoch in range(1, epochs + 1):
        idx = torch.randint(0, points_all.shape[0], (sample_count,), device=device)
        points = points_all[idx].detach().clone().requires_grad_(True)
        targets = {key: targets_all[key][idx] for key in supervised_keys}

        optimizer.zero_grad()
        residuals = equation.pde_residuals(model, points)
        balance_loss = torch.mean(residuals["balance_x"] ** 2) + torch.mean(residuals["balance_y"] ** 2)
        predictions = equation.prediction(model, points)
        data_loss = sum(torch.mean((predictions[key] - targets[key]) ** 2) for key in supervised_keys)
        boundary_loss = equation.boundary_loss(model, boundary_samples, device)

        constitutive_loss = torch.zeros((), device=device)
        if use_constitutive:
            hooke = linear_elasticity_hooke_residual(
                predictions,
                points,
                lambda_=equation.lambda_,
                mu=equation.mu,
            )
            constitutive_loss = sum(torch.mean(value**2) for value in hooke.values())

        loss = 0.1 * balance_loss + 8.0 * data_loss + 3.0 * boundary_loss + constitutive_weight * constitutive_loss
        loss.backward()
        optimizer.step()

        row = {
            "epoch": epoch,
            "weighted_loss": float(loss.detach().cpu()),
            "balance_loss": float(balance_loss.detach().cpu()),
            "data_loss": float(data_loss.detach().cpu()),
            "boundary_loss": float(boundary_loss.detach().cpu()),
            "constitutive_loss": float(constitutive_loss.detach().cpu()),
        }
        history.append(row)
        if epoch == 1 or epoch % max(1, epochs // 10) == 0:
            print(json.dumps({"run": run_dir.name, **row}, ensure_ascii=False), flush=True)

    np.save(run_dir / "training_history.npy", np.array(history, dtype=object))
    torch.save(model.state_dict(), run_dir / "model.pt")
    with (run_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "equation": "linear_elasticity",
                "dataset": str(dataset),
                "epochs": epochs,
                "samples": sample_count,
                "boundary_samples": boundary_samples,
                "hidden": hidden,
                "lr": lr,
                "device": str(device),
                "ablation": "with_constitutive" if use_constitutive else "without_constitutive",
                "balance_weight": 0.1,
                "data_weight": 8.0,
                "boundary_weight": 3.0,
                "constitutive_weight": constitutive_weight if use_constitutive else 0.0,
            },
            f,
            indent=2,
        )


def _plot_summary(output_root: Path) -> None:
    rows = []
    for run_dir in (output_root / "without_constitutive", output_root / "with_constitutive"):
        metrics_path = run_dir / "metrics.json"
        history_path = run_dir / "training_history.npy"
        if not metrics_path.exists() or not history_path.exists():
            continue
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        history = [dict(item) for item in np.load(history_path, allow_pickle=True).tolist()]
        rows.append((run_dir.name, metrics, history))
    if not rows:
        return

    plt.figure(figsize=(7, 5))
    for name, _metrics, history in rows:
        plt.plot([row["epoch"] for row in history], [row["weighted_loss"] for row in history], label=name)
    plt.yscale("log")
    plt.xlabel("epoch")
    plt.ylabel("weighted loss")
    plt.title("Linear elasticity constitutive ablation")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_root / "ablation_loss.png", dpi=200)
    plt.close()

    summary = {}
    for name, metrics, history in rows:
        summary[name] = {
            "final_weighted_loss": history[-1]["weighted_loss"],
            "final_constitutive_loss": history[-1]["constitutive_loss"],
            "relative_l2": {key: value["relative_l2"] for key, value in metrics.items()},
        }
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def run_ablation(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    dataset = Path(args.dataset)
    output_root = Path(args.output_root)

    for name, use_constitutive in (("without_constitutive", False), ("with_constitutive", True)):
        run_dir = output_root / name
        if not args.skip_train:
            _train_one(
                run_dir=run_dir,
                dataset=dataset,
                device=device,
                epochs=args.epochs,
                samples=args.samples,
                boundary_samples=args.boundary_samples,
                hidden=tuple(args.hidden),
                lr=args.lr,
                use_constitutive=use_constitutive,
                constitutive_weight=args.constitutive_weight,
                seed=args.seed,
            )
        if not args.skip_evaluate:
            evaluate_run(run_dir, dataset, t_value=None, device_name=args.device)

    _plot_summary(output_root)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Linear elasticity Hooke constitutive-residual ablation.")
    parser.add_argument("--dataset", default="data/linear_elasticity.npz")
    parser.add_argument("--output-root", default="runs/ablation/linear_elasticity_constitutive")
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--epochs", type=int, default=3000)
    parser.add_argument("--samples", type=int, default=2048)
    parser.add_argument("--boundary-samples", type=int, default=1024)
    parser.add_argument("--hidden", type=int, nargs="+", default=[64, 64, 64, 64])
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--constitutive-weight", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-evaluate", action="store_true")
    return parser


def main() -> None:
    run_ablation(build_parser().parse_args())


if __name__ == "__main__":
    main()
