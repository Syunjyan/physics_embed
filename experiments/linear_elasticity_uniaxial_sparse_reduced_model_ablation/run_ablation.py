from __future__ import annotations

from pathlib import Path
import argparse
import json
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from physics_embed.equations import LinearElasticityUniaxial2D
from physics_embed.evaluate import evaluate_run
from physics_embed.models import MLP
from physics_embed.reduced_models import linear_elasticity_uniaxial_stress


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
    use_reduced_model: bool,
    reduced_model_weight: float,
    seed: int,
) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    equation = LinearElasticityUniaxial2D()
    data = np.load(dataset)
    points_all = torch.tensor(data["points"], dtype=torch.float32, device=device)
    targets_all = _load_targets(data, device)
    model = MLP(equation.input_dim, equation.raw_output_dim, hidden_layers=hidden).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    run_dir.mkdir(parents=True, exist_ok=True)

    history = []
    sample_count = min(samples, points_all.shape[0])

    for epoch in range(1, epochs + 1):
        idx = torch.randint(0, points_all.shape[0], (sample_count,), device=device)
        points = points_all[idx].detach().clone().requires_grad_(True)

        optimizer.zero_grad()
        residuals = equation.pde_residuals(model, points)
        # Sparse/engineering setting:
        # - use displacement labels only,
        # - use equilibrium residual only,
        # - do not use internal stress labels or the full constitutive residual.
        # The reduced model is expected to supply the missing stress-displacement relation.
        balance_loss = torch.mean(residuals["balance_x"] ** 2) + torch.mean(residuals["balance_y"] ** 2)
        predictions = equation.prediction(model, points)
        displacement_loss = torch.mean((predictions["ux"] - targets_all["ux"][idx]) ** 2)
        displacement_loss = displacement_loss + torch.mean((predictions["uy"] - targets_all["uy"][idx]) ** 2)
        boundary_loss = equation.boundary_loss(model, boundary_samples, device)

        reduced_model_loss = torch.zeros((), device=device)
        if use_reduced_model:
            reduced_residuals = linear_elasticity_uniaxial_stress(
                predictions,
                points,
                young_modulus=equation.young_modulus,
                poisson_ratio=equation.poisson_ratio,
            )
            reduced_model_loss = sum(torch.mean(value**2) for value in reduced_residuals.values())

        loss = 0.1 * balance_loss + 20.0 * displacement_loss + 3.0 * boundary_loss
        loss = loss + reduced_model_weight * reduced_model_loss
        loss.backward()
        optimizer.step()

        row = {
            "epoch": epoch,
            "weighted_loss": float(loss.detach().cpu()),
            "balance_loss": float(balance_loss.detach().cpu()),
            "displacement_loss": float(displacement_loss.detach().cpu()),
            "boundary_loss": float(boundary_loss.detach().cpu()),
            "reduced_model_loss": float(reduced_model_loss.detach().cpu()),
        }
        history.append(row)
        if epoch == 1 or epoch % max(1, epochs // 10) == 0:
            print(json.dumps({"run": run_dir.name, **row}, ensure_ascii=False), flush=True)

    np.save(run_dir / "training_history.npy", np.array(history, dtype=object))
    torch.save(model.state_dict(), run_dir / "model.pt")
    with (run_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "equation": "linear_elasticity_uniaxial",
                "dataset": str(dataset),
                "epochs": epochs,
                "samples": sample_count,
                "boundary_samples": boundary_samples,
                "hidden": hidden,
                "lr": lr,
                "seed": seed,
                "device": str(device),
                "ablation": "with_reduced_model" if use_reduced_model else "without_reduced_model",
                "training_setting": "sparse_displacement_only",
                "balance_weight": 0.1,
                "displacement_weight": 20.0,
                "boundary_weight": 3.0,
                "reduced_model_weight": reduced_model_weight if use_reduced_model else 0.0,
            },
            f,
            indent=2,
        )


def _plot_summary(output_root: Path) -> None:
    rows = []
    for run_dir in (output_root / "without_reduced_model", output_root / "with_reduced_model"):
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
    plt.title("Uniaxial sparse reduced-model ablation")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_root / "ablation_loss.png", dpi=200)
    plt.close()

    summary = {}
    for name, metrics, history in rows:
        summary[name] = {
            "final_weighted_loss": history[-1]["weighted_loss"],
            "final_reduced_model_loss": history[-1]["reduced_model_loss"],
            "relative_l2": {key: value["relative_l2"] for key, value in metrics.items()},
        }
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def run_ablation(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    dataset = Path(args.dataset)
    output_root = Path(args.output_root)

    for name, use_reduced_model in (("without_reduced_model", False), ("with_reduced_model", True)):
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
                use_reduced_model=use_reduced_model,
                reduced_model_weight=args.reduced_model_weight,
                seed=args.seed,
            )
        if not args.skip_evaluate:
            evaluate_run(run_dir, dataset, t_value=None, device_name=args.device)

    _plot_summary(output_root)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sparse uniaxial elasticity ablation for reduced-model knowledge.")
    parser.add_argument("--dataset", default="data/linear_elasticity_uniaxial.npz")
    parser.add_argument("--output-root", default="runs/ablation/linear_elasticity_uniaxial_sparse_reduced_model")
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--samples", type=int, default=512)
    parser.add_argument("--boundary-samples", type=int, default=256)
    parser.add_argument("--hidden", type=int, nargs="+", default=[32, 32, 32])
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--reduced-model-weight", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-evaluate", action="store_true")
    return parser


def main() -> None:
    run_ablation(build_parser().parse_args())


if __name__ == "__main__":
    main()
