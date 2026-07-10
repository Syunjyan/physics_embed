from __future__ import annotations

from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import torch

from physics_embed.equations import get_equation
from physics_embed.evaluate import evaluate_run
from physics_embed.models import MLP
from physics_embed.reduced_models import REDUCED_MODELS


def load_targets(npz: np.lib.npyio.NpzFile, device: torch.device) -> dict[str, torch.Tensor]:
    return {
        key: torch.tensor(npz[key], dtype=torch.float32, device=device)
        for key in npz.files
        if key not in {"points", "equation"}
    }


def _make_reduced_model_loss(model_name: str, model_kwargs: dict):
    reduced_model = REDUCED_MODELS[model_name]

    def loss_fn(fields, points):
        return reduced_model(fields, points, **model_kwargs)

    return loss_fn


def train_one(
    equation_name: str,
    reduced_model_name: str,
    reduced_model_kwargs: dict,
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
    pde_weight: float,
    data_weight: float,
    boundary_weight: float,
) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    equation = get_equation(equation_name)
    data = np.load(dataset)
    points_all = torch.tensor(data["points"], dtype=torch.float32, device=device)
    targets_all = load_targets(data, device)
    model = MLP(equation.input_dim, equation.raw_output_dim, hidden_layers=hidden).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    reduced_model_fn = _make_reduced_model_loss(reduced_model_name, reduced_model_kwargs)
    run_dir.mkdir(parents=True, exist_ok=True)

    history = []
    sample_count = min(samples, points_all.shape[0])
    supervised_keys = tuple(equation.supervised_keys)

    for epoch in range(1, epochs + 1):
        idx = torch.randint(0, points_all.shape[0], (sample_count,), device=device)
        points = points_all[idx].detach().clone().requires_grad_(True)
        targets = {key: targets_all[key][idx] for key in supervised_keys if key in targets_all}

        optimizer.zero_grad()
        pde_loss = sum(torch.mean(value**2) for value in equation.pde_residuals(model, points).values())
        predictions = equation.prediction(model, points)
        data_loss = sum(torch.mean((predictions[key] - targets[key]) ** 2) for key in targets)
        boundary_loss = equation.boundary_loss(model, boundary_samples, device)
        reduced_model_loss = torch.zeros((), device=device)
        if use_reduced_model:
            residuals = reduced_model_fn(predictions, points)
            reduced_model_loss = sum(torch.mean(value**2) for value in residuals.values())

        loss = (
            pde_weight * pde_loss
            + data_weight * data_loss
            + boundary_weight * boundary_loss
            + reduced_model_weight * reduced_model_loss
        )
        loss.backward()
        optimizer.step()

        row = {
            "epoch": epoch,
            "weighted_loss": float(loss.detach().cpu()),
            "pde_loss": float(pde_loss.detach().cpu()),
            "data_loss": float(data_loss.detach().cpu()),
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
                "equation": equation_name,
                "dataset": str(dataset),
                "epochs": epochs,
                "samples": sample_count,
                "boundary_samples": boundary_samples,
                "hidden": hidden,
                "lr": lr,
                "seed": seed,
                "device": str(device),
                "ablation": "with_reduced_model" if use_reduced_model else "without_reduced_model",
                "reduced_model_name": reduced_model_name if use_reduced_model else None,
                "reduced_model_kwargs": reduced_model_kwargs if use_reduced_model else {},
                "pde_weight": pde_weight,
                "data_weight": data_weight,
                "boundary_weight": boundary_weight,
                "reduced_model_weight": reduced_model_weight if use_reduced_model else 0.0,
            },
            f,
            indent=2,
        )


def summarize(output_root: Path) -> None:
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
    plt.title(output_root.name)
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


def run_reduced_model_ablation(args) -> None:
    device = torch.device(args.device)
    dataset = Path(args.dataset)
    output_root = Path(args.output_root)
    model_kwargs = json.loads(args.reduced_model_kwargs)
    for name, use_reduced_model in (("without_reduced_model", False), ("with_reduced_model", True)):
        run_dir = output_root / name
        if not args.skip_train:
            train_one(
                equation_name=args.equation,
                reduced_model_name=args.reduced_model,
                reduced_model_kwargs=model_kwargs,
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
                pde_weight=args.pde_weight,
                data_weight=args.data_weight,
                boundary_weight=args.boundary_weight,
            )
        if not args.skip_evaluate:
            evaluate_run(run_dir, dataset, t_value=None, device_name=args.device)
    summarize(output_root)
