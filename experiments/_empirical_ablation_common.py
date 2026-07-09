from __future__ import annotations

from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import torch

from physics_embed.autodiff import column_grad
from physics_embed.empirical import burgers_energy_decay, navier_stokes_bernoulli
from physics_embed.equations import get_equation
from physics_embed.evaluate import evaluate_run
from physics_embed.models import MLP


def load_targets(npz: np.lib.npyio.NpzFile, device: torch.device) -> dict[str, torch.Tensor]:
    return {
        key: torch.tensor(npz[key], dtype=torch.float32, device=device)
        for key in npz.files
        if key not in {"points", "equation"}
    }


def empirical_loss_for(equation_name: str, equation, model: MLP, points: torch.Tensor, device: torch.device) -> torch.Tensor:
    predictions = equation.prediction(model, points)
    if equation_name == "heat":
        # Fourier heat-flux boundary residual on the top and bottom boundaries.
        # This is intentionally treated as empirical/engineering knowledge in this ablation.
        count = max(32, points.shape[0] // 4)
        s = torch.rand((count, 1), device=device)
        zeros, ones = torch.zeros_like(s), torch.ones_like(s)
        bottom = torch.cat([s, zeros], dim=1).requires_grad_(True)
        top = torch.cat([s, ones], dim=1).requires_grad_(True)
        u_bottom = equation.prediction(model, bottom)["u"]
        u_top = equation.prediction(model, top)["u"]
        target_bottom = torch.pi * torch.sin(torch.pi * bottom[:, 0:1])
        target_top = -torch.pi * torch.sin(torch.pi * top[:, 0:1])
        return torch.mean((column_grad(u_bottom, bottom, 1) - target_bottom) ** 2) + torch.mean(
            (column_grad(u_top, top, 1) - target_top) ** 2
        )
    if equation_name == "burgers":
        residuals = burgers_energy_decay(predictions, points)
        return sum(torch.mean(value**2) for value in residuals.values())
    if equation_name == "navier_stokes":
        residuals = navier_stokes_bernoulli(predictions, points)
        return sum(torch.mean(value**2) for value in residuals.values())
    raise ValueError(f"No empirical ablation is defined for {equation_name}")


def train_one(
    equation_name: str,
    run_dir: Path,
    dataset: Path,
    device: torch.device,
    epochs: int,
    samples: int,
    boundary_samples: int,
    hidden: tuple[int, ...],
    lr: float,
    use_empirical: bool,
    empirical_weight: float,
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
        empirical_loss = torch.zeros((), device=device)
        if use_empirical:
            empirical_loss = empirical_loss_for(equation_name, equation, model, points, device)

        loss = (
            pde_weight * pde_loss
            + data_weight * data_loss
            + boundary_weight * boundary_loss
            + empirical_weight * empirical_loss
        )
        loss.backward()
        optimizer.step()

        row = {
            "epoch": epoch,
            "weighted_loss": float(loss.detach().cpu()),
            "pde_loss": float(pde_loss.detach().cpu()),
            "data_loss": float(data_loss.detach().cpu()),
            "boundary_loss": float(boundary_loss.detach().cpu()),
            "empirical_loss": float(empirical_loss.detach().cpu()),
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
                "device": str(device),
                "ablation": "with_empirical" if use_empirical else "without_empirical",
                "pde_weight": pde_weight,
                "data_weight": data_weight,
                "boundary_weight": boundary_weight,
                "empirical_weight": empirical_weight if use_empirical else 0.0,
            },
            f,
            indent=2,
        )


def summarize(output_root: Path) -> None:
    rows = []
    for run_dir in (output_root / "without_empirical", output_root / "with_empirical"):
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
            "final_empirical_loss": history[-1]["empirical_loss"],
            "relative_l2": {key: value["relative_l2"] for key, value in metrics.items()},
        }
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def run_empirical_ablation(args) -> None:
    device = torch.device(args.device)
    dataset = Path(args.dataset)
    output_root = Path(args.output_root)
    for name, use_empirical in (("without_empirical", False), ("with_empirical", True)):
        run_dir = output_root / name
        if not args.skip_train:
            train_one(
                equation_name=args.equation,
                run_dir=run_dir,
                dataset=dataset,
                device=device,
                epochs=args.epochs,
                samples=args.samples,
                boundary_samples=args.boundary_samples,
                hidden=tuple(args.hidden),
                lr=args.lr,
                use_empirical=use_empirical,
                empirical_weight=args.empirical_weight,
                seed=args.seed,
                pde_weight=args.pde_weight,
                data_weight=args.data_weight,
                boundary_weight=args.boundary_weight,
            )
        if not args.skip_evaluate:
            evaluate_run(run_dir, dataset, t_value=None, device_name=args.device)
    summarize(output_root)
