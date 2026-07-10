from __future__ import annotations

from pathlib import Path
import argparse
import json
import sys
from typing import Any

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

REPO_ROOT = Path(__file__).resolve().parents[2]

from physics_embed.equations import NavierStokesTaylorGreen2D
from physics_embed.models import MLP
from physics_embed.ns_spectral import generate_pdebench_style_ns_dataset


def generate_taylor_green_dataset(out: Path, viscosity: float, spatial_resolution: int, time_steps: int) -> dict[str, float]:
    meta = generate_pdebench_style_ns_dataset(
        out,
        viscosity=viscosity,
        spatial_resolution=spatial_resolution,
        time_steps=time_steps,
        t_final=1.0,
    )
    print(
        f"saved PDEBench-style Taylor-Green NS dataset: {out} "
        f"(nu={viscosity}, points={meta['points']}, solver_max_abs_error={meta['solver_max_abs_error']:.3e})"
    )
    return meta


def load_targets(npz: np.lib.npyio.NpzFile, device: torch.device) -> dict[str, torch.Tensor]:
    skip = {
        "points",
        "equation",
        "viscosity_value",
        "spatial_resolution",
        "time_steps",
        "fields_st",
        "times",
        "x",
        "y",
        "dt",
        "solver_max_abs_error",
        "generation_method",
    }
    return {
        key: torch.tensor(npz[key], dtype=torch.float32, device=device)
        for key in npz.files
        if key not in skip
    }


def relative_l2(pred: np.ndarray, truth: np.ndarray) -> float:
    denom = np.linalg.norm(truth.reshape(-1))
    return float(np.linalg.norm((pred - truth).reshape(-1)) / denom) if denom else float(np.linalg.norm(pred - truth))


def interpolate_state_dict(state_a: dict[str, torch.Tensor], state_b: dict[str, torch.Tensor], alpha: float) -> dict[str, torch.Tensor]:
    return {key: (1.0 - alpha) * state_a[key] + alpha * state_b[key] for key in state_a}


def evaluate_model(model: MLP, equation: NavierStokesTaylorGreen2D, dataset: Path, device: torch.device, run_dir: Path | None = None) -> dict:
    data = np.load(dataset)
    points = torch.tensor(data["points"], dtype=torch.float32, device=device).requires_grad_(True)
    pred_tensors = equation.prediction(model, points)
    exact_tensors = equation.exact(points)
    pred = {key: pred_tensors[key].detach().cpu().numpy() for key in ("u", "v", "p")}
    exact = {key: exact_tensors[key].detach().cpu().numpy() for key in ("u", "v", "p")}
    metrics = {
        key: {
            "relative_l2": relative_l2(pred[key], exact[key]),
            "mse": float(np.mean((pred[key] - exact[key]) ** 2)),
        }
        for key in ("u", "v", "p")
    }
    if run_dir is not None:
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        np.savez_compressed(
            run_dir / "predictions.npz",
            points=data["points"],
            **{f"pred_{key}": value for key, value in pred.items()},
            **{f"exact_{key}": value for key, value in exact.items()},
        )
        plot_loss(run_dir)
        make_gt_pred_gif(data["points"], pred, exact, run_dir / "gt_pred_speed.gif")
    return metrics


def train_model(
    run_dir: Path,
    dataset: Path,
    viscosity: float,
    device: torch.device,
    hidden: tuple[int, ...],
    epochs: int,
    samples: int,
    boundary_samples: int,
    lr: float,
    seed: int,
    init_state: dict[str, torch.Tensor] | None = None,
    eval_every: int | None = None,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    equation = NavierStokesTaylorGreen2D(viscosity=viscosity)
    data = np.load(dataset)
    points_all = torch.tensor(data["points"], dtype=torch.float32, device=device)
    targets_all = load_targets(data, device)
    model = MLP(equation.input_dim, equation.raw_output_dim, hidden_layers=hidden).to(device)
    if init_state is not None:
        model.load_state_dict(init_state)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    run_dir.mkdir(parents=True, exist_ok=True)

    metrics_init = evaluate_model(model, equation, dataset, device)
    history = []
    sample_count = min(samples, points_all.shape[0])
    eval_every = eval_every or max(1, epochs // 10)

    for epoch in range(1, epochs + 1):
        idx = torch.randint(0, points_all.shape[0], (sample_count,), device=device)
        points = points_all[idx].detach().clone().requires_grad_(True)
        targets = {key: targets_all[key][idx] for key in ("u", "v", "p")}

        optimizer.zero_grad()
        residuals = equation.pde_residuals(model, points)
        pde_loss = sum(torch.mean(value**2) for value in residuals.values())
        pred = equation.prediction(model, points)
        data_loss = sum(torch.mean((pred[key] - targets[key]) ** 2) for key in ("u", "v", "p"))
        boundary_loss = equation.boundary_loss(model, boundary_samples, device)
        loss = pde_loss + 8.0 * data_loss + 3.0 * boundary_loss
        loss.backward()
        optimizer.step()

        row = {
            "epoch": epoch,
            "weighted_loss": float(loss.detach().cpu()),
            "pde_loss": float(pde_loss.detach().cpu()),
            "data_loss": float(data_loss.detach().cpu()),
            "boundary_loss": float(boundary_loss.detach().cpu()),
        }
        history.append(row)
        if epoch == 1 or epoch % eval_every == 0 or epoch == epochs:
            print(json.dumps({"run": run_dir.name, **row}), flush=True)

    torch.save(model.state_dict(), run_dir / "model.pt")
    np.save(run_dir / "training_history.npy", np.array(history, dtype=object))
    config = {
        "equation": "navier_stokes_taylor_green",
        "dataset": str(dataset),
        "viscosity": viscosity,
        "hidden": hidden,
        "epochs": epochs,
        "samples": sample_count,
        "boundary_samples": boundary_samples,
        "lr": lr,
        "seed": seed,
        "init": "interpolated" if init_state is not None else "scratch",
        "data_generation": "taylor_green_analytical_pdebench_layout",
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (run_dir / "metrics_init.json").write_text(json.dumps(metrics_init, indent=2), encoding="utf-8")
    metrics_final = evaluate_model(model, equation, dataset, device, run_dir)
    return {"model": model, "metrics_init": metrics_init, "metrics": metrics_final, "history": history}


def plot_loss(run_dir: Path) -> None:
    history = [dict(item) for item in np.load(run_dir / "training_history.npy", allow_pickle=True).tolist()]
    plt.figure(figsize=(7, 5))
    for key in ("weighted_loss", "pde_loss", "data_loss", "boundary_loss"):
        plt.plot([row["epoch"] for row in history], [row[key] for row in history], label=key)
    plt.yscale("log")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title(run_dir.name)
    plt.legend()
    plt.tight_layout()
    plt.savefig(run_dir / "loss.png", dpi=200)
    plt.close()


def make_gt_pred_gif(points: np.ndarray, pred: dict[str, np.ndarray], exact: dict[str, np.ndarray], out: Path) -> None:
    x_unique = np.unique(points[:, 0])
    y_unique = np.unique(points[:, 1])
    t_unique = np.unique(points[:, 2])
    nx, ny = len(x_unique), len(y_unique)

    speed_exact = np.sqrt(exact["u"][:, 0] ** 2 + exact["v"][:, 0] ** 2)
    speed_pred = np.sqrt(pred["u"][:, 0] ** 2 + pred["v"][:, 0] ** 2)
    err = np.abs(speed_pred - speed_exact)
    vmin = min(speed_exact.min(), speed_pred.min())
    vmax = max(speed_exact.max(), speed_pred.max())
    err_max = err.max()

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))

    def frame(i: int):
        for ax in axes:
            ax.clear()
        mask = np.isclose(points[:, 2], t_unique[i])
        exact_grid = speed_exact[mask].reshape(nx, ny).T
        pred_grid = speed_pred[mask].reshape(nx, ny).T
        err_grid = err[mask].reshape(nx, ny).T
        extent = [x_unique.min(), x_unique.max(), y_unique.min(), y_unique.max()]
        panels = (
            (exact_grid, "GT |u|", vmin, vmax),
            (pred_grid, "Pred |u|", vmin, vmax),
            (err_grid, "|error|", 0.0, err_max),
        )
        artists = []
        for ax, (field, title, lo, hi) in zip(axes, panels):
            im = ax.imshow(field, origin="lower", extent=extent, aspect="auto", cmap="viridis", vmin=lo, vmax=hi)
            ax.set_title(f"{title}, t={t_unique[i]:.3f}")
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            artists.append(im)
        fig.tight_layout()
        return artists

    ani = animation.FuncAnimation(fig, frame, frames=len(t_unique), interval=250, blit=False)
    out.parent.mkdir(parents=True, exist_ok=True)
    ani.save(out, writer=animation.PillowWriter(fps=4))
    plt.close(fig)


def aggregate_metric(metrics: dict, metric_name: str) -> float:
    return float(np.mean([metrics[field][metric_name] for field in ("u", "v", "p")]))


def markdown_summary(scratch: dict, interp: dict) -> str:
    lines = [
        "# Navier-Stokes viscosity interpolation ablation",
        "",
        "## Final metrics (after target training)",
        "",
        "| Field | Metric | Scratch c | Interpolated pretrain c | Delta | Relative improvement |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for field in ("u", "v", "p"):
        for metric_name in ("mse", "relative_l2"):
            before = scratch["metrics"][field][metric_name]
            after = interp["metrics"][field][metric_name]
            delta = after - before
            improvement = (before - after) / before * 100.0 if before else 0.0
            lines.append(f"| {field} | {metric_name} | {before:.6g} | {after:.6g} | {delta:.6g} | {improvement:.2f}% |")

    lines.extend(
        [
            "",
            "## Initial metrics (before target training)",
            "",
            "| Field | Metric | Scratch c | Interpolated init c | Delta | Relative improvement |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for field in ("u", "v", "p"):
        for metric_name in ("mse", "relative_l2"):
            before = scratch["metrics_init"][field][metric_name]
            after = interp["metrics_init"][field][metric_name]
            delta = after - before
            improvement = (before - after) / before * 100.0 if before else 0.0
            lines.append(f"| {field} | {metric_name} | {before:.6g} | {after:.6g} | {delta:.6g} | {improvement:.2f}% |")

    lines.extend(
        [
            "",
            "## Aggregated mean relative L2",
            "",
            f"- Scratch init: {aggregate_metric(scratch['metrics_init'], 'relative_l2'):.6g}",
            f"- Interpolated init: {aggregate_metric(interp['metrics_init'], 'relative_l2'):.6g}",
            f"- Scratch final: {aggregate_metric(scratch['metrics'], 'relative_l2'):.6g}",
            f"- Interpolated final: {aggregate_metric(interp['metrics'], 'relative_l2'):.6g}",
        ]
    )
    return "\n".join(lines)


def run(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    data_root = Path(args.data_root)
    if not data_root.is_absolute():
        data_root = REPO_ROOT / data_root
    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = REPO_ROOT / output_root
    output_root.mkdir(parents=True, exist_ok=True)
    data_root.mkdir(parents=True, exist_ok=True)

    datasets = {
        "a": data_root / f"ns_tg_nu_{args.nu_a:g}.npz",
        "b": data_root / f"ns_tg_nu_{args.nu_b:g}.npz",
        "c": data_root / f"ns_tg_nu_{args.nu_c:g}.npz",
    }
    dataset_meta = {}
    if not args.skip_generate:
        dataset_meta["a"] = generate_taylor_green_dataset(datasets["a"], args.nu_a, args.spatial_resolution, args.time_steps)
        dataset_meta["b"] = generate_taylor_green_dataset(datasets["b"], args.nu_b, args.spatial_resolution, args.time_steps)
        dataset_meta["c"] = generate_taylor_green_dataset(datasets["c"], args.nu_c, args.spatial_resolution, args.time_steps)

    hidden = tuple(args.hidden)
    if not args.skip_sources:
        train_model(
            run_root / "source_nu_a",
            datasets["a"],
            args.nu_a,
            device,
            hidden,
            args.source_epochs,
            args.samples,
            args.boundary_samples,
            args.lr,
            args.seed,
        )
        train_model(
            run_root / "source_nu_b",
            datasets["b"],
            args.nu_b,
            device,
            hidden,
            args.source_epochs,
            args.samples,
            args.boundary_samples,
            args.lr,
            args.seed + 1,
        )

    state_a = torch.load(run_root / "source_nu_a" / "model.pt", map_location=device)
    state_b = torch.load(run_root / "source_nu_b" / "model.pt", map_location=device)
    alpha = (args.nu_c - args.nu_a) / (args.nu_b - args.nu_a)
    init_c = interpolate_state_dict(state_a, state_b, alpha)

    scratch = train_model(
        run_root / "target_nu_c_scratch",
        datasets["c"],
        args.nu_c,
        device,
        hidden,
        args.target_epochs,
        args.samples,
        args.boundary_samples,
        args.lr,
        args.seed + 2,
    )
    interp = train_model(
        run_root / "target_nu_c_interpolated",
        datasets["c"],
        args.nu_c,
        device,
        hidden,
        args.target_epochs,
        args.samples,
        args.boundary_samples,
        args.lr,
        args.seed + 2,
        init_state=init_c,
    )

    table = markdown_summary(scratch, interp)
    summary = {
        "nu_a": args.nu_a,
        "nu_b": args.nu_b,
        "nu_c": args.nu_c,
        "interpolation_alpha": alpha,
        "dataset_meta": dataset_meta,
        "scratch": {
            "metrics_init": scratch["metrics_init"],
            "metrics_final": scratch["metrics"],
            "mean_relative_l2_init": aggregate_metric(scratch["metrics_init"], "relative_l2"),
            "mean_relative_l2_final": aggregate_metric(scratch["metrics"], "relative_l2"),
        },
        "interpolated": {
            "metrics_init": interp["metrics_init"],
            "metrics_final": interp["metrics"],
            "mean_relative_l2_init": aggregate_metric(interp["metrics_init"], "relative_l2"),
            "mean_relative_l2_final": aggregate_metric(interp["metrics"], "relative_l2"),
        },
    }
    (output_root / "metrics_table.md").write_text(table + "\n", encoding="utf-8")
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output_root / "experiment_config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    print(table)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PINN weight interpolation pretraining for viscosity-parametric Taylor-Green NS.")
    parser.add_argument("--output-root", default="runs/ablation/ns_viscosity_interpolation")
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--nu-a", type=float, default=0.005)
    parser.add_argument("--nu-b", type=float, default=0.02)
    parser.add_argument("--nu-c", type=float, default=0.01)
    parser.add_argument("--spatial-resolution", type=int, default=32)
    parser.add_argument("--time-steps", type=int, default=21)
    parser.add_argument("--hidden", type=int, nargs="+", default=[64, 64, 64, 64])
    parser.add_argument("--source-epochs", type=int, default=1500)
    parser.add_argument("--target-epochs", type=int, default=800)
    parser.add_argument("--samples", type=int, default=2048)
    parser.add_argument("--boundary-samples", type=int, default=512)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--skip-generate", action="store_true")
    parser.add_argument("--skip-sources", action="store_true", help="Reuse source models in output-root/runs.")
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
