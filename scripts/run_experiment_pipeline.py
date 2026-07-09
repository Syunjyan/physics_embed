from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import argparse
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from physics_embed.data import generate_dataset
from physics_embed.evaluate import evaluate_run
from physics_embed.train import train_from_dataset


@dataclass(frozen=True)
class ExperimentConfig:
    equation: str
    dataset: Path
    run_dir: Path
    spatial_resolution: int
    time_steps: int
    epochs: int
    samples: int
    boundary_samples: int
    hidden: tuple[int, ...]
    lr: float
    pde_weight: float
    data_weight: float
    boundary_weight: float


EXPERIMENTS: dict[str, ExperimentConfig] = {
    "heat": ExperimentConfig(
        equation="heat",
        dataset=Path("data/heat.npz"),
        run_dir=Path("runs/baseline/heat"),
        spatial_resolution=48,
        time_steps=1,
        epochs=3000,
        samples=2048,
        boundary_samples=512,
        hidden=(64, 64, 64, 64),
        lr=1e-3,
        pde_weight=1.0,
        data_weight=5.0,
        boundary_weight=3.0,
    ),
    "burgers": ExperimentConfig(
        equation="burgers",
        dataset=Path("data/burgers.npz"),
        run_dir=Path("runs/baseline/burgers"),
        spatial_resolution=32,
        time_steps=21,
        epochs=2500,
        samples=4096,
        boundary_samples=1024,
        hidden=(64, 64, 64, 64),
        lr=1e-3,
        pde_weight=1.0,
        data_weight=5.0,
        boundary_weight=3.0,
    ),
    "navier_stokes": ExperimentConfig(
        equation="navier_stokes",
        dataset=Path("data/navier_stokes.npz"),
        run_dir=Path("runs/baseline/navier_stokes"),
        spatial_resolution=32,
        time_steps=21,
        epochs=3000,
        samples=4096,
        boundary_samples=1024,
        hidden=(64, 64, 64, 64),
        lr=8e-4,
        pde_weight=1.0,
        data_weight=10.0,
        boundary_weight=3.0,
    ),
    "linear_elasticity": ExperimentConfig(
        equation="linear_elasticity",
        dataset=Path("data/linear_elasticity.npz"),
        run_dir=Path("runs/baseline/linear_elasticity"),
        spatial_resolution=48,
        time_steps=1,
        epochs=5000,
        samples=2048,
        boundary_samples=1024,
        hidden=(64, 64, 64, 64, 64),
        lr=5e-4,
        pde_weight=0.1,
        data_weight=8.0,
        boundary_weight=3.0,
    ),
}


def _select_experiments(names: list[str] | None) -> list[ExperimentConfig]:
    if not names:
        return list(EXPERIMENTS.values())
    unknown = sorted(set(names) - set(EXPERIMENTS))
    if unknown:
        choices = ", ".join(sorted(EXPERIMENTS))
        raise ValueError(f"Unknown experiment(s): {unknown}. Choose from: {choices}")
    return [EXPERIMENTS[name] for name in names]


def _write_pipeline_config(configs: list[ExperimentConfig], out: Path, device: str) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "device": device,
        "experiments": [
            {
                **asdict(config),
                "dataset": str(config.dataset),
                "run_dir": str(config.run_dir),
                "hidden": list(config.hidden),
            }
            for config in configs
        ],
    }
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run_pipeline(
    configs: list[ExperimentConfig],
    device: str,
    skip_generate: bool,
    skip_train: bool,
    skip_evaluate: bool,
    force_generate: bool,
) -> None:
    _write_pipeline_config(configs, Path("runs/baseline/pipeline_config.json"), device)

    for config in configs:
        print(f"=== {config.equation} ===", flush=True)
        if not skip_generate and (force_generate or not config.dataset.exists()):
            generate_dataset(
                config.equation,
                config.dataset,
                spatial_resolution=config.spatial_resolution,
                time_steps=config.time_steps,
            )
        if not skip_train:
            train_from_dataset(
                equation_name=config.equation,
                dataset_path=config.dataset,
                output_dir=config.run_dir,
                epochs=config.epochs,
                samples=config.samples,
                boundary_samples=config.boundary_samples,
                lr=config.lr,
                hidden=config.hidden,
                seed=1234,
                device_name=device,
                pde_weight=config.pde_weight,
                data_weight=config.data_weight,
                boundary_weight=config.boundary_weight,
            )
        if not skip_evaluate:
            evaluate_run(config.run_dir, config.dataset, t_value=None, device_name=device)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the full dataset -> train -> evaluate PINN pipeline.")
    parser.add_argument("--only", nargs="*", choices=sorted(EXPERIMENTS), help="Run selected experiments only.")
    parser.add_argument("--device", default="cuda:1", help="Torch device. Default uses physical GPU 1.")
    parser.add_argument("--skip-generate", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-evaluate", action="store_true")
    parser.add_argument("--force-generate", action="store_true", help="Regenerate datasets even if files exist.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    configs = _select_experiments(args.only)
    run_pipeline(
        configs=configs,
        device=args.device,
        skip_generate=args.skip_generate,
        skip_train=args.skip_train,
        skip_evaluate=args.skip_evaluate,
        force_generate=args.force_generate,
    )


if __name__ == "__main__":
    main()
