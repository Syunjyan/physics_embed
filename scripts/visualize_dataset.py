from __future__ import annotations

from pathlib import Path
import argparse
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


DEFAULT_FIELDS = {
    "heat": ("u", "f"),
    "burgers": ("u", "v", "fu", "fv"),
    "navier_stokes": ("u", "v", "p", "fu", "fv"),
    "linear_elasticity": ("ux", "uy", "sigmaxx", "sigmayy", "sigmaxy", "fx", "fy"),
}


def _infer_equation(dataset: np.lib.npyio.NpzFile, path: Path) -> str:
    if "equation" in dataset.files:
        value = dataset["equation"]
        return str(value.item() if value.shape == () else value)
    return path.stem


def _slice_mask(points: np.ndarray, t: float | None) -> tuple[np.ndarray, str]:
    if points.shape[1] == 2:
        return np.ones(points.shape[0], dtype=bool), ""
    unique_t = np.unique(points[:, 2])
    target = unique_t[len(unique_t) // 2] if t is None else unique_t[np.argmin(np.abs(unique_t - t))]
    return np.isclose(points[:, 2], target), f" at t={target:.3f}"


def _grid(points: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_unique = np.unique(points[:, 0])
    y_unique = np.unique(points[:, 1])
    return x_unique, y_unique, values.reshape(len(x_unique), len(y_unique)).T


def _cmap_and_limits(field: np.ndarray) -> tuple[str, float | None, float | None]:
    min_value = float(np.nanmin(field))
    max_value = float(np.nanmax(field))
    if min_value < 0.0 < max_value:
        bound = max(abs(min_value), abs(max_value))
        return "coolwarm", -bound, bound
    return "viridis", None, None


def visualize_dataset(dataset_path: Path, out: Path, fields: list[str] | None, t: float | None) -> None:
    data = np.load(dataset_path)
    equation = _infer_equation(data, dataset_path)
    points = data["points"]
    chosen_fields = fields or [field for field in DEFAULT_FIELDS.get(equation, ()) if field in data.files]
    if not chosen_fields:
        chosen_fields = [key for key in data.files if key not in {"points", "equation"}]

    mask, time_title = _slice_mask(points, t)
    sliced_points = points[mask]
    cols = min(3, len(chosen_fields))
    rows = int(np.ceil(len(chosen_fields) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.6 * rows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")

    for idx, field_name in enumerate(chosen_fields):
        ax = axes.ravel()[idx]
        values = data[field_name][mask, 0]
        x_unique, y_unique, grid = _grid(sliced_points[:, :2], values)
        cmap, vmin, vmax = _cmap_and_limits(grid)
        im = ax.imshow(
            grid,
            origin="lower",
            extent=[x_unique.min(), x_unique.max(), y_unique.min(), y_unique.max()],
            aspect="auto",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_title(field_name)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.axis("on")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(f"{equation} dataset fields{time_title}")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"saved {out}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Visualize exact fields stored in a generated dataset.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--fields", nargs="*", default=None)
    parser.add_argument("--t", type=float, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    visualize_dataset(args.dataset, args.out, args.fields, args.t)


if __name__ == "__main__":
    main()
