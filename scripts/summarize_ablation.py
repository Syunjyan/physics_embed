from __future__ import annotations

from pathlib import Path
import argparse
import json


def _load_metrics(run_dir: Path) -> dict:
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing metrics file: {metrics_path}")
    return json.loads(metrics_path.read_text(encoding="utf-8"))


def _format_float(value: float) -> str:
    return f"{value:.6g}"


def summarize(output_root: Path, without_name: str, with_name: str, label: str) -> str:
    without = _load_metrics(output_root / without_name)
    with_constraint = _load_metrics(output_root / with_name)
    keys = [key for key in without if key in with_constraint]

    lines = [
        f"| Field | Metric | Without {label} | With {label} | Delta | Relative improvement |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for field in keys:
        metric_names = sorted(set(without[field]) & set(with_constraint[field]))
        for metric_name in metric_names:
            before = float(without[field][metric_name])
            after = float(with_constraint[field][metric_name])
            delta = after - before
            improvement = (before - after) / before * 100.0 if before != 0.0 else 0.0
            lines.append(
                "| "
                f"{field} | {metric_name} | {_format_float(before)} | {_format_float(after)} | "
                f"{_format_float(delta)} | {improvement:.2f}% |"
            )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Print a Markdown table for ablation metrics.")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--without-name", default="without_reduced_model")
    parser.add_argument("--with-name", default="with_reduced_model")
    parser.add_argument("--label", default="constraint")
    parser.add_argument("--out", type=Path, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    table = summarize(args.output_root, args.without_name, args.with_name, args.label)
    print(table)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(table + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
