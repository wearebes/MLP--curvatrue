from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from project_config import DEFAULT_PROJECT_CONFIG_PATH, load_project_config
from project_runtime import cleanup_bytecode_caches, disable_bytecode_cache

from .paper_alignment import load_origin_stats, load_payload

disable_bytecode_cache()

SHIFT_VARIANTS = ("raw", "to_model_h", "from_model_h")


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure stencil distribution shift against origin trainStats.")
    parser.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_PROJECT_CONFIG_PATH),
        help=f"Shared project config path (default: {DEFAULT_PROJECT_CONFIG_PATH})",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Optional explicit report path. Defaults to test/results/distribution_shift_MMDD.txt.",
    )
    args = parser.parse_args()

    project_config = load_project_config(args.config)
    report = build_distribution_shift_report(project_config.project_root, config_path=args.config)
    output_path = resolve_output_path(project_config.project_root, args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report + "\n", encoding="utf-8")
    print(report)
    print(f"\nDistribution shift report saved to {output_path}")


def build_distribution_shift_report(project_root: Path, *, config_path: str | Path | None = None) -> str:
    project_config = load_project_config(config_path or DEFAULT_PROJECT_CONFIG_PATH)
    data_root = project_config.project.data_dir
    origin_model_root = project_root / "model" / "origin"

    lines: list[str] = []
    lines.append("Distribution Shift Report")
    lines.append(f"Generated: {datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"Config: {project_config.source_path}")
    lines.append(f"Data root: {data_root}")
    lines.append("")

    for scenario in project_config.evaluation.scenarios:
        stats_mu, stats_sigma = load_origin_stats(origin_model_root / f"trainStats_{int(scenario.rho_model)}.csv")
        case_h = float(scenario.h)
        model_h = 1.0 / float(int(scenario.rho_model) - 1)
        to_model_factor = model_h / case_h
        from_model_factor = case_h / model_h

        lines.append(f"[{scenario.exp_id}] rho_model={scenario.rho_model}")
        lines.append(
            f"meta: case_h={case_h:.9e} | model_h={model_h:.9e} | "
            f"to_model_factor={to_model_factor:.9e} | from_model_factor={from_model_factor:.9e}"
        )
        for step in project_config.evaluation.test_iters:
            payload = load_payload(data_root / scenario.exp_id / f"iter_{int(step)}.h5")
            stencils = payload["stencils"]
            variants = {
                "raw": stencils,
                "to_model_h": stencils * to_model_factor,
                "from_model_h": stencils * from_model_factor,
            }
            lines.append(f"  step={int(step)}")
            for variant_name, variant_stencils in variants.items():
                summary = summarize_shift(variant_stencils, stats_mu, stats_sigma)
                lines.append(
                    f"    {variant_name}: "
                    f"mean_shift={summary['mean_shift']:.6e} | "
                    f"mean_abs_shift_z={summary['mean_abs_shift_z']:.6e} | "
                    f"std_ratio_mean={summary['std_ratio_mean']:.6e} | "
                    f"std_ratio_maxdev={summary['std_ratio_maxdev']:.6e}"
                )
        lines.append("")
    return "\n".join(lines).rstrip()


def summarize_shift(stencils: np.ndarray, train_mu: np.ndarray, train_sigma: np.ndarray) -> dict[str, float]:
    sample_mu = np.mean(stencils, axis=0)
    sample_sigma = np.std(stencils, axis=0)
    safe_sigma = np.maximum(train_sigma.astype(np.float64), 1e-15)
    mean_shift = float(np.mean(np.abs(sample_mu - train_mu)))
    mean_abs_shift_z = float(np.mean(np.abs((sample_mu - train_mu) / safe_sigma)))
    std_ratio = sample_sigma / safe_sigma
    std_ratio_mean = float(np.mean(std_ratio))
    std_ratio_maxdev = float(np.max(np.abs(std_ratio - 1.0)))
    return {
        "mean_shift": mean_shift,
        "mean_abs_shift_z": mean_abs_shift_z,
        "std_ratio_mean": std_ratio_mean,
        "std_ratio_maxdev": std_ratio_maxdev,
    }


def resolve_output_path(project_root: Path, raw_output: str) -> Path:
    if raw_output:
        output_path = Path(raw_output)
        if not output_path.is_absolute():
            output_path = project_root / output_path
        return output_path
    stamp = datetime.now().strftime("%m%d")
    return project_root / "test" / "results" / f"distribution_shift_{stamp}.txt"


if __name__ == "__main__":
    try:
        main()
    finally:
        cleanup_bytecode_caches(PROJECT_ROOT)
