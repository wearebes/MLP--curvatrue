from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from project_config import DEFAULT_PROJECT_CONFIG_PATH, load_project_config
from project_runtime import cleanup_bytecode_caches, disable_bytecode_cache

from .paper_alignment import (
    OriginPredictorLite,
    compute_metrics,
    format_metric_delta,
    format_metric_triple,
    load_payload,
    normalized_distance,
)
from .paper_reference import PAPER_UNIFORM_REFERENCES, get_paper_reference

disable_bytecode_cache()

SCALE_VARIANTS = ("raw", "to_model_h", "from_model_h")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare origin-model sensitivity to stencil scale and h mismatch.")
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
        help="Optional explicit report path. Defaults to test/results/scale_sensitivity_MMDD.txt.",
    )
    args = parser.parse_args()

    project_config = load_project_config(args.config)
    report = build_scale_sensitivity_report(project_config.project_root, config_path=args.config)
    output_path = resolve_output_path(project_config.project_root, args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report + "\n", encoding="utf-8")
    print(report)
    print(f"\nScale sensitivity report saved to {output_path}")


def build_scale_sensitivity_report(project_root: Path, *, config_path: str | Path | None = None) -> str:
    project_config = load_project_config(config_path or DEFAULT_PROJECT_CONFIG_PATH)
    origin_predictor = OriginPredictorLite(project_root)
    data_root = project_config.project.data_dir

    case_wins = {name: 0 for name in SCALE_VARIANTS}
    step_wins = {name: 0 for name in SCALE_VARIANTS}
    total_cases = 0
    total_steps = 0

    lines: list[str] = []
    lines.append("Scale Sensitivity Report")
    lines.append(f"Generated: {datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"Config: {project_config.source_path}")
    lines.append(f"Data root: {data_root}")
    lines.append("")

    for scenario in project_config.evaluation.scenarios:
        exp_id = scenario.exp_id
        if exp_id not in PAPER_UNIFORM_REFERENCES:
            continue

        total_cases += 1
        model_h = read_model_h(project_root / "model" / "origin" / f"trainStats_{int(scenario.rho_model)}.csv")
        case_h = float(scenario.h)
        factor_to_model = model_h / case_h
        factor_from_model = case_h / model_h
        case_scores = {name: 0.0 for name in SCALE_VARIANTS}

        lines.append(f"[{exp_id}] {PAPER_UNIFORM_REFERENCES[exp_id].table_id}")
        lines.append(
            f"meta: case_h={case_h:.9e} | model_h={model_h:.9e} | "
            f"to_model_factor={factor_to_model:.9e} | from_model_factor={factor_from_model:.9e}"
        )

        for step in project_config.evaluation.test_iters:
            total_steps += 1
            payload = load_payload(data_root / exp_id / f"iter_{int(step)}.h5")
            paper_ref = get_paper_reference(exp_id, int(step))
            stencils = payload["stencils"]
            predictions = {
                "raw": origin_predictor.predict(int(scenario.rho_model), stencils),
                "to_model_h": origin_predictor.predict(int(scenario.rho_model), stencils * factor_to_model),
                "from_model_h": origin_predictor.predict(int(scenario.rho_model), stencils * factor_from_model),
            }
            step_scores: dict[str, float] = {}
            lines.append(f"  step={int(step)}")
            for variant_name, prediction in predictions.items():
                metrics = compute_metrics(payload["target"], prediction)
                score = normalized_distance(metrics, paper_ref.paper_model)
                case_scores[variant_name] += score
                step_scores[variant_name] = score
                lines.append(
                    f"    {variant_name}: "
                    f"{format_metric_triple(metrics)} | "
                    f"|d_paper|={format_metric_delta(metrics, paper_ref.paper_model)} | "
                    f"score={score:.6f}"
                )
            winner = min(step_scores, key=step_scores.get)
            step_wins[winner] += 1
            lines.append(
                "    step_summary: "
                f"winner={winner} | "
                f"raw={step_scores['raw']:.6f} | "
                f"to_model_h={step_scores['to_model_h']:.6f} | "
                f"from_model_h={step_scores['from_model_h']:.6f}"
            )

        case_winner = min(case_scores, key=case_scores.get)
        case_wins[case_winner] += 1
        lines.append(
            "  case_summary: "
            f"winner={case_winner} | "
            f"raw={case_scores['raw']:.6f} | "
            f"to_model_h={case_scores['to_model_h']:.6f} | "
            f"from_model_h={case_scores['from_model_h']:.6f}"
        )
        lines.append("")

    lines.insert(
        4,
        "Summary: "
        + ", ".join(
            f"{name} {case_wins[name]}/{max(total_cases,1)} cases {step_wins[name]}/{max(total_steps,1)} steps"
            for name in SCALE_VARIANTS
        ),
    )
    return "\n".join(lines).rstrip()


def read_model_h(csv_path: Path) -> float:
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        next(reader, None)
        for row in reader:
            if len(row) < 2:
                continue
            if row[0].strip().strip('"') == "h":
                return float(row[1])
    raise ValueError(f"Missing h row in {csv_path}")


def resolve_output_path(project_root: Path, raw_output: str) -> Path:
    if raw_output:
        output_path = Path(raw_output)
        if not output_path.is_absolute():
            output_path = project_root / output_path
        return output_path
    stamp = datetime.now().strftime("%m%d")
    return project_root / "test" / "results" / f"scale_sensitivity_{stamp}.txt"


if __name__ == "__main__":
    try:
        main()
    finally:
        cleanup_bytecode_caches(PROJECT_ROOT)
