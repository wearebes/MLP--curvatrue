from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import re
import sys

import h5py
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from generate.dataset_test import (
    TEST_DATA_MODE_PAPER_ALIGNED,
    TEST_DATA_MODE_REBUILT,
    generate_test_datasets,
    resolve_test_output_root,
)
from project_config import DEFAULT_PROJECT_CONFIG_PATH, load_project_config
from project_runtime import cleanup_bytecode_caches, disable_bytecode_cache

from .paper_reference import PAPER_UNIFORM_REFERENCES, PaperMetricTriple, PaperStepReference, get_paper_reference

disable_bytecode_cache()


@dataclass(frozen=True)
class MetricTriple:
    mae: float
    max_ae: float
    mse: float


@dataclass(frozen=True)
class DatasetSummary:
    count: int
    minimum: float
    maximum: float
    mean: float
    std: float


class OriginPredictorLite:
    def __init__(self, project_root: Path):
        self.model_dir = project_root / "model" / "origin"
        self.cache: dict[int, tuple[list[np.ndarray], list[np.ndarray], np.ndarray, np.ndarray]] = {}

    def predict(self, rho: int, stencils_raw: np.ndarray, batch_size: int = 16384) -> np.ndarray:
        weights, biases, mu, sigma = self._ensure_bundle(rho)
        outputs: list[np.ndarray] = []
        for start in range(0, stencils_raw.shape[0], batch_size):
            chunk = stencils_raw[start:start + batch_size].astype(np.float32, copy=False)
            norm_chunk = (chunk - mu) / sigma
            outputs.append(forward_relu_mlp(norm_chunk, weights, biases).astype(np.float32, copy=False))
        return np.concatenate(outputs, axis=0) if outputs else np.empty((0,), dtype=np.float32)

    def _ensure_bundle(self, rho: int) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray, np.ndarray]:
        if rho not in self.cache:
            weights, biases = load_origin_weights(self.model_dir / f"nnet_{rho}.h5")
            mu, sigma = load_origin_stats(self.model_dir / f"trainStats_{rho}.csv")
            self.cache[rho] = (weights, biases, mu, sigma)
        return self.cache[rho]


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare current rebuilt explicit tests against a paper-aligned mode.")
    parser.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_PROJECT_CONFIG_PATH),
        help=f"Shared project config path (default: {DEFAULT_PROJECT_CONFIG_PATH})",
    )
    parser.add_argument(
        "--refresh-paper-aligned",
        action="store_true",
        help="Regenerate data/paper_aligned before building the report.",
    )
    parser.add_argument(
        "--refresh-rebuilt",
        action="store_true",
        help="Regenerate the current rebuilt explicit test HDF5 before building the report.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Optional explicit report path. Defaults to test/results/paper_alignment_MMDD.txt.",
    )
    args = parser.parse_args()

    project_config = load_project_config(args.config)
    ensure_test_datasets(project_config, TEST_DATA_MODE_REBUILT, refresh=args.refresh_rebuilt)
    ensure_test_datasets(project_config, TEST_DATA_MODE_PAPER_ALIGNED, refresh=args.refresh_paper_aligned)

    report = build_paper_alignment_report(project_config.project_root, config_path=args.config)
    output_path = resolve_output_path(project_config.project_root, args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report + "\n", encoding="utf-8")
    print(report)
    print(f"\nPaper alignment report saved to {output_path}")


def ensure_test_datasets(project_config, mode: str, *, refresh: bool) -> None:
    root = resolve_test_output_root(project_config.project.data_dir, mode)
    if refresh or not test_dataset_root_is_complete(root, project_config.evaluation.scenarios, project_config.evaluation.test_iters):
        generate_test_datasets(project_config.project.data_dir, project_config=project_config, mode=mode)


def test_dataset_root_is_complete(root: Path, scenarios, test_iters: tuple[int, ...]) -> bool:
    for scenario in scenarios:
        exp_dir = root / scenario.exp_id
        if not (exp_dir / "meta.json").exists():
            return False
        for step in test_iters:
            if not (exp_dir / f"iter_{int(step)}.h5").exists():
                return False
    return True


def build_paper_alignment_report(project_root: Path, *, config_path: str | Path | None = None) -> str:
    project_config = load_project_config(config_path or DEFAULT_PROJECT_CONFIG_PATH)
    origin_predictor = OriginPredictorLite(project_root)
    rebuilt_root = resolve_test_output_root(project_config.project.data_dir, TEST_DATA_MODE_REBUILT)
    aligned_root = resolve_test_output_root(project_config.project.data_dir, TEST_DATA_MODE_PAPER_ALIGNED)

    lines: list[str] = []
    lines.append("Paper Alignment Report")
    lines.append(f"Generated: {datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"Config: {project_config.source_path}")
    lines.append(f"Rebuilt root: {rebuilt_root}")
    lines.append(f"Paper-aligned root: {aligned_root}")
    lines.append("")

    case_win_count = 0
    total_cases = 0
    step_win_count = 0
    total_steps = 0

    for scenario in project_config.evaluation.scenarios:
        exp_id = scenario.exp_id
        if exp_id not in PAPER_UNIFORM_REFERENCES:
            continue

        total_cases += 1
        rebuilt_meta = load_meta(rebuilt_root / exp_id / "meta.json")
        aligned_meta = load_meta(aligned_root / exp_id / "meta.json")
        case_current_score = 0.0
        case_aligned_score = 0.0
        case_origin_trend_ok = True

        lines.append(f"[{exp_id}] {PAPER_UNIFORM_REFERENCES[exp_id].table_id}")
        lines.append(
            "meta: "
            f"h={float(rebuilt_meta['h']):.9e} | "
            f"rho_eq={float(rebuilt_meta['rho_eq']):.4f} | "
            f"paper_samples={PAPER_UNIFORM_REFERENCES[exp_id].sample_count} | "
            f"rebuilt_samples={count_samples(rebuilt_root / exp_id / 'iter_5.h5')} | "
            f"paper_aligned_samples={count_samples(aligned_root / exp_id / 'iter_5.h5')}"
        )
        lines.append(
            "mode: "
            f"current_rebuilt sampling={rebuilt_meta.get('sampling_rule', 'n/a')} | "
            f"paper_aligned sampling={aligned_meta.get('sampling_rule', 'n/a')}"
        )

        for step in project_config.evaluation.test_iters:
            total_steps += 1
            paper_ref = get_paper_reference(exp_id, int(step))
            rebuilt_payload = load_payload(rebuilt_root / exp_id / f"iter_{int(step)}.h5")
            aligned_payload = load_payload(aligned_root / exp_id / f"iter_{int(step)}.h5")

            rebuilt_numerical = compute_metrics(rebuilt_payload["target"], rebuilt_payload["numerical"])
            aligned_numerical = compute_metrics(aligned_payload["target"], aligned_payload["numerical"])
            rebuilt_origin = compute_origin_metrics(origin_predictor, scenario.rho_model, rebuilt_payload)
            aligned_origin = compute_origin_metrics(origin_predictor, scenario.rho_model, aligned_payload)

            current_origin_score = normalized_distance(rebuilt_origin, paper_ref.paper_model)
            aligned_origin_score = normalized_distance(aligned_origin, paper_ref.paper_model)
            case_current_score += current_origin_score
            case_aligned_score += aligned_origin_score
            if aligned_origin_score < current_origin_score:
                step_win_count += 1

            target_line = format_target_line(rebuilt_payload, aligned_payload)
            trend_line = format_trend_line(paper_ref, rebuilt_numerical, rebuilt_origin, aligned_numerical, aligned_origin)
            case_origin_trend_ok = case_origin_trend_ok and trend_matches_paper(
                paper_ref,
                aligned_numerical,
                aligned_origin,
            )

            lines.append(f"  step={int(step)}")
            lines.append(
                "    sample_count: "
                f"paper={PAPER_UNIFORM_REFERENCES[exp_id].sample_count} | "
                f"current_rebuilt={rebuilt_payload['target'].shape[0]} | "
                f"paper_aligned={aligned_payload['target'].shape[0]}"
            )
            lines.append(f"    target: {target_line}")
            lines.append(
                "    numerical: "
                f"paper={format_metric_triple(paper_ref.numerical)} | "
                f"current_rebuilt={format_metric_triple(rebuilt_numerical)} | "
                f"paper_aligned={format_metric_triple(aligned_numerical)} | "
                f"|d_paper| current={format_metric_delta(rebuilt_numerical, paper_ref.numerical)} | "
                f"|d_paper| aligned={format_metric_delta(aligned_numerical, paper_ref.numerical)}"
            )
            lines.append(
                "    origin: "
                f"paper={format_metric_triple(paper_ref.paper_model)} | "
                f"current_rebuilt={format_metric_triple(rebuilt_origin)} | "
                f"paper_aligned={format_metric_triple(aligned_origin)} | "
                f"|d_paper| current={format_metric_delta(rebuilt_origin, paper_ref.paper_model)} | "
                f"|d_paper| aligned={format_metric_delta(aligned_origin, paper_ref.paper_model)}"
            )
            lines.append(
                "    origin_distance_to_paper: "
                f"current={current_origin_score:.6f} | "
                f"aligned={aligned_origin_score:.6f} | "
                f"winner={'paper_aligned' if aligned_origin_score < current_origin_score else 'current_rebuilt'}"
            )
            lines.append(f"    trend: {trend_line}")

        if case_aligned_score < case_current_score:
            case_win_count += 1
        lines.append(
            "  case_summary: "
            f"origin_distance current={case_current_score:.6f} | "
            f"aligned={case_aligned_score:.6f} | "
            f"winner={'paper_aligned' if case_aligned_score < case_current_score else 'current_rebuilt'} | "
            f"aligned_origin_trend_matches_paper={case_origin_trend_ok}"
        )
        lines.append("")

    lines.insert(
        5,
        "Summary: "
        f"paper_aligned closer on {case_win_count}/{max(total_cases, 1)} cases "
        f"and {step_win_count}/{max(total_steps, 1)} case-steps for the origin row.",
    )
    return "\n".join(lines).rstrip()


def compute_origin_metrics(origin_predictor: OriginPredictorLite, rho_model: int, payload: dict[str, np.ndarray]) -> MetricTriple:
    prediction = origin_predictor.predict(rho_model, payload["stencils"])
    return compute_metrics(payload["target"], prediction)


def compute_metrics(target: np.ndarray, pred: np.ndarray) -> MetricTriple:
    error = np.asarray(pred, dtype=np.float64) - np.asarray(target, dtype=np.float64)
    abs_error = np.abs(error)
    return MetricTriple(
        mae=float(abs_error.mean()) if abs_error.size else 0.0,
        max_ae=float(abs_error.max()) if abs_error.size else 0.0,
        mse=float(np.mean(error**2)) if error.size else 0.0,
    )


def load_meta(meta_path: Path) -> dict[str, object]:
    return json.loads(meta_path.read_text(encoding="utf-8"))


def load_payload(h5_path: Path) -> dict[str, np.ndarray]:
    with h5py.File(h5_path, "r") as handle:
        return {
            "indices": handle["indices"][:].astype(np.int64),
            "stencils": handle["stencils_raw"][:].astype(np.float64),
            "target": handle["hkappa_target"][:].astype(np.float64),
            "numerical": handle["hkappa_fd"][:].astype(np.float64),
        }


def count_samples(h5_path: Path) -> int:
    with h5py.File(h5_path, "r") as handle:
        return int(handle["hkappa_target"].shape[0])


def summarize_array(values: np.ndarray) -> DatasetSummary:
    if values.size == 0:
        return DatasetSummary(count=0, minimum=0.0, maximum=0.0, mean=0.0, std=0.0)
    cast = np.asarray(values, dtype=np.float64)
    return DatasetSummary(
        count=int(cast.size),
        minimum=float(cast.min()),
        maximum=float(cast.max()),
        mean=float(cast.mean()),
        std=float(cast.std()),
    )


def format_target_line(rebuilt_payload: dict[str, np.ndarray], aligned_payload: dict[str, np.ndarray]) -> str:
    rebuilt_target = summarize_array(rebuilt_payload["target"])
    aligned_target = summarize_array(aligned_payload["target"])
    same_indices = np.array_equal(rebuilt_payload["indices"], aligned_payload["indices"])
    same_shape = rebuilt_payload["target"].shape == aligned_payload["target"].shape
    direct_gap = ""
    if same_shape:
        diff = np.asarray(aligned_payload["target"], dtype=np.float64) - np.asarray(rebuilt_payload["target"], dtype=np.float64)
        direct_gap = (
            f" | direct_diff_mae={np.abs(diff).mean():.6e}"
            f" | direct_diff_max={np.abs(diff).max():.6e}"
            f" | same_indices={same_indices}"
        )
    else:
        direct_gap = f" | direct_diff=n/a | same_indices={same_indices}"
    return (
        f"current_rebuilt(count={rebuilt_target.count}, mean={rebuilt_target.mean:.6e}, std={rebuilt_target.std:.6e}) | "
        f"paper_aligned(count={aligned_target.count}, mean={aligned_target.mean:.6e}, std={aligned_target.std:.6e})"
        f"{direct_gap}"
    )


def normalized_distance(observed: MetricTriple, reference: PaperMetricTriple) -> float:
    eps = 1e-15
    return (
        abs(observed.mae - reference.mae) / max(reference.mae, eps)
        + abs(observed.max_ae - reference.max_ae) / max(reference.max_ae, eps)
        + abs(observed.mse - reference.mse) / max(reference.mse, eps)
    )


def format_metric_triple(metrics: MetricTriple | PaperMetricTriple) -> str:
    return (
        f"MAE={metrics.mae:.6e}, "
        f"MaxAE={metrics.max_ae:.6e}, "
        f"MSE={metrics.mse:.6e}"
    )


def format_metric_delta(observed: MetricTriple, reference: PaperMetricTriple) -> str:
    return (
        f"MAE={abs(observed.mae - reference.mae):.6e}, "
        f"MaxAE={abs(observed.max_ae - reference.max_ae):.6e}, "
        f"MSE={abs(observed.mse - reference.mse):.6e}"
    )


def trend_matches_paper(
    reference: PaperStepReference,
    numerical: MetricTriple,
    origin: MetricTriple,
) -> bool:
    paper_flags = compare_origin_vs_numerical(reference.paper_model, reference.numerical)
    observed_flags = compare_origin_vs_numerical(origin, numerical)
    return paper_flags == observed_flags


def format_trend_line(
    reference: PaperStepReference,
    current_numerical: MetricTriple,
    current_origin: MetricTriple,
    aligned_numerical: MetricTriple,
    aligned_origin: MetricTriple,
) -> str:
    paper_flags = compare_origin_vs_numerical(reference.paper_model, reference.numerical)
    current_flags = compare_origin_vs_numerical(current_origin, current_numerical)
    aligned_flags = compare_origin_vs_numerical(aligned_origin, aligned_numerical)
    return (
        f"paper={render_trend_flags(paper_flags)} | "
        f"current_rebuilt={render_trend_flags(current_flags)} | "
        f"paper_aligned={render_trend_flags(aligned_flags)}"
    )


def compare_origin_vs_numerical(origin: MetricTriple | PaperMetricTriple, numerical: MetricTriple | PaperMetricTriple) -> tuple[bool, bool, bool]:
    return (
        origin.mae < numerical.mae,
        origin.max_ae < numerical.max_ae,
        origin.mse < numerical.mse,
    )


def render_trend_flags(flags: tuple[bool, bool, bool]) -> str:
    return f"MAE:{'<' if flags[0] else '>='}, MaxAE:{'<' if flags[1] else '>='}, MSE:{'<' if flags[2] else '>='}"


def resolve_output_path(project_root: Path, raw_output: str) -> Path:
    if raw_output:
        output_path = Path(raw_output)
        if not output_path.is_absolute():
            output_path = project_root / output_path
        return output_path
    stamp = datetime.now().strftime("%m%d")
    return project_root / "test" / "results" / f"paper_alignment_{stamp}.txt"


def load_origin_stats(csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
    mu_values: list[float] = []
    sigma_values: list[float] = []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        next(reader, None)
        for row in reader:
            if len(row) < 3:
                continue
            feature_name = row[0].strip().strip('"')
            if feature_name == "h":
                continue
            mu_values.append(float(row[1]))
            sigma_values.append(float(row[2]))
    return np.asarray(mu_values, dtype=np.float32), np.asarray(sigma_values, dtype=np.float32)


def dense_layer_sort_key(name: str) -> int:
    match = re.fullmatch(r"dense(?:_(\d+))?", name)
    if match is None:
        raise ValueError(f"Unexpected Keras dense layer group name: {name}")
    return int(match.group(1) or 0)


def load_origin_weights(checkpoint_path: Path) -> tuple[list[np.ndarray], list[np.ndarray]]:
    weights: list[np.ndarray] = []
    biases: list[np.ndarray] = []
    with h5py.File(checkpoint_path, "r") as handle:
        model_weights = handle["model_weights"]
        layer_names: list[str] = []
        for layer_name, layer_group in model_weights.items():
            if not isinstance(layer_group, h5py.Group):
                continue
            nested_groups = [child for child in layer_group.values() if isinstance(child, h5py.Group)]
            if not nested_groups:
                continue
            dense_group = nested_groups[0]
            if "kernel:0" in dense_group and "bias:0" in dense_group:
                layer_names.append(layer_name)

        for layer_name in sorted(layer_names, key=dense_layer_sort_key):
            dense_group = next(
                child for child in model_weights[layer_name].values() if isinstance(child, h5py.Group)
            )
            weights.append(np.asarray(dense_group["kernel:0"], dtype=np.float32))
            biases.append(np.asarray(dense_group["bias:0"], dtype=np.float32))

    if not weights or len(weights) != len(biases):
        raise ValueError(f"Failed to load dense-layer weights from {checkpoint_path}")
    return weights, biases


def forward_relu_mlp(
    inputs: np.ndarray,
    weights: list[np.ndarray],
    biases: list[np.ndarray],
) -> np.ndarray:
    activations = np.asarray(inputs, dtype=np.float32)
    for weight, bias in zip(weights[:-1], biases[:-1], strict=True):
        activations = np.maximum(activations @ weight + bias, 0.0)
    return (activations @ weights[-1] + biases[-1]).reshape(-1)


if __name__ == "__main__":
    try:
        main()
    finally:
        cleanup_bytecode_caches(PROJECT_ROOT)
