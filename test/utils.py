from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

try:
    import pandas as pd
except ImportError:  # pragma: no cover
    pd = None

try:
    import torch
except ImportError:  # pragma: no cover - allows origin-only tooling without PyTorch installed
    torch = None

from generate.train_data import encode_patch_legacy_flat, encode_patch_training_order
from generate.numerics import ReinitQualityEvaluator
from generate.config import load_generate_config
from train.config import load_training_config
if torch is not None:
    from train.model import load_inference_bundle
else:  # pragma: no cover - neural predictor is unavailable without torch
    load_inference_bundle = None


METHOD_LABELS = {
    "numerical": "Numerical",
    "neural": "Neural",
    "origin": "Origin",
}
CURVATURE_METHOD_STANDARD = "expanded_formula"
CURVATURE_METHOD_NEURAL = "model_prediction"
METRIC_VIEW_FIXED = "fixed_phi0_nodes"
METRIC_VIEW_CURRENT = "current_interface_nodes"
DATA_SOURCE_H5_RAW = "h5_raw"
DATA_SOURCE_REBUILT = "rebuilt"
MODEL_TAG_BASELINE = "baseline_numerical"
MODEL_TAG_ROOT = "model_root"
MODEL_TAG_ORIGIN = "origin"


RESULT_COLUMNS = [
    "data_split",
    "group_id",
    "rho_model",
    "step_or_iter",
    "method",
    "model_tag",
    "curvature_method",
    "metric_view",
    "data_source",
    "N_samples",
    "MAE_hk",
    "MaxAE_hk",
    "MSE_hk",
]
GROUP_COLUMNS = [
    "data_split",
    "method",
    "model_tag",
    "curvature_method",
    "metric_view",
    "data_source",
]
TABLE_COLUMNS = [
    "group_id",
    "rho_model",
    "step_or_iter",
    "N_samples",
    "MAE_hk",
    "MaxAE_hk",
    "MSE_hk",
]


_ENCODING_DIAGNOSTIC_PRINTED = False


def create_result_output_path(output_dir: str | Path) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / "unified_evaluation.txt"


def save_unified_results(rows: list[dict[str, object]], output_dir: str | Path) -> Path:
    output_path = create_result_output_path(output_dir)
    table = format_results_table(rows)
    output_path.write_text(table + "\n", encoding="utf-8")
    return output_path


def format_results_table(rows: list[dict[str, object]]) -> str:
    normalized = [{column: row.get(column, "") for column in RESULT_COLUMNS} for row in rows]
    groups: dict[tuple[str, str, str, str, str, str], list[dict[str, object]]] = {}
    for row in normalized:
        key = tuple(str(row[column]) for column in GROUP_COLUMNS)
        groups.setdefault(key, []).append(row)

    sections: list[str] = []
    for key in sorted(groups):
        data_split, method, model_tag, curvature_method, metric_view, data_source = key
        title = (
            f"[{data_split}] {method} | model={model_tag} | curvature={curvature_method} | "
            f"nodes={metric_view} | source={data_source}"
        )
        sections.append(title)
        sections.append(_format_single_table(groups[key]))
    return "\n\n".join(sections)


def _format_single_table(rows: list[dict[str, object]]) -> str:
    if pd is not None:
        frame = pd.DataFrame(rows, columns=TABLE_COLUMNS)
        formatters = {
            "MAE_hk": lambda value: f"{float(value):.6e}",
            "MaxAE_hk": lambda value: f"{float(value):.6e}",
            "MSE_hk": lambda value: f"{float(value):.6e}",
        }
        return frame.to_string(index=False, columns=TABLE_COLUMNS, formatters=formatters)

    prepared_rows = []
    for row in rows:
        prepared_rows.append(
            {
                "group_id": str(row["group_id"]),
                "rho_model": str(row["rho_model"]),
                "step_or_iter": str(row["step_or_iter"]),
                "N_samples": str(row["N_samples"]),
                "MAE_hk": f"{float(row['MAE_hk']):.6e}",
                "MaxAE_hk": f"{float(row['MaxAE_hk']):.6e}",
                "MSE_hk": f"{float(row['MSE_hk']):.6e}",
            }
        )

    widths = {
        column: max(len(column), *(len(prepared[column]) for prepared in prepared_rows))
        for column in TABLE_COLUMNS
    }
    header = " ".join(column.rjust(widths[column]) for column in TABLE_COLUMNS)
    lines = [header]
    for prepared in prepared_rows:
        lines.append(" ".join(prepared[column].rjust(widths[column]) for column in TABLE_COLUMNS))
    return "\n".join(lines)


@dataclass
class MetricAccumulator:
    count: int = 0
    sum_abs: float = 0.0
    sum_sq: float = 0.0
    max_abs: float = 0.0

    def update(self, target: np.ndarray, pred: np.ndarray) -> None:
        error = pred.astype(np.float64) - target.astype(np.float64)
        abs_error = np.abs(error)
        self.count += int(target.shape[0])
        self.sum_abs += float(abs_error.sum())
        self.sum_sq += float((error**2).sum())
        self.max_abs = max(self.max_abs, float(abs_error.max()) if abs_error.size else 0.0)

    def as_metrics(self) -> dict[str, float | int]:
        if self.count == 0:
            return {"N_samples": 0, "MAE_hk": 0.0, "MaxAE_hk": 0.0, "MSE_hk": 0.0}
        return {
            "N_samples": self.count,
            "MAE_hk": self.sum_abs / self.count,
            "MaxAE_hk": self.max_abs,
            "MSE_hk": self.sum_sq / self.count,
        }


def build_metric_row(
    *,
    data_split: str,
    group_id: str,
    rho_model: int,
    step: int,
    method: str,
    model_tag: str,
    curvature_method: str,
    metric_view: str,
    data_source: str,
    target: np.ndarray,
    pred: np.ndarray,
) -> dict[str, object]:
    accumulator = MetricAccumulator()
    accumulator.update(target, pred)
    metrics = accumulator.as_metrics()
    return {
        "data_split": data_split,
        "group_id": group_id,
        "rho_model": rho_model,
        "step_or_iter": step,
        "method": method,
        "model_tag": model_tag,
        "curvature_method": curvature_method,
        "metric_view": metric_view,
        "data_source": data_source,
        **metrics,
    }


def _infer_model_tag(model_dir: Path, default_root: Path) -> str:
    return MODEL_TAG_ROOT if model_dir.resolve() == default_root.resolve() else model_dir.name


def discover_model_directories(project_root: Path) -> list[Path]:
    model_root = project_root / "model"
    if not model_root.exists() or not model_root.is_dir():
        return []

    discovered: list[Path] = []
    if any(model_root.glob("model_rho*.pth")):
        discovered.append(model_root)

    for subdir in sorted(path for path in model_root.iterdir() if path.is_dir()):
        if subdir.name != "origin" and any(subdir.glob("model_rho*.pth")):
            discovered.append(subdir)
    return discovered


def discover_train_rhos(data_dir: Path, configured_rhos: tuple[int, ...]) -> list[int]:
    discovered: list[int] = []
    for h5_path in sorted(data_dir.glob("train_rho*.h5")):
        match = re.fullmatch(r"train_rho(\d+)\.h5", h5_path.name)
        if match:
            discovered.append(int(match.group(1)))
    if discovered:
        return discovered
    return list(configured_rhos)


class NeuralPredictor:
    def __init__(self, project_root: Path, *, model_dir: Path, config_path: str | Path | None = None):
        if torch is None:
            raise RuntimeError("PyTorch is required to use NeuralPredictor but is not installed in this environment.")
        self.project_root = project_root
        self.model_dir = model_dir
        self.model_tag = _infer_model_tag(model_dir, project_root / "model")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = load_training_config(config_path)
        self.cache: dict[int, tuple[torch.nn.Module, np.ndarray, np.ndarray]] = {}
        self._missing_bundle_rhos: set[int] = set()

    def has_bundle(self, rho: int) -> bool:
        checkpoint = self.model_dir / f"model_rho{rho}.pth"
        stats = self.model_dir / f"zscore_stats_{rho}.csv"
        return checkpoint.exists() and stats.exists()

    def note_missing_bundle_once(self, rho: int) -> None:
        if rho in self._missing_bundle_rhos:
            return
        self._missing_bundle_rhos.add(rho)
        print(
            f"Skipping model={self.model_tag} for rho={rho}: "
            f"missing model_rho{rho}.pth or zscore_stats_{rho}.csv in {self.model_dir}"
        )

    def predict(self, rho: int, stencils_raw: np.ndarray, batch_size: int = 16384) -> np.ndarray:
        model, mu, sigma = self._ensure_bundle(rho)
        outputs = []
        for start in range(0, stencils_raw.shape[0], batch_size):
            chunk = stencils_raw[start:start + batch_size].astype(np.float32, copy=False)
            norm_chunk = (chunk - mu) / sigma
            with torch.no_grad():
                inputs = torch.from_numpy(norm_chunk).to(self.device)
                preds = model(inputs).detach().cpu().numpy().reshape(-1)
            outputs.append(preds)
        return np.concatenate(outputs, axis=0) if outputs else np.empty((0,), dtype=np.float32)

    def _ensure_bundle(self, rho: int):
        if rho not in self.cache:
            model, stats = load_inference_bundle(
                rho,
                self.config.model,
                model_dir=self.model_dir,
                map_location=self.device,
            )
            model = model.to(self.device)
            model.eval()
            self.cache[rho] = (model, stats["mu"], stats["sigma"])
        return self.cache[rho]


class OriginPredictor:
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.model_dir = project_root / "model" / "origin"
        self.model_tag = MODEL_TAG_ORIGIN
        self.cache: dict[int, tuple[list[np.ndarray], list[np.ndarray], np.ndarray, np.ndarray]] = {}
        self._missing_bundle_rhos: set[int] = set()

    def has_bundle(self, rho: int) -> bool:
        checkpoint = self.model_dir / f"nnet_{rho}.h5"
        stats = self.model_dir / f"trainStats_{rho}.csv"
        return checkpoint.exists() and stats.exists()

    def note_missing_bundle_once(self, rho: int) -> None:
        if rho in self._missing_bundle_rhos:
            return
        self._missing_bundle_rhos.add(rho)
        print(
            f"Skipping model={self.model_tag} for rho={rho}: "
            f"missing nnet_{rho}.h5 or trainStats_{rho}.csv in {self.model_dir}"
        )

    def predict(self, rho: int, stencils_raw: np.ndarray, batch_size: int = 16384) -> np.ndarray:
        weights, biases, mu, sigma = self._ensure_bundle(rho)
        outputs = []
        for start in range(0, stencils_raw.shape[0], batch_size):
            chunk = stencils_raw[start:start + batch_size].astype(np.float32, copy=False)
            norm_chunk = (chunk - mu) / sigma
            preds = _forward_relu_mlp(norm_chunk, weights, biases)
            outputs.append(preds.astype(np.float32, copy=False))
        return np.concatenate(outputs, axis=0) if outputs else np.empty((0,), dtype=np.float32)

    def _ensure_bundle(self, rho: int) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray, np.ndarray]:
        if rho not in self.cache:
            checkpoint_path = self.model_dir / f"nnet_{rho}.h5"
            stats_path = self.model_dir / f"trainStats_{rho}.csv"
            weights, biases = _load_origin_weights(checkpoint_path)
            mu, sigma = _load_origin_stats(stats_path)
            self.cache[rho] = (weights, biases, mu, sigma)
        return self.cache[rho]


def _load_origin_stats(csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
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

    return (
        np.asarray(mu_values, dtype=np.float32),
        np.asarray(sigma_values, dtype=np.float32),
    )


def _dense_layer_sort_key(name: str) -> int:
    match = re.fullmatch(r"dense(?:_(\d+))?", name)
    if match is None:
        raise ValueError(f"Unexpected Keras dense layer group name: {name}")
    return int(match.group(1) or 0)


def _load_origin_weights(checkpoint_path: Path) -> tuple[list[np.ndarray], list[np.ndarray]]:
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

        for layer_name in sorted(layer_names, key=_dense_layer_sort_key):
            dense_group = next(
                child for child in model_weights[layer_name].values() if isinstance(child, h5py.Group)
            )
            weights.append(np.asarray(dense_group["kernel:0"], dtype=np.float32))
            biases.append(np.asarray(dense_group["bias:0"], dtype=np.float32))

    if not weights or len(weights) != len(biases):
        raise ValueError(f"Failed to load dense-layer weights from {checkpoint_path}")
    return weights, biases


def _forward_relu_mlp(
    inputs: np.ndarray,
    weights: list[np.ndarray],
    biases: list[np.ndarray],
) -> np.ndarray:
    activations = np.asarray(inputs, dtype=np.float32)
    for weight, bias in zip(weights[:-1], biases[:-1], strict=True):
        activations = np.maximum(activations @ weight + bias, 0.0)
    return (activations @ weights[-1] + biases[-1]).reshape(-1)


def hkappa_from_full_field_standard(phi: np.ndarray, indices: np.ndarray, h: float) -> np.ndarray:
    if indices.size == 0:
        return np.zeros((0,), dtype=np.float64)

    rows = indices[:, 0]
    cols = indices[:, 1]
    phi_x = (phi[rows, cols + 1] - phi[rows, cols - 1]) / (2.0 * h)
    phi_y = (phi[rows + 1, cols] - phi[rows - 1, cols]) / (2.0 * h)
    phi_xx = (phi[rows, cols + 1] - 2.0 * phi[rows, cols] + phi[rows, cols - 1]) / (h**2)
    phi_yy = (phi[rows + 1, cols] - 2.0 * phi[rows, cols] + phi[rows - 1, cols]) / (h**2)
    phi_xy = (
        phi[rows + 1, cols + 1]
        - phi[rows + 1, cols - 1]
        - phi[rows - 1, cols + 1]
        + phi[rows - 1, cols - 1]
    ) / (4.0 * h**2)

    numerator = phi_x**2 * phi_yy - 2.0 * phi_x * phi_y * phi_xy + phi_y**2 * phi_xx
    denominator = (phi_x**2 + phi_y**2) ** 1.5
    prediction = np.zeros_like(numerator)
    mask = denominator > 1e-12
    prediction[mask] = numerator[mask] / denominator[mask]
    return h * prediction


def compute_curvature_predictions(phi: np.ndarray, indices: np.ndarray, h: float) -> dict[str, np.ndarray]:
    return {
        CURVATURE_METHOD_STANDARD: hkappa_from_full_field_standard(phi, indices, h),
    }


def train_style_sampling_indices(phi: np.ndarray) -> np.ndarray:
    i_coords, j_coords = ReinitQualityEvaluator.get_sampling_coordinates(phi)
    if len(i_coords) == 0:
        return np.zeros((0, 2), dtype=np.int64)
    return np.column_stack((i_coords, j_coords)).astype(np.int64, copy=False)


def select_payload(
    fixed_payload: dict[str, np.ndarray | float],
    current_payload: dict[str, np.ndarray | float],
    *,
    metric_view_policy: str,
) -> tuple[str, dict[str, np.ndarray | float]]:
    if metric_view_policy != "larger_node_set":
        raise ValueError(f"Unsupported metric_view_policy: {metric_view_policy}")
    fixed_count = int(np.asarray(fixed_payload["indices"]).shape[0])
    current_count = int(np.asarray(current_payload["indices"]).shape[0])
    if current_count >= fixed_count:
        return METRIC_VIEW_CURRENT, current_payload
    return METRIC_VIEW_FIXED, fixed_payload


def print_encoding_diagnostic_once(
    *,
    enabled: bool,
    group_id: str,
    step: int,
    phi: np.ndarray,
    indices: np.ndarray,
    encoded_stencils: np.ndarray,
) -> None:
    global _ENCODING_DIAGNOSTIC_PRINTED
    if not enabled or _ENCODING_DIAGNOSTIC_PRINTED or indices.size == 0 or encoded_stencils.size == 0:
        return

    row, col = indices[0]
    patch = phi[row - 1:row + 2, col - 1:col + 2]
    legacy = encode_patch_legacy_flat(patch)
    training_order = encode_patch_training_order(patch)
    stored = encoded_stencils[0]
    differs = not np.allclose(training_order, legacy)
    stored_matches = np.allclose(stored, training_order)
    max_diff = float(np.max(np.abs(training_order - legacy)))
    print(
        "\nStencil encoding diagnostic:"
        f" group={group_id} | step={step} | "
        f"train-order differs from legacy={differs} | "
        f"max_diff={max_diff:.6e} | "
        f"selected payload matches train-order={stored_matches}"
    )
    _ENCODING_DIAGNOSTIC_PRINTED = True


def print_neural_step_variation(rows: list[dict[str, object]]) -> None:
    smooth_rows = [
        row for row in rows
        if row.get("data_split") == "test"
        and row.get("method") == METHOD_LABELS["neural"]
        and row.get("group_id") == "smooth/smooth_256"
    ]
    if not smooth_rows:
        return

    print("\nNeural smooth_256 step-variation check:")
    model_tags = sorted({str(row["model_tag"]) for row in smooth_rows})
    for model_tag in model_tags:
        model_rows = sorted(
            (row for row in smooth_rows if str(row["model_tag"]) == model_tag),
            key=lambda row: int(row["step_or_iter"]),
        )
        mae_by_step = [float(row["MAE_hk"]) for row in model_rows]
        exact_repeat = len({f"{value:.16e}" for value in mae_by_step}) <= 1
        formatted = ", ".join(
            f"{int(row['step_or_iter'])}:{float(row['MAE_hk']):.6e}"
            for row in model_rows
        )
        print(f"  model={model_tag} | mae_by_step=[{formatted}] | exact_repeat={exact_repeat}")


def build_test_diagnostics_row(
    *,
    group_id: str,
    step: int,
    h: float,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    phi0_points: np.ndarray,
    fixed_indices: np.ndarray,
    current_payload: dict[str, np.ndarray | float],
) -> dict[str, object]:
    current_phi = np.asarray(current_payload["phi"], dtype=np.float64)
    current_points = collect_zero_crossing_points(current_phi, x_grid, y_grid)
    fixed_xy = np.column_stack((x_grid[fixed_indices[:, 0], fixed_indices[:, 1]], y_grid[fixed_indices[:, 0], fixed_indices[:, 1]])) if fixed_indices.size else np.zeros((0, 2), dtype=np.float64)
    fixed_to_current_h = nearest_distances(fixed_xy, current_points) / h if fixed_xy.size and current_points.size else np.zeros((0,), dtype=np.float64)
    interface_shift_h = symmetric_nearest_distances(phi0_points, current_points) / h if phi0_points.size and current_points.size else np.zeros((0,), dtype=np.float64)
    eikonal_abs = np.asarray(current_payload["eikonal_abs_on_samples"], dtype=np.float64)

    return {
        "group_id": group_id,
        "step": int(step),
        "fixed_to_current_interface_mean_h": safe_mean(fixed_to_current_h),
        "fixed_to_current_interface_p95_h": safe_quantile(fixed_to_current_h, 0.95),
        "fixed_to_current_interface_max_h": safe_max(fixed_to_current_h),
        "interface_shift_mean_h": safe_mean(interface_shift_h),
        "interface_shift_p95_h": safe_quantile(interface_shift_h, 0.95),
        "interface_shift_max_h": safe_max(interface_shift_h),
        "current_interface_eikonal_mean": safe_mean(eikonal_abs),
        "current_interface_eikonal_max": safe_max(eikonal_abs),
    }


def print_test_diagnostics(diagnostics: list[dict[str, object]]) -> None:
    print("\nExplicit test diagnostics:")
    for row in diagnostics:
        print(
            "  "
            f"{row['group_id']} | step={int(row['step']):>2} | "
            "fixed->current(h) "
            f"mean/p95/max={float(row['fixed_to_current_interface_mean_h']):.6f}/"
            f"{float(row['fixed_to_current_interface_p95_h']):.6f}/"
            f"{float(row['fixed_to_current_interface_max_h']):.6f} | "
            "shift(h) "
            f"mean/p95/max={float(row['interface_shift_mean_h']):.6f}/"
            f"{float(row['interface_shift_p95_h']):.6f}/"
            f"{float(row['interface_shift_max_h']):.6f} | "
            "eikonal "
            f"mean/max={float(row['current_interface_eikonal_mean']):.6e}/"
            f"{float(row['current_interface_eikonal_max']):.6e}"
        )


def central_grad_norm(phi: np.ndarray, h: float) -> np.ndarray:
    phi_x = np.zeros_like(phi, dtype=np.float64)
    phi_y = np.zeros_like(phi, dtype=np.float64)
    phi_x[:, 1:-1] = (phi[:, 2:] - phi[:, :-2]) / (2.0 * h)
    phi_y[1:-1, :] = (phi[2:, :] - phi[:-2, :]) / (2.0 * h)
    return np.sqrt(phi_x**2 + phi_y**2)


def collect_zero_crossing_points(phi: np.ndarray, x_grid: np.ndarray, y_grid: np.ndarray) -> np.ndarray:
    x_coords = x_grid[0, :]
    y_coords = y_grid[:, 0]
    points: list[tuple[float, float]] = []

    horizontal_mask = phi[:, :-1] * phi[:, 1:] <= 0.0
    horizontal_rows, horizontal_cols = np.where(horizontal_mask)
    for row, col in zip(horizontal_rows.tolist(), horizontal_cols.tolist()):
        phi0 = float(phi[row, col])
        phi1 = float(phi[row, col + 1])
        denom = phi0 - phi1
        t = 0.5 if abs(denom) <= 1e-12 else phi0 / denom
        t = float(np.clip(t, 0.0, 1.0))
        x = float(x_coords[col] + t * (x_coords[col + 1] - x_coords[col]))
        y = float(y_coords[row])
        points.append((x, y))

    vertical_mask = phi[:-1, :] * phi[1:, :] <= 0.0
    vertical_rows, vertical_cols = np.where(vertical_mask)
    for row, col in zip(vertical_rows.tolist(), vertical_cols.tolist()):
        phi0 = float(phi[row, col])
        phi1 = float(phi[row + 1, col])
        denom = phi0 - phi1
        t = 0.5 if abs(denom) <= 1e-12 else phi0 / denom
        t = float(np.clip(t, 0.0, 1.0))
        x = float(x_coords[col])
        y = float(y_coords[row] + t * (y_coords[row + 1] - y_coords[row]))
        points.append((x, y))

    if not points:
        return np.zeros((0, 2), dtype=np.float64)
    return np.asarray(points, dtype=np.float64)


def symmetric_nearest_distances(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.concatenate([nearest_distances(a, b), nearest_distances(b, a)])


def nearest_distances(source: np.ndarray, target: np.ndarray, chunk_size: int = 512) -> np.ndarray:
    if source.size == 0 or target.size == 0:
        return np.zeros((0,), dtype=np.float64)
    result = np.empty(source.shape[0], dtype=np.float64)
    for start in range(0, source.shape[0], chunk_size):
        chunk = source[start:start + chunk_size]
        diff = chunk[:, None, :] - target[None, :, :]
        dist_sq = np.sum(diff * diff, axis=2)
        result[start:start + chunk.shape[0]] = np.sqrt(np.min(dist_sq, axis=1))
    return result


def safe_mean(values: np.ndarray) -> float:
    return float(np.mean(values)) if values.size else 0.0


def safe_quantile(values: np.ndarray, quantile: float) -> float:
    return float(np.quantile(values, quantile)) if values.size else 0.0


def safe_max(values: np.ndarray) -> float:
    return float(np.max(values)) if values.size else 0.0
