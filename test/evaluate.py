from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import torch

from generate.config import CFL, GEOMETRY_SEED, REINIT_STEPS, TARGET_RHOS, VARIATIONS
from generate.field_builder import LevelSetFieldBuilder
from generate.geometry import CircleGeometryGenerator
from generate.reinitializer import ReinitFieldPackBuilder, ReinitQualityEvaluator
from generate.test_data import (
    TEST_CFL,
    TEST_ITERS,
    LevelSetReinitializer,
    build_flower_phi0,
    build_grid,
    extract_3x3_stencils,
    find_projection_theta,
    get_valid_indices,
    hkappa_analytic,
    hkappa_div_normal_from_field,
    indices_to_xy,
    interface_band_mask,
)
from generate.test_blueprints import TEST_BLUEPRINTS
from train.config import load_training_config
from train.inference import load_inference_bundle


METHOD_LABELS = {
    "numerical": "Numerical",
    "neural": "Neural",
}
CURVATURE_METHOD_DIV = "div_normal"
CURVATURE_METHOD_STANDARD = "expanded_formula"
CURVATURE_METHOD_NEURAL = "model_prediction"
METRIC_VIEW_FIXED = "fixed_phi0_nodes"
METRIC_VIEW_CURRENT = "current_interface_nodes"
DATA_SOURCE_H5_RAW = "h5_raw"
DATA_SOURCE_REBUILT = "rebuilt"


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


class NeuralPredictor:
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = load_training_config(project_root / "train" / "config.txt")
        self.cache: dict[int, tuple[torch.nn.Module, np.ndarray, np.ndarray]] = {}

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
                model_dir=self.project_root / "model",
                map_location=self.device,
            )
            model = model.to(self.device)
            model.eval()
            self.cache[rho] = (model, stats["mu"], stats["sigma"])
        return self.cache[rho]


def evaluate_train_data(
    project_root: Path,
    methods: list[str],
    *,
    chunk_size: int = 100000,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    if "numerical" in methods:
        rows.extend(_evaluate_train_data_numerical_full_field())
    if "neural" in methods:
        rows.extend(_evaluate_train_data_neural(project_root, chunk_size=chunk_size))

    method_order = {METHOD_LABELS[method]: index for index, method in enumerate(methods)}
    curvature_order = {
        CURVATURE_METHOD_DIV: 0,
        CURVATURE_METHOD_STANDARD: 1,
        CURVATURE_METHOD_NEURAL: 2,
    }
    metric_view_order = {
        METRIC_VIEW_FIXED: 0,
        METRIC_VIEW_CURRENT: 1,
    }
    data_source_order = {
        DATA_SOURCE_H5_RAW: 0,
        DATA_SOURCE_REBUILT: 1,
    }
    rows.sort(
        key=lambda row: (
            int(row["rho_model"]),
            int(row["step_or_iter"]),
            method_order[row["method"]],
            curvature_order.get(str(row["curvature_method"]), 99),
            metric_view_order.get(str(row["metric_view"]), 99),
            data_source_order.get(str(row["data_source"]), 99),
        )
    )
    return rows


def _evaluate_train_data_numerical_full_field() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    field_builder = LevelSetFieldBuilder(dtype=np.float64)
    reinit_builder = ReinitFieldPackBuilder(cfl=CFL)
    print("Evaluating train Numerical with full-field div(normal); this rebuilds train fields in memory and can take a while.")

    for rho in TARGET_RHOS:
        grouped: dict[tuple[int, str], MetricAccumulator] = {}
        generator = CircleGeometryGenerator(resolution_rho=rho, seed=GEOMETRY_SEED, variations=VARIATIONS)
        blueprints = generator.generate_blueprints()
        print(f"  [train Numerical] rho={rho} | rebuilding {len(blueprints)} blueprints")

        for index, blueprint in enumerate(blueprints, start=1):
            pack_sdf = field_builder.build_circle_sdf(blueprint, return_grid=False)
            _update_train_numerical_accumulator(grouped, step=0, field_pack=pack_sdf)

            pack_nonsdf = field_builder.build_circle_nonsdf(blueprint, return_grid=False)
            reinitialized = reinit_builder.build(pack_nonsdf, steps_list=REINIT_STEPS)
            for step in REINIT_STEPS:
                _update_train_numerical_accumulator(grouped, step=step, field_pack=reinitialized[str(step)])

            if index % 100 == 0 or index == len(blueprints):
                print(f"    progress: {index}/{len(blueprints)} blueprints")

        for step, curvature_method in sorted(grouped):
            metrics = grouped[(step, curvature_method)].as_metrics()
            rows.append(
                {
                    "data_split": "train",
                    "group_id": f"rho{rho}",
                    "rho_model": rho,
                    "step_or_iter": step,
                    "method": METHOD_LABELS["numerical"],
                    "curvature_method": curvature_method,
                    "metric_view": METRIC_VIEW_CURRENT,
                    "data_source": DATA_SOURCE_REBUILT,
                    **metrics,
                }
            )
    return rows


def _update_train_numerical_accumulator(
    grouped: dict[tuple[int, str], MetricAccumulator],
    *,
    step: int,
    field_pack: dict[str, object],
) -> None:
    phi = field_pack["field"]["phi"]
    h = float(field_pack["params"]["h"])
    target_hk = float(field_pack["label"]["h_kappa"])
    if target_hk <= 0.0:
        raise ValueError(
            "Training circle h*kappa must stay positive under the current "
            "outside-positive / inside-negative sign convention. If the raw "
            "phi convention changes, flip label['kappa'] and label['h_kappa'] too."
        )
    i_coords, j_coords = ReinitQualityEvaluator.get_sampling_coordinates(phi)
    if len(i_coords) == 0:
        return

    indices = np.column_stack((i_coords, j_coords)).astype(np.int64, copy=False)
    # Circles have constant curvature on the entire interface, so a single h/r
    # target remains correct for every current-interface sample node.
    target = np.full(indices.shape[0], target_hk, dtype=np.float64)
    predictions = _compute_curvature_predictions(phi, indices, h)

    for curvature_method, pred in predictions.items():
        accumulator = grouped.setdefault((int(step), curvature_method), MetricAccumulator())

        # Training HDF5 augments each sample with (-phi, -hk); mirror that here so
        # train numerical metrics match the effective train dataset.
        accumulator.update(target, pred)
        accumulator.update(-target, -pred)


def _evaluate_train_data_neural(
    project_root: Path,
    *,
    chunk_size: int,
) -> list[dict[str, object]]:
    predictor = NeuralPredictor(project_root)
    rows: list[dict[str, object]] = []
    for rho in TARGET_RHOS:
        h5_path = project_root / "data" / f"train_rho{rho}.h5"
        if not h5_path.exists():
            continue

        grouped: dict[int, MetricAccumulator] = {}
        with h5py.File(h5_path, "r") as handle:
            total_samples = handle["X"].shape[0]
            for start in range(0, total_samples, chunk_size):
                end = min(start + chunk_size, total_samples)
                X = handle["X"][start:end]
                y_true = handle["Y"][start:end, 0]
                steps = handle["reinit_steps"][start:end, 0].astype(np.int32)
                pred = predictor.predict(rho, X)

                for step in np.unique(steps):
                    mask = steps == step
                    accumulator = grouped.setdefault(int(step), MetricAccumulator())
                    accumulator.update(y_true[mask], pred[mask])

        for step in sorted(grouped):
            metrics = grouped[step].as_metrics()
            rows.append(
                {
                    "data_split": "train",
                    "group_id": f"rho{rho}",
                    "rho_model": rho,
                    "step_or_iter": step,
                    "method": METHOD_LABELS["neural"],
                    "curvature_method": CURVATURE_METHOD_NEURAL,
                    "metric_view": METRIC_VIEW_CURRENT,
                    "data_source": DATA_SOURCE_H5_RAW,
                    **metrics,
                }
            )
    return rows


def evaluate_test_data(project_root: Path, methods: list[str]) -> list[dict[str, object]]:
    predictor = NeuralPredictor(project_root) if "neural" in methods else None
    rows: list[dict[str, object]] = []
    diagnostics: list[dict[str, object]] = []
    data_dir = project_root / "data"
    ordered_exp_ids = [config["exp_id"] for config in TEST_BLUEPRINTS]
    seen = set()
    candidates: list[Path] = []
    for exp_id in ordered_exp_ids:
        exp_dir = data_dir / exp_id
        if exp_dir.is_dir() and (exp_dir / "meta.json").exists():
            candidates.append(exp_dir)
            seen.add(exp_dir.name)
    for exp_dir in sorted(path for path in data_dir.iterdir() if path.is_dir() and (path / "meta.json").exists()):
        if exp_dir.name not in seen:
            candidates.append(exp_dir)

    for exp_dir in candidates:
        meta = json.loads((exp_dir / "meta.json").read_text(encoding="utf-8"))
        rho_model = int(meta["rho_model"])
        blueprint_id = str(meta.get("blueprint_id", meta.get("exp_id", exp_dir.name)))
        experiment_type = str(meta.get("experiment_type", meta.get("gamma_type", "test")))
        group_id = f"{experiment_type}/{blueprint_id}"
        fixed_payloads = _build_fixed_test_payloads(meta)
        current_payloads = _build_current_test_payloads(meta)
        x_grid, y_grid, h = build_grid(float(meta["L"]), int(meta["N"]))
        phi0 = build_flower_phi0(x_grid, y_grid, float(meta["a"]), float(meta["b"]), float(meta["p"]))
        phi0_points = _collect_zero_crossing_points(phi0, x_grid, y_grid)

        for step in TEST_ITERS:
            h5_path = exp_dir / f"iter_{step}.h5"
            if not h5_path.exists():
                continue
            fixed_payload = fixed_payloads.get(int(step))
            current_payload = current_payloads.get(int(step))
            if fixed_payload is None or current_payload is None:
                continue
            raw_payloads = _load_raw_test_payloads(h5_path, h)

            if "numerical" in methods:
                raw_fixed_payload = raw_payloads.get(METRIC_VIEW_FIXED)
                raw_current_payload = raw_payloads.get(METRIC_VIEW_CURRENT)
                if raw_fixed_payload is not None:
                    rows.extend(
                        _build_numerical_test_rows(
                            group_id=group_id,
                            rho_model=rho_model,
                            step=step,
                            metric_view=METRIC_VIEW_FIXED,
                            data_source=DATA_SOURCE_H5_RAW,
                            payload=raw_fixed_payload,
                        )
                    )
                if raw_current_payload is not None:
                    rows.extend(
                        _build_numerical_test_rows(
                            group_id=group_id,
                            rho_model=rho_model,
                            step=step,
                            metric_view=METRIC_VIEW_CURRENT,
                            data_source=DATA_SOURCE_H5_RAW,
                            payload=raw_current_payload,
                        )
                    )
                rows.extend(
                    _build_numerical_test_rows(
                        group_id=group_id,
                        rho_model=rho_model,
                        step=step,
                        metric_view=METRIC_VIEW_FIXED,
                        data_source=DATA_SOURCE_REBUILT,
                        payload=fixed_payload,
                    )
                )
                rows.extend(
                    _build_numerical_test_rows(
                        group_id=group_id,
                        rho_model=rho_model,
                        step=step,
                        metric_view=METRIC_VIEW_CURRENT,
                        data_source=DATA_SOURCE_REBUILT,
                        payload=current_payload,
                    )
                )

            if "neural" in methods and predictor is not None:
                fixed_pred = predictor.predict(rho_model, fixed_payload["stencils_raw"])
                rows.append(
                    _build_metric_row(
                        data_split="test",
                        group_id=group_id,
                        rho_model=rho_model,
                        step=step,
                        method=METHOD_LABELS["neural"],
                        curvature_method=CURVATURE_METHOD_NEURAL,
                        metric_view=METRIC_VIEW_FIXED,
                        data_source=DATA_SOURCE_REBUILT,
                        target=fixed_payload["target"],
                        pred=fixed_pred,
                    )
                )
                current_pred = predictor.predict(rho_model, current_payload["stencils_raw"])
                rows.append(
                    _build_metric_row(
                        data_split="test",
                        group_id=group_id,
                        rho_model=rho_model,
                        step=step,
                        method=METHOD_LABELS["neural"],
                        curvature_method=CURVATURE_METHOD_NEURAL,
                        metric_view=METRIC_VIEW_CURRENT,
                        data_source=DATA_SOURCE_REBUILT,
                        target=current_payload["target"],
                        pred=current_pred,
                    )
                )

            diagnostics.append(
                _build_test_diagnostics_row(
                    group_id=group_id,
                    step=step,
                    h=h,
                    x_grid=x_grid,
                    y_grid=y_grid,
                    phi0_points=phi0_points,
                    fixed_indices=np.asarray(fixed_payload["indices"], dtype=np.int64),
                    current_payload=current_payload,
                )
            )
    if diagnostics:
        _print_test_diagnostics(diagnostics)
    return rows


def _build_fixed_test_payloads(meta: dict[str, object]) -> dict[int, dict[str, np.ndarray | float]]:
    x_grid, y_grid, h = build_grid(float(meta["L"]), int(meta["N"]))
    phi0 = build_flower_phi0(x_grid, y_grid, float(meta["a"]), float(meta["b"]), float(meta["p"]))
    fixed_indices = get_valid_indices(interface_band_mask(phi0))
    fixed_xy = indices_to_xy(fixed_indices, x_grid, y_grid)
    fixed_theta_proj = find_projection_theta(fixed_xy, float(meta["a"]), float(meta["b"]), float(meta["p"]))
    fixed_target = hkappa_analytic(fixed_theta_proj, h, float(meta["a"]), float(meta["b"]), float(meta["p"]))
    reinitializer = LevelSetReinitializer(cfl=TEST_CFL)
    payloads: dict[int, dict[str, np.ndarray | float]] = {}

    for step in TEST_ITERS:
        phi = reinitializer.reinitialize(phi0, h, int(step))
        payloads[int(step)] = {
            "phi": phi,
            "indices": fixed_indices,
            "stencils_raw": extract_3x3_stencils(phi, fixed_indices),
            "target": fixed_target,
            "curvature_predictions": _compute_curvature_predictions(phi, fixed_indices, h),
        }

    return payloads


def _build_current_test_payloads(meta: dict[str, object]) -> dict[int, dict[str, np.ndarray | float]]:
    x_grid, y_grid, h = build_grid(float(meta["L"]), int(meta["N"]))
    phi0 = build_flower_phi0(x_grid, y_grid, float(meta["a"]), float(meta["b"]), float(meta["p"]))
    reinitializer = LevelSetReinitializer(cfl=TEST_CFL)
    payloads: dict[int, dict[str, np.ndarray | float]] = {}

    for step in TEST_ITERS:
        phi = reinitializer.reinitialize(phi0, h, int(step))
        indices = get_valid_indices(interface_band_mask(phi))
        if indices.size == 0:
            payloads[int(step)] = {
                "phi": phi,
                "indices": indices,
                "stencils_raw": np.zeros((0, 9), dtype=np.float64),
                "target": np.zeros((0,), dtype=np.float64),
                "curvature_predictions": {
                    CURVATURE_METHOD_DIV: np.zeros((0,), dtype=np.float64),
                    CURVATURE_METHOD_STANDARD: np.zeros((0,), dtype=np.float64),
                },
                "eikonal_abs_on_samples": np.zeros((0,), dtype=np.float64),
            }
            continue

        xy = indices_to_xy(indices, x_grid, y_grid)
        theta_proj = find_projection_theta(xy, float(meta["a"]), float(meta["b"]), float(meta["p"]))
        target = hkappa_analytic(theta_proj, h, float(meta["a"]), float(meta["b"]), float(meta["p"]))
        stencils_raw = extract_3x3_stencils(phi, indices)
        grad_norm = _central_grad_norm(phi, h)

        payloads[int(step)] = {
            "phi": phi,
            "indices": indices,
            "stencils_raw": stencils_raw,
            "target": target,
            "curvature_predictions": _compute_curvature_predictions(phi, indices, h),
            "eikonal_abs_on_samples": np.abs(grad_norm[indices[:, 0], indices[:, 1]] - 1.0),
        }

    return payloads


def _build_test_diagnostics_row(
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
    current_points = _collect_zero_crossing_points(current_phi, x_grid, y_grid)
    fixed_xy = indices_to_xy(fixed_indices, x_grid, y_grid)
    fixed_to_current_h = _nearest_distances(fixed_xy, current_points) / h if fixed_xy.size and current_points.size else np.zeros((0,), dtype=np.float64)
    interface_shift_h = _symmetric_nearest_distances(phi0_points, current_points) / h if phi0_points.size and current_points.size else np.zeros((0,), dtype=np.float64)
    eikonal_abs = np.asarray(current_payload["eikonal_abs_on_samples"], dtype=np.float64)

    return {
        "group_id": group_id,
        "step": int(step),
        "fixed_to_current_interface_mean_h": _safe_mean(fixed_to_current_h),
        "fixed_to_current_interface_p95_h": _safe_quantile(fixed_to_current_h, 0.95),
        "fixed_to_current_interface_max_h": _safe_max(fixed_to_current_h),
        "interface_shift_mean_h": _safe_mean(interface_shift_h),
        "interface_shift_p95_h": _safe_quantile(interface_shift_h, 0.95),
        "interface_shift_max_h": _safe_max(interface_shift_h),
        "current_interface_eikonal_mean": _safe_mean(eikonal_abs),
        "current_interface_eikonal_max": _safe_max(eikonal_abs),
    }


def _print_test_diagnostics(diagnostics: list[dict[str, object]]) -> None:
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


def _central_grad_norm(phi: np.ndarray, h: float) -> np.ndarray:
    phi_x = np.zeros_like(phi, dtype=np.float64)
    phi_y = np.zeros_like(phi, dtype=np.float64)
    phi_x[:, 1:-1] = (phi[:, 2:] - phi[:, :-2]) / (2.0 * h)
    phi_y[1:-1, :] = (phi[2:, :] - phi[:-2, :]) / (2.0 * h)
    return np.sqrt(phi_x**2 + phi_y**2)


def _collect_zero_crossing_points(phi: np.ndarray, x_grid: np.ndarray, y_grid: np.ndarray) -> np.ndarray:
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


def _symmetric_nearest_distances(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.concatenate([_nearest_distances(a, b), _nearest_distances(b, a)])


def _nearest_distances(source: np.ndarray, target: np.ndarray, chunk_size: int = 512) -> np.ndarray:
    if source.size == 0 or target.size == 0:
        return np.zeros((0,), dtype=np.float64)
    result = np.empty(source.shape[0], dtype=np.float64)
    for start in range(0, source.shape[0], chunk_size):
        chunk = source[start:start + chunk_size]
        diff = chunk[:, None, :] - target[None, :, :]
        dist_sq = np.sum(diff * diff, axis=2)
        result[start:start + chunk.shape[0]] = np.sqrt(np.min(dist_sq, axis=1))
    return result


def _safe_mean(values: np.ndarray) -> float:
    return float(np.mean(values)) if values.size else 0.0


def _safe_quantile(values: np.ndarray, quantile: float) -> float:
    return float(np.quantile(values, quantile)) if values.size else 0.0


def _safe_max(values: np.ndarray) -> float:
    return float(np.max(values)) if values.size else 0.0


def _compute_curvature_predictions(phi: np.ndarray, indices: np.ndarray, h: float) -> dict[str, np.ndarray]:
    return {
        CURVATURE_METHOD_DIV: hkappa_div_normal_from_field(phi, indices, h),
        CURVATURE_METHOD_STANDARD: _hkappa_from_full_field_standard(phi, indices, h),
    }


def _hkappa_from_full_field_standard(phi: np.ndarray, indices: np.ndarray, h: float) -> np.ndarray:
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


def _build_numerical_test_rows(
    *,
    group_id: str,
    rho_model: int,
    step: int,
    metric_view: str,
    data_source: str,
    payload: dict[str, np.ndarray | float],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    target = np.asarray(payload["target"], dtype=np.float64)
    predictions = payload["curvature_predictions"]
    for curvature_method, pred in predictions.items():
        rows.append(
            _build_metric_row(
                data_split="test",
                group_id=group_id,
                rho_model=rho_model,
                step=step,
                method=METHOD_LABELS["numerical"],
                curvature_method=curvature_method,
                metric_view=metric_view,
                data_source=data_source,
                target=target,
                pred=np.asarray(pred, dtype=np.float64),
            )
        )
    return rows


def _build_metric_row(
    *,
    data_split: str,
    group_id: str,
    rho_model: int,
    step: int,
    method: str,
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
        "curvature_method": curvature_method,
        "metric_view": metric_view,
        "data_source": data_source,
        **metrics,
    }


def _load_raw_test_payloads(h5_path: Path, h: float) -> dict[str, dict[str, np.ndarray | float]]:
    payloads: dict[str, dict[str, np.ndarray | float]] = {}
    with h5py.File(h5_path, "r") as handle:
        payloads[METRIC_VIEW_CURRENT] = _build_raw_payload_from_handle(handle, h, prefix="")
        if "fixed_indices" in handle:
            payloads[METRIC_VIEW_FIXED] = _build_raw_payload_from_handle(handle, h, prefix="fixed_")
    return payloads


def _build_raw_payload_from_handle(
    handle: h5py.File,
    h: float,
    *,
    prefix: str,
) -> dict[str, np.ndarray | float]:
    indices = np.asarray(handle[f"{prefix}indices"][:], dtype=np.int64)
    stencils_raw = np.asarray(handle[f"{prefix}stencils_raw"][:], dtype=np.float64)
    target = np.asarray(handle[f"{prefix}hkappa_target"][:], dtype=np.float64)
    div_prediction = np.asarray(handle[f"{prefix}hkappa_fd"][:], dtype=np.float64)
    return {
        "indices": indices,
        "stencils_raw": stencils_raw,
        "target": target,
        "curvature_predictions": {
            CURVATURE_METHOD_DIV: div_prediction,
            CURVATURE_METHOD_STANDARD: _hkappa_from_stencils_standard(stencils_raw, h),
        },
    }


def _hkappa_from_stencils_standard(stencils_raw: np.ndarray, h: float) -> np.ndarray:
    if stencils_raw.size == 0:
        return np.zeros((0,), dtype=np.float64)

    phi_x = (stencils_raw[:, 5] - stencils_raw[:, 3]) / (2.0 * h)
    phi_y = (stencils_raw[:, 7] - stencils_raw[:, 1]) / (2.0 * h)
    phi_xx = (stencils_raw[:, 5] - 2.0 * stencils_raw[:, 4] + stencils_raw[:, 3]) / (h**2)
    phi_yy = (stencils_raw[:, 7] - 2.0 * stencils_raw[:, 4] + stencils_raw[:, 1]) / (h**2)
    phi_xy = (stencils_raw[:, 8] - stencils_raw[:, 6] - stencils_raw[:, 2] + stencils_raw[:, 0]) / (4.0 * h**2)

    numerator = phi_x**2 * phi_yy - 2.0 * phi_x * phi_y * phi_xy + phi_y**2 * phi_xx
    denominator = (phi_x**2 + phi_y**2) ** 1.5
    prediction = np.zeros_like(numerator)
    mask = denominator > 1e-12
    prediction[mask] = numerator[mask] / denominator[mask]
    return h * prediction
