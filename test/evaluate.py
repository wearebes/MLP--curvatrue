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
from generate.test_data import hkappa_div_normal_from_field
from generate.test_blueprints import TEST_BLUEPRINTS
from train.config import load_training_config
from train.inference import load_inference_bundle


METHOD_LABELS = {
    "numerical": "Numerical",
    "neural": "Neural",
}


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
    rows.sort(key=lambda row: (int(row["rho_model"]), int(row["step_or_iter"]), method_order[row["method"]]))
    return rows


def _evaluate_train_data_numerical_full_field() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    field_builder = LevelSetFieldBuilder(dtype=np.float64)
    reinit_builder = ReinitFieldPackBuilder(cfl=CFL)
    print("Evaluating train Numerical with full-field div(normal); this rebuilds train fields in memory and can take a while.")

    for rho in TARGET_RHOS:
        grouped: dict[int, MetricAccumulator] = {}
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

        for step in sorted(grouped):
            metrics = grouped[step].as_metrics()
            rows.append(
                {
                    "data_split": "train",
                    "group_id": f"rho{rho}",
                    "rho_model": rho,
                    "step_or_iter": step,
                    "method": METHOD_LABELS["numerical"],
                    **metrics,
                }
            )
    return rows


def _update_train_numerical_accumulator(
    grouped: dict[int, MetricAccumulator],
    *,
    step: int,
    field_pack: dict[str, object],
) -> None:
    phi = field_pack["field"]["phi"]
    h = float(field_pack["params"]["h"])
    target_hk = float(field_pack["label"]["h_kappa"])
    i_coords, j_coords = ReinitQualityEvaluator.get_sampling_coordinates(phi)
    if len(i_coords) == 0:
        return

    indices = np.column_stack((i_coords, j_coords)).astype(np.int64, copy=False)
    target = np.full(indices.shape[0], target_hk, dtype=np.float64)
    pred = hkappa_div_normal_from_field(phi, indices, h)
    accumulator = grouped.setdefault(int(step), MetricAccumulator())

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
                    **metrics,
                }
            )
    return rows


def evaluate_test_data(project_root: Path, methods: list[str]) -> list[dict[str, object]]:
    predictor = NeuralPredictor(project_root) if "neural" in methods else None
    rows: list[dict[str, object]] = []
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

        for step in [5, 10, 20]:
            h5_path = exp_dir / f"iter_{step}.h5"
            if not h5_path.exists():
                continue
            with h5py.File(h5_path, "r") as handle:
                stencils_raw = handle["stencils_raw"][:]
                target = handle["hkappa_target"][:]
                numerical = handle["hkappa_fd"][:]

            predictions = {}
            if "numerical" in methods:
                predictions["numerical"] = numerical
            if "neural" in methods and predictor is not None:
                predictions["neural"] = predictor.predict(rho_model, stencils_raw)

            for method in methods:
                accumulator = MetricAccumulator()
                accumulator.update(target, predictions[method])
                metrics = accumulator.as_metrics()
                rows.append(
                    {
                        "data_split": "test",
                        "group_id": group_id,
                        "rho_model": rho_model,
                        "step_or_iter": step,
                        "method": METHOD_LABELS[method],
                        **metrics,
                    }
                )
    return rows
