from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np

from generate.numerics import (
    ReinitQualityEvaluator,
    build_grid,
    build_flower_phi0,
    compute_hkappa,
    find_projection_theta,
    hkappa_analytic,
)
from generate.pde import LevelSetReinitializer
from generate.train_data import (
    CircleGeometryGenerator,
    LevelSetFieldBuilder,
    ReinitFieldPackBuilder,
    extract_3x3_stencils,
)
from generate.config import load_generate_config

from .utils import (
    CURVATURE_METHOD_NEURAL,
    CURVATURE_METHOD_STANDARD,
    DATA_SOURCE_H5_RAW,
    DATA_SOURCE_REBUILT,
    METHOD_LABELS,
    METRIC_VIEW_CURRENT,
    MODEL_TAG_BASELINE,
    MetricAccumulator,
    NeuralPredictor,
    OriginPredictor,
    build_metric_row,
    build_test_diagnostics_row,
    central_grad_norm,
    collect_zero_crossing_points,
    compute_curvature_predictions,
    discover_model_directories,
    discover_train_rhos,
    print_encoding_diagnostic_once,
    print_neural_step_variation,
    print_test_diagnostics,
    select_payload,
    train_style_sampling_indices,
)


def evaluate_train_data(
    project_root: Path,
    methods: list[str],
    *,
    config_path: str | Path | None = None,
    data_dir: Path | None = None,
    chunk_size: int = 100000,
    run_id: str | None = None,
) -> list[dict[str, object]]:
    gen_cfg = load_generate_config()
    if data_dir is None:
        data_dir = gen_cfg.data_dir

    rows: list[dict[str, object]] = []
    if "numerical" in methods:
        rows.extend(_evaluate_train_data_numerical_full_field(gen_cfg, data_dir=data_dir))
    if "neural" in methods:
        if run_id:
            model_dirs = [project_root / "model" / run_id]
        else:
            model_dirs = discover_model_directories(project_root)
        if not model_dirs:
            print("No model checkpoints found under model/. Skipping neural train evaluation.")
        for model_dir in model_dirs:
            predictor = NeuralPredictor(project_root, model_dir=model_dir, config_path=config_path)
            print(f"Evaluating train Neural using model directory: {model_dir}")
            rows.extend(
                _evaluate_train_data_neural(
                    data_dir,
                    gen_cfg.train_data.resolutions,
                    predictor=predictor,
                    chunk_size=chunk_size,
                )
            )
    if "origin" in methods:
        predictor = OriginPredictor(project_root)
        if not predictor.model_dir.exists():
            print("No origin model directory found under model/origin. Skipping origin train evaluation.")
        else:
            print(f"Evaluating train Origin using model directory: {predictor.model_dir}")
            rows.extend(
                _evaluate_train_data_neural(
                    gen_cfg.data_dir,
                    gen_cfg.train_data.resolutions,
                    predictor=predictor,
                    chunk_size=chunk_size,
                    method_label=METHOD_LABELS["origin"],
                )
            )

    method_order = {METHOD_LABELS[method]: index for index, method in enumerate(methods)}
    curvature_order = {
        CURVATURE_METHOD_STANDARD: 0,
        CURVATURE_METHOD_NEURAL: 1,
    }
    rows.sort(
        key=lambda row: (
            int(row["rho_model"]),
            int(row["step_or_iter"]),
            method_order[row["method"]],
            str(row.get("model_tag", "")),
            curvature_order.get(str(row["curvature_method"]), 99),
            str(row.get("metric_view", "")),
            str(row.get("data_source", "")),
        )
    )
    return rows


def _evaluate_train_data_numerical_full_field(gen_cfg, data_dir=None) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    if data_dir is None:
        data_dir = gen_cfg.data_dir
    field_builder = LevelSetFieldBuilder(dtype=np.float64)
    reinit_builder = ReinitFieldPackBuilder(cfl=gen_cfg.train_data.cfl)
    rho_values = discover_train_rhos(data_dir, gen_cfg.train_data.resolutions)
    print("Evaluating train Numerical with full-field expanded_formula; this rebuilds train fields in memory and can take a while.")

    for rho in rho_values:
        grouped: dict[tuple[int, str], MetricAccumulator] = {}
        generator = CircleGeometryGenerator(
            resolution_rho=rho,
            seed=gen_cfg.train_data.geometry_seed,
            variations=gen_cfg.train_data.variations,
        )
        blueprints = generator.generate_blueprints()
        print(f"  [train Numerical] rho={rho} | rebuilding {len(blueprints)} blueprints")

        for index, blueprint in enumerate(blueprints, start=1):
            pack_sdf = field_builder.build_circle_sdf(blueprint, return_grid=False)
            _update_train_numerical_accumulator(grouped, step=0, field_pack=pack_sdf)

            pack_nonsdf = field_builder.build_circle_nonsdf(blueprint, return_grid=False)
            reinitialized = reinit_builder.build(pack_nonsdf, steps_list=list(gen_cfg.train_data.reinit_steps))
            for step in gen_cfg.train_data.reinit_steps:
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
                    "model_tag": MODEL_TAG_BASELINE,
                    "curvature_method": curvature_method,
                    "metric_view": "current_interface_nodes",
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
        raise ValueError("Training circle h*kappa must stay positive under the current sign convention.")
    i_coords, j_coords = ReinitQualityEvaluator.get_sampling_coordinates(phi)
    if len(i_coords) == 0:
        return

    indices = np.column_stack((i_coords, j_coords)).astype(np.int64, copy=False)
    target = np.full(indices.shape[0], target_hk, dtype=np.float64)
    predictions = compute_curvature_predictions(phi, indices, h)

    for curvature_method, pred in predictions.items():
        accumulator = grouped.setdefault((int(step), curvature_method), MetricAccumulator())
        accumulator.update(target, pred)
        accumulator.update(-target, -pred)


def _evaluate_train_data_neural(
    data_dir: Path,
    configured_rhos: tuple[int, ...],
    *,
    predictor: NeuralPredictor,
    chunk_size: int,
    method_label: str = METHOD_LABELS["neural"],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for rho in discover_train_rhos(data_dir, configured_rhos):
        if not predictor.has_bundle(rho):
            predictor.note_missing_bundle_once(rho)
            continue
        h5_path = data_dir / f"train_rho{rho}.h5"
        if not h5_path.exists():
            continue

        grouped: dict[int, MetricAccumulator] = {}
        with h5py.File(h5_path, "r") as handle:
            total_samples = handle["X"].shape[0]
            for start in range(0, total_samples, chunk_size):
                end = min(start + chunk_size, total_samples)
                x = handle["X"][start:end]
                y_true = handle["Y"][start:end, 0]
                steps = handle["reinit_steps"][start:end, 0].astype(np.int32)
                pred = predictor.predict(rho, x)

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
                    "method": method_label,
                    "model_tag": predictor.model_tag,
                    "curvature_method": CURVATURE_METHOD_NEURAL,
                    "metric_view": "current_interface_nodes",
                    "data_source": DATA_SOURCE_H5_RAW,
                    **metrics,
                }
            )
    return rows


def evaluate_test_data(
    project_root: Path,
    methods: list[str],
    *,
    config_path: str | Path | None = None,
    data_dir: Path | None = None,
    data_source_opt: str | None = None,
    run_id: str | None = None,
) -> list[dict[str, object]]:
    gen_cfg = load_generate_config()
    neural_predictors: list[NeuralPredictor] = []
    origin_predictor: OriginPredictor | None = None
    if "neural" in methods:
        if run_id:
            model_dirs = [project_root / "model" / run_id]
        else:
            model_dirs = discover_model_directories(project_root)
        if not model_dirs:
            print("No model checkpoints found under model/. Skipping neural test evaluation.")
        for model_dir in model_dirs:
            print(f"Evaluating test Neural using model directory: {model_dir}")
            neural_predictors.append(NeuralPredictor(project_root, model_dir=model_dir, config_path=config_path))
    if "origin" in methods:
        origin_predictor = OriginPredictor(project_root)
        if not origin_predictor.model_dir.exists():
            print("No origin model directory found under model/origin. Skipping origin test evaluation.")

    rows: list[dict[str, object]] = []
    diagnostics: list[dict[str, object]] = []
    if data_dir is None:
        data_dir = gen_cfg.data_dir
    if data_source_opt is None:
        data_source_opt = DATA_SOURCE_REBUILT
    use_stored_h5_payloads = data_source_opt != DATA_SOURCE_REBUILT
    ordered_exp_ids = [s.exp_id for s in gen_cfg.test_data.scenarios]
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
        active_predictors: list[NeuralPredictor] = []
        for predictor in neural_predictors:
            if predictor.has_bundle(rho_model):
                active_predictors.append(predictor)
            else:
                predictor.note_missing_bundle_once(rho_model)
        blueprint_id = str(meta.get("blueprint_id", meta.get("exp_id", exp_dir.name)))
        experiment_type = str(meta.get("experiment_type", meta.get("gamma_type", "test")))
        group_id = f"{experiment_type}/{blueprint_id}"
        fixed_payloads: dict[int, dict[str, np.ndarray | float]] = {}
        current_payloads: dict[int, dict[str, np.ndarray | float]] = {}
        x_grid = y_grid = phi0 = phi0_points = None
        h = 0.0
        if not use_stored_h5_payloads:
            fixed_payloads = _build_fixed_test_payloads(
                meta,
                cfl=gen_cfg.test_data.cfl,
                test_iters=gen_cfg.test_data.test_iters,
            )
            current_payloads = _build_current_test_payloads(
                meta,
                cfl=gen_cfg.test_data.cfl,
                test_iters=gen_cfg.test_data.test_iters,
            )
            x_grid, y_grid, h = build_grid(float(meta["L"]), int(meta["N"]))
            phi0 = build_flower_phi0(x_grid, y_grid, float(meta["a"]), float(meta["b"]), float(meta["p"]))
            phi0_points = collect_zero_crossing_points(phi0, x_grid, y_grid)

        for step in gen_cfg.test_data.test_iters:
            h5_path = exp_dir / f"iter_{step}.h5"
            if not h5_path.exists():
                continue
            if use_stored_h5_payloads:
                metric_view = str(meta.get("sampling_rule", METRIC_VIEW_CURRENT))
                selected_payload = _load_stored_test_payload(h5_path)
            else:
                fixed_payload = fixed_payloads.get(int(step))
                current_payload = current_payloads.get(int(step))
                if fixed_payload is None or current_payload is None:
                    continue
                metric_view, selected_payload = select_payload(
                    fixed_payload,
                    current_payload,
                    metric_view_policy="larger_node_set",
                )
                print_encoding_diagnostic_once(
                    enabled=False,
                    group_id=group_id,
                    step=int(step),
                    phi=np.asarray(selected_payload["phi"], dtype=np.float64),
                    indices=np.asarray(selected_payload["indices"], dtype=np.int64),
                    encoded_stencils=np.asarray(selected_payload["stencils_raw"], dtype=np.float64),
                )

            if "numerical" in methods:
                rows.extend(
                    _build_numerical_test_rows(
                        group_id=group_id,
                        rho_model=rho_model,
                        step=int(step),
                        metric_view=metric_view,
                        data_source=data_source_opt,
                        model_tag=MODEL_TAG_BASELINE,
                        payload=selected_payload,
                    )
                )

            if "neural" in methods:
                for predictor in active_predictors:
                    selected_pred = predictor.predict(rho_model, selected_payload["stencils_raw"])
                    rows.append(
                        build_metric_row(
                            data_split="test",
                            group_id=group_id,
                            rho_model=rho_model,
                            step=int(step),
                            method=METHOD_LABELS["neural"],
                            model_tag=predictor.model_tag,
                            curvature_method=CURVATURE_METHOD_NEURAL,
                            metric_view=metric_view,
                            data_source=data_source_opt,
                            target=selected_payload["target"],
                            pred=selected_pred,
                        )
                    )

            if "origin" in methods and origin_predictor is not None:
                if origin_predictor.has_bundle(rho_model):
                    selected_pred = origin_predictor.predict(rho_model, selected_payload["stencils_raw"])
                    rows.append(
                        build_metric_row(
                            data_split="test",
                            group_id=group_id,
                            rho_model=rho_model,
                            step=int(step),
                            method=METHOD_LABELS["origin"],
                            model_tag=origin_predictor.model_tag,
                            curvature_method=CURVATURE_METHOD_NEURAL,
                            metric_view=metric_view,
                            data_source=data_source_opt,
                            target=selected_payload["target"],
                            pred=selected_pred,
                        )
                    )
                else:
                    origin_predictor.note_missing_bundle_once(rho_model)

            if not use_stored_h5_payloads:
                diagnostics.append(
                    build_test_diagnostics_row(
                        group_id=group_id,
                        step=int(step),
                        h=h,
                        x_grid=x_grid,
                        y_grid=y_grid,
                        phi0_points=phi0_points,
                        fixed_indices=np.asarray(fixed_payload["indices"], dtype=np.int64),
                        current_payload=current_payload,
                    )
                )
    if diagnostics:
        print_test_diagnostics(diagnostics)
    print_neural_step_variation(rows)
    return rows


def _build_fixed_test_payloads(
    meta: dict[str, object],
    *,
    cfl: float,
    test_iters: tuple[int, ...],
) -> dict[int, dict[str, np.ndarray | float]]:
    x_grid, y_grid, h = build_grid(float(meta["L"]), int(meta["N"]))
    phi0 = build_flower_phi0(x_grid, y_grid, float(meta["a"]), float(meta["b"]), float(meta["p"]))
    fixed_indices = train_style_sampling_indices(phi0)
    fixed_xy = np.column_stack((x_grid[fixed_indices[:, 0], fixed_indices[:, 1]], y_grid[fixed_indices[:, 0], fixed_indices[:, 1]])) if len(fixed_indices) else np.zeros((0, 2))
    fixed_theta_proj = find_projection_theta(fixed_xy, float(meta["a"]), float(meta["b"]), float(meta["p"]))
    fixed_target = hkappa_analytic(fixed_theta_proj, h, float(meta["a"]), float(meta["b"]), float(meta["p"]))
    reinitializer = LevelSetReinitializer(indexing="xy", cfl=cfl)
    payloads: dict[int, dict[str, np.ndarray | float]] = {}

    for step in test_iters:
        phi = reinitializer.reinitialize(phi0, h, int(step))
        payloads[int(step)] = {
            "phi": phi,
            "indices": fixed_indices,
            "stencils_raw": extract_3x3_stencils(phi, fixed_indices),
            "target": fixed_target,
            "curvature_predictions": compute_curvature_predictions(phi, fixed_indices, h),
        }
    return payloads


def _build_current_test_payloads(
    meta: dict[str, object],
    *,
    cfl: float,
    test_iters: tuple[int, ...],
) -> dict[int, dict[str, np.ndarray | float]]:
    x_grid, y_grid, h = build_grid(float(meta["L"]), int(meta["N"]))
    phi0 = build_flower_phi0(x_grid, y_grid, float(meta["a"]), float(meta["b"]), float(meta["p"]))
    reinitializer = LevelSetReinitializer(indexing="xy", cfl=cfl)
    payloads: dict[int, dict[str, np.ndarray | float]] = {}

    for step in test_iters:
        phi = reinitializer.reinitialize(phi0, h, int(step))
        indices = train_style_sampling_indices(phi)
        if indices.size == 0:
            payloads[int(step)] = {
                "phi": phi,
                "indices": indices,
                "stencils_raw": np.zeros((0, 9), dtype=np.float64),
                "target": np.zeros((0,), dtype=np.float64),
                "curvature_predictions": {
                    CURVATURE_METHOD_STANDARD: np.zeros((0,), dtype=np.float64),
                },
                "eikonal_abs_on_samples": np.zeros((0,), dtype=np.float64),
            }
            continue

        xy = np.column_stack((x_grid[indices[:, 0], indices[:, 1]], y_grid[indices[:, 0], indices[:, 1]])) if len(indices) else np.zeros((0, 2))
        theta_proj = find_projection_theta(xy, float(meta["a"]), float(meta["b"]), float(meta["p"]))
        target = hkappa_analytic(theta_proj, h, float(meta["a"]), float(meta["b"]), float(meta["p"]))
        stencils_raw = extract_3x3_stencils(phi, indices)
        grad_norm = central_grad_norm(phi, h)

        payloads[int(step)] = {
            "phi": phi,
            "indices": indices,
            "stencils_raw": stencils_raw,
            "target": target,
            "curvature_predictions": compute_curvature_predictions(phi, indices, h),
            "eikonal_abs_on_samples": np.abs(grad_norm[indices[:, 0], indices[:, 1]] - 1.0),
        }
    return payloads


def _build_numerical_test_rows(
    *,
    group_id: str,
    rho_model: int,
    step: int,
    metric_view: str,
    data_source: str,
    model_tag: str,
    payload: dict[str, np.ndarray | float],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    target = np.asarray(payload["target"], dtype=np.float64)
    predictions = payload["curvature_predictions"]
    for curvature_method, pred in predictions.items():
        rows.append(
            build_metric_row(
                data_split="test",
                group_id=group_id,
                rho_model=rho_model,
                step=step,
                method=METHOD_LABELS["numerical"],
                model_tag=model_tag,
                curvature_method=curvature_method,
                metric_view=metric_view,
                data_source=data_source,
                target=target,
                pred=np.asarray(pred, dtype=np.float64),
            )
        )
    return rows


def _load_stored_test_payload(h5_path: Path) -> dict[str, np.ndarray]:
    with h5py.File(h5_path, "r") as handle:
        target = np.asarray(handle["hkappa_target"][:], dtype=np.float64)
        stencils_raw = np.asarray(handle["stencils_raw"][:], dtype=np.float64)
        indices = np.asarray(handle["indices"][:], dtype=np.int64)
        numerical = np.asarray(handle["hkappa_fd"][:], dtype=np.float64)
    return {
        "indices": indices,
        "stencils_raw": stencils_raw,
        "target": target,
        "curvature_predictions": {
            CURVATURE_METHOD_STANDARD: numerical,
        },
    }
