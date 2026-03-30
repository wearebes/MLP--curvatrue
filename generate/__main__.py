"""
generate/__main__.py — ``python -m generate`` 入口。

子命令：
  train   生成训练 HDF5 数据集
  test    生成可配置初值模式的评估数据集
"""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import yaml

from .config import (
    TEST_DATA_MODES,
    load_generate_config,
)


def main() -> None:
    cfg = load_generate_config()

    parser = argparse.ArgumentParser(description="Generate train or test datasets.")
    parser.add_argument(
        "--config", type=str, default=None,
        help="Override config.yaml path.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ── train ──────────────────────────────────────────────
    train_p = subparsers.add_parser("train", help="Generate training datasets.")
    train_p.add_argument(
        "--rho", type=int, nargs="+", default=list(cfg.train_data.resolutions),
        help="Resolution(s) to generate",
    )
    train_p.add_argument("--dataset-name", type=str, required=True,
                         help="Dataset folder name (e.g. ds_smooth_01)")

    # ── test ───────────────────────────────────────────────
    test_p = subparsers.add_parser("test", help="Generate test datasets.")
    test_p.add_argument("--dataset-name", type=str, required=True,
                        help="Test dataset folder name (e.g. test_flower_01)")
    test_p.add_argument(
        "--mode",
        choices=TEST_DATA_MODES,
        default=None,
        help="Override test-data initial-field mode.",
    )

    args = parser.parse_args()

    # reload config if overridden
    gen_cfg = load_generate_config(args.config) if args.config else cfg
    output_dir = gen_cfg.data_dir / args.dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.command == "train":
        from .train_data import generate_train_datasets

        td = gen_cfg.train_data
        data_config = {
            "dataset_name": args.dataset_name,
            "type":         "train",
            "generated_at": datetime.now().isoformat(),
            "target_rhos":  args.rho,
            "cfl":          td.cfl,
            "sign_mode":    td.sign_mode,
            "time_order":   td.time_order,
            "space_order":  td.space_order,
            "variations":   td.variations,
            "seed":         td.geometry_seed,
            "reinit_steps": list(td.reinit_steps),
        }
        with open(output_dir / "data_config.yaml", "w", encoding="utf-8") as f:
            yaml.dump(data_config, f, sort_keys=False)

        generate_train_datasets(
            target_rhos=args.rho,
            output_dir=output_dir,
            config=gen_cfg,
        )

    elif args.command == "test":
        from .test_data import generate_test_datasets

        td2 = gen_cfg.test_data
        test_mode = args.mode or td2.mode
        data_config = {
            "dataset_name": args.dataset_name,
            "type":         "test",
            "mode":         test_mode,
            "generated_at": datetime.now().isoformat(),
            "cfl":          td2.cfl,
            "eps_sign_factor": td2.eps_sign_factor,
            "sign_mode":    td2.sign_mode,
            "time_order":   td2.time_order,
            "space_order":  td2.space_order,
        }
        if test_mode == "formula_phi0_projection_band":
            data_config["formula_projection_band_cells"] = td2.formula_projection_band_cells
        if test_mode == "exact_sdf":
            data_config.update(
                {
                    "exact_sdf_method": td2.exact_sdf_method,
                    "exact_sdf_mp_dps": td2.exact_sdf_mp_dps,
                    "exact_sdf_newton_tol": td2.exact_sdf_newton_tol,
                    "exact_sdf_newton_max_iter": td2.exact_sdf_newton_max_iter,
                }
            )
        with open(output_dir / "data_config.yaml", "w", encoding="utf-8") as f:
            yaml.dump(data_config, f, sort_keys=False)

        generate_test_datasets(output_dir=output_dir, config=gen_cfg, mode_override=test_mode)


if __name__ == "__main__":
    main()
