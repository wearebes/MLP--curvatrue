from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from project_runtime import cleanup_bytecode_caches, disable_bytecode_cache

disable_bytecode_cache()

if __package__ in {None, ""}:
    from train.config import load_training_config
    from train.data_save import (
        append_log_line,
        create_result_output_path,
        create_training_log_output_path,
        save_results_to_txt,
        save_zscore_stats,
    )
    from train.dataloader import build_dataloaders_from_h5
    from train.model_structure import build_model_for_rho
    from train.train import fit_regression_model, set_all_seeds
else:
    from .config import load_training_config
    from .data_save import (
        append_log_line,
        create_result_output_path,
        create_training_log_output_path,
        save_results_to_txt,
        save_zscore_stats,
    )
    from .dataloader import build_dataloaders_from_h5
    from .model_structure import build_model_for_rho
    from .train import fit_regression_model, set_all_seeds


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parent.parent
    config = load_training_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_dir = project_root / "model"
    results_dir = project_root / "train" / "results"
    training_log_path = create_training_log_output_path(results_dir)
    result_rows: list[dict[str, object]] = []

    def emit(message: str) -> None:
        print(message)
        append_log_line(training_log_path, message)

    for rho in args.rho_values:
        set_all_seeds(config.training.seed)
        data_path = project_root / "data" / f"train_rho{rho}.h5"
        checkpoint_path = model_dir / f"model_rho{rho}.pth"

        emit("")
        emit("=" * 72)
        emit(f"Starting training for rho={rho} on device={device}")
        emit("=" * 72)

        train_loader, val_loader, test_loader, meta = build_dataloaders_from_h5(
            data_path,
            batch_size=config.training.batch_size,
            seed=config.training.seed,
            num_workers=0,
        )
        emit(
            f"[rho={rho}] Data split | total={meta['N_total']} train={meta['N_train']} "
            f"val={meta['N_val']} test={meta['N_test']} batch_size={meta['batch_size']}"
        )
        if meta["zero_sigma_features"]:
            emit(
                f"[rho={rho}] Warning: {meta['zero_sigma_features']} feature(s) had zero std "
                "and were kept at sigma=1.0 to avoid division by zero."
            )

        model = build_model_for_rho(rho, config.model)
        save_zscore_stats(rho, meta["mu"], meta["sigma"], model_dir)

        result = fit_regression_model(
            rho,
            model,
            train_loader,
            val_loader,
            test_loader,
            optimizer_config=config.optimizer,
            training_config=config.training,
            device=device,
            save_path=checkpoint_path,
            log_message=lambda message: append_log_line(training_log_path, message),
        )
        result_rows.append(result)

    output_path = create_result_output_path(results_dir)
    table = save_results_to_txt(result_rows, output_path)
    emit("")
    emit("Final summary:")
    for line in table.splitlines():
        emit(line)
    emit("")
    emit(f"Final summary saved to {output_path}")
    emit(f"Training log saved to {training_log_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train rho-specific curvature MLP models.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    rho_parser = subparsers.add_parser("rho", help="Train one or more rho-specific models.")
    rho_parser.add_argument("rho_values", nargs="+", type=int, help="rho values to train, e.g. 256 266 276")
    rho_parser.add_argument(
        "--config",
        default=str(Path(__file__).resolve().with_name("config.txt")),
        help="Path to the training config file.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    try:
        main()
    finally:
        cleanup_bytecode_caches(PROJECT_ROOT)
