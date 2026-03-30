"""
train/__main__.py — ``python -m train`` 入口。
"""
from __future__ import annotations

import argparse
import os
from datetime import datetime
from pathlib import Path

import yaml

os.environ["PYTHONUNBUFFERED"] = "1"

import torch
import torch.multiprocessing as mp
import concurrent.futures

from .config import load_training_config, DEFAULT_CONFIG_PATH

try:
    import pandas as pd
except ImportError:
    pd = None


RESULT_COLUMNS = [
    "rho", "best_epoch", "best_val_mae",
    "test_mse", "test_mae", "test_max_ae", "elapsed_seconds",
]


def format_results_table(rows: list[dict[str, object]]) -> str:
    if pd is not None:
        frame = pd.DataFrame(rows, columns=RESULT_COLUMNS)
        formatters = {
            "best_val_mae": lambda v: f"{float(v):.6e}",
            "test_mse": lambda v: f"{float(v):.6e}",
            "test_mae": lambda v: f"{float(v):.6e}",
            "test_max_ae": lambda v: f"{float(v):.6e}",
            "elapsed_seconds": lambda v: f"{float(v):.2f}",
        }
        return frame.to_string(index=False, formatters=formatters)

    prepared = []
    for row in rows:
        prepared.append({
            "rho": str(row["rho"]),
            "best_epoch": str(row["best_epoch"]),
            "best_val_mae": f"{float(row['best_val_mae']):.6e}",
            "test_mse": f"{float(row['test_mse']):.6e}",
            "test_mae": f"{float(row['test_mae']):.6e}",
            "test_max_ae": f"{float(row['test_max_ae']):.6e}",
            "elapsed_seconds": f"{float(row['elapsed_seconds']):.2f}",
        })
    if not prepared:
        return "No results."
    widths = {c: max(len(c), *(len(r[c]) for r in prepared)) for c in RESULT_COLUMNS}
    lines = [" ".join(c.rjust(widths[c]) for c in RESULT_COLUMNS)]
    for r in prepared:
        lines.append(" ".join(r[c].rjust(widths[c]) for c in RESULT_COLUMNS))
    return "\n".join(lines)


def save_results_to_txt(results: list[dict[str, object]], output_path: str | Path) -> str:
    rows = [{c: r[c] for c in RESULT_COLUMNS} for r in results]
    table = format_results_table(rows)
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(table + "\n", encoding="utf-8")
    return table


def main() -> None:
    args = parse_args()
    cfg = load_training_config(args.config, overrides=args.config_overrides)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    data_dir = cfg.data_dir / args.dataset
    run_id = args.run_id if args.run_id else f"{args.dataset}_bs{cfg.batch_size}"
    model_dir = cfg.model_dir / run_id
    model_dir.mkdir(parents=True, exist_ok=True)

    dump = {
        "run_id": run_id,
        "dataset_source": args.dataset,
        "trained_at": datetime.now().isoformat(),
        "training_parameters": {
            "batch_size": cfg.batch_size,
            "max_epochs": cfg.max_epochs,
            "patience": cfg.patience,
            "optimizer": {"lr": cfg.optimizer.lr, "weight_decay": cfg.optimizer.weight_decay},
        },
    }
    with open(model_dir / "train_config.yaml", "w", encoding="utf-8") as f:
        yaml.dump(dump, f, sort_keys=False)

    # 根据 --sequential 参数选择并行或顺序执行
    if args.sequential:
        print(f"Running SEQUENTIAL training for rhos: {args.rho_values} on device: {device}")
        from .trainer import run_training_worker
        
        result_rows = []
        stop_info_list = []
        
        for rho in args.rho_values:
            try:
                print(f"[Main] Starting training for rho={rho}...")
                result = run_training_worker(
                    rho,
                    device,
                    args.config,
                    args.config_overrides,
                    str(data_dir),
                    str(model_dir),
                )
                result_rows.append(result)
                if result.get("stop_info"):
                    stop_info_list.append(f"[rho={rho}] {result['stop_info']}")
                print(f"[Main] Completed rho={rho}.")
            except Exception as e:
                print(f"[Main] Exception for rho={rho}: {e}")
    else:
        print(f"Submitting parallel jobs for rhos: {args.rho_values} on device: {device}")

        from .trainer import run_training_worker

        result_rows = []
        stop_info_list = []
        ctx = mp.get_context("spawn")

        with concurrent.futures.ProcessPoolExecutor(max_workers=len(args.rho_values), mp_context=ctx) as executor:
            futures = {
                executor.submit(
                    run_training_worker,
                    rho,
                    device,
                    args.config,
                    args.config_overrides,
                    str(data_dir),
                    str(model_dir),
                ): rho
                for rho in args.rho_values
            }
            for future in concurrent.futures.as_completed(futures):
                rho = futures[future]
                try:
                    result = future.result()
                    result_rows.append(result)
                    if result.get("stop_info"):
                        stop_info_list.append(f"[rho={rho}] {result['stop_info']}")
                    print(f"[Main] Completed rho={rho}.")
                except Exception as e:
                    print(f"[Main] Exception for rho={rho}: {e}")

    output_path = model_dir / "training_summary.txt"
    table = save_results_to_txt(result_rows, output_path)

    if stop_info_list:
        (model_dir / "stop_summary.txt").write_text("\n".join(stop_info_list) + "\n", encoding="utf-8")

    print("\nFinal Training Summary:")
    for line in table.splitlines():
        print(line)
    print(f"\nSaved to {output_path}")


def parse_args() -> argparse.Namespace:
    cfg = load_training_config()
    parser = argparse.ArgumentParser(description="Train rho-specific curvature MLP models.")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset ID (e.g., ds01)")
    parser.add_argument("--run-id", type=str, default="", help="Optional Run ID")
    parser.add_argument("--rho", nargs="+", type=int, default=list(cfg.resolutions),
                        help=f"rho values to train (default: {list(cfg.resolutions)})")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Config yaml path.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size from config.")
    parser.add_argument("--sequential", action="store_true",
                        help="Run training sequentially (one rho at a time) instead of in parallel")
    args = parser.parse_args()
    args.rho_values = args.rho
    args.config_overrides = {"batch_size": args.batch_size}
    return args


if __name__ == "__main__":
    import multiprocessing
    try:
        multiprocessing.set_start_method("spawn")
    except RuntimeError:
        pass
    try:
        main()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
