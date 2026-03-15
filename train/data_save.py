from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from typing import Iterable

try:
    import pandas as pd
except ImportError:  # pragma: no cover - fallback when pandas is unavailable
    pd = None


RESULT_COLUMNS = [
    "rho",
    "best_epoch",
    "best_val_mae",
    "test_mse",
    "test_mae",
    "test_max_ae",
    "elapsed_seconds",
]


def create_result_output_path(results_dir: str | Path, *, now: datetime | None = None) -> Path:
    return _create_timestamped_output_path(results_dir, prefix="result", now=now)


def create_training_log_output_path(results_dir: str | Path, *, now: datetime | None = None) -> Path:
    return _create_timestamped_output_path(results_dir, prefix="train_log", now=now)


def append_log_line(output_path: str | Path, message: str) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as handle:
        handle.write(message.rstrip("\n") + "\n")


def _create_timestamped_output_path(
    results_dir: str | Path,
    *,
    prefix: str,
    now: datetime | None = None,
) -> Path:
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    timestamp = (now or datetime.now()).strftime("%m%d")
    base_path = results_dir / f"{prefix}_{timestamp}.txt"
    if not base_path.exists():
        return base_path

    suffix = 2
    while True:
        candidate = results_dir / f"{prefix}_{timestamp}_{suffix}.txt"
        if not candidate.exists():
            return candidate
        suffix += 1


def save_results_to_txt(results_list: Iterable[dict[str, object]], output_path: str | Path) -> str:
    rows = [{column: result[column] for column in RESULT_COLUMNS} for result in results_list]
    table = format_results_table(rows)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(table + "\n", encoding="utf-8")
    return table


def save_zscore_stats(rho: int, mu, sigma, output_dir: str | Path) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"zscore_stats_{rho}.csv"

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["feature_index", "mu", "sigma"])
        for index, (mu_value, sigma_value) in enumerate(zip(_to_flat_list(mu), _to_flat_list(sigma), strict=True)):
            writer.writerow([index, float(mu_value), float(sigma_value)])

    return csv_path


def format_results_table(rows: list[dict[str, object]]) -> str:
    if pd is not None:
        frame = pd.DataFrame(rows, columns=RESULT_COLUMNS)
        formatters = {
            "best_val_mae": lambda value: f"{float(value):.6f}",
            "test_mse": lambda value: f"{float(value):.6f}",
            "test_mae": lambda value: f"{float(value):.6f}",
            "test_max_ae": lambda value: f"{float(value):.6f}",
            "elapsed_seconds": lambda value: f"{float(value):.2f}",
        }
        return frame.to_string(index=False, formatters=formatters)

    prepared_rows = []
    for row in rows:
        prepared_rows.append(
            {
                "rho": str(row["rho"]),
                "best_epoch": str(row["best_epoch"]),
                "best_val_mae": f"{float(row['best_val_mae']):.6f}",
                "test_mse": f"{float(row['test_mse']):.6f}",
                "test_mae": f"{float(row['test_mae']):.6f}",
                "test_max_ae": f"{float(row['test_max_ae']):.6f}",
                "elapsed_seconds": f"{float(row['elapsed_seconds']):.2f}",
            }
        )

    widths = {
        column: max(len(column), *(len(prepared_row[column]) for prepared_row in prepared_rows))
        for column in RESULT_COLUMNS
    }
    header = " ".join(column.rjust(widths[column]) for column in RESULT_COLUMNS)
    lines = [header]
    for prepared_row in prepared_rows:
        lines.append(" ".join(prepared_row[column].rjust(widths[column]) for column in RESULT_COLUMNS))
    return "\n".join(lines)


def _to_flat_list(values) -> list[float]:
    if hasattr(values, "detach"):
        values = values.detach().cpu().reshape(-1).tolist()
    elif hasattr(values, "reshape"):
        values = values.reshape(-1).tolist()
    else:
        values = list(values)
    return [float(value) for value in values]
