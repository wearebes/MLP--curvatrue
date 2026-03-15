from __future__ import annotations

from pathlib import Path

try:
    import pandas as pd
except ImportError:  # pragma: no cover
    pd = None

from train.data_save import create_result_output_path


RESULT_COLUMNS = [
    "data_split",
    "group_id",
    "rho_model",
    "step_or_iter",
    "method",
    "N_samples",
    "MAE_hk",
    "MaxAE_hk",
    "MSE_hk",
]


def save_unified_results(rows: list[dict[str, object]], output_dir: str | Path) -> Path:
    output_path = create_result_output_path(output_dir)
    table = format_results_table(rows)
    output_path.write_text(table + "\n", encoding="utf-8")
    return output_path


def format_results_table(rows: list[dict[str, object]]) -> str:
    normalized = [{column: row.get(column, "") for column in RESULT_COLUMNS} for row in rows]
    if pd is not None:
        frame = pd.DataFrame(normalized, columns=RESULT_COLUMNS)
        formatters = {
            "MAE_hk": lambda value: f"{float(value):.6e}",
            "MaxAE_hk": lambda value: f"{float(value):.6e}",
            "MSE_hk": lambda value: f"{float(value):.6e}",
        }
        return frame.to_string(index=False, formatters=formatters)

    prepared_rows = []
    for row in normalized:
        prepared_rows.append(
            {
                "data_split": str(row["data_split"]),
                "group_id": str(row["group_id"]),
                "rho_model": str(row["rho_model"]),
                "step_or_iter": str(row["step_or_iter"]),
                "method": str(row["method"]),
                "N_samples": str(row["N_samples"]),
                "MAE_hk": f"{float(row['MAE_hk']):.6e}",
                "MaxAE_hk": f"{float(row['MaxAE_hk']):.6e}",
                "MSE_hk": f"{float(row['MSE_hk']):.6e}",
            }
        )

    widths = {
        column: max(len(column), *(len(prepared[column]) for prepared in prepared_rows))
        for column in RESULT_COLUMNS
    }
    header = " ".join(column.rjust(widths[column]) for column in RESULT_COLUMNS)
    lines = [header]
    for prepared in prepared_rows:
        lines.append(" ".join(prepared[column].rjust(widths[column]) for column in RESULT_COLUMNS))
    return "\n".join(lines)
