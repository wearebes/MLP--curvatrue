from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


METRICS = ["MSE_hk", "MAE_hk", "MaxAE_hk"]
METRIC_TITLES = ["Test MSE", "Test MAE", "Test Max AE"]
CFL_MAP = {
    "data02_001": "CFL=0.1",
    "data02_002": "CFL=0.2",
    "data02_004": "CFL=0.4",
    "data02_005": "CFL=0.5",
}
CFL_VALUES = [0.1, 0.2, 0.4, 0.5]
RHO_VALUES = [256, 266, 276]
GEOMETRY_GROUPS = {
    "smooth": [f"smooth/smooth_{rho}" for rho in RHO_VALUES],
    "acute": [f"acute/acute_{rho}" for rho in RHO_VALUES],
}
METHOD_COLORS = {
    "Numerical": "black",
    "Origin": "gray",
    "CFL=0.1": "#1f77b4",
    "CFL=0.2": "#ff7f0e",
    "CFL=0.4": "#2ca02c",
    "CFL=0.5": "#d62728",
}
METHOD_MARKERS = {
    "Numerical": "X",
    "Origin": "s",
    "CFL=0.1": "o",
    "CFL=0.2": "o",
    "CFL=0.4": "o",
    "CFL=0.5": "o",
}
METHOD_LINESTYLES = {
    "Numerical": "--",
    "Origin": "-.",
    "CFL=0.1": "-",
    "CFL=0.2": "-",
    "CFL=0.4": "-",
    "CFL=0.5": "-",
}


def parse_unified_evaluation(filepath):
    data = []
    current_method = None
    current_model = None
    with open(filepath, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("[test]"):
                parts = line.split("|")
                method_part = parts[0].replace("[test]", "").strip()
                model_part = [p.strip() for p in parts if p.strip().startswith("model=")][0]
                current_method = method_part
                current_model = model_part.split("=")[1].strip()
                continue
            if line.startswith("group_id"):
                continue

            cols = line.split()
            if len(cols) < 7:
                continue

            data.append(
                {
                    "method": current_method,
                    "model_tag": current_model,
                    "group_id": cols[0],
                    "rho_model": int(cols[1]),
                    "step_or_iter": int(cols[2]),
                    "N_samples": int(cols[3]),
                    "MAE_hk": float(cols[4]),
                    "MaxAE_hk": float(cols[5]),
                    "MSE_hk": float(cols[6]),
                }
            )
    return pd.DataFrame(data)


def scientific_formatter(x, pos):
    if x == 0:
        return "0"
    exp = int(np.floor(np.log10(abs(x))))
    coeff = x / 10**exp
    return f"${coeff:.2f} \\times 10^{{{exp}}}$"


def infer_cfl_label(folder_name):
    for key, label in CFL_MAP.items():
        if key in folder_name:
            return label
    return "Unknown CFL"


def infer_display_name(method, cfl_label):
    if method == "Neural":
        return cfl_label
    return method


def extract_cfl_value(display_name):
    if not isinstance(display_name, str) or not display_name.startswith("CFL="):
        return np.nan
    return float(display_name.split("=")[1])


def ordered_display_names(df):
    names = set(df["Display_Name"].dropna())
    ordered = []
    for fixed in ["Numerical", "Origin"]:
        if fixed in names:
            ordered.append(fixed)
    cfl_names = sorted(
        [name for name in names if name.startswith("CFL=")],
        key=extract_cfl_value,
    )
    ordered.extend(cfl_names)
    ordered.extend(sorted(name for name in names if name not in ordered))
    return ordered


def load_combined_evaluations(base_dir):
    eval_files = list(base_dir.glob("data02_*_batch256/evals/unified_evaluation.txt"))
    dfs = []
    for eval_file in eval_files:
        df = parse_unified_evaluation(eval_file)
        if df.empty:
            continue
        cfl_label = infer_cfl_label(eval_file.parent.parent.name)
        df["Display_Name"] = df["method"].map(lambda method: infer_display_name(method, cfl_label))
        df["CFL_Val"] = df["Display_Name"].map(extract_cfl_value)
        dfs.append(df)

    if not dfs:
        return pd.DataFrame()

    combo_df = pd.concat(dfs, ignore_index=True)
    return combo_df.drop_duplicates(subset=["Display_Name", "group_id", "step_or_iter"])


def ensure_axis_has_data(ax, has_data):
    if has_data:
        return
    ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center", fontsize=12)
    ax.set_xticks([])
    ax.set_yticks([])


def setup_axis(ax, formatter, metric_title=None, row_title=None, xlabel=None):
    ax.grid(True, alpha=0.35, linestyle="--")
    ax.yaxis.set_major_formatter(formatter)
    if metric_title:
        ax.set_title(metric_title, fontsize=15, pad=10)
    if row_title:
        ax.set_ylabel(row_title, fontsize=15, fontweight="bold")
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=12)


def plot_geometry_evolution(combo_df, geometry, formatter):
    fig, axes = plt.subplots(3, 3, figsize=(18, 14), sharex=True)
    group_ids = GEOMETRY_GROUPS[geometry]
    ordered_names = ordered_display_names(combo_df)

    for r, (rho, group_id) in enumerate(zip(RHO_VALUES, group_ids, strict=True)):
        rho_df = combo_df[combo_df["group_id"] == group_id].sort_values("step_or_iter")
        for c, metric in enumerate(METRICS):
            ax = axes[r, c]
            has_data = False
            for name in ordered_names:
                method_df = rho_df[rho_df["Display_Name"] == name]
                if method_df.empty:
                    continue
                has_data = True
                ax.plot(
                    method_df["step_or_iter"],
                    method_df[metric],
                    marker=METHOD_MARKERS.get(name, "o"),
                    linestyle=METHOD_LINESTYLES.get(name, "-"),
                    linewidth=2.2,
                    markersize=7,
                    color=METHOD_COLORS.get(name, "#444444"),
                    label=name,
                )
            setup_axis(
                ax,
                formatter,
                metric_title=METRIC_TITLES[c] if r == 0 else None,
                row_title=f"rho={rho}" if c == 0 else None,
                xlabel="Iterations" if r == 2 else None,
            )
            ax.set_xticks([5, 10, 20])
            ensure_axis_has_data(ax, has_data)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="lower center",
            ncol=min(6, len(labels)),
            bbox_to_anchor=(0.5, -0.02),
            title="Method / Training CFL",
            fontsize=12,
            title_fontsize=13,
        )
    title = "Smooth Error Evolution Across Resolutions" if geometry == "smooth" else "Acute Error Evolution Across Resolutions"
    plt.suptitle(title, fontsize=22, y=0.98)
    plt.tight_layout(rect=(0, 0.05, 1, 0.95))
    plt.savefig(f"fig4_{geometry}_all_metrics_evolution.png" if geometry == "smooth" else f"fig5_{geometry}_all_metrics_evolution.png", dpi=400, bbox_inches="tight")
    plt.close()


def plot_geometry_vs_cfl(combo_df, geometry, formatter):
    fig, axes = plt.subplots(3, 3, figsize=(18, 14), sharex=True)
    iter20_df = combo_df[
        (combo_df["step_or_iter"] == 20)
        & combo_df["group_id"].isin(GEOMETRY_GROUPS[geometry])
        & combo_df["Display_Name"].str.startswith("CFL=", na=False)
    ].copy()

    for r, (rho, group_id) in enumerate(zip(RHO_VALUES, GEOMETRY_GROUPS[geometry], strict=True)):
        rho_df = iter20_df[iter20_df["group_id"] == group_id].sort_values("CFL_Val")
        for c, metric in enumerate(METRICS):
            ax = axes[r, c]
            has_data = not rho_df.empty
            if has_data:
                ax.plot(
                    rho_df["CFL_Val"],
                    rho_df[metric],
                    marker="o",
                    linestyle="-",
                    linewidth=2.4,
                    markersize=7,
                    color=METHOD_COLORS["CFL=0.2"] if geometry == "smooth" else METHOD_COLORS["CFL=0.4"],
                )
            setup_axis(
                ax,
                formatter,
                metric_title=METRIC_TITLES[c] if r == 0 else None,
                row_title=f"rho={rho}" if c == 0 else None,
                xlabel="Training CFL" if r == 2 else None,
            )
            ax.set_xticks(CFL_VALUES)
            ensure_axis_has_data(ax, has_data)

    title = "Smooth Accuracy vs. Training CFL (Iter=20)" if geometry == "smooth" else "Acute Accuracy vs. Training CFL (Iter=20)"
    plt.suptitle(title, fontsize=22, y=0.98)
    plt.tight_layout(rect=(0, 0.02, 1, 0.95))
    plt.savefig(f"fig6_{geometry}_all_metrics_vs_cfl.png" if geometry == "smooth" else f"fig7_{geometry}_all_metrics_vs_cfl.png", dpi=400, bbox_inches="tight")
    plt.close()


def best_cfl_label_for_group(iter20_df, group_id):
    group_df = iter20_df[
        (iter20_df["group_id"] == group_id)
        & iter20_df["Display_Name"].str.startswith("CFL=", na=False)
    ]
    if group_df.empty:
        return None
    return group_df.sort_values("MSE_hk").iloc[0]["Display_Name"]


def plot_geometry_baseline_comparison(combo_df, geometry, formatter):
    fig, axes = plt.subplots(3, 3, figsize=(18, 14), sharex=False)
    iter20_df = combo_df[
        (combo_df["step_or_iter"] == 20)
        & combo_df["group_id"].isin(GEOMETRY_GROUPS[geometry])
    ].copy()

    for r, (rho, group_id) in enumerate(zip(RHO_VALUES, GEOMETRY_GROUPS[geometry], strict=True)):
        best_cfl = best_cfl_label_for_group(iter20_df, group_id)
        methods = ["Numerical", "Origin"] + ([best_cfl] if best_cfl else [])
        for c, metric in enumerate(METRICS):
            ax = axes[r, c]
            plot_df = iter20_df[
                (iter20_df["group_id"] == group_id)
                & (iter20_df["Display_Name"].isin(methods))
            ].copy()
            has_data = not plot_df.empty
            if has_data:
                plot_df["Display_Name"] = pd.Categorical(plot_df["Display_Name"], categories=methods, ordered=True)
                plot_df = plot_df.sort_values("Display_Name")
                x = np.arange(len(plot_df))
                colors = [METHOD_COLORS.get(name, "#444444") for name in plot_df["Display_Name"]]
                ax.bar(x, plot_df[metric], color=colors, width=0.7)
                ax.set_xticks(x)
                ax.set_xticklabels(plot_df["Display_Name"], rotation=15)
            setup_axis(
                ax,
                formatter,
                metric_title=METRIC_TITLES[c] if r == 0 else None,
                row_title=f"rho={rho}" if c == 0 else None,
            )
            ensure_axis_has_data(ax, has_data)

    title = "Smooth Baseline Check by Resolution (Iter=20)" if geometry == "smooth" else "Acute Baseline Check by Resolution (Iter=20)"
    plt.suptitle(title, fontsize=22, y=0.98)
    plt.tight_layout(rect=(0, 0.02, 1, 0.95))
    plt.savefig(f"fig8_{geometry}_all_metrics_comparison.png" if geometry == "smooth" else f"fig9_{geometry}_all_metrics_comparison.png", dpi=400, bbox_inches="tight")
    plt.close()


def plot_vs_resolution(combo_df, formatter):
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharex=True)
    iter20_df = combo_df[combo_df["step_or_iter"] == 20].copy()
    ordered_names = ordered_display_names(combo_df)

    for r, geometry in enumerate(["smooth", "acute"]):
        geometry_df = iter20_df[iter20_df["group_id"].isin(GEOMETRY_GROUPS[geometry])].copy()
        group_to_rho = dict(zip(GEOMETRY_GROUPS[geometry], RHO_VALUES))
        geometry_df["plot_rho"] = geometry_df["group_id"].map(group_to_rho)

        for c, metric in enumerate(METRICS):
            ax = axes[r, c]
            has_data = False
            for name in ordered_names:
                series_df = geometry_df[geometry_df["Display_Name"] == name].sort_values("plot_rho")
                if series_df.empty:
                    continue
                has_data = True
                ax.plot(
                    series_df["plot_rho"],
                    series_df[metric],
                    marker=METHOD_MARKERS.get(name, "o"),
                    linestyle=METHOD_LINESTYLES.get(name, "-"),
                    linewidth=2.2,
                    markersize=7,
                    color=METHOD_COLORS.get(name, "#444444"),
                    label=name,
                )
            setup_axis(
                ax,
                formatter,
                metric_title=METRIC_TITLES[c] if r == 0 else None,
                row_title=geometry.capitalize() if c == 0 else None,
                xlabel="Model Resolution (rho)" if r == 1 else None,
            )
            ax.set_xticks(RHO_VALUES)
            ensure_axis_has_data(ax, has_data)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="lower center",
            ncol=min(6, len(labels)),
            bbox_to_anchor=(0.5, -0.02),
            title="Method / Training CFL",
            fontsize=12,
            title_fontsize=13,
        )
    plt.suptitle("Accuracy vs. Model Resolution (Iter=20)", fontsize=20, y=0.98)
    plt.tight_layout(rect=(0, 0.06, 1, 0.94))
    plt.savefig("fig10_all_metrics_vs_resolution.png", dpi=400, bbox_inches="tight")
    plt.close()


def generate_test_plots():
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 12,
            "axes.grid": True,
            "grid.alpha": 0.35,
            "grid.linestyle": "--",
        }
    )
    formatter = ticker.FuncFormatter(scientific_formatter)
    combo_df = load_combined_evaluations(Path("model"))
    if combo_df.empty:
        print("No evaluation files found!")
        return

    for geometry in ["smooth", "acute"]:
        plot_geometry_evolution(combo_df, geometry, formatter)
        plot_geometry_vs_cfl(combo_df, geometry, formatter)
        plot_geometry_baseline_comparison(combo_df, geometry, formatter)
    plot_vs_resolution(combo_df, formatter)
    print("Success! Generated geometry-first publication-style plots.")


if __name__ == "__main__":
    generate_test_plots()
