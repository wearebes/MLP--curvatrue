"""
Publication-quality plotting script for PDE model comparison.
Produces 3 separate figures following top-journal (Nature/NeurIPS) standards.
"""

import os
import re
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

matplotlib.use("Agg")  # headless backend for WSL

# ─────────────────────────────────────────────────────────────────────────────
# 1.  GLOBAL STYLE  (journal-grade: clean, serif, tight ticks)
# ─────────────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    # --- font ---
    "font.family":        "serif",
    "font.serif":         ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset":   "stix",          # LaTeX-style math

    # --- sizes ---
    "font.size":          11,
    "axes.labelsize":     12,
    "axes.titlesize":     12,
    "legend.fontsize":    9.5,
    "xtick.labelsize":    10,
    "ytick.labelsize":    10,

    # --- axes ---
    "axes.linewidth":     1.2,
    "axes.spines.top":    True,
    "axes.spines.right":  True,

    # --- ticks ---
    "xtick.direction":    "in",
    "ytick.direction":    "in",
    "xtick.top":          True,
    "ytick.right":        True,
    "xtick.major.size":   5,
    "ytick.major.size":   5,
    "xtick.minor.size":   3,
    "ytick.minor.size":   3,
    "xtick.major.width":  1.0,
    "ytick.major.width":  1.0,

    # --- lines / markers ---
    "lines.linewidth":    1.6,
    "lines.markersize":   7,

    # --- grid ---
    "axes.grid":          False,   # no grid by default (cleaner)

    # --- legend ---
    "legend.frameon":     True,
    "legend.framealpha":  1.0,
    "legend.edgecolor":   "black",
    "legend.fancybox":    False,

    # --- figure ---
    "figure.dpi":         150,
    "savefig.dpi":        400,
    "savefig.bbox":       "tight",
    "figure.facecolor":   "white",
    "axes.facecolor":     "white",
})

# ─────────────────────────────────────────────────────────────────────────────
# 2.  COLOR / MARKER PALETTES  (colorblind-safe, journal-standard)
# ─────────────────────────────────────────────────────────────────────────────
# rho: 3 lines
RHO_COLORS  = {256: "#0072B2",   # deep blue
               266: "#D55E00",   # vermillion
               276: "#009E73"}   # green
RHO_MARKERS = {256: "o", 266: "s", 276: "^"}

# Batch size: solid vs dashed
BATCH_LS    = {32: "-",  256: "--"}
BATCH_COLOR = {32: "#E69F00",   # amber
               256: "#56B4E9"}  # sky blue

# Dataset scales → x-axis labels
SCALE_ORDER  = ["001", "002", "004", "005"]
SCALE_LABELS = ["CFL=0.1", "CFL=0.2", "CFL=0.4", "CFL=0.5"]
CFL_XLABEL   = "CFL Number"
SCALE_NUM    = {k: i for i, k in enumerate(SCALE_ORDER)}  # for numeric x-axis

# ─────────────────────────────────────────────────────────────────────────────
# 3.  DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────
def load_all_data(base_dir="model"):
    records = []
    pattern = re.compile(r"data02_(00[1245])_batch(\d+)$")
    for folder in sorted(os.listdir(base_dir)):
        m = pattern.match(folder)
        if not m:
            continue
        scale = m.group(1)
        batch = int(m.group(2))
        summary = os.path.join(base_dir, folder, "training_summary.txt")
        if not os.path.exists(summary):
            continue
        df = pd.read_csv(summary, sep=r"\s+")
        for _, row in df.iterrows():
            records.append({
                "scale":           scale,
                "batch":           batch,
                "rho":             int(row["rho"]),
                "best_epoch":      int(row["best_epoch"]),
                "test_mse":        float(row["test_mse"]),
                "test_mae":        float(row["test_mae"]),
                "test_max_ae":     float(row["test_max_ae"]),
                "elapsed_hours":  float(row["elapsed_seconds"]) / 3600.0,
                "scale_idx":       SCALE_NUM[scale],
            })
    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────────────────────
# 4.  FIGURE 1 — Test MSE vs. Dataset Scale, grouped by rho
#     (Batch 32 data used; MSE identical across batch sizes)
# ─────────────────────────────────────────────────────────────────────────────
# 4.  FIGURE 1 — Test MSE vs. CFL Number  [1×2: Batch 32 | Batch 256]
# ─────────────────────────────────────────────────────────────────────────────
def plot_fig1_mse_vs_scale(df, outdir):
    def full_sci_label(val, pos):
        if val == 0:
            return "0"
        exp = int(np.floor(np.log10(abs(val))))
        coef = val / 10**exp
        return rf"${coef:.2f} \times 10^{{{exp}}}$"

    xs = np.array(range(len(SCALE_ORDER)))

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.2), sharey=True)
    subtitles = {32: "(a) Batch Size = 32", 256: "(b) Batch Size = 256"}

    for col, batch in enumerate([32, 256]):
        ax = axes[col]
        dfsub = df[df["batch"] == batch].copy()

        for rho in [256, 266, 276]:
            sub = dfsub[dfsub["rho"] == rho].sort_values("scale_idx")
            ax.plot(sub["scale_idx"].values, sub["test_mse"].values,
                    color=RHO_COLORS[rho],
                    marker=RHO_MARKERS[rho],
                    linestyle="-",
                    label=r"$\rho = $" + str(rho))

        ax.set_xticks(xs)
        ax.set_xticklabels(SCALE_LABELS)
        ax.set_xlabel(CFL_XLABEL)
        ax.set_title(subtitles[batch], fontweight="bold")

        ax.yaxis.set_major_formatter(ticker.FuncFormatter(full_sci_label))
        ax.yaxis.get_offset_text().set_visible(False)
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
        ax.tick_params(which="minor", length=3, width=0.8)

        # y-label only on leftmost panel
        if col == 0:
            ax.set_ylabel(r"Test MSE")
        # legend only on rightmost panel
        if col == 1:
            ax.legend(loc="best", title=r"Resolution $(\rho)$",
                      title_fontsize=9.5, handlelength=2)

    fig.suptitle("Test MSE vs. CFL Number", fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    out = os.path.join(outdir, "fig1_mse_vs_scale.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved → {out}")



# ─────────────────────────────────────────────────────────────────────────────
# 5.  FIGURE 2 — Test MAE vs. rho  [1×2: Batch 32 | Batch 256]
# ─────────────────────────────────────────────────────────────────────────────
def plot_fig2_mae_vs_rho(df, outdir):
    def full_sci_label_mae(val, pos):
        if val == 0:
            return "0"
        exp = int(np.floor(np.log10(abs(val))))
        coef = val / 10**exp
        return rf"${coef:.2f} \times 10^{{{exp}}}$"

    scale_colors  = {"001": "#0072B2", "002": "#D55E00",
                     "004": "#009E73", "005": "#CC79A7"}
    scale_markers = {"001": "o", "002": "s", "004": "^", "005": "D"}
    CFL_LABELS    = {"001": "CFL=0.1", "002": "CFL=0.2",
                     "004": "CFL=0.4", "005": "CFL=0.5"}
    rho_values    = [256, 266, 276]

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.2), sharey=True)
    subtitles = {32: "(a) Batch Size = 32", 256: "(b) Batch Size = 256"}

    for col, batch in enumerate([32, 256]):
        ax = axes[col]
        dfsub = df[df["batch"] == batch].copy()

        for scale in SCALE_ORDER:
            sub = dfsub[dfsub["scale"] == scale].sort_values("rho")
            ax.plot(sub["rho"].values, sub["test_mae"].values,
                    color=scale_colors[scale],
                    marker=scale_markers[scale],
                    linestyle="-",
                    label=CFL_LABELS[scale])

        ax.set_xticks(rho_values)
        ax.set_xlabel(r"Mesh Resolution $\rho$")
        ax.set_title(subtitles[batch], fontweight="bold")

        ax.yaxis.set_major_formatter(ticker.FuncFormatter(full_sci_label_mae))
        ax.yaxis.get_offset_text().set_visible(False)
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
        ax.tick_params(which="minor", length=3, width=0.8)

        if col == 0:
            ax.set_ylabel(r"Test MAE")
        if col == 1:
            ax.legend(loc="best", title="CFL Number",
                      title_fontsize=9.5, handlelength=2)

    fig.suptitle("Test MAE vs. Mesh Resolution", fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    out = os.path.join(outdir, "fig2_mae_vs_rho.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved → {out}")



# ─────────────────────────────────────────────────────────────────────────────
# 6.  FIGURE 3 — Training Time: Batch 32 vs. 256, per rho
# ─────────────────────────────────────────────────────────────────────────────
def plot_fig3_time_batch(df, outdir):
    fig, ax = plt.subplots(figsize=(5.8, 4.0))

    xs = np.array(range(len(SCALE_ORDER)))
    offsets = {32: -0.04, 256: 0.04}  # tiny jitter so overlapping markers are visible

    for batch in [32, 256]:
        for rho in [256, 266, 276]:
            sub = df[(df["batch"] == batch) & (df["rho"] == rho)].sort_values("scale_idx")
            jitter = offsets[batch]
            ax.plot(sub["scale_idx"].values + jitter, sub["elapsed_hours"].values,
                    color=RHO_COLORS[rho],
                    marker=RHO_MARKERS[rho],
                    linestyle=BATCH_LS[batch],
                    label=f"Batch {batch}, " + r"$\rho=$" + str(rho))

    ax.set_xticks(xs)
    ax.set_xticklabels(SCALE_LABELS)
    ax.set_xlabel(CFL_XLABEL)
    ax.set_ylabel("Training Time (h)")
    ax.set_title("(c) Training Time: Batch 32 vs. Batch 256")

    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    ax.tick_params(which="minor", length=3, width=0.8)

    # Two-column legend: left col = batch, right col = rho legend
    # Build it manually with proxy artists
    from matplotlib.lines import Line2D
    batch_proxies = [
        Line2D([0], [0], color="gray", ls="-",  lw=1.6, label="Batch 32"),
        Line2D([0], [0], color="gray", ls="--", lw=1.6, label="Batch 256"),
    ]
    rho_proxies = [
        Line2D([0], [0], color=RHO_COLORS[r], marker=RHO_MARKERS[r],
               ls="none", ms=6, label=r"$\rho=$" + str(r))
        for r in [256, 266, 276]
    ]
    all_proxies = batch_proxies + rho_proxies
    ax.legend(handles=all_proxies, loc="upper right",
              ncol=2, handlelength=2, title_fontsize=9.5,
              columnspacing=1.0, handletextpad=0.5)

    fig.tight_layout()
    out = os.path.join(outdir, "fig3_training_time.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# 7.  MAIN
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    base = os.path.dirname(os.path.abspath(__file__))
    out_dir = base   # save in project root

    print("Loading data …")
    df = load_all_data(os.path.join(base, "model"))
    print(f"  {len(df)} rows loaded.\n")
    print(df.to_string(index=False))
    print()

    print("Generating figures …")
    plot_fig1_mse_vs_scale(df, out_dir)
    plot_fig2_mae_vs_rho(df, out_dir)
    plot_fig3_time_batch(df, out_dir)
    print("\nDone.")
