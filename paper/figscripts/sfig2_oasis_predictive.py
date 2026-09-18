"""
Supplementary Fig. 2  OASIS affect-12 predictive correlations.
Two panels: valence (left) and arousal (right).
Horizontal dot plot with 95% bootstrap intervals.
183 mm wide, NC publication style.
"""
from __future__ import annotations
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ncstyle

ROOT = Path(__file__).resolve().parents[2]  # repository root; frozen runs/ must be restored alongside
FIGDIR = Path(__file__).resolve().parent.parent / "figures"

# Frozen OASIS predictive results (same data as plot_oasis_predictive.py)
OASIS_CSV = ROOT / "runs/scienceplots-oasis12-20260911-02/plotted-results.csv"


def main():
    ncstyle.apply_style()
    data = pd.read_csv(OASIS_CSV)

    methods = [
        "native-raw_logp",
        "native-reference_log_ratio",
        "native-reference_z",
        "qwen-embedding-anchor",
        "qwen-embedding-original",
        "qwen-reranker-anchor",
    ]
    labels = [
        "OmniAnchor: raw",
        "OmniAnchor: log ratio",
        "OmniAnchor: reference z",
        "Embedding: anchors",
        "Embedding: original",
        "Reranker: anchors",
    ]
    colours = ([ncstyle.BLUE] * 3 +
               [ncstyle.GREY] * 2 +
               [ncstyle.ORANGE])
    markers = (["o"] * 3 + ["^"] * 2 + ["s"])

    fig_w = ncstyle.DOUBLE_COL_INCH
    fig_h = fig_w * 0.42
    fig, axes = plt.subplots(1, 2, figsize=(fig_w, fig_h), sharey=True,
                             layout="constrained")
    fig.get_layout_engine().set(rect=(0, 0.07, 1, 0.90))

    for ax, metric, title, letter in zip(
        axes,
        ["spearman_V", "spearman_A"],
        ["Valence", "Arousal"],
        ["a", "b"],
    ):
        ncstyle.panel_label(ax, letter)
        subset = data[data.metric == metric].set_index("method")
        for y, (method, colour, marker) in enumerate(
                zip(methods, colours, markers)):
            row = subset.loc[method]
            ncstyle.dot_interval(
                ax, row.value, y, row.ci_lower, row.ci_upper,
                color=colour, marker=marker)
        ax.set_title(title, fontsize=ncstyle.SZ_TITLE, loc="left")
        ax.set(xlim=(0.25, 0.95), xlabel="Spearman $\\rho$",
               ylim=(5.6, -0.6))
        ax.set_xticks([0.3, 0.5, 0.7, 0.9])
        ncstyle.xgrid(ax)

    axes[0].set_yticks(range(6))
    axes[0].set_yticklabels(labels, fontsize=ncstyle.SZ_NOTE)

    # Legend
    handles = [
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor=ncstyle.BLUE, markeredgecolor="white",
               markeredgewidth=0.4, markersize=ncstyle.MS * 0.72,
               label="OmniAnchor"),
        Line2D([0], [0], marker="^", color="w",
               markerfacecolor=ncstyle.GREY, markeredgecolor="white",
               markeredgewidth=0.4, markersize=ncstyle.MS * 0.72,
               label="Embedding"),
        Line2D([0], [0], marker="s", color="w",
               markerfacecolor=ncstyle.ORANGE, markeredgecolor="white",
               markeredgewidth=0.4, markersize=ncstyle.MS * 0.72,
               label="Reranker"),
    ]
    fig.legend(handles=handles, ncol=3, loc="lower center",
               bbox_to_anchor=(0.5, 0.01),
               fontsize=ncstyle.SZ_ANNOT, frameon=False)

    ncstyle.save(fig, "sfig2-oasis-predictive", FIGDIR)
    print("Supplementary Fig. 2 saved.")


if __name__ == "__main__":
    main()
