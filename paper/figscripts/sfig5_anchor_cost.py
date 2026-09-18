"""
Supplementary Fig. 5  Measurement time and GPU memory vs anchor count.
a: Wall time per material and timed forward-pass duration.
b: Peak allocated and reserved GPU memory.
183 mm wide, NC publication style.
"""
from __future__ import annotations
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ncstyle

ROOT = Path(__file__).resolve().parents[2]  # repository root; frozen runs/ must be restored alongside
FIGDIR = Path(__file__).resolve().parent.parent / "figures"

# Frozen anchor-efficiency cost table (same data as plot_anchor_efficiency.py)
COST_CSV = ROOT / "paper/generated/anchor-efficiency-20260911-01/costs.csv"


def main():
    ncstyle.apply_style()
    table = pd.read_csv(COST_CSV).sort_values("anchors")

    fig_w = ncstyle.DOUBLE_COL_INCH
    fig_h = fig_w * 0.35
    fig, axes = plt.subplots(1, 2, figsize=(fig_w, fig_h),
                             layout="constrained")

    # Panel a: time
    ax = axes[0]
    ncstyle.panel_label(ax, "a")
    ax.plot(table.anchors, table.measurement_seconds / 32,
            "-o", color=ncstyle.BLUE,
            markersize=ncstyle.MS * 0.7,
            markeredgecolor="white", markeredgewidth=ncstyle.MW,
            linewidth=ncstyle.LW,
            label="Measurement wall time")
    ax.plot(table.anchors, table.forward_seconds / 32,
            "-s", color=ncstyle.ORANGE,
            markersize=ncstyle.MS * 0.6,
            markeredgecolor="white", markeredgewidth=ncstyle.MW,
            linewidth=ncstyle.LW,
            label="Timed forwards")
    ax.set(xlabel="Anchors", ylabel="Seconds per material",
           xticks=[16, 64, 128, 256],
           title="Measurement time")
    ax.legend(fontsize=ncstyle.SZ_ANNOT, frameon=False, loc="upper left")
    ncstyle.ygrid(ax)

    # Panel b: memory
    ax = axes[1]
    ncstyle.panel_label(ax, "b")
    ax.plot(table.anchors, table.peak_allocated_bytes / 2**30,
            "-o", color=ncstyle.BLUE,
            markersize=ncstyle.MS * 0.7,
            markeredgecolor="white", markeredgewidth=ncstyle.MW,
            linewidth=ncstyle.LW,
            label="Peak allocated")
    ax.plot(table.anchors, table.peak_reserved_bytes / 2**30,
            "-s", color=ncstyle.ORANGE,
            markersize=ncstyle.MS * 0.6,
            markeredgecolor="white", markeredgewidth=ncstyle.MW,
            linewidth=ncstyle.LW,
            label="Peak reserved")
    ax.set(xlabel="Anchors", ylabel="Peak GPU memory (GiB)",
           xticks=[16, 64, 128, 256], ylim=(0, 10),
           title="GPU memory")
    ax.legend(fontsize=ncstyle.SZ_ANNOT, frameon=False, loc="lower right")
    ncstyle.ygrid(ax)

    ncstyle.save(fig, "sfig5-anchor-cost", FIGDIR)
    print("Supplementary Fig. 5 saved.")


if __name__ == "__main__":
    main()
