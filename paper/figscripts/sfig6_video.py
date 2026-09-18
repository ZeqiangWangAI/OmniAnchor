"""
Supplementary Fig. 6  Video scoring time and GPU memory at 8 and 16 frames.
a: Measurement time per mixed batch for three scoring paths.
b: Peak allocated GPU memory.
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

# Frozen video cost table (same data as plot_video_costs.py)
VIDEO_CSV = ROOT / "paper/generated/video-capability-20260911-01/costs.csv"


def main():
    ncstyle.apply_style()
    data = pd.read_csv(VIDEO_CSV)

    methods = ["native", "qwen-embedding", "qwen-reranker"]
    labels = ["OmniAnchor (native)", "Embedding head", "Reranker head"]
    colours = [ncstyle.BLUE, ncstyle.GREY, ncstyle.ORANGE]
    markers = ["o", "^", "s"]

    fig_w = ncstyle.DOUBLE_COL_INCH
    fig_h = fig_w * 0.42
    fig, axes = plt.subplots(1, 2, figsize=(fig_w, fig_h),
                             layout="constrained")
    fig.get_layout_engine().set(rect=(0, 0.07, 1, 0.90))

    for method, label, colour, marker in zip(
        methods, labels, colours, markers
    ):
        rows = data[data.method == method].sort_values("frames")
        # Panel a: time in minutes
        axes[0].plot(rows.frames, rows.measurement_seconds / 60,
                     marker=marker, color=colour,
                     markersize=ncstyle.MS * 0.7,
                     markeredgecolor="white", markeredgewidth=ncstyle.MW,
                     linewidth=ncstyle.LW, label=label)
        # Panel b: memory in GiB
        axes[1].plot(rows.frames, rows.peak_allocated_bytes / 2**30,
                     marker=marker, color=colour,
                     markersize=ncstyle.MS * 0.7,
                     markeredgecolor="white", markeredgewidth=ncstyle.MW,
                     linewidth=ncstyle.LW, label=label)

    # Panel a formatting
    ax = axes[0]
    ncstyle.panel_label(ax, "a")
    ax.set(xlabel="Requested frames per video",
           ylabel="Minutes per mixed batch",
           xticks=[8, 16], xlim=(6, 18),
           title="Measurement time")
    ax.set_ylim(bottom=0)
    ncstyle.ygrid(ax)

    # Panel b formatting
    ax = axes[1]
    ncstyle.panel_label(ax, "b")
    ax.set(xlabel="Requested frames per video",
           ylabel="Peak allocated (GiB)",
           xticks=[8, 16], xlim=(6, 18),
           title="GPU memory")
    ax.set_ylim(bottom=0)
    ncstyle.ygrid(ax)

    # Shared legend
    handles = [
        Line2D([0], [0], marker=m, color=c,
               markeredgecolor="white", markeredgewidth=ncstyle.MW * 0.5,
               markersize=ncstyle.MS * 0.72,
               label=l, linewidth=ncstyle.LW)
        for m, c, l in zip(markers, colours, labels)
    ]
    fig.legend(handles=handles, ncol=3, loc="lower center",
               bbox_to_anchor=(0.5, 0.01),
               fontsize=ncstyle.SZ_ANNOT, frameon=False)

    ncstyle.save(fig, "sfig6-video", FIGDIR)
    print("Supplementary Fig. 6 saved.")


if __name__ == "__main__":
    main()
