"""
Supplementary Fig. 1  Study coverage ledger.
Compact matrix: studies (rows) x model families (columns).
Cells: completed (blue tint), capability-only (orange tint), empty (neutral).
Letter inside each cell.  183 mm wide, NC publication style.
"""
from __future__ import annotations
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ncstyle

ROOT = Path(__file__).resolve().parents[2]  # repository root; frozen runs/ must be restored alongside
FIGDIR = Path(__file__).resolve().parent.parent / "figures"

# Frozen coverage ledger (same data as plot_coverage.py)
COVERAGE_CSV = ROOT / "paper/generated/coverage-20260911-01/coverage.csv"


def _family(method: str) -> str:
    if method.startswith("native") or method == "VLanchor" or method.startswith("qwen35-"):
        return "Qwen3.5-4B"
    if method.startswith("qwen3vl-"):
        return "Qwen3-VL-4B"
    if method.startswith("e5-"):
        return "E5"
    if method.startswith("qwen-embedding"):
        return "Qwen embedding"
    if method.startswith("qwen-reranker"):
        return "Qwen reranker"
    if method.startswith("bert-"):
        return "BERT"
    if method == "roberta-base":
        return "RoBERTa"
    return "Other FMAT"


def main():
    ncstyle.apply_style()
    data = pd.read_csv(COVERAGE_CSV)

    families = [
        "Qwen3.5-4B", "Qwen3-VL-4B", "E5",
        "Qwen embedding", "Qwen reranker",
        "BERT", "RoBERTa", "Other FMAT",
    ]
    study_order = [
        "FMAT d1a", "FMAT d1b", "ValueEval", "EmoBank",
        "Chinese EmoBank", "OASIS direct",
        "VIVA", "DWUG",
        "Video demo 8 frames", "Video demo 16 frames",
    ]
    studies = [s for s in study_order if s in data.study.values]

    matrix = np.zeros((len(studies), len(families)), dtype=int)
    row_labels = []
    for i, study in enumerate(studies):
        rows = data[data.study == study]
        row_labels.append(f"{study}")
        for row in rows.itertuples():
            j = families.index(_family(row.method))
            matrix[i, j] = 2 if "capability" in row.status else 1

    cmap = ListedColormap([ncstyle.COV_EMPTY,
                           ncstyle.COV_BLUE_TINT,
                           ncstyle.COV_ORANGE_TINT])

    fig_w = ncstyle.DOUBLE_COL_INCH
    fig_h = fig_w * 0.42
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    ax.imshow(matrix, cmap=cmap, vmin=0, vmax=2, aspect="auto")

    # Cell text
    for i in range(len(studies)):
        for j in range(len(families)):
            if matrix[i, j] == 1:
                ax.text(j, i, "C", ha="center", va="center",
                        color=ncstyle.BLUE, fontsize=ncstyle.SZ_NOTE,
                        fontweight="bold")
            elif matrix[i, j] == 2:
                ax.text(j, i, "D", ha="center", va="center",
                        color=ncstyle.ORANGE, fontsize=ncstyle.SZ_NOTE,
                        fontweight="bold")

    # Grid lines (white separator)
    ax.set_xticks(np.arange(-0.5, len(families), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(studies), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    ax.tick_params(which="both", length=0)

    ax.set_xticks(range(len(families)))
    ax.set_xticklabels(families, rotation=40, ha="right",
                       fontsize=ncstyle.SZ_ANNOT)
    ax.set_yticks(range(len(studies)))
    ax.set_yticklabels(row_labels, fontsize=ncstyle.SZ_ANNOT)

    # Restore all spines for matrix border
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(ncstyle.SPINE_W)
        spine.set_color(ncstyle.SPINE_CLR)

    # Legend at bottom
    handles = [
        Patch(facecolor=ncstyle.COV_BLUE_TINT,
              edgecolor=ncstyle.SPINE_CLR, linewidth=0.3,
              label="C: completed evaluation"),
        Patch(facecolor=ncstyle.COV_ORANGE_TINT,
              edgecolor=ncstyle.SPINE_CLR, linewidth=0.3,
              label="D: capability demonstration"),
        Patch(facecolor=ncstyle.COV_EMPTY,
              edgecolor=ncstyle.SPINE_CLR, linewidth=0.3,
              label="Not applicable"),
    ]
    fig.legend(handles=handles, ncol=3, loc="lower center",
               bbox_to_anchor=(0.5, 0.0),
               fontsize=ncstyle.SZ_ANNOT, frameon=False)

    fig.subplots_adjust(left=0.18, right=0.97, top=0.97, bottom=0.30)
    ncstyle.save(fig, "sfig1-coverage", FIGDIR, tight=False)
    print("Supplementary Fig. 1 saved.")


if __name__ == "__main__":
    main()
