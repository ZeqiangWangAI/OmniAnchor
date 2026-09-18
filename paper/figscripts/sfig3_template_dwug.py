"""
Supplementary Fig. 3  Bridge-versus-averaged partition stability on DWUG.
a: ARI between per-bridge and averaged k-means partitions, plotted vs k.
b: Template radius per material.
Uses the frozen base-raw_logp cube (general-128, 3 fixed bridges, dev set).
183 mm wide, NC publication style.
"""
from __future__ import annotations
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ncstyle

ROOT = Path(__file__).resolve().parents[2]  # repository root; frozen runs/ must be restored alongside
FIGDIR = Path(__file__).resolve().parent.parent / "figures"

# Frozen cube (same data path as the DWUG sensitivity analysis)
CUBE_PATH = ROOT / "runs/dwug-sensitivity-analysis-47431/runs/analysis-47431/analysis/base-raw_logp.npz"


def main():
    ncstyle.apply_style()

    # Load frozen cube: (n_materials, n_anchors, n_bridges)
    npz = np.load(CUBE_PATH)
    cube = npz["cube"]  # shape: (128, 128, 3)
    n_materials, n_anchors, n_bridges = cube.shape

    # Bridge-averaged coordinates
    avg = cube.mean(axis=2)  # (128, 128)

    # ── Panel a: ARI vs k ─────────────────────────────────────────────
    ks = np.arange(2, 11)
    bridge_colours = [ncstyle.BLUE, ncstyle.ORANGE, ncstyle.TEAL]
    bridge_markers = ["o", "s", "D"]
    bridge_labels = ["Bridge 1", "Bridge 2", "Bridge 3"]

    ari_matrix = np.zeros((len(ks), n_bridges))
    for ki, k in enumerate(ks):
        km_avg = KMeans(n_clusters=k, random_state=42, n_init=10).fit(avg)
        for b in range(n_bridges):
            km_b = KMeans(n_clusters=k, random_state=42, n_init=10).fit(
                cube[:, :, b])
            ari_matrix[ki, b] = adjusted_rand_score(
                km_avg.labels_, km_b.labels_)

    # ── Panel b: template radius per material ─────────────────────────
    radii = np.zeros(n_materials)
    for i in range(n_materials):
        max_r = 0.0
        for b in range(n_bridges):
            diff = cube[i, :, b] - avg[i, :]
            dist = np.sqrt(np.sum(diff ** 2))
            if dist > max_r:
                max_r = dist
        radii[i] = max_r

    # ── Plot ──────────────────────────────────────────────────────────
    fig_w = ncstyle.DOUBLE_COL_INCH
    fig_h = fig_w * 0.38
    fig, axes = plt.subplots(1, 2, figsize=(fig_w, fig_h),
                             layout="constrained")

    # Panel a
    ax = axes[0]
    ncstyle.panel_label(ax, "a")
    for b in range(n_bridges):
        ax.plot(ks, ari_matrix[:, b],
                marker=bridge_markers[b], color=bridge_colours[b],
                markersize=ncstyle.MS * 0.7,
                markeredgecolor="white", markeredgewidth=ncstyle.MW,
                linewidth=ncstyle.LW, label=bridge_labels[b])
    ax.set(xlabel="Number of clusters $k$",
           ylabel="Adjusted Rand index",
           title="Bridge vs averaged partition",
           xticks=ks, ylim=(-0.05, 1.05))
    ax.legend(fontsize=ncstyle.SZ_ANNOT, loc="lower left")
    ncstyle.ygrid(ax)

    # Panel b
    ax = axes[1]
    ncstyle.panel_label(ax, "b")
    sorted_radii = np.sort(radii)
    ax.bar(range(n_materials), sorted_radii, width=1.0,
           color=ncstyle.BLUE, alpha=0.7, edgecolor="none")
    ax.set(xlabel="Material (sorted)",
           ylabel="Template radius $r_x$",
           title="Per-material template radius")
    ncstyle.ygrid(ax)

    ncstyle.save(fig, "sfig3-template-dwug", FIGDIR)
    print("Supplementary Fig. 3 saved.")


if __name__ == "__main__":
    main()
