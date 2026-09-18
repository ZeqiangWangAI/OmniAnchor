"""Fig. 4  Images alone and with text (r3 revision)."""
from __future__ import annotations
import sys
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import ncstyle

ROOT = Path(__file__).resolve().parents[2]  # repository root; frozen runs/ must be restored alongside
FIGDIR = Path(__file__).resolve().parent.parent / "figures"
OASIS_DIR = ROOT / "runs/oasis-direct-final201-analysis-20260910-02"
VIVA_DIR = ROOT / "runs/final-results-update-20260911-01/runs/analysis-45872/analysis"


def _q(v): return f"{v:.3f}".lstrip("0")


def main():
    ncstyle.apply_style()
    corr = pd.read_csv(OASIS_DIR / "direct-correlations.csv"); opaired = pd.read_csv(OASIS_DIR / "paired-differences.csv")
    metrics = pd.read_csv(VIVA_DIR / "metrics.csv"); paired = pd.read_csv(VIVA_DIR / "paired-differences.csv")

    fig_w = ncstyle.DOUBLE_COL_INCH
    fig = plt.figure(figsize=(fig_w, fig_w * 0.74), layout="constrained")
    fig.get_layout_engine().set(h_pad=0.10, w_pad=0.08, hspace=0.08, wspace=0.06)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.0], height_ratios=[1.0, 0.9])
    ax_a = fig.add_subplot(gs[0, 0]); ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0]); ax_d = fig.add_subplot(gs[1, 1])

    # a  OASIS direct correlations
    ax = ax_a; ncstyle.panel_label(ax, "a", x=-0.22, y=1.12)
    methods = [("VLanchor", "OmniAnchor", ncstyle.BLUE, "o"), ("qwen-embedding", "Embedding head", ncstyle.GREY, "^"),
               ("qwen-reranker", "Reranker head", ncstyle.ORANGE, "s")]
    qlook = {(r.dimension, r.comparator): r.primary_family_bh_q for _, r in opaired[opaired.metric == "Pearson"].iterrows()}
    pearson = corr[corr.metric == "Pearson"]; mx = float(pearson.ci_upper.max()) + 0.04
    for j, dim in enumerate(["V", "A"]):
        for i, (m, label, colour, marker) in enumerate(methods):
            row = pearson[(pearson.method == m) & (pearson.dimension == dim)].iloc[0]; y = j * 4 + i
            ncstyle.dot_interval(ax, row.correlation, y, row.ci_lower, row.ci_upper, color=colour, marker=marker)
            txt = f"$r$ = .{int(round(row.correlation * 1000)):03d}"
            if (dim, m) in qlook: txt += f"; $q$ = {_q(qlook[(dim, m)])}"
            ax.text(mx, y, txt, ha="left", va="center", fontsize=ncstyle.SZ_NOTE, color=ncstyle.INK2)
    ax.set_yticks([0, 1, 2, 4, 5, 6]); ax.set_yticklabels([l for _, l, _, _ in methods] * 2, fontsize=ncstyle.SZ_NOTE)
    ax.axhline(3, color=ncstyle.GRID, linewidth=0.5, zorder=0)
    for y, lab in [(-0.6, "Valence"), (3.4, "Arousal")]:
        ax.text(0.02, y, lab, fontsize=ncstyle.SZ_ANNOT, fontweight="medium", transform=ax.get_yaxis_transform(), ha="left", va="center", color=ncstyle.INK2)
    ax.set(xlabel="Pearson $r$ with published norms", xlim=(0, 1.42), xticks=[0, 0.5, 1.0], ylim=(-1.0, 7.0))
    ax.set_title("Photographs: OmniAnchor above\nboth heads on valence", fontsize=ncstyle.SZ_TITLE, loc="left")
    ax.invert_yaxis(); ncstyle.xgrid(ax)
    ax.text(0.02, 0.02, "$n$ = 201; $q$: paired test vs OmniAnchor", transform=ax.transAxes, fontsize=ncstyle.SZ_NOTE, color=ncstyle.MUTED)

    # b  VIVA conditions
    ax = ax_b; ncstyle.panel_label(ax, "b", x=-0.20, y=1.12)
    conds = ["action_only", "image_action", "mismatched_image_action"]
    clabels = ["Text\nonly", "Correct\nimage + text", "Mismatched\nimage + text"]
    inst = [("native", "OmniAnchor", ncstyle.BLUE, "o"), ("qwen-reranker", "Reranker head", ncstyle.ORANGE, "s"), ("qwen-embedding", "Embedding head", ncstyle.GREY, "^")]
    for k, (m, label, colour, marker) in enumerate(inst):
        d = metrics[(metrics.method == m) & (metrics.metric == "AP")].set_index("condition").loc[conds]; off = (k - 1) * 0.12
        ax.plot(np.arange(3) + off, d["value"], "-", lw=ncstyle.LW * 0.7, color=colour, zorder=2)
        for xi, c in enumerate(conds):
            r = d.loc[c]; ncstyle.dot_interval_v(ax, xi + off, r["value"], r.ci_lower, r.ci_upper, color=colour, marker=marker, label=label if xi == 0 else None)
    ax.set_xticks(range(3)); ax.set_xticklabels(clabels, fontsize=ncstyle.SZ_NOTE)
    ax.set(ylabel="Mean per-item AP", ylim=(0.64, 0.92), xlim=(-0.5, 2.5))
    ax.set_title("Images alter scores without\nimproving this criterion", fontsize=ncstyle.SZ_TITLE, loc="left"); ncstyle.ygrid(ax)
    ax.legend(frameon=False, loc="center right", bbox_to_anchor=(1.02, 0.36), fontsize=ncstyle.SZ_NOTE, handletextpad=0.3, markerscale=0.8)
    ax.text(0.02, 0.02, "$n$ = 216 items, 215 groups;\naction text held fixed", transform=ax.transAxes, fontsize=ncstyle.SZ_NOTE, color=ncstyle.MUTED, linespacing=1.3)

    # c  input contrasts (OmniAnchor), d model contrasts at the correct image
    ap = paired[paired.metric == "AP"].reset_index(drop=True)
    inp = ap[ap.comparator == "native"]; mod = ap[ap.comparator != "native"]
    for ax, tab, labels, title, xlim, xticks, letter in [
        (ax_c, inp, ["Correct image + text\n$-$ text only", "Correct image + text\n$-$ mismatched image + text"],
         "Input contrasts, OmniAnchor", (-0.05, 0.062), [-0.04, -0.02, 0, 0.02, 0.04], "c"),
        (ax_d, mod, ["OmniAnchor $-$ embedding head", "OmniAnchor $-$ reranker head"],
         "Model contrasts, correct image + text", (-0.06, 0.26), [0, 0.1, 0.2], "d")]:
        ncstyle.panel_label(ax, letter, x=-0.40 if letter == "c" else -0.34, y=1.16)
        qx = xlim[1] - 0.004
        for i, (_, r) in enumerate(tab.iterrows()):
            ncstyle.dot_interval(ax, r.difference, i, r.ci_lower, r.ci_upper, color=ncstyle.BLUE, marker="o")
            ax.text(qx, i, f"$q$ = {_q(r.primary_family_bh_q)}", ha="right", va="center", fontsize=ncstyle.SZ_ANNOT, color=ncstyle.INK2)
            print(f"  {letter}: {labels[i]} {r.difference:.4f} [{r.ci_lower:.3f}, {r.ci_upper:.3f}] q={r.primary_family_bh_q:.3f}")
        ncstyle.forest_zero(ax); ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=ncstyle.SZ_NOTE)
        ax.set(xlabel="Paired AP difference", xlim=xlim, xticks=xticks, ylim=(-0.6, 1.6))
        ax.set_title(title, fontsize=ncstyle.SZ_TITLE, loc="left"); ax.invert_yaxis()
    ax_c.text(0.02, 0.04, "Axis is five times narrower than in d", transform=ax_c.transAxes, fontsize=ncstyle.SZ_NOTE, color=ncstyle.MUTED)
    ncstyle.save(fig, "fig4-images", FIGDIR); print("Fig. 4 saved.")


if __name__ == "__main__":
    main()
