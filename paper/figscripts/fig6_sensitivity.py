"""Fig. 6  Calibration and sensitivity (r3 revision). Intervals parsed from the frozen
evidence tables e3-emobank-sensitivity.md and e3-dwug-sensitivity.md (raw_logp rows)."""
from __future__ import annotations
import re, sys
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np, pandas as pd
from scipy.spatial.distance import pdist
sys.path.insert(0, str(Path(__file__).resolve().parent))
import ncstyle

ROOT = Path(__file__).resolve().parents[2]  # repository root; frozen runs/ must be restored alongside
FIGDIR = Path(__file__).resolve().parent.parent / "figures"
CALIB_DIR = ROOT / "runs/e3-emobank-final-analysis-47430"
BRIDGE_DIR = ROOT / "runs/final-results-update-20260911-01/runs/analysis-46395/analysis"
EN_MD = ROOT / "paper/v2/evidence/e3-emobank-sensitivity.md"
DWUG_MD = ROOT / "paper/v2/evidence/e3-dwug-sensitivity.md"
COMPONENTS = [("Second model checkpoint", "second-model"), ("Terminated event", "terminated"), ("Anchor aliases", "aliases"),
              ("Phrase anchors", "phrases"), ("Augmented anchor bank", "augmented"), ("General 256 anchors", "general256"),
              ("Derived 64 anchors", "general64-derived")]


def parse(md):
    out = {}
    for m in re.finditer(r"\| ([\w-]+) \| raw_logp \| ([\d.]+) \| \[([\d.-]+), ([\d.-]+)\] \|", md.read_text()):
        out[m.group(1)] = tuple(float(m.group(k)) for k in (2, 3, 4))
    return out


def _q(v): return f"{v:.3f}".lstrip("0")


def main():
    ncstyle.apply_style()
    fig_w = ncstyle.DOUBLE_COL_INCH
    fig, axes = plt.subplots(1, 3, figsize=(fig_w, fig_w * 0.44), gridspec_kw={"width_ratios": [0.85, 1.25, 1.05]}, layout="constrained")
    fig.get_layout_engine().set(w_pad=0.10, wspace=0.05)

    ax = axes[0]; ncstyle.panel_label(ax, "a", y=1.18)
    arrays = {v: np.load(CALIB_DIR / f"base-{v}.npz") for v in ["raw_logp", "reference_log_ratio"]}
    dist = {v: pdist(d["cube"].mean(axis=2)) for v, d in arrays.items()}
    err = float(np.max(np.abs(dist["raw_logp"] - dist["reference_log_ratio"]))); print(f"  a: max plotted error {err:.3e}")
    ax.scatter(dist["raw_logp"], dist["reference_log_ratio"], s=1.5, alpha=0.25, color=ncstyle.BLUE, edgecolors="none", rasterized=True)
    mx = max(dist["raw_logp"]); ax.plot([0, mx], [0, mx], color=ncstyle.SPINE_CLR, lw=ncstyle.SPINE_W)
    ax.set(xlabel="Raw Euclidean distance", ylabel="Fixed-reference log-ratio distance", xticks=[0, 5, 10, 15], yticks=[0, 5, 10, 15])
    ax.set_title("Fixed-reference log ratio:\ndistances preserved", fontsize=ncstyle.SZ_TITLE, loc="left")
    ax.text(0.05, 0.92, f"Max error: {err:.1e}", transform=ax.transAxes, fontsize=ncstyle.SZ_ANNOT, color=ncstyle.INK2)
    ax.text(0.05, 0.84, "$n$ = 128 materials, 8,128 pairs", transform=ax.transAxes, fontsize=ncstyle.SZ_NOTE, color=ncstyle.MUTED)

    ax = axes[1]; ncstyle.panel_label(ax, "b", y=1.18)
    en, dw = parse(EN_MD), parse(DWUG_MD)
    for y, (label, key) in enumerate(COMPONENTS):
        for tab, colour, marker, off, name in [(en, ncstyle.BLUE, "o", -0.16, "English affect"), (dw, ncstyle.ORANGE, "s", 0.16, "Word usage")]:
            if key in tab:
                v, lo, hi = tab[key]
                ncstyle.dot_interval(ax, v, y + off, lo, hi, color=colour, marker=marker)
                print(f"  b: {label} {name} {v} [{lo}, {hi}]")
            else:
                ax.text(1.04, y + off, "not evaluated", va="center", ha="left", fontsize=5.5, color=colour, style="italic")
    ax.set_yticks(range(len(COMPONENTS))); ax.set_yticklabels([c[0] for c in COMPONENTS], fontsize=ncstyle.SZ_NOTE)
    ax.set(xlabel="Distance-rank agreement with the base instrument", xlim=(-0.08, 1.30), xticks=[0, 0.25, 0.5, 0.75, 1.0])
    ax.set_title("Checkpoint changes alter geometry;\nanchor-bank changes leave it close", fontsize=ncstyle.SZ_TITLE, loc="left")
    ax.invert_yaxis(); ncstyle.xgrid(ax); ax.axvline(0, color=ncstyle.GRID, lw=ncstyle.SPINE_W, zorder=0)
    handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor=ncstyle.BLUE, markeredgecolor="white", markersize=ncstyle.MS * 0.72, label="English affect, 128 materials"),
               Line2D([0], [0], marker="s", color="w", markerfacecolor=ncstyle.ORANGE, markeredgecolor="white", markersize=ncstyle.MS * 0.72, label="Word usage, 128 materials")]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, fontsize=ncstyle.SZ_NOTE, handletextpad=0.3, frameon=False)

    ax = axes[2]; ncstyle.panel_label(ax, "c", x=-0.42, y=1.18)
    bridge = pd.read_csv(BRIDGE_DIR / "paired-differences.csv"); ref_z = bridge[bridge.variant == "reference_z"].set_index("method")
    names = ["reference_guided", "validity", "reliability", "alpha", "random"]
    labels = ["Reference-guided", "Validity only", "Reliability only", "Paraphrase consistency", "Random"]
    rows = ref_z.loc[names]; qx = float(rows.ci_upper.max()) + 0.004
    for i, (name, r) in enumerate(rows.iterrows()):
        ncstyle.dot_interval(ax, r.difference, i, r.ci_lower, r.ci_upper, color=ncstyle.BLUE, marker="o")
        ax.text(qx, i, f"$q$ = {_q(r.bh_q)}", va="center", fontsize=ncstyle.SZ_ANNOT, color=ncstyle.INK2)
    ncstyle.forest_zero(ax); ax.set_yticks(range(5)); ax.set_yticklabels(labels, fontsize=ncstyle.SZ_NOTE)
    ax.set(xlabel="Macro AP difference", xlim=(-0.022, 0.030), xticks=[-0.02, 0, 0.02], ylim=(-0.7, 4.8))
    ax.set_title("Wording search:\nintervals include zero", fontsize=ncstyle.SZ_TITLE, loc="left"); ax.invert_yaxis()
    ax.text(0.02, 0.03, "$n$ = 1,576 arguments", transform=ax.transAxes, fontsize=ncstyle.SZ_NOTE, color=ncstyle.MUTED)
    ncstyle.save(fig, "fig6-sensitivity", FIGDIR); print("Fig. 6 saved.")


if __name__ == "__main__":
    main()
