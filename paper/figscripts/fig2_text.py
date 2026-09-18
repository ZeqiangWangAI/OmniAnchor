"""Fig. 2  Text validity (r3 revision). Data loaded verbatim from frozen runs;
the no-content gains come from the SI no-content table (tab:controls)."""
from __future__ import annotations
import json, re, sys
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import ncstyle

ROOT = Path(__file__).resolve().parents[2]  # repository root; frozen runs/ must be restored alongside
FIGDIR = Path(__file__).resolve().parent.parent / "figures"
SI_TEX = Path(__file__).resolve().parent.parent / "si.tex"
FMAT_DIR = ROOT / "runs/fmat-matched-validity-45045-01"
TEXT_CSV = ROOT / "runs/scienceplots-text-final-20260911-01/plotted-results.csv"
DWUG_DIR = ROOT / "runs/final-results-update-20260911-01/runs/analysis-46173/analysis"

PRIMARY = "qwen35-association"
GEN_LABEL = {"qwen35-association": "OmniAnchor (Qwen3.5-4B, association)",
             "qwen35-contextual_gap": "Qwen3.5-4B, contextual gap",
             "qwen35-full-sentence-PLC": "Qwen3.5-4B, full-sentence PLC",
             "qwen3vl-association": "Qwen3-VL-4B, association",
             "qwen3vl-contextual_gap": "Qwen3-VL-4B, contextual gap"}
INST = [("native-raw_logp", "OmniAnchor"), ("e5-anchor", "E5"),
        ("qwen-embedding-anchor", "Embedding"), ("qwen-reranker-anchor", "Reranker")]


def _q(v): return f"{v:.4f}".rstrip("0").lstrip("0") if v < 0.01 else f"{v:.3f}".lstrip("0")


def no_content_gains():
    """Direct-kind ValueEval rows of the SI no-content table: gain, CI, BH q."""
    t = SI_TEX.read_text()
    block = t[t.index("label{tab:controls}"):t.index("end{longtable}", t.index("label{tab:controls}"))]
    rows = {}
    for m in re.finditer(r"valueeval & ([\w\\_-]+) & direct & macro\\_ap & ([\d.]+) & ([\d.]+) & ([\d.]+) & ([\d.]+)", block):
        rows[m.group(1).replace("\\_", "_")] = tuple(float(m.group(i)) for i in (2, 3, 4, 5))
    return rows


def main():
    ncstyle.apply_style()
    fmat = pd.read_csv(FMAT_DIR / "matched-correlations.csv")
    fmat = fmat[fmat.subset == "all50"]
    order = fmat[fmat.study == "d1a"].sort_values("pearson_r").method.tolist()
    text = pd.read_csv(TEXT_CSV)
    pairs = json.loads((DWUG_DIR / "pooled-pair-validity.json").read_text())
    gains = no_content_gains()

    fig_w = ncstyle.DOUBLE_COL_INCH
    fig = plt.figure(figsize=(fig_w, fig_w * 0.86), layout="constrained")
    fig.get_layout_engine().set(h_pad=0.10, w_pad=0.06, hspace=0.06, wspace=0.04)
    gs = fig.add_gridspec(2, 3, height_ratios=[1.55, 1.0], width_ratios=[1.0, 1.0, 1.0])
    ax_occ = fig.add_subplot(gs[0, 0:2]); ax_nam = fig.add_subplot(gs[0, 2])
    ax_val = fig.add_subplot(gs[1, 0]); ax_gain = fig.add_subplot(gs[1, 1]); ax_dwug = fig.add_subplot(gs[1, 2])

    # a  FMAT criteria
    for ax, study, title, xlab, show in [
        (ax_occ, "d1a", "Gendered association across 17 scoring methods: occupations", "Pearson $r$, 50 occupations", True),
        (ax_nam, "d1b", "Names", "Pearson $r$, 50 names", False)]:
        frame = fmat[fmat.study == study].set_index("method").loc[order]
        for i, (method, row) in enumerate(frame.iterrows()):
            if method == PRIMARY:
                ncstyle.dot_interval(ax, row.pearson_r, i, row.pearson_ci_lower, row.pearson_ci_upper,
                                     color=ncstyle.BLUE, marker="o")
            elif method in GEN_LABEL:
                ax.errorbar(row.pearson_r, i, xerr=[[row.pearson_r - row.pearson_ci_lower], [row.pearson_ci_upper - row.pearson_r]],
                            fmt="none", ecolor=ncstyle.BLUE, elinewidth=ncstyle.LW_ERR * 0.8, capsize=0, zorder=3)
                ax.plot(row.pearson_r, i, "o", markerfacecolor="white", markeredgecolor=ncstyle.BLUE,
                        markeredgewidth=0.9, markersize=ncstyle.MS * 0.9, zorder=4)
            else:
                ncstyle.dot_interval(ax, row.pearson_r, i, row.pearson_ci_lower, row.pearson_ci_upper,
                                     color=ncstyle.GREY, marker="^")
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels([GEN_LABEL.get(m, m.replace("vinai/", "")) for m in order] if show else [],
                           fontsize=ncstyle.SZ_NOTE)
        ax.set(xlabel=xlab, xlim=(0.25, 1.0)); ax.set_title(title, fontsize=ncstyle.SZ_TITLE, loc="left")
        ax.invert_yaxis(); ncstyle.xgrid(ax)
    ncstyle.panel_label(ax_occ, "a", x=-0.36)
    handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor=ncstyle.BLUE, markeredgecolor="white", markersize=ncstyle.MS * 0.72, label="OmniAnchor (primary event)"),
               Line2D([0], [0], marker="o", color="w", markerfacecolor="white", markeredgecolor=ncstyle.BLUE, markeredgewidth=0.9, markersize=ncstyle.MS * 0.72, label="Other generative events"),
               Line2D([0], [0], marker="^", color="w", markerfacecolor=ncstyle.GREY, markeredgecolor="white", markersize=ncstyle.MS * 0.72, label="Stored masked models")]
    ax_occ.legend(handles=handles, loc="lower left", fontsize=ncstyle.SZ_NOTE, handletextpad=0.3)

    # b  direct macro AP and gains over the no-content control
    ax = ax_val; ncstyle.panel_label(ax, "b", x=-0.30, y=1.14)
    for j, (m, name) in enumerate(INST):
        row = text[(text.panel.str.startswith("A")) & (text.kind == "direct") & (text.method == m) & (text.metric == "macro_ap")].iloc[0]
        ncstyle.dot_interval_v(ax, j, row["value"], row.ci_lower, row.ci_upper,
                               color=ncstyle.INSTRUMENT_COLOURS[name], marker=ncstyle.INSTRUMENT_MARKERS[name])
    ax.set_xticks(range(4)); ax.set_xticklabels(["Omni-\nAnchor", "E5", "Embed-\nding", "Re-\nranker"], fontsize=ncstyle.SZ_NOTE)
    ax.set(ylabel="Macro AP, direct", ylim=(0.14, 0.30), xlim=(-0.5, 3.5))
    ax.set_title("Values: direct AP", fontsize=ncstyle.SZ_TITLE, loc="left"); ncstyle.ygrid(ax)
    ax.text(0.03, 0.04, "$n$ = 1,576 arguments", transform=ax.transAxes, fontsize=ncstyle.SZ_NOTE, color=ncstyle.MUTED)

    ax = ax_gain
    ax.text(0.03, 0.04, "same arguments;\npaired within instrument", transform=ax.transAxes, fontsize=ncstyle.SZ_NOTE, color=ncstyle.MUTED, linespacing=1.3)
    for j, (m, name) in enumerate(INST):
        g, lo, hi, q = gains[m]
        ncstyle.dot_interval(ax, g, j, lo, hi, color=ncstyle.INSTRUMENT_COLOURS[name], marker=ncstyle.INSTRUMENT_MARKERS[name])
        ax.text(0.162, j, f"$q$ = {_q(q)}", va="center", ha="right", fontsize=ncstyle.SZ_NOTE, color=ncstyle.INK2)
        print(f"  b gain {name}: {g:.3f} [{lo:.3f}, {hi:.3f}] q={q}")
    ncstyle.forest_zero(ax)
    ax.set_yticks(range(4)); ax.set_yticklabels([n for _, n in INST], fontsize=ncstyle.SZ_NOTE)
    ax.set(xlabel="Gain over no-content input", xlim=(-0.02, 0.168), xticks=[0, 0.05, 0.10, 0.15], ylim=(-0.6, 3.9))
    ax.set_title("Gain over no-content", fontsize=ncstyle.SZ_TITLE, loc="left")
    ax.invert_yaxis()

    # c  DWUG relatedness
    ax = ax_dwug; ncstyle.panel_label(ax, "c", x=-0.34, y=1.14)
    for i, (m, name) in enumerate([("native-raw_logp", "OmniAnchor"), ("qwen-reranker-anchor", "Reranker"),
                                   ("e5-original", "E5"), ("qwen-embedding-original", "Embedding")]):
        d = pairs[m]["target_group_bootstrap"]
        ncstyle.dot_interval(ax, d["estimate"], i, d["lower"], d["upper"],
                             color=ncstyle.INSTRUMENT_COLOURS[name], marker=ncstyle.INSTRUMENT_MARKERS[name])
    ax.set_yticks(range(4)); ax.set_yticklabels(["OmniAnchor", "Reranker", "E5", "Embedding"], fontsize=ncstyle.SZ_NOTE)
    ax.set(xlabel="Spearman $\\rho$, human relatedness\n(10,120 pairs; 9 words resampled)", xlim=(-0.05, 0.72), ylim=(3.9, -0.6))
    ax.set_title("Word-usage relatedness", fontsize=ncstyle.SZ_TITLE, loc="left")
    ax.axvline(0, color=ncstyle.GRID, lw=ncstyle.SPINE_W, zorder=0); ncstyle.xgrid(ax)
    ncstyle.save(fig, "fig2-text", FIGDIR); print("Fig. 2 saved.")


if __name__ == "__main__":
    main()
