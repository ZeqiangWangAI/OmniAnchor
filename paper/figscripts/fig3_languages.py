"""Fig. 3  Trained prediction (r3 revision)."""
from __future__ import annotations
import re, sys
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import ncstyle

ROOT = Path(__file__).resolve().parents[2]  # repository root; frozen runs/ must be restored alongside
FIGDIR = Path(__file__).resolve().parent.parent / "figures"
SI_TEX = Path(__file__).resolve().parent.parent / "si.tex"
TEXT_CSV = ROOT / "runs/scienceplots-text-final-20260911-01/plotted-results.csv"
ZH_PAIRED = ROOT / "runs/cpu-affect-controls-evidence-20260911-01/runs/analysis-45857/analysis/paired-differences.csv"
EN_PAIRED = ROOT / "runs/cpu-affect-controls-evidence-20260911-01/runs/analysis-45859/analysis/paired-differences.csv"

TRAINED = ["native-raw_logp", "e5-original", "qwen-embedding-original", "qwen-reranker-anchor"]
INST = {"native-raw_logp": "OmniAnchor", "e5-original": "E5", "qwen-embedding-original": "Embedding", "qwen-reranker-anchor": "Reranker"}
DIM = {"spearman_V": "Valence", "spearman_A": "Arousal", "spearman_D": "Dominance"}


def _pick(d, pfx, kind, m, metric):
    s = d[(d.panel.str.startswith(pfx)) & (d.kind == kind) & (d.method == m) & (d.metric == metric)]
    assert len(s) == 1; return s.iloc[0]


def si_trained():
    t = SI_TEX.read_text(); i = t.index("label{tab:valuepred}")
    block = t[i:t.index("end{table}", i)]
    out = {}
    for m in re.finditer(r"([\w\\_-]+) & ([\d.]+) & ([\d.]+) & ([\d.]+) & 1576", block):
        out[m.group(1).replace("\\_", "_")] = tuple(float(m.group(k)) for k in (2, 3, 4))
    return out


def _d(v): return ("$-$" + f"{-v:.3f}".lstrip("0")) if v < 0 else f"{v:.3f}".lstrip("0")


def main():
    ncstyle.apply_style()
    data = pd.read_csv(TEXT_CSV); trained = si_trained()
    zh = pd.read_csv(ZH_PAIRED); en = pd.read_csv(EN_PAIRED)

    fig_w = ncstyle.DOUBLE_COL_INCH
    fig = plt.figure(figsize=(fig_w, fig_w * 0.80), layout="constrained")
    fig.get_layout_engine().set(h_pad=0.10, w_pad=0.08, hspace=0.08, wspace=0.05)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.95], width_ratios=[1.3, 1.0])
    ax_en = fig.add_subplot(gs[0, 0]); ax_zh = fig.add_subplot(gs[0, 1]); ax_c = fig.add_subplot(gs[1, :])

    for ax, letter, pfx, metrics, title, note in [
        (ax_en, "a", "C", ["spearman_V", "spearman_A", "spearman_D"],
         "English affect: E5 ahead on\nvalence and dominance", "$n$ = 1,000 sentences\nEnglish anchor pack"),
        (ax_zh, "b", "D", ["spearman_V", "spearman_A"],
         "Chinese affect: differences\nfrom E5 include zero", "$n$ = 528 sentences; Chinese anchor pack")]:
        ncstyle.panel_label(ax, letter, x=-0.12, y=1.24)
        for j, m in enumerate(TRAINED):
            name = INST[m]
            for i, metric in enumerate(metrics):
                row = _pick(data, pfx, "probe", m, metric)
                ncstyle.dot_interval_v(ax, i + (j - 1.5) * 0.14, row["value"], row.ci_lower, row.ci_upper,
                                       color=ncstyle.INSTRUMENT_COLOURS[name], marker=ncstyle.INSTRUMENT_MARKERS[name])
        ax.set_xticks(range(len(metrics))); ax.set_xticklabels([DIM[m] for m in metrics], fontsize=ncstyle.SZ_ANNOT)
        ax.set(ylabel="Spearman $\\rho$, trained prediction", ylim=(0, 0.95), xlim=(-0.5, len(metrics) - 0.5))
        ax.set_title(title, fontsize=ncstyle.SZ_TITLE, loc="left"); ncstyle.ygrid(ax)
        ax.text(0.02, 0.96, note, transform=ax.transAxes, va="top", fontsize=ncstyle.SZ_NOTE, color=ncstyle.MUTED)
    # paired differences OmniAnchor - E5, protected sets
    def diff_text(tab, metrics):
        parts = []
        for m in metrics:
            r = tab[(tab.comparator == "e5-original") & (tab.metric == m)].iloc[0]
            parts.append(f"{DIM[m][0]}: {_d(r.difference)} [{_d(r.ci_lower)}, {_d(r.ci_upper)}], $q$ = {r.family_bh_q:.3f}".replace("= 0.", "= ."))
            print(f"  diff {m}: {r.difference:.4f} [{r.ci_lower:.3f}, {r.ci_upper:.3f}] q={r.family_bh_q:.3f}")
        return "Paired difference, OmniAnchor $-$ E5\n" + "\n".join(parts)
    ax_zh.text(0.98, 0.04, diff_text(zh, ["spearman_V", "spearman_A"]), transform=ax_zh.transAxes,
               ha="right", va="bottom", fontsize=ncstyle.SZ_NOTE, color=ncstyle.INK2, linespacing=1.35)
    ax_en.text(0.98, 0.97, diff_text(en, ["spearman_V", "spearman_A", "spearman_D"]), transform=ax_en.transAxes,
               ha="right", va="top", fontsize=ncstyle.SZ_NOTE, color=ncstyle.INK2, linespacing=1.35)

    # c  ValueEval: direct -> trained with the representation held fixed
    ax = ax_c; ncstyle.panel_label(ax, "c", x=-0.06, y=1.12)
    rows = [("OmniAnchor coordinates", "OmniAnchor", "native-raw_logp", "native-raw_logp"),
            ("E5 anchor projection", "E5", "e5-anchor", "e5-anchor"),
            ("Embedding-head anchor projection", "Embedding", "qwen-embedding-anchor", "qwen-embedding-anchor"),
            ("Reranker-head anchor coordinates", "Reranker", "qwen-reranker-anchor", "qwen-reranker-anchor"),
            ("E5 original vectors (trained only)", "E5", None, "e5-original"),
            ("Embedding-head original vectors (trained only)", "Embedding", None, "qwen-embedding-original")]
    for i, (label, name, dm, tm) in enumerate(rows):
        colour = ncstyle.INSTRUMENT_COLOURS[name]; marker = ncstyle.INSTRUMENT_MARKERS[name]
        tv, tlo, thi = trained[tm]
        if dm is not None:
            d = _pick(data, "A", "direct", dm, "macro_ap")
            ax.plot([d["value"], tv], [i, i], "-", color=ncstyle.SPINE_CLR, lw=ncstyle.SPINE_W, zorder=1)
            ax.errorbar(d["value"], i, xerr=[[d["value"] - d.ci_lower], [d.ci_upper - d["value"]]], fmt="none", ecolor=colour, elinewidth=ncstyle.LW_ERR, capsize=0, zorder=2)
            ax.plot(d["value"], i, marker=marker, color=colour, markerfacecolor="white", markeredgecolor=colour, markeredgewidth=0.9, markersize=ncstyle.MS, zorder=3)
            print(f"  c {label}: direct {d['value']:.3f} -> trained {tv:.3f} [{tlo:.3f},{thi:.3f}]")
        else:
            print(f"  c {label}: trained {tv:.3f} [{tlo:.3f},{thi:.3f}]")
        ax.errorbar(tv, i, xerr=[[tv - tlo], [thi - tv]], fmt="none", ecolor=colour, elinewidth=ncstyle.LW_ERR, capsize=0, zorder=2)
        ax.plot(tv, i, marker=marker, color=colour, markeredgecolor="white", markeredgewidth=ncstyle.MW, markersize=ncstyle.MS, zorder=4)
    ax.axhline(3.5, color=ncstyle.GRID, lw=ncstyle.SPINE_W, zorder=0)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows], fontsize=ncstyle.SZ_ANNOT)
    ax.set(xlabel="Macro AP", xlim=(0.14, 0.50)); ax.set_ylim(len(rows) - 0.2, -0.9)
    ax.set_title("Values: direct (open) and trained (filled) with the representation held fixed",
                 fontsize=ncstyle.SZ_TITLE, loc="left"); ncstyle.xgrid(ax)
    d0 = _pick(data, "A", "direct", "native-raw_logp", "macro_ap")["value"]; t0 = trained["native-raw_logp"][0]
    ax.annotate("Direct", (d0, 0), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=ncstyle.SZ_NOTE, color=ncstyle.INK2)
    ax.annotate("Trained", (t0, 0), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=ncstyle.SZ_NOTE, color=ncstyle.INK2)
    ax.text(0.02, 0.02, "$n$ = 1,576 arguments; one predictor, grid fixed on development rows",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=ncstyle.SZ_NOTE, color=ncstyle.MUTED)
    ncstyle.instrument_legend(fig, ["OmniAnchor", "E5", "Embedding", "Reranker"], ncol=4, loc="lower center", bbox_to_anchor=(0.5, 0.0))
    fig.get_layout_engine().set(rect=(0, 0.04, 1, 0.96))
    ncstyle.save(fig, "fig3-languages", FIGDIR); print("Fig. 3 saved.")


if __name__ == "__main__":
    main()
