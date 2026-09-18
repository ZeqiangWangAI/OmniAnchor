"""Fig. 7  Aggregate structure (r3 revision): readable node names and stated edge definitions."""
from __future__ import annotations
import json, sys
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import ncstyle

ROOT = Path(__file__).resolve().parents[2]  # repository root; frozen runs/ must be restored alongside
FIGDIR = Path(__file__).resolve().parent.parent / "figures"
DWUG_DIR = ROOT / "runs/final-results-update-20260911-01/runs/analysis-46173/analysis"
NET_DIR = ROOT / "runs/cpu-valueeval-evidence-20260911-01/runs/analysis-45995/analysis"
SHORT = {"Self-direction": "Self-dir.", "Universalism": "Univ.", "Benevolence": "Benev.", "Conformity": "Conf.", "Security": "Sec."}


def short(node):
    for k, v in SHORT.items():
        node = node.replace(k, v)
    return node


def _d(v): return ("$-$" + f"{-v:.3f}".lstrip("0")) if v < 0 else f"{v:.3f}".lstrip("0")


def main():
    ncstyle.apply_style()
    fig_w = ncstyle.DOUBLE_COL_INCH
    fig, axes = plt.subplots(1, 3, figsize=(fig_w, fig_w * 0.54), gridspec_kw={"width_ratios": [0.95, 0.95, 1.15]}, layout="constrained")
    fig.get_layout_engine().set(w_pad=0.10, wspace=0.05)

    ax = axes[0]; ncstyle.panel_label(ax, "a", x=-0.18, y=1.16)
    shift = pd.read_csv(DWUG_DIR / "per-target-shift.csv"); nat = shift[shift.method == "native-raw_logp"]
    ax.scatter(nat.human_change, nat.energy_distance, s=ncstyle.MS ** 2 * 0.6, color=ncstyle.BLUE, zorder=3, edgecolors="white", linewidths=ncstyle.MW)
    offsets = {"include_vb": (-18, 22), "face_nn": (-24, 4), "part_nn": (-10, -16), "land_nn": (12, -12), "multitude_nn": (24, 10), "grain_nn": (12, 6), "tip_nn": (8, 6), "thump_nn": (8, 4), "plane_nn": (-28, 2)}
    for r in nat.itertuples():
        ax.annotate(r.target.rsplit("_", 1)[0], (r.human_change, r.energy_distance), xytext=offsets.get(r.target, (5, 5)), textcoords="offset points",
                    fontsize=ncstyle.SZ_NOTE, color=ncstyle.INK2, arrowprops={"arrowstyle": "-", "color": ncstyle.MUTED, "lw": 0.4, "shrinkB": 2})
    ax.margins(x=0.22, y=0.26); ax.set(xlabel="Human semantic-change score", ylabel="OmniAnchor energy distance between periods")
    ax.set_title("Semantic change across\nnine held-out words", fontsize=ncstyle.SZ_TITLE, loc="left")
    ax.text(0.04, 0.96, "$\\rho$ = .200 [$-$.684, .893]\nresampling unit: target word", transform=ax.transAxes, ha="left", va="top", fontsize=ncstyle.SZ_NOTE, color=ncstyle.INK2, linespacing=1.35)

    ax = axes[1]; ncstyle.panel_label(ax, "b", x=-0.18, y=1.16)
    human = pd.read_csv(NET_DIR / "human-all-nodes-adjacency.csv", index_col=0)
    native = pd.read_csv(NET_DIR / "native-raw_logp-all-nodes-adjacency.csv", index_col=0)
    summary = json.loads((NET_DIR / "native-raw_logp/summary.json").read_text())
    nodes = human.index.tolist(); gold = human.loc[nodes, nodes].to_numpy(); pred = native.loc[nodes, nodes].to_numpy(); up = np.triu_indices(20, 1)
    ax.scatter(gold[up], pred[up], s=6, alpha=0.5, color=ncstyle.BLUE, edgecolors="none")
    lo, hi = summary["edge_weight_spearman_ci"]; rho = summary["point"]["edge_weight_spearman"]; jac = summary["point"]["top_k_edge_jaccard"]
    ax.set(xlabel="Human label-network edge\n(correlation of two label columns)", ylabel="OmniAnchor edge\n(correlation of two anchor columns)")
    ax.set_title("Edges miss the\nhuman structure", fontsize=ncstyle.SZ_TITLE, loc="left")
    ax.text(0.96, 0.04, f"Edge $\\rho$ = {_d(rho)} [{_d(lo)}, {_d(hi)}]\nTop-20 Jaccard = {_d(jac)}\n20 nodes, 190 edges;\n1,576 materials in 105 source groups",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=ncstyle.SZ_NOTE, color=ncstyle.INK2, linespacing=1.3)
    print(f"  b: rho {rho:.3f} [{lo:.3f},{hi:.3f}] jac {jac:.3f}")

    ax = axes[2]; ncstyle.panel_label(ax, "c", x=-0.62, y=1.16)
    et = pd.read_csv(NET_DIR / "native-raw_logp/edges.csv"); edges = {frozenset([r.source, r.target]): r for r in et.itertuples()}
    pairs = list(zip(*up)); order = [int(k) for k in np.argsort(-np.abs(pred[up]), kind="stable") if pred[up][k] != 0][:20]
    recs = [edges[frozenset([nodes[pairs[k][0]], nodes[pairs[k][1]]])] for k in order]
    labels = [f"{short(nodes[pairs[k][0]])} – {short(nodes[pairs[k][1]])}" for k in order]
    freqs = [r.top20_bootstrap_frequency for r in recs]
    ax.barh(range(20), freqs, color=ncstyle.BLUE, height=0.55)
    ax.set_yticks(range(20)); ax.set_yticklabels(labels, fontsize=6.0); ax.invert_yaxis()
    ax.set(xlim=(0, 1), xticks=[0, 0.5, 1.0], xlabel="Selection frequency across 1,000\nsource-group bootstraps")
    ax.set_title("The 20 strongest edges\nrecur across draws", fontsize=ncstyle.SZ_TITLE, loc="left"); ncstyle.xgrid(ax)
    print(f"  c: {min(freqs):.3f}-{max(freqs):.3f}; labels {labels[:3]}")
    ncstyle.save(fig, "fig7-structure", FIGDIR); print("Fig. 7 saved.")


if __name__ == "__main__":
    main()
