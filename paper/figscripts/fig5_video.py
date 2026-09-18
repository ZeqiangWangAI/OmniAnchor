"""Fig. 5  Video demonstration (r3 revision): cosine matrix with the actual argmax, and match counts at 8 and 16 frames."""
from __future__ import annotations
import json, sys
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import ncstyle

FIGDIR = Path(__file__).resolve().parent.parent / "figures"
EVID = Path(__file__).resolve().parent.parent / "evidence"
RUNS = {"development-47130": 8, "development-46727": 16}
BLUE_TINTS = LinearSegmentedColormap.from_list("tints", ["#f2f6fa", "#bccfe5", "#789fcb", "#1f5fa8"])


def _clip(item): return item.split(":")[1]


def load(run):
    cos = pd.read_csv(EVID / f"video-demo-cosine-{run}.csv", index_col=0)
    rows = [r for r in cos.index if ":en:" in r] + [r for r in cos.index if ":zh:" in r]
    cols = [f"vatex:{_clip(r)}:video" for r in rows]
    return cos.loc[rows, cols].to_numpy()


def main():
    ncstyle.apply_style()
    match = json.loads((EVID / "video-demo-match.json").read_text())["runs"]
    M = load("development-46727")
    c2v = int(sum(M[i].argmax() == i for i in range(16))); v2c = int(sum(M[:, j].argmax() == j for j in range(16)))
    assert (c2v, v2c) == (match["development-46727"]["caption_to_video_recall_at_1"], match["development-46727"]["video_to_caption_recall_at_1"])

    fig_w = ncstyle.DOUBLE_COL_INCH
    fig, axes = plt.subplots(1, 2, figsize=(fig_w, fig_w * 0.52), gridspec_kw={"width_ratios": [1.25, 0.75]}, layout="constrained")
    ax = axes[0]; ncstyle.panel_label(ax, "a", x=-0.20, y=1.12)
    im = ax.imshow(M, cmap=BLUE_TINTS, vmin=float(M.min()), vmax=float(M.max()), interpolation="nearest", aspect="equal")
    for i in range(16):
        ax.add_patch(Rectangle((i - 0.5, i - 0.5), 1, 1, fill=False, edgecolor=ncstyle.INK, linewidth=0.6, zorder=4))
        j = int(M[i].argmax())
        ax.plot(j, i, marker="x", color=ncstyle.ORANGE if j != i else "white", markersize=3.2 if j != i else 2.2, markeredgewidth=0.9, zorder=5)
    ax.set_xticks(range(16)); ax.set_yticks(range(16))
    ax.set_xticklabels([str(i + 1) for i in range(16)], fontsize=ncstyle.SZ_NOTE); ax.set_yticklabels([str(i + 1) for i in range(16)], fontsize=ncstyle.SZ_NOTE)
    ax.set_xlabel("Clip"); ax.set_ylabel("Caption", labelpad=24)
    ax.set_title(f"Matched clips rank first for {c2v} of 16 captions", fontsize=ncstyle.SZ_TITLE, loc="left")
    for s in ax.spines.values(): s.set_visible(False)
    ax.tick_params(length=0)
    trans = ax.get_yaxis_transform()
    for y0, y1, name in [(-0.5, 7.5, "English"), (7.5, 15.5, "Chinese")]:
        ax.plot([-0.075, -0.075], [y0 + 0.1, y1 - 0.1], transform=trans, color=ncstyle.SPINE_CLR, lw=ncstyle.SPINE_W, clip_on=False, zorder=5)
        ax.text(-0.105, (y0 + y1) / 2, name, transform=trans, rotation=90, va="center", ha="center", fontsize=ncstyle.SZ_NOTE, color=ncstyle.INK2, clip_on=False)
    cbar = fig.colorbar(im, ax=ax, fraction=0.040, pad=0.02, aspect=20, ticks=[0.990, 0.994, 0.998])
    cbar.ax.set_yticklabels([".990", ".994", ".998"], fontsize=ncstyle.SZ_NOTE); cbar.set_label("Cosine similarity (raw coordinates)", fontsize=ncstyle.SZ_NOTE, color=ncstyle.INK2)
    cbar.outline.set_visible(False); cbar.ax.tick_params(length=1.5, color=ncstyle.MUTED)
    handles = [Line2D([0], [0], marker="s", color="w", markerfacecolor="none", markeredgecolor=ncstyle.INK, markersize=5, label="True pair"),
               Line2D([0], [0], marker="x", color="w", markeredgecolor=ncstyle.ORANGE, markeredgewidth=0.9, markersize=4, label="Row maximum elsewhere")]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.0, -0.09), ncol=2, fontsize=ncstyle.SZ_NOTE, frameon=False, handletextpad=0.3)
    ax.text(0.0, -0.20, f"At 16 frames: caption to clip {c2v}/16; clip to caption {v2c}/16", transform=ax.transAxes, ha="left", va="top", fontsize=ncstyle.SZ_NOTE, color=ncstyle.INK2)

    # b  match counts at 8 and 16 frames
    ax = axes[1]; ncstyle.panel_label(ax, "b", x=-0.16, y=1.12)
    series = [("caption_to_video_recall_at_1", "Caption to clip", "o", ncstyle.BLUE), ("video_to_caption_recall_at_1", "Clip to caption", "s", ncstyle.INK2)]
    for k, (key, label, marker, colour) in enumerate(series):
        xs, ys = [], []
        for run, frames in RUNS.items():
            xs.append(frames); ys.append(match[run][key])
        ax.plot(xs, ys, "-", color=colour, lw=ncstyle.LW, zorder=2)
        ax.plot(xs, ys, marker, color=colour, linestyle="none", markersize=ncstyle.MS, markeredgecolor="white", markeredgewidth=ncstyle.MW, zorder=3, label=label)
        for x, y in zip(xs, ys):
            ax.annotate(f"{y}/16", (x, y), xytext=(0, 7 if k == 0 else -11), textcoords="offset points", ha="center", fontsize=ncstyle.SZ_NOTE, color=colour)
            print(f"  b: {label} {x} frames {y}/16")
    en = [match[r]["caption_to_video_recall_at_1_en"] for r in RUNS]; zh = [match[r]["caption_to_video_recall_at_1_zh"] for r in RUNS]
    ax.text(0.03, 0.05, f"Caption to clip, by language\nEnglish {en[0]}/8 and {en[1]}/8; Chinese {zh[0]}/8 and {zh[1]}/8",
            transform=ax.transAxes, fontsize=ncstyle.SZ_NOTE, color=ncstyle.MUTED, linespacing=1.35)
    ax.set(xlabel="Sampled frames per clip", ylabel="Items whose own match ranks first", xlim=(5, 19), xticks=[8, 16], ylim=(10, 16.9), yticks=[10, 12, 14, 16])
    ax.set_title("Counts at 8 and 16 frames", fontsize=ncstyle.SZ_TITLE, loc="left"); ncstyle.ygrid(ax)
    ax.legend(loc="upper left", fontsize=ncstyle.SZ_NOTE, frameon=False, handletextpad=0.3)
    fig.text(0.015, 0.01, "16 clips, one caption each (8 English, 8 Chinese); 128 general anchors; one relation. Demonstration only, no human criterion; no interval (single frozen run).",
             ha="left", va="bottom", fontsize=ncstyle.SZ_NOTE, color=ncstyle.INK2)
    fig.get_layout_engine().set(rect=(0, 0.05, 1, 0.95), w_pad=0.12)
    ncstyle.save(fig, "fig5-video", FIGDIR); print("Fig. 5 saved.")


if __name__ == "__main__":
    main()
