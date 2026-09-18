"""
Fig. 1  OmniAnchor: the scored event, the coordinates, and the validation design.

r3 redesign (2026-09-18).  Schematic; no data loaded.  183 mm wide, Helvetica
Neue, two accents (blue = instrument path, orange = human-criterion path).

Band a walks one photograph through the event: material, relation sentence,
frozen model, and the log probability of each researcher-named anchor.
Band b shows the coordinate matrix, the open anchor set and the two
calibrations.  Band c shows direct measurement and trained prediction, and
the five material cases the paper reports.
"""
from __future__ import annotations
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Polygon, Circle

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ncstyle as S
import matplotlib.font_manager as _fm
_fm.fontManager.addfont("/Library/Fonts/Arial Unicode.ttf")
CJK = "Arial Unicode MS"

FIGDIR = Path(__file__).resolve().parent.parent / "figures"
FONT = "Helvetica Neue"


def box(ax, x, y, w, h, text="", fc=S.FIG1_BLUE_TINT, ec=S.SPINE_CLR, fs=None,
        weight="normal", color=S.INK2, lw=0.6, ls="-", z=2, pad=0.02, r=0.05, **kw):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle=f"round,pad={pad},rounding_size={r}",
                                facecolor=fc, edgecolor=ec, linewidth=lw,
                                linestyle=ls, zorder=z))
    if text:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fs or S.SZ_ANNOT, fontweight=weight, color=color,
                fontfamily=FONT, linespacing=1.3, zorder=z + 1, **kw)


def arrow(ax, x0, y0, x1, y1, color=S.INK2, lw=0.7, z=3, style="-|>", ls="-"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style,
                                 color=color, lw=lw, mutation_scale=7,
                                 shrinkA=1.5, shrinkB=1.5, zorder=z, linestyle=ls))


def photo_icon(ax, x, y, w, h):
    """A tiny landscape: sky, hill, sun."""
    ax.add_patch(Rectangle((x, y), w, h, facecolor="#dbe7f3", edgecolor=S.SPINE_CLR,
                           linewidth=0.5, zorder=3))
    ax.add_patch(Polygon([(x, y), (x + w, y), (x + w, y + 0.32 * h),
                          (x + 0.62 * w, y + 0.62 * h), (x + 0.30 * w, y + 0.36 * h),
                          (x, y + 0.48 * h)], closed=True, facecolor="#9bb7a3",
                         edgecolor="none", zorder=4))
    ax.add_patch(Circle((x + 0.74 * w, y + 0.74 * h), 0.11 * h, facecolor="#f0c46b",
                        edgecolor="none", zorder=4))


def text_icon(ax, x, y, w, h, lines=4):
    ax.add_patch(Rectangle((x, y), w, h, facecolor="white", edgecolor=S.SPINE_CLR,
                           linewidth=0.5, zorder=3))
    for i in range(lines):
        yy = y + h * (0.82 - i * 0.2)
        ww = w * (0.72 if i < lines - 1 else 0.45)
        ax.plot([x + 0.14 * w, x + 0.14 * w + ww], [yy, yy], color=S.MUTED,
                lw=0.9, solid_capstyle="round", zorder=4)


def clip_icon(ax, x, y, w, h):
    for k in range(3):
        dx = k * 0.11 * w
        ax.add_patch(Rectangle((x + dx, y - k * 0.06 * h), 0.78 * w, h,
                               facecolor="#e6ebf2" if k else "#dbe7f3",
                               edgecolor=S.SPINE_CLR, linewidth=0.5, zorder=3 + (2 - k)))
    ax.add_patch(Polygon([(x + 0.30 * w, y + 0.30 * h), (x + 0.30 * w, y + 0.70 * h),
                          (x + 0.58 * w, y + 0.50 * h)], closed=True,
                         facecolor=S.BLUE, edgecolor="none", zorder=6))


def main():
    S.apply_style()
    W, H = 10.0, 8.1
    fig, ax = plt.subplots(figsize=(S.DOUBLE_COL_INCH, S.DOUBLE_COL_INCH * H / W))
    ax.set_position([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")

    bx, bw = 0.40, 9.20
    L, R = bx + 0.20, bx + bw - 0.20
    bands = {"a": (5.60, 7.80, S.FIG1_BLUE_BAND), "b": (2.95, 5.40, S.FIG1_BLUE_BAND),
             "c": (0.20, 2.75, S.FIG1_ORANGE_BAND)}
    titles = {"a": "The scored event: one photograph, four anchors",
              "b": "Coordinates: one row per material, one column per anchor",
              "c": "Validation against published human ratings"}
    for k, (b0, b1, col) in bands.items():
        ax.add_patch(FancyBboxPatch((bx, b0), bw, b1 - b0,
                                    boxstyle="round,pad=0.015,rounding_size=0.08",
                                    facecolor=col, edgecolor="none", zorder=0))
        ax.text(bx - 0.22, b1 - 0.03, k, fontsize=S.SZ_PANEL, fontweight="bold",
                va="top", ha="left", color=S.INK, fontfamily=FONT)
        ax.text(L, b1 - 0.19, titles[k], fontsize=S.SZ_TITLE, fontweight="medium",
                va="center", ha="left", color=S.INK, fontfamily=FONT)

    # ───────────────────────── band a ─────────────────────────
    a0, a1 = bands["a"][0], bands["a"][1]
    ym = (a0 + a1 - 0.36) / 2                      # vertical centre of the row

    # material stack (three cards)
    cw, ch, gap = 1.30, 0.46, 0.09
    y_top = ym + ch + gap / 2 + ch / 2 - 0.02
    cards = [("Sentence", text_icon), ("Photograph", photo_icon), ("Clip + caption", clip_icon)]
    for i, (lab, icon) in enumerate(cards):
        y = y_top - i * (ch + gap)
        box(ax, L, y - ch / 2, cw, ch, fc="white", z=2)
        icon(ax, L + 0.08, y - ch / 2 + 0.09, 0.42, ch - 0.18)
        ax.text(L + 0.58, y, lab, fontsize=S.SZ_ANNOT, va="center", ha="left",
                color=S.INK2, fontfamily=FONT, zorder=4)
    ax.text(L + cw / 2, y_top - 3 * (ch + gap) + ch / 2 - 0.04, "any material the model reads",
            fontsize=S.SZ_NOTE, ha="center", va="top", color=S.MUTED, fontfamily=FONT)

    # prompt box
    px, pw, ph = L + cw + 0.40, 3.05, 1.34
    py = ym - ph / 2
    box(ax, px, py, pw, ph, fc="white", z=2)
    ax.text(px + 0.14, py + ph - 0.16, "Input to the model", fontsize=S.SZ_NOTE,
            color=S.MUTED, va="top", ha="left", fontfamily=FONT, zorder=4)
    photo_icon(ax, px + 0.14, py + 0.42, 0.62, 0.50)
    ax.text(px + 0.86, py + 0.84, "material", fontsize=S.SZ_NOTE, color=S.MUTED,
            va="center", ha="left", fontfamily=FONT, zorder=4, style="italic")
    ax.text(px + 0.86, py + 0.58, "“The feeling this photograph\nevokes in a viewer is:”",
            fontsize=S.SZ_NOTE + 0.3, color=S.INK2, va="center", ha="left", fontfamily=FONT,
            zorder=4, linespacing=1.25)
    ax.text(px + 0.14, py + 0.16, "relation sentence, one of three fixed wordings",
            fontsize=S.SZ_NOTE, color=S.MUTED, va="center", ha="left", fontfamily=FONT, zorder=4)
    for i in range(3):
        yy = y_top - i * (ch + gap)
        arrow(ax, L + cw, yy, px, ym + (1 - i) * 0.18, color=S.SPINE_CLR, lw=0.6)

    # model box
    mx, mw, mh = px + pw + 0.34, 1.22, 0.80
    my = ym - mh / 2
    box(ax, mx, my, mw, mh, "Frozen multimodal\ngenerative model", fc=S.FIG1_BLUE_TINT,
        ec=S.BLUE, lw=0.7, color=S.BLUE, weight="medium", z=2)
    ax.text(mx + mw / 2, my - 0.13, "no weights trained", fontsize=S.SZ_NOTE,
            color=S.MUTED, ha="center", va="top", fontfamily=FONT)
    arrow(ax, px + pw, ym, mx, ym)

    # anchor readout: horizontal bars of log probability
    rx = mx + mw + 0.34
    rw = R - rx
    anchors = [("pleasant", 0.86), ("calm", 0.62), ("aroused", 0.33), ("unpleasant", 0.18)]
    bar_h, bar_gap = 0.20, 0.10
    tot = 4 * bar_h + 3 * bar_gap
    by0 = ym + tot / 2 - bar_h
    lab_w = 0.74
    bar_x0 = rx + lab_w
    bar_max = rw - lab_w - 0.10
    ax.plot([bar_x0, bar_x0], [by0 - 3 * (bar_h + bar_gap) - 0.04, by0 + bar_h + 0.04],
            color=S.SPINE_CLR, lw=0.6, zorder=2)
    for i, (name, v) in enumerate(anchors):
        y = by0 - i * (bar_h + bar_gap)
        ax.text(bar_x0 - 0.08, y + bar_h / 2, name, fontsize=S.SZ_ANNOT, ha="right",
                va="center", color=S.INK2, fontfamily=FONT, zorder=4)
        ax.add_patch(Rectangle((bar_x0, y), v * bar_max, bar_h, facecolor=S.BLUE,
                               alpha=0.25 + 0.75 * v, edgecolor="none", zorder=3))
    ax.text(rx, ym + tot / 2 + 0.16, "Anchors the researcher names",
            fontsize=S.SZ_NOTE, color=S.MUTED, ha="left", va="bottom", fontfamily=FONT)
    ax.text(rx, ym - tot / 2 - 0.12,
            "log p(anchor | input), averaged over the\nthree wordings, is the anchor coordinate",
            fontsize=S.SZ_NOTE, color=S.BLUE, ha="left", va="top", fontfamily=FONT, linespacing=1.25)
    arrow(ax, mx + mw, ym, rx + 0.02, ym)

    # ───────────────────────── band b ─────────────────────────
    b0, b1 = bands["b"][0], bands["b"][1]
    ymb = (b0 + b1 - 0.36) / 2 - 0.04
    # matrix
    ncol, nrow, cs = 5, 3, 0.34
    gx, gy = L + 0.95, ymb - nrow * cs / 2 - 0.10
    vals = [[0.9, 0.55, 0.25, 0.4, 0.7], [0.3, 0.8, 0.6, 0.2, 0.5], [0.5, 0.35, 0.85, 0.65, 0.3]]
    rows = ["argument", "photograph", "clip"]
    cols = ["benevolence", "security", "pleasant", "快乐", "freedom"]
    for r in range(nrow):
        ax.text(gx - 0.08, gy + (nrow - 1 - r) * cs + cs / 2, rows[r], fontsize=S.SZ_NOTE,
                ha="right", va="center", color=S.INK2, fontfamily=FONT)
        for c in range(ncol):
            v = vals[r][c]
            ax.add_patch(Rectangle((gx + c * cs, gy + (nrow - 1 - r) * cs), cs, cs,
                                   facecolor=S.BLUE, alpha=0.12 + 0.75 * v,
                                   edgecolor="white", linewidth=0.8, zorder=3))
    for c in range(ncol):
        ax.text(gx + c * cs + cs / 2, gy + nrow * cs + 0.06, cols[c], fontsize=S.SZ_NOTE,
                ha="left", va="bottom", rotation=35, color=S.INK2,
                fontfamily=FONT if c != 3 else CJK)
    # open set: dashed new column
    nx = gx + ncol * cs + 0.10
    ax.add_patch(Rectangle((nx, gy), cs, nrow * cs, facecolor="none", edgecolor=S.BLUE,
                           linewidth=0.7, linestyle=(0, (2, 1.5)), zorder=3))
    ax.text(nx + cs / 2, gy + nrow * cs + 0.06, "+ new anchor", fontsize=S.SZ_NOTE,
            ha="left", va="bottom", rotation=35, color=S.BLUE, fontfamily=FONT)
    ax.text(gx, gy - 0.14, "Open anchor set: adding a column leaves every existing\n"
            "coordinate unchanged, because no candidate list reaches the model",
            fontsize=S.SZ_NOTE, ha="left", va="top", color=S.MUTED, fontfamily=FONT,
            linespacing=1.25)
    # calibration on the right
    cx0 = nx + cs + 0.70
    cbw, cbh = 1.55, 0.52
    cy = ymb - cbh / 2 + 0.16
    box(ax, cx0, cy, cbw, cbh, "Raw coordinate\nlog probability, nats", fc="white")
    arrow(ax, cx0 + cbw, cy + cbh / 2, cx0 + cbw + 0.42, cy + cbh / 2)
    cx1 = cx0 + cbw + 0.42
    cbw2 = R - cx1
    box(ax, cx1, cy + 0.36, cbw2, cbh * 0.78, "Log ratio against a reference set", fc=S.FIG1_NEUTRAL,
        fs=S.SZ_NOTE)
    box(ax, cx1, cy - 0.36 + 0.06, cbw2, cbh * 0.78, "Reference $z$ score", fc=S.FIG1_NEUTRAL,
        fs=S.SZ_NOTE)
    ax.text(cx1 + cbw2 / 2, cy - 0.44, "calibration moves the origin;\ndistances and rankings stay put (Propositions 1 and 2)",
            fontsize=S.SZ_NOTE, ha="center", va="top", color=S.MUTED, fontfamily=FONT,
            linespacing=1.25)

    # ───────────────────────── band c ─────────────────────────
    c0, c1 = bands["c"][0], bands["c"][1]
    top = c1 - 0.42
    hw = (R - L - 0.40) / 2
    bh = 1.02
    yb = top - bh
    box(ax, L, yb, hw, bh, fc="white", ec=S.ORANGE, lw=0.7)
    ax.text(L + 0.14, yb + bh - 0.14, "Direct measurement", fontsize=S.SZ_ANNOT,
            fontweight="medium", color=S.ORANGE, ha="left", va="top", fontfamily=FONT, zorder=4)
    ax.text(L + 0.14, yb + bh - 0.40,
            "Coordinates that no label has\ntouched are correlated with\npublished human ratings",
            fontsize=S.SZ_NOTE, color=S.INK2, ha="left", va="top", fontfamily=FONT,
            linespacing=1.25, zorder=4)
    # mini scatter
    sx, sy, ss = L + hw - 0.86, yb + 0.16, 0.62
    ax.plot([sx, sx + ss], [sy, sy], color=S.SPINE_CLR, lw=0.5, zorder=3)
    ax.plot([sx, sx], [sy, sy + ss], color=S.SPINE_CLR, lw=0.5, zorder=3)
    pts = [(0.15, 0.2), (0.3, 0.42), (0.45, 0.38), (0.55, 0.6), (0.7, 0.66), (0.85, 0.82), (0.4, 0.22), (0.62, 0.5)]
    for u, v in pts:
        ax.plot(sx + u * ss, sy + v * ss, "o", ms=2.2, color=S.ORANGE, mec="white", mew=0.3, zorder=4)
    ax.text(sx + ss / 2, sy - 0.03, "human rating", fontsize=5.5, color=S.MUTED, ha="center", va="top", fontfamily=FONT)
    ax.text(sx - 0.03, sy + ss / 2, "coordinate", fontsize=5.5, color=S.MUTED, ha="right", va="center", rotation=90, fontfamily=FONT)

    x2 = L + hw + 0.40
    box(ax, x2, yb, hw, bh, fc="white", ec=S.ORANGE, lw=0.7)
    ax.text(x2 + 0.14, yb + bh - 0.14, "Trained prediction", fontsize=S.SZ_ANNOT,
            fontweight="medium", color=S.ORANGE, ha="left", va="top", fontfamily=FONT, zorder=4)
    ax.text(x2 + 0.14, yb + bh - 0.40,
            "One linear predictor, fitted on\nlabelled training rows, to the\ncoordinates and to general vectors",
            fontsize=S.SZ_NOTE, color=S.INK2, ha="left", va="top", fontfamily=FONT,
            linespacing=1.25, zorder=4)
    # mini pipeline
    p0 = x2 + hw - 2.20
    for i, t in enumerate(["coordinates", "predictor", "rating"]):
        box(ax, p0 + i * 0.72, yb + bh / 2 - 0.13, 0.60, 0.26, t, fc=S.FIG1_ORANGE_TINT, fs=5.5, z=3)
        if i < 2:
            arrow(ax, p0 + i * 0.72 + 0.60, yb + bh / 2, p0 + (i + 1) * 0.72, yb + bh / 2, lw=0.5)

    ax.text(L, yb - 0.12, "Every protected set is scored once after the instrument is frozen; comparisons are paired and corrected within registered test families.",
            fontsize=S.SZ_NOTE, color=S.MUTED, ha="left", va="top", fontfamily=FONT)
    chips = ["Text", "English and Chinese", "Photographs", "Photograph + action text", "Video (demonstration)"]
    n = len(chips); cg = 0.12; cwid = (R - L - (n - 1) * cg) / n
    cy0 = c0 + 0.16
    for i, t in enumerate(chips):
        box(ax, L + i * (cwid + cg), cy0, cwid, 0.34, t, fc=S.FIG1_ORANGE_TINT, fs=S.SZ_NOTE)

    S.save(fig, "fig1-workflow-alt", FIGDIR, tight=False)


if __name__ == "__main__":
    main()
