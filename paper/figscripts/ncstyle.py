"""
Nature Communications visual system for OmniAnchor figures.

One source of truth for typography, colour tokens, instrument slots, mark
specs, panel layout and file output.  Every figure script imports this
module and uses its helpers; no token is overridden locally.
"""
from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.lines import Line2D
from pathlib import Path
from typing import Sequence

# ── Font registration ─────────────────────────────────────────────────
# Times New Roman (system files); the manuscript figures use a serif face.
import glob as _glob
for _ttf in sorted(_glob.glob("/System/Library/Fonts/Supplemental/Times New Roman*.ttf")):
    fm.fontManager.addfont(_ttf)
FONT_FAMILY = "Times New Roman"

# ── Ink tokens ────────────────────────────────────────────────────────
INK       = "#0b0b0b"   # primary: titles, panel letters
INK2      = "#52514e"   # secondary: axis labels, annotations
MUTED     = "#898781"   # tick labels, notes
GRID      = "#e1e0d9"   # gridlines
SPINE_CLR = "#c3c2b7"   # axis baselines, reference lines

# ── Categorical instrument slots (fixed order, never cycled) ──────────
BLUE   = "#1f5fa8"   # slot 1  OmniAnchor (native declared coordinates)
ORANGE = "#d9761a"   # slot 2  Reranker head
TEAL   = "#2a9d8f"   # slot 3  E5 (original vectors / anchor projection)
GREY   = "#6b6a66"   # slot 4  Embedding head (achromatic comparator)

INSTRUMENT_COLOURS = {
    "OmniAnchor": BLUE, "Reranker": ORANGE,
    "E5": TEAL, "Embedding": GREY,
}
INSTRUMENT_MARKERS = {
    "OmniAnchor": "o", "Reranker": "s",   # circle, square
    "E5": "D", "Embedding": "^",        # diamond, triangle-up
}

# FMAT shorthand (Fig. 2a)
GENERATIVE_COLOUR = BLUE
GENERATIVE_MARKER = "o"
STORED_COLOUR     = GREY
STORED_MARKER     = "^"

# Fig. 1 diagram accents (8 % tint of slot colour on white)
FIG1_BLUE_TINT   = "#edf2f8"
FIG1_ORANGE_TINT = "#fcf4ed"
FIG1_NEUTRAL     = "#f4f3ef"
# Band backgrounds (5 % tint)
FIG1_BLUE_BAND   = "#f4f7fb"
FIG1_ORANGE_BAND = "#fdf8f4"
FIG1_NEUTRAL_BAND = "#f8f7f5"

# Coverage-matrix tints (25 % of instrument colour on white)
COV_BLUE_TINT   = "#c7d7e9"
COV_ORANGE_TINT = "#f5dcc6"
COV_EMPTY       = "#f4f3ef"

# ── Dimensions ────────────────────────────────────────────────────────
def mm(v: float) -> float:
    """Millimetres to inches."""
    return v / 25.4

SINGLE_COL_MM  = 89
DOUBLE_COL_MM  = 183
SINGLE_COL_INCH = mm(SINGLE_COL_MM)
DOUBLE_COL_INCH = mm(DOUBLE_COL_MM)

def figsize_mm(w_mm: float, h_mm: float) -> tuple[float, float]:
    """Return (width_inches, height_inches) from mm values."""
    return mm(w_mm), mm(h_mm)

# ── Typographic sizes (pt at print scale) ─────────────────────────────
SZ_PANEL = 10.0    # Bold
SZ_TITLE = 9.0    # Medium (weight 500)
SZ_LABEL = 8.5    # axis labels, Regular
SZ_TICK  = 8.0    # tick labels, Regular
SZ_ANNOT = 8.0    # annotations, legends, Regular
SZ_NOTE  = 7.0    # footnotes (hard minimum)

# ── Mark specs ────────────────────────────────────────────────────────
LW       = 1.0    # data-line width (pt)
LW_ERR   = 0.8    # interval-line width
MS       = 5.5    # marker diameter (pt)
MW       = 0.8    # white-ring width around markers
SPINE_W  = 0.6    # spine and reference-line width
GRID_W   = 0.4    # gridline width
TICK_LEN = 2.0    # tick length, outward
BAR_MAX  = 6.0    # max bar thickness (pt)


# ──────────────────────────────────────────────────────────────────────
def apply_style() -> None:
    """Set all rcParams for the NC visual system."""
    mpl.rcParams.update({
        # Font
        "font.family":       FONT_FAMILY,
        "font.size":         SZ_TICK,
        "pdf.fonttype":      42,
        "ps.fonttype":       42,
        "mathtext.fontset":  "custom",
        "mathtext.rm":       FONT_FAMILY,
        "mathtext.it":       FONT_FAMILY + ":italic",
        "mathtext.bf":       FONT_FAMILY + ":bold",
        # Axes
        "axes.linewidth":    SPINE_W,
        "axes.edgecolor":    SPINE_CLR,
        "axes.labelsize":    SZ_LABEL,
        "axes.labelcolor":   INK2,
        "axes.titlesize":    SZ_TITLE,
        "axes.titleweight":  "bold",
        "axes.titlepad":     4,
        "axes.labelpad":     3,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.titlecolor":   INK,
        # Ticks
        "xtick.labelsize":   SZ_TICK,
        "ytick.labelsize":   SZ_TICK,
        "xtick.labelcolor":  MUTED,
        "ytick.labelcolor":  MUTED,
        "xtick.color":       MUTED,
        "ytick.color":       MUTED,
        "xtick.major.width": SPINE_W,
        "ytick.major.width": SPINE_W,
        "xtick.major.size":  TICK_LEN,
        "ytick.major.size":  TICK_LEN,
        "xtick.major.pad":   2,
        "ytick.major.pad":   2,
        "xtick.direction":   "out",
        "ytick.direction":   "out",
        # Lines / markers
        "lines.linewidth":   LW,
        "lines.markersize":  MS,
        # Legend
        "legend.fontsize":   SZ_ANNOT,
        "legend.frameon":    False,
        "legend.handlelength":   1.2,
        "legend.handletextpad":  0.4,
        "legend.columnspacing":  0.8,
        # Grid
        "grid.linewidth":    GRID_W,
        "grid.color":        GRID,
        "grid.alpha":        1.0,
        # Output
        "savefig.dpi":       300,
        "savefig.pad_inches": mm(4),
        "figure.dpi":        150,
    })


# ── Panel letters ─────────────────────────────────────────────────────
def panel_label(ax, letter: str, x: float = -0.06, y: float = 1.08) -> None:
    """Bold 9 pt panel letter, outside the axes, top-left."""
    ax.text(x, y, letter, transform=ax.transAxes,
            fontsize=SZ_PANEL, fontweight="bold",
            va="top", ha="right", color=INK)


# ── Grid helpers ──────────────────────────────────────────────────────
def xgrid(ax) -> None:
    """Vertical gridlines only, below data."""
    ax.grid(axis="x", linewidth=GRID_W, color=GRID, zorder=0)
    ax.set_axisbelow(True)

def ygrid(ax) -> None:
    """Horizontal gridlines only, below data."""
    ax.grid(axis="y", linewidth=GRID_W, color=GRID, zorder=0)
    ax.set_axisbelow(True)


# ── Dot-interval helpers ──────────────────────────────────────────────
def dot_interval(ax, x, y, lo, hi, *,
                 color=BLUE, marker="o", label=None, zorder=3):
    """Horizontal dot + 95 % interval, white-ring marker, no caps."""
    ax.errorbar(x, y, xerr=[[x - lo], [hi - x]],
                fmt="none", ecolor=color, elinewidth=LW_ERR,
                capsize=0, zorder=zorder)
    ax.plot(x, y, marker=marker, color=color,
            markersize=MS, markeredgecolor="white",
            markeredgewidth=MW, zorder=zorder + 1, label=label)


def dot_interval_v(ax, x, y, lo, hi, *,
                   color=BLUE, marker="o", label=None, zorder=3):
    """Vertical dot + 95 % interval, white-ring marker, no caps."""
    ax.errorbar(x, y, yerr=[[y - lo], [hi - y]],
                fmt="none", ecolor=color, elinewidth=LW_ERR,
                capsize=0, zorder=zorder)
    ax.plot(x, y, marker=marker, color=color,
            markersize=MS, markeredgecolor="white",
            markeredgewidth=MW, zorder=zorder + 1, label=label)


# ── Forest-plot helpers ───────────────────────────────────────────────
def forest_row(ax, diff, lo, hi, y, *,
               color=BLUE, marker="o", q_val=None, q_x=None, zorder=3):
    """One forest-plot row: dot + interval + optional q annotation.

    q_x: fixed x position for q label (for column alignment);
         if None, places label just right of the interval end.
    """
    dot_interval(ax, diff, y, lo, hi,
                 color=color, marker=marker, zorder=zorder)
    if q_val is not None:
        x = q_x if q_x is not None else max(hi + 0.008, 0.025)
        ax.text(x, y, f"$q$ = {q_val:.3f}", va="center",
                fontsize=SZ_ANNOT, color=INK2)


def forest_zero(ax) -> None:
    """Solid zero baseline at the spine colour."""
    ax.axvline(0, color=SPINE_CLR, lw=SPINE_W, zorder=0)


# ── Legend helper ─────────────────────────────────────────────────────
def instrument_legend(target, instruments: Sequence[str], **kwargs):
    """Shared legend with marker + colour per named instrument."""
    handles = []
    for name in instruments:
        h = Line2D([0], [0], marker=INSTRUMENT_MARKERS[name],
                   color="w",
                   markerfacecolor=INSTRUMENT_COLOURS[name],
                   markeredgecolor="white", markeredgewidth=0.4,
                   markersize=MS * 0.72, label=name)
        handles.append(h)
    kw = dict(fontsize=SZ_ANNOT, frameon=False,
              handletextpad=0.3, columnspacing=0.8)
    kw.update(kwargs)
    return target.legend(handles=handles, **kw)


# ── Save helper ───────────────────────────────────────────────────────
def save(fig, name: str, out_dir: Path | None = None, *,
         tight: bool = False) -> None:
    """Write PDF (vector, TrueType only) + 300 dpi PNG preview.

    Default tight=False preserves the exact figsize (183 mm or 89 mm).
    Use tight=True only for schematic figures with axis('off').
    """
    if out_dir is None:
        out_dir = Path(__file__).resolve().parent.parent / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    preview = out_dir / "preview"
    preview.mkdir(parents=True, exist_ok=True)
    if tight:
        bbox, pad = "tight", mm(1)
    else:
        bbox, pad = None, 0.02
    fig.savefig(out_dir / f"{name}.pdf",
                bbox_inches=bbox, pad_inches=pad)
    fig.savefig(preview / f"{name}.png", dpi=300,
                bbox_inches=bbox, pad_inches=pad)
    plt.close(fig)


# Backward-compat alias used by some call-sites
save_figure = save
