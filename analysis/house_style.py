#!/usr/bin/env python3
"""Shared figure house style.

Matches the medace_aud project: 7 pt Arial, 600 dpi, no top/right spines,
hairline axes, and the Okabe-Ito colourblind-safe palette. Import this at the
top of every display script so a style change happens in one place.

    from house_style import apply, OKABE, COHORT, save

Encoding discipline: colour carries category only. Magnitude is carried by
position, and series are additionally separated by marker shape so every panel
survives grayscale printing.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

# --- Okabe-Ito, colourblind safe -------------------------------------------
OKABE = {
    "black":     "#000000",
    "orange":    "#E69F00",
    "sky":       "#56B4E9",
    "green":     "#009E73",
    "yellow":    "#F0E442",
    "blue":      "#0072B2",
    "vermilion": "#D55E00",
    "purple":    "#CC79A7",
    "grey":      "#999999",
}

# One vocabulary for the whole paper, four inks. Wong recommends blue and
# vermilion for a two-group contrast, and every contrast here is a two-group
# contrast, so the same pair carries all of them. Cohort detail is context and
# is drawn in grey; the pooled summary is the black foreground. Five saturated
# hues for five cohorts encoded nothing and read as a rainbow.
#
#   vermilion   the heavier exposure, or the nearer horizon
#   blue        the lighter exposure, or the later horizon
#   black       the pooled or any-withdrawal summary
#   grey        cohort detail, null lines, everything in the background
REFERENCE = OKABE["grey"]        # null line, background
ONE_LOSS = OKABE["blue"]         # one withdrawal
TWO_LOSS = OKABE["vermilion"]    # two or more withdrawals
ANY_LOSS = OKABE["black"]        # any withdrawal
RETIREMENT = OKABE["blue"]       # retirement-linked work exit
OTHER_EXIT = OKABE["vermilion"]  # work exit not linked to retirement
NEAR = OKABE["vermilion"]        # immediately following interval
LATER = OKABE["blue"]            # one wave later
POOLED = OKABE["black"]          # random-effects summary
CONTEXT = "#8C8C8C"              # a single cohort line

# Cohort colours come from the four Wong inks that the contrast vocabulary above
# does not already spend. Blue and vermilion are reserved, because Figure 3 puts
# cohort panels next to a panel where blue and vermilion mean one withdrawal and
# two or more; the same ink cannot mean a cohort in one panel and an exposure
# level in the next. Marker shape repeats the distinction so the panels still
# read in grayscale.
#
# CHARLS appears in no colour-coded panel: its comparable-window mortality model
# is not evaluable and it does not contribute to the standardized risks. It is
# assigned the remaining Wong ink, which is the weakest of them on white, so if
# CHARLS ever enters one of these panels the assignment needs revisiting.
COHORT = {
    "charls": OKABE["yellow"],
    "elsa":   OKABE["sky"],
    "hrs":    OKABE["orange"],
    "klosa":  OKABE["green"],
    "share":  OKABE["purple"],
}
COHORT_MARKER = {
    "charls": "v", "elsa": "o", "hrs": "s", "klosa": "^", "share": "D",
}

LABEL = {
    "charls": "CHARLS", "elsa": "ELSA", "hrs": "HRS",
    "klosa": "KLoSA", "mhas": "MHAS", "share": "SHARE",
}

# Journal column widths in inches (BMC Medicine follows the usual 85/183 mm).
SINGLE_COL = 3.35
DOUBLE_COL = 7.20


# Panels are drawn for a 183 mm assembled figure, which is the width BMC sets a
# full-width figure at. Nine point is the normal band for journal figure text at
# that size; going higher looks right on a panel viewed alone and oversized once
# the panels are assembled. The manuscript template renders figures narrower
# than 183 mm, which is handled in the LaTeX, not here.
PANEL_BASE = 9.0


def apply(base: float = 7.0) -> None:
    """Install the house rcParams. Every size derives from base."""
    mpl.rcParams.update({
        "figure.dpi": 600,
        "savefig.dpi": 600,
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": base,
        "axes.titlesize": base + 1,
        "axes.labelsize": base,
        "xtick.labelsize": base - 1,
        "ytick.labelsize": base - 1,
        "legend.fontsize": base - 1,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.6,
        "lines.linewidth": 1.0,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "legend.frameon": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.pad_inches": 0.02,
        "axes.grid": False,
        "pdf.fonttype": 42,   # embed as TrueType so text stays editable
        "ps.fonttype": 42,
    })


def small() -> float:
    """Size for in-panel annotations: one step below body text."""
    return mpl.rcParams["font.size"] - 1.0


def panel_label(ax, letter: str, dx: float = -0.15, dy: float = 1.08,
                ghost: bool = False) -> None:
    """Bold lower-case panel letter, consistent position (Nature convention).

    ghost=True draws it fully transparent. The letter still reports its extent,
    so a panel saved on its own reserves the same corner space, and the letters
    can be set once on the assembled figure instead of five times here.
    """
    ax.text(dx, dy, letter, transform=ax.transAxes,
            fontsize=mpl.rcParams["font.size"] + 1.0, fontweight="bold",
            va="top", ha="right", alpha=0.0 if ghost else 1.0)


def rr_axis(ax, lo: float, hi: float, ticks=None) -> None:
    """Log risk-ratio axis with a null reference line."""
    ax.set_xscale("log")
    ax.axvline(1.0, color=REFERENCE, lw=0.6, ls=(0, (3, 2)), zorder=0)
    if ticks is None:
        ticks = [t for t in (0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0) if lo <= t <= hi]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t:g}" for t in ticks])
    ax.set_xlim(lo, hi)
    ax.minorticks_off()
    ax.set_axisbelow(True)
    # A forest plot has no meaningful left edge; the null line is the reference.
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)


def save(fig, outdir: Path, stem: str, also_png: bool = True,
         tight: bool = True) -> None:
    """Write a figure as PDF and PNG.

    tight=False keeps the saved size exactly equal to figsize. Panels meant for
    hand assembly need that: a tight box crops to the artists, so a panel with
    text outside its axes comes out wider than it was asked to be and stops
    lining up with its neighbours.
    """
    outdir.mkdir(parents=True, exist_ok=True)
    box = "tight" if tight else None
    fig.savefig(outdir / f"{stem}.pdf", bbox_inches=box, facecolor="white")
    if also_png:
        fig.savefig(outdir / f"{stem}.png", dpi=600, bbox_inches=box,
                    facecolor="white")
    plt.close(fig)
