#!/usr/bin/env python3
"""Panels for the figure on what a recent cessation is compared against.

Two panels at the size Figure 3's panels use, written separately so they can be
assembled by analysis/assemble_figures.py or by hand. Panel letters are drawn
transparent: they reserve the corner space and are set once at assembly.

Every value is read from the frozen synthesis at draw time; none is typed here.
"""
from __future__ import annotations

import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
import house_style as hs  # noqa: E402

SRC = ROOT / "artifacts/behavioral_withdrawal_competing_context/final"
PANELS = ROOT / "figures/panels"

d = pd.read_csv(SRC / "reference-group-synthesis.csv")
d = d[d.outcome_id.eq("mortality")]


def get(model: str, domain: str, term: str) -> tuple[float, float, float]:
    row = d[d.model_id.eq(model) & d.domain.eq(domain) & d.term.eq(term)]
    if len(row) != 1:
        sys.exit(f"expected one row for {model}/{domain}/{term}, found {len(row)}")
    r = row.iloc[0]
    return float(r.pooled_estimate), float(r.ci_low), float(r.ci_high)


def rows(ax, n: int) -> np.ndarray:
    """Evenly spaced rows filling the panel, whatever the row count.

    A panel with three rows and a panel with four should each use their whole
    height; matching the row pitch across panels leaves one of them with a
    hole in it.
    """
    ax.set_ylim(-0.62, n - 0.38)
    return np.arange(n)[::-1]


# ------------------------------------------------------------------ panel a
STATES = [
    ("Never active", None, hs.REFERENCE, "o", False),
    ("Stopped", "activity_stopped", hs.TWO_LOSS, "s", True),
    ("Started", "activity_started", hs.REFERENCE, "^", False),
    ("Continued", "activity_continued", hs.ONE_LOSS, "o", True),
]


def transition_panel(ax) -> None:
    ys = rows(ax, len(STATES))
    for y, (label, term, colour, marker, filled) in zip(ys, STATES):
        if term is None:
            est = 1.0
        else:
            est, lo, hi = get("four_state_never_had_reference", "activity", term)
            ax.plot([lo, hi], [y, y], color=colour, lw=1.4,
                    solid_capstyle="butt", zorder=2)
        ax.plot([est], [y], marker=marker, ms=5.0, color=colour, zorder=3,
                mfc=colour if filled else "white", mew=1.1, mec=colour)
    ax.set_yticks(ys)
    ax.set_yticklabels([s[0] for s in STATES])
    hs.rr_axis(ax, 0.34, 1.13, ticks=[0.4, 0.6, 0.8, 1.0])
    ax.set_xlabel("Risk ratio for death,\nagainst never regularly active")


# ------------------------------------------------------------------ panel b
DOMAINS = [
    ("Regular activity", "activity", "activity_loss"),
    ("Paid work", "work", "work_loss"),
    ("Drinking", "alcohol", "alcohol_loss"),
]


def contrast_panel(ax) -> None:
    ys = rows(ax, len(DOMAINS))
    for y, (label, domain, term) in zip(ys, DOMAINS):
        est, lo, hi = get("restricted_stopped_vs_continued", domain, term)
        ax.plot([lo, hi], [y, y], color=hs.TWO_LOSS, lw=1.4,
                solid_capstyle="butt", zorder=2)
        ax.plot([est], [y], marker="s", ms=5.0, color=hs.TWO_LOSS, zorder=3)
    ax.set_yticks(ys)
    ax.set_yticklabels([row[0] for row in DOMAINS])
    hs.rr_axis(ax, 0.94, 2.55, ticks=[1.0, 1.5, 2.0, 2.5])
    ax.set_xlabel("Risk ratio for death,\nstopped against continued")


# Same width and height as Figure 3's panels, so the two figures sit at the same
# scale on the page.
PANEL_SPECS = (
    ("panel_reference_a_transition_states", (3.60, 3.20), "a",
     (0.26, 0.98, 0.24, 0.94), transition_panel),
    ("panel_reference_b_stopped_vs_continued", (3.60, 3.20), "b",
     (0.26, 0.98, 0.24, 0.94), contrast_panel),
)


def main() -> None:
    hs.apply(base=hs.PANEL_BASE)
    for stem, size, letter, margins, draw in PANEL_SPECS:
        left, right, bottom, top = margins
        fig, ax = plt.subplots(figsize=size)
        fig.subplots_adjust(left=left, right=right, bottom=bottom, top=top)
        draw(ax)
        hs.panel_label(ax, letter, dx=-0.02, dy=1.06, ghost=True)
        hs.save(fig, PANELS, stem, tight=False)
        print(f"wrote panel {stem}  {size[0]:.2f} x {size[1]:.2f} in")
    hs.apply()


if __name__ == "__main__":
    main()
