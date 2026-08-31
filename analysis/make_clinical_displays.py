#!/usr/bin/env python3
"""Data figures for the behavioural-cessation manuscript.

Figure 1  The question, the count, and what follows.
          A vector visual abstract: the three activities, how many
          were lost, and the two-year risk of death that followed.

Figure 2  What happens, and how much.
          (a) two-year risk of death by number of activities stopped, per cohort
          (b) the same for new ADL limitation
          (c) adjusted risk ratios across every outcome

Figure 3  Whether it holds up.
          (a) does the reason for leaving work change its meaning
          (b) how far past the change the signal extends

Supplementary Figure S1  Participant flow.

Every plotted value is read from the frozen aggregate CSVs.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np
import pandas as pd
from matplotlib.patches import Ellipse, Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parent))
import house_style as hs

ROOT = Path(__file__).resolve().parents[1]
PILOT = ROOT / "artifacts/multidomain_behavioral_withdrawal_pilot/final"
EXT = ROOT / "artifacts/work_exit_reason_extension/final"
DESC = ROOT / "artifacts/behavioral_withdrawal_frailty_extension/manuscript_descriptive"
OUT = ROOT / "figures/manuscript"

SCOPE = "comparable_22_30_months"
DASH = "–"
SCEN = ["withdrawal_0", "withdrawal_1", "withdrawal_2plus"]
XTICKS = ["None", "One", "Two or more"]


# ---------------------------------------------------------------------------
# Figure 2
# ---------------------------------------------------------------------------

def risk_panel(ax, outcome_id: str, ylab: str, ymax: float) -> None:
    """Standardized two-year risk by number of activities stopped, one line per cohort."""
    c = pd.read_csv(EXT / "cohort_standardized_risks.csv")
    p = pd.read_csv(EXT / "pooled_standardized_risks.csv")
    c = c[c.outcome_id.eq(outcome_id) & c.model_status.eq("PASS")]

    x = np.arange(3)
    present = [k for k in hs.COHORT if k in set(c.cohort)]
    for i, cohort in enumerate(present):
        g = c[c.cohort.eq(cohort)].set_index("scenario")
        if not all(s in g.index for s in SCEN):
            continue
        y = [100 * g.loc[s, "standardized_risk"] for s in SCEN]
        lo = [100 * g.loc[s, "ci_low"] for s in SCEN]
        hi = [100 * g.loc[s, "ci_high"] for s in SCEN]
        # A small horizontal offset per cohort keeps the intervals apart.
        # Filled bands were used here, and four of them overlapping merged into
        # one grey area that read as a single interval.
        xo = x + (i - (len(present) - 1) / 2) * 0.075
        col = hs.COHORT[cohort]
        for xi, a, b in zip(xo, lo, hi):
            ax.plot([xi, xi], [a, b], color=col, lw=0.8, zorder=2,
                    clip_on=False)
        ax.plot(xo, y, color=col, lw=1.2, marker=hs.COHORT_MARKER[cohort],
                ms=3.6, markerfacecolor="white", markeredgewidth=1.0, zorder=3,
                clip_on=False)
        ax.annotate(hs.LABEL[cohort], xy=(2.30, y[2]), xytext=(0, 0),
                    textcoords="offset points", color=col,
                    fontsize=hs.small(), va="center", ha="left",
                    annotation_clip=False)

    pooled = p[p.outcome_id.eq(outcome_id)].set_index("scenario")
    if all(s in pooled.index for s in SCEN):
        yp = [100 * pooled.loc[s, "pooled_standardized_risk"] for s in SCEN]
        ax.plot(x, yp, color=hs.POOLED, lw=1.8, ls=(0, (3.6, 1.5)), zorder=4,
                clip_on=False)
        ax.legend(handles=[plt.Line2D([], [], color=hs.POOLED, lw=1.8,
                                      ls=(0, (3.6, 1.5)), label="Pooled")],
                  loc="upper left", handlelength=1.8, handletextpad=0.5,
                  borderaxespad=0.15)


    ax.set_xticks(x)
    ax.set_xticklabels(XTICKS)
    ax.set_xlim(-0.34, 2.34)
    ax.set_ylim(0, ymax)
    ax.set_yticks(np.arange(0, ymax + 0.1, 5.0))
    ax.set_xlabel("Activities recently stopped")
    ax.set_ylabel(ylab)
    ax.spines["left"].set_bounds(0, ymax)


def forest_panel(ax) -> None:
    s = pd.read_csv(PILOT / "cross-cohort-summary.csv")

    graded = [
        ("mortality", "Death"),
        ("incident_any_adl", "New ADL limitation"),
        ("incident_any_iadl", "New IADL limitation"),
        ("multimorbidity_progression", "Multimorbidity progression"),
    ]
    OFF = 0.21
    y = 0.0
    marks: list[tuple] = []
    ylabels: list[tuple] = []

    for oid, label in graded:
        for term, colour, marker, face, off in (
            ("loss_1", hs.ONE_LOSS, "o", "white", +OFF),
            ("loss_2plus", hs.TWO_LOSS, "s", hs.TWO_LOSS, -OFF),
        ):
            q = s[s.exposure_model.eq("score_categorical")
                  & s.outcome_id.eq(oid) & s.term.eq(term)]
            if not len(q):
                continue
            r = q.iloc[0]
            marks.append((y + off, r.pooled_estimate, r.ci_low, r.ci_high,
                          int(r.k_cohorts), colour, marker, face))
        ylabels.append((y, label))
        y -= 1.0

    hs.rr_axis(ax, 0.8, 4.35, ticks=[1.0, 1.5, 2.0, 3.0, 4.0])
    trans = mtransforms.blended_transform_factory(ax.transAxes, ax.transData)

    for yy, est, lo, hi, k, colour, marker, face in marks:
        ax.plot([lo, hi], [yy, yy], color=colour, lw=1.0, solid_capstyle="butt",
                zorder=2)
        ax.plot([est], [yy], marker=marker, ms=3.6, color=colour,
                markerfacecolor=face, markeredgecolor=colour, mew=0.8, zorder=3)
        ax.text(1.020, yy, f"{est:.2f} ({lo:.2f}{DASH}{hi:.2f})", transform=trans,
                fontsize=hs.small(), va="center", ha="left", color="#1A1A1A")
        ax.text(1.310, yy, str(k), transform=trans, fontsize=hs.small(), va="center",
                ha="center", color="#1A1A1A")

    ax.set_yticks([p for p, _ in ylabels])
    ax.set_yticklabels([t for _, t in ylabels])
    ax.set_ylim(y + 0.62, 0.82)
    ax.set_xlabel("Adjusted risk ratio (95% CI)")
    ax.text(1.020, 0.70, "RR (95% CI)", transform=trans, fontsize=hs.small(),
            ha="left", color="#1A1A1A")
    ax.text(1.310, 0.70, "Cohorts", transform=trans, fontsize=hs.small(),
            ha="center", color="#1A1A1A")

    handles = [
        plt.Line2D([], [], marker="o", ls="none", ms=3.6, color=hs.ONE_LOSS,
                   markerfacecolor="white", mew=0.8, label="One stopped"),
        plt.Line2D([], [], marker="s", ls="none", ms=3.6, color=hs.TWO_LOSS,
                   label="Two or more stopped"),
        plt.Line2D([], [], marker="D", ls="none", ms=3.2, color=hs.ANY_LOSS,
                   label="Any cessation"),
    ]
    ax.legend(handles=handles, loc="upper center", ncol=3,
              bbox_to_anchor=(0.5, -0.155), handletextpad=0.4,
              borderaxespad=0.0, columnspacing=1.6)


def work_exit_panel(ax) -> None:
    m = pd.read_csv(PILOT / "systematic-results-matrix.csv")
    q = m[m.scope.eq(SCOPE) & m.adjustment.eq("full")
          & m.exposure_model.eq("work_exit_phenotype")
          & m.model_status.eq("PASS")].drop_duplicates(
              ["cohort", "outcome_id", "term"])

    wanted = [
        ("hrs", "mortality", "HRS, death"),
        ("hrs", "incident_any_adl", "HRS, new ADL"),
        ("elsa", "incident_any_adl", "ELSA, new ADL"),
    ]
    ylabels, y = [], 0.0
    for cohort, oid, label in wanted:
        for term, colour, marker, face, off in (
            ("work_exit_retirement", hs.RETIREMENT, "o", "white", +0.16),
            ("work_exit_no_retirement", hs.OTHER_EXIT, "s", hs.OTHER_EXIT, -0.16),
        ):
            r = q[q.cohort.eq(cohort) & q.outcome_id.eq(oid) & q.term.eq(term)]
            if not len(r):
                continue
            r = r.iloc[0]
            yy = y + off
            ax.plot([r.ci_low, r.ci_high], [yy, yy], color=colour, lw=1.0,
                    zorder=2)
            ax.plot([r.estimate], [yy], marker=marker, ms=3.6, color=colour,
                    markerfacecolor=face, markeredgecolor=colour, mew=0.8,
                    zorder=3)
        ylabels.append((y, label))
        y -= 1.0

    hs.rr_axis(ax, 0.80, 3.15, ticks=[1.0, 1.5, 2.0, 2.5, 3.0])
    ax.set_yticks([p for p, _ in ylabels])
    ax.set_yticklabels([t for _, t in ylabels])
    ax.set_xlabel("Adjusted risk ratio versus continued work")
    ax.set_ylim(y + 0.55, 0.45)

    handles = [
        plt.Line2D([], [], marker="o", ls="none", ms=3.6, color=hs.RETIREMENT,
                   markerfacecolor="white", mew=0.8,
                   label="Retirement-linked exit"),
        plt.Line2D([], [], marker="s", ls="none", ms=3.6, color=hs.OTHER_EXIT,
                   label="Other work exit"),
    ]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.20),
              handletextpad=0.4, borderaxespad=0.0, labelspacing=0.3)


def figure_2() -> None:
    fig = plt.figure(figsize=(hs.DOUBLE_COL, 5.15))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.42],
                          hspace=0.40, wspace=0.46,
                          left=0.085, right=0.775, top=0.955, bottom=0.115)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, :])
    risk_panel(ax_a, "mortality", "Two-year risk of death (%)", 18.0)
    risk_panel(ax_b, "incident_any_adl",
               "Two-year risk of new\nADL limitation (%)", 24.0)
    forest_panel(ax_c)
    hs.panel_label(ax_a, "a", dx=-0.22, dy=1.13)
    hs.panel_label(ax_b, "b", dx=-0.26, dy=1.13)
    hs.panel_label(ax_c, "c", dx=-0.28, dy=1.08)
    hs.save(fig, OUT, "figure_2_multidomain_clinical_results")
    print("wrote figure_2_multidomain_clinical_results")


def figure_3() -> None:
    fig = plt.figure(figsize=(hs.DOUBLE_COL * 0.40, 2.60))
    gs = fig.add_gridspec(1, 1, left=0.37, right=0.97, top=0.94, bottom=0.315)
    work_exit_panel(fig.add_subplot(gs[0, 0]))
    hs.save(fig, OUT, "figure_3_multidomain_specificity")
    print("wrote figure_3_multidomain_specificity")


# ---------------------------------------------------------------------------
# Participant flow
# ---------------------------------------------------------------------------

STAGES = [
    ("source_respondents",
     "Respondents in the six harmonized ageing cohorts"),
    ("age_60_three_wave_intervals",
     "Aged 60 years or older with three consecutive interviews"),
    ("comparable_22_30_month_intervals",
     "Outcome interview scheduled 22 to 30 months after the second interview"),
    ("three_behaviors_observed",
     "All three activities observed at the first and second interview"),
    ("primary_behavior_risk_set",
     "At least one activity present initially, with complete model covariates"),
]


def flow_figure() -> None:
    flow = pd.read_csv(DESC / "manuscript-flow.csv")
    mat = pd.read_csv(PILOT / "systematic-results-matrix.csv")
    contrib = mat[mat.scope.eq(SCOPE) & mat.adjustment.eq("full")
                  & mat.exposure_model.eq("any_withdrawal")
                  & mat.term.eq("any_withdrawal")
                  & mat.model_status.eq("PASS")].drop_duplicates(
                      ["cohort", "outcome_id"])

    fig, ax = plt.subplots(figsize=(5.6, 5.9))
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    x, w, h = 0.020, 0.625, 0.112
    tops = np.linspace(0.975, 0.375, len(STAGES))

    prev_bottom = None
    for i, ((stage, title), top) in enumerate(zip(STAGES, tops)):
        rows = flow[flow.stage.eq(stage)]
        people = int(rows.people.sum())
        iv = rows.intervals.sum()
        intervals = None if pd.isna(iv) or iv == 0 else int(iv)

        final = stage == "primary_behavior_risk_set"
        edge = hs.OKABE["vermilion"] if final else hs.OKABE["blue"]
        face = "#FDF2EA" if final else "#EDF4FA"
        ax.add_patch(plt.Rectangle((x, top - h), w, h, transform=ax.transAxes,
                                   facecolor=face, edgecolor=edge, lw=0.8,
                                   zorder=2))
        ax.text(x + 0.018, top - 0.026, title, transform=ax.transAxes,
                fontsize=7.9, va="top", ha="left", color="#1A1A1A")
        count = (f"{intervals:,} person-intervals from {people:,} participants"
                 if intervals else f"{people:,} participants")
        ax.text(x + 0.018, top - h + 0.022, count, transform=ax.transAxes,
                fontsize=7.9, va="bottom", ha="left", fontweight="bold",
                color=edge)

        if prev_bottom is not None:
            mid = (top + prev_bottom) / 2
            ax.annotate("", xy=(x + w / 2, top), xytext=(x + w / 2, prev_bottom),
                        xycoords=ax.transAxes, textcoords=ax.transAxes,
                        arrowprops=dict(arrowstyle="-|>", lw=0.7,
                                        color="#7A7A7A", shrinkA=0, shrinkB=0))
            pr = flow[flow.stage.eq(STAGES[i - 1][0])].intervals.sum()
            if intervals is not None and not pd.isna(pr) and pr > 0:
                drop = int(pr) - intervals
                if drop > 0:
                    ax.annotate("", xy=(x + w + 0.045, mid),
                                xytext=(x + w / 2, mid),
                                xycoords=ax.transAxes, textcoords=ax.transAxes,
                                arrowprops=dict(arrowstyle="-|>", lw=0.6,
                                                color="#A0A0A0"))
                    ax.text(x + w + 0.055, mid,
                            f"{drop:,} intervals\nexcluded",
                            transform=ax.transAxes, fontsize=7.4, va="center",
                            ha="left", color="#5A5A5A")
        prev_bottom = top - h

    y0 = 0.205
    outcomes = [("mortality", "Death"),
                ("incident_any_adl", "New ADL limitation"),
                ("incident_any_iadl", "New IADL limitation")]
    bw = 0.285
    for j, (oid, label) in enumerate(outcomes):
        g = contrib[contrib.outcome_id.eq(oid)]
        n, ev, k = int(g.n.sum()), int(g.events.sum()), len(g)
        xx = 0.030 + j * (bw + 0.028)
        ax.add_patch(plt.Rectangle((xx, y0 - 0.135), bw, 0.135,
                                   transform=ax.transAxes, facecolor="#F5F7F9",
                                   edgecolor="#7A7A7A", lw=0.7, zorder=2))
        ax.text(xx + bw / 2, y0 - 0.014, label, transform=ax.transAxes,
                fontsize=8.2, ha="center", va="top", fontweight="bold",
                color="#1A1A1A")
        ax.text(xx + bw / 2, y0 - 0.128,
                f"{n:,} intervals\n{ev:,} events\n{k} cohorts",
                transform=ax.transAxes, fontsize=7.5, ha="center", va="bottom",
                color="#3A3A3A")
        ax.annotate("", xy=(xx + bw / 2, y0), xytext=(x + w / 2, tops[-1] - h),
                    xycoords=ax.transAxes, textcoords=ax.transAxes,
                    arrowprops=dict(arrowstyle="-|>", lw=0.6, color="#7A7A7A"))

    ax.text(0.5, 0.008,
            "MHAS had no outcome interview scheduled within the 22 to 30 month "
            "window\nand contributed to sensitivity analyses only.",
            transform=ax.transAxes, fontsize=7.5, ha="center", va="bottom",
            color="#5A5A5A")

    hs.save(fig, OUT, "supp_figure_s1_participant_flow")
    print("wrote supp_figure_s1_participant_flow")


# ---------------------------------------------------------------------------
# One file per panel, for hand assembly
# ---------------------------------------------------------------------------
# Assemble at 1:1 and do not resize individual panels: every panel is drawn at
# the same absolute type size, so scaling one of them breaks the match. The two
# figures are laid out for a 183 mm assembled width.
#
#   Fig. 2   specificity_a | specificity_b               side by side
#   Fig. 3   outcomes_a | outcomes_b  over  outcomes_c    two up, one across
#
# Panel letters are drawn transparent. They reserve the corner space so the
# panels align, and the visible letters go on once, on the assembled figure.

PANELS = ROOT / "figures/panels"

# stem, (width, height), letter, (left, right, bottom, top), draw
# Widths are exact and add up: 3.60 + 3.60 across the top, 7.20 below, so the
# assembled figure is 183 mm without any panel being resized.
PANEL_SPECS = (
    ("panel_fig3_a_work_exit_reason", (3.60, 3.20), "a",
     (0.27, 0.97, 0.26, 0.95), work_exit_panel),
    ("panel_fig2_a_two_year_death_risk", (3.60, 3.00), "a",
     (0.20, 0.82, 0.17, 0.96),
     lambda ax: risk_panel(ax, "mortality", "Two-year risk of death (%)", 18.0)),
    ("panel_fig2_b_two_year_adl_risk", (3.60, 3.00), "b",
     (0.24, 0.82, 0.17, 0.96),
     lambda ax: risk_panel(ax, "incident_any_adl",
                           "Two-year risk of new\nADL limitation (%)", 24.0)),
    # The forest was 7.2 by 3.4, which stretched nine rows across a very long
    # axis and left the type small against the width. Taller, same width, and
    # the right margin now holds the estimate column inside the panel.
    ("panel_fig2_c_outcome_gradient", (7.20, 4.20), "c",
     (0.24, 0.76, 0.17, 0.96), forest_panel),
)


def panels() -> None:
    """Write every panel on its own, at panel type size and exact size."""
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


# ---------------------------------------------------------------------------
# Figure 1  The count and the risk that follows
# ---------------------------------------------------------------------------
# One chart, not a panel of boxes. The three activities are named once as a key,
# and the finding is a single bar chart with a real axis. Every plotted value is
# read from the frozen aggregate CSVs, so the figure cannot drift from the
# abstract. Confidence intervals are deliberately absent: the pooled interval on
# a standardized absolute risk is dominated by the severalfold difference in
# baseline mortality between cohorts, not by uncertainty about the gradient, and
# it belongs with the cohort detail in Fig. 2a and Supplementary Table S6.

NAVY = "#12355B"
FOOTER = "#E2E8ED"
VA_W = hs.DOUBLE_COL
VA_H = hs.DOUBLE_COL * 2 / 3
ASPECT = VA_W / VA_H


def _pt(cx, cy, dx, dy, h):
    """A point offset from (cx, cy) by (dx, dy) in units of the icon height."""
    return cx + dx * h / ASPECT, cy + dy * h


def _stroke(ax, pts, colour, lw=1.1):
    xs, ys = zip(*pts)
    ax.plot(xs, ys, color=colour, lw=lw, solid_capstyle="round",
            solid_joinstyle="round", zorder=3)


def _disc(ax, x, y, r, face, edge, lw=0.8):
    ax.add_patch(Ellipse((x, y), width=2 * r / ASPECT, height=2 * r,
                         facecolor=face, edgecolor=edge, lw=lw, zorder=3))


def _icon_glass(ax, cx, cy, h, colour):
    """A tapered tumbler, part filled."""
    top, bot = 0.30, 0.21
    _stroke(ax, [_pt(cx, cy, -top, 0.46, h), _pt(cx, cy, -bot, -0.46, h),
                 _pt(cx, cy, bot, -0.46, h), _pt(cx, cy, top, 0.46, h)], colour)
    ax.add_patch(Ellipse(_pt(cx, cy, 0, 0.46, h), width=2 * top * h / ASPECT,
                         height=0.13 * h, facecolor="none", edgecolor=colour,
                         lw=1.1, zorder=3))
    fill = 0.27
    span = bot + (top - bot) * (fill + 0.46) / 0.92
    _stroke(ax, [_pt(cx, cy, -span, fill, h), _pt(cx, cy, span, fill, h)],
            colour, lw=0.8)


def _icon_walker(ax, cx, cy, h, colour):
    """A walking figure, mid stride, facing right."""
    _disc(ax, *_pt(cx, cy, 0.00, 0.40, h), 0.100 * h, "none", colour, lw=1.1)
    _stroke(ax, [_pt(cx, cy, 0.00, 0.29, h), _pt(cx, cy, 0.02, 0.02, h)], colour)
    _stroke(ax, [_pt(cx, cy, 0.02, 0.02, h), _pt(cx, cy, 0.20, -0.15, h),
                 _pt(cx, cy, 0.20, -0.44, h)], colour)
    _stroke(ax, [_pt(cx, cy, 0.02, 0.02, h), _pt(cx, cy, -0.13, -0.19, h),
                 _pt(cx, cy, -0.27, -0.44, h)], colour)
    _stroke(ax, [_pt(cx, cy, 0.01, 0.24, h), _pt(cx, cy, 0.16, 0.13, h),
                 _pt(cx, cy, 0.14, -0.02, h)], colour)
    _stroke(ax, [_pt(cx, cy, 0.01, 0.24, h), _pt(cx, cy, -0.16, 0.07, h)], colour)


def _icon_case(ax, cx, cy, h, colour):
    """A briefcase with a squared handle and a clasp."""
    _stroke(ax, [_pt(cx, cy, -0.44, 0.22, h), _pt(cx, cy, -0.44, -0.34, h),
                 _pt(cx, cy, 0.44, -0.34, h), _pt(cx, cy, 0.44, 0.22, h),
                 _pt(cx, cy, -0.44, 0.22, h)], colour)
    _stroke(ax, [_pt(cx, cy, -0.44, 0.00, h), _pt(cx, cy, 0.44, 0.00, h)],
            colour, lw=0.8)
    _stroke(ax, [_pt(cx, cy, -0.15, 0.22, h), _pt(cx, cy, -0.15, 0.40, h),
                 _pt(cx, cy, 0.15, 0.40, h), _pt(cx, cy, 0.15, 0.22, h)], colour)
    _stroke(ax, [_pt(cx, cy, -0.07, -0.06, h), _pt(cx, cy, -0.07, 0.06, h),
                 _pt(cx, cy, 0.07, 0.06, h), _pt(cx, cy, 0.07, -0.06, h),
                 _pt(cx, cy, -0.07, -0.06, h)], colour, lw=0.8)


def visual_abstract() -> None:
    """Figure 1. The count, and the two-year risk of death that follows it."""
    p = pd.read_csv(EXT / "pooled_standardized_risks.csv")
    m = p[p.outcome_id.eq("mortality")].set_index("scenario")
    pct = [100 * m.loc[s, "pooled_standardized_risk"] for s in SCEN]

    # The figure shows the mortality gradient, so it is described by the mortality
    # risk set. The study-wide count of 84,924 across five cohorts includes
    # CHARLS, whose comparable-window mortality model is not evaluable.
    mat = pd.read_csv(PILOT / "systematic-results-matrix.csv")
    risk = mat[mat.scope.eq(SCOPE) & mat.adjustment.eq("full")
               & mat.exposure_model.eq("score_categorical")
               & mat.outcome_id.eq("mortality")
               & mat.model_status.eq("PASS")].drop_duplicates(["cohort"])
    people = int(risk.people.sum())
    cohorts = int(risk.cohort.nunique())
    deaths = int(risk.events.sum())

    blue, verm, grey = hs.OKABE["blue"], hs.OKABE["vermilion"], hs.OKABE["grey"]
    fig = plt.figure(figsize=(VA_W, VA_H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # --- title bar and footer strip ---
    ax.add_patch(Rectangle((0, 0.870), 1, 0.130, facecolor=NAVY, lw=0))
    ax.add_patch(Rectangle((0, 0), 1, 0.075, facecolor=FOOTER, lw=0))
    ax.text(0.060, 0.958, "Recent cessation of everyday activities and "
            "two-year risk of death", color="white", fontsize=11.5,
            fontweight="bold", va="center", ha="left")
    ax.text(0.060, 0.898, f"{people:,} adults aged 60 years or older in "
            f"{_word(cohorts)} international ageing cohorts, {deaths:,} deaths",
            color="white", fontsize=8.5, va="center", ha="left")
    ax.text(0.5, 0.037, "Ask what stopped, when, and why", color=NAVY,
            fontsize=9.5, fontweight="bold", va="center", ha="center")

    # --- what the count is made of, named once ---
    ax.text(0.060, 0.826, "Three everyday activities, counted between two "
            "consecutive interviews about two years apart",
            color="black", fontsize=8, va="center", ha="left")
    key = ((_icon_glass, "Stopped drinking", 0.088),
           (_icon_walker, "Stopped regular activity", 0.398),
           (_icon_case, "Left paid work", 0.726))
    for draw, name, x in key:
        draw(ax, x, 0.752, 0.070, blue)
        ax.text(x + 0.036, 0.752, name, color="black", fontsize=8.5,
                va="center", ha="left")
    ax.plot([0.060, 0.940], [0.690, 0.690], color=grey, lw=0.5)

    # --- the finding, as one chart ---
    base, scale = 0.235, 0.0370
    axis_x, axis_top = 0.185, 0.235 + 10 * scale
    ax.plot([axis_x, axis_x], [base, axis_top], color="black", lw=0.6)
    ax.plot([axis_x, 0.815], [base, base], color="black", lw=0.6)
    for tick in (0, 5, 10):
        y = base + tick * scale
        ax.plot([axis_x - 0.008, axis_x], [y, y], color="black", lw=0.6)
        ax.text(axis_x - 0.016, y, str(tick), color="black", fontsize=7.5,
                va="center", ha="right")
    ax.text(0.098, (base + axis_top) / 2, "Two-year risk of death (%)",
            color="black", fontsize=8.5, va="center", ha="center", rotation=90)

    # Each cohort is drawn on its own bar. The pooled bar alone reads as a single
    # certain number; the four points show that baseline risk differs severalfold
    # between cohorts while the ordering does not.
    cohort_risk = pd.read_csv(EXT / "cohort_standardized_risks.csv")
    cohort_risk = cohort_risk[cohort_risk.outcome_id.eq("mortality")
                              & cohort_risk.model_status.eq("PASS")]

    centres = (0.330, 0.530, 0.730)
    for x, value, label, scen in zip(centres, pct, XTICKS, SCEN):
        ax.add_patch(Rectangle((x - 0.065, base), 0.130, value * scale,
                               facecolor=verm, lw=0, zorder=3))
        ax.text(x, base + value * scale + 0.020, f"{value:.1f}%", color=verm,
                fontsize=14, fontweight="bold", va="bottom", ha="center")
        ax.text(x, 0.186, label, color="black", fontsize=8.5,
                va="center", ha="center")
    ax.text(0.530, 0.126, "Everyday activities recently stopped", color="black",
            fontsize=8.5, va="center", ha="center")

    near, far = round(100 / pct[-1]), round(100 / pct[0] / 5) * 5
    ax.text(0.222, 0.552, f"About 1 in {near} versus 1 in {far}", color=NAVY,
            fontsize=10.5, fontweight="bold", va="center", ha="left")
    spread = {}
    for scen in SCEN:
        v = sorted(100 * x for x in
                   cohort_risk[cohort_risk.scenario.eq(scen)].standardized_risk)
        spread[scen] = (v[0], v[-1])
    ax.text(0.530, 0.086,
            f"Across cohorts: {spread[SCEN[0]][0]:.1f}\u2013{spread[SCEN[0]][1]:.1f}% "
            f"with none, {spread[SCEN[2]][0]:.1f}\u2013{spread[SCEN[2]][1]:.1f}% "
            "with two or more", color=NAVY, fontsize=7.6,
            va="center", ha="center")

    hs.save(fig, OUT, "figure_1_visual_abstract")
    print(f"figure 1: {pct[0]:.1f} / {pct[1]:.1f} / {pct[2]:.1f} pooled | "
          f"mortality risk set {people:,} people, {deaths:,} deaths, "
          f"{cohorts} cohorts")


def _word(n: int) -> str:
    return {3: "three", 4: "four", 5: "five", 6: "six"}.get(n, str(n))


if __name__ == "__main__":
    hs.apply()
    visual_abstract()
    flow_figure()
    panels()
    # figure_2() and figure_3() composed the two multi-panel figures here. The
    # panels are now assembled by analysis/assemble_figures.py, which places each
    # one at its exact size and keeps the result vector, so the auto-composed
    # versions are no longer written. Both functions are kept for reference.
