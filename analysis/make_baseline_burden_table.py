#!/usr/bin/env python3
"""Appendix table: who the people who stopped were, before anything stopped."""
from __future__ import annotations

import pathlib

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "artifacts/behavioral_withdrawal_baseline_burden"
OUT = ROOT / "manuscript/generated"

NAME = {"charls": "CHARLS", "elsa": "ELSA", "hrs": "HRS", "klosa": "KLoSA",
        "share": "SHARE"}
GROUP = {"none": "None", "one": "One", "two_or_more": "Two or more"}
DASH = "--"

d = pd.concat([pd.read_csv(f) for f in sorted(SRC.glob("*-baseline-burden.csv"))])
d = d[d.n_intervals.gt(0)]

ROWS = [
    ("conditions_at_first_interview", "Conditions at first interview", "{:.2f}"),
    ("conditions_at_landmark", "Conditions at the landmark interview", "{:.2f}"),
    ("conditions_gained_across_window", "Conditions gained across the window", "{:.2f}"),
    ("frailty_index_at_landmark", "Frailty index at the landmark interview", "{:.3f}"),
    ("frailty_index_change", "Frailty index change across the window", "{:.3f}"),
    ("self_rated_health_worsening", "Self-rated health worsened, \\%", "{:.1f}"),
    ("transition_hospitalization", "Hospital admission in the window, \\%", "{:.1f}"),
]

body = []
for cohort in NAME:
    block = d[d.cohort.eq(cohort)]
    if block.empty:
        continue
    counts = [f"{int(block[block.cessation_group.eq(g)].n_intervals.iloc[0]):,}"
              for g in GROUP]
    body.append(" & ".join([NAME[cohort], "Person-intervals", *counts]) + r" \\")
    for field, label, fmt in ROWS:
        if field not in block or block[field].isna().all():
            continue
        cells = []
        for g in GROUP:
            row = block[block.cessation_group.eq(g)]
            value = row[field].iloc[0] if not row.empty else None
            cells.append(DASH if value is None or pd.isna(value) else fmt.format(value))
        body.append(" & ".join(["", label, *cells]) + r" \\")
    body.append(r"\addlinespace")

lines = [
    r"\begin{table}[htbp]", r"\centering",
    r"\caption{\textbf{Health at and before the interview that identifies the "
    r"cessation.} Respondents are grouped by how many of the three everyday "
    r"activities they had recently stopped. The first row of each block is the "
    r"disease burden already carried at the first interview, before the window in "
    r"which the activity stopped; the third is what accumulated during that window. "
    r"The frailty index is the 26-item deficit count and is recorded in four "
    r"cohorts; self-rated health and hospital admission are recorded in the cohorts "
    r"shown. Every model in this study adjusts for the condition count and for "
    r"outcome-specific baseline status.}",
    r"\label{tab:burden}", r"\footnotesize", r"\setlength{\tabcolsep}{4pt}",
    r"\begin{tabular}{llrrr}", r"\hline",
    r"Cohort & Measure & None & One & Two or more \\",
    r"\hline", *body, r"\hline", r"\end{tabular}",
    r"\begin{flushleft}\footnotesize Values are means unless a percentage is "
    r"stated. MHAS has no interval in the comparable window and does not appear. "
    r"CHARLS has no comparable self-rated health item and ELSA no comparable hospital "
    r"admission item; KLoSA records neither. Where a cohort has no comparable item "
    r"the cell is empty.\end{flushleft}",
    r"\end{table}",
]
dest = OUT / "supp_table_baseline_burden.tex"
dest.write_text("\n".join(lines) + "\n")
print(f"wrote {dest.relative_to(ROOT)}")
