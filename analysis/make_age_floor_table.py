#!/usr/bin/env python3
"""Appendix table for the age floor, the first question a reader asks."""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "artifacts/behavioral_withdrawal_age_floor"
OUT = ROOT / "manuscript/generated"
sys.path.insert(0, str(ROOT / "scripts"))
from postprocess_behavior_outcome_landscape import reml_hk  # noqa: E402

LABEL = {"mortality": "Death", "incident_any_adl": "New ADL limitation",
         "incident_any_iadl": "New IADL limitation",
         "multimorbidity_progression": "Multimorbidity progression"}
TERM = {"loss_1": "One", "loss_2plus": "Two or more"}
DASH = "--"

d = pd.concat([pd.read_csv(f) for f in sorted(SRC.glob("*-age-floor-models.csv"))])

body = []
for outcome in LABEL:
    for term in TERM:
        cells = []
        for floor in [60, 65, 70]:
            g = d[d.outcome_id.eq(outcome) & d.term.eq(term) & d.age_floor.eq(floor)
                  & d.model_status.eq("PASS")].drop_duplicates("cohort")
            if len(g) < 3:
                cells.append("Not estimable")
                continue
            fit = reml_hk(np.log(g.estimate.to_numpy(float)),
                          g.standard_error.to_numpy(float) ** 2)
            cells.append(
                f"{np.exp(fit['pooled']):.2f} ({np.exp(fit['ci_low']):.2f}{DASH}"
                f"{np.exp(fit['ci_high']):.2f})")
        first = LABEL[outcome] if term == "loss_1" else ""
        body.append(" & ".join([first, TERM[term], *cells]) + r" \\")

counts = []
for floor in [60, 65, 70]:
    g = d[d.outcome_id.eq("mortality") & d.term.eq("loss_2plus")
          & d.age_floor.eq(floor) & d.model_status.eq("PASS")].drop_duplicates("cohort")
    counts.append(f"{int(g.n.sum()):,} intervals and {int(g.events.sum()):,} deaths"
                  f" in {len(g)} cohorts at {floor}")

dist = pd.concat([pd.read_csv(f) for f in sorted(SRC.glob("*-age-distribution.csv"))])
dist = dist[dist.eligible_intervals.gt(0)]
share = ", ".join(f"{r.cohort.upper() if r.cohort != 'klosa' else 'KLoSA'} "
                  f"{r.pct_65_plus:.0f}\\%" for _, r in dist.iterrows())

lines = [
    r"\begin{table}[htbp]", r"\centering",
    r"\caption{\textbf{Where the age floor is put.} The protocol takes adults aged "
    r"60 years or older at the interview that identifies the cessation, which is the "
    r"threshold the World Health Organization uses for older people, and the one these "
    r"cohorts are built around. "
    r"Each cell repeats the primary synthesis with the floor raised, on the same "
    r"protocol, covariates and support rules. Cells give the risk ratio with its 95\% "
    r"CI. $I^2$ was zero for death at every floor. The mortality risk set holds "
    + "; ".join(counts) + r".}",
    r"\label{tab:agefloor}", r"\footnotesize", r"\setlength{\tabcolsep}{4pt}",
    r"\begin{tabular}{llccc}", r"\hline",
    r"Outcome & Stopped & 60 years+ & 65 years+ & 70 years+ \\",
    r"\hline", *body, r"\hline", r"\end{tabular}",
    r"\begin{flushleft}\footnotesize The eligible risk set is already old: median age "
    r"at the landmark interview is 66 to 71 years, and the share aged 65 or older is "
    + share + r". Raising the floor to 65 removes ELSA from the mortality synthesis, "
    r"because its group with two or more cessations then falls below the prespecified "
    r"minimum of 20 events, and removes about two fifths of the person-intervals. "
    r"Combinations leaving fewer than three contributing cohorts are reported as not "
    r"estimable rather than fitted on fewer.\end{flushleft}",
    r"\end{table}",
]
dest = OUT / "supp_table_age_floor.tex"
dest.write_text("\n".join(lines) + "\n")
print(f"wrote {dest.relative_to(ROOT)}")
