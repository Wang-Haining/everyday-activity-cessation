#!/usr/bin/env python3
"""Supplementary table: the count restricted to respondents who could lose two.

Two or more withdrawals can only be observed in someone who had at least two
activities at the first interview. Every model already adjusts for how many were
available to lose; this restricts the risk set instead, so that every respondent
could in principle have reached every level of the count.

Reads only frozen aggregate model output. No respondent-level data.
"""

from __future__ import annotations

import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from postprocess_behavior_outcome_landscape import reml_hk  # noqa: E402

PILOT = ROOT / "artifacts/multidomain_behavioral_withdrawal_pilot"
GEN = ROOT / "manuscript/generated"
SCOPE = "comparable_22_30_months"
LABEL = {"elsa": "ELSA", "hrs": "HRS", "share": "SHARE", "klosa": "KLoSA", "charls": "CHARLS"}
OUTCOMES = [("mortality", "Death"), ("incident_any_adl", "New ADL limitation")]
TERMS = [("loss_1", "One activity"), ("loss_2plus", "Two or more")]


def restricted() -> pd.DataFrame:
    frames = [pd.read_csv(f) for f in glob.glob(str(PILOT / "opportunity2/*-models.csv"))]
    d = pd.concat(frames, ignore_index=True)
    return d[d.scope.eq(SCOPE) & d.adjustment.eq("full")
             & d.exposure_model.eq("score_categorical")
             & d.model_status.eq("PASS")
             & d.minimum_baseline_opportunities.eq(2)].drop_duplicates(
                 ["cohort", "outcome_id", "term"])


def primary() -> pd.DataFrame:
    s = pd.read_csv(PILOT / "final/cross-cohort-summary.csv")
    return s[s.exposure_model.eq("score_categorical")]


def pooled(rows: pd.DataFrame) -> tuple[float, float, float, int]:
    yi = np.log(rows.estimate.to_numpy(dtype=float))
    se = (np.log(rows.ci_high.to_numpy(dtype=float))
          - np.log(rows.ci_low.to_numpy(dtype=float))) / (2 * 1.959963985)
    fit = reml_hk(yi, se ** 2)
    return (float(np.exp(fit["pooled"])), float(np.exp(fit["ci_low"])),
            float(np.exp(fit["ci_high"])), len(rows))


def cell(estimate: float, low: float, high: float) -> str:
    return f"{estimate:.2f} ({low:.2f}--{high:.2f})"


def main() -> None:
    r, p = restricted(), primary()
    lines = [
        r"\begin{table*}[htbp]",
        r"\caption{Activities recently stopped and outcome among respondents who had at least "
        r"two activities available to lose}",
        r"\label{tab:opportunity}",
        r"\centering\footnotesize",
        r"\begin{tabular}{@{}llccc@{}}",
        r"\toprule",
        r"Outcome & Activities & Cohort & \multicolumn{2}{c}{RR (95\% CI)} \\",
        r"\cmidrule(l){4-5}",
        r" & stopped & & Restricted & Primary \\",
        r"\midrule",
    ]
    for outcome, olabel in OUTCOMES:
        for ti, (term, tlabel) in enumerate(TERMS):
            rows = r[r.outcome_id.eq(outcome) & r.term.eq(term)].sort_values("cohort")
            if rows.empty:
                continue
            main_row = p[p.outcome_id.eq(outcome) & p.term.eq(term)]
            main_cell = (cell(float(main_row.pooled_estimate.iloc[0]),
                              float(main_row.ci_low.iloc[0]),
                              float(main_row.ci_high.iloc[0]))
                         if len(main_row) else "--")
            first = f"{olabel}" if ti == 0 else ""
            for j, (_, row) in enumerate(rows.iterrows()):
                lines.append(
                    f"{first if j == 0 else ''} & {tlabel if j == 0 else ''} & "
                    f"{LABEL[row.cohort]} & {cell(row.estimate, row.ci_low, row.ci_high)} & "
                    f"{'' if j else main_cell} " + r"\\")
                first = ""
            if len(rows) >= 3:
                est, lo, hi, k = pooled(rows)
                lines.append(
                    r" &  & \textit{Pooled} & \textit{" + cell(est, lo, hi)
                    + "}" + f" ($k$={k}) & " + r"\\")
        lines.append(r"\addlinespace")
    lines.extend([
        r"\bottomrule", r"\end{tabular}",
        r"\begin{flushleft}\footnotesize Restricted analyses include only respondents "
        r"with at least two of the three activities present at the first interview, so "
        r"that every level of the count was attainable for every respondent. Covariates "
        r"are those of the primary models. KLoSA and CHARLS did not meet the support "
        r"threshold for the restricted graded models. The primary column repeats the "
        r"pooled estimate reported in the main text for comparison. $k$ is the number "
        r"of contributing cohorts.\end{flushleft}",
        r"\end{table*}", "",
    ])
    out = GEN / "supp_table_baseline_opportunity.tex"
    out.write_text("\n".join(lines))
    print("wrote", out.relative_to(ROOT))
    for outcome, olabel in OUTCOMES:
        for term, tlabel in TERMS:
            rows = r[r.outcome_id.eq(outcome) & r.term.eq(term)]
            if len(rows) >= 3:
                est, lo, hi, k = pooled(rows)
                m = p[p.outcome_id.eq(outcome) & p.term.eq(term)]
                base = float(m.pooled_estimate.iloc[0]) if len(m) else float("nan")
                print(f"  {olabel:20s} {tlabel:16s} restricted {est:.2f} "
                      f"({lo:.2f}-{hi:.2f}) k={k}   primary {base:.2f}")


if __name__ == "__main__":
    main()
