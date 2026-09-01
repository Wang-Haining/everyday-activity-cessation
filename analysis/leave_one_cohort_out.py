"""Leave-one-cohort-out re-synthesis of the primary estimates.

Zhang's P1-2 asks whether one cohort carries the pooled result. The question is
answerable from the frozen per-cohort log-estimates and standard errors that the
manuscript already reports, using the same REML plus Hartung-Knapp routine as the
primary synthesis. No respondent-level data and no new cohort run is involved.

Emits manuscript/generated/supp_table_leave_one_out.tex.
"""
import pathlib
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from postprocess_behavior_outcome_landscape import reml_hk  # noqa: E402

M = pd.read_csv(
    ROOT / "artifacts/multidomain_behavioral_withdrawal_pilot/final/systematic-results-matrix.csv"
)

ROWS = [
    ("mortality", "score_categorical", "loss_1", "Death, one stopped"),
    ("mortality", "score_categorical", "loss_2plus", "Death, two or more"),
    ("incident_any_adl", "score_categorical", "loss_1", "New ADL, one stopped"),
    ("incident_any_adl", "score_categorical", "loss_2plus", "New ADL, two or more"),
    ("multimorbidity_progression", "score_categorical", "loss_1", "Multimorbidity, one stopped"),
    ("multimorbidity_progression", "score_categorical", "loss_2plus", "Multimorbidity, two or more"),
]

NAME = {"charls": "CHARLS", "elsa": "ELSA", "hrs": "HRS", "klosa": "KLoSA",
        "mhas": "MHAS", "share": "SHARE"}


def contributors(outcome, model, term):
    q = M[
        M.scope.eq("comparable_22_30_months")
        & M.adjustment.eq("full")
        & M.exposure_model.eq(model)
        & M.model_status.eq("PASS")
        & M.outcome_id.eq(outcome)
        & M.term.eq(term)
    ].drop_duplicates("cohort")
    return q[["cohort", "estimate", "standard_error"]].sort_values("cohort")


def synth(y, v):
    fit = reml_hk(np.asarray(y, float), np.asarray(v, float))
    return (np.exp(fit["pooled"]), np.exp(fit["ci_low"]), np.exp(fit["ci_high"]),
            np.exp(fit["prediction_low"]), np.exp(fit["prediction_high"]), fit["i2"] * 100, fit["tau2"])


lines = []
print(f"{'outcome / term':42s} {'omitted':8s} {'k':>2s} {'RR':>6s} {'95% CI':>16s} {'PI':>16s} {'I2%':>6s}")
print("-" * 104)
for outcome, model, term, label in ROWS:
    c = contributors(outcome, model, term)
    if c.empty:
        continue
    y = np.log(c.estimate.to_numpy())  # the matrix stores risk ratios; synthesis is on the log scale
    v = (c.standard_error.to_numpy()) ** 2
    names = [NAME[x] for x in c.cohort]
    k = len(y)

    rr, lo, hi, pl, ph, i2, tau2 = synth(y, v)
    print(f"{label:42s} {'none':8s} {k:2d} {rr:6.2f} {lo:7.2f}-{hi:7.2f} {pl:7.2f}-{ph:7.2f} {i2:6.1f}")
    lines.append((label, "None", k, rr, lo, hi, pl, ph, i2))

    for j, drop in enumerate(names):
        keep = [i for i in range(k) if i != j]
        if len(keep) >= 3:
            rr2, lo2, hi2, pl2, ph2, i22, _ = synth(y[keep], v[keep])
            print(f"{'':42s} {drop:8s} {len(keep):2d} {rr2:6.2f} {lo2:7.2f}-{hi2:7.2f} "
                  f"{pl2:7.2f}-{ph2:7.2f} {i22:6.1f}")
            lines.append(("", drop, len(keep), rr2, lo2, hi2, pl2, ph2, i22))
        else:
            rem = ", ".join(f"{names[i]} {np.exp(y[i]):.2f}" for i in keep)
            print(f"{'':42s} {drop:8s} {len(keep):2d}   two cohorts remain, not synthesised: {rem}")
            lines.append(("", drop, len(keep), None, None, None, None, None, None))
    print()

CELL = "{label} & {drop} & {k} & {est} & {pi} & {i2} \\\\"
out = [
    r"\begin{table}[htbp]",
    r"\centering",
    r"\caption{\textbf{Leave-one-cohort-out synthesis.} Each block repeats the primary "
    r"random-effects synthesis with one cohort omitted, using the same restricted maximum "
    r"likelihood estimator and Hartung--Knapp interval. Prediction intervals and $I^2$ "
    r"require at least three contributing cohorts, so rows that would leave two are shown "
    r"as the remaining cohort estimates in the text.}",
    r"\label{tab:leaveoneout}",
    r"\footnotesize",
    r"\setlength{\tabcolsep}{4pt}",
    r"\begin{tabular}{llrcc c}",
    r"\hline",
    # "Cohorts", not "$k$". A clinical reader should not have to decode a
    # letter, and figure 1 already prints the same quantity under that word.
    r"Outcome and term & Omitted & Cohorts & Risk ratio (95\% CI) & Prediction interval & $I^2$ \\",
    r"\hline",
]
for label, drop, k, rr, lo, hi, pl, ph, i2 in lines:
    if rr is None:
        out.append(CELL.format(label=label, drop=drop, k=k, est="Not synthesised",
                               pi="--", i2="--"))
    else:
        out.append(CELL.format(
            label=label, drop=drop, k=k,
            est=f"{rr:.2f} ({lo:.2f}--{hi:.2f})",
            pi=f"{pl:.2f}--{ph:.2f}",
            i2=f"{i2:.1f}\\%"))
out += [r"\hline", r"\end{tabular}", r"\end{table}"]

dest = ROOT / "manuscript/generated/supp_table_leave_one_out.tex"
dest.write_text("\n".join(out) + "\n")
print(f"wrote {dest.relative_to(ROOT)}")
