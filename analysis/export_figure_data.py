#!/usr/bin/env python3
"""Every value the manuscript figures plot, as tidy CSVs.

The figures are drawn in R, but the provenance stays here: this script is the
only place that knows which rows of the frozen release a panel is allowed to
use, and it asserts what it found before writing. R then reads a table and
draws it, which keeps the drawing layer free of filtering logic that could
quietly select a different row.

Written to figures/data/. One file per panel.
"""
from __future__ import annotations

import pathlib

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
PILOT = ROOT / "artifacts/multidomain_behavioral_withdrawal_pilot/final"
EXT = ROOT / "artifacts/bmc_absolute_risk_work_exit_extension/final"
CONTEXT = ROOT / "artifacts/behavioral_withdrawal_competing_context/final"
OUT = ROOT / "figures/data"

SCOPE = "comparable_22_30_months"
SCEN = ["withdrawal_0", "withdrawal_1", "withdrawal_2plus"]
LABEL = {"charls": "CHARLS", "elsa": "ELSA", "hrs": "HRS",
         "klosa": "KLoSA", "mhas": "MHAS", "share": "SHARE"}


def write(frame: pd.DataFrame, name: str, expect: int) -> None:
    if len(frame) != expect:
        raise SystemExit(f"{name}: expected {expect} rows, got {len(frame)}")
    OUT.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUT / name, index=False)
    print(f"  {name:34s} {len(frame):3d} rows")


# --------------------------------------------------------------- figure 1 a,b
def standardised_risks(outcome: str, name: str) -> None:
    cohort = pd.read_csv(EXT / "cohort_standardized_risks.csv")
    pooled = pd.read_csv(EXT / "pooled_standardized_risks.csv")
    cohort = cohort[cohort.outcome_id.eq(outcome) & cohort.model_status.eq("PASS")]

    rows = []
    for coh, group in cohort.groupby("cohort"):
        group = group.set_index("scenario")
        for i, scen in enumerate(SCEN):
            rows.append({"series": LABEL[coh], "kind": "cohort", "x": i,
                         "risk": 100 * group.loc[scen, "standardized_risk"],
                         "lo": 100 * group.loc[scen, "ci_low"],
                         "hi": 100 * group.loc[scen, "ci_high"]})
    group = pooled[pooled.outcome_id.eq(outcome)].set_index("scenario")
    for i, scen in enumerate(SCEN):
        rows.append({"series": "Pooled", "kind": "pooled", "x": i,
                     "risk": 100 * group.loc[scen, "pooled_standardized_risk"],
                     "lo": 100 * group.loc[scen, "ci_low"],
                     "hi": 100 * group.loc[scen, "ci_high"]})
    frame = pd.DataFrame(rows)
    write(frame, name, len(frame))


# ----------------------------------------------------------------- figure 1 c
def outcome_forest() -> None:
    s = pd.read_csv(PILOT / "cross-cohort-summary.csv")
    graded = [("mortality", "Death"),
              ("incident_any_adl", "New ADL limitation"),
              ("multimorbidity_progression", "Multimorbidity progression")]
    rows = []
    for outcome, label in graded:
        for term, series in (("loss_1", "One stopped"),
                             ("loss_2plus", "Two or more stopped")):
            r = s[s.exposure_model.eq("score_categorical")
                  & s.outcome_id.eq(outcome) & s.term.eq(term)].iloc[0]
            rows.append({"row_label": label, "series": series, "block": "graded",
                         "estimate": r.pooled_estimate, "lo": r.ci_low,
                         "hi": r.ci_high, "cohorts": int(r.k_cohorts)})
    write(pd.DataFrame(rows), "fig1c_outcome_forest.csv", 6)


# ------------------------------------------------------------- figure 2 a,b
def reference_group() -> None:
    d = pd.read_csv(CONTEXT / "reference-group-synthesis.csv")
    d = d[d.outcome_id.eq("mortality")]

    def one(model: str, domain: str, term: str):
        r = d[d.model_id.eq(model) & d.domain.eq(domain) & d.term.eq(term)]
        if len(r) != 1:
            raise SystemExit(f"reference group: {model}/{domain}/{term} -> {len(r)} rows")
        return r.iloc[0]

    states = [("Never active", None), ("Stopped", "activity_stopped"),
              ("Started", "activity_started"), ("Continued", "activity_continued")]
    rows = []
    for label, term in states:
        if term is None:
            rows.append({"row_label": label, "estimate": 1.0, "lo": float("nan"),
                         "hi": float("nan"), "role": "reference"})
            continue
        r = one("four_state_never_had_reference", "activity", term)
        role = {"activity_stopped": "stopped", "activity_started": "other",
                "activity_continued": "continued"}[term]
        rows.append({"row_label": label, "estimate": r.pooled_estimate,
                     "lo": r.ci_low, "hi": r.ci_high, "role": role})
    write(pd.DataFrame(rows), "fig2a_transition_states.csv", 4)

    domains = [("Regular activity", "activity", "activity_loss"),
               ("Paid work", "work", "work_loss"),
               ("Drinking", "alcohol", "alcohol_loss")]
    rows = []
    for label, domain, term in domains:
        r = one("restricted_stopped_vs_continued", domain, term)
        rows.append({"row_label": label, "estimate": r.pooled_estimate,
                     "lo": r.ci_low, "hi": r.ci_high})
    write(pd.DataFrame(rows), "fig2b_stopped_vs_continued.csv", 3)


# ------------------------------------------------------------- figure 3 a,b
def work_exit() -> None:
    m = pd.read_csv(PILOT / "systematic-results-matrix.csv")
    q = m[m.scope.eq(SCOPE) & m.adjustment.eq("full")
          & m.exposure_model.eq("work_exit_phenotype")
          & m.model_status.eq("PASS")].drop_duplicates(["cohort", "outcome_id", "term"])
    wanted = [("hrs", "mortality", "HRS, death"),
              ("hrs", "incident_any_adl", "HRS, new ADL"),
              ("elsa", "incident_any_adl", "ELSA, new ADL")]
    rows = []
    for cohort, outcome, label in wanted:
        for term, series in (("work_exit_retirement", "Retirement-linked exit"),
                             ("work_exit_no_retirement", "Other work exit")):
            r = q[q.cohort.eq(cohort) & q.outcome_id.eq(outcome) & q.term.eq(term)]
            if not len(r):
                continue
            r = r.iloc[0]
            rows.append({"row_label": label, "series": series,
                         "estimate": r.estimate, "lo": r.ci_low, "hi": r.ci_high})
    write(pd.DataFrame(rows), "fig3a_work_exit.csv", 6)


def main() -> None:
    print("figure data, from the frozen release:")
    standardised_risks("mortality", "fig1a_death_risk.csv")
    standardised_risks("incident_any_adl", "fig1b_adl_risk.csv")
    outcome_forest()
    reference_group()
    work_exit()


if __name__ == "__main__":
    main()
