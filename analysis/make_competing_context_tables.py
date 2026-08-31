#!/usr/bin/env python3
"""Appendix tables for the three analyses added after the second clinical review.

Reads only the aggregate synthesis from artifacts/behavioral_withdrawal_competing_context.
"""
from __future__ import annotations

import pathlib

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
FINAL = ROOT / "artifacts/behavioral_withdrawal_competing_context/final"
OUT = ROOT / "manuscript/generated"

LABEL = {
    "mortality": "Death",
    "incident_any_adl": "New ADL limitation",
    "incident_any_iadl": "New IADL limitation",
    "multimorbidity_progression": "Multimorbidity progression",
}
TERM = {"loss_1": "One", "loss_2plus": "Two or more"}
DOMAIN = {"alcohol": "Stopped drinking", "activity": "Stopped regular activity",
          "work": "Left paid work"}
CONTEXT = {
    "self_rated_health_worsening": "Self-rated health worsened",
    "transition_hospitalization": "Hospital admission in the window",
    "cesd_worsening": "Depressive symptoms worsened",
    "bmi_or_weight_loss": "Weight loss",
    "incident_non_target_disease": "Newly reported common condition",
    "fi_change": "Frailty index change",
    "joint_context": "All of the above together",
    "joint_context_no_fi": "All except frailty index change",
}
DASH = "--"


def ci(row) -> str:
    if pd.isna(row.pooled_estimate):
        return "Not estimable"
    return f"{row.pooled_estimate:.2f} ({row.ci_low:.2f}{DASH}{row.ci_high:.2f})"


def pi(row) -> str:
    if pd.isna(row.prediction_low):
        return DASH
    return f"{row.prediction_low:.2f}{DASH}{row.prediction_high:.2f}"


def i2(row) -> str:
    return DASH if pd.isna(row.i2) else f"{row.i2:.0f}\\%"


def table(path: pathlib.Path, caption: str, label: str, header: str,
          spec: str, body: list[str], note: str = "") -> None:
    lines = [
        r"\begin{table}[htbp]", r"\centering",
        r"\caption{" + caption + r"}", r"\label{" + label + r"}",
        r"\footnotesize", r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{" + spec + r"}", r"\hline", header, r"\hline",
        *body, r"\hline", r"\end{tabular}",
    ]
    if note:
        lines.append(r"\begin{flushleft}\footnotesize " + note + r"\end{flushleft}")
    lines.append(r"\end{table}")
    path.write_text("\n".join(lines) + "\n")
    print(f"wrote {path.relative_to(ROOT)}")


# ------------------------------------------------------- composite
c = pd.read_csv(FINAL / "composite-synthesis.csv")
c = c[c.exposure_model.eq("graded") & c.state_adjustment.eq("unconditional")]
body = []
for outcome in ["incident_any_adl", "incident_any_iadl", "multimorbidity_progression"]:
    for term in ["loss_1", "loss_2plus"]:
        cells = []
        for model in ["survivors", "death_or_outcome"]:
            row = c[c.outcome_id.eq(outcome) & c.term.eq(term) & c.model_id.eq(model)]
            if row.empty:
                cells += ["Not estimable", DASH]
                continue
            row = row.iloc[0]
            cells += [ci(row), f"{int(row.k_cohorts)}, {i2(row)}"]
        deaths = c.loc[c.outcome_id.eq(outcome) & c.model_id.eq("death_or_outcome"),
                       "entered_through_death_total"]
        added = f"{int(deaths.iloc[0]):,}" if len(deaths) else DASH
        first = LABEL[outcome] if term == "loss_1" else ""
        body.append(" & ".join([first, TERM[term], *cells, added]) + r" \\")
table(
    OUT / "supp_table_competing_composite.tex",
    r"\textbf{Death counted as an event.} The non-fatal outcomes are ascertained at "
    r"the outcome interview, so respondents who died first are not observed for them. "
    r"Which respondents can enter each outcome's risk set is decided at the landmark "
    r"interview that ends the behaviour-transition window, before any death, "
    r"and the composite then counts death as an event within that risk set. The "
    r"number of deaths added therefore differs by outcome, because freedom from an "
    r"activity-of-daily-living limitation at the landmark interview identifies a "
    r"different group from freedom from an instrumental limitation, and "
    r"multimorbidity progression excludes nobody at baseline. Both columns are the "
    r"primary model with no adjustment for behavioural status at the landmark "
    r"interview, so the survivor column reproduces the main analysis.",
    "tab:composite",
    r"Outcome & Stopped & Survivors & $k$, $I^2$ & Death or outcome & $k$, $I^2$ & Deaths added \\",
    "llccccr", body)

# ------------------------------------------------------- reference group
r = pd.read_csv(FINAL / "reference-group-synthesis.csv")
body = []
for outcome in ["mortality", "incident_any_adl", "incident_any_iadl"]:
    for domain in ["alcohol", "activity", "work"]:
        row = r[r.outcome_id.eq(outcome) & r.domain.eq(domain)
                & r.model_id.eq("restricted_stopped_vs_continued")]
        four = r[r.outcome_id.eq(outcome) & r.domain.eq(domain)
                 & r.model_id.eq("four_state_never_had_reference")
                 & r.term.eq(f"{domain}_stopped")]
        cells = []
        for source in [row, four]:
            if source.empty or pd.isna(source.iloc[0].pooled_estimate):
                cells += ["Not estimable", DASH]
            else:
                s = source.iloc[0]
                cells += [ci(s), f"{int(s.k_cohorts)}, {i2(s)}"]
        first = LABEL[outcome] if domain == "alcohol" else ""
        body.append(" & ".join([first, DOMAIN[domain], *cells]) + r" \\")
table(
    OUT / "supp_table_reference_group.tex",
    r"\textbf{What the single-domain comparison is against.} The restricted "
    r"analysis keeps only respondents who had the activity at the first interview, "
    r"so stopping is compared with continuing and with nothing else. The "
    r"four-state column takes respondents who never had the activity as the "
    r"reference instead, and reports the estimate for stopping. Both hold the "
    r"other two domains' status at the landmark interview constant.",
    "tab:reference",
    r"Outcome & Domain & vs continued & $k$, $I^2$ & vs never had & $k$, $I^2$ \\",
    "llcccc", body,
    note=r"Combinations that left fewer than three contributing cohorts, or that "
         r"failed a prespecified support gate, are reported as not estimable "
         r"rather than fitted on fewer.")

# ------------------------------------------------------- context
a = pd.read_csv(FINAL / "context-attenuation-by-cohort.csv")
a = a[a.state_adjustment.eq("unconditional") & a.term.eq("loss_2plus")
      & a.outcome_id.eq("mortality")]
body = []
for context in ["self_rated_health_worsening", "transition_hospitalization",
                "cesd_worsening", "bmi_or_weight_loss",
                "incident_non_target_disease", "fi_change",
                "joint_context_no_fi", "joint_context"]:
    for cohort, name in [("hrs", "HRS"), ("share", "SHARE")]:
        row = a[a.context_id.eq(context) & a.cohort.eq(cohort)]
        if row.empty:
            continue
        row = row.iloc[0]
        first = CONTEXT[context] if cohort == "hrs" else ""
        body.append(" & ".join([
            first, name, f"{int(row.n):,}", f"{int(row.events):,}",
            f"{row.estimate:.2f} ({row.ci_low:.2f}{DASH}{row.ci_high:.2f})",
            f"{row.log_rr_attenuation_percent:.0f}\\%",
        ]) + r" \\")
table(
    OUT / "supp_table_clinical_context.tex",
    r"\textbf{The mortality gradient against health change that was already "
    r"measurable.} Each row adds one measure of change between the two behaviour "
    r"interviews to the primary model for two or more activities stopped, on the "
    r"risk set where that measure is recorded. Attenuation is the proportion of "
    r"the log risk ratio removed. The two cohorts with the largest risk sets are "
    r"shown; the frailty index is recorded on a subset of waves, so the joint "
    r"model is given with and without it.",
    "tab:context",
    r"Adjusted for & Cohort & Intervals & Deaths & Risk ratio (95\% CI) & Attenuation \\",
    "llrrcc", body,
    note=r"Pooling across cohorts was not possible for most of these models "
         r"because the contributing measures are recorded in different subsets of "
         r"cohorts and waves, so cohort-specific estimates are reported.")

# ------------------------------------------------------- transition counts
tc = pd.read_csv(FINAL / "transition-counts.csv")
STATE = {"0_to_0": "Never had it", "0_to_1": "Started", "1_to_0": "Stopped",
         "1_to_1": "Continued"}
COHORT = {"charls": "CHARLS", "elsa": "ELSA", "hrs": "HRS", "klosa": "KLoSA",
          "mhas": "MHAS", "share": "SHARE"}
body = []
for domain in ["alcohol", "activity", "work"]:
    for state in ["0_to_0", "0_to_1", "1_to_0", "1_to_1"]:
        cells = []
        for cohort in COHORT:
            row = tc[tc.domain.eq(domain) & tc.transition_state.eq(state)
                     & tc.cohort.eq(cohort)]
            cells.append(f"{int(row.iloc[0].n):,}" if not row.empty else DASH)
        first = DOMAIN[domain] if state == "0_to_0" else ""
        body.append(" & ".join([first, STATE[state], *cells]) + r" \\")
table(
    OUT / "supp_table_transitions.tex",
    r"\textbf{The four transition states behind each single-domain comparison.} "
    r"Person-intervals in the comparable 22 to 30 month window. The mutually "
    r"adjusted component analysis compares stopped against the other three states "
    r"combined; the restricted analysis compares stopped against continued only.",
    "tab:transitions",
    r"Domain & State & " + " & ".join(COHORT.values()) + r" \\",
    "ll" + "r" * len(COHORT), body)
