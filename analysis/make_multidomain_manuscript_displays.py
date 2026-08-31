#!/usr/bin/env python3
"""Create the positive multidomain manuscript displays from frozen aggregates."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from status_labels import label as status_label  # noqa: E402  (needs the path above)

PILOT = ROOT / "artifacts/multidomain_behavioral_withdrawal_pilot/final"
EXT = ROOT / "artifacts/work_exit_reason_extension/final"
DESC = ROOT / "artifacts/behavioral_withdrawal_frailty_extension/manuscript_descriptive"
FRAILTY = ROOT / "artifacts/behavioral_withdrawal_frailty_extension/final"
FIG = ROOT / "figures/manuscript"
GEN = ROOT / "manuscript/generated"


def write_if_changed(path, body: str) -> None:
    """Write only when the content differs.

    A generator that rewrites an identical file gives it a new timestamp, and
    every product built from it then reports as stale; the rebuild that silences
    that is a rebuild of a document that did not move. This file writes a dozen
    tables on every run and most runs change one of them.
    """
    if path.exists() and path.read_text(encoding="utf-8") == body:
        return
    path.write_text(body, encoding="utf-8")



NAVY = "#12355B"
BLUE = "#3B6FB6"
ORANGE = "#E56B2F"
TEAL = "#0C7C86"
GREY = "#667085"
LIGHT = "#E7EDF3"
GOLD = "#D6A84B"
COHORT_COLORS = {
    "charls": "#7A5195",
    "elsa": "#2F6B9A",
    "hrs": "#D45087",
    "klosa": "#2A9D8F",
    "share": "#F28E2B",
}

COHORTS = ["charls", "elsa", "hrs", "klosa", "share"]
LABELS = {"charls": "CHARLS", "elsa": "ELSA", "hrs": "HRS", "klosa": "KLoSA", "mhas": "MHAS", "share": "SHARE"}
# The four newly reported diagnoses left this paper's outcome set. The graded
# count was evaluable for them in HRS and SHARE alone, below the prespecified
# three-cohort minimum, so they could only ever carry the binary any-cessation
# exposure while the paper's claim is about the count. They are reported
# separately.
OUTCOMES = [
    ("multimorbidity_progression", "Multimorbidity progression"),
    ("incident_any_adl", "Incident ADL limitation"),
    ("incident_any_iadl", "Incident IADL limitation"),
    ("mortality", "Mortality"),
]


def save(fig: plt.Figure, name: str) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{name}.png", dpi=320, bbox_inches="tight", facecolor="white")
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _cohort_names(r: pd.Series) -> str:
    """Name the contributing cohorts so the count is never a bare number."""
    return ", ".join(LABELS[c] for c in str(r.cohorts).split(";") if c)


def _row(summary: pd.DataFrame, outcome: str, exposure: str, term: str) -> pd.Series | None:
    rows = summary[
        summary.outcome_id.eq(outcome)
        & summary.exposure_model.eq(exposure)
        & summary.term.eq(term)
    ]
    return None if rows.empty else rows.iloc[0]


def _panel_label(ax: plt.Axes, label: str, title: str) -> None:
    ax.text(0.0, 1.04, label, transform=ax.transAxes, fontsize=13, fontweight="bold", color=NAVY)
    ax.text(0.08, 1.04, title, transform=ax.transAxes, fontsize=11.5, fontweight="bold", color=NAVY)


def _fmt_count(row: pd.Series) -> str:
    return f"{int(row['count']):,} ({row['percent']:.1f})"


def _fmt_mean(row: pd.Series, digits: int = 1) -> str:
    return f"{row['mean']:.{digits}f} ({row['sd']:.{digits}f})"


def table1() -> None:
    data = pd.read_csv(DESC / "manuscript-table1-long.csv")
    flow = pd.read_csv(DESC / "manuscript-flow.csv")
    primary = flow[flow.stage.eq("primary_behavior_risk_set")].set_index("cohort")

    def value(cohort: str, variable: str, kind: str) -> str:
        r = data[(data.cohort.eq(cohort)) & data.group.eq("all") & data.variable.eq(variable)].iloc[0]
        return _fmt_mean(r, 1 if variable != "baseline_disease_count" else 1) if kind == "mean" else _fmt_count(r)

    rows: list[tuple[str, list[str]]] = [
        ("Person-intervals", [f"{int(primary.loc[c, 'intervals']):,}" for c in COHORTS]),
        ("Participants", [f"{int(primary.loc[c, 'people']):,}" for c in COHORTS]),
    ]
    for label, variable, kind in [
        ("Age, years", "age_years", "mean"),
        ("Women", "female", "count"),
        ("Current smoking", "current_smoking", "count"),
        ("Previously diagnosed conditions, count", "baseline_disease_count", "mean"),
        ("Current drinking", "current_drinking", "count"),
        ("Current regular activity", "current_regular_activity", "count"),
        ("Current paid work", "current_paid_work", "count"),
        ("Stopped drinking", "alcohol_withdrawal", "count"),
        ("Stopped regular activity", "activity_withdrawal", "count"),
        ("Stopped paid work", "paid_work_withdrawal", "count"),
    ]:
        rows.append((label, [value(c, variable, kind) for c in COHORTS]))
    for label, cat in [("None stopped", "0"), ("One stopped", "1"), ("Two or more stopped", "2_plus")]:
        values = []
        for cohort in COHORTS:
            r = data[(data.cohort.eq(cohort)) & data.group.eq(f"withdrawal_{cat}") & data.variable.eq("age_years")].iloc[0]
            n = int(r["denominator"])
            total = int(primary.loc[cohort, "intervals"])
            values.append(f"{n:,} ({100*n/total:.1f})")
        rows.append((label, values))

    lines = [
        r"\begin{table*}[t]", r"\caption{Characteristics of the primary person-interval risk set}",
        r"\label{tab:baseline}", r"\centering\footnotesize", r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{@{}lccccc@{}}", r"\toprule",
        "Characteristic & " + " & ".join(LABELS[c] for c in COHORTS) + " \\\\", r"\midrule",
    ]
    for label, values in rows:
        lines.append(label + " & " + " & ".join(values) + " \\\\")
    lines.extend([
        r"\bottomrule", r"\end{tabular}", r"}",
        r"\begin{flushleft}\footnotesize Values are mean (SD) or n (\%). Characteristics were measured at the interview ending the behaviour-transition window. The condition count included cancer, diabetes, heart disease, hypertension, stroke and arthritis. Percentages use each variable's non-missing denominator; this differs from the interval count only for current smoking, recorded for 1364 CHARLS, 34,605 ELSA, 72,799 HRS, 17,751 KLoSA and 46,141 SHARE intervals. Participants could contribute more than one interval.\end{flushleft}",
        r"\end{table*}", "",
    ])
    GEN.mkdir(parents=True, exist_ok=True)
    write_if_changed(GEN / "table1_multidomain_characteristics.tex", "\n".join(lines))


def _effect(r: pd.Series | None) -> str:
    if r is None:
        return "--"
    return f"{r.pooled_estimate:.2f} ({r.ci_low:.2f}--{r.ci_high:.2f})"


def _effect_with_k(r: pd.Series | None) -> str:
    if r is None:
        return "--"
    return f"{_effect(r)}; {int(r.k_cohorts)}"


def _absolute_risk_triplet(pooled: pd.DataFrame, outcome: str) -> str:
    group = pooled.loc[pooled["outcome_id"].eq(outcome)].set_index("scenario")
    scenarios = ["withdrawal_0", "withdrawal_1", "withdrawal_2plus"]
    if not set(scenarios).issubset(group.index):
        return "--"
    risks = " / ".join(f"{100 * group.loc[name, 'pooled_standardized_risk']:.1f}" for name in scenarios)
    return f"{risks}; {int(group.iloc[0].k_cohorts)}"


def table2() -> None:
    summary = pd.read_csv(PILOT / "cross-cohort-summary.csv")
    pooled_risks = pd.read_csv(EXT / "pooled_standardized_risks.csv")
    lines = [
        r"\begin{table*}[t]", r"\caption{Recent cessation of everyday activities and subsequent clinical outcomes}",
        r"\label{tab:primary}", r"\centering\footnotesize",
        r"\makebox[\textwidth][c]{%",
        r"\begin{tabular}{@{}lcccc@{}}", r"\toprule",
        r"& Any cessation & One stopped & Two or more & Standardised risk \\",
        r"Outcome & RR (95\% CI); cohorts & RR (95\% CI); cohorts & RR (95\% CI); cohorts & 0 / 1 / 2+ (\%); cohorts \\",
        r"\midrule",
    ]
    graded = ["mortality", "incident_any_adl", "incident_any_iadl",
              "multimorbidity_progression"]
    by_id = dict(OUTCOMES)
    order = ([(o, by_id[o]) for o in graded]
             + [(o, lab) for o, lab in OUTCOMES if o not in graded])
    for outcome, label in order:
        assert outcome in graded, outcome
        any_r = _row(summary, outcome, "any_withdrawal", "any_withdrawal")
        one_r = _row(summary, outcome, "score_categorical", "loss_1")
        two_r = _row(summary, outcome, "score_categorical", "loss_2plus")
        cells = [_effect_with_k(any_r)]
        if outcome in graded:
            cells += [_effect_with_k(one_r), _effect_with_k(two_r),
                      _absolute_risk_triplet(pooled_risks, outcome)]
        else:
            cells += ["", "", ""]
        lines.append(f"{label} & " + " & ".join(cells) + r" \\")
    lines.extend([
        r"\bottomrule", r"\end{tabular}}",
        r"\begin{flushleft}\footnotesize The last figure in each cell is the number of cohorts contributing to that estimate. It is not a number of participants or person-intervals, and it is given separately for each estimate because categorical support differed from the any-cessation model. Risk ratios were adjusted for age, sex, interview wave, education, economic position, smoking, prior disease burden, baseline behaviour opportunity and outcome-specific baseline status. Absolute risks are logistic-model standardised estimates from the same full risk set and covariates; they are shown only for the four prespecified absolute-risk outcomes.\end{flushleft}",
        r"\end{table*}", "",
    ])
    write_if_changed(GEN / "table2_multidomain_results.tex", "\n".join(lines))


def supplementary_tables() -> None:
    systematic = pd.read_csv(PILOT / "systematic-results-matrix.csv")
    summary = pd.read_csv(PILOT / "cross-cohort-summary.csv")

    lines = [
        r"\begin{table*}[t]", r"\caption{Cohort-specific estimates for any recent cessation}",
        r"\label{tab:cohort-results}", r"\centering\footnotesize",
        r"\begin{tabular}{@{}llrrrrll@{}}", r"\toprule",
        r"Outcome & Cohort & n & Events & Cessation n & Cessation events & RR (95\% CI) & Status \\",
        r"\midrule",
    ]
    for outcome, label in OUTCOMES:
        for cohort in COHORTS:
            z = systematic[
                systematic.scope.eq("comparable_22_30_months")
                & systematic.adjustment.eq("full")
                & systematic.exposure_model.eq("any_withdrawal")
                & systematic.outcome_id.eq(outcome)
                & systematic.cohort.eq(cohort)
            ]
            if z.empty:
                continue
            r = z.iloc[0]
            if r.model_status == "PASS":
                effect = f"{r.estimate:.2f} ({r.ci_low:.2f}--{r.ci_high:.2f})"
                wn = f"{int(r.any_withdrawal_n):,}"
                we = f"{int(r.any_withdrawal_events):,}"
                status = status_label("PASS")
            else:
                effect, wn, we = "--", "--", "--"
                status = status_label(r.model_status)
            lines.append(
                f"{label} & {LABELS[cohort]} & {int(r.n):,} & {int(r.events):,} & {wn} & {we} & {effect} & {status} " + r"\\"
            )
        lines.append(r"\addlinespace[2pt]")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"}", r"\end{table*}", ""])
    write_if_changed(GEN / "supp_table_s1_multidomain_cohort_results.tex", "\n".join(lines))

    component_outcomes = [
        ("multimorbidity_progression", "Multimorbidity progression"),
        ("incident_any_adl", "ADL limitation"),
        ("incident_any_iadl", "IADL limitation"),
        ("mortality", "Mortality"),
    ]
    components = [
        ("alcohol_loss", "Stopped drinking"),
        ("activity_loss", "Stopped regular activity"),
        ("work_loss", "Left paid work"),
    ]
    lines = [
        r"\begin{table*}[t]", r"\caption{Mutually adjusted cessation-component estimates}",
        r"\label{tab:components}", r"\centering\footnotesize",
        r"\begin{tabular}{@{}lllrrrl@{}}", r"\toprule",
        r"Outcome & Component & Contributing cohorts & RR & 95\% CI & Prediction interval & Direction \\",
        r"\midrule",
    ]
    for outcome, label in component_outcomes:
        for term, term_label in components:
            r = _row(summary, outcome, "mutually_adjusted_components", term)
            if r is None:
                continue
            lines.append(
                f"{label} & {term_label} & {_cohort_names(r)} & {r.pooled_estimate:.2f} & "
                f"{r.ci_low:.2f}--{r.ci_high:.2f} & {r.prediction_low:.2f}--{r.prediction_high:.2f} & "
                f"{int(round(r.direction_consistency * r.k_cohorts))}/{int(r.k_cohorts)} above 1 " + r"\\"
            )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""])
    write_if_changed(GEN / "supp_table_s4_multidomain_components.tex", "\n".join(lines))

    gradient_outcomes = [
        ("multimorbidity_progression", "Multimorbidity progression"),
        ("incident_any_adl", "Incident ADL limitation"),
        ("incident_any_iadl", "Incident IADL limitation"),
        ("mortality", "Mortality"),
    ]
    lines = [
        r"\begin{table*}[t]", r"\caption{Graded associations by number of activities recently stopped}",
        r"\label{tab:gradient}", r"\centering\footnotesize",
        r"\begin{tabular}{@{}>{\raggedright\arraybackslash}p{.17\textwidth}l"
        r">{\raggedright\arraybackslash}p{.15\textwidth}rrrrr@{}}", r"\toprule",
        r"Outcome & Activities stopped & Contributing cohorts & RR & 95\% CI & Prediction interval & $I^2$ (\%) & $\tau^2$ \\",
        r"\midrule",
    ]
    for outcome, label in gradient_outcomes:
        for term, term_label in [("loss_1", "One"), ("loss_2plus", "Two or more")]:
            r = _row(summary, outcome, "score_categorical", term)
            lines.append(
                f"{label} & {term_label} & {_cohort_names(r)} & {r.pooled_estimate:.2f} & "
                f"{r.ci_low:.2f}--{r.ci_high:.2f} & {r.prediction_low:.2f}--{r.prediction_high:.2f} & "
                f"{100*r.i2:.1f} & {r.tau2_analysis_scale:.3f} " + r"\\"
            )
    lines.extend([
        r"\bottomrule", r"\end{tabular}",
        r"\begin{flushleft}\footnotesize $\tau^2$ is the between-cohort variance of the "
        r"log risk ratio. With three or four cohorts an $I^2$ of zero means heterogeneity "
        r"was not detected rather than absent; $\tau^2$ gives its size. The two are "
        r"estimated differently and need not agree: $I^2$ is computed from Cochran's Q, "
        r"and $\tau^2$ by restricted maximum likelihood, which cannot return a value "
        r"below zero. Both multimorbidity progression rows sit on that boundary.\end{flushleft}",
        r"\end{table*}", ""])
    write_if_changed(GEN / "supp_table_s5_multidomain_gradient.tex", "\n".join(lines))


def extension_supplementary_tables() -> None:
    risks = pd.read_csv(EXT / "cohort_standardized_risks.csv", low_memory=False)
    pooled = pd.read_csv(EXT / "pooled_standardized_risks.csv", low_memory=False)
    contrasts = pd.read_csv(EXT / "cohort_work_exit_contrasts.csv", low_memory=False)
    pooled_contrasts = pd.read_csv(EXT / "pooled_work_exit_contrasts.csv", low_memory=False)
    intervals = pd.read_csv(EXT / "comparable_interval_provenance.csv", low_memory=False)

    retirement_rows = [
        ("ELSA", r"r\{wave\}retemp", "0 = no; 1 = yes", "1"),
        ("HRS", r"r\{wave\}retemp", "0 = no retire empstat; 1 = only retire empstat; 2 = retire plus other empstat", "1 or 2"),
        ("MHAS", r"r\{wave\}retemp", "0 = working; 1 = retired; 2 = retired and other status", "1 or 2"),
        ("SHARE", r"r\{wave\}retemp", "0 = not retired empstat; 1 = retired empstat", "1"),
    ]
    lines = [
        r"\begin{table*}[t]", r"\caption{Source coding of retirement-linked employment status}",
        r"\label{tab:retirement-provenance}", r"\centering\footnotesize",
        r"\begin{tabular}{@{}ll>{\raggedright\arraybackslash}p{.38\textwidth}"
        r">{\raggedright\arraybackslash}p{.18\textwidth}@{}}", r"\toprule",
        r"Cohort & Wave-specific source field & Source value labels & Values classified as retirement-linked \\",
        r"\midrule",
    ]
    for cohort, field, labels, retired in retirement_rows:
        lines.append(f"{cohort} & {field} & {labels} & {retired} " + r"\\")
    lines.extend([
        r"\bottomrule", r"\end{tabular}",
        r"\begin{flushleft}\footnotesize Negative special codes and Stata extended-missing values were treated as missing. These fields identify retirement-linked employment status at the interview, not whether retirement was planned or voluntary. MHAS had no primary 22--30-month interval.\end{flushleft}",
        r"\end{table*}", "",
    ])
    write_if_changed(GEN / "supp_table_retirement_provenance.tex", "\n".join(lines))

    usable_intervals = intervals.loc[intervals["scheduled_followup_months"].notna()].copy()
    usable_intervals = usable_intervals.sort_values(["cohort", "t1"])
    lines = [
        r"\begin{table*}[t]", r"\caption{Cohort wave pairs in the comparable 22--30-month outcome window}",
        r"\label{tab:interval-provenance}", r"\centering\footnotesize",

        r"\begin{tabular}{@{}lrrr@{}}", r"\toprule",
        r"Cohort & Interview ending the cessation window & Outcome interview & Scheduled follow-up, months \\",
        r"\midrule",
    ]
    for row in usable_intervals.itertuples(index=False):
        lines.append(
            f"{LABELS.get(row.cohort, row.cohort)} & {int(row.t1)} & {int(row.outcome_wave)} & "
            f"{int(row.scheduled_followup_months)} " + r"\\"
        )
    lines.extend([
        r"\bottomrule", r"\end{tabular}",
        r"\begin{flushleft}\footnotesize MHAS had no scheduled outcome interval between 22 and 30 months and therefore did not enter the primary cross-cohort comparison.\end{flushleft}",
        r"\end{table*}", "",
    ])
    write_if_changed(GEN / "supp_table_interval_provenance.tex", "\n".join(lines))

    scenario_order = ["withdrawal_0", "withdrawal_1", "withdrawal_2plus"]
    outcome_labels = {
        "mortality": "Mortality",
        "incident_any_adl": "Incident ADL limitation",
        "incident_any_iadl": "Incident IADL limitation",
        "multimorbidity_progression": "Multimorbidity progression",
    }

    def risk_cell(group: pd.DataFrame, scenario: str, pooled_row: bool = False) -> str:
        row = group.loc[group["scenario"].eq(scenario)].iloc[0]
        value_name = "pooled_standardized_risk" if pooled_row else "standardized_risk"
        return f"{100 * row[value_name]:.1f} ({100 * row.ci_low:.1f}--{100 * row.ci_high:.1f})"

    lines = [
        r"\begin{table*}[t]", r"\caption{\textbf{Adjusted standardised risks by number of activities recently stopped.} Each cell is the standardised two-year risk as a percentage, with its 95\% CI.}",
        r"\label{tab:absolute-risks}", r"\centering\footnotesize",
        r"\begin{tabular}{@{}>{\raggedright\arraybackslash}p{.17\textwidth}"
        r">{\raggedright\arraybackslash}p{.22\textwidth}rrr@{}}", r"\toprule",
        r"Outcome & Cohort & None stopped & One stopped & Two or more stopped \\",
        r"\midrule",
    ]
    for outcome, label in outcome_labels.items():
        passed = risks.loc[risks["outcome_id"].eq(outcome) & risks["model_status"].eq("PASS")]
        for cohort, group in passed.groupby("cohort", sort=True):
            lines.append(
                f"{label} & {LABELS[cohort]} & "
                + " & ".join(risk_cell(group, scenario) for scenario in scenario_order)
                + " " + r"\\"
            )
        group = pooled.loc[pooled["outcome_id"].eq(outcome)]
        cohorts = ", ".join(LABELS[name] for name in group.iloc[0].cohorts.split(";"))
        lines.append(
            f"{label} & Pooled ({cohorts}; k={int(group.iloc[0].k_cohorts)}) & "
            + " & ".join(risk_cell(group, scenario, pooled_row=True) for scenario in scenario_order)
            + " " + r"\\"
        )
        lines.append(r"\addlinespace[2pt]")
    lines.extend([
        r"\bottomrule", r"\end{tabular}",
        r"\begin{flushleft}\footnotesize Risks were standardised from cohort-specific logistic models using the same full risk sets and covariates as the categorical risk-ratio analyses. Pooled values were synthesised on the logit scale.\end{flushleft}",
        r"\end{table*}", "",
    ])
    write_if_changed(GEN / "supp_table_absolute_risks.tex", "\n".join(lines))

    lines = [
        r"\begin{table*}[t]", r"\caption{\textbf{Retirement-linked and other work exits.} The retirement-linked, other-exit and direct-ratio columns are all risk ratios.}",
        r"\label{tab:work-exit}", r"\centering\footnotesize",
        # Only the three text columns wrap. Giving the numeric columns a fixed
        # width as well left the status column too little room, and it broke
        # one word to a line; as r columns they take what they need and the
        # rest of the width goes where the sentences are.
        r"\begin{tabular}{@{}>{\raggedright\arraybackslash}p{.12\textwidth}"
        r">{\raggedright\arraybackslash}p{.12\textwidth}"
        r">{\raggedright\arraybackslash}p{.07\textwidth}rrrr"
        r">{\raggedright\arraybackslash}p{.17\textwidth}@{}}", r"\toprule",
        r"Outcome & Cohort & n / events & Retirement-linked & Other exit & Direct ratio & 95\% CI & Status \\",
        r"\midrule",
    ]
    for outcome, label in [("incident_any_adl", "Incident ADL limitation"), ("mortality", "Mortality")]:
        for row in contrasts.loc[contrasts["outcome_id"].eq(outcome)].itertuples(index=False):
            if row.model_status == "PASS":
                values = [
                    f"{int(row.n):,} / {int(row.events):,}",
                    f"{row.retirement_exit_rr_vs_continued_work:.2f}",
                    f"{row.other_exit_rr_vs_continued_work:.2f}",
                    f"{row.estimate:.2f}",
                    f"{row.ci_low:.2f}--{row.ci_high:.2f}",
                    status_label("PASS"),
                ]
            else:
                n_events = "--" if pd.isna(row.n) or pd.isna(row.events) else f"{int(row.n):,} / {int(row.events):,}"
                values = [n_events, "--", "--", "--", "--", status_label(row.model_status)]
            lines.append(f"{label} & {LABELS.get(row.cohort, row.cohort)} & " + " & ".join(values) + " " + r"\\")
        meta = pooled_contrasts.loc[pooled_contrasts["outcome_id"].eq(outcome)].iloc[0]
        meta_status = status_label(meta.model_status)
        # PASS_DESCRIPTIVE_K2 is a two-cohort pooling the frozen pipeline marks as
        # descriptive. The prespecified minimum for a cross-cohort estimate is three,
        # so no number is printed for it, as for the one-cohort mortality row.
        if meta.model_status == "PASS":
            lines.append(
                f"{label} & Random-effects synthesis & -- & -- & -- & {meta.pooled_ratio_of_risk_ratios:.2f} & "
                f"{meta.ci_low:.2f}--{meta.ci_high:.2f} & {meta_status} " + r"\\"
            )
        else:
            lines.append(
                f"{label} & Random-effects synthesis & -- & -- & -- & -- & -- & "
                f"{meta_status} " + r"\\"
            )
        lines.append(r"\addlinespace[2pt]")
    lines.extend([
        r"\bottomrule", r"\end{tabular}",
        r"\begin{flushleft}\footnotesize Direct ratios compare other work exit with retirement-linked exit by subtracting the two model coefficients and using their covariance. A ratio above 1 indicates greater risk after another work exit. Retirement-linked status was not measured in CHARLS or KLoSA; MHAS had no comparable interval.\end{flushleft}",
        r"\end{table*}", "",
    ])
    write_if_changed(GEN / "supp_table_work_exit.tex", "\n".join(lines))


def main() -> None:
    # Every manuscript figure is built by analysis/make_clinical_displays.py,
    # which owns the shared house style. This file writes the tables.
    table1()
    table2()
    supplementary_tables()
    extension_supplementary_tables()
    print("wrote positive multidomain manuscript displays")


if __name__ == "__main__":
    main()
