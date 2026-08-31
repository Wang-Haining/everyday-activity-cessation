#!/usr/bin/env python3
"""Aggregate-only Table 1 and participant-flow probe for the frailty manuscript.

Respondent-level data are read and transformed only on Quartz. Outputs contain
cohort-level counts, percentages, means, and standard deviations; no IDs or
respondent-level rows are written.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from probe_behavior_outcome_feasibility import binary, numeric, outcome_values

from behavioral_withdrawal_frailty_core import (
    COMMON_DISEASE_FIELDS,
    add_direction_categories,
    build_frailty_frame,
    delayed_outcome,
    extend_four_wave,
    load_extension_data,
    load_extension_specs,
    load_fi_long,
    load_lookup,
    load_source_components,
    multidomain_frame,
)
from cohort_core import baseline_multimorbidity


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--universe-config", required=True, type=Path)
    parser.add_argument("--multidomain-config", required=True, type=Path)
    parser.add_argument("--extension-config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--code-commit", required=True)
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError(f"empty descriptive output: {path.name}")
    frame.to_csv(path, index=False)
    os.chmod(path, 0o600)


def people(episodes: pd.DataFrame, mask: pd.Series) -> int:
    return int(episodes.loc[mask.fillna(False), "person_id"].nunique())


def mean_sd(series: pd.Series, mask: pd.Series) -> tuple[float, float, int]:
    values = pd.to_numeric(series.loc[mask], errors="coerce").dropna()
    if values.empty:
        return np.nan, np.nan, 0
    return float(values.mean()), float(values.std(ddof=1)), int(len(values))


def count_pct(series: pd.Series, mask: pd.Series, value: Any = 1) -> tuple[int, int, float]:
    values = series.loc[mask].dropna()
    denominator = int(len(values))
    count = int(values.eq(value).sum())
    return count, denominator, 100.0 * count / denominator if denominator else np.nan


def add_continuous(
    rows: list[dict[str, Any]], cohort: str, group: str, variable: str,
    series: pd.Series, mask: pd.Series,
) -> None:
    mean, sd, denominator = mean_sd(series, mask)
    rows.append({
        "cohort": cohort,
        "group": group,
        "variable": variable,
        "summary_type": "mean_sd",
        "mean": mean,
        "sd": sd,
        "count": np.nan,
        "denominator": denominator,
        "percent": np.nan,
    })


def add_binary(
    rows: list[dict[str, Any]], cohort: str, group: str, variable: str,
    series: pd.Series, mask: pd.Series, value: Any = 1,
) -> None:
    count, denominator, percent = count_pct(series, mask, value)
    rows.append({
        "cohort": cohort,
        "group": group,
        "variable": variable,
        "summary_type": "count_percent",
        "mean": np.nan,
        "sd": np.nan,
        "count": count,
        "denominator": denominator,
        "percent": percent,
    })


def table1_rows(
    episodes: pd.DataFrame,
    behavior: pd.DataFrame,
    universe: dict[str, Any],
    cohort: str,
    primary: pd.Series,
    frailty: pd.DataFrame | None,
) -> list[dict[str, Any]]:
    cc = universe["cohorts"][cohort]
    rows: list[dict[str, Any]] = []
    age = numeric(episodes[f"t1__{cc['age']}"])
    sex = numeric(episodes[f"t1__{cc['sex']}"]).where(lambda x: x.isin([1, 2]))
    education = numeric(episodes[f"t1__{cc['education']}"])
    income = numeric(episodes[f"t1__{cc['income']}"])
    education_rank = education.groupby(episodes["t1"]).rank(method="average", pct=True)
    income_rank = income.groupby(episodes["t1"]).rank(method="average", pct=True)
    smoking = binary(episodes["t1__smoken"])
    disease_count = baseline_multimorbidity(episodes, exclude=None, prefix="t1")

    loss_category = pd.Series(pd.NA, index=episodes.index, dtype="string")
    loss_category.loc[behavior["loss_count"].eq(0)] = "0"
    loss_category.loc[behavior["loss_count"].eq(1)] = "1"
    loss_category.loc[behavior["loss_count"].ge(2)] = "2_plus"
    groups = {
        "all": primary,
        "withdrawal_0": primary & loss_category.eq("0"),
        "withdrawal_1": primary & loss_category.eq("1"),
        "withdrawal_2_plus": primary & loss_category.eq("2_plus"),
    }
    for group, mask in groups.items():
        add_continuous(rows, cohort, group, "age_years", age, mask)
        add_binary(rows, cohort, group, "female", sex, mask, 2)
        add_continuous(rows, cohort, group, "education_rank", education_rank, mask)
        add_continuous(rows, cohort, group, "economic_rank", income_rank, mask)
        add_binary(rows, cohort, group, "current_smoking", smoking, mask)
        add_continuous(rows, cohort, group, "baseline_disease_count", disease_count, mask)
        add_binary(rows, cohort, group, "current_drinking", behavior["alcohol_t1"], mask)
        add_binary(rows, cohort, group, "current_regular_activity", behavior["activity_t1"], mask)
        add_binary(rows, cohort, group, "current_paid_work", behavior["work_t1"], mask)
        add_binary(rows, cohort, group, "alcohol_withdrawal", behavior["alcohol_loss"], mask)
        add_binary(rows, cohort, group, "activity_withdrawal", behavior["activity_loss"], mask)
        add_binary(rows, cohort, group, "paid_work_withdrawal", behavior["work_loss"], mask)
        for outcome_id, field in COMMON_DISEASE_FIELDS.items():
            add_binary(rows, cohort, group, f"baseline_{outcome_id.removeprefix('incident_')}", binary(episodes[f"t1__{field}"]), mask)

        if frailty is not None:
            add_binary(rows, cohort, group, "fried_robust", frailty["fried_category"], mask, "robust")
            add_binary(rows, cohort, group, "fried_prefrail", frailty["fried_category"], mask, "prefrail")
            add_binary(rows, cohort, group, "fried_frail", frailty["fried_category"], mask, "frail")
            add_continuous(rows, cohort, group, "frailty_index", frailty["fi_t1"], mask)
    return rows


def main() -> None:
    args = arguments()
    universe, multidomain, extension = load_extension_specs(
        args.universe_config, args.multidomain_config, args.extension_config
    )
    root = Path(extension["release_root"])
    lookup, _ = load_lookup(root, universe)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    flow: list[dict[str, Any]] = []
    table: list[dict[str, Any]] = []
    outcomes: list[dict[str, Any]] = []
    delayed: list[dict[str, Any]] = []

    for cohort in extension["specificity_cohorts"]:
        episodes, formal, status, _intervals, _audits = load_extension_data(
            root, universe, cohort, lookup
        )
        behavior = add_direction_categories(
            multidomain_frame(episodes, universe, multidomain, cohort)
        )
        comparable = episodes["comparable_window"].fillna(False)
        behavior_complete = comparable & behavior["core_valid"]
        primary = behavior_complete & behavior["baseline_engagement_count"].ge(1)

        frailty = None
        if cohort in extension["frailty_cohorts"]:
            source, _ = load_source_components(root, extension, cohort)
            fi_long, _ = load_fi_long(extension, cohort)
            frailty = build_frailty_frame(
                episodes, formal, source, fi_long, universe, extension, cohort
            )

        flow.extend([
            {"cohort": cohort, "stage": "source_respondents", "intervals": np.nan, "people": int(formal["person_id"].nunique())},
            {"cohort": cohort, "stage": "age_60_three_wave_intervals", "intervals": int(len(episodes)), "people": int(episodes["person_id"].nunique())},
            {"cohort": cohort, "stage": "comparable_22_30_month_intervals", "intervals": int(comparable.sum()), "people": people(episodes, comparable)},
            {"cohort": cohort, "stage": "three_behaviors_observed", "intervals": int(behavior_complete.sum()), "people": people(episodes, behavior_complete)},
            {"cohort": cohort, "stage": "primary_behavior_risk_set", "intervals": int(primary.sum()), "people": people(episodes, primary)},
        ])
        if frailty is not None:
            fried_set = primary & frailty["fried5_t1"].notna()
            fi_set = primary & frailty["fi_t1"].notna()
            flow.extend([
                {"cohort": cohort, "stage": "fried5_complete_risk_set", "intervals": int(fried_set.sum()), "people": people(episodes, fried_set)},
                {"cohort": cohort, "stage": "fi26_complete_risk_set", "intervals": int(fi_set.sum()), "people": people(episodes, fi_set)},
            ])

        table.extend(table1_rows(episodes, behavior, universe, cohort, primary, frailty))

        for outcome_id in extension["specificity_outcomes"]:
            outcome, coding = outcome_values(
                episodes, outcome_id, universe["outcomes"][outcome_id], universe["outcomes"], cohort
            )
            eligible = primary & outcome.notna()
            row: dict[str, Any] = {
                "cohort": cohort,
                "outcome_id": outcome_id,
                "coding_status": coding,
                "intervals": int(eligible.sum()),
                "people": people(episodes, eligible),
                "events": int(outcome.loc[eligible].sum()),
                "withdrawal_intervals": int((eligible & behavior["any_withdrawal"].eq(1)).sum()),
                "withdrawal_events": int(outcome.loc[eligible & behavior["any_withdrawal"].eq(1)].sum()),
            }
            if frailty is not None:
                fried = eligible & frailty["fried5_t1"].notna()
                row["fried_intervals"] = int(fried.sum())
                row["fried_people"] = people(episodes, fried)
                row["fried_events"] = int(outcome.loc[fried].sum())
            outcomes.append(row)

        episodes4 = extend_four_wave(episodes, formal, status, universe, cohort)
        for outcome_id in extension["delayed_outcomes"]:
            outcome = delayed_outcome(episodes4, outcome_id)
            eligible = behavior["core_valid"] & behavior["baseline_engagement_count"].ge(1) & outcome.notna()
            delayed.append({
                "cohort": cohort,
                "outcome_id": outcome_id,
                "intervals": int(eligible.sum()),
                "people": people(episodes, eligible),
                "events": int(outcome.loc[eligible].sum()),
                "withdrawal_intervals": int((eligible & behavior["any_withdrawal"].eq(1)).sum()),
                "withdrawal_events": int(outcome.loc[eligible & behavior["any_withdrawal"].eq(1)].sum()),
            })

    write_csv(args.output_dir / "manuscript-flow.csv", flow)
    write_csv(args.output_dir / "manuscript-table1-long.csv", table)
    write_csv(args.output_dir / "manuscript-outcome-risk-sets.csv", outcomes)
    write_csv(args.output_dir / "manuscript-delayed-risk-sets.csv", delayed)
    manifest = {
        "analysis_id": extension["analysis_id"],
        "code_commit": args.code_commit,
        "aggregate_only": True,
        "respondent_rows_exported": 0,
        "cohorts": extension["specificity_cohorts"],
        "outputs": [
            "manuscript-flow.csv",
            "manuscript-table1-long.csv",
            "manuscript-outcome-risk-sets.csv",
            "manuscript-delayed-risk-sets.csv",
        ],
    }
    path = args.output_dir / "manuscript-descriptive-manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    os.chmod(path, 0o600)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
