#!/usr/bin/env python3
"""Aggregate-only probe for the manuscript Table 1 and participant flow.

The script reads respondent-level data only inside the approved compute
environment and writes cohort-level counts and summaries. It never writes IDs
or respondent-level rows.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from probe_behavior_outcome_feasibility import (
    binary,
    load_lookup,
    numeric,
    read_formal,
)

from cohort_core import (
    COMMON_DISEASES,
    contraction_frame,
    load_episodes,
    load_specs,
    outcome_and_baseline,
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--universe-config", required=True, type=Path)
    parser.add_argument("--pilot-config", required=True, type=Path)
    parser.add_argument("--assay-config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def n_people(frame: pd.DataFrame, mask: pd.Series) -> int:
    return int(frame.loc[mask, "person_id"].nunique())


def mean_sd(series: pd.Series) -> tuple[float, float, int]:
    value = numeric(series).dropna()
    return float(value.mean()), float(value.std(ddof=1)), int(len(value))


def count_pct(series: pd.Series, value: float = 1.0) -> tuple[int, int, float]:
    valid = series.dropna()
    n = int(valid.eq(value).sum())
    denominator = int(len(valid))
    pct = 100.0 * n / denominator if denominator else np.nan
    return n, denominator, pct


def add_count(row: dict, name: str, series: pd.Series, value: float = 1.0) -> None:
    n, denominator, pct = count_pct(series, value)
    row[f"{name}_n"] = n
    row[f"{name}_denominator"] = denominator
    row[f"{name}_pct"] = pct


def candidate_pairs(formal: pd.DataFrame, waves: list[int]) -> pd.DataFrame:
    previous_wave = {waves[index]: waves[index - 1] for index in range(1, len(waves))}
    t1 = formal.loc[formal["wave_int"].isin(waves[1:-1]), ["person_id", "wave_int"]].copy()
    if t1.empty:
        return pd.DataFrame(columns=["person_id", "t0", "t1"])
    t1["t0"] = t1["wave_int"].map(previous_wave)
    t1 = t1.rename(columns={"wave_int": "t1"})
    observed = formal[["person_id", "wave_int"]].rename(columns={"wave_int": "t0"})
    return t1.merge(observed, on=["person_id", "t0"], how="inner", validate="many_to_one")


def main() -> None:
    args = arguments()
    universe, pilot, _assay = load_specs(
        args.universe_config, args.pilot_config, args.assay_config
    )
    root = Path(pilot["release_root"])
    lookup, _ = load_lookup(root, universe)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    flow_rows: list[dict] = []
    table_rows: list[dict] = []
    outcome_rows: list[dict] = []

    for cohort in universe["cohorts"]:
        cc = universe["cohorts"][cohort]
        formal, _ = read_formal(root, universe, cohort, lookup)
        pairs = candidate_pairs(formal, list(cc["primary_waves"]))
        flow = {
            "cohort": cohort,
            "primary_wave_observations": int(len(formal)),
            "primary_wave_people": int(formal["person_id"].nunique()),
            "consecutive_t0_t1_intervals": int(len(pairs)),
            "consecutive_t0_t1_people": int(pairs["person_id"].nunique()) if len(pairs) else 0,
            "age_60_intervals": 0,
            "age_60_people": 0,
            "comparable_intervals": 0,
            "comparable_people": 0,
            "behavior_complete_intervals": 0,
            "behavior_complete_people": 0,
        }

        if cohort not in pilot["candidate_cohorts"]:
            flow_rows.append(flow)
            continue

        episodes, _, _ = load_episodes(root, universe, cohort, lookup)
        behavior = contraction_frame(episodes, universe, cohort)
        comparable = episodes["comparable_window"].fillna(False)
        final = comparable & behavior["valid_behavior_pair"]
        flow.update({
            "age_60_intervals": int(len(episodes)),
            "age_60_people": int(episodes["person_id"].nunique()),
            "comparable_intervals": int(comparable.sum()),
            "comparable_people": n_people(episodes, comparable),
            "behavior_complete_intervals": int(final.sum()),
            "behavior_complete_people": n_people(episodes, final),
        })
        flow_rows.append(flow)

        if not final.any():
            continue

        selected = episodes.loc[final].copy()
        b = behavior.loc[final].copy()
        age = numeric(selected[f"t1__{cc['age']}"])
        sex = numeric(selected[f"t1__{cc['sex']}"]).where(lambda x: x.isin([1.0, 2.0]))
        education = numeric(selected[f"t1__{cc['education']}"]).where(
            lambda x: x.isin([1.0, 2.0, 3.0])
        )
        income = numeric(selected[f"t1__{cc['income']}"])
        income_rank = income.groupby(selected["t1"]).rank(method="average", pct=True)
        smoke_current = binary(selected["t1__smoken"])
        diseases = pd.DataFrame(
            {field: binary(selected[f"t1__{field}"]) for field in COMMON_DISEASES},
            index=selected.index,
        )
        complete_disease = diseases.notna().all(axis=1)
        disease_count = diseases.sum(axis=1).where(complete_disease)

        row = {
            "cohort": cohort,
            "people": int(selected["person_id"].nunique()),
            "person_intervals": int(len(selected)),
        }
        row["age_mean"], row["age_sd"], row["age_denominator"] = mean_sd(age)
        row["disease_count_mean"], row["disease_count_sd"], row[
            "disease_count_denominator"
        ] = mean_sd(disease_count)
        add_count(row, "female", sex, 2.0)
        add_count(row, "education_low", education, 1.0)
        add_count(row, "education_middle", education, 2.0)
        add_count(row, "education_high", education, 3.0)
        add_count(row, "lowest_income_quintile", income_rank.le(0.20).astype(float).where(income_rank.notna()))
        add_count(row, "current_smoking", smoke_current)
        add_count(row, "baseline_two_behaviors", b["baseline_engagement_count"], 2.0)
        add_count(row, "baseline_current_drinking", b["alcohol_transition"].isin(["1_to_0", "1_to_1"]).astype(float))
        add_count(row, "baseline_regular_activity", b["activity_transition"].isin(["1_to_0", "1_to_1"]).astype(float))
        add_count(row, "behavioral_contraction", b["any_contraction"])
        add_count(row, "alcohol_loss", b["alcohol_loss"])
        add_count(row, "activity_loss", b["activity_loss"])
        for field in ["diabe", "hibpe", "hearte", "stroke"]:
            add_count(row, f"baseline_{field}", diseases[field])
        table_rows.append(row)

        for outcome_id in pilot["primary_binary_outcomes"]:
            outcome, _, coding_status, _ = outcome_and_baseline(
                episodes, universe, cohort, outcome_id
            )
            risk = final & outcome.notna()
            outcome_rows.append({
                "cohort": cohort,
                "outcome_id": outcome_id,
                "coding_status": coding_status,
                "risk_set_intervals": int(risk.sum()),
                "risk_set_people": n_people(episodes, risk),
                "events": int(outcome.loc[risk].sum()),
                "contraction_intervals": int((risk & behavior["any_contraction"].eq(1)).sum()),
                "contraction_events": int(outcome.loc[risk & behavior["any_contraction"].eq(1)].sum()),
            })

    pd.DataFrame(flow_rows).to_csv(args.output_dir / "manuscript-flow.csv", index=False)
    pd.DataFrame(table_rows).to_csv(args.output_dir / "manuscript-table1.csv", index=False)
    pd.DataFrame(outcome_rows).to_csv(args.output_dir / "manuscript-outcome-flow.csv", index=False)
    manifest = {
        "aggregate_only": True,
        "respondent_rows_exported": 0,
        "cohorts_in_flow": len(flow_rows),
        "cohorts_in_table1": len(table_rows),
    }
    (args.output_dir / "manuscript-descriptive-manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(manifest)


if __name__ == "__main__":
    main()
