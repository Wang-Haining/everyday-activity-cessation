#!/usr/bin/env python3
"""Shared loading for E1/E2: harmonized episodes, primary risk sets, t-1 merge.

Respondent-level frames live only in memory. Nothing here writes rows.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CODE = ROOT / "scripts"
sys.path.insert(0, str(CODE))
DATA = Path(os.environ["HARMONIZED_ROOT"]).expanduser().resolve() if os.environ.get("HARMONIZED_ROOT") else None
META = ROOT / "config"
OUT = ROOT / "results" / "prediction" / "cohort"

from behavioral_contraction_biomarker_core import covariate_frame  # noqa: E402
from multidomain_behavioral_withdrawal_core import (  # noqa: E402
    coefficient_rows,
    fit_clustered,
    load_specs,
    multidomain_frame,
    prepare_model,
    support_gate,
)
from probe_behavior_outcome_feasibility import (  # noqa: E402
    binary,
    build_episodes,
    load_lookup,
    max_valid,
    numeric,
    outcome_fields,
    read_formal,
    row_count,
    source_status_and_elsa_dates,
)
from run_behavior_outcome_models import restricted_cubic_spline  # noqa: E402

SCOPE = "comparable_22_30_months"

# Published risk sets (full adjustment, score_categorical, 22-30 months):
# (n person-intervals, events, people)
PUB = {
    "mortality": {
        "elsa": (28927, 509, 8865), "hrs": (72785, 4554, 21202),
        "klosa": (17193, 611, 5827), "share": (43677, 1917, 30506),
    },
    "incident_any_adl": {
        "elsa": (23018, 2200, 7512), "hrs": (54427, 5028, 17198), "share": (31923, 2368, 22735),
    },
}
# Published cohort RRs (loss_1, loss_2plus) used to check the E2 risk sets.
PUB_RR = {
    "mortality": {
        "elsa": (1.785335, 3.547409), "hrs": (1.717691, 3.354014),
        "klosa": (1.824665, 3.625117), "share": (1.896338, 3.469139),
    },
    "incident_any_adl": {
        "elsa": (1.473496, 2.075735), "hrs": (1.455052, 2.040885), "share": (1.526432, 1.823908),
    },
}
MORT_COHORTS = ["elsa", "hrs", "klosa", "share"]
ADL_COHORTS = ["elsa", "hrs", "share"]


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    assert not ({"person_id", "id", "respondent_id", "prediction"} & set(frame.columns))
    frame.to_csv(path, index=False, lineterminator="\n")
    print(f"{path.name}: {len(frame)} aggregate rows")


@dataclass
class Cohort:
    name: str
    universe: dict[str, Any]
    pilot: dict[str, Any]
    episodes: pd.DataFrame
    behavior: pd.DataFrame
    formal: pd.DataFrame
    adl_fields: list[str] = field(default_factory=list)
    iadl_fields: list[str] = field(default_factory=list)
    gap: pd.DataFrame | None = None


def load_cohort(cohort: str) -> Cohort:
    global DATA
    value = os.environ.get("HARMONIZED_ROOT")
    if not value:
        raise ValueError("HARMONIZED_ROOT must name the harmonized data directory")
    DATA = Path(value).expanduser().resolve()
    if not DATA.is_dir():
        raise ValueError(f"HARMONIZED_ROOT is not a directory: {DATA}")
    universe, pilot = load_specs(
        META / "behavioral_transition_outcome_scan_universe_v0.1.json",
        META / "multidomain_behavioral_withdrawal_pilot_v0.1.json",
    )
    if cohort == "share":
        # Include EURO-D when loading SHARE health measures.
        universe["model_covariates_by_cohort"]["share"] = ["smokev", "smoken", "eurod"]
    lookup, _ = load_lookup(DATA, universe)
    formal, _audit = read_formal(DATA, universe, cohort, lookup)
    status, dates, _src = source_status_and_elsa_dates(DATA, universe, cohort)
    episodes, _intervals = build_episodes(formal, status, universe, cohort, dates)
    behavior = multidomain_frame(episodes, universe, pilot, cohort)
    if dates is not None:
        formal = formal.drop(columns=["interview_month_index"]).merge(
            dates, on=["person_id", "wave_int"], how="left", validate="one_to_one"
        )
    outcomes = universe["outcomes"]
    co = Cohort(
        cohort, universe, pilot, episodes, behavior, formal,
        adl_fields=outcome_fields("incident_any_adl", outcomes["incident_any_adl"], outcomes, cohort),
        iadl_fields=outcome_fields("incident_any_iadl", outcomes["incident_any_iadl"], outcomes, cohort),
    )
    add_tm1(co)
    return co


def add_tm1(co: Cohort) -> None:
    """Merge the interview before t0 (previous primary wave) with prefix tm1__."""
    waves = co.universe["cohorts"][co.name]["primary_waves"]
    prev = {waves[i]: waves[i - 1] for i in range(1, len(waves))}
    ep = co.episodes
    key = pd.DataFrame({"person_id": ep["person_id"].to_numpy(), "tm1_wave": ep["t0"].map(prev).to_numpy()})
    values = [c for c in co.formal.columns if c not in {"person_id", "wave_int"}]
    tm1 = co.formal.rename(columns={"wave_int": "tm1_wave", **{f: f"tm1__{f}" for f in values}})
    tm1["tm1_present"] = True
    key["tm1_wave"] = key["tm1_wave"].astype("Int64")
    tm1["tm1_wave"] = tm1["tm1_wave"].astype("Int64")
    merged = key.merge(tm1, on=["person_id", "tm1_wave"], how="left", validate="many_to_one")
    assert len(merged) == len(ep)
    merged.index = ep.index
    for column in merged.columns:
        if column in {"person_id"}:
            continue
        ep[column] = merged[column]
    ep["tm1_present"] = ep["tm1_present"].fillna(False).astype(bool)
    # Scheduled gap between wave medians (same construction as the pipeline's t1 -> t2 schedule).
    med = co.formal.groupby("wave_int", observed=True)["interview_month_index"].median()
    ep["tm1_scheduled_gap_months"] = [
        float(med.get(int(a), np.nan) - med.get(int(b), np.nan)) if pd.notna(b) else np.nan
        for a, b in zip(ep["t0"], ep["tm1_wave"])
    ]
    ep["tm1_observed_gap_months"] = ep["t0__interview_month_index"] - ep["tm1__interview_month_index"]
    ep["tm1_gap_22_30"] = ep["tm1_scheduled_gap_months"].between(22, 30)


def primary_set(co: Cohort, outcome_id: str, exposure_model: str = "score_categorical"):
    data, y, X, terms, is_binary, status = prepare_model(
        co.episodes, co.behavior, co.universe, co.pilot, co.name, outcome_id,
        SCOPE, "full", exposure_model, co.pilot["minimum_baseline_opportunities"],
    )
    assert is_binary
    return data, y, X, terms


def assert_published(co: Cohort, outcome_id: str, data: pd.DataFrame, y: pd.Series) -> dict[str, int]:
    n, events, people = int(len(data)), int(y.sum()), int(data["person_id"].nunique())
    expected = PUB[outcome_id][co.name]
    assert (n, events, people) == expected, (
        f"{co.name}/{outcome_id}: risk set (n, events, people) = {(n, events, people)} "
        f"differs from published {expected}"
    )
    return {"n": n, "events": events, "people": people}


def srh(co: Cohort, prefix: str) -> pd.Series:
    if f"{prefix}__shlt" not in co.episodes:
        return pd.Series(np.nan, index=co.episodes.index)
    return numeric(co.episodes[f"{prefix}__shlt"]).where(lambda x: x.isin([1, 2, 3, 4, 5]))


def adl_count(co: Cohort, prefix: str) -> pd.Series:
    return row_count(co.episodes, prefix, co.adl_fields)


def iadl_count(co: Cohort, prefix: str) -> pd.Series:
    return row_count(co.episodes, prefix, co.iadl_fields)


def state_t1(co: Cohort, short: str) -> pd.Series:
    b = co.behavior
    return (b[f"{short}_baseline"] - b[f"{short}_loss"] + b[f"{short}_gain"]).where(b["core_valid"])
