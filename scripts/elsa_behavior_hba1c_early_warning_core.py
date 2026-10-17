#!/usr/bin/env python3
"""Shared row-safe construction for the ELSA behavioral-change HbA1c pilot."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from behavioral_contraction_biomarker_core import (
    assay_values,
    contraction_frame,
    covariate_frame,
    load_episodes,
    load_specs,
)
from probe_behavior_outcome_feasibility import binary, load_lookup, sha256
from probe_paper3_alcohol_diabetes_mechanisms import load_elsa_labs


def load_early_specs(
    universe_path: Path,
    pilot_path: Path,
    assay_path: Path,
    early_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    universe, pilot, assay = load_specs(universe_path, pilot_path, assay_path)
    early = json.loads(early_path.read_text())
    parents = early["parent_sha256"]
    expected = {
        "universe_config": sha256(universe_path),
        "pilot_config": sha256(pilot_path),
        "assay_config": sha256(assay_path),
    }
    if parents != expected:
        raise RuntimeError(f"early-warning parent config drift: {parents} != {expected}")
    if early["release_root"] != universe["release_root"]:
        raise RuntimeError("early-warning release root mismatch")
    return universe, pilot, assay, early


def glycemic_category(values: pd.Series, prediabetes: float, diabetes: float) -> pd.Series:
    result = pd.Series(np.nan, index=values.index, dtype=float)
    result.loc[values.lt(prediabetes)] = 0.0
    result.loc[values.ge(prediabetes) & values.lt(diabetes)] = 1.0
    result.loc[values.ge(diabetes)] = 2.0
    return result


def build_frame(
    root: Path,
    universe: dict[str, Any],
    pilot: dict[str, Any],
    assay_config: dict[str, Any],
    early: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    cohort = "elsa"
    lookup, lookup_path = load_lookup(root, universe)
    episodes, interval_rows, source_audit = load_episodes(root, universe, cohort, lookup)
    behavior = contraction_frame(episodes, universe, cohort)
    assays, assay_audit = assay_values(root, universe, assay_config, pilot, cohort, episodes)
    labs, lab_audit = load_elsa_labs(
        root, assay_config["biomarker_modules"]["elsa_landmark_repeated"]
    )
    source = labs.set_index(["person_id", "lab_wave"])
    follow_index = pd.MultiIndex.from_arrays(
        [episodes["person_id"].astype("string"), episodes["outcome_wave"]]
    )

    data = covariate_frame(episodes, behavior, universe, cohort, "incident_diabetes")
    data["t0_wave"] = pd.to_numeric(episodes["t0"], errors="coerce")
    data["t2_wave"] = pd.to_numeric(episodes["outcome_wave"], errors="coerce")
    data["observed_followup_months"] = pd.to_numeric(
        episodes["observed_followup_months"], errors="coerce"
    )
    data["valid_behavior_pair"] = behavior["valid_behavior_pair"]
    for field in [
        "any_contraction",
        "loss_count_2",
        "alcohol_loss",
        "activity_loss",
        "baseline_engagement_count",
    ]:
        data[field] = behavior[field]
    data["baseline_hba1c"] = assays["hba1c"]
    data["followup_hba1c"] = pd.Series(
        source["hba1c_percent"].reindex(follow_index).to_numpy(),
        index=episodes.index,
        dtype=float,
    )
    data["baseline_fasting_glucose"] = pd.Series(
        source["fasting_glucose_mmol_l"]
        .reindex(
            pd.MultiIndex.from_arrays(
                [episodes["person_id"].astype("string"), episodes["t0"]]
            )
        )
        .to_numpy(),
        index=episodes.index,
        dtype=float,
    )
    data["followup_fasting_glucose"] = pd.Series(
        source["fasting_glucose_mmol_l"].reindex(follow_index).to_numpy(),
        index=episodes.index,
        dtype=float,
    )
    data["diagnosed_diabetes_t1"] = binary(episodes["t1__diabe"])

    landmark = pd.Series(False, index=episodes.index)
    for item in early["landmarks"]:
        landmark |= (
            episodes["t0"].eq(item["t0"])
            & episodes["t1"].eq(item["t1"])
            & episodes["outcome_wave"].eq(item["t2"])
        )
    data["eligible_landmark"] = landmark

    thresholds = early["hba1c_thresholds_percent"]
    baseline_category = glycemic_category(
        data["baseline_hba1c"], thresholds["prediabetes"], thresholds["diabetes_range"]
    )
    followup_category = glycemic_category(
        data["followup_hba1c"], thresholds["prediabetes"], thresholds["diabetes_range"]
    )
    data["baseline_glycemic_category"] = baseline_category
    data["followup_glycemic_category"] = followup_category
    data["glycemic_category_progression"] = (
        followup_category.gt(baseline_category).astype(float)
    ).where(baseline_category.lt(2) & followup_category.notna())
    data["incident_diabetes_range_hba1c"] = (
        data["followup_hba1c"].ge(thresholds["diabetes_range"]).astype(float)
    ).where(data["baseline_hba1c"].lt(thresholds["diabetes_range"]) & data["followup_hba1c"].notna())
    data["incident_prediabetes_or_higher"] = (
        data["followup_hba1c"].ge(thresholds["prediabetes"]).astype(float)
    ).where(data["baseline_hba1c"].lt(thresholds["prediabetes"]) & data["followup_hba1c"].notna())
    glucose_threshold = early["fasting_glucose_diabetes_mmol_l"]
    data["incident_fasting_glucose_diabetes_range"] = (
        data["followup_fasting_glucose"].ge(glucose_threshold).astype(float)
    ).where(
        data["baseline_fasting_glucose"].lt(glucose_threshold)
        & data["followup_fasting_glucose"].notna()
    )

    primary_base = (
        data["eligible_landmark"]
        & data["valid_behavior_pair"]
        & data["baseline_engagement_count"].ge(early["minimum_baseline_engagement"])
        & data["baseline_hba1c"].lt(thresholds["diabetes_range"])
        & data["diagnosed_diabetes_t1"].eq(0)
    )
    data["primary_base"] = primary_base

    audit = {
        "formal": source_audit,
        "intervals": interval_rows,
        "assay": assay_audit,
        "labs": lab_audit,
        "lookup_path": str(lookup_path),
        "lookup_sha256": sha256(lookup_path),
        "episode_rows": int(len(data)),
    }
    return data, audit


def outcome_mask(data: pd.DataFrame, outcome: str) -> pd.Series:
    base = data["primary_base"].copy()
    if outcome == "incident_prediabetes_or_higher":
        base &= data["baseline_hba1c"].lt(5.7)
    if outcome == "incident_fasting_glucose_diabetes_range":
        base &= data["baseline_fasting_glucose"].lt(7.0)
    return base & data[outcome].notna()
