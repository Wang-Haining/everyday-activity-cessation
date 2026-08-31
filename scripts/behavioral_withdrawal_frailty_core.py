#!/usr/bin/env python3
"""Core ETL and estimand helpers for the frailty head-to-head extension.

Respondent-level frames created here are memory-only on Quartz. Public callers
must write aggregate counts or model summaries only.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from probe_behavior_outcome_feasibility import (
    baseline_eligible,
    binary,
    build_episodes,
    load_lookup,
    normalize_id,
    numeric,
    outcome_values,
    read_formal,
    sha256,
    source_status_and_elsa_dates,
)

from cohort_core import (
    add_base_design,
    baseline_multimorbidity,
    coefficient_rows,
    fit_clustered,
)
from multidomain_behavioral_withdrawal_core import multidomain_frame

COMMON_DISEASE_FIELDS = {
    "incident_diabetes": "diabe",
    "incident_stroke": "stroke",
    "incident_heart_disease": "hearte",
    "incident_hypertension": "hibpe",
    "incident_cancer": "cancre",
    "incident_arthritis": "arthre",
}


def stable_person_fold(cohort: str, person_id: str, folds: int) -> int:
    digest = hashlib.sha256(f"{cohort}|{person_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % folds


def load_extension_specs(
    universe_path: Path,
    multidomain_path: Path,
    extension_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    universe = json.loads(universe_path.read_text())
    multidomain = json.loads(multidomain_path.read_text())
    extension = json.loads(extension_path.read_text())
    expected = {
        str(universe_path): extension["universe_config_sha256"],
        str(multidomain_path): extension["multidomain_config_sha256"],
    }
    for path, frozen_hash in expected.items():
        if sha256(Path(path)) != frozen_hash:
            raise RuntimeError(f"frozen parent config drift: {path}")
    if len({universe["release_root"], multidomain["release_root"], extension["release_root"]}) != 1:
        raise RuntimeError("release-root mismatch")
    universe = json.loads(json.dumps(universe))
    universe["minimum_age_at_t1"] = extension["minimum_age_at_t1"]
    universe["comparable_outcome_window_months"] = extension["comparable_window_months"]
    universe["model_covariates_by_cohort"] = {}
    for cohort in extension["specificity_cohorts"]:
        fields = ["smokev", "smoken"]
        fields += extension["formal_frailty_fields"].get(cohort, [])
        universe["model_covariates_by_cohort"][cohort] = list(dict.fromkeys(fields))
    return universe, multidomain, extension


def load_extension_data(
    root: Path,
    universe: dict[str, Any],
    cohort: str,
    lookup: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[dict[str, Any]], dict[str, Any]]:
    formal, formal_audit = read_formal(root, universe, cohort, lookup)
    status, dates, source_audit = source_status_and_elsa_dates(root, universe, cohort)
    episodes, intervals = build_episodes(formal, status, universe, cohort, dates)
    return episodes, formal, status, intervals, {
        "formal": formal_audit,
        "status": source_audit,
    }


def add_direction_categories(behavior: pd.DataFrame) -> pd.DataFrame:
    result = behavior.copy()
    result["loss_count"] = result["loss_count_core"]
    result["gain_count"] = (
        result["alcohol_gain"] + result["activity_gain"] + result["work_gain"]
    ).where(result["core_valid"])
    direction = pd.Series(pd.NA, index=result.index, dtype="string")
    valid = result["core_valid"]
    direction.loc[valid & result["loss_count"].eq(0) & result["gain_count"].eq(0)] = "stable"
    direction.loc[valid & result["loss_count"].eq(0) & result["gain_count"].ge(1)] = "expansion"
    direction.loc[valid & result["loss_count"].ge(1) & result["gain_count"].ge(1)] = "mixed"
    direction.loc[valid & result["loss_count"].eq(1) & result["gain_count"].eq(0)] = "contraction_1"
    direction.loc[valid & result["loss_count"].ge(2) & result["gain_count"].eq(0)] = "contraction_2_plus"
    result["transition_direction"] = direction
    result["alcohol_t1"] = (
        result["alcohol_baseline"] - result["alcohol_loss"] + result["alcohol_gain"]
    ).where(valid)
    result["activity_t1"] = (
        result["activity_baseline"] - result["activity_loss"] + result["activity_gain"]
    ).where(valid)
    result["work_t1"] = (
        result["work_baseline"] - result["work_loss"] + result["work_gain"]
    ).where(valid)
    return result


def _stata_available_and_labels(path: Path) -> tuple[set[str], dict[str, str]]:
    with pd.io.stata.StataReader(path, convert_categoricals=False) as reader:
        labels = reader.variable_labels()
    return set(labels), labels


def _source_piece(
    path: Path,
    expected_hash: str,
    id_field: str,
    waves: list[int],
    patterns: dict[str, str],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if sha256(path) != expected_hash:
        raise RuntimeError(f"source hash drift: {path}")
    available, labels = _stata_available_and_labels(path)
    requested = {key: {wave: pattern.format(wave=wave) for wave in waves} for key, pattern in patterns.items()}
    fields = [field for values in requested.values() for field in values.values() if field in available]
    fields = list(dict.fromkeys(fields))
    if id_field not in available:
        raise RuntimeError(f"source ID missing: {path}")
    wide = pd.read_stata(
        path,
        columns=[id_field, *fields],
        convert_categoricals=False,
        preserve_dtypes=False,
    )
    wide["person_id"] = normalize_id(wide[id_field])
    if wide["person_id"].isna().any() or wide["person_id"].duplicated().any():
        raise RuntimeError(f"source ID not unique: {path}")
    pieces = []
    for wave in waves:
        piece = pd.DataFrame({"person_id": wide["person_id"], "wave_int": wave})
        for key, values in requested.items():
            field = values[wave]
            piece[key] = pd.to_numeric(wide[field], errors="coerce") if field in wide else np.nan
        pieces.append(piece)
    audit = {
        "path": str(path),
        "sha256": expected_hash,
        "rows": int(len(wide)),
        "id_unique": True,
        "available_fields": fields,
        "missing_requested_fields": sorted(
            field for values in requested.values() for field in values.values() if field not in available
        ),
        "variable_labels": {field: labels[field] for field in fields},
    }
    return pd.concat(pieces, ignore_index=True), audit


def load_source_components(
    root: Path,
    extension: dict[str, Any],
    cohort: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    spec = extension["source_component_fields"][cohort]
    primary, primary_audit = _source_piece(
        root / spec["relative_path"],
        spec["sha256"],
        spec["id"],
        spec["waves"],
        spec["patterns"],
    )
    audits: dict[str, Any] = {"primary": primary_audit}
    if "completion_supplement" in spec:
        sub = spec["completion_supplement"]
        supplemental, supplemental_audit = _source_piece(
            root / sub["relative_path"],
            sub["sha256"],
            spec["id"],
            spec["waves"],
            sub["patterns"],
        )
        primary = primary.merge(
            supplemental,
            on=["person_id", "wave_int"],
            how="left",
            validate="one_to_one",
            suffixes=("", "_supplement"),
        )
        for key in sub["patterns"]:
            supplemental_key = f"{key}_supplement"
            if supplemental_key in primary:
                if key in primary:
                    primary[key] = primary[supplemental_key].combine_first(primary[key])
                else:
                    primary[key] = primary[supplemental_key]
                primary = primary.drop(columns=supplemental_key)
        audits["completion_supplement"] = supplemental_audit
    if primary[["person_id", "wave_int"]].duplicated().any():
        raise RuntimeError(f"{cohort}: duplicate source component ID-wave")
    return primary, audits


def _align_long(
    long: pd.DataFrame,
    episodes: pd.DataFrame,
    value: str,
    wave_field: str = "t1",
) -> pd.Series:
    source = long.set_index(["person_id", "wave_int"])[value]
    index = pd.MultiIndex.from_arrays(
        [episodes["person_id"].astype("string"), pd.to_numeric(episodes[wave_field], errors="coerce")]
    )
    return pd.Series(source.reindex(index).to_numpy(), index=episodes.index)


def _prior_formal_value(
    formal: pd.DataFrame,
    episodes: pd.DataFrame,
    field: str,
    waves: list[int],
    maximum_prior_visits: int,
) -> pd.Series:
    source = formal.set_index(["person_id", "wave_int"])[field]
    positions = {wave: index for index, wave in enumerate(waves)}
    result = pd.Series(np.nan, index=episodes.index, dtype=float)
    for lag in range(1, maximum_prior_visits + 1):
        prior_wave = episodes["t1"].map(
            lambda wave: waves[positions[int(wave)] - lag]
            if int(wave) in positions and positions[int(wave)] >= lag
            else np.nan
        )
        index = pd.MultiIndex.from_arrays([episodes["person_id"].astype("string"), prior_wave])
        values = pd.to_numeric(pd.Series(source.reindex(index).to_numpy(), index=episodes.index), errors="coerce")
        values = values.where(values >= 0)
        result = result.combine_first(values)
    return result


def _valid_binary_or_missing(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    return values.where(values.isin([0.0, 1.0])).astype(float)


def _bmi_stratum(sex: pd.Series, bmi: pd.Series) -> pd.Series:
    result = pd.Series(pd.NA, index=bmi.index, dtype="string")
    female = sex.eq(2)
    male = sex.eq(1)
    for mask, bounds, prefix in [
        (female, [23, 26, 29], "f"),
        (male, [24, 26, 28], "m"),
    ]:
        result.loc[mask & bmi.le(bounds[0])] = f"{prefix}1"
        result.loc[mask & bmi.gt(bounds[0]) & bmi.le(bounds[1])] = f"{prefix}2"
        result.loc[mask & bmi.gt(bounds[1]) & bmi.le(bounds[2])] = f"{prefix}3"
        result.loc[mask & bmi.gt(bounds[2])] = f"{prefix}4"
    return result


def absolute_fried_weakness(sex: pd.Series, bmi: pd.Series, grip: pd.Series) -> pd.Series:
    strata = _bmi_stratum(sex, bmi)
    cutoffs = {
        "f1": 17.0, "f2": 17.3, "f3": 18.0, "f4": 21.0,
        "m1": 29.0, "m2": 30.0, "m3": 30.0, "m4": 32.0,
    }
    threshold = strata.map(cutoffs).astype(float)
    result = pd.Series(np.nan, index=grip.index, dtype=float)
    valid = grip.notna() & threshold.notna()
    result.loc[valid] = grip.loc[valid].le(threshold.loc[valid]).astype(float)
    return result


def apply_grip_completion_reason(
    weakness: pd.Series,
    grip: pd.Series,
    health: pd.Series,
    unable: pd.Series,
    refusal: pd.Series,
    equipment: pd.Series,
) -> pd.Series:
    result = weakness.copy()
    health_unable = health.eq(1) | unable.eq(1)
    nonhealth_missing = refusal.eq(1) | equipment.eq(1)
    result.loc[health_unable] = 1.0
    result.loc[nonhealth_missing & ~health_unable & grip.isna()] = np.nan
    return result


def wave_stratified_lowest_fifth(
    value: pd.Series,
    wave: pd.Series,
    sex: pd.Series,
    stratum: pd.Series,
    minimum_group_n: int = 20,
) -> pd.Series:
    data = pd.DataFrame({"value": value, "wave": wave, "sex": sex, "stratum": stratum})
    groups = [data["wave"], data["sex"], data["stratum"]]
    count = data["value"].groupby(groups, dropna=False).transform("count")
    cutoff = data["value"].groupby(groups, dropna=False).transform(lambda x: x.quantile(0.20))
    result = pd.Series(np.nan, index=value.index, dtype=float)
    valid = value.notna() & cutoff.notna() & count.ge(minimum_group_n)
    result.loc[valid] = value.loc[valid].le(cutoff.loc[valid]).astype(float)
    return result


def wave_stratified_highest_fifth(
    value: pd.Series,
    wave: pd.Series,
    sex: pd.Series,
    height: pd.Series,
    minimum_group_n: int = 20,
) -> pd.Series:
    data = pd.DataFrame({"value": value, "wave": wave, "sex": sex, "height": height})
    median_height = data["height"].groupby([data["wave"], data["sex"]]).transform("median")
    height_group = pd.Series(pd.NA, index=value.index, dtype="string")
    height_group.loc[height.notna() & median_height.notna() & height.le(median_height)] = "shorter"
    height_group.loc[height.notna() & median_height.notna() & height.gt(median_height)] = "taller"
    groups = [data["wave"], data["sex"], height_group]
    count = value.groupby(groups, dropna=False).transform("count")
    cutoff = value.groupby(groups, dropna=False).transform(lambda x: x.quantile(0.80))
    result = pd.Series(np.nan, index=value.index, dtype=float)
    valid = value.notna() & cutoff.notna() & count.ge(minimum_group_n)
    result.loc[valid] = value.loc[valid].ge(cutoff.loc[valid]).astype(float)
    return result


def _activity_inactive(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    result = pd.Series(np.nan, index=series.index, dtype=float)
    result.loc[values.isin([1, 2, 3])] = 0.0
    result.loc[values.isin([4, 5])] = 1.0
    return result


def _complete_sum(frame: pd.DataFrame, fields: list[str]) -> pd.Series:
    result = pd.Series(np.nan, index=frame.index, dtype=float)
    valid = frame[fields].notna().all(axis=1)
    result.loc[valid] = frame.loc[valid, fields].sum(axis=1)
    return result


def fried_scores(component_frame: pd.DataFrame) -> pd.DataFrame:
    components = ["shrinking", "exhaustion", "weakness", "slowness", "low_activity"]
    result = pd.DataFrame(index=component_frame.index)
    result["fried_components_observed"] = component_frame[components].notna().sum(axis=1)
    result["fried5_t1"] = _complete_sum(component_frame, components)
    result["fried4_no_activity_t1"] = _complete_sum(component_frame, components[:-1])
    category = pd.Series(pd.NA, index=component_frame.index, dtype="string")
    category.loc[result["fried5_t1"].eq(0)] = "robust"
    category.loc[result["fried5_t1"].between(1, 2)] = "prefrail"
    category.loc[result["fried5_t1"].ge(3)] = "frail"
    result["fried_category"] = category
    return result


def first_eligible_mask(episodes: pd.DataFrame, eligible: pd.Series) -> pd.Series:
    order = episodes.assign(_eligible=eligible.fillna(False)).sort_values(
        ["person_id", "t1"], kind="mergesort"
    )
    selected = order["_eligible"] & ~order.loc[order["_eligible"], "person_id"].duplicated().reindex(
        order.index, fill_value=False
    )
    return selected.reindex(episodes.index, fill_value=False)


def load_fi_long(extension: dict[str, Any], cohort: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    spec = extension["fi_sources"][cohort]
    path = Path(spec["path"])
    if sha256(path) != spec["sha256"]:
        raise RuntimeError(f"{cohort}: FI source hash drift")
    frame = pd.read_parquet(path, columns=["respondent_id", "wave", "fi_26", "fi_items_valid"])
    frame = frame.rename(columns={"respondent_id": "person_id", "wave": "wave_int"})
    frame["person_id"] = normalize_id(frame["person_id"])
    frame["wave_int"] = pd.to_numeric(frame["wave_int"], errors="coerce").astype("Int64")
    if frame[["person_id", "wave_int"]].duplicated().any():
        raise RuntimeError(f"{cohort}: FI source duplicate ID-wave")
    return frame, {"path": str(path), "sha256": spec["sha256"], "rows": int(len(frame))}


def build_frailty_frame(
    episodes: pd.DataFrame,
    formal: pd.DataFrame,
    source: pd.DataFrame,
    fi_long: pd.DataFrame,
    universe: dict[str, Any],
    extension: dict[str, Any],
    cohort: str,
) -> pd.DataFrame:
    result = pd.DataFrame(index=episodes.index)
    waves = universe["cohorts"][cohort]["primary_waves"]
    sex = numeric(episodes[f"t1__{universe['cohorts'][cohort]['sex']}"]).where(lambda x: x.isin([1, 2]))
    wave = pd.to_numeric(episodes["t1"], errors="coerce")
    measurement = {
        "charls": ("bmi", "mweight", "mheight"),
        "elsa": ("mbmi", "mweight", "mheight"),
        "hrs": ("mbmi", "mweight", "mheight"),
        "share": ("bmi", "weight", "height"),
    }[cohort]
    bmi_field, weight_field, height_field = measurement
    bmi = numeric(episodes[f"t1__{bmi_field}"], [10, 80])
    weight = numeric(episodes[f"t1__{weight_field}"], [20, 300])
    height = numeric(episodes[f"t1__{height_field}"], [1.0, 2.3])
    prior_weight = _prior_formal_value(
        formal,
        episodes,
        weight_field,
        waves,
        extension["fried"]["shrinking"]["maximum_prior_scheduled_visits"],
    ).where(lambda x: x.between(20, 300))
    weight_loss_fraction = (prior_weight - weight) / prior_weight
    shrinking = pd.Series(np.nan, index=episodes.index, dtype=float)
    shrink_observed = bmi.notna() | (weight.notna() & prior_weight.notna())
    shrinking.loc[shrink_observed] = (
        bmi.lt(extension["fried"]["shrinking"]["bmi_threshold"])
        | weight_loss_fraction.gt(extension["fried"]["shrinking"]["weight_loss_fraction"])
    ).loc[shrink_observed].astype(float)
    result["shrinking"] = shrinking
    result["weight_loss_fraction"] = weight_loss_fraction

    if cohort == "charls":
        effort = pd.to_numeric(episodes["t1__effortl"], errors="coerce")
        going = pd.to_numeric(episodes["t1__goingl"], errors="coerce")
        valid_effort = effort.isin([1, 2, 3, 4])
        valid_going = going.isin([1, 2, 3, 4])
        exhaustion = pd.Series(np.nan, index=episodes.index, dtype=float)
        observed = valid_effort | valid_going
        positive = effort.isin(extension["fried"]["exhaustion"]["charls_positive_values"]) | going.isin(
            extension["fried"]["exhaustion"]["charls_positive_values"]
        )
        exhaustion.loc[observed] = positive.loc[observed].astype(float)
    else:
        keys = {
            "elsa": ("exhaustion_effort", "exhaustion_going"),
            "hrs": ("exhaustion_effort", "exhaustion_going"),
            "share": ("exhaustion_fatigue",),
        }[cohort]
        values = [_valid_binary_or_missing(_align_long(source, episodes, key)) for key in keys]
        observed = pd.concat(values, axis=1).notna().any(axis=1)
        positive = pd.concat(values, axis=1).eq(1).any(axis=1)
        exhaustion = pd.Series(np.nan, index=episodes.index, dtype=float)
        exhaustion.loc[observed] = positive.loc[observed].astype(float)
    result["exhaustion"] = exhaustion

    left = numeric(episodes["t1__lgrip"], [1, 100])
    right = numeric(episodes["t1__rgrip"], [1, 100])
    grip = pd.concat([left, right], axis=1).max(axis=1, skipna=True)
    grip = grip.where(left.notna() | right.notna())
    health = _valid_binary_or_missing(_align_long(source, episodes, "grip_health")) if "grip_health" in source else pd.Series(np.nan, index=episodes.index)
    unable = _valid_binary_or_missing(_align_long(source, episodes, "grip_unable")) if "grip_unable" in source else pd.Series(np.nan, index=episodes.index)
    refusal = _valid_binary_or_missing(_align_long(source, episodes, "grip_refusal")) if "grip_refusal" in source else pd.Series(np.nan, index=episodes.index)
    equipment = _valid_binary_or_missing(_align_long(source, episodes, "grip_equipment")) if "grip_equipment" in source else pd.Series(np.nan, index=episodes.index)
    if extension["fried"]["weakness"][cohort]["method"].startswith("cohort_wave"):
        weakness = wave_stratified_lowest_fifth(grip, wave, sex, _bmi_stratum(sex, bmi))
    else:
        weakness = absolute_fried_weakness(sex, bmi, grip)
    health_unable = health.eq(1) | unable.eq(1)
    nonhealth_missing = refusal.eq(1) | equipment.eq(1)
    weakness = apply_grip_completion_reason(weakness, grip, health, unable, refusal, equipment)
    result["weakness"] = weakness
    result["grip_kg"] = grip
    result["grip_health_unable"] = health_unable.astype(float).where(health.notna() | unable.notna())
    result["grip_refusal_or_equipment"] = nonhealth_missing.astype(float).where(refusal.notna() | equipment.notna())

    if cohort == "share":
        walk = binary(episodes["t1__walk100a"])
        climb = binary(episodes["t1__clim1a"])
        observed = walk.notna() & climb.notna()
        slowness = pd.Series(np.nan, index=episodes.index, dtype=float)
        slowness.loc[observed] = (walk.eq(1) | climb.eq(1)).loc[observed].astype(float)
    else:
        walk_seconds = numeric(episodes["t1__wspeed"], extension["fried"]["slowness"]["walk_seconds_valid_range"])
        slowness = wave_stratified_highest_fifth(walk_seconds, wave, sex, height)
        result["walk_seconds"] = walk_seconds
    result["slowness"] = slowness

    if cohort == "charls":
        activity_values = pd.concat(
            [binary(episodes[f"t1__{field}"]) for field in ["vgact_c", "mdact_c", "ltact_c"]],
            axis=1,
        )
        observed = activity_values.notna().all(axis=1)
        low_activity = pd.Series(np.nan, index=episodes.index, dtype=float)
        low_activity.loc[observed] = activity_values.loc[observed].eq(0).all(axis=1).astype(float)
    elif cohort in {"elsa", "hrs"}:
        fields = ["vgactx_e", "mdactx_e", "ltactx_e"] if cohort == "elsa" else ["vgactx", "mdactx", "ltactx"]
        inactive = pd.concat([_activity_inactive(episodes[f"t1__{field}"]) for field in fields], axis=1)
        observed = inactive.notna().all(axis=1)
        low_activity = pd.Series(np.nan, index=episodes.index, dtype=float)
        low_activity.loc[observed] = inactive.loc[observed].eq(1).all(axis=1).astype(float)
    else:
        low_activity = _activity_inactive(episodes["t1__mdactx"])
    result["low_activity"] = low_activity

    scores = fried_scores(result)
    result = pd.concat([result, scores], axis=1)

    fi_t0 = _align_long(fi_long, episodes, "fi_26", "t0")
    fi_t1 = _align_long(fi_long, episodes, "fi_26", "t1")
    result["fi_t0"] = pd.to_numeric(fi_t0, errors="coerce").where(lambda x: x.between(0, 1))
    result["fi_t1"] = pd.to_numeric(fi_t1, errors="coerce").where(lambda x: x.between(0, 1))
    result["fi_change"] = result["fi_t1"] - result["fi_t0"]
    result["fi_worsening_005"] = result["fi_change"].ge(0.05).astype(float).where(result["fi_change"].notna())

    if cohort == "share" and "appetite_loss" in source:
        result["share_appetite_loss"] = _valid_binary_or_missing(_align_long(source, episodes, "appetite_loss"))
    return result


def build_context_frame(
    episodes: pd.DataFrame,
    frailty: pd.DataFrame,
    universe: dict[str, Any],
    cohort: str,
) -> pd.DataFrame:
    result = pd.DataFrame(index=episodes.index)
    if cohort in {"elsa", "hrs", "share"}:
        health0 = numeric(episodes["t0__shlt"]).where(lambda x: x.isin([1, 2, 3, 4, 5]))
        health1 = numeric(episodes["t1__shlt"]).where(lambda x: x.isin([1, 2, 3, 4, 5]))
        valid = health0.notna() & health1.notna()
        result["self_rated_health_worsening"] = health1.gt(health0).astype(float).where(valid)
    else:
        result["self_rated_health_worsening"] = np.nan

    if cohort == "charls":
        nights = numeric(episodes["t1__hspnite"], [0, 365])
        result["transition_hospitalization"] = nights.gt(0).astype(float).where(nights.notna())
    elif cohort == "hrs":
        result["transition_hospitalization"] = binary(episodes["t1__hosp"])
    elif cohort == "share":
        result["transition_hospitalization"] = binary(episodes["t1__hosp1y"])
    else:
        result["transition_hospitalization"] = np.nan

    if cohort == "charls":
        c0 = numeric(episodes["t0__cesd10"], [0, 30])
        c1 = numeric(episodes["t1__cesd10"], [0, 30])
        threshold = 3.0
    elif cohort in {"elsa", "hrs"}:
        c0 = numeric(episodes["t0__cesd"], [0, 8])
        c1 = numeric(episodes["t1__cesd"], [0, 8])
        threshold = 1.0
    else:
        c0 = c1 = pd.Series(np.nan, index=episodes.index)
        threshold = 1.0
    delta = c1 - c0
    result["cesd_worsening"] = delta.ge(threshold).astype(float).where(delta.notna())
    result["bmi_or_weight_loss"] = frailty["shrinking"]

    disease_values = []
    for field in COMMON_DISEASE_FIELDS.values():
        at0 = binary(episodes[f"t0__{field}"])
        at1 = binary(episodes[f"t1__{field}"])
        disease_values.append((at0.eq(0) & at1.eq(1)).astype(float).where(at0.notna() & at1.notna()))
    disease_frame = pd.concat(disease_values, axis=1)
    observed = disease_frame.notna().all(axis=1)
    result["incident_non_target_disease"] = disease_frame.eq(1).any(axis=1).astype(float).where(observed)
    result["fi_change"] = frailty["fi_change"]
    result["fi_worsening_005"] = frailty["fi_worsening_005"]
    binary_context = [
        "self_rated_health_worsening", "transition_hospitalization", "cesd_worsening",
        "bmi_or_weight_loss", "incident_non_target_disease", "fi_worsening_005",
    ]
    result["concurrent_health_change"] = result[binary_context].eq(1).any(axis=1).astype(float).where(
        result[binary_context].notna().any(axis=1)
    )
    return result


def extend_four_wave(
    episodes: pd.DataFrame,
    formal: pd.DataFrame,
    status: pd.DataFrame,
    universe: dict[str, Any],
    cohort: str,
) -> pd.DataFrame:
    waves = universe["cohorts"][cohort]["primary_waves"]
    next_wave = {waves[index]: waves[index + 1] for index in range(len(waves) - 1)}
    result = episodes.copy()
    result["t3_wave"] = result["outcome_wave"].map(next_wave)
    values = [column for column in formal.columns if column not in {"person_id", "wave_int"}]
    t3 = formal.rename(columns={"wave_int": "t3_wave", **{field: f"t3__{field}" for field in values}})
    result = result.merge(t3, on=["person_id", "t3_wave"], how="left", validate="many_to_one")
    delayed_status = status.rename(columns={"outcome_wave": "t3_wave", "next_status": "t3_status"})
    result = result.merge(delayed_status, on=["person_id", "t3_wave"], how="left", validate="many_to_one")
    t2_date = result["t2__interview_month_index"]
    t3_date = result["t3__interview_month_index"]
    result["delayed_followup_months"] = t3_date - t2_date
    low, high = universe["comparable_outcome_window_months"]
    result["delayed_comparable_window"] = result["delayed_followup_months"].between(low, high)
    return result


def delayed_outcome(
    episodes4: pd.DataFrame,
    outcome_id: str,
) -> pd.Series:
    field = COMMON_DISEASE_FIELDS[outcome_id]
    at1 = binary(episodes4[f"t1__{field}"])
    at2 = binary(episodes4[f"t2__{field}"])
    at3 = binary(episodes4[f"t3__{field}"])
    result = pd.Series(np.nan, index=episodes4.index, dtype=float)
    valid = at1.eq(0) & at2.eq(0) & at3.notna() & episodes4["delayed_comparable_window"].fillna(False)
    result.loc[valid] = at3.loc[valid]
    return result


def competing_outcome(
    episodes: pd.DataFrame,
    universe: dict[str, Any],
    cohort: str,
    outcome_id: str,
) -> tuple[pd.Series, pd.Series]:
    spec = universe["outcomes"][outcome_id]
    primary, _ = outcome_values(
        episodes, outcome_id, spec, universe["outcomes"], cohort,
    )
    # The risk set is frozen from the first interview, before anyone can have
    # died. Writing the event over every missing value instead added the deaths
    # that were never eligible for this outcome and left the equally ineligible
    # survivors out, which is selection on the outcome's own competing event.
    eligible = baseline_eligible(
        episodes, outcome_id, spec, universe["outcomes"], cohort,
    )
    assert primary[~eligible].isna().all(), (
        f"{outcome_id}: an interval outside the baseline risk set carries a value"
    )
    death = episodes["next_status"].isin(universe["cohorts"][cohort]["death_codes"])
    composite = primary.copy()
    composite.loc[death & eligible] = 1.0
    state = pd.Series("unknown", index=episodes.index, dtype="string")
    state.loc[primary.eq(0)] = "alive_no_diagnosis"
    state.loc[primary.eq(1)] = "alive_new_diagnosis"
    state.loc[death & eligible] = "death"
    state.loc[~eligible] = "not_in_baseline_risk_set"
    return composite, state


def routine_data_and_design(
    episodes: pd.DataFrame,
    behavior: pd.DataFrame,
    universe: dict[str, Any],
    extension: dict[str, Any],
    cohort: str,
    outcome_id: str,
    outcome: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cc = universe["cohorts"][cohort]
    data = pd.DataFrame(index=episodes.index)
    data["person_id"] = episodes["person_id"].astype("string")
    data["t1_wave"] = pd.to_numeric(episodes["t1"], errors="coerce")
    data["age"] = numeric(episodes[f"t1__{cc['age']}"])
    data["sex"] = numeric(episodes[f"t1__{cc['sex']}"]).where(lambda x: x.isin([1, 2]))
    education = numeric(episodes[f"t1__{cc['education']}"])
    income = numeric(episodes[f"t1__{cc['income']}"])
    data["education_rank"] = education.groupby(episodes["t1"]).rank(method="average", pct=True)
    data["income_rank"] = income.groupby(episodes["t1"]).rank(method="average", pct=True)
    data["smoke_ever"] = binary(episodes["t1__smokev"])
    data["smoke_current"] = binary(episodes["t1__smoken"])
    exclude = COMMON_DISEASE_FIELDS.get(outcome_id)
    data["baseline_multimorbidity"] = baseline_multimorbidity(episodes, exclude=exclude, prefix="t1")
    data["baseline_engagement_count"] = behavior["baseline_engagement_count"]
    data["alcohol_t1"] = behavior["alcohol_t1"]
    data["activity_t1"] = behavior["activity_t1"]
    data["work_t1"] = behavior["work_t1"]
    data["loss_1"] = behavior["loss_1"]
    data["loss_2plus"] = behavior["loss_2plus"]
    data["any_withdrawal"] = behavior["any_withdrawal"]
    data["outcome"] = outcome
    X = add_base_design(data.dropna(subset=["age", "sex", "t1_wave", "baseline_engagement_count"]), extension, "full")
    X["alcohol_t1"] = data.loc[X.index, "alcohol_t1"].astype(float)
    X["activity_t1"] = data.loc[X.index, "activity_t1"].astype(float)
    X["work_t1"] = data.loc[X.index, "work_t1"].astype(float)
    return data, X


def c_statistic(y: np.ndarray, prediction: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    positive = y == 1
    negative = y == 0
    n_positive = int(positive.sum())
    n_negative = int(negative.sum())
    if not n_positive or not n_negative:
        return float("nan")
    ranks = pd.Series(prediction).rank(method="average").to_numpy()
    return float((ranks[positive].sum() - n_positive * (n_positive + 1) / 2) / (n_positive * n_negative))


def standardized_risks(
    fit: Any,
    X: pd.DataFrame,
    scenarios: dict[str, dict[str, float]],
) -> list[dict[str, Any]]:
    rows = []
    for scenario, settings in scenarios.items():
        design = X.copy()
        for field, value in settings.items():
            if field in design:
                design[field] = value
        prediction = np.clip(np.asarray(fit.predict(design), dtype=float), 0, 1)
        rows.append({
            "scenario": scenario,
            "standardized_risk": float(prediction.mean()),
            "prediction_n": int(len(prediction)),
        })
    return rows


__all__ = [
    "COMMON_DISEASE_FIELDS", "absolute_fried_weakness", "add_direction_categories",
    "apply_grip_completion_reason",
    "build_context_frame", "build_frailty_frame", "c_statistic", "coefficient_rows",
    "competing_outcome", "delayed_outcome", "extend_four_wave", "fit_clustered",
    "first_eligible_mask", "fried_scores", "load_extension_data", "load_extension_specs", "load_fi_long", "load_lookup",
    "load_source_components", "multidomain_frame", "routine_data_and_design", "sha256",
    "stable_person_fold", "standardized_risks", "wave_stratified_highest_fifth",
    "wave_stratified_lowest_fifth",
]
