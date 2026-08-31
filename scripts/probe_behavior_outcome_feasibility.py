#!/usr/bin/env python3
"""Aggregate-only feasibility scan for frozen behavior x outcome cells.

All respondent-level data remain in memory on Quartz. The only outputs are
cell counts, support labels, interval summaries, and provenance manifests.
No effect model is fit in this phase.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

FORBIDDEN_FORMAL_OUTCOME = "iwstat"
BASIC_PARAMETER_COUNT = 7
FULL_PARAMETER_COUNT = 11


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--universe-commit", required=True)
    parser.add_argument("--universe-commit-time", required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_id(series: pd.Series) -> pd.Series:
    result = series.astype("string").str.strip()
    result = result.replace({"": pd.NA, "nan": pd.NA, "<NA>": pd.NA})
    numeric = pd.to_numeric(result, errors="coerce")
    integer_like = numeric.notna() & np.isclose(numeric, np.round(numeric))
    result.loc[integer_like] = numeric.loc[integer_like].round().astype("Int64").astype("string")
    return result


def numeric(series: pd.Series, valid_range: Iterable[float] | None = None) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").astype(float)
    values = values.where(values >= 0)
    if valid_range is not None:
        low, high = list(valid_range)
        values = values.where(values.between(float(low), float(high)))
    return values


def binary(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").astype(float)
    return values.where(values.isin([0.0, 1.0]))


def json_ready(value):
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return [json_ready(item) for item in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise RuntimeError(f"refusing to write empty output: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.chmod(path, 0o600)


def load_lookup(root: Path, config: dict[str, Any]) -> tuple[pd.DataFrame, Path]:
    path = root / config["lookup_relative_path"]
    if sha256(path) != config["lookup_sha256"]:
        raise RuntimeError("live lookup hash drift")
    lookup = pd.read_csv(path, dtype="string", encoding="utf-8-sig")
    required = {"cohort", "canonical_variable", "column_name_in_csv"}
    if missing := required.difference(lookup.columns):
        raise RuntimeError(f"lookup missing columns: {sorted(missing)}")
    return lookup, path


def cohort_lookup(lookup: pd.DataFrame, cohort: str) -> pd.DataFrame:
    frame = lookup.loc[lookup["cohort"].eq(cohort)].copy()
    if frame.empty or frame["canonical_variable"].duplicated().any():
        raise RuntimeError(f"{cohort}: missing or nonunique live lookup")
    return frame


def assert_ordered_header(formal_path: Path, frame: pd.DataFrame) -> None:
    with formal_path.open("r", encoding="utf-8-sig", newline="") as handle:
        observed = next(csv.reader(handle))
    if observed != frame["column_name_in_csv"].tolist():
        raise RuntimeError(f"{formal_path.name}: ordered header differs from lookup")


def outcome_fields(
    outcome_id: str, spec: dict[str, Any], outcomes: dict[str, Any], cohort: str
) -> list[str]:
    if "shares_fields_with" in spec:
        parent = spec["shares_fields_with"]
        return outcome_fields(parent, outcomes[parent], outcomes, cohort)
    if "field" in spec:
        return [spec["field"]] if cohort in spec.get("cohorts", []) else []
    fields = spec.get("fields", {})
    if isinstance(fields, list):
        return list(fields) if cohort in spec.get("cohorts", []) else []
    return list(fields.get(cohort, []))


def selected_canonicals(config: dict[str, Any], cohort: str) -> list[str]:
    cc = config["cohorts"][cohort]
    fields = [cc["id"], cc["wave"], cc["age"], cc["sex"], cc["education"], cc["income"]]
    if cc.get("interview_year"):
        fields.append(cc["interview_year"])
    if cc.get("interview_month"):
        fields.append(cc["interview_month"])
    for spec in config["behavioral_transitions"].values():
        fields.extend(spec.get("fields", {}).get(cohort, []))
    for outcome_id, spec in config["outcomes"].items():
        if spec["type"] != "binary_source_status":
            fields.extend(outcome_fields(outcome_id, spec, config["outcomes"], cohort))
        if cohort == "klosa" and "klosa_module" in spec:
            fields.extend(spec["klosa_module"]["fields_by_waves"].values())
    fields.extend(config.get("model_covariates_by_cohort", {}).get(cohort, []))
    fields = list(dict.fromkeys(fields))
    if FORBIDDEN_FORMAL_OUTCOME in fields:
        raise RuntimeError("formal iwstat is forbidden")
    return fields


def read_formal(
    root: Path, config: dict[str, Any], cohort: str, lookup: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, Any]]:
    cc = config["cohorts"][cohort]
    c_lookup = cohort_lookup(lookup, cohort)
    mapping = dict(zip(c_lookup["canonical_variable"], c_lookup["column_name_in_csv"]))
    path = root / config["formal_subdir"] / cc["formal_file"]
    assert_ordered_header(path, c_lookup)
    actual_hash = sha256(path)
    if actual_hash != cc["formal_sha256"]:
        raise RuntimeError(f"{cohort}: formal CSV hash drift")
    canonicals = selected_canonicals(config, cohort)
    missing = [field for field in canonicals if field not in mapping]
    if missing:
        raise RuntimeError(f"{cohort}: missing selected canonical fields {missing}")
    decorated = [mapping[field] for field in canonicals]
    frame = pd.read_csv(
        path,
        usecols=decorated,
        dtype={mapping[cc["id"]]: "string"},
        encoding="utf-8-sig",
        low_memory=False,
    ).rename(columns={mapping[field]: field for field in canonicals})
    frame["person_id"] = normalize_id(frame[cc["id"]])
    frame["wave_int"] = pd.to_numeric(frame[cc["wave"]], errors="coerce").astype("Int64")
    frame = frame.loc[frame["wave_int"].isin(cc["primary_waves"])].copy()
    if frame["person_id"].isna().any():
        raise RuntimeError(f"{cohort}: missing formal person ID")
    duplicate_n = int(frame[["person_id", "wave_int"]].duplicated().sum())
    if duplicate_n:
        raise RuntimeError(f"{cohort}: {duplicate_n} duplicate ID-wave rows")
    if cc.get("interview_year") and cc.get("interview_month"):
        year = numeric(frame[cc["interview_year"]], [1900, 2100])
        month = numeric(frame[cc["interview_month"]], [1, 12])
        frame["interview_month_index"] = year * 12 + month
    else:
        frame["interview_month_index"] = np.nan
    keep = ["person_id", "wave_int", "interview_month_index", *canonicals]
    keep = list(dict.fromkeys(keep))
    return frame[keep], {
        "formal_relative_path": str(path.relative_to(root)),
        "formal_sha256": actual_hash,
        "ordered_header_matches_lookup": True,
        "selected_rows": int(len(frame)),
        "selected_fields": canonicals,
        "id_wave_unique": True,
    }


def source_status_and_elsa_dates(
    root: Path, config: dict[str, Any], cohort: str
) -> tuple[pd.DataFrame, pd.DataFrame | None, dict[str, Any]]:
    cc = config["cohorts"][cohort]
    source_path = root / cc["primary_source_relative_path"]
    actual_hash = sha256(source_path)
    if actual_hash != cc["primary_source_sha256"]:
        raise RuntimeError(f"{cohort}: primary source hash drift")
    waves = list(cc["primary_waves"])
    status_fields = [f"r{wave}iwstat" for wave in waves]
    date_fields = []
    if cohort == "elsa":
        date_fields = [field for wave in waves for field in (f"r{wave}iwy", f"r{wave}iwm")]
    with pd.io.stata.StataReader(source_path, convert_categoricals=False) as reader:
        reader.variable_labels()
        available = list(reader._varlist)
        label_names = list(reader._lbllist)
        value_labels = reader.value_labels()
        required = [cc["id"], *status_fields, *date_fields]
        missing = [field for field in required if field not in available]
        if missing:
            raise RuntimeError(f"{cohort}: source status/date fields missing {missing}")
        label_audit = []
        for field in status_fields:
            label_name = label_names[available.index(field)]
            mapping = {int(key): str(value) for key, value in value_labels.get(label_name, {}).items()}
            for code in cc["death_codes"]:
                if code not in mapping or "died" not in mapping[code].lower():
                    raise RuntimeError(f"{cohort}/{field}: death code {code} unsupported by source label")
            for code in cc["alive_codes"]:
                if code not in mapping or "alive" not in mapping[code].lower():
                    raise RuntimeError(f"{cohort}/{field}: alive code {code} unsupported by source label")
            label_audit.append({"variable": field, "value_label_name": label_name, "value_labels": mapping})
    wide = pd.read_stata(
        source_path,
        columns=[cc["id"], *status_fields, *date_fields],
        convert_categoricals=False,
        preserve_dtypes=False,
    )
    wide["person_id"] = normalize_id(wide[cc["id"]])
    if wide["person_id"].isna().any() or wide["person_id"].duplicated().any():
        raise RuntimeError(f"{cohort}: source person ID missing or nonunique")
    status_pieces = []
    date_pieces = []
    for wave in waves:
        status = wide[["person_id", f"r{wave}iwstat"]].rename(columns={f"r{wave}iwstat": "next_status"})
        status = status.copy()
        status["outcome_wave"] = wave
        status["next_status"] = pd.to_numeric(status["next_status"], errors="coerce")
        status_pieces.append(status)
        if cohort == "elsa":
            year = numeric(wide[f"r{wave}iwy"], [1900, 2100])
            month = numeric(wide[f"r{wave}iwm"], [1, 12])
            dates = pd.DataFrame(
                {"person_id": wide["person_id"], "wave_int": wave, "interview_month_index": year * 12 + month}
            )
            date_pieces.append(dates)
    status_long = pd.concat(status_pieces, ignore_index=True)
    dates_long = pd.concat(date_pieces, ignore_index=True) if date_pieces else None
    return status_long, dates_long, {
        "primary_source_relative_path": cc["primary_source_relative_path"],
        "primary_source_sha256": actual_hash,
        "source_id_unique": True,
        "source_rows": int(len(wide)),
        "status_value_labels": label_audit,
        "elsa_interview_dates_read_from_source": cohort == "elsa",
    }


def add_binary_behavior_state(
    frame: pd.DataFrame, spec: dict[str, Any], cohort: str, prefix: str
) -> pd.Series:
    fields = spec["fields"][cohort]
    values = frame[[f"{prefix}__{field}" for field in fields]].apply(pd.to_numeric, errors="coerce")
    cohort_rule = spec.get("rules_by_cohort", {}).get(cohort, {})
    rule = cohort_rule.get("rule", spec.get("rule"))
    result = pd.Series(np.nan, index=frame.index, dtype=float)
    if rule == "binary_single":
        raw = values.iloc[:, 0]
        if "active_values" in cohort_rule or "inactive_values" in cohort_rule:
            result.loc[raw.isin(cohort_rule.get("active_values", [1]))] = 1.0
            result.loc[raw.isin(cohort_rule.get("inactive_values", [0]))] = 0.0
        else:
            result.loc[raw.isin(spec.get("valid_values", [0, 1]))] = raw
    elif rule in {"binary_any", "ordinal_any"}:
        active_values = cohort_rule.get("active_values", [1])
        inactive_values = cohort_rule.get("inactive_values", [0])
        valid = values.isin(active_values + inactive_values)
        active = values.isin(active_values).any(axis=1)
        inactive = valid.all(axis=1) & values.isin(inactive_values).all(axis=1)
        result.loc[active] = 1.0
        result.loc[inactive] = 0.0
    elif rule is None and len(fields) == 1:
        raw = values.iloc[:, 0]
        result.loc[raw.isin(cohort_rule.get("active_values", [1]))] = 1.0
        result.loc[raw.isin(cohort_rule.get("inactive_values", [0]))] = 0.0
    else:
        raise RuntimeError(f"{cohort}: unsupported binary behavior rule {rule}")
    return result


def behavior_usable_waves(spec: dict[str, Any], cohort: str, primary_waves: list[int]) -> set[int]:
    rule = spec.get("rules_by_cohort", {}).get(cohort, {})
    usable = rule.get("usable_waves", spec.get("usable_waves", {}).get(cohort))
    return set(int(wave) for wave in (usable if usable is not None else primary_waves))


def behavior_states(
    episodes: pd.DataFrame, transition_id: str, spec: dict[str, Any], cohort: str, primary_waves: list[int]
) -> pd.Series:
    usable = behavior_usable_waves(spec, cohort, primary_waves)
    both_usable = episodes["t0"].isin(usable) & episodes["t1"].isin(usable)
    result = pd.Series(pd.NA, index=episodes.index, dtype="string")
    if spec["kind"] == "binary":
        state0 = add_binary_behavior_state(episodes, spec, cohort, "t0")
        state1 = add_binary_behavior_state(episodes, spec, cohort, "t1")
        valid = both_usable & state0.isin([0.0, 1.0]) & state1.isin([0.0, 1.0])
        result.loc[valid] = (
            state0.loc[valid].astype(int).astype(str) + "_to_" + state1.loc[valid].astype(int).astype(str)
        )
        return result

    field = spec["fields"][cohort][0]
    raw0 = pd.to_numeric(episodes[f"t0__{field}"], errors="coerce")
    raw1 = pd.to_numeric(episodes[f"t1__{field}"], errors="coerce")
    rule = spec.get("rules_by_cohort", {}).get(cohort, {})
    if transition_id == "smoking_quantity":
        low, high = spec["valid_range"]
        valid = both_usable & raw0.between(low, high) & raw1.between(low, high)
        change = raw1 - raw0
        result.loc[valid & change.le(-5)] = "decrease"
        result.loc[valid & change.between(-4, 4)] = "stable"
        result.loc[valid & change.ge(5)] = "increase"
        return result
    if rule.get("status") == "CODEBOOK_DIRECTION_REVIEW":
        return result
    if rule.get("scale") == "days_per_week":
        low, high = rule["valid_range"]
        valid = both_usable & raw0.between(low, high) & raw1.between(low, high)
        change = raw1 - raw0
        result.loc[valid & change.le(-2)] = "decrease"
        result.loc[valid & change.abs().lt(2)] = "stable"
        result.loc[valid & change.ge(2)] = "increase"
        return result
    if rule.get("scale") == "ordinal_frequency" and rule.get("direction") == "higher_code_more_frequent":
        low, high = rule.get("valid_range", [0, 4])
        valid = both_usable & raw0.between(low, high) & raw1.between(low, high)
        result.loc[valid & raw1.lt(raw0)] = "decrease"
        result.loc[valid & raw1.eq(raw0)] = "stable"
        result.loc[valid & raw1.gt(raw0)] = "increase"
        return result
    return result


def row_count(frame: pd.DataFrame, prefix: str, fields: list[str]) -> pd.Series:
    values = frame[[f"{prefix}__{field}" for field in fields]].apply(pd.to_numeric, errors="coerce")
    valid = values.isin([0.0, 1.0]).all(axis=1)
    result = pd.Series(np.nan, index=frame.index, dtype=float)
    result.loc[valid] = values.loc[valid].sum(axis=1)
    return result


def max_valid(frame: pd.DataFrame, prefix: str, fields: list[str], valid_range: list[float]) -> pd.Series:
    values = frame[[f"{prefix}__{field}" for field in fields]].apply(pd.to_numeric, errors="coerce")
    values = values.where(values.ge(valid_range[0]) & values.le(valid_range[1]))
    return values.max(axis=1, skipna=True)


def baseline_eligible(
    episodes: pd.DataFrame,
    outcome_id: str,
    spec: dict[str, Any],
    outcomes: dict[str, Any],
    cohort: str,
) -> pd.Series:
    """Who belongs in this outcome's risk set, decided from t1 alone.

    outcome_values returns NaN for two different reasons, and they look the
    same: the interval was never eligible, because the outcome was already
    present at the first interview, and the outcome was never observed,
    because the respondent died before the outcome interview.

    A composite that counts death as an event has to tell them apart. Writing
    the event over every NaN adds the deaths that were never eligible, while
    the survivors who were not eligible either stay out, which selects into
    the risk set on the very thing being modelled. This returns the mask that
    keeps that from happening, and it is computed without touching t2 so a
    death cannot change it.
    """
    fields = outcome_fields(outcome_id, spec, outcomes, cohort)
    if not fields and not (cohort == "klosa" and "klosa_module" in spec):
        return pd.Series(False, index=episodes.index)

    count1 = row_count(episodes, "t1", fields)
    if outcome_id in {"incident_any_adl", "incident_any_iadl"}:
        return count1.notna() & count1.eq(0)
    if outcome_id in {"adl_count_increase", "iadl_count_increase",
                      "multimorbidity_progression"}:
        return count1.notna()
    if spec["type"] == "incident_binary":
        return binary(episodes[f"t1__{fields[0]}"]).eq(0)
    raise ValueError(
        f"{outcome_id}: no baseline risk set is defined for type {spec['type']!r}, "
        "so it cannot carry a death-as-event composite"
    )


def outcome_values(
    episodes: pd.DataFrame,
    outcome_id: str,
    spec: dict[str, Any],
    outcomes: dict[str, Any],
    cohort: str,
) -> tuple[pd.Series, str]:
    result = pd.Series(np.nan, index=episodes.index, dtype=float)
    outcome_type = spec["type"]
    if outcome_type == "binary_source_status":
        known = episodes["next_status"].isin(
            episodes.attrs["death_codes"] + episodes.attrs["alive_codes"]
        )
        result.loc[known] = episodes.loc[known, "next_status"].isin(episodes.attrs["death_codes"]).astype(float)
        return result, "PASS_SOURCE_STATUS"

    fields = outcome_fields(outcome_id, spec, outcomes, cohort)
    if not fields and not (cohort == "klosa" and "klosa_module" in spec):
        return result, "NOT_AVAILABLE_IN_COHORT"

    if outcome_id in {"incident_any_adl", "adl_count_increase"}:
        count1 = row_count(episodes, "t1", fields)
        count2 = row_count(episodes, "t2", fields)
        valid = count1.notna() & count2.notna()
        if outcome_id == "incident_any_adl":
            valid &= count1.eq(0)
            result.loc[valid] = count2.loc[valid].ge(1).astype(float)
        else:
            result.loc[valid] = count2.loc[valid].sub(count1.loc[valid]).ge(1).astype(float)
        return result, "PASS_STRICT_COMPLETE_ITEM_COUNTS"

    if outcome_id in {"incident_any_iadl", "iadl_count_increase"}:
        count1 = row_count(episodes, "t1", fields)
        count2 = row_count(episodes, "t2", fields)
        valid = count1.notna() & count2.notna()
        if outcome_id == "incident_any_iadl":
            valid &= count1.eq(0)
            result.loc[valid] = count2.loc[valid].ge(1).astype(float)
        else:
            result.loc[valid] = count2.loc[valid].sub(count1.loc[valid]).ge(1).astype(float)
        return result, "PASS_STRICT_COMPLETE_ITEM_COUNTS"

    if outcome_id == "multimorbidity_progression":
        count1 = row_count(episodes, "t1", fields)
        count2 = row_count(episodes, "t2", fields)
        valid = count1.notna() & count2.notna()
        result.loc[valid] = count2.loc[valid].sub(count1.loc[valid]).ge(1).astype(float)
        return result, "PASS_STRICT_COMMON_SIX"

    if outcome_type == "incident_binary":
        field = fields[0]
        value1 = binary(episodes[f"t1__{field}"])
        value2 = binary(episodes[f"t2__{field}"])
        valid = value1.eq(0) & value2.notna()
        result.loc[valid] = value2.loc[valid]
        return result, "PASS_BASELINE_FREE_RISK_SET"

    if outcome_type == "binary_interval_event":
        field = fields[0]
        value2 = binary(episodes[f"t2__{field}"])
        result.loc[value2.notna()] = value2.dropna()
        return result, "PASS_INTERVAL_EVENT"

    if outcome_type == "ordinal_worsening":
        direction = spec.get("coding_direction_by_cohort", {}).get(cohort)
        if direction not in {"higher_worse", "lower_worse"}:
            return result, "NOT_EVALUABLE_CODING_DIRECTION_UNRESOLVED"
        field = fields[0]
        value1 = numeric(episodes[f"t1__{field}"]).where(lambda x: x.isin(spec["valid_values"]))
        value2 = numeric(episodes[f"t2__{field}"]).where(lambda x: x.isin(spec["valid_values"]))
        valid = value1.notna() & value2.notna()
        worse = value2.gt(value1) if direction == "higher_worse" else value2.lt(value1)
        result.loc[valid] = worse.loc[valid].astype(float)
        return result, f"PASS_{direction.upper()}"

    if outcome_id == "depressive_symptom_change" and cohort == "klosa":
        module = spec["klosa_module"]
        same_a = episodes["t1"].isin([3, 4]) & episodes["outcome_wave"].isin([3, 4])
        same_b = episodes["t1"].isin([5, 6, 7, 8, 9]) & episodes["outcome_wave"].isin([5, 6, 7, 8, 9])
        for mask, field in ((same_a, module["fields_by_waves"]["3-4"]), (same_b, module["fields_by_waves"]["5-10"])):
            value1 = numeric(episodes[f"t1__{field}"], [0, 30])
            value2 = numeric(episodes[f"t2__{field}"], [0, 30])
            valid = mask & value1.notna() & value2.notna()
            result.loc[valid] = value2.loc[valid] - value1.loc[valid]
        return result, "PASS_SCALE_VERSION_STRATIFIED"

    if outcome_type.startswith("continuous_change"):
        valid_range = spec.get("valid_range_by_cohort", {}).get(cohort, spec.get("valid_range"))
        if outcome_id == "grip_strength_change":
            value1 = max_valid(episodes, "t1", fields, valid_range)
            value2 = max_valid(episodes, "t2", fields, valid_range)
        else:
            field = fields[0]
            value1 = numeric(episodes[f"t1__{field}"], valid_range)
            value2 = numeric(episodes[f"t2__{field}"], valid_range)
        valid = value1.notna() & value2.notna()
        result.loc[valid] = value2.loc[valid] - value1.loc[valid]
        return result, "PASS_BASELINE_ADJUSTED_CHANGE_CANDIDATE"

    return result, f"NOT_IMPLEMENTED_{outcome_type}"


def build_episodes(
    formal: pd.DataFrame,
    status: pd.DataFrame,
    config: dict[str, Any],
    cohort: str,
    elsa_dates: pd.DataFrame | None,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    cc = config["cohorts"][cohort]
    waves = list(cc["primary_waves"])
    if elsa_dates is not None:
        formal = formal.drop(columns=["interview_month_index"]).merge(
            elsa_dates, on=["person_id", "wave_int"], how="left", validate="one_to_one"
        )
    wave_median_dates = formal.groupby("wave_int", observed=True)["interview_month_index"].median()
    previous_wave = {waves[index]: waves[index - 1] for index in range(1, len(waves))}
    next_wave = {waves[index]: waves[index + 1] for index in range(len(waves) - 1)}
    value_fields = [column for column in formal.columns if column not in {"person_id", "wave_int"}]

    t1 = formal.loc[formal["wave_int"].isin(waves[1:-1])].copy()
    t1["t0"] = t1["wave_int"].map(previous_wave)
    t1["outcome_wave"] = t1["wave_int"].map(next_wave)
    t1 = t1.rename(columns={"wave_int": "t1", **{field: f"t1__{field}" for field in value_fields}})
    t0 = formal.rename(columns={"wave_int": "t0", **{field: f"t0__{field}" for field in value_fields}})
    t2 = formal.rename(columns={"wave_int": "outcome_wave", **{field: f"t2__{field}" for field in value_fields}})
    episodes = t1.merge(t0, on=["person_id", "t0"], how="inner", validate="many_to_one")
    episodes = episodes.merge(t2, on=["person_id", "outcome_wave"], how="left", validate="many_to_one")
    age = numeric(episodes[f"t1__{cc['age']}"])
    episodes = episodes.loc[age.ge(config["minimum_age_at_t1"])].copy()
    before = len(episodes)
    episodes = episodes.merge(status, on=["person_id", "outcome_wave"], how="left", validate="many_to_one")
    if len(episodes) != before:
        raise RuntimeError(f"{cohort}: status merge changed episode count")
    episodes.attrs["death_codes"] = list(cc["death_codes"])
    episodes.attrs["alive_codes"] = list(cc["alive_codes"])

    t1_date = episodes["t1__interview_month_index"]
    t2_date = episodes["t2__interview_month_index"]
    episodes["observed_followup_months"] = t2_date - t1_date
    pair_rows = []
    for (t1_wave, outcome_wave), group in episodes.groupby(["t1", "outcome_wave"], sort=True):
        durations = group["observed_followup_months"].dropna()
        t1_wave_median = wave_median_dates.get(int(t1_wave), np.nan)
        outcome_wave_median = wave_median_dates.get(int(outcome_wave), np.nan)
        scheduled = (
            float(outcome_wave_median - t1_wave_median)
            if pd.notna(t1_wave_median) and pd.notna(outcome_wave_median)
            else np.nan
        )
        pair_rows.append(
            {
                "cohort": cohort,
                "t1": int(t1_wave),
                "outcome_wave": int(outcome_wave),
                "scheduled_followup_months": scheduled,
                "observed_duration_n": int(len(durations)),
                "observed_duration_p25": float(durations.quantile(.25)) if len(durations) else np.nan,
                "observed_duration_median": float(durations.median()) if len(durations) else np.nan,
                "observed_duration_p75": float(durations.quantile(.75)) if len(durations) else np.nan,
            }
        )
    schedule = {(row["t1"], row["outcome_wave"]): row["scheduled_followup_months"] for row in pair_rows}
    episodes["scheduled_followup_months"] = [schedule[(int(t1), int(t2))] for t1, t2 in zip(episodes["t1"], episodes["outcome_wave"])]
    low, high = config["comparable_outcome_window_months"]
    episodes["comparable_window"] = episodes["scheduled_followup_months"].between(low, high)
    return episodes, pair_rows


def support_label_binary(
    transitioned_events: int,
    transitioned_nonevents: int,
    comparator_n: int,
    total_events: int,
    minimum_events: int,
    minimum_nonevents: int,
    epv: int,
    parameters: int,
) -> str:
    if comparator_n == 0:
        return "NOT_EVALUABLE_NO_COMPARATOR"
    if transitioned_events < minimum_events:
        return "NOT_EVALUABLE_TRANSITION_EVENTS_LT_MINIMUM"
    if transitioned_nonevents < minimum_nonevents:
        return "NOT_EVALUABLE_TRANSITION_NONEVENTS_LT_MINIMUM"
    if total_events < epv * parameters:
        return "NOT_EVALUABLE_TOTAL_EVENTS_LT_EPV"
    return "ESTIMABLE"


def feasibility_rows_for_cell(
    episodes: pd.DataFrame,
    config: dict[str, Any],
    cohort: str,
    transition_id: str,
    transition_spec: dict[str, Any],
    outcome_id: str,
    outcome_spec: dict[str, Any],
) -> list[dict[str, Any]]:
    states = behavior_states(episodes, transition_id, transition_spec, cohort, config["cohorts"][cohort]["primary_waves"])
    outcome, outcome_status = outcome_values(episodes, outcome_id, outcome_spec, config["outcomes"], cohort)
    if transition_spec["kind"] == "binary":
        contrasts = {
            "withdrawal": ("1_to_0", "1_to_1"),
            "initiation_or_resumption": ("0_to_1", "0_to_0"),
        }
    else:
        contrasts = {"decrease_vs_stable": ("decrease", "stable"), "increase_vs_stable": ("increase", "stable")}
    rows = []
    scopes = {
        "all_primary_wave_intervals": pd.Series(True, index=episodes.index),
        "comparable_22_30_months": episodes["comparable_window"].fillna(False),
    }
    binary_outcome = outcome_spec["type"] not in {"continuous_change", "continuous_change_standardized_within_cohort"}
    for scope, scope_mask in scopes.items():
        for contrast, (transition_state, comparator_state) in contrasts.items():
            analytic = scope_mask & states.isin([transition_state, comparator_state]) & outcome.notna()
            transitioned = analytic & states.eq(transition_state)
            comparator = analytic & states.eq(comparator_state)
            base = {
                "cohort": cohort,
                "scope": scope,
                "transition_id": transition_id,
                "transition_family": transition_spec["family"],
                "transition_kind": transition_spec["kind"],
                "contrast": contrast,
                "transition_state": transition_state,
                "comparator_state": comparator_state,
                "outcome_id": outcome_id,
                "outcome_family": outcome_spec["family"],
                "outcome_type": outcome_spec["type"],
                "outcome_coding_status": outcome_status,
                "n_episodes": int(analytic.sum()),
                "n_people": int(episodes.loc[analytic, "person_id"].nunique()),
                "transitioned_n": int(transitioned.sum()),
                "transitioned_people": int(episodes.loc[transitioned, "person_id"].nunique()),
                "comparator_n": int(comparator.sum()),
                "comparator_people": int(episodes.loc[comparator, "person_id"].nunique()),
                "t0_waves": ";".join(map(str, sorted(episodes.loc[analytic, "t0"].unique()))) if analytic.any() else "",
                "t1_waves": ";".join(map(str, sorted(episodes.loc[analytic, "t1"].unique()))) if analytic.any() else "",
                "outcome_waves": ";".join(map(str, sorted(episodes.loc[analytic, "outcome_wave"].unique()))) if analytic.any() else "",
                "scheduled_months_min": float(episodes.loc[analytic, "scheduled_followup_months"].min()) if analytic.any() else np.nan,
                "scheduled_months_median": float(episodes.loc[analytic, "scheduled_followup_months"].median()) if analytic.any() else np.nan,
                "scheduled_months_max": float(episodes.loc[analytic, "scheduled_followup_months"].max()) if analytic.any() else np.nan,
            }
            if not outcome_status.startswith("PASS"):
                descriptive = outcome_status
                basic = outcome_status
                full = outcome_status
                base.update(
                    {
                        "outcome_events_or_observations": 0,
                        "transitioned_events_or_observations": 0,
                        "transitioned_nonevents": 0 if binary_outcome else "",
                        "comparator_events": 0 if binary_outcome else "",
                        "total_events": 0 if binary_outcome else "",
                        "descriptive_support": descriptive,
                        "basic_model_support": basic,
                        "full_model_support": full,
                    }
                )
            elif binary_outcome:
                transitioned_events = int(outcome.loc[transitioned].sum())
                transitioned_nonevents = int(transitioned.sum()) - transitioned_events
                comparator_events = int(outcome.loc[comparator].sum())
                total_events = transitioned_events + comparator_events
                descriptive = support_label_binary(
                    transitioned_events,
                    transitioned_nonevents,
                    int(comparator.sum()),
                    total_events,
                    config["minimum_binary_transition_events"],
                    config["minimum_binary_transition_nonevents"],
                    0,
                    0,
                )
                basic = support_label_binary(
                    transitioned_events,
                    transitioned_nonevents,
                    int(comparator.sum()),
                    total_events,
                    config["minimum_binary_transition_events"],
                    config["minimum_binary_transition_nonevents"],
                    config["events_per_parameter"],
                    BASIC_PARAMETER_COUNT,
                )
                full = support_label_binary(
                    transitioned_events,
                    transitioned_nonevents,
                    int(comparator.sum()),
                    total_events,
                    config["minimum_binary_transition_events"],
                    config["minimum_binary_transition_nonevents"],
                    config["events_per_parameter"],
                    FULL_PARAMETER_COUNT,
                )
                base.update(
                    {
                        "outcome_events_or_observations": total_events,
                        "transitioned_events_or_observations": transitioned_events,
                        "transitioned_nonevents": transitioned_nonevents,
                        "comparator_events": comparator_events,
                        "total_events": total_events,
                        "descriptive_support": descriptive,
                        "basic_model_support": basic,
                        "full_model_support": full,
                    }
                )
            else:
                n = int(analytic.sum())
                transitioned_n = int(transitioned.sum())
                support = "ESTIMABLE" if n >= config["minimum_continuous_episodes"] and transitioned_n >= config["minimum_continuous_transitioned"] else "NOT_EVALUABLE_CONTINUOUS_SUPPORT"
                base.update(
                    {
                        "outcome_events_or_observations": n,
                        "transitioned_events_or_observations": transitioned_n,
                        "transitioned_nonevents": "",
                        "comparator_events": "",
                        "total_events": "",
                        "descriptive_support": support,
                        "basic_model_support": support,
                        "full_model_support": support,
                    }
                )
            rows.append(base)
    return rows


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    root = Path(config["release_root"])
    lookup, lookup_path = load_lookup(root, config)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    os.chmod(args.output_dir, 0o700)

    rows: list[dict[str, Any]] = []
    interval_rows: list[dict[str, Any]] = []
    cohort_audits: dict[str, Any] = {}
    for cohort in config["cohorts"]:
        formal, formal_audit = read_formal(root, config, cohort, lookup)
        status, elsa_dates, source_audit = source_status_and_elsa_dates(root, config, cohort)
        episodes, pair_rows = build_episodes(formal, status, config, cohort, elsa_dates)
        interval_rows.extend(pair_rows)
        cohort_rows_before = len(rows)
        for transition_id, transition_spec in config["behavioral_transitions"].items():
            if cohort not in transition_spec.get("fields", {}):
                continue
            for outcome_id, outcome_spec in config["outcomes"].items():
                if outcome_spec["type"] != "binary_source_status" and not outcome_fields(
                    outcome_id, outcome_spec, config["outcomes"], cohort
                ) and not (cohort == "klosa" and "klosa_module" in outcome_spec):
                    continue
                rows.extend(
                    feasibility_rows_for_cell(
                        episodes,
                        config,
                        cohort,
                        transition_id,
                        transition_spec,
                        outcome_id,
                        outcome_spec,
                    )
                )
        cohort_audits[cohort] = {
            "formal": formal_audit,
            "source": source_audit,
            "eligible_age_t1_three_wave_episodes": int(len(episodes)),
            "people": int(episodes["person_id"].nunique()),
            "comparable_episodes": int(episodes["comparable_window"].sum()),
            "feasibility_rows": len(rows) - cohort_rows_before,
        }

    rows.sort(key=lambda row: (row["scope"], row["outcome_family"], row["outcome_id"], row["transition_id"], row["contrast"], row["cohort"]))
    interval_rows.sort(key=lambda row: (row["cohort"], row["t1"], row["outcome_wave"]))
    feasibility_path = args.output_dir / "feasibility-matrix.csv"
    interval_path = args.output_dir / "followup-interval-audit.csv"
    write_csv(feasibility_path, rows)
    write_csv(interval_path, interval_rows)
    status_counts = pd.Series([row["basic_model_support"] for row in rows]).value_counts().to_dict()
    manifest = {
        "analysis_id": config["analysis_id"],
        "phase": "aggregate_counts_only_feasibility",
        "config_path": str(args.config),
        "config_sha256": sha256(args.config),
        "universe_commit": args.universe_commit,
        "universe_commit_time": args.universe_commit_time,
        "first_exposure_specific_outcome_count_started_after_frozen_commit": True,
        "lookup_relative_path": str(lookup_path.relative_to(root)),
        "lookup_sha256": sha256(lookup_path),
        "cohorts": cohort_audits,
        "feasibility_rows": len(rows),
        "basic_model_support_counts": status_counts,
        "feasibility_matrix_sha256": sha256(feasibility_path),
        "followup_interval_audit_sha256": sha256(interval_path),
        "effect_models_fit": 0,
        "respondent_rows_exported": 0,
        "formal_iwstat_policy": "FORBIDDEN_NOT_READ",
        "negative_results_retained": True,
    }
    manifest_path = args.output_dir / "feasibility-run-manifest.json"
    manifest_path.write_text(json.dumps(json_ready(manifest), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.chmod(manifest_path, 0o600)


if __name__ == "__main__":
    main()
