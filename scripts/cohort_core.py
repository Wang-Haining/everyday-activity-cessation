#!/usr/bin/env python3
"""Shared, aggregate-safe cohort machinery.

Loading the harmonised specs and episodes, building the contraction frame and
its covariates, the clustered Poisson fit, the support rules that decide what
is estimable, and the writers. Every analysis in this repository runs on it.

Nothing here writes a respondent-level file: the writers take rows that are
already aggregated.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from probe_behavior_outcome_feasibility import (
    add_binary_behavior_state,
    behavior_states,
    binary,
    build_episodes,
    numeric,
    outcome_values,
    read_formal,
    sha256,
    source_status_and_elsa_dates,
)
from run_behavior_outcome_models import baseline_outcome, restricted_cubic_spline

COMMON_DISEASES = ["cancre", "diabe", "hearte", "hibpe", "stroke", "arthre"]


def load_specs(
    universe_path: Path, pilot_path: Path, assay_path: Path
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    universe = json.loads(universe_path.read_text())
    pilot = json.loads(pilot_path.read_text())
    assay = json.loads(assay_path.read_text())
    if sha256(universe_path) != pilot["parent_universe_sha256"]:
        raise RuntimeError("parent universe config drift")
    if sha256(assay_path) != pilot["assay_parent_sha256"]:
        raise RuntimeError("assay parent config drift")
    if universe["release_root"] != pilot["release_root"] or assay["release_root"] != pilot["release_root"]:
        raise RuntimeError("release-root mismatch")
    universe = json.loads(json.dumps(universe))
    universe["minimum_age_at_t1"] = pilot["minimum_age_at_t1"]
    universe["comparable_outcome_window_months"] = pilot["comparable_window_months"]
    universe["model_covariates_by_cohort"] = {
        cohort: ["smokev", "smoken"] for cohort in pilot["candidate_cohorts"]
    }
    return universe, pilot, assay


def load_episodes(
    root: Path, universe: dict[str, Any], cohort: str, lookup: pd.DataFrame
) -> tuple[pd.DataFrame, list[dict[str, Any]], dict[str, Any]]:
    formal, formal_audit = read_formal(root, universe, cohort, lookup)
    status, dates, source_audit = source_status_and_elsa_dates(root, universe, cohort)
    episodes, intervals = build_episodes(formal, status, universe, cohort, dates)
    audit = {"formal": formal_audit, "source_status": source_audit}
    return episodes, intervals, audit


def contraction_frame(
    episodes: pd.DataFrame, universe: dict[str, Any], cohort: str
) -> pd.DataFrame:
    cc = universe["cohorts"][cohort]
    alcohol_spec = universe["behavioral_transitions"]["alcohol_current"]
    activity_spec = universe["behavioral_transitions"]["moderate_or_vigorous_activity"]
    alcohol = behavior_states(episodes, "alcohol_current", alcohol_spec, cohort, cc["primary_waves"])
    activity = behavior_states(
        episodes, "moderate_or_vigorous_activity", activity_spec, cohort, cc["primary_waves"]
    )
    alcohol0 = add_binary_behavior_state(episodes, alcohol_spec, cohort, "t0")
    activity0 = add_binary_behavior_state(episodes, activity_spec, cohort, "t0")
    valid = alcohol.notna() & activity.notna()
    result = pd.DataFrame(index=episodes.index)
    result["alcohol_transition"] = alcohol
    result["activity_transition"] = activity
    result["alcohol_loss"] = alcohol.eq("1_to_0").astype(float).where(valid)
    result["activity_loss"] = activity.eq("1_to_0").astype(float).where(valid)
    result["alcohol_gain"] = alcohol.eq("0_to_1").astype(float).where(valid)
    result["activity_gain"] = activity.eq("0_to_1").astype(float).where(valid)
    result["baseline_engagement_count"] = (alcohol0 + activity0).where(valid)
    result["loss_count_2"] = (result["alcohol_loss"] + result["activity_loss"]).where(valid)
    result["any_contraction"] = result["loss_count_2"].ge(1).astype(float).where(valid)
    result["any_resumption"] = (result["alcohol_gain"].eq(1) | result["activity_gain"].eq(1)).astype(float).where(valid)
    result["valid_behavior_pair"] = valid
    return result


def baseline_multimorbidity(
    episodes: pd.DataFrame, exclude: str | None = None, prefix: str = "t1"
) -> pd.Series:
    fields = [field for field in COMMON_DISEASES if field != exclude]
    values = pd.DataFrame(
        {field: binary(episodes[f"{prefix}__{field}"]) for field in fields},
        index=episodes.index,
    )
    valid = values.notna().all(axis=1)
    result = pd.Series(np.nan, index=episodes.index, dtype=float)
    result.loc[valid] = values.loc[valid].sum(axis=1)
    return result


def covariate_frame(
    episodes: pd.DataFrame,
    behavior: pd.DataFrame,
    universe: dict[str, Any],
    cohort: str,
    outcome_id: str,
    anchor: str = "t1",
) -> pd.DataFrame:
    cc = universe["cohorts"][cohort]
    prefix = f"{anchor}__"
    data = pd.DataFrame(index=episodes.index)
    data["person_id"] = episodes["person_id"].astype("string")
    data["t1_wave"] = pd.to_numeric(episodes["t1"], errors="coerce")
    data["age"] = numeric(episodes[f"{prefix}{cc['age']}"])
    data["sex"] = numeric(episodes[f"{prefix}{cc['sex']}"]).where(lambda x: x.isin([1.0, 2.0]))
    education = numeric(episodes[f"{prefix}{cc['education']}"])
    income = numeric(episodes[f"{prefix}{cc['income']}"])
    data["education_rank"] = education.groupby(episodes["t1"]).rank(method="average", pct=True)
    data["income_rank"] = income.groupby(episodes["t1"]).rank(method="average", pct=True)
    data["smoke_ever"] = binary(episodes[f"{prefix}smokev"])
    data["smoke_current"] = binary(episodes[f"{prefix}smoken"])
    exclude = {
        "incident_diabetes": "diabe",
        "incident_heart_disease": "hearte",
        "incident_stroke": "stroke",
        "incident_hypertension": "hibpe",
        "incident_dementia": None,
    }.get(outcome_id)
    data["baseline_multimorbidity"] = baseline_multimorbidity(episodes, exclude=exclude, prefix=anchor)
    data["baseline_engagement_count"] = behavior["baseline_engagement_count"]
    return data


def add_base_design(
    data: pd.DataFrame, pilot: dict[str, Any], adjustment: str
) -> pd.DataFrame:
    X = pd.DataFrame({"intercept": 1.0}, index=data.index)
    X = pd.concat([X, restricted_cubic_spline(data["age"], pilot["age_rcs_knots"], "age")], axis=1)
    X["female_code2"] = data["sex"].eq(2.0).astype(float)
    X["baseline_engagement_count"] = data["baseline_engagement_count"].astype(float)
    wave = pd.get_dummies(
        data["t1_wave"].astype(int).astype(str), prefix="t1_wave", drop_first=True, dtype=float
    )
    X = pd.concat([X, wave], axis=1)
    if adjustment == "full":
        full_fields = [
            "education_rank", "income_rank", "smoke_ever", "smoke_current",
            "baseline_multimorbidity",
        ]
        if "baseline_outcome" in data:
            comparable = data[["baseline_outcome", "baseline_multimorbidity"]].dropna()
            if len(comparable) and np.allclose(
                comparable["baseline_outcome"].to_numpy(float),
                comparable["baseline_multimorbidity"].to_numpy(float),
            ):
                full_fields.remove("baseline_multimorbidity")
        for field in full_fields:
            X[field] = data[field].astype(float)
    if "baseline_outcome" in data:
        X["baseline_outcome"] = data["baseline_outcome"].astype(float)
    return X.astype(float)


def coefficient_rows(result: Any, terms: list[str], binary_outcome: bool) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    ci = result.conf_int()
    for term in terms:
        beta = float(result.params[term])
        se = float(result.bse[term])
        low, high = [float(value) for value in ci.loc[term]]
        if binary_outcome:
            rows.append({
                "term": term, "estimate_scale": "risk_ratio", "estimate": math.exp(beta),
                "standard_error": se, "ci_low": math.exp(low), "ci_high": math.exp(high),
                "p_value": float(result.pvalues[term]),
            })
        else:
            rows.append({
                "term": term, "estimate_scale": "mean_difference", "estimate": beta,
                "standard_error": se, "ci_low": low, "ci_high": high,
                "p_value": float(result.pvalues[term]),
            })
    return rows


def fit_clustered(
    y: pd.Series, X: pd.DataFrame, groups: pd.Series, binary_outcome: bool
) -> tuple[Any, str]:
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        if binary_outcome:
            result = sm.GLM(y, X, family=sm.families.Poisson()).fit(
                cov_type="cluster", cov_kwds={"groups": groups}
            )
        else:
            result = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": groups})
    warning_text = " | ".join(str(item.message) for item in captured)
    return result, warning_text


def binary_support(
    data: pd.DataFrame,
    y: pd.Series,
    X: pd.DataFrame,
    terms: list[str],
    pilot: dict[str, Any],
) -> tuple[str, dict[str, int]]:
    counts: dict[str, int] = {
        "n": int(len(data)), "people": int(data["person_id"].nunique()),
        "events": int(y.sum()), "parameters": int(X.shape[1]),
        "matrix_rank": int(np.linalg.matrix_rank(X.to_numpy(float))) if len(X) else 0,
    }
    if counts["people"] < pilot["minimum_clusters"]:
        return "NOT_EVALUABLE_CLUSTERS", counts
    if counts["matrix_rank"] < counts["parameters"]:
        return "NOT_EVALUABLE_RANK", counts
    if counts["events"] < pilot["events_per_parameter"] * counts["parameters"]:
        return "NOT_EVALUABLE_EPV", counts
    for term in terms:
        exposed = data[term].eq(1)
        counts[f"{term}_n"] = int(exposed.sum())
        counts[f"{term}_events"] = int(y.loc[exposed].sum())
        counts[f"{term}_nonevents"] = int(exposed.sum() - y.loc[exposed].sum())
        if counts[f"{term}_events"] < pilot["minimum_exposed_events"]:
            return f"NOT_EVALUABLE_{term.upper()}_EVENTS", counts
        if counts[f"{term}_nonevents"] < pilot["minimum_exposed_nonevents"]:
            return f"NOT_EVALUABLE_{term.upper()}_NONEVENTS", counts
    return "ESTIMABLE", counts


def continuous_support(
    data: pd.DataFrame, group_field: str, pilot: dict[str, Any]
) -> tuple[str, dict[str, int]]:
    group = data[group_field].eq(1)
    counts = {
        "n": int(len(data)), "people": int(data["person_id"].nunique()),
        "exposed_n": int(group.sum()), "unexposed_n": int((~group).sum()),
    }
    if len(data) < pilot["minimum_continuous_n"]:
        return "NOT_EVALUABLE_CONTINUOUS_N", counts
    if min(counts["exposed_n"], counts["unexposed_n"]) < pilot["minimum_continuous_group_n"]:
        return "NOT_EVALUABLE_CONTINUOUS_GROUP_N", counts
    return "ESTIMABLE", counts


def add_burden(
    assays: pd.DataFrame, sex: pd.Series, pilot: dict[str, Any]
) -> pd.DataFrame:
    result = assays.copy()
    transformed = pd.DataFrame(index=result.index)
    transformed["log_crp"] = np.log(result["crp"] + pilot["crp_log_offset"])
    transformed["hba1c"] = result["hba1c"]
    transformed["log_triglycerides"] = np.log(result["triglycerides"])
    transformed["negative_hdl"] = -result["hdl"]
    group = pd.DataFrame({"wave": result["assay_wave"], "sex": sex}, index=result.index)
    z = pd.DataFrame(index=result.index)
    for marker in pilot["primary_burden_components"]:
        means = transformed[marker].groupby([group["wave"], group["sex"]]).transform("mean")
        sds = transformed[marker].groupby([group["wave"], group["sex"]]).transform("std")
        z[f"z_{marker}"] = (transformed[marker] - means) / sds.where(sds > 0)
    result = pd.concat([result, z], axis=1)
    n_components = z.notna().sum(axis=1)
    raw_burden = z.mean(axis=1, skipna=True).where(n_components.ge(pilot["minimum_burden_components"]))
    bmean = raw_burden.groupby([group["wave"], group["sex"]]).transform("mean")
    bsd = raw_burden.groupby([group["wave"], group["sex"]]).transform("std")
    result["burden_z"] = (raw_burden - bmean) / bsd.where(bsd > 0)
    result["burden_components_n"] = n_components
    cutoff = result["burden_z"].groupby(result["assay_wave"]).transform(
        lambda x: x.quantile(pilot["high_burden_quantile"])
    )
    result["high_burden"] = result["burden_z"].ge(cutoff).astype(float).where(result["burden_z"].notna())
    result["z_crp"] = z["z_log_crp"]
    result["z_hba1c"] = z["z_hba1c"]
    result["z_triglycerides"] = z["z_log_triglycerides"]
    result["z_hdl_adverse"] = z["z_negative_hdl"]
    return result


def outcome_and_baseline(
    episodes: pd.DataFrame,
    universe: dict[str, Any],
    cohort: str,
    outcome_id: str,
) -> tuple[pd.Series, pd.Series | None, str, bool]:
    spec = universe["outcomes"][outcome_id]
    outcome, status = outcome_values(episodes, outcome_id, spec, universe["outcomes"], cohort)
    baseline = baseline_outcome(episodes, outcome_id, spec, universe["outcomes"], cohort)
    binary_outcome = not spec["type"].startswith("continuous_change")
    return outcome, baseline, status, binary_outcome


def write_frame(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise RuntimeError(f"refusing empty output {path}")
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    pd.DataFrame(rows).reindex(columns=fields).to_csv(path, index=False, lineterminator="\n")
    os.chmod(path, 0o600)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n")
    os.chmod(path, 0o600)


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
