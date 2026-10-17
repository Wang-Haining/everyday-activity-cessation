#!/usr/bin/env python3
"""Load the CHARLS blood assay module."""

from __future__ import annotations
from pathlib import Path
from typing import Any
import pandas as pd
from probe_behavior_outcome_feasibility import normalize_id, sha256
from probe_paper3_alcohol_diabetes_mechanisms import clean_assay, read_zip_dta

def load_charls_blood(
    root: Path, parent: dict[str, Any], config: dict[str, Any]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    module = config["biomarker_modules"]["charls_pretransition"]
    path = root / module["relative_path"]
    if sha256(path) != module["sha256"]:
        raise RuntimeError("CHARLS blood hash drift")
    columns = [module["id_field"], module["weight_field"], module["fasting_field"], *module["fields"].values()]
    blood = read_zip_dta(path, module["inner_dta"], columns)
    blood["wave1_person_id"] = normalize_id(blood[module["id_field"]])
    linkage = module["linkage"]
    source_path = root / parent["cohorts"]["charls"]["primary_source_relative_path"]
    with pd.io.stata.StataReader(source_path, convert_categoricals=False) as reader:
        labels = reader.variable_labels()
    if linkage["source_wave1_label_contains"].lower() not in labels.get(
        linkage["source_wave1_id_field"], ""
    ).lower():
        raise RuntimeError("CHARLS linkage source label mismatch")
    crosswalk = pd.read_stata(
        source_path,
        columns=[linkage["source_current_id_field"], linkage["source_wave1_id_field"]],
        convert_categoricals=False,
        preserve_dtypes=False,
    )
    crosswalk["person_id"] = normalize_id(crosswalk[linkage["source_current_id_field"]])
    crosswalk["wave1_person_id"] = normalize_id(crosswalk[linkage["source_wave1_id_field"]])
    crosswalk = crosswalk.loc[crosswalk["wave1_person_id"].notna(), ["wave1_person_id", "person_id"]]
    if crosswalk["wave1_person_id"].duplicated().any() or crosswalk["person_id"].duplicated().any():
        raise RuntimeError("CHARLS linkage crosswalk nonunique")
    blood = blood.merge(crosswalk, on="wave1_person_id", how="left", validate="one_to_one")
    if float(blood["person_id"].notna().mean()) != linkage["required_blood_id_match_fraction"]:
        raise RuntimeError("CHARLS blood linkage incomplete")
    blood = blood.rename(columns={source: target for target, source in module["fields"].items()})
    for marker in module["fields"]:
        blood[marker] = clean_assay(blood[marker])
    return blood, {
        "source": str(path.relative_to(root)), "sha256": module["sha256"],
        "rows": int(len(blood)), "matched_rows": int(blood["person_id"].notna().sum()),
        "linkage": "blood.ID_to_ID_w1_to_current_ID",
    }

