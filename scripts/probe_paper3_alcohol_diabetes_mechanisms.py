#!/usr/bin/env python3
"""Read and harmonize source biomarker assays."""

from __future__ import annotations
import io
import zipfile
from pathlib import Path
from typing import Any
import pandas as pd
from probe_behavior_outcome_feasibility import normalize_id, sha256

def clean_assay(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").astype(float).where(lambda x: x.ge(0))


def read_zip_dta(path: Path, inner: str, columns: list[str]) -> pd.DataFrame:
    with zipfile.ZipFile(path) as archive:
        with archive.open(inner) as handle:
            payload = io.BytesIO(handle.read())
    return pd.read_stata(
        payload, columns=columns, convert_categoricals=False, preserve_dtypes=False
    )


def load_elsa_labs(root: Path, module: dict[str, Any]) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    pieces: list[pd.DataFrame] = []
    audits: list[dict[str, Any]] = []
    for key, relative in module["relative_files"].items():
        path = root / relative
        if sha256(path) != module["sha256"][key]:
            raise RuntimeError(f"ELSA nurse hash drift: {key}")
        if key == "8_9":
            columns = [
                module["id_field"], module["combined_wave_field"], "hba1c", "fglu",
                "trig", "hdl", "hscrp", "fastelig",
            ]
            frame = pd.read_stata(
                path, columns=columns, convert_categoricals=False, preserve_dtypes=False
            )
            frame["lab_wave"] = pd.to_numeric(
                frame[module["combined_wave_field"]], errors="coerce"
            )
        else:
            wave = int(key)
            fasting = module["fasting_fields"][key]
            columns = [module["id_field"], "hba1c", "fglu", "trig", "hdl", "hscrp", fasting]
            frame = pd.read_stata(
                path, columns=columns, convert_categoricals=False, preserve_dtypes=False
            )
            frame["lab_wave"] = wave
            if fasting != "fastelig":
                frame = frame.rename(columns={fasting: "fastelig"})
        frame["person_id"] = normalize_id(frame[module["id_field"]])
        frame = frame.loc[frame["lab_wave"].isin(module["waves"])].copy()
        for field in ["hba1c", "fglu", "trig", "hdl", "hscrp"]:
            frame[field] = clean_assay(frame[field])
        frame["fasting_eligible"] = pd.to_numeric(
            frame["fastelig"], errors="coerce"
        ).eq(1).astype(float).where(pd.to_numeric(frame["fastelig"], errors="coerce").ge(0))
        frame["hba1c_percent"] = frame["hba1c"]
        ifcc = frame["lab_wave"].isin([6, 8, 9])
        frame.loc[ifcc, "hba1c_percent"] = frame.loc[ifcc, "hba1c"] / 10.929 + 2.15
        frame["fasting_glucose_mmol_l"] = frame["fglu"].where(frame["fasting_eligible"].eq(1))
        frame["triglycerides_mmol_l"] = frame["trig"]
        frame["hdl_mmol_l"] = frame["hdl"]
        frame["crp_mg_l"] = frame["hscrp"]
        keep = [
            "person_id", "lab_wave", "hba1c_percent", "fasting_glucose_mmol_l",
            "triglycerides_mmol_l", "hdl_mmol_l", "crp_mg_l", "fasting_eligible",
        ]
        pieces.append(frame[keep])
        audits.append(
            {
                "module": "elsa_nurse",
                "source_key": key,
                "relative_path": relative,
                "sha256": module["sha256"][key],
                "rows": int(len(frame)),
                "id_missing": int(frame["person_id"].isna().sum()),
                "duplicate_id_wave": int(frame[["person_id", "lab_wave"]].duplicated().sum()),
            }
        )
    labs = pd.concat(pieces, ignore_index=True)
    if labs["person_id"].isna().any() or labs[["person_id", "lab_wave"]].duplicated().any():
        raise RuntimeError("ELSA nurse ID-wave integrity failure")
    return labs, audits

