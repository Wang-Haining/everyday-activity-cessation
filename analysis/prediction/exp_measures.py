#!/usr/bin/env python3
"""E2 pre-window measures: change from t-1 to t0 (all aligned to episodes.index)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from exp_common import (
    Cohort, adl_count, binary, iadl_count, max_valid, numeric, srh,
)


@dataclass
class Measure:
    name: str
    form: str            # "binary" or "continuous"
    y: pd.Series         # outcome of the E2 model (binary 0/1, or raw change for continuous)
    level: pd.Series     # t-1 level used as covariate
    poolable: bool = True
    note: str = ""
    primary: bool = True


def _both(a: pd.Series, b: pd.Series) -> pd.Series:
    return a.notna() & b.notna()


def build_measures(co: Cohort) -> list[Measure]:
    ep, name = co.episodes, co.name
    idx = ep.index
    nan = pd.Series(np.nan, index=idx)
    out: list[Measure] = []

    # 1. self-rated health (1 excellent ... 5 poor; higher = worse)
    s_m, s_0 = srh(co, "tm1"), srh(co, "t0")
    v = _both(s_m, s_0)
    out.append(Measure("srh_worse_1level", "binary", (s_0 > s_m).astype(float).where(v), s_m))
    new_fp = ((s_0 >= 4).astype(float)).where(v & (s_m <= 3))
    out.append(Measure("srh_new_fair_poor", "binary", new_fp, s_m, primary=False,
                       note="restricted to t-1 excellent/very good/good"))

    # 2. ADL and 3. IADL counts
    for label, fn in (("adl", adl_count), ("iadl", iadl_count)):
        c_m, c_0 = fn(co, "tm1"), fn(co, "t0")
        v = _both(c_m, c_0)
        out.append(Measure(f"{label}_increase", "binary", (c_0 > c_m).astype(float).where(v), c_m))
        out.append(Measure(f"{label}_mean_change", "continuous", (c_0 - c_m).where(v), c_m, primary=False))

    # 4. depressive symptoms
    d_m = d_0 = nan
    thr = 1.0
    scale_note = ""
    if name in {"elsa", "hrs"}:
        d_m = numeric(ep["tm1__cesd"], [0, 8]); d_0 = numeric(ep["t0__cesd"], [0, 8]); scale_note = "CESD-8"
    elif name == "klosa":
        a = ep["tm1_wave"].isin([3, 4]) & ep["t0"].isin([3, 4])
        b = ep["tm1_wave"].isin([5, 6, 7, 8, 9]) & ep["t0"].isin([5, 6, 7, 8, 9])
        d_m = pd.Series(np.nan, index=idx); d_0 = pd.Series(np.nan, index=idx)
        d_m.loc[a] = numeric(ep.loc[a, "tm1__cesd10a"], [0, 30]); d_0.loc[a] = numeric(ep.loc[a, "t0__cesd10a"], [0, 30])
        d_m.loc[b] = numeric(ep.loc[b, "tm1__cesd10b"], [0, 30]); d_0.loc[b] = numeric(ep.loc[b, "t0__cesd10b"], [0, 30])
        thr = 3.0; scale_note = "CESD-10 (a or b version; version-crossing pairs excluded)"
    elif name == "share":
        d_m = numeric(ep["tm1__eurod"], [0, 12]); d_0 = numeric(ep["t0__eurod"], [0, 12]); scale_note = "EURO-D (no CESD in SHARE)"
    v = _both(d_m, d_0)
    delta = (d_0 - d_m).where(v)
    out.append(Measure("depressive_change_continuous", "continuous", delta, d_m, note=scale_note))
    out.append(Measure("depressive_worsening_binary", "binary", (delta >= thr).astype(float).where(v), d_m,
                       primary=False, note=f"increase >= {thr:g} point(s); {scale_note}"))

    # 5. hospitalisation in the pre-window period (level at t0; t-1 level = same variable at t-1)
    hosp_field = {"hrs": "hosp", "klosa": "hcfcl2y", "share": "hosp1y"}.get(name)
    if hosp_field:
        h_0, h_m = binary(ep[f"t0__{hosp_field}"]), binary(ep[f"tm1__{hosp_field}"])
        out.append(Measure("hospitalised_prior_period", "binary", h_0.where(_both(h_0, h_m)), h_m, poolable=False,
                           note=f"{hosp_field}; recall window differs by cohort; not pooled"))
    else:
        out.append(Measure("hospitalised_prior_period", "binary", nan, nan, poolable=False, note="no hospital variable"))

    # 6. grip strength (kg, max of valid left/right, 1-100) and gait speed (wspeed, harmonised)
    g_m = max_valid(ep, "tm1", ["lgrip", "rgrip"], [1, 100]); g_0 = max_valid(ep, "t0", ["lgrip", "rgrip"], [1, 100])
    out.append(Measure("grip_change_kg", "continuous", (g_0 - g_m).where(_both(g_m, g_0)), g_m,
                       note="positive = stronger; measured at both t-1 and t0 only"))
    if name in {"elsa", "hrs", "share"}:
        w_m, w_0 = numeric(ep["tm1__wspeed"], [0.5, 120]), numeric(ep["t0__wspeed"], [0.5, 120])
        out.append(Measure("gait_change_wspeed", "continuous", (w_0 - w_m).where(_both(w_m, w_0)), w_m,
                           note="harmonised wspeed; positive change is coded as slower"))
    else:
        out.append(Measure("gait_change_wspeed", "continuous", nan, nan, note="no gait field"))
    return out
