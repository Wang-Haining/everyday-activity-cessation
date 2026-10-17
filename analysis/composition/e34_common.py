#!/usr/bin/env python3
"""Shared building blocks for E3 (composition) and E4 (losses vs gains).

Uses the shared harmonized-data loader and writes aggregate composition results.
Respondent-level frames live only in memory. Aggregates only are written.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "analysis" / "prediction"))
from exp_common import (  # noqa: E402
    PUB, PUB_RR, MORT_COHORTS, ADL_COHORTS, assert_published, load_cohort, primary_set,
)
from exp_stats import reml_hk  # noqa: E402
from multidomain_behavioral_withdrawal_core import (  # noqa: E402
    coefficient_rows, fit_clustered, support_gate,
)

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "composition" / "cohort"
OUTCOMES = ["mortality", "incident_any_adl"]
DOMS = ["alcohol", "activity", "work"]            # bit order of the pattern string: D, A, W
LETTER = {"alcohol": "D", "activity": "A", "work": "W"}
PATTERNS = ["000", "100", "010", "001", "110", "101", "011", "111"]


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    assert not ({"person_id", "id", "respondent_id", "prediction"} & set(frame.columns))
    frame.to_csv(path, index=False, lineterminator="\n")
    print(f"{path.name}: {len(frame)} aggregate rows")


def build_frame(co, outcome: str):
    """Primary risk set (verified against the published n/events/people) with domain bits."""
    data, y, X, terms = primary_set(co, outcome)
    got = assert_published(co, outcome, data, y)
    b = co.behavior.loc[data.index]
    d = pd.DataFrame(index=data.index)
    d["person_id"] = data["person_id"]
    d["t1_wave"] = data["t1_wave"]
    d["y"] = y.astype(float)
    for s in DOMS:
        d[f"{s}_base"] = b[f"{s}_baseline"].astype(int)
        d[f"{s}_loss"] = b[f"{s}_loss"].astype(int)
        d[f"{s}_gain"] = b[f"{s}_gain"].astype(int)
    d["opp"] = d[[f"{s}_base" for s in DOMS]].sum(axis=1)
    d["loss_n"] = d[[f"{s}_loss" for s in DOMS]].sum(axis=1)
    d["gain_n"] = d[[f"{s}_gain" for s in DOMS]].sum(axis=1)
    d["pattern"] = d[[f"{s}_loss" for s in DOMS]].astype(str).agg("".join, axis=1)
    # integrity checks (identities of the three-domain state)
    assert (d["opp"] == data["baseline_engagement_count"]).all()
    for s in DOMS:
        assert (d[f"{s}_loss"] <= d[f"{s}_base"]).all() and (d[f"{s}_gain"] <= 1 - d[f"{s}_base"]).all()
    assert ((d["loss_n"] == 1) == data["loss_1"].eq(1)).all() and ((d["loss_n"] >= 2) == data["loss_2plus"].eq(1)).all()
    assert d["pattern"].isin(PATTERNS).all()
    assert (d["opp"] >= 1).all()
    X0 = X.drop(columns=terms)
    return d, X0, got


def fit_groups(d, X0, groups, focal, pilot, count_override=None, min_ev=20, min_nonev=20):
    """Modified-Poisson fit with several mutually exclusive indicator groups vs a reference.

    d       frame (already restricted to the analysis population), columns person_id, t1_wave, y
    X0      covariate design for the same index (primary covariates, without exposure terms)
    groups  dict name -> boolean Series (mutually exclusive); the reference is every row in no group
    focal   names reported; the other groups are nuisance (kept in the model, not reported)
    Per-term support requirement: a group with <min_ev events or <min_nonev non-events is
    NOT_EVALUABLE and its rows are dropped from that fit; the reference must also pass.
    Model-level gate (clusters, rank, events >= 10 x parameters) is the pipeline's support_gate.
    Returns (rows, fit_or_None, kept_terms).
    """
    y = d["y"]
    ref = ~pd.concat(list(groups.values()), axis=1).any(axis=1)
    cnt = {}
    for name, m in groups.items():
        ev = int(y[m].sum()); n = int(m.sum())
        cnt[name] = (n, ev, n - ev)
    rev = int(y[ref].sum()); rn = int(ref.sum())
    status = {}
    for name, (n, ev, ne) in cnt.items():
        status[name] = "ESTIMABLE" if (ev >= min_ev and ne >= min_nonev) else ("NOT_EVALUABLE_EVENTS" if ev < min_ev else "NOT_EVALUABLE_NONEVENTS")
    ref_ok = rev >= min_ev and (rn - rev) >= min_nonev
    base = {"n_population": int(len(d)), "people_population": int(d["person_id"].nunique()),
            "events_population": int(y.sum()), "n_reference": rn, "events_reference": rev}

    def rows_nofit(note):
        out = []
        for name in focal:
            n, ev, ne = cnt[name]
            out.append({**base, "term": name, "n_group": n, "events_group": ev,
                        "risk_group": ev / n if n else np.nan, "status": note if status[name] == "ESTIMABLE" else status[name]})
        return out

    if not ref_ok:
        return rows_nofit("NOT_EVALUABLE_REFERENCE"), None, []
    kept = [n for n in groups if status[n] == "ESTIMABLE"]
    if not any(n in focal for n in kept):
        return rows_nofit("NOT_EVALUABLE_NO_FOCAL"), None, []
    dropped_rows = pd.concat([groups[n] for n in groups if n not in kept], axis=1).any(axis=1) if len(kept) < len(groups) else pd.Series(False, index=d.index)
    keep_idx = d.index[~dropped_rows.to_numpy()]
    ds = d.loc[keep_idx]
    ys = y.loc[keep_idx]
    Xs = X0.loc[keep_idx]
    # rebuild wave indicators inside the subset (lowest wave present dropped), drop constant columns
    Xs = Xs.drop(columns=[c for c in Xs.columns if c.startswith("t1_wave_")])
    wave = pd.get_dummies(ds["t1_wave"].astype(int).astype(str), prefix="t1_wave", drop_first=True, dtype=float)
    Xs = pd.concat([Xs, wave], axis=1)
    if count_override is not None:
        Xs = Xs.copy(); Xs["baseline_engagement_count"] = count_override.loc[keep_idx].astype(float)
    Xs = Xs.loc[:, [c == "intercept" or Xs[c].std() > 0 for c in Xs.columns]]
    dd = pd.DataFrame({"person_id": ds["person_id"]})
    for n in kept:
        dd[n] = groups[n].loc[keep_idx].astype(float)
        Xs[n] = dd[n]
    gate, counts = support_gate(dd, ys, Xs, kept, pilot, True)
    base2 = {**base, "n_model": int(len(ds)), "events_model": int(ys.sum()), "n_parameters": int(Xs.shape[1]),
             "n_dropped_rows_failed_groups": int(len(d) - len(ds))}
    if gate != "ESTIMABLE":
        out = []
        for name in focal:
            n, ev, ne = cnt[name]
            out.append({**base2, "term": name, "n_group": n, "events_group": ev, "risk_group": ev / n if n else np.nan,
                        "status": gate if status[name] == "ESTIMABLE" else status[name]})
        return out, None, kept
    fit, warn = fit_clustered(ys, Xs, ds["person_id"], True)
    crow = {r["term"]: r for r in coefficient_rows(fit, [n for n in kept if n in focal], True)}
    out = []
    for name in focal:
        n, ev, ne = cnt[name]
        r = {**base2, "term": name, "n_group": n, "events_group": ev, "risk_group": ev / n if n else np.nan}
        if name in crow:
            c = crow[name]
            r.update({"status": "ESTIMABLE", "rr": c["estimate"], "se_log_rr": c["standard_error"],
                      "ci_low": c["ci_low"], "ci_high": c["ci_high"], "p_value": c["p_value"], "warnings": warn})
        else:
            r["status"] = status[name]
        out.append(r)
    return out, fit, kept


def contrast_rows(fit, kept, pairs, base_info):
    """log-ratio of two group RRs from the cluster-robust covariance of one fit."""
    rows = []
    if fit is None:
        return rows
    V = fit.cov_params()
    for label, a, b in pairs:
        if a in kept and b in kept:
            diff = float(fit.params[a] - fit.params[b])
            var = float(V.loc[a, a] + V.loc[b, b] - 2 * V.loc[a, b])
            se = var ** 0.5
            rows.append({**base_info, "term": label, "status": "ESTIMABLE", "rr": float(np.exp(diff)), "se_log_rr": se,
                         "ci_low": float(np.exp(diff - 1.959964 * se)), "ci_high": float(np.exp(diff + 1.959964 * se)),
                         "is_contrast": True})
    return rows
