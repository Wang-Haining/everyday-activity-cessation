#!/usr/bin/env python3
"""Shared building blocks for analyses A (composition and gains without HRS), B (delayed landmarks)
and C (selection weights), including support checks and aggregate effect estimates.
Respondent-level frames live only in memory. Aggregates only are written.
"""
from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "analysis" / "prediction"))
from exp_common import (  # noqa: E402
    DATA, PUB, PUB_RR, MORT_COHORTS, ADL_COHORTS, assert_published, load_cohort, primary_set,
    srh, adl_count, iadl_count,
)
from exp_stats import reml_hk, fit_logit, Prep  # noqa: E402
from multidomain_behavioral_withdrawal_core import (  # noqa: E402
    coefficient_rows, fit_clustered, support_gate,
)
from behavioral_contraction_biomarker_core import covariate_frame, outcome_and_baseline  # noqa: E402
from probe_behavior_outcome_feasibility import (  # noqa: E402
    load_lookup, normalize_id, numeric, binary, baseline_eligible,
)
from run_behavior_outcome_models import restricted_cubic_spline  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "sensitivity" / "cohort"
OUTCOMES = ["mortality", "incident_any_adl"]
DOMS = ["alcohol", "activity", "work"]            # bit order of the pattern string: D, A, W
LETTER = {"alcohol": "D", "activity": "A", "work": "W"}
PATTERNS = ["000", "100", "010", "001", "110", "101", "011", "111"]
PAIRS = [("alcohol", "activity"), ("alcohol", "work"), ("activity", "work")]
Z975 = 1.959963984540054


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    assert not ({"person_id", "id", "respondent_id", "prediction"} & set(frame.columns))
    frame.to_csv(path, index=False, lineterminator="\n")
    print(f"{path.name}: {len(frame)} aggregate rows")


# ----------------------------------------------------------------------------------------------
# primary risk set with domain bits (copied from v0.2 e34_common.build_frame)
# ----------------------------------------------------------------------------------------------
def build_frame(co, outcome: str):
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
    assert (d["opp"] == data["baseline_engagement_count"]).all()
    for s in DOMS:
        assert (d[f"{s}_loss"] <= d[f"{s}_base"]).all() and (d[f"{s}_gain"] <= 1 - d[f"{s}_base"]).all()
    assert ((d["loss_n"] == 1) == data["loss_1"].eq(1)).all() and ((d["loss_n"] >= 2) == data["loss_2plus"].eq(1)).all()
    assert d["pattern"].isin(PATTERNS).all()
    assert (d["opp"] >= 1).all()
    X0 = X.drop(columns=terms)
    return d, X0, got


def _nuniq(d, m):
    return int(d["person_id"][m].nunique())


def fit_groups(d, X0, groups, focal, pilot, count_override=None, min_ev=20, min_nonev=20):
    """Modified-Poisson fit with mutually exclusive indicator groups against a reference.
    Same logic as v0.2 fit_groups; rows additionally carry respondent counts."""
    y = d["y"]
    ref = ~pd.concat(list(groups.values()), axis=1).any(axis=1)
    cnt = {}
    for name, m in groups.items():
        ev = int(y[m].sum()); n = int(m.sum())
        cnt[name] = (n, ev, n - ev, _nuniq(d, m))
    rev = int(y[ref].sum()); rn = int(ref.sum())
    status = {}
    for name, (n, ev, ne, _p) in cnt.items():
        status[name] = "ESTIMABLE" if (ev >= min_ev and ne >= min_nonev) else ("NOT_EVALUABLE_EVENTS" if ev < min_ev else "NOT_EVALUABLE_NONEVENTS")
    ref_ok = rev >= min_ev and (rn - rev) >= min_nonev
    base = {"n_population": int(len(d)), "people_population": int(d["person_id"].nunique()),
            "events_population": int(y.sum()), "n_reference": rn, "events_reference": rev,
            "people_reference": _nuniq(d, ref)}

    def grp_cols(name):
        n, ev, ne, pp = cnt[name]
        return {"n_group": n, "events_group": ev, "people_group": pp, "risk_group": ev / n if n else np.nan}

    def rows_nofit(note):
        return [{**base, "term": name, **grp_cols(name), "status": note if status[name] == "ESTIMABLE" else status[name]} for name in focal]

    if not ref_ok:
        return rows_nofit("NOT_EVALUABLE_REFERENCE"), None, []
    kept = [n for n in groups if status[n] == "ESTIMABLE"]
    if not any(n in focal for n in kept):
        return rows_nofit("NOT_EVALUABLE_NO_FOCAL"), None, []
    dropped_rows = pd.concat([groups[n] for n in groups if n not in kept], axis=1).any(axis=1) if len(kept) < len(groups) else pd.Series(False, index=d.index)
    keep_idx = d.index[~dropped_rows.to_numpy()]
    ds = d.loc[keep_idx]; ys = y.loc[keep_idx]; Xs = X0.loc[keep_idx]
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
             "events_per_parameter": float(ys.sum()) / Xs.shape[1], "n_dropped_rows_failed_groups": int(len(d) - len(ds))}
    if gate != "ESTIMABLE":
        return [{**base2, "term": name, **grp_cols(name), "status": gate if status[name] == "ESTIMABLE" else status[name]} for name in focal], None, kept
    fit, warn = fit_clustered(ys, Xs, ds["person_id"], True)
    crow = {r["term"]: r for r in coefficient_rows(fit, [n for n in kept if n in focal], True)}
    out = []
    for name in focal:
        r = {**base2, "term": name, **grp_cols(name)}
        if name in crow:
            c = crow[name]
            r.update({"status": "ESTIMABLE", "rr": c["estimate"], "se_log_rr": c["standard_error"],
                      "ci_low": c["ci_low"], "ci_high": c["ci_high"], "p_value": c["p_value"], "warnings": warn})
        else:
            r["status"] = status[name]
        out.append(r)
    return out, fit, kept


def contrast_rows(fit, kept, pairs, base_info):
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
                         "ci_low": float(np.exp(diff - Z975 * se)), "ci_high": float(np.exp(diff + Z975 * se)),
                         "is_contrast": True})
    return rows


# ----------------------------------------------------------------------------------------------
# one-stage cohort-stratified common-coefficient fit (copied from v0.2 onestage_e34.stacked_fit)
# ----------------------------------------------------------------------------------------------
def stacked_fit(plist, fn, focal, label_info, pilot, contrasts=(), count_fn=None, extra_fn=None, drop_cols=()):
    """plist: {cohort: (d, X0)}. Every covariate has cohort-specific coefficients; the exposure group
    terms are common to all cohorts. The model-level events-per-parameter requirement is reported
    (events_per_parameter, epp_ge10) rather than enforced; term gates, cluster count and rank are enforced."""
    pilot_use = {**pilot, "events_per_parameter": 0}
    sub = {}
    for c, (d, X0) in plist.items():
        mask, groups = fn(d)
        X0c = X0.copy()
        if count_fn is not None:
            X0c["baseline_engagement_count"] = count_fn(d).astype(float)
        if extra_fn is not None:
            X0c = pd.concat([X0c, extra_fn(d)], axis=1)
        if len(drop_cols):
            X0c = X0c.drop(columns=list(drop_cols), errors="ignore")
        sub[c] = (d.loc[mask], X0c.loc[mask], {k: v.loc[mask] for k, v in groups.items()})
    names = list(next(iter(sub.values()))[2])
    ev = {k: sum(int(s[0]["y"][s[2][k]].sum()) for s in sub.values()) for k in names}
    nn = {k: sum(int(s[2][k].sum()) for s in sub.values()) for k in names}
    pp = {k: sum(int(s[0]["person_id"][s[2][k]].nunique()) for s in sub.values()) for k in names}
    percoh = {k: ";".join(f"{c}:{int(s[2][k].sum())}/{int(s[0]['y'][s[2][k]].sum())}" for c, s in sub.items()) for k in names}
    ref_n = ref_ev = ref_pp = 0
    for dsub, _, g in sub.values():
        r = ~pd.concat(list(g.values()), axis=1).any(axis=1)
        ref_n += int(r.sum()); ref_ev += int(dsub["y"][r].sum()); ref_pp += int(dsub["person_id"][r].nunique())
    n_pop = sum(len(s[0]) for s in sub.values()); ev_pop = sum(int(s[0]["y"].sum()) for s in sub.values())
    base = {**label_info, "n_population": n_pop, "events_population": ev_pop, "n_reference": ref_n,
            "events_reference": ref_ev, "people_reference": ref_pp, "cohorts_included": ";".join(sub)}
    status = {k: ("ESTIMABLE" if ev[k] >= 20 and (nn[k] - ev[k]) >= 20 else "NOT_EVALUABLE_EVENTS" if ev[k] < 20 else "NOT_EVALUABLE_NONEVENTS") for k in names}

    def gcols(k):
        return {"n_group": nn[k], "events_group": ev[k], "people_group": pp[k],
                "risk_group": ev[k] / nn[k] if nn[k] else np.nan, "cohort_n_events_group": percoh[k]}

    def nofit(note):
        return [{**base, "term": k, **gcols(k), "status": note if status[k] == "ESTIMABLE" else status[k]} for k in focal]

    if not (ref_ev >= 20 and ref_n - ref_ev >= 20):
        return nofit("NOT_EVALUABLE_REFERENCE")
    kept = [k for k in names if status[k] == "ESTIMABLE"]
    if not any(k in focal for k in kept):
        return nofit("NOT_EVALUABLE_NO_FOCAL")
    Xl, dl, gl = [], [], {k: [] for k in kept}
    dropped = 0
    for c, (dsub, Xsub, g) in sub.items():
        bad = pd.concat([g[k] for k in names if k not in kept], axis=1).any(axis=1) if len(kept) < len(names) else pd.Series(False, index=dsub.index)
        keep = dsub.index[~bad.to_numpy()]
        dropped += int(len(dsub) - len(keep))
        ds, Xs = dsub.loc[keep], Xsub.loc[keep]
        Xs = Xs.drop(columns=[x for x in Xs.columns if x.startswith("t1_wave_")])
        wave = pd.get_dummies(ds["t1_wave"].astype(int).astype(str), prefix="t1_wave", drop_first=True, dtype=float)
        Xs = pd.concat([Xs, wave], axis=1)
        Xs = Xs.loc[:, [x == "intercept" or Xs[x].std() > 0 for x in Xs.columns]]
        Xs.columns = [f"{c}__{x}" for x in Xs.columns]
        Xl.append(Xs); dl.append(ds[["person_id", "y"]])
        for k in kept:
            gl[k].append(g[k].loc[keep].astype(float))
    Xall = pd.concat(Xl, axis=0, ignore_index=True).fillna(0.0)
    dall = pd.concat(dl, axis=0, ignore_index=True)
    for k in kept:
        Xall[k] = pd.concat(gl[k], axis=0, ignore_index=True)
    gate, counts = support_gate(pd.DataFrame({"person_id": dall["person_id"], **{k: Xall[k] for k in kept}}), dall["y"], Xall, kept, pilot_use, True)
    npar = int(Xall.shape[1]); evm = int(dall["y"].sum())
    base2 = {**base, "n_model": int(len(dall)), "events_model": evm, "n_parameters": npar,
             "events_per_parameter": evm / npar, "epp_ge10": bool(evm >= 10 * npar), "n_dropped_rows_failed_groups": dropped}
    if gate != "ESTIMABLE":
        return [{**base2, "term": k, **gcols(k), "status": gate if status[k] == "ESTIMABLE" else status[k]} for k in focal]
    fit, warn = fit_clustered(dall["y"], Xall, dall["person_id"], True)
    crow = {r["term"]: r for r in coefficient_rows(fit, [k for k in kept if k in focal], True)}
    out = []
    for k in focal:
        r = {**base2, "term": k, **gcols(k)}
        if k in crow:
            c_ = crow[k]
            r.update({"status": "ESTIMABLE", "rr": c_["estimate"], "se_log_rr": c_["standard_error"], "ci_low": c_["ci_low"],
                      "ci_high": c_["ci_high"], "p_value": c_["p_value"], "warnings": warn})
        else:
            r["status"] = status[k]
        out.append(r)
    out += contrast_rows(fit, kept, list(contrasts), {**base2})
    return out


# ----------------------------------------------------------------------------------------------
# death dates (analysis B)
# ----------------------------------------------------------------------------------------------
def load_death_dates(co):
    """Person-level death year/month. HRS, KLoSA and SHARE: primary source Stata file (radyear, radmonth).
    ELSA: the primary source has no death-date fields; the formal CSV carries an unmapped legacy radyear
    (year only), read here only to document its coverage. Returns (frame indexed by person_id, source label)."""
    cc = co.universe["cohorts"][co.name]
    if co.name == "elsa":
        lookup, _ = load_lookup(DATA, co.universe)
        lk = lookup.loc[lookup["cohort"].eq("elsa")]
        mp = dict(zip(lk["canonical_variable"], lk["column_name_in_csv"]))
        path = DATA / co.universe["formal_subdir"] / cc["formal_file"]
        f = pd.read_csv(path, usecols=[mp[cc["id"]], mp["radyear"]], dtype={mp[cc["id"]]: "string"}, encoding="utf-8-sig", low_memory=False)
        f["person_id"] = normalize_id(f[mp[cc["id"]]])
        f["yr"] = numeric(f[mp["radyear"]], [1900, 2100]); f["mo"] = np.nan
        src = "formal_csv_legacy_radyear_year_only"
    else:
        path = DATA / cc["primary_source_relative_path"]
        w = pd.read_stata(path, columns=[cc["id"], "radyear", "radmonth"], convert_categoricals=False, preserve_dtypes=False)
        w["person_id"] = normalize_id(w[cc["id"]])
        assert not w["person_id"].duplicated().any()
        f = pd.DataFrame({"person_id": w["person_id"], "yr": numeric(w["radyear"], [1900, 2100]), "mo": numeric(w["radmonth"], [1, 12])})
        src = "primary_source_dta_radyear_radmonth"
    g = f.groupby("person_id").agg(yr_min=("yr", "min"), yr_max=("yr", "max"), mo_min=("mo", "min"), mo_max=("mo", "max"))
    g["consistent"] = (g["yr_min"].fillna(-1) == g["yr_max"].fillna(-1)) & (g["mo_min"].fillna(-1) == g["mo_max"].fillna(-1))
    g["yr"] = g["yr_min"]; g["mo"] = g["mo_min"]
    g["idx"] = g["yr"] * 12 + g["mo"]
    return g[["yr", "mo", "idx", "consistent"]], src


# ----------------------------------------------------------------------------------------------
# selection frame and inclusion design (analysis C)
# ----------------------------------------------------------------------------------------------
COV_REQUIRED = ["age", "sex", "education_rank", "income_rank", "smoke_ever", "smoke_current",
                "baseline_multimorbidity", "baseline_engagement_count"]


def selection_frame(co, outcome: str):
    """Exposure-eligible population for one outcome with retention flags.
    Mortality: scope & core_valid & initial count >= 1 (known-status and covariates enter as exclusion reasons).
    ADL: additionally baseline ADL count observed and zero at t1, and known alive at the outcome wave (survivors);
    deaths and unknown-status intervals are reported separately and are not 'missing outcome'."""
    ep, b, u = co.episodes, co.behavior, co.universe
    elig = ep["comparable_window"].fillna(False).astype(bool) & b["core_valid"].fillna(False).astype(bool) & b["baseline_engagement_count"].ge(1)
    out, _base, _st, _bin = outcome_and_baseline(ep, u, co.name, outcome)
    dc, ac = ep.attrs["death_codes"], ep.attrs["alive_codes"]
    status_known = ep["next_status"].isin(dc + ac)
    died = ep["next_status"].isin(dc)
    alive = ep["next_status"].isin(ac)
    cov = covariate_frame(ep, b, u, co.name, outcome)
    cov_ok = cov[COV_REQUIRED].notna().all(axis=1)
    f = pd.DataFrame(index=ep.index)
    if outcome == "mortality":
        f["eligible"] = elig
        f["miss_outcome"] = out.isna()
    else:
        spec = u["outcomes"][outcome]
        base_ok = baseline_eligible(ep, outcome, spec, u["outcomes"], co.name)
        f["adl_free_t1"] = elig & base_ok
        f["died_before_t2"] = f["adl_free_t1"] & died
        f["status_unknown"] = f["adl_free_t1"] & ~status_known
        f["eligible"] = elig & base_ok & alive
        f["miss_outcome"] = out.isna()
    f["cov_ok"] = cov_ok
    f["retained_calc"] = f["eligible"] & ~f["miss_outcome"] & f["cov_ok"]
    return f, cov


def pre_exposure_frame(co, idx):
    """Values measured at or before t0, on the given episode index. Missing values stay NaN."""
    ep, b, u = co.episodes, co.behavior, co.universe
    c0 = covariate_frame(ep, b, u, co.name, "mortality", anchor="t0").loc[idx]
    out = pd.DataFrame(index=idx)
    out["srh_t0"] = srh(co, "t0").loc[idx]
    out["adl_t0"] = adl_count(co, "t0").loc[idx]
    out["iadl_t0"] = iadl_count(co, "t0").loc[idx]
    # depressive symptoms at t0 (CESD-8 ELSA/HRS; CESD-10 version a/b KLoSA; EURO-D SHARE)
    e = ep.loc[idx]
    if co.name in {"elsa", "hrs"}:
        dep = numeric(e["t0__cesd"], [0, 8])
    elif co.name == "klosa":
        dep = pd.Series(np.nan, index=idx)
        a = e["t0"].isin([3, 4]); bb = e["t0"].isin([5, 6, 7, 8, 9])
        dep.loc[a] = numeric(e.loc[a, "t0__cesd10a"], [0, 30]); dep.loc[bb] = numeric(e.loc[bb, "t0__cesd10b"], [0, 30])
    else:
        dep = numeric(e["t0__eurod"], [0, 12])
    z = dep.groupby(e["t0"]).transform(lambda s: (s - s.mean()) / s.std(ddof=0) if s.notna().sum() > 1 and s.std(ddof=0) > 0 else s * 0.0)
    out["dep_t0_z"] = z
    out["education_rank_t0"] = c0["education_rank"]
    out["income_rank_t0"] = c0["income_rank"]
    out["smoke_ever_t0"] = c0["smoke_ever"]
    out["smoke_current_t0"] = c0["smoke_current"]
    out["multimorbidity_t0"] = c0["baseline_multimorbidity"]
    out["tm1_present"] = e["tm1_present"].astype(float)
    return out


def inclusion_design(co, idx, cov_t1):
    """Predictors of retention, all measured at or before t0 or defining the exposure-eligible interval.
    Every predictor with missing values enters with a missing indicator (zero-filled)."""
    ep, b = co.episodes, co.behavior
    pre = pre_exposure_frame(co, idx)
    X = pd.DataFrame({"intercept": 1.0}, index=idx)
    age = cov_t1.loc[idx, "age"]
    X = pd.concat([X, restricted_cubic_spline(age, [60, 70, 80, 90], "age")], axis=1)
    sex = cov_t1.loc[idx, "sex"]
    X["female"] = sex.eq(2.0).astype(float)
    X["sex_missing"] = sex.isna().astype(float)
    wave = pd.get_dummies(ep.loc[idx, "t1"].astype(int).astype(str), prefix="wave", drop_first=True, dtype=float)
    X = pd.concat([X, wave], axis=1)
    for s in DOMS:
        X[f"{s}_base"] = b.loc[idx, f"{s}_baseline"].astype(float)
        X[f"{s}_loss"] = b.loc[idx, f"{s}_loss"].astype(float)
        X[f"{s}_gain"] = b.loc[idx, f"{s}_gain"].astype(float)
    srh_t0 = pre["srh_t0"]
    for k in [2, 3, 4, 5]:
        X[f"srh_t0_{k}"] = srh_t0.eq(k).astype(float)
    adl0 = pre["adl_t0"]
    X["adl_t0_1"] = adl0.eq(1).astype(float); X["adl_t0_2p"] = adl0.ge(2).astype(float)
    iadl0 = pre["iadl_t0"]
    X["iadl_t0_1"] = iadl0.eq(1).astype(float); X["iadl_t0_2p"] = iadl0.ge(2).astype(float)
    mm = pre["multimorbidity_t0"]
    X["mm_t0_1"] = mm.eq(1).astype(float); X["mm_t0_2"] = mm.eq(2).astype(float); X["mm_t0_3p"] = mm.ge(3).astype(float)
    for nm in ["dep_t0_z", "education_rank_t0", "income_rank_t0", "smoke_ever_t0", "smoke_current_t0"]:
        X[nm] = pre[nm].fillna(0.0)
    for nm, s in [("srh_t0", srh_t0), ("adl_t0", adl0), ("iadl_t0", iadl0), ("dep_t0", pre["dep_t0_z"]),
                  ("education_t0", pre["education_rank_t0"]), ("income_t0", pre["income_rank_t0"]),
                  ("smoke_t0", pre["smoke_ever_t0"]), ("mm_t0", mm)]:
        X[f"{nm}_missing"] = s.isna().astype(float)
    X["tm1_present"] = pre["tm1_present"]
    X = X.loc[:, [c == "intercept" or X[c].std() > 0 for c in X.columns]]
    return X.astype(float), pre


def fit_weighted(y, X, groups, w, family="poisson"):
    fam = sm.families.Poisson() if family == "poisson" else sm.families.Binomial()
    with warnings.catch_warnings(record=True) as cap:
        warnings.simplefilter("always")
        res = sm.GLM(np.asarray(y, float), X, family=fam, freq_weights=np.asarray(w, float)).fit(
            cov_type="cluster", cov_kwds={"groups": np.asarray(groups)})
    return res, " | ".join(str(c.message) for c in cap)


def ess_kish(w):
    w = np.asarray(w, float)
    return float(w.sum() ** 2 / np.sum(w ** 2))


def pool_log(est: pd.DataFrame, rr_col="rr", se_col="se_log_rr"):
    """REML + Hartung-Knapp (safeguarded CI primary) on log RR; requires >= 3 cohorts."""
    fitr = reml_hk(np.log(est[rr_col].to_numpy(float)), est[se_col].to_numpy(float) ** 2)
    return {"rr": float(np.exp(fitr["pooled"])), "ci_low": float(np.exp(fitr["ci_low_safeguarded"])),
            "ci_high": float(np.exp(fitr["ci_high_safeguarded"])),
            "ci_low_unmodified_hk": float(np.exp(fitr["ci_low"])), "ci_high_unmodified_hk": float(np.exp(fitr["ci_high"])),
            "pi_low": float(np.exp(fitr["prediction_low_safeguarded"])), "pi_high": float(np.exp(fitr["prediction_high_safeguarded"])),
            "i2": fitr["i2"], "tau2": fitr["tau2"], "hk_scale": fitr["hk_scale"]}


def fixed_effect_log(est: pd.DataFrame, rr_col="rr", se_col="se_log_rr"):
    yi = np.log(est[rr_col].to_numpy(float)); vi = est[se_col].to_numpy(float) ** 2
    w = 1 / vi; mu = float((w * yi).sum() / w.sum()); se = float((1 / w.sum()) ** 0.5)
    return {"rr": float(np.exp(mu)), "ci_low": float(np.exp(mu - Z975 * se)), "ci_high": float(np.exp(mu + Z975 * se))}
