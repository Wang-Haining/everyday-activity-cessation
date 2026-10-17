#!/usr/bin/env python3
"""C: selection sensitivity. Retained vs excluded exposure-eligible intervals; inclusion models; stabilised
inverse-probability-of-inclusion weights; weighted vs complete-case 0/1/2+ models and common-target
standardised risks. One cohort per task; mortality and categorical ADL. Aggregates only."""
import argparse, hashlib, json
import numpy as np, pandas as pd
from scipy.special import expit
from x3_common import *    # noqa

ap = argparse.ArgumentParser(); ap.add_argument("--cohort", required=True); a = ap.parse_args()
co = load_cohort(a.cohort); pilot = co.pilot
TERMS = ["loss_1", "loss_2plus"]
SCEN = {"withdrawal_0": {"loss_1": 0, "loss_2plus": 0}, "withdrawal_1": {"loss_1": 1, "loss_2plus": 0}, "withdrawal_2plus": {"loss_1": 0, "loss_2plus": 1}}
sel_rows, mod_rows = [], []
manifest = {"cohort": a.cohort, "verified": {}, "aggregate_only": True, "respondent_rows_exported": 0}
Q = [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]


def qstats(w, prefix):
    w = np.asarray(w, float)
    r = {f"{prefix}_mean": float(w.mean()), f"{prefix}_sd": float(w.std(ddof=1)), f"{prefix}_min": float(w.min()), f"{prefix}_max": float(w.max()),
         f"{prefix}_ess": ess_kish(w), f"{prefix}_ess_share": ess_kish(w) / len(w)}
    for q, v in zip(Q, np.quantile(w, Q)):
        r[f"{prefix}_p{int(round(q * 100))}"] = float(v)
    return r


def wmean(x, w):
    m = x.notna().to_numpy()
    return float(np.sum(w[m] * x.to_numpy(float)[m]) / np.sum(w[m])) if m.any() else np.nan


for outcome in OUTCOMES:
    if a.cohort not in PUB[outcome]:
        continue
    d, X0, got = build_frame(co, outcome)               # asserts the published risk set
    manifest["verified"][outcome] = got
    data, y, X, terms = primary_set(co, outcome)
    f, cov_t1 = selection_frame(co, outcome)
    assert set(f.index[f["retained_calc"]]) == set(data.index), "retained flags differ from published risk set"
    pop = f.index[f["eligible"]]
    S = f.loc[pop, "retained_calc"].astype(float)
    Z, pre = inclusion_design(co, pop, cov_t1)
    beta = fit_logit(Z.to_numpy(float), S.to_numpy(float), ridge=1e-6)
    p = pd.Series(expit(Z.to_numpy(float) @ beta), index=pop)
    auc = Prep(p.to_numpy(), S.to_numpy()).auc(None)
    ret = S.eq(1.0)
    pbar = float(S.mean())
    ret_idx = pop[ret.to_numpy()]
    lc = co.behavior.loc[pop, "loss_count_core"]
    grp = pd.Series(np.where(lc.eq(0), "0", np.where(lc.eq(1), "1", "2plus")), index=pop)
    p_grp = S.groupby(grp).mean()
    w_raw = pbar / p.loc[ret_idx]
    w_exp = grp.loc[ret_idx].map(p_grp) / p.loc[ret_idx]
    def trunc(w):
        lo, hi = np.quantile(w, [0.01, 0.99]); return w.clip(lo, hi), float(lo), float(hi)
    w_t, lo_t, hi_t = trunc(w_raw)
    w_et, lo_e, hi_e = trunc(w_exp)
    w_t = w_t / w_t.mean(); w_et = w_et / w_et.mean()
    base = {"cohort": a.cohort, "outcome": outcome}
    # ---- flow rows by exposure group
    for lab in ["0", "1", "2plus"]:
        m = grp.eq(lab)
        sel_rows.append({**base, "row_type": "flow", "item": "eligible_retained_by_loss_group", "group": lab,
                         "n_eligible": int(m.sum()), "n_retained": int((m & ret).sum()), "retained_share": float((m & ret).sum() / m.sum()),
                         "n_excluded_outcome_missing": int((m & f.loc[pop, "miss_outcome"]).sum()),
                         "n_excluded_covariates_only": int((m & ~f.loc[pop, "miss_outcome"] & ~f.loc[pop, "cov_ok"]).sum())})
    ex = {"n_eligible": int(len(pop)), "n_retained": int(ret.sum()), "retained_share": pbar,
          "n_excluded_outcome_missing": int(f.loc[pop, "miss_outcome"].sum()),
          "n_excluded_covariates_only": int((~f.loc[pop, "miss_outcome"] & ~f.loc[pop, "cov_ok"]).sum())}
    if outcome != "mortality":
        ex.update({"adl_free_t1_all": int(f["adl_free_t1"].sum()), "died_before_t2_excluded_from_population": int(f["died_before_t2"].sum()),
                   "status_unknown_excluded_from_population": int(f["status_unknown"].sum())})
    sel_rows.append({**base, "row_type": "flow", "item": "overall", "group": "all", **ex})
    # ---- diagnostic: which required covariates are missing, overall and by t1 wave (eligible intervals)
    e_w = co.episodes.loc[pop, "t1"].astype(int)
    for cv in COV_REQUIRED[2:7]:
        miss = cov_t1.loc[pop, cv].isna()
        sel_rows.append({**base, "row_type": "covariate_missingness", "item": cv, "group": "all", "share_missing": float(miss.mean()),
                         "n_missing": int(miss.sum()), "n_eligible": int(len(pop)),
                         "share_missing_by_wave": ";".join(f"{int(w)}:{miss[e_w.eq(w)].mean():.3f}" for w in sorted(e_w.unique()))})
    # ---- retained vs excluded comparisons and balance after weighting
    e1 = co.episodes.loc[pop]
    b = co.behavior.loc[pop]
    vars_ = {
        "age_t1": cov_t1.loc[pop, "age"], "female": cov_t1.loc[pop, "sex"].eq(2.0).astype(float).where(cov_t1.loc[pop, "sex"].notna()),
        "initial_count": b["baseline_engagement_count"].astype(float),
        "drinking_t0": b["alcohol_baseline"].astype(float), "activity_t0": b["activity_baseline"].astype(float), "work_t0": b["work_baseline"].astype(float),
        "loss_1": b["loss_count_core"].eq(1).astype(float), "loss_2plus": b["loss_count_core"].ge(2).astype(float),
        "any_gain": (b[["alcohol_gain", "activity_gain", "work_gain"]].sum(axis=1) >= 1).astype(float),
        "prior_wave_interview_present": pre["tm1_present"],
        "srh_fair_or_poor_t0": pre["srh_t0"].ge(4).astype(float).where(pre["srh_t0"].notna()),
        "adl_any_t0": pre["adl_t0"].ge(1).astype(float).where(pre["adl_t0"].notna()),
        "iadl_any_t0": pre["iadl_t0"].ge(1).astype(float).where(pre["iadl_t0"].notna()),
        "depressive_symptoms_t0_z": pre["dep_t0_z"],
        "conditions_ge2_t0": pre["multimorbidity_t0"].ge(2).astype(float).where(pre["multimorbidity_t0"].notna()),
        "smoking_current_t0": pre["smoke_current_t0"],
        "education_rank_t0": pre["education_rank_t0"], "income_rank_t0": pre["income_rank_t0"],
    }
    s1 = srh(co, "t1").loc[pop]; ad1 = adl_count(co, "t1").loc[pop]; iad1 = iadl_count(co, "t1").loc[pop]
    vars_["srh_fair_or_poor_t1"] = s1.ge(4).astype(float).where(s1.notna())
    vars_["adl_any_t1"] = ad1.ge(1).astype(float).where(ad1.notna())
    vars_["iadl_any_t1"] = iad1.ge(1).astype(float).where(iad1.notna())
    comp = {"excluded_any": ~ret, "excluded_outcome_missing": f.loc[pop, "miss_outcome"].to_numpy() & ~ret.to_numpy(),
            "excluded_covariates_only": (~f.loc[pop, "miss_outcome"] & ~f.loc[pop, "cov_ok"]).to_numpy()}
    wfull = pd.Series(0.0, index=pop); wfull.loc[ret_idx] = w_t.to_numpy()
    for vname, x in vars_.items():
        xr = x[ret.to_numpy()]
        for cname, cm in comp.items():
            xe = x[np.asarray(cm)]
            if xe.notna().sum() < 2 or xr.notna().sum() < 2:
                continue
            v = (xr.var(ddof=0) + xe.var(ddof=0)) / 2
            row = {**base, "row_type": "compare", "item": vname, "group": cname,
                   "mean_retained": float(xr.mean()), "mean_excluded": float(xe.mean()), "n_valid_retained": int(xr.notna().sum()),
                   "n_valid_excluded": int(xe.notna().sum()), "smd_retained_minus_excluded": float((xr.mean() - xe.mean()) / v ** 0.5) if v > 0 else np.nan}
            if cname == "excluded_any":
                sd_all = float(x.std(ddof=0))
                mw = wmean(xr, w_t.to_numpy())
                row.update({"mean_eligible_all": float(x.mean()), "mean_retained_ipw": mw,
                            "smd_retained_vs_eligible_unweighted": float((xr.mean() - x.mean()) / sd_all) if sd_all > 0 else np.nan,
                            "smd_retained_vs_eligible_ipw": float((mw - x.mean()) / sd_all) if sd_all > 0 else np.nan})
            sel_rows.append(row)
    # ---- inclusion model and weight distribution
    wave = e1["t1"].astype(int)
    wave_ret = S.groupby(wave).agg(["sum", "count"])
    zero_waves = [int(k) for k, r in wave_ret.iterrows() if r["sum"] == 0]
    sel_rows.append({**base, "row_type": "inclusion_model", "item": "logistic_inclusion_model", "group": "all",
                     "n_parameters": int(Z.shape[1]), "n_eligible": int(len(pop)), "n_retained": int(ret.sum()), "auc_in_sample": auc,
                     "share_eligible_pred_lt_0.05": float((p < 0.05).mean()), "share_retained_pred_lt_0.05": float((p.loc[ret_idx] < 0.05).mean()),
                     "min_pred_retained": float(p.loc[ret_idx].min()), "waves_with_zero_retention": ";".join(map(str, zero_waves)),
                     "n_eligible_in_zero_retention_waves": int(wave.isin(zero_waves).sum()),
                     "retention_by_wave": ";".join(f"{int(k)}:{int(r['sum'])}/{int(r['count'])}" for k, r in wave_ret.iterrows())})
    for wname, wr, wt, lo, hi in [("stabilised_marginal_numerator", w_raw, w_t, lo_t, hi_t), ("stabilised_exposure_numerator", w_exp, w_et, lo_e, hi_e)]:
        sel_rows.append({**base, "row_type": "weights", "item": wname, "group": "all", "trunc_lo": lo, "trunc_hi": hi,
                         "n_truncated_low": int((wr < lo).sum()), "n_truncated_high": int((wr > hi).sum()),
                         **qstats(wr.to_numpy(), "raw"), **qstats(wt.to_numpy() if wname.endswith("marginal_numerator") else w_et.to_numpy(), "truncated")})
    # ---- weighted models on the published risk set
    gate, counts = support_gate(data, y, X, terms, pilot, True)
    assert gate == "ESTIMABLE", gate
    wser = {"complete_case": pd.Series(1.0, index=data.index),
            "ipw_truncated": pd.Series(w_t.to_numpy(), index=ret_idx).loc[data.index],
            "ipw_untruncated": (w_raw / w_raw.mean()).loc[data.index],
            "ipw_exposure_numerator_truncated": pd.Series(w_et.to_numpy(), index=ret_idx).loc[data.index]}
    for est, w in wser.items():
        assert w.notna().all() and (w > 0).all()
        fit, warn = fit_weighted(y, X, data["person_id"], w, "poisson")
        cr = {r["term"]: r for r in coefficient_rows(fit, TERMS, True)}
        if est == "complete_case":
            assert np.allclose([cr[t]["estimate"] for t in TERMS], PUB_RR[outcome][a.cohort], rtol=2e-5), (outcome, a.cohort)
        for t in TERMS:
            mod_rows.append({**base, "model": "rr_count_model", "estimator": est, "term": t, "rr": cr[t]["estimate"], "se_log_rr": cr[t]["standard_error"],
                             "ci_low": cr[t]["ci_low"], "ci_high": cr[t]["ci_high"], "n": int(len(data)), "events": int(y.sum()),
                             "ess": ess_kish(w.to_numpy()), "warnings": warn})
    # ---- minimal-covariate model on all eligible intervals with the outcome observed (selection by covariate
    #      completeness removed for covariates available in < 95% of those intervals) and on the retained rows
    pop_obs = pop[~f.loc[pop, "miss_outcome"].to_numpy()]
    optional = ["education_rank", "income_rank", "smoke_ever", "smoke_current", "baseline_multimorbidity"]
    avail = {c: float(cov_t1.loc[pop_obs, c].notna().mean()) for c in optional}
    inc = [c for c in optional if avail[c] >= 0.95]
    need = ["age", "sex", *inc]
    okm = cov_t1.loc[pop_obs, need].notna().all(axis=1)
    sub_all = pop_obs[okm.to_numpy()]
    sub_ret = sub_all.intersection(data.index)
    outc, _b, _s, _bi = outcome_and_baseline(co.episodes, co.universe, co.name, outcome)
    for lab, sub in [("minimal_covariates_all_eligible", sub_all), ("minimal_covariates_retained_rows", sub_ret)]:
        dsub = cov_t1.loc[sub].copy()
        dsub["person_id"] = co.episodes.loc[sub, "person_id"].astype("string")
        Xm = pd.DataFrame({"intercept": 1.0}, index=sub)
        Xm = pd.concat([Xm, restricted_cubic_spline(dsub["age"], [60, 70, 80, 90], "age")], axis=1)
        Xm["female_code2"] = dsub["sex"].eq(2.0).astype(float)
        Xm["baseline_engagement_count"] = dsub["baseline_engagement_count"].astype(float)
        wv = pd.get_dummies(co.episodes.loc[sub, "t1"].astype(int).astype(str), prefix="t1_wave", drop_first=True, dtype=float)
        Xm = pd.concat([Xm, wv], axis=1)
        for c in inc:
            Xm[c] = dsub[c].astype(float)
        Xm = Xm.loc[:, [c == "intercept" or Xm[c].std() > 0 for c in Xm.columns]]
        for t in TERMS:
            Xm[t] = co.behavior.loc[sub, t].astype(float)
        ym = outc.loc[sub].astype(float)
        dd_ = pd.DataFrame({"person_id": dsub["person_id"]})
        for t in TERMS:
            dd_[t] = Xm[t]
        gate_m, _c = support_gate(dd_, ym, Xm, TERMS, pilot, True)
        for t in TERMS:
            row = {**base, "model": "rr_count_model", "estimator": lab, "term": t, "n": int(len(sub)), "events": int(ym.sum()),
                   "covariates_included": ";".join(inc), "covariates_omitted_lt95pct_available": ";".join(c for c in optional if c not in inc),
                   "n_parameters": int(Xm.shape[1]), "status": gate_m}
            if gate_m == "ESTIMABLE":
                fit_m, warn_m = fit_weighted(ym, Xm, dd_["person_id"], np.ones(len(sub)), "poisson")
                cr_m = {r["term"]: r for r in coefficient_rows(fit_m, TERMS, True)}
                row.update({"rr": cr_m[t]["estimate"], "se_log_rr": cr_m[t]["standard_error"], "ci_low": cr_m[t]["ci_low"], "ci_high": cr_m[t]["ci_high"], "warnings": warn_m})
            mod_rows.append(row)
    sel_rows.append({**base, "row_type": "covariate_availability", "item": "optional_covariate_availability_among_eligible_outcome_observed", "group": "all",
                     "availability": ";".join(f"{c}:{avail[c]:.3f}" for c in optional), "n_eligible": int(len(pop_obs)),
                     "n_all_eligible_minimal": int(len(sub_all)), "n_retained_minimal": int(len(sub_ret))})
    # ---- common-target standardised risks (initial count >= 2), using logistic regression
    tgt = data["baseline_engagement_count"].ge(2)
    for est in ["complete_case", "ipw_truncated"]:
        w = wser[est]
        fit, warn = fit_weighted(y, X, data["person_id"], w, "binomial")
        cov = fit.cov_params()
        covm = np.asarray(cov)
        wt_ = w.loc[tgt].to_numpy(float)
        res = {}
        for sc, spec in SCEN.items():
            Xs = X.loc[tgt].copy()
            for k, v in spec.items():
                Xs[k] = float(v)
            pr = np.asarray(fit.predict(Xs), float)
            risk = float(np.sum(wt_ * pr) / wt_.sum())
            g = (wt_ * pr * (1 - pr))[:, None] * Xs.to_numpy(float)
            g = g.sum(axis=0) / wt_.sum()
            se = float(np.sqrt(g @ covm @ g))
            res[sc] = (risk, se, g)
        for sc, (risk, se, g) in res.items():
            row = {**base, "model": "standardised_risk_common_target_k_ge_2", "estimator": est, "term": sc, "risk": risk, "se_risk": se,
                   "ci_low": max(0.0, risk - Z975 * se), "ci_high": min(1.0, risk + Z975 * se), "n_target_rows": int(tgt.sum()),
                   "n": int(len(data)), "events": int(y.sum()), "ess": ess_kish(wt_), "warnings": warn}
            if sc != "withdrawal_0":
                rd = risk - res["withdrawal_0"][0]
                g0 = res["withdrawal_0"][2]
                rdse = float(np.sqrt(max(0.0, (g - g0) @ covm @ (g - g0))))
                row.update({"risk_difference_vs_0": rd, "se_risk_difference": rdse})
            mod_rows.append(row)

write_csv(OUT / f"c_sel_{a.cohort}.csv", pd.DataFrame(sel_rows))
write_csv(OUT / f"c_mod_{a.cohort}.csv", pd.DataFrame(mod_rows))


print(json.dumps({"status": "PASS", "cohort": a.cohort, "sel_rows": len(sel_rows), "mod_rows": len(mod_rows)}))
