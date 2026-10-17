#!/usr/bin/env python3
"""E2: pre-window trajectories by later cessation group, and the already-healthy (stable-subset) check."""
import argparse, hashlib, json
import numpy as np, pandas as pd
from exp_common import *    # noqa
from exp_measures import build_measures

AGE_KNOTS = [60, 70, 80, 90]
TERMS = ["loss_1", "loss_2plus"]


def count_cells(sel: pd.DataFrame, y: pd.Series, binary_form: bool) -> dict:
    g0 = sel["loss_1"].eq(0) & sel["loss_2plus"].eq(0)
    out = {}
    for label, mask in [("0", g0), ("1", sel["loss_1"].eq(1)), ("2plus", sel["loss_2plus"].eq(1))]:
        out[f"n_group_{label}"] = int(mask.sum())
        out[f"{'events' if binary_form else 'mean_raw'}_group_{label}"] = float(y.loc[mask].sum()) if binary_form else float(y.loc[mask].mean()) if mask.any() else np.nan
    return out


def e2_models(co, data, cov, measures):
    rows, support = [], []
    ep = co.episodes
    idx0 = data.index
    for gap_rule in ["all", "22_30"]:
        for adjustment in ["A", "B"]:
            for m in measures:
                base = dict(cohort=co.name, measure=m.name, form=m.form, adjustment=adjustment, gap_rule=gap_rule,
                            poolable=m.poolable, primary_measure=m.primary, note=m.note)
                req = ["age", "sex", "education_rank", "income_rank"]
                if adjustment == "B":
                    req += ["smoke_ever", "smoke_current", "baseline_multimorbidity", "baseline_engagement_count"]
                ok = ep["tm1_present"].copy()
                if gap_rule == "22_30":
                    ok &= ep["tm1_gap_22_30"]
                ok &= m.y.notna() & m.level.notna()
                ok &= cov[req].notna().all(axis=1)
                keep = idx0[ok.reindex(idx0).fillna(False).to_numpy()]
                if len(keep) == 0:
                    rows.append({**base, "term": "all", "n": 0, "people": 0, "status": "NOT_AVAILABLE"}); continue
                d = pd.DataFrame(index=keep)
                d["person_id"] = data.loc[keep, "person_id"]
                d["loss_1"] = data.loc[keep, "loss_1"]; d["loss_2plus"] = data.loc[keep, "loss_2plus"]
                X = pd.DataFrame({"intercept": 1.0}, index=keep)
                X = pd.concat([X, restricted_cubic_spline(cov.loc[keep, "age"], AGE_KNOTS, "age")], axis=1)
                X["female_code2"] = cov.loc[keep, "sex"].eq(2.0).astype(float)
                X = pd.concat([X, pd.get_dummies(data.loc[keep, "t1_wave"].astype(int).astype(str), prefix="t1_wave", drop_first=True, dtype=float)], axis=1)
                X["education_rank"] = cov.loc[keep, "education_rank"]; X["income_rank"] = cov.loc[keep, "income_rank"]
                X["tm1_level"] = m.level.loc[keep].astype(float)
                if adjustment == "B":
                    for c in ["baseline_engagement_count", "smoke_ever", "smoke_current", "baseline_multimorbidity"]:
                        X[c] = cov.loc[keep, c].astype(float)
                X = X.loc[:, [c == "intercept" or X[c].std() > 0 for c in X.columns]]
                binary_form = m.form == "binary"
                yv = m.y.loc[keep].astype(float)
                sd = np.nan
                if not binary_form:
                    sd = float(yv.std(ddof=1)); yv = yv / sd
                for t in TERMS:
                    X[t] = d[t].astype(float)
                status, counts = support_gate(d, yv, X, TERMS, co.pilot, binary_form)
                cells = count_cells(d, m.y.loc[keep], binary_form)
                info = {**base, "n": len(keep), "people": int(d["person_id"].nunique()), "n_parameters": X.shape[1],
                        "sd_change": sd, **cells, "status": status,
                        "prevalence_or_mean_all": float(m.y.loc[keep].mean())}
                if status != "ESTIMABLE":
                    rows.append({**info, "term": "all"}); continue
                fit, warn = fit_clustered(yv, X, d["person_id"], binary_form)
                for r in coefficient_rows(fit, TERMS, binary_form):
                    row = {**info, "term": r["term"], "estimate": r["estimate"], "se": r["standard_error"],
                           "ci_low": r["ci_low"], "ci_high": r["ci_high"], "p_value": r["p_value"], "warnings": warn}
                    if not binary_form:
                        row["raw_unit_estimate"] = r["estimate"] * sd
                    rows.append(row)
    return rows


def stable_rows(co):
    data, y, X, terms = primary_set(co, "mortality")
    assert_published(co, "mortality", data, y)
    idx = data.index
    ep = co.episodes.loc[idx]
    s_m, s_0 = srh(co, "tm1").loc[idx], srh(co, "t0").loc[idx]
    a_m, a_0 = adl_count(co, "tm1").loc[idx], adl_count(co, "t0").loc[idx]
    T = ep["tm1_present"] & s_m.notna() & s_0.notna() & a_m.notna() & a_0.notna()
    stable = T & (s_0 <= s_m) & (a_0 == 0)
    g = ep["tm1_gap_22_30"].fillna(False)
    pops = {"all_primary": pd.Series(True, index=idx), "T_all": T, "T_stable": stable, "T_not_stable": T & ~stable,
            "T_all_gap22_30": T & g, "T_stable_gap22_30": stable & g}
    rows = []
    for label, mask in pops.items():
        sub = idx[mask.to_numpy()]
        base = {"cohort": co.name, "population": label}
        if len(sub) == 0:
            rows.append({**base, "term": "all", "n": 0, "status": "NOT_AVAILABLE"}); continue
        d = data.loc[sub]; ys = y.loc[sub]
        Xs = X.loc[sub]
        # Rebuild the wave indicators inside the subset (same coding as the pipeline: drop the lowest
        # wave present). Otherwise, when the reference wave is absent (no t-1 exists for it), the
        # remaining indicators sum to 1 and are collinear with the intercept.
        Xs = Xs.drop(columns=[c for c in Xs.columns if c.startswith("t1_wave_")])
        wave = pd.get_dummies(d["t1_wave"].astype(int).astype(str), prefix="t1_wave", drop_first=True, dtype=float)
        Xs = pd.concat([Xs, wave], axis=1)
        Xs = Xs.loc[:, [c == "intercept" or c in terms or Xs[c].std() > 0 for c in Xs.columns]]
        status, counts = support_gate(d, ys, Xs, terms, co.pilot, True)
        g0 = d["loss_1"].eq(0) & d["loss_2plus"].eq(0)
        info = {**base, "n": len(sub), "people": int(d["person_id"].nunique()), "deaths": int(ys.sum()),
                "n_group_0": int(g0.sum()), "deaths_group_0": int(ys[g0].sum()),
                "n_group_1": int(d["loss_1"].eq(1).sum()), "deaths_group_1": int(ys[d["loss_1"].eq(1)].sum()),
                "n_group_2plus": int(d["loss_2plus"].eq(1).sum()), "deaths_group_2plus": int(ys[d["loss_2plus"].eq(1)].sum()),
                "n_parameters": Xs.shape[1], "status": status}
        if status != "ESTIMABLE":
            rows.append({**info, "term": "all"}); continue
        fit, warn = fit_clustered(ys, Xs, d["person_id"], True)
        for r in coefficient_rows(fit, terms, True):
            rows.append({**info, "term": r["term"], "rr": r["estimate"], "se_log_rr": r["standard_error"],
                         "ci_low": r["ci_low"], "ci_high": r["ci_high"], "p_value": r["p_value"]})
        if label == "all_primary":
            pub = PUB_RR["mortality"][co.name]
            got = [r["estimate"] for r in coefficient_rows(fit, terms, True)]
            assert np.allclose(got, pub, rtol=2e-5), (co.name, got, pub)
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--cohort", required=True); a = ap.parse_args()
    co = load_cohort(a.cohort)
    data, y, X, terms = primary_set(co, "mortality")
    got = assert_published(co, "mortality", data, y)
    cov = covariate_frame(co.episodes, co.behavior, co.universe, co.name, "mortality", anchor="t0")
    rows = e2_models(co, data, cov, build_measures(co))
    write_csv(OUT / f"e2_cohort_{a.cohort}.csv", pd.DataFrame(rows))
    write_csv(OUT / f"e2_stable_{a.cohort}.csv", pd.DataFrame(stable_rows(co)))


    print(json.dumps({"status": "PASS", "cohort": a.cohort, "rows": len(rows)}))
