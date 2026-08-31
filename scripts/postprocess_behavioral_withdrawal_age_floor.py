import glob

import numpy as np
import pandas as pd
from postprocess_behavior_outcome_landscape import reml_hk

pd.set_option("display.width", 240)
d = pd.concat([pd.read_csv(f) for f in sorted(glob.glob("*-age-floor-models.csv"))])

print("########## death, two or more stopped: what each floor costs ##########")
q = d[d.outcome_id.eq("mortality") & d.term.eq("loss_2plus")]
print(q[["cohort", "age_floor", "model_status", "n", "events", "median_age",
         "estimate", "ci_low", "ci_high"]].round(2).to_string(index=False))

print("\n########## pooled at each floor ##########")
for oc in ["mortality", "incident_any_adl", "incident_any_iadl", "multimorbidity_progression"]:
    for term in ["loss_1", "loss_2plus"]:
        for floor in [60, 65, 70]:
            g = d[d.outcome_id.eq(oc) & d.term.eq(term) & d.age_floor.eq(floor)
                  & d.model_status.eq("PASS")].drop_duplicates("cohort")
            if len(g) < 3:
                miss = d[d.outcome_id.eq(oc) & d.term.isna() & d.age_floor.eq(floor)]
                why = ";".join(sorted(set(miss.model_status.astype(str))))
                print(f"  {oc:28s} {term:11s} floor {floor}  k={len(g)}  not pooled  {why}")
                continue
            fit = reml_hk(np.log(g.estimate.to_numpy(float)),
                          g.standard_error.to_numpy(float) ** 2)
            rr, lo, hi = (np.exp(fit["pooled"]), np.exp(fit["ci_low"]), np.exp(fit["ci_high"]))
            pl, ph = np.exp(fit["prediction_low"]), np.exp(fit["prediction_high"])
            print(f"  {oc:28s} {term:11s} floor {floor}  k={len(g)}  "
                  f"N={int(g.n.sum()):>7,}  ev={int(g.events.sum()):>6,}  "
                  f"RR {rr:.2f} ({lo:.2f}-{hi:.2f})  PI {pl:.2f}-{ph:.2f}  "
                  f"I2={fit['i2'] * 100:.0f}%")
    print()

dist = [pd.read_csv(f) for f in sorted(glob.glob("*-age-distribution.csv"))]
if dist:
    print("########## age distribution of the eligible risk set ##########")
    print(pd.concat(dist).to_string(index=False))
