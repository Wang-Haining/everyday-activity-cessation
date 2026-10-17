"""What the complete-case rule costs, pooled across cohorts."""
import glob

import numpy as np
import pandas as pd

from postprocess_behavior_outcome_landscape import reml_hk

VARIANTS = ["all_minimal", "complete_minimal", "complete_full", "all_nosmoking"]

pd.set_option("display.width", 240)
d = pd.concat([pd.read_csv(f) for f in sorted(glob.glob("*-missingness-models.csv"))])

print("########## what the complete-case rule removes, by cohort ##########")
q = (d[d.outcome_id.eq("mortality") & d.term.eq("loss_2plus")]
     .drop_duplicates(["cohort", "variant"]))
print(q[["cohort", "variant", "eligible_intervals", "complete_case_intervals",
         "n", "events", "covariate_columns", "model_status",
         "estimate", "ci_low", "ci_high"]].round(3).to_string(index=False))

print("\n########## pooled, each variant ##########")
for outcome in ["mortality", "incident_any_adl", "incident_any_iadl",
                "multimorbidity_progression"]:
    for term in ["loss_1", "loss_2plus"]:
        for variant in VARIANTS:
            g = d[d.outcome_id.eq(outcome) & d.term.eq(term) & d.variant.eq(variant)
                  & d.model_status.eq("PASS")].drop_duplicates("cohort")
            if len(g) < 3:
                print(f"  {outcome:28s} {term:11s} {variant:17s} k={len(g)}  not pooled")
                continue
            fit = reml_hk(np.log(g.estimate.to_numpy(float)),
                          g.standard_error.to_numpy(float) ** 2)
            rr, lo, hi = (np.exp(fit["pooled"]), np.exp(fit["ci_low_safeguarded"]), np.exp(fit["ci_high_safeguarded"]))
            print(f"  {outcome:28s} {term:11s} {variant:17s} k={len(g)}  "
                  f"N={int(g.n.sum()):>7,}  ev={int(g.events.sum()):>6,}  "
                  f"RR {rr:.2f} ({lo:.2f}-{hi:.2f})  I2={fit['i2']*100:.0f}%  "
                  f"tau2={fit['tau2']:.3f}")
    print()

