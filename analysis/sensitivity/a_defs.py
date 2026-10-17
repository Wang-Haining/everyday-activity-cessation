"""Analysis A definitions shared by the per-cohort and one-stage runners."""
import numpy as np, pandas as pd
from x3_common import DOMS, LETTER, PAIRS

# E4 stratum: exactly one loss, gain attainable (initial count 1 or 2)
def e4_mask(d):
    return d["loss_n"].eq(1) & d["opp"].isin([1, 2])


def lost_domain_cols(d):
    """Two indicators for which domain was lost (work lost = reference); exactly one loss in the stratum."""
    return pd.DataFrame({"lost_alcohol": d["alcohol_loss"].astype(float), "lost_activity": d["activity_loss"].astype(float)}, index=d.index)


def cell_cols(d):
    """Initial pattern x lost domain cells inside the E4 stratum (most common cell is the reference).
    Rows outside the stratum get all zeros (they are masked out before fitting)."""
    m = e4_mask(d)
    lab = d[[f"{s}_base" for s in DOMS]].astype(str).agg("".join, axis=1) + "_" + d[[f"{s}_loss" for s in DOMS]].astype(str).agg("".join, axis=1)
    lab = lab.where(m, "OUT")
    ref = lab[m].value_counts().index[0]
    dm = pd.get_dummies(lab, prefix="cell", dtype=float)
    dm = dm.drop(columns=[c for c in dm.columns if c in (f"cell_{ref}", "cell_OUT")])
    return dm


E4_VARIANTS = {
    "attainable_unadjusted": None,
    "attainable_lost_domain_adjusted": lost_domain_cols,
    "attainable_initial_pattern_x_lost_domain": cell_cols,
}
# the linear initial-count covariate is a function of the cell dummies, so it is dropped in the cell specification
E4_DROP = {"attainable_initial_pattern_x_lost_domain": ["baseline_engagement_count"]}


def pair_groups(d, d1, d2):
    third = [s for s in DOMS if s not in (d1, d2)][0]
    pair = f"{LETTER[d1]}{LETTER[d2]}"
    names = [f"pair_{pair}", f"single_{LETTER[d1]}", f"single_{LETTER[d2]}", f"nuisance_{LETTER[third]}_lost"]
    m = d[f"{d1}_base"].eq(1) & d[f"{d2}_base"].eq(1)
    l1, l2, l3 = d[f"{d1}_loss"].eq(1), d[f"{d2}_loss"].eq(1), d[f"{third}_loss"].eq(1)
    return pair, names, m, {names[0]: l1 & l2 & ~l3, names[1]: l1 & ~l2 & ~l3, names[2]: ~l1 & l2 & ~l3, names[3]: l3}
