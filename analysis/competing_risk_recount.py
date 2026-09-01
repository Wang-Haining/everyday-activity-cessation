"""Discrete-time competing-risk recount from the frozen aggregate cell counts.

Non-fatal outcomes are ascertained at the third visit, so anyone who died first
leaves the risk set. Cessation predicts death, so the exposed group loses more
people before the outcome is measured, and a reviewer will ask what that does.

Fine-Gray needs event times. This design has one fixed 22-30 month interval and
a binary outcome at the third visit, so the subdistribution estimate reduces to
a recount: keep the people who died in the denominator and code them as
non-events. Both quantities below follow in closed form from the published cell
counts, given one assumption stated in the output.
"""
import sys, pathlib
import numpy as np, pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
from postprocess_behavior_outcome_landscape import reml_hk

ROOT = pathlib.Path(__file__).resolve().parents[1]
M = pd.read_csv(ROOT / "artifacts/multidomain_behavioral_withdrawal_pilot/final/systematic-results-matrix.csv")
S = pd.read_csv(ROOT / "artifacts/multidomain_behavioral_withdrawal_pilot/final/cross-cohort-summary.csv")

def cells(outcome):
    q = M[M.scope.eq("comparable_22_30_months") & M.adjustment.eq("full")
          & M.exposure_model.eq("score_categorical") & M.model_status.eq("PASS")
          & M.outcome_id.eq(outcome)].drop_duplicates("cohort")
    out = {}
    for _, r in q.iterrows():
        ref_n = r.n - r.loss_1_n - r.loss_2plus_n
        ref_e = r.events - r.loss_1_events - r.loss_2plus_events
        out[r.cohort] = {"0": (ref_n, ref_e),
                         "1": (r.loss_1_n, r.loss_1_events),
                         "2": (r.loss_2plus_n, r.loss_2plus_events)}
    return out

def logrr(n1, e1, n0, e0):
    """log RR and its variance, with a 0.5 correction if a cell is empty."""
    if min(e1, e0) == 0:
        e1, e0, n1, n0 = e1 + .5, e0 + .5, n1 + 1, n0 + 1
    y = np.log((e1 / n1) / (e0 / n0))
    v = 1 / e1 - 1 / n1 + 1 / e0 - 1 / n0
    return y, v

DASH = "--"


def fmt(x):
    return f"{x[0]:.2f} ({x[1]:.2f}{DASH}{x[2]:.2f})"


death = cells("mortality")
OUTCOMES = [("incident_any_adl", "New ADL limitation"),
            ("multimorbidity_progression", "Multimorbidity progression")]
TERMS = [("1", "loss_1", "One activity stopped"),
         ("2", "loss_2plus", "Two or more stopped")]

print("Discrete-time competing-risk recount, comparable 22-30 month window")
print("Assumption: among people free of the outcome at the middle interview, the")
print("risk of dying before the outcome interview equals the risk observed in the")
print("mortality risk set of the same cohort and exposure group. Intervals with the")
print("outcome already present end in death more often, so this OVERSTATES the deaths")
print("added back, and therefore overstates how far the estimates move.")
print()

for outcome, olabel in OUTCOMES:
    cur = cells(outcome)
    print(f"=== {olabel} ===")
    for key, term, tlabel in TERMS:
        rows_cur, rows_sub, rows_comp = [], [], []
        detail = []
        for cohort in sorted(cur):
            if cohort not in death:
                continue
            n_g, e_g = cur[cohort][key]
            n_0, e_0 = cur[cohort]["0"]
            dn_g, de_g = death[cohort][key]
            dn_0, de_0 = death[cohort]["0"]
            p_g, p_0 = de_g / dn_g, de_0 / dn_0          # death risk by group
            # survivors n_g imply a pre-death denominator of n_g / (1 - p)
            N_g, N_0 = n_g / (1 - p_g), n_0 / (1 - p_0)
            D_g, D_0 = N_g - n_g, N_0 - n_0              # deaths added back
            rows_cur.append(logrr(n_g, e_g, n_0, e_0))
            rows_sub.append(logrr(N_g, e_g, N_0, e_0))
            rows_comp.append(logrr(N_g, e_g + D_g, N_0, e_0 + D_0))
            detail.append((cohort, p_g, p_0, D_g, D_0))
        def pool(rows):
            y = np.array([r[0] for r in rows]); v = np.array([r[1] for r in rows])
            f = reml_hk(y, v)
            return np.exp(f["pooled"]), np.exp(f["ci_low"]), np.exp(f["ci_high"])
        a = pool(rows_cur); b = pool(rows_sub); c = pool(rows_comp)
        pub = S[S.exposure_model.eq("score_categorical") & S.outcome_id.eq(outcome)
                & S.term.eq(term)]
        pubrr = float(pub.pooled_estimate.iloc[0]) if len(pub) else float("nan")
        print(f"  {tlabel}")
        print(f"    published adjusted RR (survivors only) {pubrr:.2f}")
        print(f"    crude, survivors only                  {a[0]:.2f} ({a[1]:.2f}-{a[2]:.2f})")
        print(f"    crude, deaths retained                 {b[0]:.2f} ({b[1]:.2f}-{b[2]:.2f})"
              f"   shift x{b[0]/a[0]:.3f}  -> adjusted ~{pubrr*b[0]/a[0]:.2f}")
        print(f"    crude, death-or-outcome composite      {c[0]:.2f} ({c[1]:.2f}-{c[2]:.2f})"
              f"   shift x{c[0]/a[0]:.3f}  -> adjusted ~{pubrr*c[0]/a[0]:.2f}")
        print("    deaths added back: " + ", ".join(
            f"{ch} {int(round(dg)):,}/{int(round(d0)):,}" for ch, _, _, dg, d0 in detail))
        print()


# ---------------------------------------------------------------------------
# Supplementary table
# ---------------------------------------------------------------------------

def table() -> None:
    lines = [
        r"\begin{table*}[htbp]",
        r"\caption{Competing-risk treatments of death for the non-fatal outcomes}",
        r"\label{tab:competing}",
        r"\centering\footnotesize",
        r"\makebox[\textwidth][c]{%",
        r"\begin{tabular}{@{}llcccc@{}}",
        r"\toprule",
        r"Outcome & Activities & Reported & \multicolumn{3}{c}{Crude RR (95\% CI)} \\",
        r"\cmidrule(l){4-6}",
        r" & stopped & RR & Survivors only & Deaths retained & Death or outcome \\",
        r"\midrule",
    ]
    for outcome, olabel in OUTCOMES:
        cur = cells(outcome)
        for ti, (key, term, tlabel) in enumerate(TERMS):
            rc, rs, rk = [], [], []
            for cohort in sorted(cur):
                if cohort not in death:
                    continue
                n_g, e_g = cur[cohort][key]
                n_0, e_0 = cur[cohort]["0"]
                dn_g, de_g = death[cohort][key]
                dn_0, de_0 = death[cohort]["0"]
                p_g, p_0 = de_g / dn_g, de_0 / dn_0
                N_g, N_0 = n_g / (1 - p_g), n_0 / (1 - p_0)
                rc.append(logrr(n_g, e_g, n_0, e_0))
                rs.append(logrr(N_g, e_g, N_0, e_0))
                rk.append(logrr(N_g, e_g + N_g - n_g, N_0, e_0 + N_0 - n_0))

            def pool(rows):
                y = np.array([r[0] for r in rows]); v = np.array([r[1] for r in rows])
                f = reml_hk(y, v)
                return np.exp(f["pooled"]), np.exp(f["ci_low"]), np.exp(f["ci_high"])

            pub = S[S.exposure_model.eq("score_categorical") & S.outcome_id.eq(outcome)
                    & S.term.eq(term)]
            pubrr = f"{float(pub.pooled_estimate.iloc[0]):.2f}" if len(pub) else "--"
            head = olabel if ti == 0 else ""
            lines.append(f"{head} & {tlabel.replace(' stopped','').replace(' activity','')} & "
                         f"{pubrr} & {fmt(pool(rc))} & {fmt(pool(rs))} & {fmt(pool(rk))} \\\\")
        lines.append(r"\addlinespace")
    lines = lines[:-1]
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}}",
        r"\begin{flushleft}\footnotesize This is a fixed-horizon competing-death "
        r"sensitivity analysis, not an individual-level subdistribution analysis. The "
        r"design has one 22 to 30 month interval and no event times, so deaths cannot "
        r"be followed individually and are instead returned to each exposure group at "
        r"the rate observed in the mortality risk set of the same cohort and group. "
        r"Non-fatal outcomes are ascertained at the outcome interview, so a "
        r"person-interval whose respondent died first leaves the risk set. "
        r"\emph{Survivors only} is the risk set the primary models use. "
        r"\emph{Deaths retained} returns those intervals to the denominator and codes "
        r"them as not having the outcome. \emph{Death or outcome} counts death as the "
        r"outcome. Intervals already carrying the outcome end in death more often, so "
        r"the rate used returns more deaths than occurred and the movement shown is "
        r"larger than an individual-level analysis would give. The last three columns "
        r"are crude, pooled by the same random-effects synthesis, and are shown "
        r"together so that the effect of the choice can be read without the "
        r"adjustment; the reported adjusted risk ratio is repeated in the third "
        r"column.\end{flushleft}",
        r"\end{table*}",
    ])
    out = ROOT / "manuscript/generated/supp_table_competing_risk.tex"
    body = "\n".join(lines) + "\n"
    # Only write when the content actually changes. A generator that rewrites an
    # identical file gives it a new timestamp, which makes every product built
    # from it report as stale, and the rebuild that silences that is a rebuild of
    # a document that did not move.
    if out.exists() and out.read_text(encoding="utf-8") == body:
        print(f"{out.name} unchanged")
        return
    out.write_text(body, encoding="utf-8")
    print(f"wrote {out.name}")


table()
