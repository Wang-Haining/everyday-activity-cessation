# BMC absolute-risk and work-exit extension v0.2

Scope: `comparable_22_30_months`; adjustment: `full`.

The prespecified modified-Poisson standardization in v0.1 failed its probability-range gate. Version 0.2 was frozen before refitting and uses cohort-specific binomial logistic models for standardized absolute risk, with the same risk sets and covariates. Risk-ratio analyses remain modified Poisson models.

## Adjusted standardized risks

Values are pooled risks for 0, 1 and 2 or more recent withdrawals.

| Outcome | 0 | 1 | 2+ | Cohorts |
|---|---:|---:|---:|---|
| Mortality | 2.8% | 5.2% | 9.9% | ELSA, HRS, KLoSA, SHARE (k=4) |
| Incident ADL limitation | 7.7% | 11.5% | 14.9% | ELSA, HRS, SHARE (k=3) |
| Incident IADL limitation | 5.4% | 8.2% | 12.8% | ELSA, HRS, SHARE (k=3) |
| Multimorbidity progression | 14.2% | 15.6% | 17.9% | ELSA, HRS, KLoSA, SHARE (k=4) |

The full cohort-specific estimates and confidence intervals are in `final/cohort_standardized_risks.csv`; pooled estimates and confidence intervals are in `final/pooled_standardized_risks.csv`.

## Retirement-linked versus other work exit

For incident ADL limitation, the direct other-exit versus retirement-linked ratio was 1.27 (95% CI 0.82–1.96) in ELSA and 1.41 (1.14–1.75) in HRS. The random-effects synthesis was 1.38 (0.81–2.35; k=2, descriptive).

For mortality, only HRS met the support rule: retirement-linked exit RR 1.63, other-exit RR 2.26, and direct ratio 1.39 (1.05–1.83). A mortality meta-analysis was not estimable at k=1.

## CHARLS support in the comparable window

| Outcome | n | Events | Status |
|---|---:|---:|---|
| Diabetes | 1,087 | 24 | NOT_EVALUABLE_EPV |
| Stroke | 1,156 | 7 | NOT_EVALUABLE_EPV |
| Heart disease | 1,005 | 34 | NOT_EVALUABLE_EPV |
| Hypertension | 777 | 51 | NOT_EVALUABLE_EPV |
| Mortality | 1,262 | 34 | NOT_EVALUABLE_EPV |
| Incident ADL limitation | 966 | 153 | PASS |
| Incident IADL limitation | 917 | 152 | PASS |
| Multimorbidity progression | 1,144 | 125 | PASS |

These rows appear in Supplementary Table S1.

## Retirement-field provenance

| Cohort | Source field | Source labels | Retirement-linked values |
|---|---|---|---|
| ELSA | `r{wave}retemp` | 0=no; 1=yes | 1 |
| HRS | `r{wave}retemp` | 0=no retire empstat; 1=only retire empstat; 2=retire plus other empstat | 1 or 2 |
| MHAS | `r{wave}retemp` | 0=working; 1=retired; 2=retired and other status | 1 or 2 |
| SHARE | `r{wave}retemp` | 0=not retired empstat; 1=retired empstat | 1 |

The field identifies retirement-linked status, not whether retirement was planned or voluntary. CHARLS and KLoSA did not supply the harmonized retirement field used here. MHAS had no primary 22–30-month interval.

## Scheduled follow-up months

- CHARLS: 2→3 (23)
- ELSA: 2→3 (22), 3→4 (25), 4→5 (24), 5→6 (23), 6→7 (24), 7→8 (24), 8→9 (25)
- HRS: 6→7 (23), 7→8 (24), 8→9 (24), 9→10 (29), 11→12 (23), 12→13 (27), 13→14 (23), 14→15 (23), 15→16 (27)
- KLoSA: 4→5 (25), 5→6 (24), 6→7 (23), 7→8 (24), 8→9 (25)
- SHARE: 4→5 (24), 5→6 (24), 6→7 (25), 8→9 (26)
- MHAS: no scheduled outcome interval in the primary window

## Validation

Quartz validation and a separate local aggregate validator both passed. The local validator reproduced parent counts and statuses, checked all standardized risks and confidence limits were in [0,1], confirmed per-outcome contributing cohorts, verified the frozen interval window and confirmed that the mortality work-exit result was not pooled at k=1.
