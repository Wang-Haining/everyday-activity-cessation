#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
: "${HARMONIZED_ROOT:?Set HARMONIZED_ROOT to the harmonized data directory}"
U=config/behavioral_transition_outcome_scan_universe_v0.1.json
M=config/multidomain_behavioral_withdrawal_pilot_v0.1.json
F=config/behavioral_withdrawal_frailty_extension_v0.2.json
A=config/bmc_absolute_risk_work_exit_extension_v0.2.json
B=config/behavioral_contraction_biomarker_pilot_v0.1.json
Q=config/paper3_alcohol_diabetes_mechanism_v0.1.json
E=config/elsa_behavior_hba1c_early_warning_v0.2.json
for c in charls elsa hrs klosa mhas share; do
  python scripts/run_multidomain_behavioral_withdrawal_pilot.py --mode probe --data-root "$HARMONIZED_ROOT" --universe-config "$U" --pilot-config "$M" --cohort "$c" --output-dir results/primary/probe
  python scripts/run_multidomain_behavioral_withdrawal_pilot.py --mode models --data-root "$HARMONIZED_ROOT" --universe-config "$U" --pilot-config "$M" --cohort "$c" --probe-dir results/primary/probe --output-dir results/primary/models
  python scripts/run_behavioral_withdrawal_competing_context.py --data-root "$HARMONIZED_ROOT" --universe-config "$U" --multidomain-config "$M" --extension-config "$F" --cohort "$c" --output-dir results/context/models
  python scripts/run_bmc_absolute_risk_work_exit_extension.py --data-root "$HARMONIZED_ROOT" --universe-config "$U" --pilot-config "$M" --extension-config "$A" --cohort "$c" --output-dir results/standardization/models
  python scripts/run_behavioral_withdrawal_age_floor.py --data-root "$HARMONIZED_ROOT" --universe-config "$U" --multidomain-config "$M" --extension-config "$F" --cohort "$c" --output-dir results/age_floor
  python scripts/run_behavioral_withdrawal_specificity.py --data-root "$HARMONIZED_ROOT" --universe-config "$U" --multidomain-config "$M" --extension-config "$F" --cohort "$c" --output-dir results/frailty/models
  python scripts/probe_exclusion_reasons.py --data-root "$HARMONIZED_ROOT" --universe-config "$U" --pilot-config "$M" --cohort "$c" --output-dir results/descriptive
  python scripts/probe_behavioral_contraction_biomarker_pilot.py --data-root "$HARMONIZED_ROOT" --universe-config "$U" --pilot-config "$B" --assay-config "$Q" --cohort "$c" --output-dir results/biomarkers/probe
  python scripts/run_behavioral_contraction_biomarker_pilot.py --data-root "$HARMONIZED_ROOT" --universe-config "$U" --pilot-config "$B" --assay-config "$Q" --cohort "$c" --probe-dir results/biomarkers/probe --output-dir results/biomarkers/models
 done
python scripts/postprocess_multidomain_behavioral_withdrawal_pilot.py --model-dir results/primary/models --pilot-config "$M" --output-dir results/primary
python scripts/postprocess_behavioral_withdrawal_competing_context.py --model-dir results/context/models --output-dir results/context/final
python scripts/postprocess_bmc_absolute_risk_work_exit_extension.py --extension-config "$A" --model-dir results/standardization/models --output-dir results/standardization
python scripts/postprocess_behavioral_contraction_biomarker_pilot.py --pilot-config "$B" --model-dir results/biomarkers/models --probe-dir results/biomarkers/probe --output-dir results/biomarkers
for c in elsa hrs share; do
  python scripts/run_multidomain_behavioral_withdrawal_pilot.py --mode models --minimum-baseline-opportunities 2 --data-root "$HARMONIZED_ROOT" --universe-config "$U" --pilot-config "$M" --cohort "$c" --probe-dir results/primary/probe --output-dir results/opportunity2
 done
for c in charls elsa hrs klosa share; do
  python scripts/run_behavioral_withdrawal_missingness.py --data-root "$HARMONIZED_ROOT" --universe-config "$U" --multidomain-config "$M" --extension-config "$F" --cohort "$c" --output-dir results/missingness
 done
for c in elsa hrs klosa share; do
  python scripts/run_behavioral_withdrawal_component_equality.py --data-root "$HARMONIZED_ROOT" --universe-config "$U" --multidomain-config "$M" --extension-config "$F" --cohort "$c" --output-dir results/component_equality
 done
for c in hrs share; do
  python scripts/probe_design_sensitivity.py --data-root "$HARMONIZED_ROOT" --universe-config "$U" --pilot-config "$M" --cohort "$c" --output-dir results/sensitivity/design
 done
python scripts/probe_frailty_manuscript_descriptives.py --data-root "$HARMONIZED_ROOT" --universe-config "$U" --multidomain-config "$M" --extension-config "$F" --output-dir results/descriptive
python scripts/run_elsa_behavior_hba1c_early_warning_models.py --data-root "$HARMONIZED_ROOT" --universe-config "$U" --pilot-config "$B" --assay-config "$Q" --early-config "$E" --output-dir results/biomarkers/hba1c
python - <<'PYCODE'
from pathlib import Path
import pandas as pd
r = Path('results')
files = sorted((r / 'frailty/models').glob('*-specificity-models.csv'))
assert len(files) == 6, f'expected 6 specificity files, got {len(files)}'
specificity = pd.concat([pd.read_csv(p) for p in files], ignore_index=True)
specificity.to_csv(r / 'frailty/systematic-specificity-results.csv', index=False)
source = pd.read_csv(r / 'biomarkers/hba1c/model-results.csv')
source.to_csv(r / 'biomarkers/hba1c_models.csv', index=False)
selected = source.loc[source.outcome.eq('followup_hba1c') & source.exposure_model.eq('any_contraction') & source.adjustment.eq('full') & source.term.eq('any_contraction') & source.model_status.eq('PASS')]
assert len(selected) == 1, f'expected one HbA1c contrast, got {len(selected)}'
s = selected.iloc[0]
summary = pd.DataFrame([dict(cohorts='elsa', n_person_intervals=s['n'], n_people=s.people, outcome=s.outcome, risk_set='ELSA t0/t1/t2=2/3/4,4/5/6,6/7/8; >=1 baseline domain', comparator='no alcohol/activity contraction', model=s.exposure_model, term=s.term, estimate_scale='mean_difference', estimate=s.estimate, ci_low=s.ci_low, ci_high=s.ci_high, ci_level='95%')])
summary.to_csv(r / 'biomarkers/hba1c_summary.csv', index=False)
print({'specificity_rows': len(specificity), 'hba1c_model_rows': len(source), 'hba1c_summary_rows': len(summary)})
PYCODE
