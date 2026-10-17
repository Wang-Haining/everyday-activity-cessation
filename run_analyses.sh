#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
: "${HARMONIZED_ROOT:?Set HARMONIZED_ROOT to the harmonized data directory}"

for cohort in charls elsa hrs klosa mhas share; do
  python analysis/standardization/attainable_risks.py --cohort "$cohort"
done
python analysis/standardization/pool_attainable_risks.py
for cohort in charls elsa hrs share; do
  python analysis/standardization/current_state_support.py --data-root "$HARMONIZED_ROOT" --cohort "$cohort" --universe-config config/behavioral_transition_outcome_scan_universe_v0.1.json --multidomain-config config/multidomain_behavioral_withdrawal_pilot_v0.1.json --extension-config config/behavioral_withdrawal_frailty_extension_v0.2.json --output-dir results/support
done
python - <<'PY'
from pathlib import Path
import pandas as pd
paths = sorted(Path('results/support').glob('*-support-summary.csv'))
assert len(paths) == 4, f'expected 4 support summaries, got {len(paths)}'
frame = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
frame.to_csv('results/support/current_status_summary.csv', index=False)
print(f'Current-status support: {len(frame)} cohorts')
PY

for cohort in elsa hrs klosa share; do
  python analysis/prediction/run_e1.py --cohort "$cohort"
  python analysis/prediction/run_e2.py --cohort "$cohort"
  python analysis/composition/tables_e34.py --cohort "$cohort"
  python analysis/composition/fit_e34.py --cohort "$cohort"
done
python analysis/prediction/pool_e.py
python analysis/composition/onestage_e34.py
python analysis/composition/onestage_e34.py --relax-epv
python analysis/composition/pool_e34.py

for cohort in elsa hrs klosa share; do
  python analysis/sensitivity/sample_counts.py --cohort "$cohort"
  python analysis/sensitivity/run_a_cohort.py --cohort "$cohort"
  python analysis/sensitivity/run_b.py --cohort "$cohort"
  python analysis/sensitivity/run_c.py --cohort "$cohort"
done
python analysis/sensitivity/run_a_onestage.py
python analysis/sensitivity/pool_a.py
python analysis/sensitivity/pool_b.py
python analysis/sensitivity/pool_c.py

for cohort in elsa hrs klosa share; do
  python analysis/clinical/sample_counts.py --cohort "$cohort"
  python analysis/clinical/run_e5.py --cohort "$cohort"
  python analysis/clinical/hidden_risk.py --cohort "$cohort"
done
python analysis/clinical/hidden_onestage.py
python analysis/clinical/pool_e5.py
printf 'Finished attainable risks, prediction, composition, timing, weighting, and clinical models.\n'
