import sys, json, math, os
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
from multidomain_behavioral_withdrawal_core import load_specs, load_episodes, multidomain_frame, prepare_model, support_gate
from probe_behavior_outcome_feasibility import load_lookup, sha256
from run_bmc_absolute_risk_work_exit_extension import _fit_probability_model, _scenario_design, _gradient
parser = argparse.ArgumentParser(description="Standardized risks in attainable populations")
parser.add_argument('--cohort', required=True)
parser.add_argument('--data-root', type=Path, default=os.environ.get('HARMONIZED_ROOT'))
args = parser.parse_args()
if args.data_root is None or not args.data_root.is_dir():
 raise ValueError(f"expected a harmonized data directory, got {args.data_root}")
root=ROOT; cohort=args.cohort; meta=root/'config'
u,p=load_specs(meta/'behavioral_transition_outcome_scan_universe_v0.1.json',meta/'multidomain_behavioral_withdrawal_pilot_v0.1.json')
e=json.loads((meta/'bmc_absolute_risk_work_exit_extension_v0.2.json').read_text())
data_root=args.data_root
lookup,_=load_lookup(data_root,u)
episodes,_,_=load_episodes(data_root,u,cohort,lookup)
behavior=multidomain_frame(episodes,u,p,cohort)
risks=[]; opportunities=[]
valid=behavior.core_valid & episodes.comparable_window.fillna(False)
for k in range(4):
 ix=valid & behavior.baseline_engagement_count.eq(k)
 opportunities.append(dict(cohort=cohort,population='core_valid_comparable_before_complete_cases',outcome_id='all',initial_count=k,n=int(ix.sum()),people=int(episodes.loc[ix,'person_id'].nunique())))
for outcome in e['risk_outcomes']:
 data,y,X,terms,binary,coding=prepare_model(episodes,behavior,u,p,cohort,outcome,e['scope'],e['adjustment'],e['risk_exposure_model'])
 assert binary,(cohort,outcome,'expected binary outcome')
 status,counts=support_gate(data,y,X,terms,p,binary)
 base=dict(cohort=cohort,outcome_id=outcome,scope=e['scope'],adjustment=e['adjustment'],**counts)
 for k in range(4):
  ix=data.baseline_engagement_count.eq(k)
  opportunities.append(dict(cohort=cohort,population='outcome_complete_cases',outcome_id=outcome,initial_count=k,n=int(ix.sum()),people=int(data.loc[ix,'person_id'].nunique())))
 assert sum(r['n'] for r in opportunities if r['outcome_id']==outcome)==len(data),outcome
 if status!='ESTIMABLE':
  for target in ['original_all','common_k_ge_2','category_attainable']:
   risks.append(dict(**base,target=target,scenario='__model__',model_status=status))
  continue
 fit,warnings=_fit_probability_model(y,X,data.person_id,e['risk_probability_model'])
 assert fit.converged,(cohort,outcome,'logistic model did not converge')
 base['warnings']=warnings
 cov=fit.cov_params().loc[X.columns,X.columns].to_numpy(float)
 assert np.isfinite(cov).all(),(cohort,outcome,'nonfinite covariance')
 for target in ['original_all','common_k_ge_2','category_attainable']:
  scenario_results={}
  for scenario,spec in e['risk_scenarios'].items():
   threshold=2 if target=='common_k_ge_2' or (target=='category_attainable' and scenario=='withdrawal_2plus') else 1
   ix=data.baseline_engagement_count.ge(threshold)
   assert ix.any(),(cohort,outcome,target,'empty target')
   xs=_scenario_design(X.loc[ix],spec)
   pred=np.asarray(fit.predict(xs),float)
   assert np.isfinite(pred).all() and ((pred>=0)&(pred<=1)).all(),(cohort,outcome,target,scenario)
   g=_gradient(xs,pred); var=float(g@cov@g)
   assert var>=0,(cohort,outcome,target,scenario,var)
   risk=float(pred.mean()); se=math.sqrt(var)
   observed=ix & (data.loss_1.eq(1) if scenario=='withdrawal_1' else data.loss_2plus.eq(1) if scenario=='withdrawal_2plus' else data.loss_1.eq(0)&data.loss_2plus.eq(0))
   row=dict(**base,target=target,scenario=scenario,model_status='PASS',standardization_n=int(ix.sum()),standardization_people=int(data.loc[ix,'person_id'].nunique()),standardized_risk=risk,standard_error=se,ci_low=max(0,risk - 1.959963984540054*se),ci_high=min(1,risk+1.959963984540054*se),observed_n=int(observed.sum()),observed_events=int(y.loc[observed].sum()))
   scenario_results[scenario]=(row,g,var); risks.append(row)
  ref,refg,refvar=scenario_results['withdrawal_0']
  for scenario,(row,g,var) in scenario_results.items():
   if scenario=='withdrawal_0': continue
   rd=row['standardized_risk']-ref['standardized_risk']; rdvar=var+refvar-2*float(g@cov@refg)
   assert rdvar>=-1e-12,(cohort,outcome,target,rdvar)
   se=math.sqrt(max(0,rdvar))
   row.update(risk_difference_vs_0=rd,risk_difference_standard_error=se,risk_difference_ci_low=rd-1.959963984540054*se,risk_difference_ci_high=rd+1.959963984540054*se)
out=root/'results/standardization/cohort'; out.mkdir(parents=True,exist_ok=True)
pd.DataFrame(risks).to_csv(out/f'{cohort}-risks.csv',index=False)
pd.DataFrame(opportunities).to_csv(out/f'{cohort}-opportunities.csv',index=False)
print(cohort,'risk rows',len(risks),'opportunity rows',len(opportunities),'statuses',pd.Series([r['model_status'] for r in risks]).value_counts().to_dict())
