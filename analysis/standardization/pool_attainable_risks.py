import sys, math
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import expit
root=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(root/'scripts'))
from postprocess_bmc_absolute_risk_work_exit_extension import _reml_hk
from postprocess_behavior_outcome_landscape import reml_hk
out=root/'results/standardization'
cohort_dir=out/'cohort'
paths=sorted(cohort_dir.glob('*-risks.csv')); assert len(paths)==6,len(paths)
risks=pd.concat([pd.read_csv(p) for p in paths],ignore_index=True)
opportunities=pd.concat([pd.read_csv(p) for p in sorted(cohort_dir.glob('*-opportunities.csv'))],ignore_index=True)
assert len(opportunities)==120,len(opportunities)
assert risks.cohort.nunique()==6,risks.cohort.unique()
pooled=[]
for (target,outcome,scenario),g in risks.loc[risks.model_status.eq('PASS')].groupby(['target','outcome_id','scenario']):
 assert not g.cohort.duplicated().any(),(target,outcome,scenario)
 assert len(g)>=2,(target,outcome,scenario,len(g))
 p=g.standardized_risk.to_numpy(); se=g.standard_error.to_numpy(); yi=np.log(p/(1-p)); vi=(se/(p*(1-p)))**2
 fit=_reml_hk(yi,vi)
 row=dict(target=target,outcome_id=outcome,scenario=scenario,k_cohorts=len(g),cohorts=';'.join(g.cohort),standardization_n=int(g.standardization_n.sum()),model_n=int(g.n.sum()),pooled_standardized_risk=float(expit(fit['pooled'])),q=fit['hk_scale'],tau2=fit['tau2'],i2=fit['i2'])
 for field in ['ci_low','ci_high','prediction_low','prediction_high']:
  row[field]=float(expit(fit[field]));row[field+'_safeguarded']=float(expit(fit[field+'_safeguarded']))
 pooled.append(row)
pooled=pd.DataFrame(pooled)
rds=[]
for (target,outcome,scenario),g in risks.loc[risks.model_status.eq('PASS') & risks.scenario.ne('withdrawal_0')].groupby(['target','outcome_id','scenario']):
 assert len(g)>=3,(target,outcome,scenario,len(g))
 fit=reml_hk(g.risk_difference_vs_0.to_numpy(),g.risk_difference_standard_error.to_numpy()**2)
 row=dict(target=target,outcome_id=outcome,scenario=scenario,k_cohorts=len(g),cohorts=';'.join(g.cohort),pooled_risk_difference=fit['pooled'],q=fit['hk_scale'],i2=fit['i2'],tau2=fit['tau2'])
 for field in ['ci_low','ci_high','prediction_low','prediction_high']:
  row[field]=fit[field];row[field+'_safeguarded']=fit[field+'_safeguarded']
 rds.append(row)
pd.DataFrame(rds).to_csv(out/'pooled_risk_differences_v0.1.csv',index=False)
print('Risk difference rows',len(rds))
opportunities['percent']=100*opportunities.n/opportunities.groupby(['cohort','population','outcome_id']).n.transform('sum')
for name,df in [('cohort_risks_v0.1.csv',risks),('pooled_risks_v0.1.csv',pooled),('opportunity_distribution_v0.1.csv',opportunities)]:df.to_csv(out/name,index=False)
print('Cohort rows',len(risks),'pooled rows',len(pooled),'opportunity rows',len(opportunities))
print(pooled.loc[pooled.outcome_id.ne('incident_any_iadl'),['target','outcome_id','scenario','pooled_standardized_risk','standardization_n']].to_string(index=False))
