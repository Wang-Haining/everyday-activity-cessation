#!/usr/bin/env python3
"""Pool cohort estimates with REML and safeguarded Hartung-Knapp intervals."""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import expit
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
from postprocess_behavior_outcome_landscape import reml_hk
from postprocess_bmc_absolute_risk_work_exit_extension import _reml_hk
OUT = ROOT / 'results/primary'
OUT.mkdir(parents=True, exist_ok=True)
rows = []
def read(pattern):
    paths = sorted(ROOT.glob(pattern))
    assert paths, f'no aggregate inputs: {pattern}'
    frames = []
    for p in paths:
        d = pd.read_csv(p, low_memory=False)
        assert not {'person_id','respondent_id','prediction','residual'} & set(d), str(p)
        d['source_row'] = [f'{p.relative_to(ROOT)}:{i+2}' for i in range(len(d))]
        frames.append(d)
    return pd.concat(frames, ignore_index=True)
def select(d, **values):
    keep = pd.Series(True, index=d.index)
    for k,v in values.items():
        keep &= d[k].isna() if pd.isna(v) else d[k].eq(v)
    return d.loc[keep].copy()
def record(g, family, key, scale='rr', ycol='estimate', secol='standard_error', se_ci=False):
    g = select(g, model_status='PASS').drop_duplicates('cohort').sort_values('cohort')
    assert len(g)>=2 and g.cohort.nunique()==len(g), (family,key,len(g))
    x, se = g[ycol].to_numpy(float), g[secol].to_numpy(float)
    if se_ci:
        se = (np.log(g.ci_high)-np.log(g.ci_low)).to_numpy(float)/(2*1.959963985)
    trans = np.exp if scale=='rr' else expit if scale=='risk' else lambda z: z
    y = np.log(x) if scale=='rr' else np.log(x/(1-x)) if scale=='risk' else x
    if scale=='risk':
        se = se/(x*(1-x))
    assert np.isfinite(y).all() and np.isfinite(se).all() and (se>0).all(), (family,key)
    fit = (_reml_hk if len(g)==2 or scale=='risk' else reml_hk)(y,se**2)
    fields = ['ci_low','ci_high','prediction_low','prediction_high']
    result = dict(family=family,key=json.dumps({k:(v.item() if isinstance(v,np.generic) else v) for k,v in key.items()},sort_keys=True),scale=scale,k=len(g),cohorts=';'.join(g.cohort),n=float(g.n.sum()) if 'n' in g else np.nan,source_rows=';'.join(g.source_row),pooled=float(trans(fit['pooled'])),q=fit['hk_scale'],tau2=fit['tau2'])
    for field in fields:
        result[field] = float(trans(fit[field]))
        result[field+'_safeguarded'] = float(trans(fit[field+'_safeguarded']))
    assert result['ci_low_safeguarded'] <= result['ci_low']+1e-12 and result['ci_high_safeguarded'] >= result['ci_high']-1e-12, (family,key)
    rows.append(result)
def pool_groups(data, family, keys, scale='rr', ycol='estimate', secol='standard_error', minimum=3):
    passing=select(data, model_status='PASS')
    for values, group in passing.groupby(keys, dropna=False, sort=True):
        values=values if isinstance(values,tuple) else (values,)
        if group.cohort.nunique() >= minimum:
            key=dict(zip(keys,values))
            if family=='primary' and key['estimate_scale']!='risk_ratio':
                record(group,family,key,'smd','standardized_estimate','standardized_standard_error')
            else:
                record(group,family,key,scale,ycol,secol)
primary=select(read('results/primary/systematic-results-matrix.csv'),scope='comparable_22_30_months',adjustment='full',analysis_family='core')
pool_groups(primary,'primary',['outcome_id','exposure_model','term','estimate_scale'])
for source,family,keys in [('context-graded-models','context-synthesis',['outcome_id','context_id','model_id','state_adjustment','term']),('composite-models','composite-synthesis',['outcome_id','exposure_model','model_id','state_adjustment','term']),('reference-group-models','reference-group-synthesis',['outcome_id','domain','model_id','term'])]:
    cohort_rows=read('results/context/models/*-'+source+'.csv')
    if source=='context-graded-models':
        cohort_rows=pd.concat([cohort_rows,read('results/context/frailty/*-context-graded-models.csv')],ignore_index=True)
    pool_groups(cohort_rows,family,keys)
specificity=read('results/frailty/systematic-specificity-results.csv')
frailty_models=read('results/frailty/systematic-frailty-results.csv')
frailty=pd.concat([specificity[specificity.term.ne('__model__')],frailty_models[frailty_models.term.isin(['any_withdrawal','loss_1','loss_2plus','prefrail','frail','fried4','fi_per_0_1'])]])
pool_groups(frailty,'frailty_extension',['analysis_family','model_id','outcome_id','term'])
risks=select(read('results/standardization/cohort_risks_v0.1.csv'),target='original_all')
pool_groups(risks,'standardized_risks',['outcome_id','scenario'],'risk','standardized_risk',minimum=2)
pool_groups(read('results/standardization/cohort_work_exit_contrasts.csv'),'work_exit',['outcome_id'],'rr','estimate','standard_error_log_scale',minimum=2)
for (outcome,scenario),g in select(risks,model_status='PASS').groupby(['outcome_id','scenario']):
    if scenario in ['withdrawal_1','withdrawal_2plus'] and g.cohort.nunique()>=3:
        record(g,'risk_difference',dict(outcome_id=outcome,scenario=scenario),'difference','risk_difference_vs_0','risk_difference_standard_error')
for source,family,keys in [('results/age_floor/*-age-floor-models.csv','age_floor',['outcome_id','term','age_floor']),('results/missingness/*-missingness-models.csv','missingness',['outcome_id','term','variant'])]:
    for key,g in select(read(source),model_status='PASS').groupby(keys):
        if g.cohort.nunique()>=3 and key[1] in ['loss_1','loss_2plus']:
            record(g,family,dict(zip(keys,key)))
opportunity=select(read('results/opportunity2/*-models.csv'),scope='comparable_22_30_months',adjustment='full',exposure_model='score_categorical',minimum_baseline_opportunities=2,model_status='PASS')
for key,g in opportunity.groupby(['outcome_id','term']):
    if g.cohort.nunique()>=3 and key[0] in ['mortality','incident_any_adl'] and key[1] in ['loss_1','loss_2plus']:
        record(g,'opportunity2',dict(zip(['outcome_id','term'],key)))
        record(g,'opportunity2_ci_reconstructed',dict(zip(['outcome_id','term'],key)),se_ci=True)
for key,g in select(read('results/component_equality/*-component-equality.csv'),model_status='PASS').groupby(['outcome_id','contrast']):
    if g.cohort.nunique()>=3:
        g=g.copy();g['ratio']=np.exp(g.log_difference)
        record(g,'component_equality',dict(zip(['outcome_id','contrast'],key)),'rr','ratio')
for (outcome,term),g in select(primary,exposure_model='score_categorical',model_status='PASS').groupby(['outcome_id','term']):
    if outcome not in ['mortality','incident_any_adl','multimorbidity_progression'] or term not in ['loss_1','loss_2plus']:
        continue
    for omit in sorted(g.cohort):
        kept=g[g.cohort.ne(omit)]
        if kept.cohort.nunique()>=3:
            record(kept,'leave_one_out',dict(outcome_id=outcome,term=term,omitted=omit))
# Crude competing-outcome estimates use the cohort aggregate death rates.
def cells(r,term):
    return (r[term+'_n'],r[term+'_events'],r.n-r.loss_1_n-r.loss_2plus_n,r.events-r.loss_1_events-r.loss_2plus_events)
countdata=select(primary,exposure_model='score_categorical',model_status='PASS').drop_duplicates(['outcome_id','cohort'])
deaths=select(countdata,outcome_id='mortality').set_index('cohort')
for outcome in ['incident_any_adl','multimorbidity_progression']:
    current=select(countdata,outcome_id=outcome)
    for term in ['loss_1','loss_2plus']:
        for treatment in ['survivors_only','deaths_retained','death_or_outcome']:
            inputs=[]
            for _,r in current.iterrows():
                if r.cohort not in deaths.index:
                    continue
                dr=deaths.loc[r.cohort]
                ng,eg,n0,e0=cells(r,term);dng,deg,dn0,de0=cells(dr,term)
                Ng,N0=ng/(1-deg/dng),n0/(1-de0/dn0)
                if treatment=='death_or_outcome':
                    eg,e0=eg+Ng-ng,e0+N0-n0
                if treatment!='survivors_only':
                    ng,n0=Ng,N0
                if min(eg,e0)==0:
                    eg,e0,ng,n0=eg+.5,e0+.5,ng+1,n0+1
                inputs.append(dict(cohort=r.cohort,estimate=(eg/ng)/(e0/n0),standard_error=np.sqrt(1/eg-1/ng+1/e0-1/n0),model_status='PASS',source_row=r.source_row+';'+dr.source_row))
            record(pd.DataFrame(inputs),'crude_competing_recount',dict(outcome_id=outcome,term=term,treatment=treatment))
d=pd.DataFrame(rows)
assert len(d)>200, len(d)
d.to_csv(OUT/'hk_pooled.csv',index=False)
print(json.dumps(dict(status='PASS',pooled_rows=len(d),rows_widened=int(d.q.lt(1).sum()),families=d.family.value_counts().to_dict())))
