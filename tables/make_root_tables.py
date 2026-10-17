import csv,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'tables/generated'
OUT.mkdir(parents=True,exist_ok=True)
titles={'S1': 'Health before the interval between the first and second interviews and death among people previously stable', 'S2': 'Pair versus constituent single activities', 'S3': 'Activities stopped and started combinations', 'S4': 'Cohort eligibility, fitted and descriptive samples, and missingness', 'S5': 'Full categorical estimates and standardized risks', 'S6': 'Pre-window health changes under both adjustment specifications', 'S7': 'Stable-prior-health and delayed death estimates', 'S8': 'Four-state engagement histories and component estimates', 'S9': 'Inclusion weighting and sensitivity analyses', 'S10': 'Discrimination of the number of activities stopped added to brief clinical information', 'S11': 'Current-status-held and concurrent-health models', 'S12': 'Biomarker and repeated-assay findings', 'S13': 'Disease-specific outcomes', 'S14': 'Paid-work exit context and word-recall results', 'S15': 'Hartung–Knapp uncertainty estimates'}
def read(path):
 return [dict(r,_path=path,_row=i) for i,r in enumerate(csv.DictReader((ROOT/path).open()),2)]
def src(r,f): return dict(path=r['_path'],row=r['_row'],field=f,raw=r[f])
def cell(r,f,fmt=',.0f',scale=1):
 assert r[f] not in ('','nan'),(r['_path'],r['_row'],f)
 return dict(text=format(float(r[f])*scale,fmt),sources=[src(r,f)])
def ci(r,point='pooled',lo='ci_low_safeguarded',hi='ci_high_safeguarded',scale=1,dec=2):
 return dict(text=f'{float(r[point])*scale:.{dec}f} ({float(r[lo])*scale:.2f}–{float(r[hi])*scale:.2f})',sources=[src(r,f) for f in [point,lo,hi]])
def interval(r,a,b):
 if not r[a] or not r[b]:return 'Not available'
 return dict(text=f'{float(r[a]):.2f}–{float(r[b]):.2f}',sources=[src(r,a),src(r,b)])
def one(rows,**key):
 got=[r for r in rows if all(r[k]==v for k,v in key.items())];assert len(got)==1,(key,len(got));return got[0]
hk=read('results/primary/hk_pooled.csv')
for r in hk:r['_key']=json.loads(r['key'])
primary=read('results/primary/systematic-results-matrix.csv')
select=read('results/sensitivity/c_selection_weights.csv')
cohorts=['charls','elsa','hrs','klosa','mhas','share']
names=dict(zip(cohorts,['CHARLS','ELSA','HRS','KLoSA','MHAS','SHARE']))
ex={c:read(f'results/descriptive/{c}-exclusion-reasons.csv')[0] for c in cohorts}
rows=[]
for label,f in [('Three consecutive interviews, age eligible','intervals_entering'),('Comparable follow-up window','in_comparable_window'),('Activities incompletely observed','behaviour_incomplete'),('No initially present activity','no_behaviour_available_to_lose'),('Descriptive person-intervals','primary_risk_set'),('Descriptive participants','primary_risk_set_people')]:
 rows.append([label]+[('No comparable interval' if c=='mhas' and f!='intervals_entering' else cell(ex[c],f)) for c in cohorts])
for label,out,f in [('Death: fitted person-intervals','mortality','n'),('Death: fitted participants','mortality','people'),('Death: events','mortality','events'),('New ADL limitation: fitted person-intervals','incident_any_adl','n'),('New ADL limitation: fitted participants','incident_any_adl','people'),('New ADL limitation: events','incident_any_adl','events')]:
 vals=[]
 for c in cohorts:
  rr=[r for r in primary if r['cohort']==c and r['outcome_id']==out and r['adjustment']=='full' and r['scope']=='comparable_22_30_months' and r['exposure_model']=='score_categorical']
  if c=='mhas':v='No comparable interval'
  elif not any(r['model_status']=='PASS' for r in rr):v='Not supported'
  else:v=cell(next(r for r in rr if r['model_status']=='PASS'),f)
  vals.append(v)
 rows.append([label]+vals)
for out,label in [('mortality','Death'),('incident_any_adl','New ADL limitation')]:
 for f,desc in [('n_eligible','eligible for inclusion model'),('n_excluded_outcome_missing','outcome unavailable'),('n_excluded_covariates_only','covariates incomplete only')]:
  vals=[]
  for c in cohorts:
   rr=[r for r in select if r['cohort']==c and r['outcome']==out and r['row_type']=='flow' and r['item']=='overall' and r['group']=='all']
   vals.append(cell(rr[0],f) if rr else ('No comparable interval' if c=='mhas' else 'Not fitted'))
  rows.append([f'{label}: {desc}']+vals)
S4=dict(id='S4',title=titles['S4'],columns=['Sample or exclusion']+[names[c] for c in cohorts],rows=rows,footnotes=['Counts are person-intervals unless labeled participants. Unsupported categorical models do not contribute to pooled estimates. ADL, activities of daily living; CHARLS, China Health and Retirement Longitudinal Study; ELSA, English Longitudinal Study of Ageing; HRS, Health and Retirement Study; KLoSA, Korean Longitudinal Study of Ageing; MHAS, Mexican Health and Aging Study; SHARE, Survey of Health, Ageing and Retirement in Europe.'],methods=[])
# Pooled categorical estimates cover every primary outcome, with attainable risks where available.
risks=read('results/standardization/pooled_risks_v0.1.csv');cr=read('results/standardization/cohort_risks_v0.1.csv')
outs={'mortality':'Death','incident_any_adl':'New ADL limitation','multimorbidity_progression':'Multimorbidity progression','incident_any_iadl':'New IADL limitation','iadl_count_increase':'IADL count increase','word_recall_change':'Word-recall change, SD'}
rows=[]
for out,label in outs.items():
 for category,term in [('None',None),('One','loss_1'),('Two or more','loss_2plus')]:
  match=[r for r in hk if r['family']=='primary' and r['_key'].get('exposure_model')=='score_categorical' and r['_key'].get('outcome_id')==out and r['_key'].get('term')==(term or 'loss_1')]
  assert len(match)==1;h=match[0]
  vals=[label,category,cell(h,'n'),'Reference' if not term else ci(h)]
  for target in ['common_k_ge_2','category_attainable']:
   rr=[r for r in risks if r['outcome_id']==out and r['target']==target and r['scenario']=='withdrawal_'+({'None':'0','One':'1','Two or more':'2plus'}[category])]
   if rr:
    r=rr[0];v=ci(r,'pooled_standardized_risk',scale=100,dec=1);v['text']+=f"; n={int(float(r['standardization_n'])):,}";v['sources'].append(src(r,'standardization_n'));vals.append(v)
   else:vals.append('Not estimated')
  rows.append(vals)
# Explicit cohort exception: categorical risks are attainable in both target populations.
for category,term,scenario in [('None',None,'withdrawal_0'),('One','loss_1','withdrawal_1'),('Two or more','loss_2plus','withdrawal_2plus')]:
 p=one(primary,cohort='klosa',outcome_id='multimorbidity_progression',adjustment='full',scope='comparable_22_30_months',exposure_model='score_categorical',term=term or 'loss_1')
 vals=['Multimorbidity progression, KLoSA',category,cell(p,'n'),'Reference' if term is None else ci(p,'estimate','ci_low','ci_high')]
 for target in ['common_k_ge_2','category_attainable']:
  r=one(cr,cohort='klosa',outcome_id='multimorbidity_progression',target=target,scenario=scenario)
  v=ci(r,'standardized_risk','ci_low','ci_high',100,1);v['text']+=f"; n={int(float(r['standardization_n'])):,}";v['sources'].append(src(r,'standardization_n'));vals.append(v)
 rows.append(vals)
S5=dict(id='S5',title=titles['S5'],columns=['Outcome / population','Activities stopped','Model n','Estimate (95% CI)','Common-population risk, % (95% CI); target n','Category-specific risk, % (95% CI); target n'],rows=rows,footnotes=['Estimates are risk ratios except word recall (standardized mean differences). Pooled confidence intervals use safeguarded Hartung–Knapp uncertainty. Common populations initially engaged in at least two activities; category-specific populations differ across categories. ADL, activities of daily living; IADL, instrumental activities of daily living; CI, confidence interval; SD, standard deviation; n, number; KLoSA, Korean Longitudinal Study of Ageing.'],methods=[])
# Method comparison, restricted to interpretable major primary outcomes and the matched contextual comparison.
rows=[]
for out,label in outs.items():
 for term,cat in [('loss_1','One'),('loss_2plus','Two or more')]:
  h=next(r for r in hk if r['family']=='primary' and r['_key'].get('exposure_model')=='score_categorical' and r['_key'].get('outcome_id')==out and r['_key'].get('term')==term)
  rows.append([label,cat,cell(h,'k'),cell(h,'pooled','.2f'),interval(h,'ci_low','ci_high'),interval(h,'ci_low_safeguarded','ci_high_safeguarded'),interval(h,'prediction_low_safeguarded','prediction_high_safeguarded')])
for state,label in [('unconditional','Death, matched unconditional'),('current_state_held','Death, current status held')]:
 for term,cat in [('loss_1','One'),('loss_2plus','Two or more')]:
  h=next(r for r in hk if r['family']=='context-synthesis' and r['_key'].get('context_id')=='incident_non_target_disease' and r['_key'].get('model_id')=='base_same_set_incident_non_target_disease' and r['_key'].get('outcome_id')=='mortality' and r['_key'].get('state_adjustment')==state and r['_key'].get('term')==term)
  rows.append([label,cat,cell(h,'k'),cell(h,'pooled','.2f'),interval(h,'ci_low','ci_high'),interval(h,'ci_low_safeguarded','ci_high_safeguarded'),interval(h,'prediction_low_safeguarded','prediction_high_safeguarded')])
S15=dict(id='S15',title=titles['S15'],columns=['Outcome / model','Activities stopped','Cohorts, n','Estimate','Unmodified HK 95% CI','Safeguarded HK 95% CI','Safeguarded 95% PI'],rows=rows,footnotes=['None stopped is the reference. Estimates are risk ratios except word recall (standardized mean differences). Safeguarding preserves point estimates and limits the variance scale to at least one. ADL, activities of daily living; IADL, instrumental activities of daily living; SD, standard deviation; HK, Hartung–Knapp; CI, confidence interval; PI, prediction interval; n, number.'],methods=[])
(OUT/'tables_4_5_15.json').write_text(json.dumps([S4,S5,S15],indent=2,ensure_ascii=False))
print('Root tables:',[(x['id'],len(x['rows'])) for x in [S4,S5,S15]])
