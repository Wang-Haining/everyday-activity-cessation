import csv,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'tables/generated'
OUT.mkdir(parents=True,exist_ok=True)
def read(name):
 p='results/'+name
 return [dict(x,_path=p,_row=i) for i,x in enumerate(csv.DictReader((ROOT/p).open()),2)]
def one(rows,**kw):
 a=[x for x in rows if all(x.get(k)==v for k,v in kw.items())]
 assert len(a)==1,(kw,len(a))
 return a[0]
def src(r,f):return dict(path=r['_path'],row=r['_row'],field=f,raw=r[f])
def cell(text,*pairs):return dict(text=text,sources=[src(r,f) for r,f in pairs])
def integer(r,f):
 return cell(f'{float(r[f]):,.0f}',(r,f)) if r.get(f) else '—'
def counts(r,n='n_group',e='events_group'):
 return cell(f'{float(r[n]):,.0f} ({float(r[e]):,.0f})',(r,n),(r,e)) if r.get(n) and r.get(e) else '—'
def combine(*cells,sep=' / '):
 return dict(text=sep.join(c['text'] if isinstance(c,dict) else c for c in cells),sources=sum((c['sources'] for c in cells if isinstance(c,dict)),[]))
def est(r,key='rr'):
 if not r or not r.get(key):return 'Not estimable'
 return cell(f'{float(r[key]):.2f} ({float(r["ci_low"]):.2f}–{float(r["ci_high"]):.2f})',(r,key),(r,'ci_low'),(r,'ci_high'))
def summary_count(r,rows,field):
 if r.get(field):return integer(r,field)
 cohorts=r['cohorts'].split(';')
 rr=[one(rows,cohort=c,row_type='cohort',population=r['population'],landmark_months=r['landmark_months'],rule=r['rule'],term=r['term']) for c in cohorts]
 return cell(f'{sum(float(x[field]) for x in rr):,.0f}',*((x,field) for x in rr))
cohorts={'elsa':'ELSA','hrs':'HRS','klosa':'KLoSA','share':'SHARE','POOLED':'Pooled'}
abbrev='CI, confidence interval; RR, risk ratio; ELSA, English Longitudinal Study of Ageing; HRS, Health and Retirement Study; KLoSA, Korean Longitudinal Study of Ageing; SHARE, Survey of Health, Ageing and Retirement in Europe.'
T=[]
pre=read('prediction/e2_pre_window.csv');stable=read('prediction/e2_stable_subset.csv');delay=read('sensitivity/b_delayed_landmark.csv')
# S1: summary of prior health and delayed death estimates.
rows=[]
for m,label in [('srh_worse_1level','Pre-window self-rated-health worsening'),('adl_increase','Pre-window ADL increase'),('depressive_worsening_binary','Pre-window depressive worsening')]:
 r=one(pre,cohort='POOLED',measure=m,adjustment='B',gap_rule='all',term='loss_2plus')
 rows.append([label,'Two or more',', '.join(cohorts[c] for c in r['cohorts'].split(';')),integer(r,'n'),est(r,'estimate')])
for c in ['hrs','share']:
 r=one(stable,cohort=c,population='T_stable',term='loss_2plus');rows.append(['Death with stable measured prior health','Two or more',cohorts[c],integer(r,'n'),est(r)])
for mo,word in [('0.0','no'),('6.0','six-month'),('12.0','twelve-month')]:
 for term,label in [('loss_1','One'),('loss_2plus','Two or more')]:
  rt='fixed_effect_2_cohorts' if mo=='12.0' and term=='loss_2plus' else 'pooled'
  r=one(delay,row_type=rt,population='all_primary',landmark_months=mo,rule='strict',term=term)
  rows.append([f'Death: {word} delay',label,', '.join(cohorts[c] for c in r['cohorts'].split(';')),summary_count(r,delay,'n_model'),est(r)])
T.append(dict(id='S1',title='Health before the interval between the first and second interviews and death among people previously stable',columns=['Analysis','Activities stopped versus none','Contributing cohorts','Model person-intervals','RR (95% CI)'],rows=rows,footnotes=['ADL, activities of daily living; '+abbrev],methods=[]))
# S2: pair and constituent contrasts, support, and omitted-activity counts.
a=read('sensitivity/a_composition_noHRS.csv');loo=read('composition/e3_leave_one_out.csv');rows=[]
for outcome,ol in [('mortality','Death'),('incident_any_adl','New ADL limitation')]:
 for pair in ['DA','AW','DW']:
  for c,rt,cl in ([('ONE_STAGE','one_stage_all_cohorts','all cohorts'),('hrs','cohort','HRS'),('ONE_STAGE','one_stage_noHRS','outside HRS'),('share','cohort','SHARE')] if outcome=='mortality' else [('ONE_STAGE','one_stage_all_cohorts','all cohorts'),('ONE_STAGE','one_stage_noHRS','outside HRS')]):
   r=one(a,cohort=c,row_type=rt,outcome=outcome,pair=pair,term='pair_'+pair)
   contrasts=[]
   for component in pair:
    cand=[x for x in a if x['cohort']==c and x['row_type']==rt and x['outcome']==outcome and x['pair']==pair and x['term']==f'contrast_pair_{pair}_vs_single_{component}']
    contrasts.append(est(cand[0]) if cand else 'Not estimable')
   n=combine(integer(r,'n_model'),counts(r),sep='; ')
   support=cell(f'{float(r["events_per_parameter"]):.1f}',(r,'events_per_parameter')) if r.get('events_per_parameter') else '—'
   rows.append([f'{ol}: {pair[0]} + {pair[1]}, {cl}',n,est(r),*contrasts,support])
# Counts for the unsupported death-pair cells preserve the cohort support boundary.
for c in ['elsa','klosa']:
 for pair in ['DA','AW','DW']:
  r=one(a,cohort=c,row_type='cohort',outcome='mortality',pair=pair,term='pair_'+pair)
  rows.append([f'Death: {pair[0]} + {pair[1]}, {cohorts[c]}',combine(integer(r,'n_model'),counts(r),sep='; '),est(r),'Not estimable','Not estimable','Group support failed'])
for outcome,ol in [('mortality','Death'),('incident_any_adl','New ADL limitation')]:
 for omitted,label in [('alcohol','drinking'),('activity','regular physical activity'),('work','paid work')]:
  for term,tl in [('loss_1','one'),('loss_2','two')]:
   r=one(loo,row_type='one_stage_S1b_posthoc',outcome=outcome,population='same_primary_rows',omitted=omitted,term=term)
   pooled=one(loo,row_type='pooled',outcome=outcome,population='same_primary_rows',omitted=omitted,term=term)
   rows.append([f'{ol}: omit {label}, {tl} stopped',combine(integer(r,'n_model'),counts(r),sep='; '),est(r),'Not applicable','Not applicable',est(pooled) if pooled['rr'] else 'Not pooled'])
T.append(dict(id='S2',title='Pair versus constituent single activities',columns=['Outcome, pair, and population','Model; exposed person-intervals (events)','Pair or count RR (95% CI)','Pair / first constituent RR (95% CI)','Pair / second constituent RR (95% CI)','Events per parameter; omitted-count pooled RR'],rows=rows,footnotes=['D, drinking; A, regular physical activity; W, paid work; ADL, activities of daily living. '+abbrev],methods=['Pair comparisons required both activities to be present at the first interview. The pair and each single activity were compared with none stopped. Direct contrasts divide the pair risk ratio by the risk ratio for the stated constituent. Models for all cohorts and outside HRS use a common exposure coefficient with cohort-specific covariate coefficients and participant-clustered variance. These intervals omit between-cohort variation. Counts in parentheses denote deaths or new ADL limitations. No pair met the requirements for random-effects synthesis. The last column gives events per model parameter for pair rows and the safeguarded pooled risk ratio for rows omitting an activity. The drinking-plus-paid-work comparison and the work-containing comparisons outside HRS have limited model event support. Rows omitting an activity keep the original analysis population and count activities stopped among the remaining activities, with none stopped among those activities as reference.']))
# S3: stopped/started combinations, attainable-start restrictions and matched covariates.
e4=read('composition/e4_models.csv');gain=read('sensitivity/a_gain_noHRS.csv');rows=[]
for outcome,ol in [('mortality','Death'),('incident_any_adl','New ADL limitation')]:
 for term,label in [('gain_only','Started only versus neither'),('loss_only','Stopped only versus neither'),('loss_and_gain','Stopped and started versus neither'),('contrast_loss_and_gain_vs_loss_only','Stopped and started versus stopped only')]:
  r=one(e4,row_type='one_stage_S1',outcome=outcome,analysis='e4_four_group',term=term);p=one(e4,row_type='pooled',outcome=outcome,analysis='e4_four_group',term=term)
  countrow=one(e4,row_type='one_stage_S1',outcome=outcome,analysis='e4_four_group',term='loss_and_gain') if term.startswith('contrast') else r
  ref=one(e4,row_type='one_stage_S1',outcome=outcome,analysis='e4_four_group',term='loss_only') if term.startswith('contrast') else None
  rows.append([f'{ol}: {label}','All, one-stage',counts(countrow),counts(ref) if ref else counts(r,'n_reference','events_reference'),est(r),est(p) if p['rr'] else 'Not pooled'])
 for pop,label in [('stratum_all_initial_counts','One stopped, all initially engaged'),('stratum_gain_attainable_initial_counts','One stopped, able to start')]:
  r=one(e4,row_type='one_stage_S1',outcome=outcome,analysis='e4_gain_within_loss_stratum',population=pop,stratum='loss_1',term='gain');p=one(e4,row_type='pooled',outcome=outcome,analysis='e4_gain_within_loss_stratum',population=pop,stratum='loss_1',term='gain')
  rows.append([f'{ol}: {label}','All, one-stage',counts(r),counts(r,'n_reference','events_reference'),est(r),est(p) if p['rr'] else 'Not pooled'])
 r=one(e4,row_type='one_stage_S1',outcome=outcome,analysis='e4_gain_within_loss_stratum',population='stratum_gain_attainable_initial_counts',stratum='loss_2plus',term='gain')
 rows.append([f'{ol}: two or more stopped, able to start','All, one-stage',counts(r),counts(r,'n_reference','events_reference'),'Not estimable','Insufficient events'])
 for pop,label in [('attainable_lost_domain_adjusted','One stopped, adjusted for activity stopped'),('attainable_initial_pattern_x_lost_domain','One stopped, initial-pattern adjustment')]:
  groups=[('ONE_STAGE','one_stage_all_cohorts','All, one-stage'),('ONE_STAGE','one_stage_noHRS','Outside HRS, one-stage'),('share','cohort','SHARE')]
  if pop=='attainable_lost_domain_adjusted':groups += [('hrs','cohort','HRS'),('elsa','cohort','ELSA')]+([('klosa','cohort','KLoSA')] if outcome=='mortality' else [])
  for c,rt,cl in groups:
   r=one(gain,cohort=c,row_type=rt,outcome=outcome,population=pop,term='gain')
   rows.append([f'{ol}: {label}',cl,counts(r),counts(r,'n_reference','events_reference'),est(r),'Cohort estimate' if rt=='cohort' else 'One-stage estimate'])
 # Outside-HRS estimate before adjustment for which activity stopped.
 r=one(gain,cohort='ONE_STAGE',row_type='one_stage_noHRS',outcome=outcome,population='attainable_unadjusted',term='gain')
 rows.append([f'{ol}: one stopped, able to start','Outside HRS, one-stage',counts(r),counts(r,'n_reference','events_reference'),est(r),'Primary covariates'])
T.append(dict(id='S3',title='Activities stopped and started combinations',columns=['Outcome and comparison','Population and model','Comparison group person-intervals (events)','Reference group person-intervals (events)','RR (95% CI)','Pooled RR (95% CI) or model'],rows=rows,footnotes=['ADL, activities of daily living. '+abbrev],methods=['Activities started were absent at the first interview and present at the second. The four combinations compare started only, stopped only, and both stopped and started with neither. Direct comparisons of started versus none started among person-intervals with one activity stopped were repeated in participants able to start an initially absent activity. These models used the primary covariates, then indicators for which activity stopped or the combination of initial engagement pattern and activity stopped. In these rows the comparison group started an activity and the reference group did not. Both groups had one activity stopped unless stated otherwise. One-stage intervals describe within-sample uncertainty. Random-effects intervals use the Hartung–Knapp safeguard. Unavailable comparisons had insufficient group support and are not assigned a null estimate.']))
# S6: principal health measures with both adjustment specifications and counts.
rows=[]
for m,label in [('srh_worse_1level','Self-rated-health worsening'),('adl_increase','ADL increase'),('depressive_worsening_binary','Depressive worsening'),('depressive_change_continuous','Depressive change, SD'),('iadl_increase','IADL increase'),('hospitalised_prior_period','Prior-period hospitalization')]:
 for c in ['elsa','hrs','klosa','share','POOLED']:
  cand=[r for r in pre if r['cohort']==c and r['measure']==m and r['adjustment']=='A' and r['gap_rule']=='all' and r['term'] in ['loss_2plus','all']]
  if not cand:continue
  ra=cand[0]; rb=one(pre,cohort=c,measure=m,adjustment='B',gap_rule='all',term=ra['term'])
  n=combine(integer(ra,'n'),integer(rb,'n')) if ra['n']!='0' else 'Not available'
  rows.append([label,cohorts[c],'Two or more',n,est(ra,'estimate'),est(rb,'estimate')])
  if c=='POOLED':
   ca=[r for r in pre if r['cohort']==c and r['measure']==m and r['adjustment']=='A' and r['gap_rule']=='all' and r['term']=='loss_1']
   if ca:
    ra=ca[0];rb=one(pre,cohort=c,measure=m,adjustment='B',gap_rule='all',term='loss_1')
    rows.append([label,'Pooled','One',combine(integer(ra,'n'),integer(rb,'n')),est(ra,'estimate'),est(rb,'estimate')])
T.append(dict(id='S6',title='Pre-window health changes under both adjustment specifications',columns=['Pre-window health measure','Cohort','Stopped versus none','Person-intervals, A / B','A: estimate (95% CI)','Fully adjusted B: estimate (95% CI)'],rows=rows,footnotes=['A, lighter adjustment; B, full adjustment; ADL, activities of daily living; IADL, instrumental activities of daily living; SD, standard deviation. '+abbrev],methods=['Pre-window changes occurred between an earlier interview and the first interview. Specification A included age, sex, interview wave, education, income, and the preceding level of the health measure. Fully adjusted specification B also included the initial number of activities, smoking, and diagnosed-condition count. The table reports risk ratios for binary measures and adjusted mean differences in cohort-specific standard-deviation units for depressive change. Both estimates and counts refer to the stated specification. Pooled rows include only evaluable cohorts and use safeguarded Hartung–Knapp intervals. Hospitalization was not pooled because recall periods differ. IADL increase had fewer than three evaluable cohorts.']))
# S7: stable eligibility, delayed estimates, and death-date completeness in one coherent table.
rows=[]
for pop,label in [('T_all','Valid preceding-health measurements'),('T_stable','Stable measured prior health'),('T_not_stable','Other measured prior health')]:
 for c in ['elsa','hrs','klosa','share']:
  cand=[r for r in stable if r['cohort']==c and r['population']==pop]
  ra=next((r for r in cand if r['term']=='loss_1'),cand[0]);rb=next((r for r in cand if r['term']=='loss_2plus'),cand[0])
  detail='Insufficient events per parameter' if 'EPV' in ra['status'] else ('Insufficient multiple-stopped deaths' if not rb['rr'] else 'Full adjustment')
  rows.append([label,cohorts[c],counts(ra,'n','deaths'),est(ra),est(rb),detail])
for pop,label in [('all_primary','All primary'),('stable_prior_health','Stable measured prior health')]:
 for mo,ml in [('0.0','no delay'),('6.0','six-month delay'),('12.0','twelve-month delay')]:
  for c in ['hrs','klosa','share']:
   ra=one(delay,cohort=c,row_type='cohort',population=pop,landmark_months=mo,rule='strict',term='loss_1');rb=one(delay,cohort=c,row_type='cohort',population=pop,landmark_months=mo,rule='strict',term='loss_2plus')
   support=combine(integer(rb,'events_group'),'multiple-stopped deaths',sep=' ') if not rb['rr'] else 'Dated deaths only'
   rows.append([f'{label}, {ml}',cohorts[c],counts(ra,'n_model','events_model'),est(ra),est(rb),support])
  ra=one(delay,cohort='POOLED',row_type='pooled',population=pop,landmark_months=mo,rule='strict',term='loss_1')
  typ='fixed_effect_2_cohorts' if (pop=='stable_prior_health' and mo!='12.0') or (pop=='all_primary' and mo=='12.0') else 'pooled'
  rb=one(delay,cohort='POOLED',row_type=typ,population=pop,landmark_months=mo,rule='strict',term='loss_2plus')
  rows.append([f'{label}, {ml}','Summary','See cohort rows',est(ra),est(rb),'Two or more: fixed effect' if typ=='fixed_effect_2_cohorts' else ('Two or more: HRS only' if pop=='stable_prior_health' and mo=='12.0' else 'Random effects')])
# Same-cohort no-delay comparator for the two-cohort delayed estimate.
ra=one(delay,row_type='fixed_effect_HRS_SHARE_comparator',population='all_primary',landmark_months='0.0',rule='strict',term='loss_1');rb=one(delay,row_type='fixed_effect_HRS_SHARE_comparator',population='all_primary',landmark_months='0.0',rule='strict',term='loss_2plus')
rows.append(['All primary, no delay','HRS and SHARE','See cohort rows',est(ra),est(rb),'Fixed effect'])
for c in ['elsa','hrs','klosa','share']:
 total=one(delay,cohort=c,row_type='dating_coverage',item='deaths_in_window');dated=one(delay,cohort=c,row_type='dating_coverage',item='deaths_year_and_month')
 pct=cell(f'{100*float(dated["value"])/float(total["value"]):.1f}%',(dated,'value'),(total,'value'))
 rows.append(['Death-date completeness',cohorts[c],combine(integer(dated,'value'),integer(total,'value'),sep=' / '),'Not applicable','Not applicable',combine(pct,'month-resolved',sep=' ')])
T.append(dict(id='S7',title='Stable-prior-health and delayed death estimates',columns=['Population and timing','Cohort','Model person-intervals (deaths) or dated / total deaths','One stopped RR (95% CI)','Two or more stopped RR (95% CI)','Support or method'],rows=rows,footnotes=[abbrev+' All risk ratios use none stopped as reference.'],methods=['Stable measured prior health required stable or improved self-rated health and no ADL limitation at the first interview, with both measures observed at the earlier and first interviews. All death models used the main covariate specification. Delayed analyses excluded undated deaths and conditioned on survival beyond the stated time after the second interview. No delay uses the same dated-death population. Events on or before the specified month were excluded. Unsupported groups were removed from the fitted model. The model counts therefore can differ from all eligible person-intervals. Summary estimates use random-effects pooling with safeguarded Hartung–Knapp intervals when three cohorts contribute. The two-cohort estimates use fixed-effect pooling, as labeled. ELSA had no month-resolved death dates. The completeness rows show dated deaths divided by total deaths in the primary risk set.']))
for t in T:
 assert all(len(r)==len(t['columns']) for r in t['rows']),t['id']
 for row in t['rows']:
  for c in row:
   if isinstance(c,str):assert not re.search(r'\d',c),(t['id'],c)
   else:
    assert c['sources'],(t['id'],c)
    for s in c['sources']:assert s['raw']!='',(t['id'],s)
OUT.joinpath('tables_2_3_6_7.json').write_text(json.dumps(T,ensure_ascii=False,indent=2))
print('Built',', '.join(f'{t["id"]}: {len(t["rows"])} rows' for t in T))
print('Numeric cells:',sum(isinstance(c,dict) for t in T for r in t['rows'] for c in r))
