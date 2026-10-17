import csv,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'tables/generated'
OUT.mkdir(parents=True,exist_ok=True)
def read(path):
    return [dict(r,_path=path,_row=i) for i,r in enumerate(csv.DictReader((ROOT/path).open()),2)]
def src(r,f):
    assert r[f]!='', (r['_path'],r['_row'],f)
    return {'path':r['_path'],'row':r['_row'],'field':f,'raw':r[f]}
def cell(text,r,*fields):
    return {'text':text,'sources':[src(r,f) for f in fields]}
def num(r,f):return cell(f"{float(r[f]):,.0f}",r,f)
def val(r,f,d=2,m=1):return cell(f"{float(r[f])*m:.{d}f}",r,f)
def ci(r,point='pooled',lo='ci_low_safeguarded',hi='ci_high_safeguarded',m=1):
    return cell(f"{float(r[point])*m:.2f} ({float(r[lo])*m:.2f}–{float(r[hi])*m:.2f})",r,point,lo,hi)
def pi(r):return cell(f"{float(r['prediction_low_safeguarded']):.2f}–{float(r['prediction_high_safeguarded']):.2f}",r,'prediction_low_safeguarded','prediction_high_safeguarded')
def joincells(cs,sep=' / '):
    return {'text':sep.join(c['text'] for c in cs),'sources':[s for c in cs for s in c['sources']]}
HK=read('results/primary/hk_pooled.csv')
for r in HK:r['_key']=json.loads(r['key'])
def hk(fam,**keys):
    rows=[r for r in HK if r['family']==fam and all(r['_key'].get(k)==v for k,v in keys.items())]
    assert len(rows)==1,(fam,keys,len(rows))
    return rows[0]
coh={'elsa':'ELSA','hrs':'HRS','klosa':'KLoSA','share':'SHARE','charls':'CHARLS','mhas':'MHAS'}
def cohorts(r,f='cohorts'):return ', '.join(coh[x] for x in r[f].split(';'))
outcome={'mortality':'Death','incident_any_adl':'New ADL limitation','incident_any_iadl':'New IADL limitation','multimorbidity_progression':'Multimorbidity progression'}
dom={'activity':'Regular physical activity','alcohol':'Drinking','work':'Paid work'}
notes='CI, confidence interval; RR, risk ratio; ADL, activities of daily living; ELSA, English Longitudinal Study of Ageing; HRS, Health and Retirement Study; KLoSA, Korean Longitudinal Study of Ageing; SHARE, Survey of Health, Ageing and Retirement in Europe.'
# S8: every four-state or restricted row for death and ADL, with the full reference group explicit.
ref=read('results/context/final/reference-group-synthesis.csv')
rows=[]
for o in ['mortality','incident_any_adl']:
    for d in ['activity','alcohol','work']:
        for t in ['continued','started','stopped']:
            orig=next(r for r in ref if r['outcome_id']==o and r['domain']==d and r['term']==d+'_'+t and r['model_id']=='four_state_never_had_reference')
            if orig['pooled_estimate']:
                r=hk('reference-group-synthesis',outcome_id=o,domain=d,term=d+'_'+t,model_id='four_state_never_had_reference')
                est=ci(r);n=num(r,'n');support=cohorts(r)
            else:
                cs=[]
                for co in orig['cohorts'].split(';'):
                    path=f'results/context/models/{co}-reference-group-models.csv'
                    rr=[x for x in read(path) if x['outcome_id']==o and x['domain']==d and x['model_id']=='four_state_never_had_reference' and x['term']==d+'_'+t]
                    assert len(rr)==1,(path,o,d,t)
                    c=ci(rr[0],'estimate','ci_low','ci_high');c['text']=coh[co]+': '+c['text'];cs.append(c)
                est=joincells(cs,'; ');n=num(orig,'n_total');support=cohorts(orig)+'; not pooled'
            rows.append([outcome[o],dom[d],t.title()+' vs absent at both',support,n,est])
        r=hk('reference-group-synthesis',outcome_id=o,domain=d,term=d+'_loss',model_id='restricted_stopped_vs_continued')
        rows.append([outcome[o],dom[d],'Stopped vs continued',cohorts(r),num(r,'n'),ci(r)])
        r=hk('primary',outcome_id=o,exposure_model='mutually_adjusted_components',term=d+'_loss')
        rows.append([outcome[o],dom[d],'Mutually adjusted component',cohorts(r),num(r,'n'),ci(r)])
s8={'id':'S8','title':'Four-state engagement histories and component estimates','columns':['Outcome','Everyday activity','Comparison','Cohorts','Person-intervals','RR (95% CI)'],'rows':rows,'footnotes':[notes+' Absent at both interviews is the four-state reference. Component estimates compare stopping with no stopping.'],'methods':['Four-state engagement histories distinguish continued participation, stopping, starting, and absence at both interviews. The reference category was absence at both interviews. Separate stopped-versus-continued models included only participants engaged at the first interview. The component models included stopping drinking, regular physical activity, and paid work together, with current status included for each activity. Pooled estimates required at least three evaluable cohorts. Drinking and paid-work four-state death models had only two evaluable cohorts, so cohort estimates are shown.']}
# S10: brief clinical information, with paired cohort ranges and safeguarded increments.
e5=read('results/clinical/e5_pooled.csv');c5=read('results/clinical/e5_cohort_metrics.csv')
m5=read('results/clinical/e5_model_metrics.csv')
def range_cell(rs,f,m=1):
    a=min(rs,key=lambda r:float(r[f]));b=max(rs,key=lambda r:float(r[f]))
    return {'text':f"{float(a[f])*m:.2f}–{float(b[f])*m:.2f}",'sources':[src(a,f),src(b,f)]}
def paired_range(rs,f,m=1):return joincells([range_cell(rs,f+'_base',m),range_cell(rs,f+'_aug',m)],' → ')
def model_rows(o,model):
    r=[r for r in e5 if r['row_type']=='model' and r['outcome']==o and r['comparison']==model]
    assert len(r)==1,(o,model,len(r))
    rs=[x for x in m5 if x['outcome']==o and x['model']==model]
    assert len(rs)==int(r[0]['k']) and sum(int(x['n']) for x in rs)==int(r[0]['n_total']),(o,model)
    return r[0],rs
rows=[]
for o in ['mortality','incident_any_adl']:
    b0,base=model_rows(o,'B0');b1,health=model_rows(o,'B1')
    assert [(x['cohort'],x['n'],x['people'],x['events']) for x in base]==[(x['cohort'],x['n'],x['people'],x['events']) for x in health],o
    support=joincells([num(b0,'n_total'),num(b0,'k')],' / ')
    rows.append([cell(outcome[o]+'; B0',b0,'comparison'),support,range_cell(base,'C',100),'—',range_cell(base,'brier',1000),joincells([range_cell(base,'cal_intercept',1000),range_cell(base,'cal_slope')],'; '),'—'])
    selected=[('B0+i_vs_B0','B0 + count vs B0'),('B1_vs_B0','B1 vs B0'),('B1+i_vs_B1','B1 + count vs B1'),('B1_iv_vs_iii','B1: two-point history vs current engagement')]
    if o=='mortality':selected += [('B2+i_vs_B2','B2 + count vs B2'),('B2_iv_vs_iii','B2: two-point history vs current engagement')]
    for comparison,label in selected:
        if comparison=='B1_vs_B0':
            delta={'text':f"{100*(float(b1['pooled'])-float(b0['pooled'])):.2f} (CI unavailable)",'sources':[src(b0,'pooled'),src(b1,'pooled')]}
            ranges=[joincells([range_cell(base,f,m),range_cell(health,f,m)],' → ') for f,m in [('C',100),('brier',1000),('cal_intercept',1000),('cal_slope',1)]]
            nb=[]
            for j in [1,2,3]:
                f='nb_'+str(j)
                nb.append({'text':f"{1000*sum(float(y[f])-float(x[f]) for x,y in zip(base,health))/len(base):.2f}",'sources':[src(x,f) for x in base+health]})
            rows.append([cell(outcome[o]+'; '+label,b1,'comparison'),support,ranges[0],delta,ranges[1],joincells(ranges[2:],'; '),joincells(nb,' / ')])
            continue
        found=[r for r in e5 if r['row_type']=='comparison' and r['outcome']==o and r['comparison']==comparison]
        assert len(found)==1,(o,comparison,len(found))
        r=found[0];rs=[x for x in c5 if x['outcome']==o and x['comparison']==comparison]
        assert len(rs)==int(r['k']) and sum(int(x['n']) for x in rs)==int(r['n_total']),(o,comparison)
        cal=joincells([paired_range(rs,'cal_intercept',1000),paired_range(rs,'cal_slope')],'; ')
        nb=joincells([val(r,'mean_dNB_thr'+str(j),m=1000) for j in [1,2,3]],' / ')
        rows.append([cell(outcome[o]+'; '+label,r,'comparison'),joincells([num(r,'n_total'),num(r,'k')],' / '),paired_range(rs,'C',100),ci(r,m=100),paired_range(rs,'brier',1000),cal,nb])
s10={'id':'S10','title':'Discrimination of the number of activities stopped added to brief clinical information','columns':['Outcome; clinical information; comparison','Person-intervals / cohorts','C-statistic ×100, base → augmented','ΔC ×100 (95% CI)','Brier ×1000, base → augmented','Calibration: intercept ×1000; slope, base → augmented','Mean ΔNB ×1000, low / middle / high threshold'],'rows':rows,'footnotes':['B0: age, sex, interview wave. B1: B0 + self-rated health. B2: B1 + activities of daily living (ADL) limitation count (death). C, concordance statistic; CI, confidence interval; NB, net benefit; Δ, augmented minus base. Brier scores measure prediction error; calibration assesses predicted versus observed risk. Entries are cohort ranges except pooled ΔC and mean ΔNB. B1 versus B0 is the difference between pooled C statistics (CI unavailable). NB thresholds (low/middle/high): death, 2/5/10%; new ADL limitation, 5/10/20%.'],'methods':['Prediction analyses used person-grouped five-fold cross-validation repeated five times. Five hundred participant-cluster bootstrap samples held out-of-fold predictions fixed, so their uncertainty excludes variation from refitting models. Pooled changes in the C-statistic use safeguarded Hartung–Knapp intervals. The self-rated-health increment is the difference between the separately pooled C statistics for B1 and B0; a paired confidence interval is unavailable. The table gives the range of cohort-specific base and augmented C-statistics, Brier scores, and calibration intercepts and slopes. Brier scores measure prediction error; calibration intercepts and slopes assess agreement between predicted and observed risk. Net-benefit changes are simple cohort means at the stated thresholds.','B0 included age, sex, and interview wave. B1 added self-rated health, and B2 added activities-of-daily-living limitation count for death. Count means the number of activities stopped. Two-point history included status at both interviews for each everyday activity and was compared with current engagement at the second interview. The complete-health sample required recorded self-rated health, plus recorded activities-of-daily-living limitation count for death. All paired comparisons used the same person-intervals within each outcome. B0 rows show baseline values.']}
# S11: matched unconditional comparators accompany current-state and concurrent-health models.
contexts={'bmi_or_weight_loss':'Weight or body mass index loss','incident_non_target_disease':'Newly reported condition','self_rated_health_worsening':'Worsened self-rated health','joint_context_no_fi':'Joint health measures without frailty index','transition_hospitalization':'Hospital admission','cesd_worsening':'Worsened depressive symptoms','fi_change':'Frailty index change','joint_context':'Joint health measures including frailty index'}
rows=[]
for o in ['mortality','incident_any_adl','incident_any_iadl','multimorbidity_progression']:
    ctx=sorted(set(r['_key']['context_id'] for r in HK if r['family']=='context-synthesis' and r['_key']['outcome_id']==o))
    for c in ctx:
        base=hk('context-synthesis',outcome_id=o,context_id=c,model_id='base_same_set_'+c,state_adjustment='unconditional',term='loss_2plus')
        for m,state,label in [('base_same_set_'+c,'current_state_held','Current-status-held'),('plus_'+c,'unconditional','Concurrent-health adjusted')]:
            r=hk('context-synthesis',outcome_id=o,context_id=c,model_id=m,state_adjustment=state,term='loss_2plus')
            assert r['n']==base['n'] and r['cohorts']==base['cohorts']
            rows.append([outcome[o]+'; '+contexts[c]+'; '+label,cohorts(r),num(r,'n'),ci(base),ci(r),pi(r)])
for co in ['hrs','share']:
    for c in ['self_rated_health_worsening','transition_hospitalization','cesd_worsening','bmi_or_weight_loss','incident_non_target_disease','fi_change','joint_context_no_fi','joint_context']:
        fi=c in ['fi_change','joint_context']
        path=(f'results/context/frailty/{co}-context-graded-models.csv' if fi else f'results/context/models/{co}-context-graded-models.csv')
        rs=read(path)
        base=[r for r in rs if r['outcome_id']=='mortality' and r['context_id']==c and r['model_id']=='base_same_set_'+c and r['state_adjustment']=='unconditional' and r['term']=='loss_2plus' and r['model_status']=='PASS']
        adj=[r for r in rs if r['outcome_id']=='mortality' and r['context_id']==c and r['model_id']=='plus_'+c and r['state_adjustment']=='unconditional' and r['term']=='loss_2plus' and r['model_status']=='PASS']
        if co=='share' and c=='cesd_worsening':
            assert not adj
            continue
        assert len(base)==len(adj)==1,(co,c,len(base),len(adj))
        b,r=base[0],adj[0];assert b['n']==r['n'] and b['events']==r['events']
        label='Death; '+contexts[c]+(' (computed in original analysis)' if fi else '')
        rows.append([label,coh[co],num(r,'n'),ci(b,'estimate','ci_low','ci_high'),ci(r,'estimate','ci_low','ci_high'),'Not applicable'])
s11={'id':'S11','title':'Current-status-held and concurrent-health models','columns':['Outcome; health context; model','Cohorts','Person-intervals','Matched unconditional RR (95% CI)','Current-status-held or health-adjusted RR (95% CI)','Adjusted RR prediction interval'],'rows':rows,'footnotes':['All estimates compare two or more activities stopped with none. CI, confidence interval; RR, risk ratio; ADL, activities of daily living; IADL, instrumental activities of daily living; ELSA, English Longitudinal Study of Ageing; HRS, Health and Retirement Study; SHARE, Survey of Health, Ageing and Retirement in Europe.'],'methods':['Current-status-held models included the status of all everyday activities at the second interview and were compared with models without those status terms on the same person-intervals. The number of activities initially present, current status, and the number stopped are linked by construction. Some combinations cannot occur, so these conditional associations rely on model restrictions across structurally unobserved combinations. They do not identify a within-status comparison supported at every number of activities stopped.','Concurrent-health models added recorded changes between the first and second interviews. Each estimate is paired with the unconditional model fitted on the same sample. The joint model used the health measures available in that cohort. Frailty-index-dependent rows come from the original analysis. The original constructor used 32 candidate deficits and required at least 26 observed deficits, dividing the deficit sum by the number observed and requiring information from all three deficit groups. This definition follows the recorded constructor audit.']}
# S14: cohort work-exit support and standardized word-recall differences in a single table.
work=read('results/standardization/cohort_work_exit_contrasts.csv');rows=[]
for o in ['mortality','incident_any_adl']:
    for r in [x for x in work if x['outcome_id']==o]:
        co=r['cohort'];status=r['model_status']
        n=num(r,'n') if r['n'] and co!='mhas' else 'Not available'
        ev=num(r,'events') if r['events'] else 'Not available'
        if status=='PASS':
            v=ci(r,'estimate','ci_low','ci_high')
            contrast=joincells([val(r,'retirement_exit_rr_vs_continued_work'),val(r,'other_exit_rr_vs_continued_work')],' / ')
            contrast['text']='Retirement-linked / other exit RR: '+contrast['text']
        else:
            v='Not evaluable';contrast={'charls':'Retirement-linked status not recorded','klosa':'Retirement-linked status not recorded','mhas':'No comparable interval','elsa':'Too few events per model parameter','share':'Too few events after other exit'}[co]
        rows.append([outcome[o]+'; other vs retirement-linked paid-work exit',coh[co],n,ev,v,contrast])
    if o=='incident_any_adl':
        r=hk('work_exit',outcome_id=o)
        rows.append([outcome[o]+'; descriptive work-exit synthesis',cohorts(r),num(r,'n'),'Not applicable',ci(r),'Two cohorts; outside the main pooling rule'])
for term,mod,label in [('any_withdrawal','any_withdrawal','Any activity stopped'),('loss_1','score_categorical','One activity stopped'),('loss_2plus','score_categorical','Two or more activities stopped'),('activity_loss','mutually_adjusted_components','Regular physical activity component'),('alcohol_loss','mutually_adjusted_components','Drinking component'),('work_loss','mutually_adjusted_components','Paid-work component')]:
    r=hk('primary',outcome_id='word_recall_change',exposure_model=mod,term=term)
    pred=pi(r);pred['text']='Prediction interval, SD: '+pred['text']
    rows.append(['Word-recall change; '+label,cohorts(r),num(r,'n'),'Not applicable',ci(r),pred])
s14={'id':'S14','title':'Paid-work exit context and word-recall results','columns':['Outcome and comparison','Cohorts','Person-intervals','Events','Ratio of RRs or mean difference, SD (95% CI)','Context or prediction interval'],'rows':rows,'footnotes':['CI, confidence interval; RR, risk ratio; SD, standard deviation; ADL, activities of daily living; CHARLS, China Health and Retirement Longitudinal Study; ELSA, English Longitudinal Study of Ageing; HRS, Health and Retirement Study; KLoSA, Korean Longitudinal Study of Ageing; MHAS, Mexican Health and Aging Study; SHARE, Survey of Health, Ageing and Retirement in Europe.'],'methods':['Paid-work exit comparisons distinguished retirement-linked status from other exits where the source interview measured it. The direct contrast is the ratio of the other-exit risk ratio to the retirement-linked-exit risk ratio, using their within-model covariance. Both component risk ratios use continued paid work as the reference. Retirement-linked status does not establish whether an exit was planned or voluntary. The descriptive work-exit synthesis contains two cohorts and is separate from the main rule requiring at least three.','Word-recall change was analyzed by a linear model with recall at the second interview included as a covariate and participant-grouped robust variance. Recall estimates and their intervals are standardized mean differences in SD units. They are pooled from evaluable cohorts and use safeguarded Hartung–Knapp confidence and prediction intervals.']}
tables=[s8,s10,s11,s14]
for t in tables:
    for i,row in enumerate(t['rows']):
        assert len(row)==len(t['columns']),(t['id'],i)
        for c in row:
            if isinstance(c,str):assert not re.search(r'\d',c),(t['id'],i,c)
            else:
                c['text']=re.sub(r'(?<![0-9.])-0\.00(?![0-9])','0.00',c['text'])
                assert c['sources'] and c['text']
                for s in c['sources']:
                    actual=list(csv.DictReader(open(s['path'])))[s['row']-2][s['field']]
                    assert actual==s['raw'],s
(OUT/'tables_8_10_11_14.json').write_text(json.dumps(tables,ensure_ascii=False,indent=2)+'\n')
print('Built tables: '+', '.join(f"{t['id']}={len(t['rows'])} rows" for t in tables))
print('Numeric cells:',sum(isinstance(c,dict) for t in tables for r in t['rows'] for c in r))
print('Orphan numeric cells: 0; all CSV raw values and record numbers verified.')
