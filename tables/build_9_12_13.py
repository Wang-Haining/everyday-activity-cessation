import csv,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'tables/generated'
OUT.mkdir(parents=True,exist_ok=True)
HK='results/primary/hk_pooled.csv'
C='results/sensitivity/c_weighted_models.csv'
BIO='results/biomarkers/systematic-biomarker-results.csv'
MECH='results/biomarkers/hba1c_summary.csv'
loaded={}
def records(path):
    if path not in loaded: loaded[path]=[(i,r) for i,r in enumerate(csv.DictReader((ROOT/path).open()),2)]
    return loaded[path]
def one(path,**keys):
    found=[(i,r) for i,r in records(path) if all(r.get(k)==str(v) for k,v in keys.items())]
    assert len(found)==1,(path,keys,len(found))
    return found[0]
def ref(path,row,r,field):return {'path':path,'row':row,'field':field,'raw':r[field]}
def cell(path,ir,field,fmt=None):
    i,r=ir
    text=fmt(float(r[field])) if fmt else r[field]
    return {'text':text,'sources':[ref(path,i,r,field)]}
def num(path,ir,field):return cell(path,ir,field,lambda x:f'{int(x):,}')
def estimate(path,ir,field='pooled',lo='ci_low_safeguarded',hi='ci_high_safeguarded',pct=False,places=2):
    i,r=ir; factor=100 if pct else 1
    text=f'{float(r[field])*factor:.{places}f} ({float(r[lo])*factor:.2f} to {float(r[hi])*factor:.2f})'
    text=text.replace('-0.00)', '0.00)').replace('(-0.00 ', '(0.00 ')
    return {'text':text,'sources':[ref(path,i,r,k) for k in [field,lo,hi]]}
def hrow(family,**keys):
    found=[(i,r) for i,r in records(HK) if r['family']==family and all(json.loads(r['key']).get(k)==v for k,v in keys.items())]
    assert len(found)==1,(family,keys,len(found))
    return found[0]
def sumcell(path,irs,field):
    assert irs
    return {'text':f'{int(sum(float(r[field]) for _,r in irs)):,}','sources':[ref(path,i,r,field) for i,r in irs]}
def sum_c_n(outcome,model,estimator,term,field='n'):
    rows=[(i,r) for i,r in records(C) if r['row_type']=='cohort' and r['outcome']==outcome and r['model']==model and r['estimator']==estimator and r['term']==term]
    return sumcell(C,rows,field)
def cr(outcome,model,estimator,term):return one(C,row_type='pooled',outcome=outcome,model=model,estimator=estimator,term=term)
rows=[]
outs={'mortality':'Death','incident_any_adl':'New ADL limitation'}
for estimator,label in [('complete_case','Complete case'),('ipw_truncated','Inclusion weights, truncated'),('ipw_untruncated','Inclusion weights, untruncated'),('ipw_exposure_numerator_truncated','Inclusion weights, exposure numerator'),('minimal_covariates_all_eligible','Minimal covariates, all eligible'),('minimal_covariates_retained_rows','Minimal covariates, complete cases')]:
    for outcome,labelout in outs.items():
        a,b=[cr(outcome,'rr_count_model',estimator,t) for t in ['loss_1','loss_2plus']]
        rows.append([label+'; '+labelout[:1].lower()+labelout[1:],num(C,a,'k'),sum_c_n(outcome,'rr_count_model',estimator,'loss_1'),'Reference',estimate(C,a,'rr','ci_low','ci_high'),estimate(C,b,'rr','ci_low','ci_high')])
for estimator,label in [('complete_case','Common-population risk, complete case'),('ipw_truncated','Common-population risk, inclusion weighted')]:
    for outcome,labelout in outs.items():
        irs=[cr(outcome,'standardised_risk_common_target_k_ge_2',estimator,t) for t in ['withdrawal_0','withdrawal_1','withdrawal_2plus']]
        rows.append([label+'; '+labelout[:1].lower()+labelout[1:],num(C,irs[0],'k'),sum_c_n(outcome,'standardised_risk_common_target_k_ge_2',estimator,'withdrawal_0','n_target_rows'),*[estimate(C,ir,'risk','ci_low','ci_high',True,1) for ir in irs]])
for omitted,name in [('elsa','England'),('hrs','United States'),('klosa','Korea'),('share','continental Europe')]:
    a,b=[hrow('leave_one_out',omitted=omitted,outcome_id='mortality',term=t) for t in ['loss_1','loss_2plus']]
    rows.append(['Omit '+name+'; death',num(HK,a,'k'),num(HK,a,'n'),'Reference',estimate(HK,a),estimate(HK,b)])
for age,word in [(65,'sixty-five'),(70,'seventy')]:
    for outcome,labelout in outs.items():
        a,b=[hrow('age_floor',age_floor=age,outcome_id=outcome,term=t) for t in ['loss_1','loss_2plus']]
        label={'text':f'Age ≥{age} years; {labelout[:1].lower()+labelout[1:]}','sources':[ref(HK,a[0],a[1],'key')]}
        rows.append([label,num(HK,a,'k'),num(HK,a,'n'),'Reference',estimate(HK,a),estimate(HK,b)])
for variant,label in [('all_nosmoking','Smoking omitted, expanded sample'),('all_minimal','Demographic adjustment, expanded sample')]:
    for outcome,labelout in outs.items():
        a,b=[hrow('missingness',variant=variant,outcome_id=outcome,term=t) for t in ['loss_1','loss_2plus']]
        rows.append([label+'; '+labelout[:1].lower()+labelout[1:],num(HK,a,'k'),num(HK,a,'n'),'Reference',estimate(HK,a),estimate(HK,b)])
for model,label in [('survivors','New ADL limitation, survivors'),('death_or_outcome','Death or new ADL limitation')]:
    a,b=[hrow('composite-synthesis',exposure_model='graded',model_id=model,outcome_id='incident_any_adl',state_adjustment='unconditional',term=t) for t in ['loss_1','loss_2plus']]
    rows.append([label,num(HK,a,'k'),num(HK,a,'n'),'Reference',estimate(HK,a),estimate(HK,b)])
for co,name,model,label in [('hrs','United States','respondent_weighted','Survey weighted'),('hrs','United States','unweighted_same_rows','Unweighted, survey-weight sample'),('share','Continental Europe','country_clustered_variance','Country-clustered variance'),('share','Continental Europe','country_fixed_effects','Country fixed effects')]:
    path=f'results/sensitivity/design/{co}-design-sensitivity.csv'
    a,b=[one(path,model=model,term=t) for t in ['loss_1','loss_2plus']]
    rows.append([name+', '+label.lower()+'; death','One',num(path,a,'n'),'Reference',estimate(path,a,'estimate','ci_low','ci_high'),estimate(path,b,'estimate','ci_low','ci_high')])
path='results/primary/systematic-results-matrix.csv'
for outcome,labelout in outs.items():
    a,b=[one(path,cohort='mhas',outcome_id=outcome,scope='all_primary_wave_intervals',adjustment='full',exposure_model='score_categorical',term=t) for t in ['loss_1','loss_2plus']]
    rows.append(['Mexico, wider follow-up; '+labelout[:1].lower()+labelout[1:],'One',num(path,a,'n'),'Reference',estimate(path,a,'estimate','ci_low','ci_high'),estimate(path,b,'estimate','ci_low','ci_high')])
s9={'id':'S9','title':'Inclusion weighting and sensitivity analyses','columns':['Analysis and outcome','Cohorts','Person-intervals','None','One','Two or more'],'rows':rows,'footnotes':['Columns denote number of activities stopped. Values are risk ratios (95% confidence intervals), except common-population risks, shown as percentages. ADL, activities of daily living.','Pooled intervals use safeguarded Hartung–Knapp uncertainty. Common-population risks share an attainable target population; category-specific risks use different eligible populations. Mexico had no comparable interval.'], 'methods':['Inclusion models predicted complete-case retention from age, sex, interview wave, initial engagement, activities stopped and started, prior self-rated health, functional limitations, depressive symptoms, multimorbidity, education, income, smoking, and availability of an earlier interview. Missing predictors entered with missing indicators. Stabilized weights used the marginal retention probability divided by the fitted inclusion probability. We truncated weights at the first and ninety-ninth percentiles and normalized them to mean one. Sensitivities used untruncated weights and a numerator specific to the number of activities stopped. Minimal-covariate models included optional covariates only when they were observed in at least ninety-five percent of outcome-observed eligible person-intervals.','Weighted standardized risks averaged fitted probabilities over participants initially engaged in at least two activities, giving an attainable common target for all three categories. Person-interval counts in these rows describe the target population. Category-specific standardization instead uses the population eligible for each category and is reported separately. Inclusion weighting addresses measured differences in retention and does not remove selection through survival to the second interview. New ADL limitation also requires survival and observed follow-up.','Death estimates were examined after omitting each contributing cohort. New ADL limitation had only three supporting cohorts, so each omission left fewer than three and did not support random-effects synthesis. England did not meet the required event support for the two-or-more category at the higher age thresholds. Smoking-omitted models included otherwise eligible participants, including the SHARE wave with structurally unavailable smoking data. Country structure and survey-weight analyses were confined to their supported cohorts. MHAS had no comparable interval and contributed only to wider-follow-up sensitivity analyses.']}
# Two-activity biomarker model and assay-specific results.
rows=[]
markers={'burden_z':'Immunometabolic burden','crp':'C-reactive protein','hba1c':'Glycated hemoglobin','triglycerides':'Triglycerides','hdl_adverse':'HDL, adverse orientation'}
for co,name in [('charls','CHARLS'),('elsa','ELSA')]:
    for marker,label in markers.items():
        ir=one(BIO,cohort=co,assay_scope='comparable',model_family='baseline_marker_predicts_contraction',marker=marker)
        rows.append([name,label+' before stopping an activity',num(BIO,ir,'n'),'RR per SD',estimate(BIO,ir,'estimate','ci_low','ci_high')])
    ir=one(BIO,cohort=co,assay_scope='comparable__crp_lt_10',model_family='baseline_marker_predicts_contraction',marker='burden_z')
    rows.append([name,'Burden, excluding high C-reactive protein',num(BIO,ir,'n'),'RR per SD',estimate(BIO,ir,'estimate','ci_low','ci_high')])
for marker,label in [('crp','C-reactive protein'),('hba1c','Glycated hemoglobin'),('triglycerides','Triglycerides'),('hdl','HDL')]:
    ir=one(BIO,cohort='elsa',model_family='baseline_adjusted_repeated_assay',marker=marker)
    rows.append(['ELSA',label+', repeat assay',num(BIO,ir,'n'),'Difference, SD',estimate(BIO,ir,'estimate','ci_low','ci_high',places=3)])
ir=one(MECH,cohorts='elsa',outcome='followup_hba1c',model='any_contraction',n_person_intervals='4965',estimate_scale='mean_difference')
rows.append(['ELSA','Glycated hemoglobin, native-unit repeat assay',num(MECH,ir,'n_person_intervals'),'Difference, percentage points',estimate(MECH,ir,'estimate','ci_low','ci_high',places=3)])
ir=one(BIO,cohort='share',assay_scope='off_window_transport',model_family='baseline_marker_predicts_contraction',marker='burden_z')
rows.append(['SHARE','Burden, outside the core window',num(BIO,ir,'n'),'RR per SD','Unavailable: insufficient events'])
rows.append(['HRS, KLoSA, MHAS','No assay results in the available analysis inputs','Unavailable','Unavailable','Unavailable'])
s12={'id':'S12','title':'Biomarker and repeated-assay findings','columns':['Cohort','Measure and timing','Person-intervals','Scale','Estimate (95% CI)'],'rows':rows,'footnotes':['CI, confidence interval; SD, standard deviation; RR, risk ratio; HDL, high-density lipoprotein; CHARLS, China Health and Retirement Longitudinal Study; ELSA, English Longitudinal Study of Ageing.','HRS, Health and Retirement Study; KLoSA, Korean Longitudinal Study of Aging; MHAS, Mexican Health and Aging Study; SHARE, Survey of Health, Ageing and Retirement in Europe.','Exposure covers drinking or regular physical activity. Repeated assays compare stopping an activity with neither activity stopped, adjusting for the first assay. Glycated-hemoglobin samples differ by scale. The unrounded HDL upper confidence limit is below zero.'], 'methods':['The biomarker analysis assessed stopping drinking or regular physical activity. Immunometabolic burden combined standardized inflammatory, glycemic, and lipid measures from the first interview, with high-density lipoprotein oriented so that lower levels indicated greater burden. Antecedent-marker models related each measure to subsequently stopping an activity. Separate repeated-assay models compared follow-up measures by whether an activity stopped while adjusting for the first assay and available covariates. ELSA repeated assays spanned the first interview to the outcome visit, approximately four years. The glycated-hemoglobin analysis in native percentage-point units used a separate covariate-complete sample. These assay-specific populations and mixed findings do not establish a common biological mechanism for the three-activity measure.']}
# Disease-specific contrasts on matched disease-free risk sets.
rows=[]
for outcome,name in [('incident_diabetes','Diabetes'),('incident_heart_disease','Heart disease'),('incident_stroke','Stroke'),('incident_hypertension','Hypertension'),('incident_cancer','Cancer'),('incident_arthritis','Arthritis')]:
    for model,label in [('three_domain_any_without_current_states','Standard adjustment'),('any_withdrawal','Current engagement also held')]:
        ir=hrow('frailty_extension',model_id=model,outcome_id=outcome,term='any_withdrawal')
        rows.append([name,label,num(HK,ir,'k'),num(HK,ir,'n'),estimate(HK,ir)])
s13={'id':'S13','title':'Disease-specific outcomes','columns':['New diagnosis','Model','Cohorts','Person-intervals','Risk ratio (95% CI)'],'rows':rows,'footnotes':['CI, confidence interval. Each comparison is stopping any activity versus none stopped among participants free of the specified disease at the second interview and with observed follow-up.','All models use the matched disease-specific complete-case sample. Standard adjustment excludes current engagement indicators. The other model also holds current drinking, regular physical activity, and paid work status. Pooled intervals use safeguarded Hartung–Knapp uncertainty.'], 'methods':['Disease-specific analyses included participants without the specified diagnosis at the second interview and with known follow-up status for that diagnosis. Stopping any activity across drinking, regular physical activity, and paid work was compared with none stopped. Models used the standard covariates and then added current engagement indicators on the same disease-specific sample. Diagnoses were participant-reported and required survival and observed follow-up. Cohort support depended on adequate outcome events and non-events.']}
tables=[s9,s12,s13]
for table in tables:
    assert table['rows'] and len(table['footnotes'])<=6
    for row in table['rows']:
        assert len(row)==len(table['columns'])
        for item in row:
            if isinstance(item,str):assert not re.search(r'\d',item),item
            else:
                assert item['sources']
                for src in item['sources']:
                    source_row=dict(records(src['path']))[src['row']]
                    assert source_row[src['field']]==src['raw'],src
(OUT/'tables_9_12_13.json').write_text(json.dumps(tables,indent=2)+'\n')
print('Built tables:',', '.join(f"{t['id']}={len(t['rows'])} rows" for t in tables))
print('Numeric cells:',sum(isinstance(c,dict) for t in tables for r in t['rows'] for c in r),'source checks passed; no orphan numeric cells')
