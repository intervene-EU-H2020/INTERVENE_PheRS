```python
#read in the UKBB PheRS models
import pandas as pd
import pickle
import numpy as np

endpoints = ['C3_COLORECTAL','C3_PROSTATE','C3_BREAST','C3_BRONCHUS_LUNG',
             'C3_CANCER','C3_MELANOMA_SKIN','C3_PROSTATE','COX_ARTHROSIS',
             'F5_DEPRESSIO','G6_AD_WIDE','G6_EPLEPSY',
             'GOUT','I9_AF',
             'I9_CHD','I9_VTE','ILD',
             'J10_ASTHMA','K11_APPENDACUT','KNEE_ARTHROSIS',
             'RHEUMA_SEROPOS_OTH','T2D']

#get the saved models
models = {}
#key = endpoint
#value = best model file path in pickle format
data = {'endpoint':[], 'feature_name':[], 'ukbb_coefficient':[], 'fg_coefficient':[]}
#keys = endpoint, feature_name, ukbb_coefficient, fg_coefficient

#values are lists
for e in endpoints:
    path = '/home/ivm/wrk/INTERVENE_PheRS_UKBB/results_250523_with_age_no_downsampling_down/'+e+'_exposure=1999-01-01-2008-12-31-washoutend=2010-12-31-observationend=2019-01-01/'
    models[e] = path+'best_model.pkl'
    #then read in the coefficients for each phecode
    df_ep = pd.read_csv(path+'best_model_coefficients.txt',sep='\t',
                                 dtype={'feature_name':str,'coefficient':float})
    #print('UKBB '+e+" : "+str(len(df_ep))+" non-zero coefficients.")
    for index,row in df_ep.iterrows():
        data['endpoint'].append(e)
        data['feature_name'].append(row['feature_name'])
        data['ukbb_coefficient'].append(row['coefficient'])
        data['fg_coefficient'].append(np.nan)
        
        
df = pd.DataFrame.from_dict(data)
df
```


```python
#run predictions on the FinnGen data for all of the UKBB models
from os import system

for e in endpoints:
    fgdir = '/home/ivm/from_SES_sandbox/wrk/INTERVENE_pheRS/results_R10_030523_with_age_no_downsampling/'+e+'_exposure=1999-01-01-2008-12-31-washoutend=2010-12-31-observationend=2019-01-01/'
    ukbbdir = '/home/ivm/wrk/INTERVENE_PheRS_UKBB/results_250523_with_age_no_downsampling_down/'+e+'_exposure=1999-01-01-2008-12-31-washoutend=2010-12-31-observationend=2019-01-01/'
    phecodefile = fgdir+'target-'+e+'-PheRS-ML-input.txt.gz'
    outdir = ukbbdir+'finngen_'
    scaler = ukbbdir+'scaler.pkl'
    imputer = ukbbdir+'imputer.pkl'
    excludevars = ukbbdir+'target-'+e+'-excluded-phecodes.txt'
    model = models[e]
    nproc = '4'
    cmd = 'python3 /home/ivm/wrk/INTERVENE_PheRS/src/scoreLogreg.py --infile '+phecodefile+' --outdir '+outdir+' --scaler '+scaler+' --imputer '+imputer+' --excludevars '+excludevars+' --nproc '+nproc+' --model '+model
    print(model)
    print(cmd)
    system(cmd)
#predictions are now saved and scores can be fed to the Cox model pipeline
```


```python
#read in the model coefficients from FinnGen models also
for e in endpoints:
    path = '/home/ivm/from_SES_sandbox/wrk/INTERVENE_pheRS/results_R10_030523_with_age_no_downsampling/'+e+'_exposure=1999-01-01-2008-12-31-washoutend=2010-12-31-observationend=2019-01-01/'
    #then read in the coefficients for each phecode
    df_ep = pd.read_csv(path+'best_model_coefficients.txt',sep='\t',
                                 dtype={'feature_name':str,'coefficient':float})
    print('FinnGen '+e+" : "+str(len(df_ep))+" non-zero coefficients.")
    for index,row in df_ep.iterrows():
        #check if the endpoint+feature combination exists in the df
        df_aux = df.loc[(df['endpoint']==e) & (df['feature_name']==row['feature_name'])]
        if not df_aux.empty:
            df.at[(df['endpoint']==e) & (df['feature_name']==row['feature_name']),'fg_coefficient'] = row['coefficient']
        else:
            #create a new row
            new_row = {'endpoint':e, 'feature_name':row['feature_name'], 'ukbb_coefficient':np.nan,
                       'fg_coefficient':row['coefficient']}
            df = df.append(new_row,ignore_index=True)
df
```


```python
import matplotlib
%matplotlib inline
import plotly.express as px
df_nonan = df.fillna(0.0)
#plot coefficient in ukbb vs finngen
px.scatter(df_nonan,x='fg_coefficient',y='ukbb_coefficient',
           facet_col='endpoint',facet_col_wrap=4,
          height=1000,width=1000)
```


```python
from matplotlib import pyplot as plt
from sklearn.metrics import average_precision_score,roc_auc_score,roc_curve,precision_recall_curve

outdir = '/home/ivm/wrk/export/'
print(len(endpoints))
fig,axs = plt.subplots(7,3,figsize=(16,12))
i = 0
j = 0
#compare the precision-recall and ROC curves in predicting FinnGen samples between models trained in FinnGen and UKBB
for e in endpoints:
    fgdir = '/home/ivm/from_SES_sandbox/wrk/INTERVENE_pheRS/results_R10_030523_with_age_no_downsampling/'+e+'_exposure=1999-01-01-2008-12-31-washoutend=2010-12-31-observationend=2019-01-01/'
    ukbbdir = '/home/ivm/wrk/INTERVENE_PheRS_UKBB/results_250523_with_age_no_downsampling_down/'+e+'_exposure=1999-01-01-2008-12-31-washoutend=2010-12-31-observationend=2019-01-01/'
    #read in the FinnGen->FinnGen predictions
    df_fg = pd.read_csv(fgdir+'pred_probas.txt.gz',sep='\t')
    #print sample sizez
    print(e+', number of cases = '+str(len(df_fg.loc[df_fg['true_class']>0])))
    auprc_fg = average_precision_score(df_fg['true_class'],df_fg['pred_class1_prob'])
    precision_fg,recall_fg,thresholds_fg = precision_recall_curve(df_fg['true_class'],df_fg['pred_class1_prob'])
    #read in the UKBB->FinnGen predictions
    df_ukbb = pd.read_csv(ukbbdir+'finngen_pred_probas.txt.gz',sep='\t')
    auprc_ukbb = average_precision_score(df_ukbb['true_class'],df_ukbb['pred_class1_prob'])
    precision_ukbb,recall_ukbb,thresholds_ukbb = precision_recall_curve(df_ukbb['true_class'],df_ukbb['pred_class1_prob'])
    #plot
    axs[i,j].plot(recall_fg,precision_fg,label='FG->FG, AUPRC='+str(round(auprc_fg,3)))
    axs[i,j].plot(recall_ukbb,precision_ukbb,label='UKBB->FG, AUPRC='+str(round(auprc_ukbb,3)))
    axs[i,j].set_xlabel('recall')
    axs[i,j].set_ylabel('precision')
    axs[i,j].legend()
    axs[i,j].set_title(e)
    j += 1
    if j==3:
        j = 0
        i += 1
plt.tight_layout()
plt.savefig(outdir+'INTERVENE-PherS-UKBB-vs-FinnGen-prc-050723.png',dpi=300)
```


```python
fig,axs = plt.subplots(7,3,figsize=(16,12))
i = 0
j = 0
#compare the precision-recall and ROC curves in predicting FinnGen samples between models trained in FinnGen and UKBB
for e in endpoints:
    fgdir = '/home/ivm/from_SES_sandbox/wrk/INTERVENE_pheRS/results_R10_030523_with_age_no_downsampling/'+e+'_exposure=1999-01-01-2008-12-31-washoutend=2010-12-31-observationend=2019-01-01/'
    ukbbdir = '/home/ivm/wrk/INTERVENE_PheRS_UKBB/results_250523_with_age_no_downsampling_down/'+e+'_exposure=1999-01-01-2008-12-31-washoutend=2010-12-31-observationend=2019-01-01/'
    #read in the FinnGen->FinnGen predictions
    df_fg = pd.read_csv(fgdir+'pred_probas.txt.gz',sep='\t')
    auc_fg = roc_auc_score(df_fg['true_class'],df_fg['pred_class1_prob'])
    fpr_fg,tpr_fg,thresholds_fg = roc_curve(df_fg['true_class'],df_fg['pred_class1_prob'])
    #read in the UKBB->FinnGen predictions
    df_ukbb = pd.read_csv(ukbbdir+'finngen_pred_probas.txt.gz',sep='\t')
    auc_ukbb = roc_auc_score(df_ukbb['true_class'],df_ukbb['pred_class1_prob'])
    fpr_ukbb,tpr_ukbb,thresholds_ukbb = roc_curve(df_ukbb['true_class'],df_ukbb['pred_class1_prob'])
    #plot
    axs[i,j].plot(fpr_fg,tpr_fg,label='FG->FG, AUC='+str(round(auc_fg,3)))
    axs[i,j].plot(fpr_ukbb,tpr_ukbb,label='UKBB->FG, AUC='+str(round(auc_ukbb,3)))
    axs[i,j].set_xlabel('false positive rate')
    axs[i,j].set_ylabel('true positive rate')
    axs[i,j].legend()
    axs[i,j].set_title(e)
    j += 1
    if j==3:
        j = 0
        i += 1
plt.tight_layout()
plt.savefig(outdir+'INTERVENE-PherS-UKBB-vs-FinnGen-roc-050723.png',dpi=300)
```


```python
df
```


```python
df.to_csv(outdir+'INTERVENE-PheRS-coefficients-UKBB-vs-FinnGen-050723.csv',sep=',',index=False)
```


```python

```
