#!/usr/bin/env python

import argparse
import logging
import csv
import gzip
import pickle
import sys

import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import average_precision_score,roc_auc_score,roc_curve,precision_recall_curve
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from scipy.stats import norm

def fitLogReg():

    parser = argparse.ArgumentParser()

    #Command line arguments:
    
    parser.add_argument("--infile",help="Full path to an INTERVENE phecode file. If last two letters of file name are gz, gzipped file is assumed.",type=str,default=None)
    parser.add_argument("--excludevars",help="Full path to a file containing variable names that should be excluded from the model (default=nothing).",type=str,default=None)
    parser.add_argument("--paramgridfile",help="File containing the parameter grid explored using cross-validation. One parameter per row, first value is key, rest are possible values.",type=str,default=None)
    parser.add_argument("--outdir",help="Output directory (must exist, default=./).",type=str,default="./")
    parser.add_argument("--penalty",help="Penalty used with the model ('l1','l2' or 'elasticnet'=default).",type=str,choices=['l1','l2','elasticnet'],default='elasticnet')
    parser.add_argument("--cv",help="Number of folds used in stratified k-fold cross validation (default=5).",type=int,default=5)
    parser.add_argument("--scoring",help="Which scoring to use in selecting best model hyperparameters (default=average_precision_score). See possible scores from: https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter",type=str)
    parser.add_argument("--seed",help="Random number generator seed (default=42).",type=int,default=42)
    parser.add_argument("--nproc",help="Number of parallel processes used (default=1).",type=int,default=1)

    args = parser.parse_args()

    #save the command used to evoke this script into a file                                                                                                                                 
    logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', filename=args.outdir+'fitLogreg.log',filemode='w',level=logging.INFO)
    logging.info("The command used to evoke this script:")
    logging.info(" ".join(sys.argv))

    #read in the hyperparameter search grid
    with open(args.paramgridfile,'rt',encoding='utf-8') as infile:
        r = csv.reader(infile,delimiter='\t')
        hyper_grid = {} #key = parameter name, value = list of all possible values
        for row in r:
            if len(row)<1: continue
            key = row[0]
            if row[0][0]=='#': continue #skip comment lines that start with #
            if key=='penalty': hyper_grid[key] = row[1:]
            #elif key=='class_weight':
            #    min_weight = float(row[1])
            #    max_weight = float(row[2])
            #    step = float(row[3])
            #    hyper_grid[key] = []
            #    for class1_frac in np.arange(min_weight,max_weight,step):
            #        hyper_grid[key].append({1:class1_frac,0:1-class1_frac})
            elif key=='l1_ratio':
                min_l1 = float(row[1])
                max_l1 = float(row[2])
                step = float(row[3])
                hyper_grid[key] = []
                for l1 in np.arange(min_l1,max_l1,step): hyper_grid[key].append(l1)
            elif key=='Cs': hyper_grid[key] = [float(row[i]) for i in range(1,len(row))]
            elif key=='fit_intercept':
                hyper_grid[key] = []
                for entry in row[1:]:
                    if entry=='False': hyper_grid[key].append(False)
                    elif entry=='True': hyper_grid[key].append(True)
    logging.info("Hyperparamter grid read in successfully.")
    
    #read in names of variables that should be excluded from the model
    with open(args.excludevars,'rt',encoding='utf-8') as infile:
        r = csv.reader(infile,delimiter='\t')
        excludevars = set(['ID','follow_up_time','case_status','train_status']+['PC'+str(i) for i in range(1,11)])
        for row in r: excludevars.add(row[0])
    logging.info("Names of excluded variables read in successfully.")

    #read in the training and test data
    #first read in the header
    if args.infile[-2:]=='gz':
        in_handle = gzip.open(args.infile,'rt',encoding='utf-8')
    else: in_handle = open(args.infile,'rt',encoding='utf-8')
    with in_handle as infile:
        r = csv.reader(infile,delimiter='\t')
        for row in r:
            #print(row)
            if row[0]=='#': continue
            #row = row.strip().split('\t')
            if row[0]=='#ID':
                feature2index = {row[0][1:]:0} #key = feature name, value = index in input file
                for i in range(1,len(row)): feature2index[row[i]] = i
                break
    #then read in the training data
    usecols = []
    features = []
    for key in feature2index.keys():
        if key not in excludevars:
            usecols.append(feature2index[key])
            features.append((feature2index[key],key))
    features = [f[1] for f in sorted(features)] #this list now contains the names of the features in the same order as the columns are in full_data        
    full_data = np.loadtxt(args.infile,usecols=usecols,dtype=float)
    case_status = np.loadtxt(args.infile,usecols=feature2index['case_status'])
    train_status = np.loadtxt(args.infile,usecols=feature2index['train_status'])
    #filter out excluded rows (e.g. because of wrong sex)
    keep_rows = np.where(case_status>=0)[0]
    full_data = full_data[keep_rows,:]
    train_status = train_status[keep_rows]
    case_status = case_status[keep_rows]
    logging.info('Training and test data read in successfully.')
    
    #keep only training data and standardize
    y_train = case_status[np.where(train_status>0)[0]]
    #print("y_train:")
    #print(y_train)
    #print(y_train[0])
    #print(type(y_train[0]))
    #print(features.index('case_status'))
    #print("Min y_train value="+str(np.min(y_train)))
    #print("Max y_train value="+str(np.max(y_train)))
    print(np.where(train_status>0))
    X_train = full_data[np.where(train_status>0)[0],:]
    #print(np.where(full_data[:,features.index('train_status')]>0)[0])
    #print(features)
    #print(features.index('case_status'))
    #print("full_data shape: "+str(full_data.shape))
    #print(X_train.shape)
    #print(y_train.shape)

    #impute missing values as mean of the column
    imp = SimpleImputer(missing_values=np.nan,strategy='mean')
    X_train = imp.fit_transform(X_train)
    #save the fitted SimpleImputer to file for use in scoring with the model
    with open(args.outdir+"imputer.pkl",'wb') as outfile: pickle.dump(imp,outfile)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    #save the fitted StandardScaler to file for use in scoring with the model
    with open(args.outdir+"scaler.pkl",'wb') as outfile: pickle.dump(scaler,outfile)
    
    logging.info("Division to train and test sets as well as standardization of features done successfully.")

    print(X_train.shape)
    print(y_train.shape)
    print("Min X_train value="+str(np.min(X_train)))
    print("Max X_train value="+str(np.max(X_train)))
    print("Min y_train value="+str(np.min(y_train)))
    print("Max y_train value="+str(np.max(y_train)))
    #fitting the model
    logreg = LogisticRegression(random_state=args.seed,solver='saga',class_weight='balanced')
    grid = GridSearchCV(logreg,hyper_grid,cv=args.cv,n_jobs=args.nproc,refit=True,scoring=args.scoring,verbose=100,error_score='raise')
    grid.fit(X_train,y_train)
    logging.info("Model fitting done.")

    #get coefficients from the decision function of the best model and compute p-values for them
    #the meaning of p-values is not clear when using regularization
    #pvals = logit_pvalue(grid.best_estimator_,X_train)
    with open(args.outdir+"best_model_coefficients.txt",'wt') as outfile:
        w = csv.writer(outfile,delimiter='\t')
        w.writerow(['feature_name','coefficient'])
        w.writerow(['intercept',grid.best_estimator_.intercept_[0]])
        print("len(features)="+str(len(features)))
        print("coef shape:"+str(grid.best_estimator_.coef_.shape))
        for i in range(len(features)):
            #print(features[i])
            #print(grid.best_estimator_.coef_[0,i])
            w.writerow([features[i],grid.best_estimator_.coef_[0,i]])
    
    #save the cross-validation results to a file (this is a dictionary)
    with open(args.outdir+"cv_results.pkl",'wb') as outfile: pickle.dump(grid.cv_results_,outfile)
    #save the best hyperparameter combination and the best model score to file
    with open(args.outdir+"best_model_hyperparameters.txt",'wt') as outfile:
        outfile.write("Best "+args.scoring+": "+str(grid.best_score_)+"\n")
        outfile.write("Best model hyperparameters:\n")
        outfile.write(str(grid.best_params_))
    #save the best model, refit on the whole training dataset, to a file
    with open(args.outdir+"best_model.pkl",'wb') as outfile: pickle.dump(grid.best_estimator_,outfile)
    logging.info("Best model and its parameters saved.")
    
    logging.info("Program successfully terminated.")

def logit_pvalue(model, x):
    #This function is taken from https://stackoverflow.com/questions/25122999/scikit-learn-how-to-check-coefficients-significance
    #from the answer by user David Dale.
    #It returns the p-values as the significance of the Wald test.
    #Calculate z-scores for scikit-learn LogisticRegression.
    #parameters:
    #    model: fitted sklearn.linear_model.LogisticRegression with intercept
    #    x:     matrix on which the model was fit
    #This function uses asymtptics for maximum likelihood estimates.
    
    p = model.predict_proba(x)
    n = len(p)
    m = len(model.coef_[0]) + 1
    coefs = np.concatenate([model.intercept_, model.coef_[0]])
    x_full = np.matrix(np.insert(np.array(x), 0, 1, axis = 1))
    ans = np.zeros((m, m))
    for i in range(n):
        ans = ans + np.dot(np.transpose(x_full[i, :]), x_full[i, :]) * p[i,1] * p[i, 0]
    vcov = np.linalg.inv(np.matrix(ans))
    se = np.sqrt(np.diag(vcov))
    t =  coefs/se  
    p = (1 - norm.cdf(abs(t))) * 2
    return p
    
fitLogReg()
