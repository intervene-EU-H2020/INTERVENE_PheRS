#!/usr/bin/env python

import argparse
import logging
import csv
import gzip
import pickle
import sys

import pandas as pd
import numpy as np

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import average_precision_score,roc_auc_score,roc_curve,precision_recall_curve
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from scipy.stats import norm

def fitGradBoost():

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
            if key=='learning_rate':
                #min, max and step size
                hyper_grid[key] = []
                min_lr = float(row[1])
                max_lr = float(row[2])
                step = float(row[3])
                for lr in np.arange(min_lr,max_lr,step): hyper_grid[key].append(lr)
            elif key=='max_iter':
                min_max_iter = int(row[1])
                max_max_iter = int(row[2])
                step = float(row[3])
                hyper_grid[key] = []
                for max_iter in range(min_max_iter,max_max_iter,step): hyper_grid[key].append(max_iter)
            elif key=='l2_regularization':
                min_l2 = float(row[1])
                max_l2 = float(row[2])
                step = float(row[3])
                hyper_grid[key] = []
                for l2 in np.arange(min_l2,max_l2,step): hyper_grid[key].append(l2)
    logging.info("Hyperparamter grid read in successfully.")

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
    excludevars.add('case_status')
    excludevars.add('train_status')
    for key in feature2index.keys():
        if key not in excludevars:
            usecols.append(key)
            features.append((feature2index[key],key))
    features = [f[1] for f in sorted(features)] #this list now contains the names of the features in the same order as the columns are in full_data
    print(usecols)
    full_data = pd.read_csv(args.infile,delimiter='\t')
    case_status = full_data['case_status']
    train_status = full_data['train_status']
    #filter out excluded rows (e.g. because of wrong sex)
    full_data = full_data.loc[full_data['case_status']>=0]
    train_status = train_status.loc[full_data['case_status']>=0]
    case_status = case_status.loc[full_data['case_status']>=0]
    logging.info('Training and test data read in successfully.')
    
    #keep only training data and standardize
    y_train = case_status.loc[full_data['train_status']>0]
    
    X_train = full_data.loc[full_data['train_status']>0]
    X_train = X_train[usecols]

    #NOTE! No imputation is needed as the gradient booster learns rules for missing values

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
    booster = HistGradientBoostingClassifier(random_state=args.seed)
    grid = GridSearchCV(booster,hyper_grid,cv=args.cv,n_jobs=args.nproc,refit=True,scoring=args.scoring,verbose=100,error_score='raise')
    grid.fit(X_train,y_train)
    logging.info("Model fitting done.")

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

fitGradBoost()
