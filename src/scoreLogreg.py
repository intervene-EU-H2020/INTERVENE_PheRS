#!/usr/bin/env python
import argparse
import logging
import csv
import gzip
import pickle
import sys

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score,roc_auc_score,roc_curve,precision_recall_curve,ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler

def scoreLogreg():

    parser = argparse.ArgumentParser()

    #Command line arguments:
    parser.add_argument("--infile",help="Full path to an INTERVENE phecode file. If last two letters of file name are gz, gzipped file is assumed.",type=str,default=None)
    parser.add_argument("--outdir",help="Output directory (must exist, default=./).",type=str,default="./")
    parser.add_argument("--scaler",help="Scaler model fit on the training data in pickle format.",type=str,default=None)
    parser.add_argument("--imputer",help="Imputer model fit on the training data in pickle format.",type=str,default=None)
    parser.add_argument("--excludevars",help="Full path to a file containing variable names that should be excluded from the model (default=nothing).",type=str,default=None)
    parser.add_argument("--model",help="Pre-fit sklearn model in pickle-format.",type=str,default=None)
    parser.add_argument("--seed",help="Random number generator seed (default=42).",type=int,default=42)
    parser.add_argument("--nproc",help="Number of parallel processes used (default=1).",type=int,default=1)

    args = parser.parse_args()

    #save the command used to evoke this script into a file
    logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', filename=args.outdir+'scoreLogreg.log',level=logging.INFO,filemode='w')
    logging.info("The command used to evoke this script:")
    logging.info(" ".join(sys.argv))

    #Read in the scaler, the imputer and the model
    with open(args.scaler,'rb') as infile: scaler = pickle.load(infile)
    with open(args.imputer,'rb') as infile: imp = pickle.load(infile)
    with open(args.model,'rb') as infile: model = pickle.load(infile)
    logging.info("Scaler and model successfully loaded.")

    #read in names of variables that should be excluded from the model
    with open(args.excludevars,'rt') as infile:
        r = csv.reader(infile,delimiter='\t')
        excludevars = set(['ID','date_of_birth','end_of_follow_up'])#+['PC'+str(i) for i in range(1,11)])
        for row in r: excludevars.add(row[0])
    logging.info("Names of excluded variables read in successfully.")
    
    #read in the training and test data                                             #first read in the header
    if args.infile[-2:]=='gz': in_handle = gzip.open(args.infile,'rt',encoding='utf-8')
    else: in_handle = open(args.infile,'rt',encoding='utf-8')
    with in_handle as infile:
        r = csv.reader(infile,delimiter='\t')
        for row in r:
            if row[0]=='#ID':
                feature2index = {row[0][1:]:0} #key = feature name, value = index in input file
                for i in range(1,len(row)): feature2index[row[i]] = i
                break
    #then read in the data
    
    usecols = []
    features = []
    excludevars.add('case_status')
    excludevars.add('train_status')
    for key in feature2index.keys():
        if key not in excludevars:
            usecols.append(key)
            features.append((feature2index[key],key))
    features = [f[1] for f in sorted(features)] #this list now contains the names of the features in the same order as the columns are in full_data
    full_data = pd.read_csv(args.infile,delimiter='\t')
    #case_status = np.loadtxt(args.infile,usecols=feature2index['case_status'])
    #train_status = np.loadtxt(args.infile,usecols=feature2index['train_status'])
    #filter out excluded rows (e.g. because of wrong sex)
    full_data = full_data.loc[full_data['case_status']>=0]
    #filter out training samples
    full_data = full_data.loc[full_data['train_status']==0]
    #print(full_data.sort_values(by='#ID'))
    #print(full_data['train_status'].value_counts())
    #train_status = full_data['train_status']
    case_status = full_data['case_status']
    #get the test set IDs
    IDs = list(full_data['#ID'])#np.genfromtxt(args.infile,usecols=[0],dtype=str)

    logging.info('Training and test data read in successfully.')

    #keep only test data, impute missing values and standardize
    y_test = case_status.values
    #X_test = full_data.loc[full_data['train_status']<1]
    X_test = full_data[usecols]
    #y_test = full_data[np.where(full_data[:,features.index('train_status')]<1)[0][0],features.index('case_status')]
    #X_test = full_data[np.where(full_data[:,features.index('train_status')]<1)[0][0],features.index('train_status')+1:]
    #imputation and scaling
    X_test = imp.transform(X_test)
    X_test = scaler.transform(X_test)
    logging.info("Division to train and test sets as well as imputation and standardization of features done successfully.")

    #predict using the loaded model
    y_pred = model.predict_proba(X_test)
    y_pred_labels = model.predict(X_test)
    #print('y_pred')
    #print(y_pred)
    #print(y_pred.shape)
    #print('y_test')
    #print(y_test)
    #print(y_test.shape)
    logging.info("Labels for the test set predicted successfully.")

    #save predictions to a file
    with gzip.open(args.outdir+"pred_probas.txt.gz",'wt') as outfile:
        w = csv.writer(outfile,delimiter='\t')
        w.writerow(["#ID","pred_class1_prob","true_class"])
        
        for i in range(0,len(y_test)): w.writerow([IDs[i],y_pred[i,np.where(model.classes_==1)][0][0],y_test[i]])
                                                  
    #precision-recall curve and average precision score
    #print("y_pred shape:")
    #print(y_pred[:,np.where(model.classes_==1)].shape)
    auprc = average_precision_score(y_test,y_pred[:,np.where(model.classes_==1)].flatten())
    precision,recall,thresholds = precision_recall_curve(y_test,y_pred[:,np.where(model.classes_==1)].flatten())
    #print(precision.shape)
    #print(recall.shape)
    #print(thresholds.shape)
    print("auPRC="+str(round(auprc,3)))
    rand_AUprc = round(np.sum(y_test)/len(y_test),3)
    plt.plot(np.linspace(0,1),rand_AUprc*np.ones(shape=(1,50)).flatten(),'--k',label="random, auPRC="+str(rand_AUprc))
    plt.plot(recall,precision,label="elastic net, AUprc="+str(round(auprc,3)))
    plt.xlabel("recall")
    plt.ylabel("precision")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.outdir+"precision_recall_curve.png",dpi=300)
    plt.clf()
    with gzip.open(args.outdir+"precision_recall_curve.txt.gz",'wt') as outfile:
        w = csv.writer(outfile,delimiter='\t')
        w.writerow(["#AUprc="+str(auprc)])
        w.writerow(["#recall","precision","threshold"])
        for i in range(len(precision)):
            if i==thresholds.shape[0]: w.writerow([recall[i],precision[i],0.0])
            else: w.writerow([recall[i],precision[i],thresholds[i]])
                                                                            
    #receiver operator characteristics curve and AUC
    auc = roc_auc_score(y_test,y_pred[:,np.where(model.classes_==1)].flatten())
    fpr,tpr,thresholds = roc_curve(y_test,y_pred[:,np.where(model.classes_==1)].flatten())
    print("AUC="+str(round(auc,3)))
    plt.plot(np.linspace(0,1),np.linspace(0,1),'--k',label="random, AUC=0.5")
    plt.plot(fpr,tpr,label="elastic net, AUC="+str(round(auc,3)))
    plt.xlabel("fpr")
    plt.ylabel("tpr")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.outdir+"roc_curve.png",dpi=300)
    plt.clf()
    with gzip.open(args.outdir+"roc_curve.txt.gz",'wt') as outfile:
        w = csv.writer(outfile,delimiter='\t')
        w.writerow(["#AUC="+str(auc)])
        w.writerow(["#fpr","tpr","threshold"])
        for i in range(len(fpr)):
            if i==thresholds.shape[0]: w.writerow([fpr[i],tpr[i],0.0])
            else: w.writerow([fpr[i],tpr[i],thresholds[i]])

    #confusion matrix
    #print(y_pred_labels)
    #print(y_pred_labels.shape)
    ConfusionMatrixDisplay.from_predictions(y_test,y_pred_labels)
    plt.savefig(args.outdir+"confusion_matrix.png",dpi=300)
    plt.clf()
    
    logging.info("Prediction metrics computed and saved successfully.")

    
scoreLogreg()    
    

