#!/usr/bin/env python                                                                                                                                                                       
import argparse
import logging
import csv
import gzip
import pickle

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
    parser.add_argument("--excludevars",help="Full path to a file containing variable names that should be excluded from the model (default=nothing).",type=str,default=None)
    parser.add_adgument("--model",help="Pre-fit sklearn model in picke-format.",type=str,default=None)
    parser.add_argument("--seed",help="Random number generator seed (default=42).",type=int,default=42)
    parser.add_argument("--nproc",help="Number of parallel processes used (default=1).",type=int,default=1)

    args = parser.parse_args()

    #save the command used to evoke this script into a file
    logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', filename=args.outdir+'scoreLogreg.log')
    logging.info("The command used to evoke this script:")
    logging.info(" ".join(sys.argv))

    #Read in the scaler and the model
    with open(args.scaler,'rb') as infile: scaler = pickle.load(infile)
    with open(args.model,'rb') as infile: model = pickle.load(infile)
    logging.info("Scaler and model successfully loaded.")

    #read in names of variables that should be excluded from the model                                                                                                                 
    with open(args.excludevars,'rt') as infile:
        r = csv.reader(infile,delimiter='\t')
        excludevars = set()
        for row in r: excludevars.add(row[0])
    logging.info("Names of excluded variables read in successfully.")
    
    #read in the training and test data                                                                                                                                                     
    #first read in the header
    if args.infile[:-2]=='gz': in_handle = gzip.open(args.infile,'rt')
    else: in_handle = open(args.infile,'rt')
    with in_handle as infile:
	r = csv.reader(infile,delimiter='\t')
        for row in r:
            if row[0]=='#follow_up_time':
                feature2index = {row[0][1:]:0} #key = feature name, value = index in input file                                                                                             
                for i in range(1,len(row)): feature2index[row[i]] = i
	        break
    #then read in the training data                                                                                                                                                         
    usecols = []
    features = []
    for key in feature2index.keys():
        if key not in exludevars:
            usecols.append(feature2index[key])
            features.append((feature2index[key],key))
    features = [f[1] for f in sorted(features)] #this list now contains the names of the features in the same order as the columns are in full_data                                         
    full_data = np.loadtxt(args.infile,usecols=usecols)
    logging.info('Training and test data read in successfully.')

    #keep only test data and standardize                                                                                                                                               
    y_test = full_data[np.where(full_data[:,features.index('train_status')]<1)[0][0],features.index('case_status')]
    X_test = full_data[np.where(full_data[:,features.index('train_status')]<1)[0][0],features.index('train_status')+1:]
    X_test = scaler.transform(X_test)
    logging.info("Division to train and test sets as well as standardization of features done successfully.")

    #predict using the loaded model
    y_pred = model.predict_proba(X_test)
    logging.info("Labels for the test set predicted successfully.")

    #save predictions to a file
    with gzip.open(args.outdir+"pred_probas.txt.gz",'wt') as outfile:
        w = csv.writer(outfile,delimiter='\t')
        w.writerow(["#pred_class1_prob","true_class"])
        for i in range(0,len(y_test)): w.writerow([y_pred[i,model.classes_.index("1")],y_test[i]])
                                                  
    #precision-recall curve and average precision score
    auprc = average_precision_score(y_test,y_pred)
    precision,recall,thresholds = precision_recall_curve(y_test,y_pred)
    plt.plot(recall,precision,label="auPRC="+str(round(auprc,3)))
    plt.xlabel("recall")
    plt.ylabel("precision")
    plt.tight_layout()
    plt.savefig(args.outdir+"precision_recall_curve.pdf",dpi=300)
    plt.clf()
    with gzip.open(args.outdir+"precision_recall_curve.txt.gz",'wt') as outfile:
        w = csv.writer(outfile,delimiter='\t')
        w.writerow(["#AUprc="+str(auprc)])
        w.writerow(["#recall","precision"])
        for i in range(len(precision)): w.writerow([recall[i],precision[i]])
                                                                            
    #receiver operator cahracteristics curve and AUC
    auc = roc_auc_score(y_test_y_pred)
    fpr,tpr,thresholds = roc_curve(y_test,y_pred)
    plt.plot(fpr,tpr,label="AUC="+str(round(auc,3)))
    plt.xlabel("fpr")
    plt.ylabel("tpr")
    plt.tight_layout()
    plt.savefig(args.outdir+"roc_curve.pdf",dpi=300)
    plt.clf()
    with gzip.open(args.outdir+"roc_curve.txt.gz",'wt') as outfile:
        w = csv.writer(outfile,delimiter='\t')
        w.writerow(["#AUC="+str(auc)])
        w.writerow(["#fpr","tpr"])
        for i in range(len(fpr)): w.writerow([fpr[i],tpr[i]])

    #confusion matrix
    ConfusionMatrixDisplay.from_predictions(y_test,y_pred)
    plt.savefig(args.outdir+"confusion_matrix.pdf",dpi=300)
    plt.clf()
    
    logging.info("Prediction metrics computed and saved successfully.")

    
scoreLogreg()    
    

