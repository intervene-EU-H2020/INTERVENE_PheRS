#!/usr/bin/env python

import sys
import argparse
import logging
import os



def PheRSwrapper():

    #Command line arguments:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ICDfile",help="Full path to an INTERVENE longitudinal ICD code file. If last two letters of file name are gz, gzipped file is assumed.",type=str,default=None)
    parser.add_argument("--phenotypefile",help="Full path to an INTERVENE phenotype file. If last two letters of file name are gz, gzipped file is assumed",type=str,default=None)
    parser.add_argument("--includevars",help="List of variable names to include from the phenotype file (default=nothing).",type=str,nargs='+',default=[])
    parser.add_argument("--phecodefile",help="Phecode definition file (NOTE: comma-separated).",type=str,default=None)
    parser.add_argument("--ICD9tophecodefile",help="Full path to the ICD9 to phecode map (NOTE: comma-separated).",type=str,default=None)
    parser.add_argument("--ICD10tophecodefile",help="Full path to the ICD10 to phecode map (NOTE: comma-separated).",type=str,default=None)
    parser.add_argument("--ICD10CMtophecodefile",help="Full path to the ICD10CM to phecode map (NOTE: comma-separated).",type=str,default=None)
    parser.add_argument("--targetphenotype",help="Name of the target phenotype.",type=str,default=None)
    parser.add_argument("--excludephecodes",help="Path to a file containing the Phecode(s) corresponding to and related to the target phenotype that should be excluded from the analysis.",type=str,default=None)
    parser.add_argument("--outdir",help="Output directory (must exist, default=./).",type=str,default="./")
    parser.add_argument("--testfraction",help="Fraction of IDs used in test set (default=0.15). Sampling at random.",type=str,default='0.15')
    parser.add_argument("--testidfile",help="Full path to a file containing the IDs used in the test set. If given, overrides --testfraction.",type=str,default=None)
    parser.add_argument("--washout_window",help="Start and end dates for washout (default= 2009-01-01 2010-12-31).",type=str,nargs=2,default=['2009-01-01','2010-12-31'])
    parser.add_argument('--exposure_window',help='Start and end dates for exposure (default = 1999-01-01 2008-12-31).',type=str,nargs=2,default=['1999-01-01','2008-12-31'])
    parser.add_argument('--observation_window',help='Start and end dates for observation (default= 2011-01-01 2019-01-01).',type=str,nargs=2,default=['2011-01-01','2019-01-01'])
    parser.add_argument("--minage",help="Minimum age at the start of the observation period to include (default=32).",type=float,default=32)
    parser.add_argument("--maxage",help="Maximum age at the start of the observation period to include (default=100).",type=float,default=100)
    parser.add_argument("--seed",help="Random number generator seed (default=42).",type=str,default='42')
    parser.add_argument("--dateformatstr",help="Date format used (default='%%Y-%%m-%%d')",type=str,default='%Y-%m-%d')
    parser.add_argument("--excludeICDfile",help="Full path to a file containing the ICD codes to be excluded. Format per row: ICD_version, ICD_code.",type=str,default=None)
    parser.add_argument("--missing",help="Fraction of missing values allowed for a feature to be included into the analysis (default=0.5).",type=str,default='0.5')
    parser.add_argument("--frequency",help="Minimum frequency for a predictor to be included into the analysis (default=0.01).",type=str,default='0.01')
    parser.add_argument("--controlfraction",help="How many controls to use, number of cases multiplied by this flag value (default=None, meaning all controls are used).",type=str,default=None)
    parser.add_argument("--nproc",help="Number of parallel processes used (default=1).",type=str,default='1')
    parser.add_argument("--paramgridfile",help="File containing the parameter grid explored using cross-validation. One parameter per row, first value is key, rest are possible values.",type=str,default=None)
    parser.add_argument("--ICD_levels",help="Level of ICD-codes to use. 1=primary, 2=primary+secondary.",type=int,default=1)
    parser.add_argument("--secondary_ICD_weight",help="How much secondary ICDs should weigh compared to primary",type=float,default=1)
    parser.add_argument("--src_dir",help="Place of the scripts.",type=str,default=None)
    parser.add_argument("--run_prep",help="Whether to run the preprocessing. Default = 1",type=int,default=1)
    parser.add_argument("--run_fit",help="Whether to run the fitting. Default = 1",type=int,default=1)
    parser.add_argument("--run_score",help="Whether to run the scoring. Default = 1",type=int,default=1)

    args = parser.parse_args()
    print("starting: " + args.targetphenotype)

    outdir_i = args.outdir+'exposure='+args.exposure_window[0]+"-"+args.exposure_window[1]+"-washoutend="+args.washout_window[1]+"-observationend="+args.observation_window[1]+'/'
    # Create director if it does not exist
    if not os.path.exists(outdir_i):
        cmd = 'mkdir -p '+outdir_i
        os.system(cmd)
        logging.info(cmd + "\n")

    #save the command used to evoke this script into a file
    logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', filename=args.outdir+'PheRSwrapper.log',level=logging.INFO,filemode='w')
    logging.info("The command used to evoke this script:")
    logging.info(" ".join(sys.argv))
    

    ######## Preprocessing ########
    includevars = ''
    for i in args.includevars: includevars += ' '+i
    
    preprocess_cmd = 'python3 ' + args.src_dir + 'PheRS_preprocess.py --ICDfile '+ args.ICDfile +' --phenotypefile '+args.phenotypefile+' --phecodefile '+args.phecodefile+' --ICD9tophecodefile '+args.ICD9tophecodefile+' --ICD10tophecodefile '+args.ICD10tophecodefile+' --ICD10CMtophecodefile '+args.ICD10CMtophecodefile+' --targetphenotype '+args.targetphenotype+' --excludephecodes '+args.excludephecodes+' --outdir '+outdir_i+' --washout_window '+args.washout_window[0]+' '+args.washout_window[1]+' --exposure_window '+args.exposure_window[0]+' '+args.exposure_window[1]+' --observation_window '+args.observation_window[0]+' '+args.observation_window[1]+' --seed '+args.seed+' --nproc '+args.nproc+' --testfraction '+args.testfraction+' --frequency '+args.frequency+' --maxage '+str(args.maxage)+' --minage '+str(args.minage) + ' --ICD_levels '+str(args.ICD_levels) + ' --secondary_ICD_weight '+str(args.secondary_ICD_weight)
    
    if args.controlfraction!=None: preprocess_cmd += ' --controlfraction '+args.controlfraction
    if args.excludeICDfile!=None: preprocess_cmd += ' --excludeICDfile '+args.excludeICDfile
    if args.testidfile!=None: preprocess_cmd += ' --testidfile '+args.testidfile
    if len(includevars)>0: preprocess_cmd += ' --includevars '+includevars

    if args.run_prep == 1:
        print("Preprocessing....")
        logging.info(preprocess_cmd)
        os.system(preprocess_cmd)
        print("Preprocessing....done.")

    in_file = outdir_i+'target-'+args.targetphenotype+"-PheRS-ML-input.txt.gz"
    exclude_file = outdir_i+'target-'+args.targetphenotype+"-excluded-phecodes.txt"

    ######## Fitting ########
    fit_cmd = 'python3 ' + args.src_dir + 'fitLogreg.py --infile '+ in_file + " --excludevars "+ exclude_file + ' --paramgridfile ' + args.paramgridfile + ' --outdir '+ outdir_i + ' --nproc '+ args.nproc + ' --scoring neg_log_loss'

    if args.run_fit == 1:
        print("Fitting....")
        logging.info(fit_cmd)
        os.system(fit_cmd)
        print("Fitting....done.")

    ######## Scoring ########
    score_cmd = 'python3 ' + args.src_dir + 'scoreLogreg.py --infile '+ in_file + ' --outdir '+ outdir_i + ' --scaler ' + outdir_i + 'scaler.pkl --imputer ' + outdir_i + 'imputer.pkl --excludevars ' + exclude_file + ' --model ' + outdir_i + 'best_model.pkl' + " --nproc 1"

    if args.run_score == 1:
        print("Scoring....")
        logging.info(score_cmd)
        os.system(score_cmd)
        print("Scoring....done.")
        
PheRSwrapper()
