#!/usr/bin/env python

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import sys
import argparse
import logging
import csv
import gzip
import datetime
import random
import multiprocessing as mp
from operator import itemgetter
import numpy as np
from scipy.sparse import csc_matrix
from dateutil.relativedelta import relativedelta


def parser_args():
    parser = argparse.ArgumentParser()
    #Command line arguments:
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
    parser.add_argument("--testfraction",help="Fraction of IDs used in test set (default=0.15). Sampling at random.",type=float,default=0.15)
    parser.add_argument("--testidfile",help="Full path to a file containing the IDs used in the test set. If given, overrides --testfraction.",type=str,default=None)
    parser.add_argument("--washout_window",help="Start and end dates for washout (default= 2010-01-01 2011-12-31).",type=str,nargs=2,default=['2010-01-01','2011-12-31'])
    parser.add_argument('--exposure_window',help='Start and end dates for exposure (default = 2000-01-01 2009-12-31).',type=str,nargs=2,default=['2000-01-01','2009-12-31'])
    parser.add_argument('--observation_window',help='Start and end dates for observation (default= 2012-01-01 2020-01-01).',type=str,nargs=2,default=['2012-01-01','2020-01-01'])
    parser.add_argument("--minage",help="Minimum age at the start of the observation period to include (default=32).",type=float,default=32)
    parser.add_argument("--maxage",help="Maximum age at the start of the observation period to include (default=100).",type=float,default=100)
    parser.add_argument("--seed",help="Random number generator seed (default=42).",type=int,default=42)
    parser.add_argument("--dateformatstr",help="Date format used (default='%Y-%m-%d')",type=str,default='%Y-%m-%d')
    parser.add_argument("--excludeICDfile",help="Full path to a file containing the ICD codes to be excluded. Format per row: ICD_version, ICD_code.",type=str,default=None)
    parser.add_argument("--missing",help="Fraction of missing values allowed for a feature to be included into the analysis (default=0.5).",type=float,default=0.5)
    parser.add_argument("--frequency",help="Minimum frequency for a predictor to be included into the analysis (default=0.01).",type=float,default=0.01)
    parser.add_argument("--controlfraction",help="How many controls to use, number of cases multiplied by this flag value (default=None, meaning all controls are used).",type=float,default=None)
    parser.add_argument("--nproc",help="Number of parallel processes used (default=1).",type=int,default=1)
    parser.add_argument("--ICD_levels",help="Level of ICD-codes to use. 1=primary, 2=primary+secondary.",type=int,default=1)
    parser.add_argument("--secondary_ICD_weight",help="How much secondary ICDs should weigh compared to primary",type=float,default=1)

    args = parser.parse_args()

    return(args)

def read_ICD2phecode_maps(args, logging):
    #read in the phecode definitions
    phecodefiles = [args.ICD9tophecodefile, args.ICD10tophecodefile,args.ICD10CMtophecodefile]
    phecode_versions = ["9","10","10CM"]
    ICD2phecode = {} #key = phecode_version, value = {ICD1:phecode1,...}
    ind = -1
    for crnt_phecode_file in phecodefiles:
        ind += 1
        ICD2phecode[phecode_versions[ind]] = {}
        if crnt_phecode_file is None: continue
        with open(crnt_phecode_file,'rt',encoding='latin-1') as infile:
            i = 1
            for row in infile:
                row = row.strip().split(',')
                icd = row[0].strip('"').strip("'").replace('.','') #remove . from the ICD codes
                phecode = row[1].strip('"').strip("'").split(".")[0]
                if len(phecode)>0:
                    #not saving mapping for ICD-codes that do not map to a phecode
                    if icd not in ICD2phecode[phecode_versions[ind]]: ICD2phecode[phecode_versions[ind]][icd] = set([phecode])
                    else: ICD2phecode[phecode_versions[ind]][icd].add(phecode)
                    i += 1
    logging.info("ICD to phecode maps read in successfully.\n")

    return(ICD2phecode)

def read_phecode_defs(args, excluded_phecodes, logging):
    phecodes = {} #key = phecode, value = {"phenotype":string,"phecode_exclude_range":[smallest,largest],"sex":string}
    with open(args.phecodefile,'rt') as infile:
        r = csv.reader(infile,delimiter=',')
        for row in r:
            if row[0]=='phecode':
                header = row
                continue
            phecodes[row[0]] = {}
            for i in range(1,len(row)):
                if i==2:
                    phecodes[row[0]][header[i]] = []
                    if len(row[i])>0:
                        #there can be multiple exclude ranges per phecode
                        range_list = row[i].split(',')
                        for r in range_list:
                            r = r.split("-")
                            phecodes[row[0]][header[i]].append([float(r[0]),float(r[1])])
                    else: phecodes[row[0]][header[i]].append([None,None])
                elif i>3: continue #rest of the entries are not needed
    logging.info("Phecode definitions read in successfully.\n")

    include_phecodes = set()
    for phecode in phecodes.keys():
        if phecode.split(".")[0] not in excluded_phecodes["orig"]:
            include_phecodes.add(phecode.split(".")[0])
    include_phecodes = list(include_phecodes)

    return(include_phecodes)

def read_exclude_phecodes(args):
    with open(args.excludephecodes,'rt') as infile:
        r = csv.reader(infile,delimiter=',')
        for row in r: excluded_phecodes = row
    excluded_phecodes = {"orig":set(excluded_phecodes)}
    return(excluded_phecodes)

def time_windows(args):
    #convert the dates to Python datetime objects
    washout_start = datetime.datetime.strptime(args.washout_window[0],args.dateformatstr)
    washout_end = datetime.datetime.strptime(args.washout_window[1],args.dateformatstr)
    washout = {"start": washout_start, "end": washout_end}

    exposure_start = datetime.datetime.strptime(args.exposure_window[0],args.dateformatstr)
    exposure_end = datetime.datetime.strptime(args.exposure_window[1],args.dateformatstr)
    exposure = {"start": exposure_start, "end": exposure_end}

    observation_start = datetime.datetime.strptime(args.observation_window[0],args.dateformatstr)
    observation_end = datetime.datetime.strptime(args.observation_window[1],args.dateformatstr)
    observation = {"start": observation_start, "end": observation_end}

    return washout, exposure, observation

def read_phenotype_file(args,
                        phecodelist,
                        washout,
                        exposure,
                        observation,
                        logging):

    #read in the phenotype file and initialize the data dictionary containing the data needed for ML
    data = {}
    #here key = ID, value = list with following columns:
    #0: date_of_birth
    #1: end_of_followup - date for either the first occurrence of the target phenotype or death
    #2: sex - female=1, male=0, other=-1
    #3: case_status - case=1, control=0, -1=exclude
    #4: train_status - train=1, test=0
    #5-1872: one column per each phecode - Column identifier is the phecode name (e.g. 008), coded as 1 (=present) or 0 (=absent). Phecode names come from: https://phewascatalog.org/files/phecode_definitions1.2.csv.zip
    #1873-: additional features used from the phenotype file and defined with input flag --includevars
    descriptive_features = ["date_of_birth","end_of_follow_up","sex","case_status","train_status"]
    feature_names = descriptive_features+phecodelist+args.includevars

    if args.phenotypefile[-2:]=='gz': in_handle = gzip.open(args.phenotypefile,'rt',encoding='utf-8')
    else: in_handle = open(args.phenotypefile,'rt',encoding='utf-8')

    N_cases = 0 #number of cases
    N_excluded = 0
    first_row = 1
    with in_handle as infile:
        for row in infile:
            row = row.strip().split('\t')
            if first_row == 1:
                first_row = 0
                header = row
                target_column = header.index(args.targetphenotype)
                target_time_column = header.index(args.targetphenotype+"_DATE")
                continue
            ID = row[0]
            date_of_birth = datetime.datetime.strptime(row[2],args.dateformatstr)
            end_of_followup = datetime.datetime.strptime(row[header.index("END_OF_FOLLOWUP")], args.dateformatstr) #end of follow-up in the biobank
            data[ID] = [0 for i in range(len(phecodelist)+len(descriptive_features)+len(args.includevars))] #initialize the row corresponding to ID
            data[ID][0] = date_of_birth
            age_at_obs_start = (observation["start"]-date_of_birth).days/365.25
            #if age is requested as a predictor calculate age at the start of the observation window
            if 'age' in args.includevars: data[ID][feature_names.index('age')] = age_at_obs_start
            
            # exclude if
            #0) the person's end of follow-up is before the end of the washout period
            #1) the person is born after the start of the exposure period
            #2) the person is younger than args.minage at the start of the observation period
            #3) the person is older than args.maxage at the start of the observation period
            if (data[ID][feature_names.index('date_of_birth')]>exposure["start"]) or \
                    (age_at_obs_start<args.minage) or \
                     (age_at_obs_start>=args.maxage) or \
                     (end_of_followup<washout["end"]):
                data[ID][feature_names.index('case_status')] = -1
                N_excluded += 1
                continue

            #Now diagnoses are collected only from the time period:
            #start of observation -> end of observation
            #In more detail:
            #0) If the person receives the diagnosis after their end of follow-up -> EXCLUDE
            #1) If the person receives the diagnosis during observation period -> CASE
            #2) If the person never receives the diagnosis -> CONTROL
            #3) If the person receives the diagnosis before the end of the "exposure window" -> EXCLUDE from analysis
            #4) If the person receives the diagnosis during the "washout" period -> EXCLUDE from analysis
            #5) If the person receives the diagnosis after the observation period -> CONTROL
            data[ID][feature_names.index('case_status')] = row[target_column]#case/control status
            # We do not exclude cases or controls as defined in the target phenotypes except for prostate and breast cancer where excludes are opposite sex
            # NAs are control exclusions -> we want as controls
            if not (data[ID][feature_names.index('case_status')] == 'NA' and args.targetphenotype in ["C3_PROSTATE", "C3_BREAST", "T2D"]):
                if data[ID][feature_names.index('case_status')] == 'NA': data[ID][feature_names.index('case_status')] = 0
                data[ID][feature_names.index('case_status')] = int(data[ID][feature_names.index('case_status')])
                ####### CASES
                if data[ID][feature_names.index('case_status')]>0:
                    case_date = datetime.datetime.strptime(row[target_time_column],args.dateformatstr)
                    data[ID][feature_names.index('end_of_follow_up')] = case_date
                    if (data[ID][feature_names.index('end_of_follow_up')]>=observation["start"]) and (data[ID][feature_names.index('end_of_follow_up')]<=observation["end"]):
                        #The diagnosis was received during the observation period
                        N_cases += 1
                    elif case_date>end_of_followup:
                        #diagnosis is marked to have happened after the end of follow-up for this ID, which does not make sense, thus we exclude
                        data[ID][feature_names.index('case_status')] = -1
                    else:
                        #diagnosis was outside the observation period
                        #if diagnosis happens after the end of observation period, we have a control
                        #otherwise, the individual is excluded
                        if data[ID][feature_names.index('end_of_follow_up')]>observation["end"]: data[ID][feature_names.index('case_status')] = 0
                        else: data[ID][feature_names.index('case_status')] = -1
                ######## CONTROLS
                else:
                    data[ID][feature_names.index('end_of_follow_up')] = end_of_followup
                    if data[ID][feature_names.index('end_of_follow_up')]<=observation["end"]:
                        #follow-up ending before observation period end
                        data[ID][feature_names.index('case_status')] = -1
                    else:
                        data[ID][feature_names.index('case_status')] = 0
            else:
                data[ID][feature_names.index('case_status')] = -1 #mark excluded if variable value is NA
                data[ID][feature_names.index('end_of_follow_up')] = end_of_followup#End of follow-up is a separate column in the phenotype file

            # Settin sex            
            sex = row[1]
            if row[1]=='female': sex = 1
            elif row[1]=='male': sex = 0
            elif row[1]=='other': sex = -1
            else: sex = row[1]
            data[ID][2] = sex #sex

            for var in args.includevars:
                if var=='age': continue
                mpf_index = header.index(var)
                out_index = feature_names.index(var)
                
                value = row[mpf_index].strip()
                if value=='NA' or value=='-': data[ID][out_index] = np.nan
                else: data[ID][out_index] = float(value) #additional features used
    logging.info("Phenotype file read in successfully.\n")

    print("\nTotal number of IDs="+str(len(list(data.keys()))))
    logging.info("Total number of IDs="+str(len(list(data.keys()))))
    print("Number of cases="+str(N_cases))
    logging.info("Number of cases="+str(N_cases))
    print("Number of excluded IDs="+str(len([ID for ID in data if data[ID][feature_names.index('case_status')]==-1])))
    logging.info("Number of excluded IDs="+str(len([ID for ID in data if data[ID][feature_names.index('case_status')]==-1])))
                 
    return data, feature_names, descriptive_features, N_cases

def downsample_controls(feature_names, data, args, N_cases, logging):
    #Then draw args.controlfraction*N_cases controls to use
    #remove other controls and all excluded IDs (where case_status=-1)
    #get a list of all control IDs
    control_IDs = [ID for ID in data if data[ID][feature_names.index('case_status')]==0]
    print("Number of control IDs before sampling="+str(len(control_IDs)))
    logging.info("Number of control IDs before sampling="+str(len(control_IDs)))
                 
    if args.controlfraction is not None:
        #sample final controls
        random.seed(args.seed)
        final_control_IDs = random.sample(control_IDs,int(args.controlfraction*N_cases))
    else: final_control_IDs = control_IDs
    print("Number of control IDs="+str(len(final_control_IDs)))
    logging.info("Number of control IDs before sampling="+str(len(final_control_IDs)))

    excl_control_IDs = set(control_IDs)-set(final_control_IDs)
    print(len(excl_control_IDs))
    for ID in excl_control_IDs: data[ID][feature_names.index('train_status')] = -1

    case_IDs = [ID for ID in data if data[ID][feature_names.index('case_status')]==1]
    print("Number of case IDs="+str(len(case_IDs)))
    logging.info("Number of case IDs="+str(len(case_IDs)))

    return data
    

def read_phecode_diags_from_icd_file(args, data, ICD2phecode, exposure, phecodelist, descriptive_features, logging):
    #read in the INTERVENE ICD code file and convert to phecode format
    #calculate basic statistics from the ICD code file
    if args.ICDfile[-2:]=='gz': in_handle = gzip.open(args.ICDfile,'rt',encoding='latin-1')
    else: in_handle = open(args.ICDfile,'rt',encoding='latin-1')

    #ICD codes not found from phecode files
    with in_handle as infile:
        r = csv.reader(infile,delimiter='\t')
        for row in r:
            if row[0]=='#ID': continue
            ID = row[0]
            #note that only the IDs that are present in the phenotype file are used
            if ID not in data: continue
            event_age = float(row[1])
            #only include ICD codes that occur within the exposure window
            exp_start_age = (exposure["start"]-data[ID][0]).total_seconds()/(365.0*24*60*60)
            exposure_end_age = (exposure["end"]-data[ID][0]).total_seconds()/(365.0*24*60*60)
            if event_age>=exp_start_age and event_age<=exposure_end_age:
                ICD_version = row[2]
                ICD_code = row[3].strip('"').strip("'").replace('.','')
                code_level = row[4]
                phecodes = set()
                #add occurrence of primary ICD code first check if we have an exact match in the ICD to phecode mapping
                if ICD_code in ICD2phecode[ICD_version]: phecodes = ICD2phecode[ICD_version][ICD_code]
                else:
                    # try .0 -> often a problem in EstB
                    ICD_code_longer = ICD_code+"0"
                    if ICD_code_longer in ICD2phecode[ICD_version]: phecodes = ICD2phecode[ICD_version][ICD_code_longer]
                    else:
                        for trunc_len in range(1,5):
                            if trunc_len > len(ICD_code): break
                            ICD_code_truncated = ICD_code[1:len(ICD_code)-trunc_len]
                            if ICD_code_truncated in ICD2phecode[ICD_version]:
                                phecodes = ICD2phecode[ICD_version][ICD_code_truncated]
                                break
                #add the occurrence of the primary phecode
                if len(phecodes) > 0:
                    for phecode in phecodes:
                        if phecode in phecodelist:
                            if (args.ICD_levels == 1 and code_level == "1") or (args.ICD_levels == -1): data[ID][len(descriptive_features)+phecodelist.index(phecode)] = 1
                            elif args.ICD_levels == 2: 
                                if code_level == "2": data[ID][len(descriptive_features)+phecodelist.index(phecode)] = 1
                                if code_level == "1": data[ID][len(descriptive_features)+phecodelist.index(phecode)] = 1/args.secondary_ICD_weight
    return(data)

def divide_test_train(data, feature_names, args, logging):
    # Next divide the data into training and test sets
    if args.testidfile is not None:
        #read in test IDs from file
        test_IDs = set()
        with open(args.testidfile,'rt') as infile:
            r = csv.reader(infile,delimiter='\t')
            for row in r: test_IDs.add(row[0])
    else:
        all_selected = set()
        for ID in data:
            if data[ID][feature_names.index('train_status')] != -1:
                if data[ID][feature_names.index('case_status')] == -1:
                    data[ID][feature_names.index('train_status')] = -1
                else:
                    all_selected.add(ID)
                    data[ID][feature_names.index('train_status')] = 1
        #draw test IDs without replacement from all IDs
        test_IDs = set(random.sample(list(all_selected), int(args.testfraction*len(list(all_selected)))))
        for ID in test_IDs:
            data[ID][feature_names.index('train_status')] = 0

    logging.info("Division to train and test sets done successfully.")   

    with open(args.outdir+'target-'+args.targetphenotype+'-case_control_counts.txt','wt',encoding='utf-8') as outfile:
        w = csv.writer(outfile,delimiter='\t')
        w.writerow(["SET","N_CASES", "N_CONTROLS"])

        N_case = len([ID for ID in data if data[ID][feature_names.index('case_status')]==1])
        N_control = len([ID for ID in data if data[ID][feature_names.index('case_status')]==0])
        w.writerow(["All", str(N_case), str(N_control)])

        N_case_train = len([ID for ID in data if data[ID][feature_names.index('case_status')]==1 and data[ID][feature_names.index('train_status')]==1])
        N_control_train = len([ID for ID in data if data[ID][feature_names.index('case_status')]==0 and data[ID][feature_names.index('train_status')]==1])
        w.writerow(["Train", str(N_case_train), str(N_control_train)])

        N_case_test = len([ID for ID in data if data[ID][feature_names.index('case_status')]==1 and data[ID][feature_names.index('train_status')]==0])
        N_control_test = len([ID for ID in data if data[ID][feature_names.index('case_status')]==0 and data[ID][feature_names.index('train_status')]==0])
        w.writerow(["Test", str(N_case_test), str(N_control_test)])

    return data

def save_phecode_file(args, feature_names, data):
    #save the phecode file
    #feature_names = ["date_of_birth","end_of_follow_up","sex","case_status","train_status"]+phecodelist+args.includevars
    with gzip.open(args.outdir+'target-'+args.targetphenotype+'-PheRS-ML-input.txt.gz','wt',encoding='utf-8') as outfile:
        w = csv.writer(outfile,delimiter='\t')
        w.writerow(['#ID']+feature_names)
        for ID in data: w.writerow([ID]+data[ID])

    logging.info("Phecode file successfully saved.")

def count_occurrences(feature1,features):
    #computes number of joint occurrences of feature1 (a vector) with all columns of matrix features
    counts = []
    for i in range(features.shape[1]):

        counts.append(np.nansum(feature1.multiply(features[:,i]).toarray()))
    return counts

def extra_phecode_excludes(args, data, excluded_phecodes, phecodelist, descriptive_features, logging):
    # Plot statistics about phecode occurrences skip date of birth
    all_IDs = list(data.keys())
    data_matrix = np.zeros(shape=(len(all_IDs),len(data[all_IDs[0]][5:]))) # N_indvs
    for i in range(len(all_IDs)): data_matrix[i,:] = np.array(data[all_IDs[i]][5:])
    
    #Count number of missing values for each feature.
    #If more than args.missing fraction are missing, mark feature as excluded.
    logging.info("Max missing was: " + str(args.missing))

    frac_nan = np.sum(np.isnan(data_matrix),axis=0)/data_matrix.shape[0]
    excl_inds = np.where(frac_nan>args.missing)[0]
    excluded_phecodes["high_missing"] = dict()
    for i in excl_inds: excluded_phecodes["high_missing"][phecodelist[int(i)]] = frac_nan[int(i)]
    
    frac_one = np.sum(data_matrix, axis=0)/data_matrix.shape[0]
    excl_inds = np.where(frac_one<args.frequency)[0]
    excluded_phecodes["low_freq"] = dict()
    for i in excl_inds: 
        if i > len(descriptive_features): excluded_phecodes["low_freq"][phecodelist[int(i)]] = frac_one[int(i)]

    logging.info("Min frequency was: " + str(args.frequency))
    #save excluded phecodes to a file
    with open(args.outdir+'target-'+args.targetphenotype+'-excluded-phecodes.txt','wt') as outfile:
        w = csv.writer(outfile,delimiter='\t')
        w.writerow(["PheCode", "Reason", "Count"])
        for row in sorted(list(excluded_phecodes["orig"])): w.writerow([row, "phenotype definition", "NA"])

        missing_phecodes = excluded_phecodes["high_missing"]
        for code in missing_phecodes: w.writerow([code, "high missing", str(missing_phecodes[code])])

        low_phecodes = excluded_phecodes["low_freq"]
        for code in low_phecodes: w.writerow([code, "low frequency", str(low_phecodes[code])])

def PheRS_preprocess():
    args = parser_args()
    random.seed(args.seed) #set pseudorandom number generator seed
    
    #save the command used to evoke this script into a file
    logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', filename=args.outdir+'PheRS_preprocess.log',level=logging.INFO,filemode='w')
    logging.info("The command used to evoke this script:")
    logging.info("python PheRS_preprocess.py ".join(sys.argv))

    washout, exposure, observation = time_windows(args)
    #excluded phecodes should not be included as predictors in the model
    excluded_phecodes = read_exclude_phecodes(args)
    #read in the phecode definitions
    phecodelist = read_phecode_defs(args, excluded_phecodes, logging)

    
    #read in the phecode definitions
    ICD2phecode = read_ICD2phecode_maps(args, logging)

    data, feature_names, descriptive_features, N_cases = read_phenotype_file(args=args, 
                                                                              phecodelist=phecodelist, 
                                                                              washout=washout, 
                                                                              exposure=exposure, 
                                                                              observation=observation, 
                                                                              logging=logging)
                                
    data = read_phecode_diags_from_icd_file(args, data, ICD2phecode, exposure, phecodelist, descriptive_features, logging)
    data = downsample_controls(feature_names, data, args, N_cases, logging)
    data = divide_test_train(data, feature_names, args, logging)
    save_phecode_file(args, feature_names, data)
    extra_phecode_excludes(args, data, excluded_phecodes, phecodelist, descriptive_features, logging)
    logging.info("Program successfully terminated.")
    return 1
        
    
PheRS_preprocess()