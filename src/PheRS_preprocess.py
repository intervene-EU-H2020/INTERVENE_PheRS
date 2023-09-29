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

def PheRS_preprocess():

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
    
    args = parser.parse_args()

    random.seed(args.seed) #set pseudorandom number generator seed
    eps = 0.00000001 #add this to phecodes when checking if within exclude range
    
    #save the command used to evoke this script into a file
    logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', filename=args.outdir+'PheRS_preprocess.log',level=logging.INFO,filemode='w')
    logging.info("The command used to evoke this script:")
    logging.info("python PheRS_preprocess.py ".join(sys.argv))

    #convert the dates to Python datetime objects
    washout_start = datetime.datetime.strptime(args.washout_window[0],args.dateformatstr)
    washout_end = datetime.datetime.strptime(args.washout_window[1],args.dateformatstr)
    exposure_start = datetime.datetime.strptime(args.exposure_window[0],args.dateformatstr)
    exposure_end = datetime.datetime.strptime(args.exposure_window[1],args.dateformatstr)
    observation_start = datetime.datetime.strptime(args.observation_window[0],args.dateformatstr)
    observation_end = datetime.datetime.strptime(args.observation_window[1],args.dateformatstr)
    
    #read in the phecode definitions
    phecodes = {} #key = phecode, value = {"phenotype":string,"phecode_exclude_range":[smallest,largest],"sex":string,"rollup":int,"leaf":int,"category_number":int,"category":string}
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
                #elif i>3 and i<7: phecodes[row[0]][header[i]] = int(row[i])
                #else: phecodes[row[0]][header[i]] = row[i]
    phecodelist = list(phecodes.keys())
    logging.info("Phecode definitions read in successfully.")
    
    #excluded phecodes should not be included as predictors in the model
    excluded_phecodes = set()
    if args.excluded_phecodes!=None:
        with open(args.excludephecodes,'rt') as infile:
            r = csv.reader(infile,delimiter='\t')
            for row in r: excluded_phecodes = row
    
    excluded_phecodes = set(excluded_phecodes) #list of phecodes excluded
    #later we will add to this list phecodes that have less than args.frequency
    #of occurrences or more than args.missing missing values
    
    #read in the phecode definitions
    phecodefiles = [args.ICD9tophecodefile,args.ICD10tophecodefile,args.ICD10CMtophecodefile]
    phecode_versions = ["9","10","10CM"]
    ICD2phecode = {} #key = phecode_version, value = {ICD1:phecode1,...}
    ind = -1
    for f in phecodefiles:
        ind += 1
        ICD2phecode[phecode_versions[ind]] = {}
        if f is None: continue
        with open(f,'rt',encoding='latin-1') as infile:
            #r = csv.reader(infile,delimiter=',')
            i = 1
            for row in infile:
                #print(str(i)+" : "+f+" : "+str(row))
                row = row.strip().split(',')
                icd = row[0].strip('"').strip("'").replace('.','') #remove . from the ICD codes
                phecode = row[1].strip('"').strip("'")
                if len(phecode)>0:
                    #not saving mapping for ICD-codes that do not map to a phecode
                    ICD2phecode[phecode_versions[ind]][icd] = phecode
                    i += 1
    logging.info("ICD to phecode maps read in successfully.")

    logging.info("Target phenotype definition read in successfully.")

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
    feature_names = ["date_of_birth","end_of_follow_up","sex","case_status","train_status"]+phecodelist+args.includevars

    if args.phenotypefile[-2:]=='gz': in_handle = gzip.open(args.phenotypefile,'rt',encoding='utf-8')
    else: in_handle = open(args.phenotypefile,'rt',encoding='utf-8')

    N_cases = 0 #number of cases
    N_excluded = 0
    
    with in_handle as infile:
        for row in infile:
            row = row.split('\t')
            #print(row)
            if row[0]=='ID':
                header = row
                target_column = header.index(args.targetphenotype)
                target_time_column = header.index(args.targetphenotype+"_DATE")
                continue
            ID = row[0]
            date_of_birth = datetime.datetime.strptime(row[2],args.dateformatstr)
            end_of_followup = datetime.datetime.strptime(row[header.index("END_OF_FOLLOWUP")],args.dateformatstr) #end of follow-up in the biobank
            #print("dob="+str(date_of_birth))
            #start_of_followup = datetime.datetime.strptime(row[92],args.dateformatstr)
            #print("start of followup="+str(start_of_followup))
            data[ID] = [0 for i in range(len(phecodelist)+5+len(args.includevars))] #initialize the row corresponding to ID
            data[ID][0] = date_of_birth
            #if age is requested as a predictor calculate age at the start of the observation window
            if 'age' in args.includevars:
                age = (observation_start-date_of_birth).days/365.25
                data[ID][feature_names.index('age')] = age
            
            #If
            #0) the person's end of follow-up is before the end of the washout period
            #1) the person is born after the start of the exposure period
            #2) the person is younger than args.minage at the start of the observation period
            #3) the person is older than args.maxage at the start of the observation period
            # they are excluded
            if (data[ID][0]>exposure_start) or (data[ID][0]>observation_start-relativedelta(years=args.minage) or (data[ID][0]<observation_start-relativedelta(years=args.maxage)) or (end_of_followup<washout_end)):
                #print("dob="+str(date_of_birth))
                data[ID][3] = -1
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
            data[ID][3] = row[target_column]#case/control status
            
            if data[ID][3]!='NA':
                data[ID][3] = int(data[ID][3])
                if data[ID][3]>0:
                    #this means we have a case
                    case_date = datetime.datetime.strptime(row[target_time_column],args.dateformatstr)
                    #diff = case_date-date_of_birth
                    data[ID][1] = case_date#diff.days/365.0 #age at first diagnosis
                    #print('case date: '+str(case_date)+", birth: "+str(date_of_birth)+', age: '+str(data[ID][0]))
                    if (data[ID][1]>=observation_start) and (data[ID][1]<=observation_end):
                        #The diagnosis was received during the observation period
                        N_cases += 1
                    elif case_date>end_of_followup:
                        #diagnosis is marked to have happened after the end of follow-up for this ID, which does not make sense, thus we exclude
                        data[ID][3] = -1
                    else:
                        #diagnosis was outside the observation period
                        #if diagnosis happens after the end of observation period, we have a control
                        #otherwise, the individual is excluded
                        if data[ID][1]>observation_end: data[ID][3] = 0
                        else: data[ID][3] = -1
                    #print('case date: '+str(case_date)+", birth: "+str(date_of_birth)+', age: '+str(data[ID][0])+" | case status="+str(data[ID][2]))
                else:
                    #this means we have a control
                    #end_of_followup = datetime.datetime.strptime(row[header.index("END_OF_FOLLOWUP")],args.dateformatstr)w
                    #diff = end_of_followup-date_of_birth
                    data[ID][1] = end_of_followup#diff.days/365.0 #follow-up time in years
                    #print('eof:')
                    #print(data[ID][1])
                    #print('observation_end:')
                    #print(observation_end)
                    if data[ID][1]<=observation_end:
                        #follow-up ending before observation period end
                        data[ID][3] = -1
                    else:
                        #print("control!")
                        data[ID][3] = 0
                    #while True:
                    #    z = input('any')
                    #    break
            else:
                data[ID][3] = -1 #mark excluded if variable value is NA
                end_of_followup = datetime.datetime.strptime(row[header.index("END_OF_FOLLOWUP")],args.dateformatstr)
                #print("end of followup: "+str(end_of_followup))
                data[ID][1] = end_of_followup#End of follow-up is a separate column in the phenotype file
            #data[ID][1] = (start_of_followup-date_of_birth).days/365.0  #age at start of follow up in years
            
            #print(header)
            #print(row)
            sex = row[1]
            if row[1]=='female': sex = 1
            elif row[1]=='male': sex = 0
            elif row[1]=='other': sex = -1
            else: sex = row[1]
            data[ID][2] = sex #sex
            for var in args.includevars:
                if var=='age': continue
                #print(args.includevars[i])
                #print(row[header.index(args.includevars[i])])
                mpf_index = header.index(var)
                out_index = feature_names.index(var)
                
                value = row[mpf_index].strip()
                #print("i="+str(i))
                #print(feature_names[5+len(phecodelist)+i])
                if value=='NA' or value=='-': data[ID][out_index] = np.nan
                else: data[ID][out_index] = float(value) #additional features used
    logging.info("Phenotype file read in successfully.")
    
    #Then draw args.controlfraction*N_cases controls to use
    #remove other controls and all excluded IDs (where case_status=-1)
    #get a list of all control IDs
    control_IDs = [ID for ID in data if data[ID][3]==0]
    print("Total number of IDs="+str(len(list(data.keys()))))
    print("Number of cases="+str(N_cases))
    print("Number of excluded IDs="+str(len([ID for ID in data if data[ID][3]==-1])))
    print("Number of control IDs before sampling="+str(len(control_IDs)))
    
    if args.controlfraction is not None:
        #sample final controls
        random.seed(args.seed)
        final_control_IDs = random.sample(control_IDs,int(args.controlfraction*N_cases))
    else: final_control_IDs = control_IDs
    print("Number of control IDs="+str(len(final_control_IDs)))
    #keep only these and case IDs
    
    case_IDs = [ID for ID in data if data[ID][3]==1]
    data = {key: data[key] for key in case_IDs+final_control_IDs}
    print("Number of case IDs="+str(len(case_IDs)))
    
    
    #read in the INTERVENE ICD code file and convert to phecode format
    #calculate basic statistics from the ICD code file
    if args.ICDfile[-2:]=='gz': in_handle = gzip.open(args.ICDfile,'rt',encoding='latin-1')
    else: in_handle = open(args.ICDfile,'rt',encoding='latin-1')

    #ICD codes not found from phecode files
    not_found_ICD = {'9':{},'10':{},'10CM':{}} #key = ICD version, value = {code:count}
    with in_handle as infile:
        r = csv.reader(infile,delimiter='\t')
        for row in r:
            if row[0]=='ID': continue
            ID = row[0]
            #note that only the IDs that are present in the phenotype file are used
            if ID not in data: continue
            event_age = float(row[1])
            #only include ICD codes that occur within the exposure window
            exposure_start_age = (exposure_start-data[ID][0]).total_seconds()/(365.0*24*60*60)
            #print(row)
            #print(ID)
            #print('date of birth:')
            #print(data[ID][0])
            #print('exposure_start:')
            #print(exposure_start)
            #print("even_age:")
            #print(event_age)
            #print("exposure_start_age:")
            #print(exposure_start_age)
            #while True:
            #    z = input('any')
            #    break
            exposure_end_age = (exposure_end-data[ID][0]).total_seconds()/(365.0*24*60*60)
            if event_age>=exposure_start_age and event_age<=exposure_end_age:
                ICD_version = row[2]
                primary_ICD = row[3]
                secondary_ICD = row[4]
                
                #add occurrence of primary ICD code
                #first check if we have an exact match in the ICD to phecode mapping
                if primary_ICD in ICD2phecode[ICD_version]: primary_phecode = ICD2phecode[ICD_version][primary_ICD]
                else:
                    #then check if there is a match to the first three characters
                    #of the ICD code
                    primary_ICD_truncated = primary_ICD[:3]
                    if primary_ICD_truncated in ICD2phecode[ICD_version]: primary_phecode = ICD2phecode[ICD_version][primary_ICD_truncated]
                    else:
                        #discard this ICD code
                        if primary_ICD not in not_found_ICD[ICD_version]: not_found_ICD[ICD_version] = {primary_ICD:1}
                        else: not_found_ICD[ICD_version][primary_ICD] += 1
                        primary_phecode = 'NA'
                #add the occurrence of the primary phecode
                if primary_phecode!="NA": data[ID][5+phecodelist.index(primary_phecode)] = 1
                
                #add occurrence of secondary ICD code
                #first check if we have an exact match in the ICD to phecode mapping
                if secondary_ICD in ICD2phecode[ICD_version]: secondary_phecode = ICD2phecode[ICD_version][secondary_ICD]
                else:
                    #then check if there is a match to the first three characters
                    #of the ICD code
                    secondary_ICD_truncated = secondary_ICD[:3]
                    if secondary_ICD_truncated in ICD2phecode[ICD_version]: secondary_phecode = ICD2phecode[ICD_version][secondary_ICD_truncated]
                    else:
                        #discard this ICD code
                        if secondary_ICD not in not_found_ICD[ICD_version]: not_found_ICD[ICD_version] = {secondary_ICD:1}
                        else: not_found_ICD[ICD_version][secondary_ICD] += 1
                        secondary_phecode = 'NA'
                
                if secondary_phecode!="NA": data[ID][5+phecodelist.index(secondary_phecode)] = 1


    logging.info("INTERVENE ICD code file read in successfully.")

    #next divide the data into training and test sets
    if args.testidfile is not None:
        #read in test IDs from file
        test_IDs = set()
        with open(args.testidfile,'rt') as infile:
            r = csv.reader(infile,delimiter='\t')
            for row in r: test_IDs.add(row[0])
    else:
        #draw test IDs without replacement from all IDs
        test_IDs = set(random.sample(list(data.keys()),int(args.testfraction*len(list(data.keys())))))
    #mark IDs not in test_IDs for training set
    for ID in data:
        if ID not in test_IDs: data[ID][4] = 1

    logging.info("Division to train and test sets done successfully.")    

    #save the phecode file
    #feature_names = ["date_of_birth","end_of_follow_up","sex","case_status","train_status"]+phecodelist+args.includevars
    with gzip.open(args.outdir+'target-'+args.targetphenotype+'-PheRS-ML-input.txt.gz','wt',encoding='utf-8') as outfile:
        w = csv.writer(outfile,delimiter='\t')
        #w.writerow(["#INTERVENE PheRS ML input file"])
        #w.writerow(["#"+" ".join(sys.argv)])
        w.writerow(["#ID"]+feature_names)
        for ID in data: w.writerow([ID]+data[ID])

    logging.info("Phecode file successfully saved.")

    #plot statistics about phecode occurrences
    #skip date of birth
    all_IDs = list(data.keys())
    data_matrix = np.zeros(shape=(len(all_IDs),len(data[all_IDs[0]][2:])))
    
    for i in range(len(all_IDs)): data_matrix[i,:] = np.array(data[all_IDs[i]][2:])
    

    #Count number of missing values for each feature.
    #If more than args.missing fraction are missing, mark feature as excluded.
    frac_nan = np.sum(np.isnan(data_matrix),axis=0)/data_matrix.shape[0]
    excl_inds = np.where(frac_nan>args.missing)[0]
    for i in excl_inds: excluded_phecodes.add(feature_names[int(i)])

    #plot histogram of missing values
    #plt.hist(frac_nan,bins='auto')
    #plt.xlabel('fraction of missing values per feature')
    #plt.ylabel('count')
    #plt.savefig(args.outdir+"fraction_of_missing_values_per_feature_histo.pdf",dpi=300)
    #plt.clf()
    x_missing = []
    y_count = []
    colors = []
    
    data_matrix = csc_matrix(data_matrix)
    #count joint occurrences of features
    with mp.Pool(args.nproc) as pool:
        #print("data_matrix shape:"+str(data_matrix.shape))
        #print("num features:"+str(len(feature_names)+1))
        #occurrences counted only for binary variables
        res = [pool.apply_async(count_occurrences,args=(data_matrix[:,i],data_matrix[:,i:])) for i in range(0,len(feature_names[2:]))]
        res = [p.get() for p in res]
        counts = {} #key = (feature1,feature2), value = number of co-occurrences
        for i in range(len(res)):
            for j in range(len(res[i])):
                #print(len(feature_names))
                #print("i="+str(i)+",j="+str(j))
                #print("index="+str(2+i+j))
                #print(feature_names[i+3])
                #print(feature_names[i+3+j])
                counts[(feature_names[i+2],feature_names[i+2+j])] = res[i][j]

    logging.info("Joint occurrences of features computed successfully.")

    #mark as excluded those predictors where frequency is less than
    #args.frequency
    min_count = int(len(data.keys())*args.frequency)
    #print("feature_names:")
    #print(feature_names)
    for i in range(6,len(feature_names)):
        name = feature_names[i]
        if name[:2]=='PC':
            #don't exclude principal components
            continue
        #print(name+", count="+str(counts[(name,name)])+"/"+str(min_count))
        if counts[(name,name)]<min_count: excluded_phecodes.add(name)

    #save excluded phecodes to a file
    with open(args.outdir+'target-'+args.targetphenotype+'-excluded-phecodes.txt','wt') as outfile:
        w = csv.writer(outfile,delimiter='\t')
        for row in sorted(list(excluded_phecodes)): w.writerow([row])

    #Skip computing feature overlaps, it can be done in a better way if needed
    logging.info("Program successfully terminated.")
    return 1

        
    #compute all pairwise overlaps between features and save to a file
    #overlap = N(cases having both 1 and 2)/N(cases of either 1 or 2)
    with open(args.outdir+"feature_overlaps.txt",'wt') as outfile:
        w = csv.writer(outfile,delimiter='\t')
        w.writerow(['feature1','feature2','feature1_column','feature2_column','feature1_count','feature2_count','overlap','feature1_exclude','feature2_exclude','feature1_nan_fraction','feature2_nan_fraction'])
        for pair in counts.keys():
            feature1 = pair[0]
            feature2 = pair[1]
            overlap = counts[(feature1,feature2)]/float(counts[(feature1,feature1)]+counts[(feature2,feature2)])
            exclude1 = False
            if feature1 in excluded_phecodes: exclude1 = True
            exclude2 = False
            if feature2 in excluded_phecodes: exclude2 = True
            if feature1==feature2:
                x_missing.append(frac_nan[feature_names.index(feature1)])
                y_count.append(counts[feature1,feature1])
                if exclude1: colors.append('r')
                else: colors.append('b')
                
            w.writerow([feature1,feature2,feature_names.index(feature1)+1,feature_names.index(feature2)+1,counts[(feature1,feature1)],counts[feature2,feature2],overlap,exclude1,exclude2,frac_nan[feature_names.index(feature1)],frac_nan[feature_names.index(feature2)]])
    logging.info("Feature overlaps computed and saved to file "+args.outdir+"feature_overlaps.txt")

    #scatter plot of missing values vs total number of non-zeros per feature
    plt.scatter(x_missing,y_count,c=colors,marker='o',alpha=0.5)
    plt.xlabel('fraction of missing values')
    plt.ylabel('number of existing values')
    plt.title('All features')
    plt.tight_layout()
    plt.savefig(args.outdir+'missing_vs_existing_count_scatter.pdf',dpi=300)
    plt.clf()
    
    #plotting the number of occurrences for each feature and a histogram of  feature occurrences
    occurrences = []
    labels = []
    for pair in counts.keys():
        #print(pair)
        #while True:
        #    z = input("any")
        #    break
        if pair[0]==pair[1]:
            #only show codes that are observed more than 5 times
            if counts[pair]>5:
                occurrences.append([pair[0],counts[pair]])
                labels.append(pair[0])
    occurrences.sort(key=itemgetter(1),reverse=True)
    #print("occurrences:")
    #print(occurrences)
    #histogram
    plt.hist([o[1] for o in occurrences],bins='auto')
    plt.xlabel('feature occurrences')
    plt.ylabel('count')
    plt.savefig(args.outdir+"feature_occurrence_histo.pdf",dpi=300)
    plt.clf()
    
    #number of occurrences of each phecode, only for phecodes that are observed more than 5 times
    plt.scatter([i for i in range(len(occurrences))],[o[1] for o in occurrences])
    plt.ylabel('number of occurrences')
    plt.xticks([i for i in range(len(occurrences))],labels,rotation='vertical')
    plt.tight_layout()
    plt.savefig(args.outdir+"features_ranked_by_occurrence.pdf",dpi=300)
    #fig = px.scatter(x=[i for i in range(len(occurrences))],y=[o[1] for o in occurrences],hover_data=[o[0] for o in occurrences],labels={'x':'features ranked by number of occurrences','y':'number of occurrences'})
    #fig = px.scatter(x=[i for i in range(len(occurrences))],y=[o[1] for o in occurrences],labels={'x':'features ranked by number of occurrences','y':'number of occurrences'})
    #fig.write_html(args.outdir+"features_ranked_by_occurrence.html")
    plt.clf()

    logging.info("Plotting statistics successful.")
    logging.info("Program successfully terminated.")
    
def count_occurrences(feature1,features):
    #computes number of joint occurrences of feature1 (a vector) with all columns of matrix features
    counts = []
    for i in range(features.shape[1]):

        #print("i="+str(i))
        #print("feature1 shape:"+str(feature1.shape)+", features slice shape:"+str(features[:,i].shape))
        
        counts.append(np.nansum(feature1.multiply(features[:,i]).toarray()))
    return counts
        
    
PheRS_preprocess()
