import argparse
import csv
import gzip
import datetime
import numpy as np
from dateutil.relativedelta import relativedelta

def read_indvs_birth_date(args, observation_start, exposure_start):
    indvs = {}
    head_row = 1
    N_excluded = 0
    with open(args.phenotypefile,'rt') as infile:
        r = csv.reader(infile,delimiter='\t')
        for row in r:
            if head_row == 1: 
                head_row = 0
                header = row
                continue
            ID = row[0]
            sex = row[header.index("SEX")]
            birth_date = datetime.datetime.strptime(row[header.index("DATE_OF_BIRTH")], args.dateformatstr)
            end_of_followup = datetime.datetime.strptime(row[header.index("END_OF_FOLLOWUP")], args.dateformatstr)
            age_base = relativedelta(observation_start, birth_date).years
            # Age at baseline = observation start between min and maxage
            # Exlcuded if 
            #0) the person's end of follow-up is before the end of the washout period
            #1) the person is born after the start of the exposure period
            #2) the person is younger than args.minage at the start of the observation period
            #3) the person is older than args.maxage at the start of the observation period
            if (age_base<args.minage) or \
                (age_base>args.maxage) or \
                (end_of_followup<observation_start) or \
                (birth_date>exposure_start): 
                N_excluded += 1
            else:
                indvs[ID] = [birth_date, age_base, sex]
    return indvs, N_excluded

def write_indvs_stats_file(args, indvs, N_excluded):
    print("Writing indv stat file")

    out_handle = open(args.indvsoutfile, "wt")
    with out_handle as out_file:
        writer = csv.writer(out_file, delimiter="\t")
        writer.writerow(["Group", "N_individuals"])
        writer.writerow(["Excluded", N_excluded])

        age_group_counts = dict()
        sex_counts = dict()
        for indvs_id in indvs:
            age_base = indvs[indvs_id][1]
            sex = indvs[indvs_id][2]
            age_group = str(round(age_base, 0))
            if not age_group in age_group_counts: age_group_counts[age_group] = 1
            else: age_group_counts[age_group] += 1

            if not sex in sex_counts: sex_counts[sex] = 1
            else: sex_counts[sex] +=1

        for sex in sex_counts:
            writer.writerow([sex, sex_counts[sex]])

        for age_group in age_group_counts:
            if age_group_counts[age_group] >= 5: writer.writerow([age_group, age_group_counts[age_group]])

def read_ICD2phecode_maps(args):
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
    return(ICD2phecode)

def match_ICD2phecode(ICD_version,
                      primary_ICD,
                      ID,
                      ICD2phecode,
                      not_found_ICD):
    #add occurrence of primary ICD code, first check if we have an exact match in the ICD to phecode mapping
    if primary_ICD in ICD2phecode[ICD_version]: phecode = ICD2phecode[ICD_version][primary_ICD]
    else:
        #then check if there is a match to the first three characters of the ICD code
        mapped = False
        for trunc_len in range(1,5):
            if trunc_len > len(primary_ICD): break
            primary_ICD_truncated = primary_ICD[0:len(primary_ICD)-trunc_len]
            if primary_ICD_truncated in ICD2phecode[ICD_version]: 
                phecode = ICD2phecode[ICD_version][primary_ICD_truncated]
                mapped = True
                break
        if not mapped:
            #discard this ICD code
            if primary_ICD not in not_found_ICD[ICD_version]: not_found_ICD[ICD_version][primary_ICD] = dict()
            if ID not in not_found_ICD[ICD_version][primary_ICD]: not_found_ICD[ICD_version][primary_ICD][ID] = 1
            else: not_found_ICD[ICD_version][primary_ICD][ID] += 1

            phecode = set(['NA'])
    return(phecode, not_found_ICD)

def get_phecode_stats(args, indvs, ICD2phecode, exposure_end, exposure_start):
    #read in the INTERVENE ICD code file and convert to phecode format
    #calculate basic statistics from the ICD code file
    if args.ICDfile[-2:]=='gz': in_handle = gzip.open(args.ICDfile,'rt',encoding='latin-1')
    else: in_handle = open(args.ICDfile,'rt',encoding='latin-1')

    #ICD codes not found from phecode files
    not_found_ICD = {'9':{},'10':{}} #key = ICD version, value = {code:count}
    prim_counts = {}
    prim_icd_counts = {}
    prim_age_group_counts = {}
    sec_counts = {}
    sec_icd_counts = {}
    sec_age_group_counts = {}

    header_line = 1
    with in_handle as infile:
            r = csv.reader(infile,delimiter='\t')
            for row in r:
                if header_line==1: header_line = 0; continue

                ID = row[0]
                Event_age = row[1]
                if Event_age == "NA": continue # No valid age
                else: Event_age = float(Event_age)
                if ID not in indvs: continue # excluded
                Birth_date = indvs[ID][0]
                Age_base = indvs[ID][1]
                Event_date = Birth_date + relativedelta(years=np.floor(Event_age))

                # Event during exposure
                if (Event_date > exposure_end) | (Event_date < exposure_start): continue

                ICD_version = row[2]
                primary_ICD = row[3]
                secondary_ICD = row[4] #source of the ICD code

                # ICD level
                ICD_code = primary_ICD.strip('"').strip("'").replace('.','') #remove . from the ICD codes
                if (secondary_ICD=="1") or (args.ICD_levels==-1):
                    if ICD_version not in prim_icd_counts: prim_icd_counts[ICD_version] = {}
                    if ICD_code not in prim_icd_counts[ICD_version]: prim_icd_counts[ICD_version][ICD_code] = {}
                    if ID not in prim_icd_counts[ICD_version][ICD_code]: prim_icd_counts[ICD_version][ICD_code][ID] = 1
                    else: prim_icd_counts[ICD_version][ICD_code][ID] += 1
                elif not args.ICD_levels==-1:
                    if ICD_version not in sec_icd_counts: sec_icd_counts[ICD_version] = {}
                    if ICD_code not in sec_icd_counts[ICD_version]: sec_icd_counts[ICD_version][ICD_code] = {}
                    if ID not in sec_icd_counts[ICD_version][ICD_code]: sec_icd_counts[ICD_version][ICD_code][ID] = 1
                    else: sec_icd_counts[ICD_version][ICD_code][ID] += 1
                # PheCode level
                phecodes, not_found_ICD = match_ICD2phecode(ICD_version, primary_ICD, ID, ICD2phecode, not_found_ICD)
                for phecode in phecodes:
                    #add the occurrence of the primary phecode
                    if phecode=="NA": continue
                    
                    age_group = str(int(10 * np.floor(round(Event_age) / 10)))
                    # Primary diagnoses
                    if (secondary_ICD=="1") or (args.ICD_levels==-1):
                        # Overall counts
                        if phecode not in prim_counts: prim_counts[phecode] = {}
                        elif ID not in prim_counts[phecode]: prim_counts[phecode][ID] = 1
                        else: prim_counts[phecode][ID] += 1

                        # Age group specific counts
                        if phecode not in prim_age_group_counts: prim_age_group_counts[phecode] = {}
                        if age_group not in prim_age_group_counts[phecode]: prim_age_group_counts[phecode][age_group] = {}
                        if ID not in prim_age_group_counts[phecode][age_group]: prim_age_group_counts[phecode][age_group][ID] = 1
                        else: prim_age_group_counts[phecode][age_group][ID] += 1
                    # Secondary diagnoses
                    elif not args.ICD_levels==-1:
                        # Overall counts
                        if phecode not in sec_counts: sec_counts[phecode] = {}
                        elif ID not in sec_counts[phecode]: sec_counts[phecode][ID] = 1
                        else: sec_counts[phecode][ID] += 1

                        # Age group specific counts
                        if phecode not in sec_age_group_counts: sec_age_group_counts[phecode] = {}
                        if age_group not in sec_age_group_counts[phecode]: sec_age_group_counts[phecode][age_group] = {}
                        if ID not in sec_age_group_counts[phecode][age_group]: sec_age_group_counts[phecode][age_group][ID] = 1
                        else: sec_age_group_counts[phecode][age_group][ID] += 1
    return prim_counts, prim_age_group_counts, sec_counts, sec_age_group_counts, not_found_ICD, prim_icd_counts, sec_icd_counts

def write_phecode_stats(args, prim_counts, prim_age_group_counts, sec_counts, sec_age_group_counts):

    print("Starting phecode stat file writing")
    out_handle = open(args.phecodeoutfile, "wt")       
    with out_handle as out_file:
        writer = csv.writer(out_file, delimiter="\t")
        writer.writerow(["PheCode", "Group", "Level", "N_occurances", "N_individuals"])
        # All counts
        for phecode in prim_counts:
            n_indvs = len(prim_counts[phecode].keys())
            n_counts = sum(prim_counts[phecode].values())

            if n_indvs >= 5: 
                if args.ICD_levels != -1: writer.writerow([phecode, "All", "Primary", n_counts, n_indvs])
                else: writer.writerow([phecode, "All", "NA", n_counts, n_indvs])
        for phecode in sec_counts:
            n_indvs = len(sec_counts[phecode].keys())
            n_counts = sum(sec_counts[phecode].values())

            if n_indvs >= 5: writer.writerow([phecode, "All", "Secondary", n_counts, n_indvs])

        # Age group counts
        for phecode in prim_age_group_counts:
            for age_group in prim_age_group_counts[phecode]:
                n_indvs = len(prim_age_group_counts[phecode][age_group].keys())
                n_counts = sum(prim_age_group_counts[phecode][age_group].values())

                if n_indvs >= 5: 
                    if args.ICD_levels != -1: writer.writerow([phecode, age_group, "Primary", n_counts, n_indvs])
                    else: writer.writerow([phecode, age_group, "NA", n_counts, n_indvs])

        for phecode in sec_age_group_counts:
            for age_group in sec_age_group_counts[phecode]:
                n_indvs = len(sec_age_group_counts[phecode][age_group].keys())
                n_counts = sum(sec_age_group_counts[phecode][age_group].values())

                if n_indvs >= 5: writer.writerow([phecode, age_group, "Secondary", n_counts, n_indvs])

def write_unmapped_file(args, not_found_ICD):
    out_handle = open(args.unmappedfile, "wt")
    with out_handle as out_file:
        writer = csv.writer(out_file, delimiter="\t")
        writer.writerow(["ICD_version", "ICD_code", "N_occurances", "N_individuals"])
        for ICD_version in not_found_ICD:
            for ICD_code in not_found_ICD[ICD_version]:
                n_indvs = len(not_found_ICD[ICD_version][ICD_code].keys())
                n_counts = sum(not_found_ICD[ICD_version][ICD_code].values())
                if n_indvs >= 5: writer.writerow([ICD_version, ICD_code, n_counts, n_indvs])


def write_icd_stats(args, prim_icd_counts, sec_icd_counts):
    out_handle = open(args.icdstatsfile, "wt")
    with out_handle as out_file:
        writer = csv.writer(out_file, delimiter="\t")
        writer.writerow(["ICD_version", "ICD_code", "Level", "N_occurances", "N_individuals"])
        for ICD_version in prim_icd_counts:
            for ICD_code in prim_icd_counts[ICD_version]:
                n_indvs = len(prim_icd_counts[ICD_version][ICD_code].keys())
                n_counts = sum(prim_icd_counts[ICD_version][ICD_code].values())
                if n_indvs >= 5: 
                    if args.ICD_levels != -1: writer.writerow([ICD_version, ICD_code, "Primary", n_counts, n_indvs])
                    else: writer.writerow([ICD_version, ICD_code, "NA", n_counts, n_indvs])
        for ICD_version in sec_icd_counts:
            for ICD_code in sec_icd_counts[ICD_version]:
                n_indvs = len(sec_icd_counts[ICD_version][ICD_code].keys())
                n_counts = sum(sec_icd_counts[ICD_version][ICD_code].values())
                if n_indvs >= 5: writer.writerow([ICD_version, ICD_code, "Secondary", n_counts, n_indvs])

def PheCode_counts():
    parser = argparse.ArgumentParser()

    #Command line arguments:
    
    parser.add_argument("--ICDfile",help="Full path to an INTERVENE longitudinal ICD code file. If last two letters of file name are gz, gzipped file is assumed.",type=str,default=None)
    parser.add_argument("--phenotypefile",help="Full path to an INTERVENE longitudinal ICD code file. If last two letters of file name are gz, gzipped file is assumed.",type=str,default=None)
    parser.add_argument("--phecodefile",help="Phecode definition file (NOTE: comma-separated).",type=str,default=None)
    parser.add_argument("--ICD9tophecodefile",help="Full path to the ICD9 to phecode map (NOTE: comma-separated).",type=str,default=None)
    parser.add_argument("--ICD10tophecodefile",help="Full path to the ICD10 to phecode map (NOTE: comma-separated).",type=str,default=None)
    parser.add_argument("--ICD10CMtophecodefile",help="Full path to the ICD10-CM to phecode map (NOTE: comma-separated).",type=str,default=None)
    parser.add_argument("--dateformatstr",help="Date format used (default='%Y-%m-%d')",type=str,default='%Y-%m-%d')
    parser.add_argument("--phecodeoutfile",help="Full path to the output file for the phecode stats.",type=str,default=None)
    parser.add_argument("--icdstatsfile",help="Full path to the output file for the icd stats.",type=str,default=None)
    parser.add_argument("--indvsoutfile",help="Full path to the output file for the individuals stats.",type=str,default=None)
    parser.add_argument("--unmappedfile",help="Full path to the output file for the individuals stats.",type=str,default=None)
    parser.add_argument("--washout_window",help="Start and end dates for washout (default= 2010-01-01 2011-12-31).",type=str,nargs=2,default=['2010-01-01','2011-12-31'])
    parser.add_argument('--exposure_window',help='Start and end dates for exposure (default = 2000-01-01 2009-12-31).',type=str,nargs=2,default=['2000-01-01','2009-12-31'])
    parser.add_argument('--observation_window',help='Start and end dates for observation (default= 2012-01-01 2020-01-01).',type=str,nargs=2,default=['2012-01-01','2020-01-01'])
    parser.add_argument("--minage",help="Minimum age at the start of the observation period to include (default=32).",type=float,default=32)
    parser.add_argument("--maxage",help="Maximum age at the start of the observation period to include (default=70).",type=float,default=70)
    parser.add_argument("--ICD_levels",help="Level of ICD-codes to use. 1=primary, 2=primary+secondary.",type=int,default=1)

    args = parser.parse_args()

    #convert the dates to Python datetime objects
    exposure_start = datetime.datetime.strptime(args.exposure_window[0],args.dateformatstr)
    exposure_end = datetime.datetime.strptime(args.exposure_window[1],args.dateformatstr)
    observation_start = datetime.datetime.strptime(args.observation_window[0],args.dateformatstr)

    indvs, N_excluded = read_indvs_birth_date(args, observation_start, exposure_start)
    write_indvs_stats_file(args, indvs, N_excluded)

    print("Starting phecode stats")
    ICD2phecode = read_ICD2phecode_maps(args)
    prim_counts, prim_age_group_counts, sec_counts, sec_age_group_counts, not_found_ICD, prim_icd_counts, sec_icd_counts = get_phecode_stats(args, indvs, ICD2phecode, exposure_end, exposure_start)
    write_phecode_stats(args, prim_counts, prim_age_group_counts, sec_counts, sec_age_group_counts)
    write_icd_stats(args, prim_icd_counts, sec_icd_counts)
    write_unmapped_file(args, not_found_ICD)

PheCode_counts()