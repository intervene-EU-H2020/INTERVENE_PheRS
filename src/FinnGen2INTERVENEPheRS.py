#!/usr/bin/env python

import argparse
import csv
import gzip
import sys
import codecs

def FinnGen2INTERVENEPheRS():

    parser = argparse.ArgumentParser()

    #Command line arguments:
    parser.add_argument("--infile",help="Input file full path, FinnGen detailed longitudinal file format. If last two letters of file name are gz, gzipped file is assumed.",type=str,default=None)
    parser.add_argument("--outfile",help="Output file full path, INTERVENE PheRS longitudinal file format. If last two letters of file name are gz, gzipped file is assumed",type=str,default='./INTERVENE_longitudinal_PheRS_FinnGen_default_output_file_name.txt')
    parser.add_argument("--allowed_ICDvers",help="List of allowed ICD versions, all other entries are skipped (default=9, 10)",type=str,nargs='+',default=['9','10'])

    args = parser.parse_args()

    #save the command used to evoke this script into a file
    with open(args.outfile+'.log','wt') as logfile: logfile.write(" ".join(sys.argv))

    #ICD-codes only come from certain "source" registries.
    #for each source, we only use certain "categories"
    #key = source register, value = list of accepted categories
    allowed_sources = {"INPAT":["0"],#inpatient HILMO, 0=main diagnosis category
                       "OUTPAT":["0"],#outpatient HILMO, 0=main diagnosis category
                       "PRIM_OUT":["ICD0"],#primary healthcare outpatient visits, ICD0=cause of visit, main diagnosis
                       "DEATH":["U","I"]#cause of death register, U=underlying cause of death, I=immediate cause of death
    }
    
    #read in the input row by row and convert to INTERVENE format
    if args.infile[:-2]=='gz': in_handle = gzip.open(args.infile,'rt',encoding='utf-8')
    else: in_handle = open(args.infile,'rt')

    if args.outfile[:-2]=='gz': out_handle = gzip.open(args.outfile,'wt')
    else: out_handle = open(args.outfile,'wt')

    with out_handle as outfile:
        w = csv.writer(outfile,delimiter='\t')
        w.writerow(["ID","Event_age","ICD_version","primary_ICD","secondary_ICD"])
        with in_handle as infile:
            r = csv.reader(infile,delimiter='\t')
            for row in r:
                ICD_version = row[8]
                source = row[1]
                #if the ICD code version is different than what is listed as desired, the entry is skipped
                #also if the source is different than what is listed in the allowed_sources, the entry is skipped
                if (ICD_version not in args.allowed_ICDvers) or (source not in allowed_sources): continue

                category = row[9]
                #check that the category is listed in allowed_sources
                if category not in allowed_sources[source]: continue
                ID = row[0]
                Event_age = row[2]
                primary_ICD = row[4]
                secondary_ICD = row[5]
                w.writerow([ID,Event_age,ICD_version,primary_ICD,secondary_ICD])
                
FinnGen2INTERVENEPheRS()                
