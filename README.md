# INTERVENE_PheRS

This repository contains code for training Phenotype Risk Score models (PheRS) based on the INTERVENE data formats. Below you can find a description of the data files needed and instructions on how to run the analysis. The commands used to replicate the analyses done in FinnGen are listed in section [Calls to replicate the FinnGen analyses](#replicate) .

The scripts have been run in the FinnGen sandbox using Python version 3.7.3 and the following packages:

* matplotlib (version 2.2.5)
* numpy (1.16.6)
* scipy (1.2.3)
* pandas (0.24.2)
* scikit-learn (0.20.4)

Other packages used should be standard Python, but a full list of installed packages in the FinnGen sandbox environment can be viewed at `data/requirements_finngen.txt`.

Note that in order for the scripts to work, you will need to add the package root to your PYTHONPATH:

```
export PYTHONPATH=/absolute/path/to/INTERVENE_PheRS/
```


## Table of contents

1. [Usage instructions](#usage)
	1. [The required input files](#input)
	2. [Output file descriptions](#output)
	3. [Calls to replicate the FinnGen analyses](#replicate)
2. [Overview of changelog](#changes)

## Usage instructions <a name="usage"></a>

This section is under construction. The Python script
```
src/PheRSwrapper.py
```
converts ICD10 codes input int the INTERVENE longitudinal file format into phecodes and preprocesses data for PheRS model fitting. It then fits the PheRS-model (elastic net) and makes predictions on a test set also reporting prediction performance. The script also writes several log files that list all the files and parameter settings of the run. Help can be viewed by typing

```
python /finngen/red/thartone/git/INTERVENE_PheRS/src/PheRSwrapper.py -h
usage: PheRSwrapper.py [-h] [--ICDfile ICDFILE]
                       [--phenotypefile PHENOTYPEFILE]
                       [--includevars INCLUDEVARS [INCLUDEVARS ...]]
                       [--phecodefile PHECODEFILE]
                       [--ICD9tophecodefile ICD9TOPHECODEFILE]
                       [--ICD10tophecodefile ICD10TOPHECODEFILE]
                       [--ICD10CMtophecodefile ICD10CMTOPHECODEFILE]
                       [--targetphenotype TARGETPHENOTYPE]
                       [--excludephecodes EXCLUDEPHECODES] [--outdir OUTDIR]
                       [--testfraction TESTFRACTION] [--testidfile TESTIDFILE]
                       [--washout_window WASHOUT_WINDOW WASHOUT_WINDOW]
                       [--exposure_window EXPOSURE_WINDOW EXPOSURE_WINDOW]
                       [--observation_window OBSERVATION_WINDOW OBSERVATION_WINDOW]
                       [--minage MINAGE] [--maxage MAXAGE] [--seed SEED]
                       [--dateformatstr DATEFORMATSTR]
                       [--excludeICDfile EXCLUDEICDFILE] [--missing MISSING]
                       [--frequency FREQUENCY]
                       [--controlfraction CONTROLFRACTION] [--nproc NPROC]
                       [--paramgridfile PARAMGRIDFILE]

optional arguments:
  -h, --help            show this help message and exit
  --ICDfile ICDFILE     Full path to an INTERVENE longitudinal ICD code file.
                        If last two letters of file name are gz, gzipped file
                        is assumed.
  --phenotypefile PHENOTYPEFILE
                        Full path to an INTERVENE phenotype file. If last two
                        letters of file name are gz, gzipped file is assumed
  --includevars INCLUDEVARS [INCLUDEVARS ...]
                        List of variable names to include from the phenotype
                        file (default=nothing).
  --phecodefile PHECODEFILE
                        Phecode definition file (NOTE: comma-separated).
  --ICD9tophecodefile ICD9TOPHECODEFILE
                        Full path to the ICD9 to phecode map (NOTE: comma-
                        separated).
  --ICD10tophecodefile ICD10TOPHECODEFILE
                        Full path to the ICD10 to phecode map (NOTE: comma-
                        separated).
  --ICD10CMtophecodefile ICD10CMTOPHECODEFILE
                        Full path to the ICD10CM to phecode map (NOTE: comma-
                        separated).
  --targetphenotype TARGETPHENOTYPE
                        Name of the target phenotype.
  --excludephecodes EXCLUDEPHECODES
                        Path to a file containing the Phecode(s) corresponding
                        to and related to the target phenotype that should be
                        excluded from the analysis.
  --outdir OUTDIR       Output directory (must exist, default=./).
  --testfraction TESTFRACTION
                        Fraction of IDs used in test set (default=0.15).
                        Sampling at random.
  --testidfile TESTIDFILE
                        Full path to a file containing the IDs used in the
                        test set. If given, overrides --testfraction.
  --washout_window WASHOUT_WINDOW WASHOUT_WINDOW
                        Start and end dates for washout (default= 2009-01-01
                        2010-12-31).
  --exposure_window EXPOSURE_WINDOW EXPOSURE_WINDOW
                        Start and end dates for exposure (default = 1999-01-01
                        2008-12-31).
  --observation_window OBSERVATION_WINDOW OBSERVATION_WINDOW
                        Start and end dates for observation (default=
                        2011-01-01 2019-01-01).
  --minage MINAGE       Minimum age at the start of the observation period to
                        include (default=32).
  --maxage MAXAGE       Maximum age at the start of the observation period to
                        include (default=100).
  --seed SEED           Random number generator seed (default=42).
  --dateformatstr DATEFORMATSTR
                        Date format used (default='%Y-%m-%d')
  --excludeICDfile EXCLUDEICDFILE
                        Full path to a file containing the ICD codes to be
                        excluded. Format per row: ICD_version, ICD_code.
  --missing MISSING     Fraction of missing values allowed for a feature to be
                        included into the analysis (default=0.5).
  --frequency FREQUENCY
                        Minimum frequency for a predictor to be included into
                        the analysis (default=0.01).
  --controlfraction CONTROLFRACTION
                        How many controls to use, number of cases multiplied
                        by this flag value (default=None, meaning all controls
                        are used).
  --nproc NPROC         Number of parallel processes used (default=1).
  --paramgridfile PARAMGRIDFILE
                        File containing the parameter grid explored using
                        cross-validation. One parameter per row, first value
                        is key, rest are possible values.
```

### The required input files <a name="input"></a>

There are two types of input files needed, data files that are biobank-specific, and general input files detailing phecode definitions and parameters for elastic net model fitting. In the following, I will first go through the biobank-specific files.

```
  --phenotypefile PHENOTYPEFILE
                        Full path to an INTERVENE phenotype file. If last two
                        letters of file name are gz, gzipped file is assumed

```
This file must follow the INTERVENE phenotype file format described here: https://docs.google.com/document/d/1GbZszpPeyf-hyb0V_YDx828YbM7woh8OBJhvzkEwo2g/edit .

```
  --ICDfile ICDFILE     Full path to an INTERVENE longitudinal ICD code file.
                        If last two letters of file name are gz, gzipped file
                        is assumed.

```
This file must follow the INTERVENE ICD longitudinal file format described here: https://docs.google.com/document/d/1E2Jc72CmMItEchgQaCvfA4MhZUkQYjALwTu3dCl7qd8/edit .

These two above are the only files that need to be provided by the biobanks. The remaining required input files are provided with this repository and can be used as such. In the following I will indicate where to find these files.

```
  --phecodefile PHECODEFILE
                        Phecode definition file (NOTE: comma-separated).

```
This file can be found from `data/phecode_definitions1.2_manual_additions_v1.csv`.

```
  --ICD9tophecodefile ICD9TOPHECODEFILE
                        Full path to the ICD9 to phecode map (NOTE: comma-
                        separated).
```
This file can be found from `data/phecode_icd9_map_unrolled.csv`.
```
  --ICD10tophecodefile ICD10TOPHECODEFILE
                        Full path to the ICD10 to phecode map (NOTE: comma-
                        separated).
```
This file can be found from `data/Phecode_map_v1_2_icd10_beta_manual_additions_v1.csv`.
```
  --ICD10CMtophecodefile ICD10CMTOPHECODEFILE
                        Full path to the ICD10CM to phecode map (NOTE: comma-
                        separated).
```
This file can be found from `data/Phecode_map_v1_2_icd10cm_beta.csv`.

```
  --excludephecodes EXCLUDEPHECODES
                        Path to a file containing the Phecode(s) corresponding
                        to and related to the target phenotype that should be
                        excluded from the analysis.
```
These files are specific to endpoints, and have been pre-created based on FinnGen endpoint definitions. The files are in `data/ENDPOINTNAME-excluded-phecodes.txt`. Insert name of the endpoint in place of ENDPOINTNAME.

```
  --paramgridfile PARAMGRIDFILE
                        File containing the parameter grid explored using
                        cross-validation. One parameter per row, first value
                        is key, rest are possible values.
```
By default, you can use the file `data/INTERVENE_logreg_paramgrid_C3_breast.txt` for all endpoints.

### Output file descriptions <a name="output"></a>

All output is written into path defined by the following flag:

```
  --outdir OUTDIR       Output directory (must exist, default=./).
```

A file named `ENDPOINTNAME_PheRSwrapper.log` will be written to the root of `--outdir`. This file records the call string of `PheRSwrapper.py` as well as call strings of subsequent function calls within the script. Subsequent output is written into a directory created by `PheRSwrapper.py`, whose name is of format `ENDPOINTNAME_exposure=1999-01-01-2008-12-31-washoutend=2010-12-31-observationend=2019-01-01` (with the actual dates corresponding to the dates specified as input for `PheRSwrapper.py`). This folder contains more log files (`*.log`), the fitted model (`best_model.pkl`) and its coefficients (`best_model_coefficients.txt`) and hyperparameters (`best_model_hyperparameters.txt`). The predicted PheRS scores for the test set individuals are in file `pred_probas.txt.gz`, and the `*.png` files contain some classification performance measures computed on the test set. The file `target-ENDPOINTNAME-PheRS-ML-input.txt.gz` contains the preprocessed input data used for the actual model fitting and scoring.

### Calls to replicate the FinnGen analyses <a name="replicate"></a>

Here, I will give examples to replicate the analyses done in FinnGen. These should be directly replicable in ohter biobanks given that one replaces the file paths with file paths specific to your system. It is worth noting that the command line flag `--nproc` only affects the training of the elastic net models, and can be changed according to what type of computational resources are available.

#### Fitting PheRS for ages 32-70, no downsampling of controls

```
for e in {C3_COLORECTAL,C3_PROSTATE,AUD_SWEDISH,C3_BREAST,C3_BRONCHUS_LUNG,C3_CANCER,C3_MELANOMA_SKIN,C3_PROSTATE,COX_ARTHROSIS,E4_HYTHYNAS,E4_THYTOXGOITDIF,F5_DEPRESSIO,FE_STRICT,G6_AD_WIDE,G6_EPLEPSY,G6_SLEEPAPNO,GE_STRICT,GOUT,H7_GLAUCOMA,I9_ABAORTANEUR,I9_AF,I9_CHD,I9_HEARTFAIL_NS,I9_SAH,I9_STR,I9_THAORTANEUR,I9_VTE,ILD,IPF,J10_ASTHMA,K11_APPENDACUT,K11_IBD_STRICT,KNEE_ARTHROSIS,M13_OSTEOPOROSIS,N14_CHRONKIDNEYDIS,RHEUMA_SEROPOS_OTH,T1D,T2D}; do python /finngen/red/thartone/git/INTERVENE_PheRS/src/PheRSwrapper.py --ICDfile data/finngen_R8_detailed_longituinal_INTERVENE_format_ICD10_only_262022.txt --phenotypefile /finngen/red/Zhiyu/Share/Pheno_R8_HighEdu --phecodefile /finngen/red/thartone/git/INTERVENE_PheRS/data/phecode_definitions1.2_manual_additions.csv --ICD9tophecodefile /finngen/red/thartone/git/INTERVENE_PheRS/data/phecode_icd9_map_unrolled.csv --ICD10tophecodefile /finngen/red/thartone/git/INTERVENE_PheRS/data/Phecode_map_v1_2_icd10_beta_manual_additions.csv --ICD10CMtophecodefile /finngen/red/thartone/git/INTERVENE_PheRS/data/Phecode_map_v1_2_icd10cm_beta.csv --targetphenotype $e --excludephecodes /finngen/red/thartone/INTERVENE_pheRS/data/"$e"-excluded-phecodes.txt --outdir results_250123_with_age_no_downsampling/"$e"_ --nproc 6 --paramgridfile /finngen/red/thartone/git/INTERVENE_PheRS/data/INTERVENE_logreg_paramgrid_C3_breast.txt --testfraction 0.5 --includevars age --minage 32 --maxage 70; done;
```

#### Fitting PheRS for ages 32-70, downsampling 4 controls per each case

```
for e in {C3_COLORECTAL,C3_PROSTATE,AUD_SWEDISH,C3_BREAST,C3_BRONCHUS_LUNG,C3_CANCER,C3_MELANOMA_SKIN,C3_PROSTATE,COX_ARTHROSIS,E4_HYTHYNAS,E4_THYTOXGOITDIF,F5_DEPRESSIO,FE_STRICT,G6_AD_WIDE,G6_EPLEPSY,G6_SLEEPAPNO,GE_STRICT,GOUT,H7_GLAUCOMA,I9_ABAORTANEUR,I9_AF,I9_CHD,I9_HEARTFAIL_NS,I9_SAH,I9_STR,I9_THAORTANEUR,I9_VTE,ILD,IPF,J10_ASTHMA,K11_APPENDACUT,K11_IBD_STRICT,KNEE_ARTHROSIS,M13_OSTEOPOROSIS,N14_CHRONKIDNEYDIS,RHEUMA_SEROPOS_OTH,T1D,T2D}; do python /finngen/red/thartone/git/INTERVENE_PheRS/src/PheRSwrapper.py --ICDfile data/finngen_R8_detailed_longituinal_INTERVENE_format_ICD10_only_262022.txt --phenotypefile /finngen/red/Zhiyu/Share/Pheno_R8_HighEdu --phecodefile /finngen/red/thartone/git/INTERVENE_PheRS/data/phecode_definitions1.2_manual_additions.csv --ICD9tophecodefile /finngen/red/thartone/git/INTERVENE_PheRS/data/phecode_icd9_map_unrolled.csv --ICD10tophecodefile /finngen/red/thartone/git/INTERVENE_PheRS/data/Phecode_map_v1_2_icd10_beta_manual_additions.csv --ICD10CMtophecodefile /finngen/red/thartone/git/INTERVENE_PheRS/data/Phecode_map_v1_2_icd10cm_beta.csv --targetphenotype $e --excludephecodes /finngen/red/thartone/INTERVENE_pheRS/data/"$e"-excluded-phecodes.txt --outdir results_250123_with_age/"$e"_ --controlfraction 4.0 --nproc 6 --paramgridfile /finngen/red/thartone/git/INTERVENE_PheRS/data/INTERVENE_logreg_paramgrid_C3_breast.txt --testfraction 0.5 --includevars age --minage 32 --maxage 70; done;
```

#### Fitting PheRS for ages 32-50, no downsamplling of controls

```
for e in {C3_COLORECTAL,C3_PROSTATE,AUD_SWEDISH,C3_BREAST,C3_BRONCHUS_LUNG,C3_CANCER,C3_MELANOMA_SKIN,C3_PROSTATE,COX_ARTHROSIS,E4_HYTHYNAS,E4_THYTOXGOITDIF,F5_DEPRESSIO,FE_STRICT,G6_AD_WIDE,G6_EPLEPSY,G6_SLEEPAPNO,GE_STRICT,GOUT,H7_GLAUCOMA,I9_ABAORTANEUR,I9_AF,I9_CHD,I9_HEARTFAIL_NS,I9_SAH,I9_STR,I9_THAORTANEUR,I9_VTE,ILD,IPF,J10_ASTHMA,K11_APPENDACUT,K11_IBD_STRICT,KNEE_ARTHROSIS,M13_OSTEOPOROSIS,N14_CHRONKIDNEYDIS,RHEUMA_SEROPOS_OTH,T1D,T2D}; do python /finngen/red/thartone/git/INTERVENE_PheRS/src/PheRSwrapper.py --ICDfile data/finngen_R8_detailed_longituinal_INTERVENE_format_ICD10_only_262022.txt --phenotypefile /finngen/red/Zhiyu/Share/Pheno_R8_HighEdu --phecodefile /finngen/red/thartone/git/INTERVENE_PheRS/data/phecode_definitions1.2_manual_additions.csv --ICD9tophecodefile /finngen/red/thartone/git/INTERVENE_PheRS/data/phecode_icd9_map_unrolled.csv --ICD10tophecodefile /finngen/red/thartone/git/INTERVENE_PheRS/data/Phecode_map_v1_2_icd10_beta_manual_additions.csv --ICD10CMtophecodefile /finngen/red/thartone/git/INTERVENE_PheRS/data/Phecode_map_v1_2_icd10cm_beta.csv --targetphenotype $e --excludephecodes /finngen/red/thartone/INTERVENE_pheRS/data/"$e"-excluded-phecodes.txt --outdir results_250123_with_age_young_no_downsampling/"$e"_ --nproc 6 --paramgridfile /finngen/red/thartone/git/INTERVENE_PheRS/data/INTERVENE_logreg_paramgrid_C3_breast.txt --testfraction 0.5 --includevars age --minage 32 --maxage 50; done;

```

#### Fitting PheRS for ages 32-50, downsampling 4 controls per each case

```
for e in {C3_COLORECTAL,C3_PROSTATE,AUD_SWEDISH,C3_BREAST,C3_BRONCHUS_LUNG,C3_CANCER,C3_MELANOMA_SKIN,C3_PROSTATE,COX_ARTHROSIS,E4_HYTHYNAS,E4_THYTOXGOITDIF,F5_DEPRESSIO,FE_STRICT,G6_AD_WIDE,G6_EPLEPSY,G6_SLEEPAPNO,GE_STRICT,GOUT,H7_GLAUCOMA,I9_ABAORTANEUR,I9_AF,I9_CHD,I9_HEARTFAIL_NS,I9_SAH,I9_STR,I9_THAORTANEUR,I9_VTE,ILD,IPF,J10_ASTHMA,K11_APPENDACUT,K11_IBD_STRICT,KNEE_ARTHROSIS,M13_OSTEOPOROSIS,N14_CHRONKIDNEYDIS,RHEUMA_SEROPOS_OTH,T1D,T2D}; do python /finngen/red/thartone/git/INTERVENE_PheRS/src/PheRSwrapper.py --ICDfile data/finngen_R8_detailed_longituinal_INTERVENE_format_ICD10_only_262022.txt --phenotypefile /finngen/red/Zhiyu/Share/Pheno_R8_HighEdu --phecodefile /finngen/red/thartone/git/INTERVENE_PheRS/data/phecode_definitions1.2_manual_additions.csv --ICD9tophecodefile /finngen/red/thartone/git/INTERVENE_PheRS/data/phecode_icd9_map_unrolled.csv --ICD10tophecodefile /finngen/red/thartone/git/INTERVENE_PheRS/data/Phecode_map_v1_2_icd10_beta_manual_additions.csv --ICD10CMtophecodefile /finngen/red/thartone/git/INTERVENE_PheRS/data/Phecode_map_v1_2_icd10cm_beta.csv --targetphenotype $e --excludephecodes /finngen/red/thartone/INTERVENE_pheRS/data/"$e"-excluded-phecodes.txt --outdir results_250123_with_age_young/"$e"_ --controlfraction 4.0 --nproc 6 --paramgridfile /finngen/red/thartone/git/INTERVENE_PheRS/data/INTERVENE_logreg_paramgrid_C3_breast.txt --testfraction 0.5 --includevars age --minage 32 --maxage 50; done;
```

#### Fitting PheRS for ages 51-70, no downsamplling of controls

```
for e in {C3_COLORECTAL,C3_PROSTATE,AUD_SWEDISH,C3_BREAST,C3_BRONCHUS_LUNG,C3_CANCER,C3_MELANOMA_SKIN,C3_PROSTATE,COX_ARTHROSIS,E4_HYTHYNAS,E4_THYTOXGOITDIF,F5_DEPRESSIO,FE_STRICT,G6_AD_WIDE,G6_EPLEPSY,G6_SLEEPAPNO,GE_STRICT,GOUT,H7_GLAUCOMA,I9_ABAORTANEUR,I9_AF,I9_CHD,I9_HEARTFAIL_NS,I9_SAH,I9_STR,I9_THAORTANEUR,I9_VTE,ILD,IPF,J10_ASTHMA,K11_APPENDACUT,K11_IBD_STRICT,KNEE_ARTHROSIS,M13_OSTEOPOROSIS,N14_CHRONKIDNEYDIS,RHEUMA_SEROPOS_OTH,T1D,T2D}; do python /finngen/red/thartone/git/INTERVENE_PheRS/src/PheRSwrapper.py --ICDfile data/finngen_R8_detailed_longituinal_INTERVENE_format_ICD10_only_262022.txt --phenotypefile /finngen/red/Zhiyu/Share/Pheno_R8_HighEdu --phecodefile /finngen/red/thartone/git/INTERVENE_PheRS/data/phecode_definitions1.2_manual_additions.csv --ICD9tophecodefile /finngen/red/thartone/git/INTERVENE_PheRS/data/phecode_icd9_map_unrolled.csv --ICD10tophecodefile /finngen/red/thartone/git/INTERVENE_PheRS/data/Phecode_map_v1_2_icd10_beta_manual_additions.csv --ICD10CMtophecodefile /finngen/red/thartone/git/INTERVENE_PheRS/data/Phecode_map_v1_2_icd10cm_beta.csv --targetphenotype $e --excludephecodes /finngen/red/thartone/INTERVENE_pheRS/data/"$e"-excluded-phecodes.txt --outdir results_250123_with_age_old_no_downsampling/"$e"_ --nproc 6 --paramgridfile /finngen/red/thartone/git/INTERVENE_PheRS/data/INTERVENE_logreg_paramgrid_C3_breast.txt --testfraction 0.5 --includevars age --minage 51 --maxage 70; done;
```

#### Fitting PheRS for ages 51-70, downsampling 4 controls per each case

```
for e in {C3_COLORECTAL,C3_PROSTATE,AUD_SWEDISH,C3_BREAST,C3_BRONCHUS_LUNG,C3_CANCER,C3_MELANOMA_SKIN,C3_PROSTATE,COX_ARTHROSIS,E4_HYTHYNAS,E4_THYTOXGOITDIF,F5_DEPRESSIO,FE_STRICT,G6_AD_WIDE,G6_EPLEPSY,G6_SLEEPAPNO,GE_STRICT,GOUT,H7_GLAUCOMA,I9_ABAORTANEUR,I9_AF,I9_CHD,I9_HEARTFAIL_NS,I9_SAH,I9_STR,I9_THAORTANEUR,I9_VTE,ILD,IPF,J10_ASTHMA,K11_APPENDACUT,K11_IBD_STRICT,KNEE_ARTHROSIS,M13_OSTEOPOROSIS,N14_CHRONKIDNEYDIS,RHEUMA_SEROPOS_OTH,T1D,T2D}; do python /finngen/red/thartone/git/INTERVENE_PheRS/src/PheRSwrapper.py --ICDfile data/finngen_R8_detailed_longituinal_INTERVENE_format_ICD10_only_262022.txt --phenotypefile /finngen/red/Zhiyu/Share/Pheno_R8_HighEdu --phecodefile /finngen/red/thartone/git/INTERVENE_PheRS/data/phecode_definitions1.2_manual_additions.csv --ICD9tophecodefile /finngen/red/thartone/git/INTERVENE_PheRS/data/phecode_icd9_map_unrolled.csv --ICD10tophecodefile /finngen/red/thartone/git/INTERVENE_PheRS/data/Phecode_map_v1_2_icd10_beta_manual_additions.csv --ICD10CMtophecodefile /finngen/red/thartone/git/INTERVENE_PheRS/data/Phecode_map_v1_2_icd10cm_beta.csv --targetphenotype $e --excludephecodes /finngen/red/thartone/INTERVENE_pheRS/data/"$e"-excluded-phecodes.txt --outdir results_250123_with_age_old/"$e"_ --controlfraction 4.0 --nproc 6 --paramgridfile /finngen/red/thartone/git/INTERVENE_PheRS/data/INTERVENE_logreg_paramgrid_C3_breast.txt --testfraction 0.5 --includevars age --minage 51 --maxage 70; done;
```

## Overview of changelog <a name="changes"></a>

### Update corresponding to the version for replication outside FinnGen

Tuomo Hartonen 31.1.2023

Main updates to previous version

* Greatly extended documentation, aimed at replication of the PheRS training in other biobanks.

* Some bug fixes, for example fixed a bug that lead to wrong IDs being written for the pred_probas-files that contain predicted PheRS scores for test set individuals.

### Update corresponding to the first run of PheRS results in FinnGen

Tuomo Hartonen 2.12.2022

Main updates to previous version

* ICD10 to phecode mapping was updated so that the ICD10 code M50.8 with no matching phecode was by manual curation set to map to phecode 714.1: Rheumatoid arthritis.

* PheRS models have been fitted in FinnGen for all endpoints and the results can be viewed from the INTERVENE flagship Google Drive under folder `PheRS-project`.

* Several minor changes to the study setup were made so that everything is comparable with the setup used by Kira. Detailed description of the study setup can also be found from INTERVENE flagship Google Drive in the same folder.

### Update after fitting first models in FinnGen

Tuomo Hartonen 5.4.2022

Main updates to previous version

* The preprocessing, fitting and scoring scripts have now been tested with one endpoint on FinnGen data, and seem to work for that.

* folder data/ contains files mapping ICD codes to phecodes, a list of ICD codes that map to multiple phecodes and a file listing all related endpoints for each FinnGen endpoint made by Lisa Eick. This file is used to define what predictors are excluded from the models, by mapping the FinnGen endpoints back to ICD10 codes and from there to phecodes.

There were only three one-to-many mapping ICD10 codes with more than 100 longitudinal entries in FinnGen. We have decided to represent these as new custom phecodes that are added to the file ´phecode_definitions1.2_manual_additions.csv´. These added entries are:

> "phecode","phenotype","phecode_exclude_range","sex","rollup","leaf","category_number","category"
> "1200","Exfoliative dermatitis","816-819.99","Both","","","",""
> "1201","Gonarthrosis [arthrosis of knee]","710-716.99","Both","","","",""
> "1202","Sequelae of injuries of head","816-819.99","Both","","","",""

Notice that at this point, only ICD10 codes have been used in the analyses and other phecode maps have not been utilized.

### Initial upload

Tuomo Hartonen 21.2.2022

This repository is under construction and the scripts are not guaranteed to work. For each script, get help by running python script_name.py -h.
