# INTERVENE_PheRS

## Usage instructions

This section is under construction. The Python script

```
src/PheRSwrapper.py
```

converts ICD10 codes input int the INTERVENE longitudinal file format into phecodes and preprocesses data for PheRS model fitting. It then fits the PheRS-model (elastic net) and makes predictions on a test set also reporting prediction performance. The script also writes several log files that list all the files and parameter settings of the run. More detailed usage instructions will follow later.

## Update corresponding to the first run of PheRS results in FinnGen

Tuomo Hartonen 2.12.2022

Main updates to previous version

* ICD10 to phecode mapping was updated so that the ICD10 code M50.8 with no matching phecode was by manual curation set to map to phecode 714.1: “Rheumatoid arthritis”.

* PheRS models have been fitted in FinnGen for all endpoints and the results can be viewed from the INTERVENE flagship Google Drive under folder `PheRS-project`.

* Several minor changes to the study setup were made so that everything is comparable with the setup used by Kira. Detailed description of the study setup can also be found from INTERVENE flagship Google Drive in the same folder.

## Update after fitting first models in FinnGen

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

## Initial upload

Tuomo Hartonen 21.2.2022

This repository is under construction and the scripts are not guaranteed to work. For each script, get help by running python script_name.py -h.
