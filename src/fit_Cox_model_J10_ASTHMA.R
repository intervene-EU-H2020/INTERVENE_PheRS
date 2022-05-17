#load needed libraries
library(data.table)
library(dplyr)
library(lubridate)
library(survival)

#Read in the datasets
phers <- fread(input="/finngen/red/thartone/INTERVENE/data/finngen_R8/J10_ASTHMA_052022-balanced-noPCs/090522-J10_ASTHMA-pred_probas.txt.gz")
prs <- fread(input="/finngen/red/Bradley/INTERVENE/Results/MegaPRS/PRS_HapMap/Asthma_PRS_hm3.sscore",select=c("IID","SCORE1_AVG"),col.names=c("ID","PRS"))
covariates <- fread(input="/finngen/red/thartone/INTERVENE/data/finngen_R8/J10_ASTHMA_052022/090522-J10_ASTHMA-target-J10_ASTHMA-PheRS-ML-input.txt.gz",select=c("#ID","follow_up_time","sex","PC1","PC2","PC3","PC4","PC5","PC6","PC7","PC8","PC9","PC10"))

#read in the phenotype file
pheno <- fread(input="/finngen/red/Zhiyu/Share/Phenotype/FinnGenR8_Phenotype", select=c("ID","DATE_OF_BIRTH","PC1","PC2","PC3","PC4","PC5","PC6","PC7","PC8","PC9","PC10","J10_ASTHMA","J10_ASTHMA_DATE","END_OF_FOLLOWUP","ANCESTRY"))
batch <- fread(input="/finngen/library-red/finngen_R8/analysis_covariates/finngen_R8_cov_1.0.txt.gz", select=c("FINNGENID","batch"), data.table=FALSE,col.names=c("ID","BATCH"))
pheno[["J10_ASTHMA_DATE"]] <- as.Date(pheno[["J10_ASTHMA_DATE"]],format='%Y-%m-%d')

pheno <- merge(pheno,prs,by='ID')
pheno <- merge(pheno,batch,by='ID')

#First preprocess PRS data
pheno <- subset(pheno, !is.na(pheno[["PRS"]]))

#Subset to those of european ancestry/those that have principal components calculated for EUROPEAN ancestry, i.e. within ancestry principal components, not global genetic principal components.
#As we have been unable to use the standardised method for computing ancestry, if you have this information available from your centralised QC please use this. 
#Feel free to subset using your own code: only provided as a reminder.
pheno <- subset(pheno, ANCESTRY=='EUR')

#Assign PRS into percentiles
q_prs <- quantile(pheno[["PRS"]], probs=c(0,0.01,0.05,0.1,0.2,0.4,0.6,0.8,0.9,0.95,0.99,1))
pheno[["PRS_group"]] <- cut(pheno[["PRS"]], q_prs, include.lowest=TRUE,
                                            labels=paste("Group",1:11))

#Make all necessary variables factors
pheno$BATCH <- as.factor(pheno$BATCH)
pheno[["PRS_group"]] <- as.factor(pheno[["PRS_group"]])
pheno[["PRS_group"]] <- relevel(pheno[["PRS_group"]], ref="Group 6")

#Specify age as either the Age at Onset or End of Follow-up (if not a case)
pheno$AGE <- ifelse(pheno[["J10_ASTHMA"]]==1, time_length(difftime(pheno[["J10_ASTHMA_DATE"]], pheno$DATE_OF_BIRTH), 'years'), time_length(difftime(pheno$END_OF_FOLLOWUP, pheno$DATE_OF_BIRTH), 'years'))
#Adjust to censor at age between 0 and 50
pheno[["J10_ASTHMA"]] <- ifelse(pheno[["J10_ASTHMA"]]==1 & pheno$AGE > 50, 0, pheno[["J10_ASTHMA"]])
#pheno$AGE <- ifelse(pheno$AGE > 80, 80, pheno$AGE)
pheno <- pheno[pheno$AGE>=42]
#Then preprocess PheRS data
#merge dataframes
df_phers <- merge(phers,covariates,by='#ID')

#Add a column describing the PheRS/PRS risk groups
q_phers <- quantile(phers[["pred_class1_prob"]], probs=c(0,0.01,0.05,0.1,0.2,0.4,0.6,0.8,0.9,0.95,0.99,1))
df_phers[["PheRS_group"]] <- cut(phers[["pred_class1_prob"]], q_phers, include.lowest=TRUE,labels=paste("Group",1:11))

#Make all necessary variables factors
df_phers[["PheRS_group"]] <- as.factor(df_phers[["PheRS_group"]])
df_phers[["PheRS_group"]] <- relevel(df_phers[["PheRS_group"]], ref="Group 6")

#fit the Cox models
survival_phers <- coxph(as.formula(paste0("Surv(follow_up_time,","true_class",") ~ PheRS_group + PC1 + PC2 + PC3 + PC4 + PC5 + PC6 + PC7 + PC8 + PC9 + PC10")), data=df_phers, na.action=na.exclude)
survival_prs <- coxph(as.formula(paste0("Surv(AGE,","J10_ASTHMA",") ~ PRS_group + PC1 + PC2 + PC3 + PC4 + PC5 + PC6 + PC7 + PC8 + PC9 + PC10 + BATCH")), data=pheno, na.action=na.exclude)

#count the number of cases and controls per PheRS risk group
N_cases_phers <- c()
for (i in 1:11)
{
  if(i==6) next
  N_cases_phers <- rbind(N_cases_phers,nrow(subset(df_phers,true_class==1 & PheRS_group==paste0("Group ",i))))
}

N_controls_phers <- c()
for (i in 1:11)
{
  if(i==6) next
  N_controls_phers <- rbind(N_controls_phers,nrow(subset(df_phers,true_class==0 & PheRS_group==paste0("Group ",i))))
}

#Extract hazard ratios, betas, standard errors and p-vals
phenotype <- rep("J10_ASTHMA",10)
phers <- rep("PheRS",10)
group <- c(paste0("PheRS_groupGroup ",c(1:5,7:11)))
betas <- summary(survival)$coefficients[group,"coef"]
std_errs <- summary(survival)$coefficients[group,"se(coef)"]
pvals <- summary(survival)$coefficients[group,"Pr(>|z|)"]
OR <- exp(betas)
CIpos <- exp(betas+1.96*std_errs)
CIneg <- exp(betas-1.96*std_errs)
result <- matrix(c(phenotype, phers, group, N_controls, N_cases, betas, std_errs, pvals, OR, CIpos, CIneg), nrow=10, ncol=11)
colnames(result) <- c('Endpoint','Score','Group','N_controls','N_cases','beta','std_err','p-val','HR','CI_pos','CI_neg')
#save results to a file
write.csv(result,'/finngen/red/thartone/INTERVENE/data/finngen_R8/J10_ASTHMA_052022-balanced-noPCs/Cox_model_results_PheRS.csv')

#then the same for PRS

#Define number of cases and controls in each PRS group.
controls <- table(pheno[["PRS_group"]], pheno[["J10_ASTHMA"]])[2:11,1]
cases <- if(sum(nrow(pheno[pheno[["J10_ASTHMA"]]==0,]))==length(pheno[["J10_ASTHMA"]])){
  rep(0,10)} else {table(pheno[["PRS_group"]], pheno[["J10_ASTHMA"]])[2:11,2]}
#Extract hazard ratios, betas, standard errors and p-vals
phenotype <- rep("J10_ASTHMA",10)
prs <- rep("PRS",10)
group <- c(paste0("PRS_groupGroup ",c(1:5,7:11)))
betas <- summary(survival_prs)$coefficients[group,"coef"]
std_errs <- summary(survival_prs)$coefficients[group,"se(coef)"]
pvals <- summary(survival_prs)$coefficients[group,"Pr(>|z|)"]
OR <- exp(betas)
CIpos <- exp(betas+1.96*std_errs)
CIneg <- exp(betas-1.96*std_errs)
result <- matrix(c(phenotype, prs, group, controls, cases, betas, std_errs, pvals, OR, CIpos, CIneg), nrow=10, ncol=11)
results <- rbind(results, result)
colnames(result) <- c('Endpoint','Score','Group','N_controls','N_cases','beta','std_err','p-val','HR','CI_pos','CI_neg')
#save results to a file
write.csv(result,'/finngen/red/thartone/INTERVENE/data/finngen_R8/J10_ASTHMA_052022-balanced-noPCs/Cox_model_results_PRS.csv')