##############################################################
# import data and create a smaller sample with
# balanced class proportions
##############################################################

library('tidyverse')

# read in dataset
all_data = read.csv('creditcard-1.csv')
# check dimensions
dim(all_data)
# check structure
str(all_data)

# identify missing values
sapply(all_data, function(x) sum(is.na(x)))

# review counts of non-fraud and fraud
table(all_data$Class)

# identify fraud rate in original dataset
(sum(all_data$Class)/nrow(all_data))

# set random seed for sampling
set.seed(5)

# create a smaller sample, with balanced class proportions
fraud_obs = subset(all_data, all_data$Class == 1)
nonfraud_obs = subset(all_data, all_data$Class == 0)
fraud_sample = fraud_obs[sample(nrow(fraud_obs), 492, replace = FALSE, prob = NULL), ]
nonfraud_sample = nonfraud_obs[sample(nrow(nonfraud_obs), 492, replace = FALSE, prob = NULL), ]
bal_sample = rbind(fraud_sample,nonfraud_sample)

# identify fraud rate in sample
(sum(bal_sample$Class)/nrow(bal_sample))

# split into training and test sets
s_size = floor(0.8 * nrow(bal_sample))
bal_train_idx = sample(seq_len(nrow(bal_sample)), size = s_size)
bal_sample_train = bal_sample[bal_train_idx, ]
bal_sample_test = bal_sample[-bal_train_idx,]

# fraud and non fraud counts in training and test
table(bal_sample_train$Class)
table(bal_sample_test$Class)

##############################################################
# Exploratory Data Analysis
##############################################################  

library(corrplot)
corr_matrix = cor(bal_sample_train[ ,-c(31)])
png(filename="corr_plot.jpeg")
corrplot(corr_matrix, method = "circle")
dev.off()

library(GGally)
png(filename="parallel_plot.jpeg")
ggparcoord(bal_sample_train, columns=c(1:30), groupColumn=31)
dev.off()
        
##############################################################
# Generate MCMC Chains
##############################################################  

require(runjags)

# set model parameters
fileNameRoot = "creditcard-" 
numSavedSteps=5000
thinSteps=2
nChains = 2
parameters = c( "beta0" ,  "beta" ,  
                "zbeta0" , "zbeta" )
adaptSteps = 500  
burnInSteps = 1000

# the data
y = bal_sample_train[ ,31]
x = as.matrix(bal_sample_train[ ,1:30])
dataList = list(
        x = x ,
        y = y ,
        Nx = dim(x)[2] ,
        Ntotal = dim(x)[1]
)

# the model
modelString = "
        # Standardize the data:
        data {
                for ( j in 1:Nx ) {
                        xm[j]  <- mean(x[,j])
                        xsd[j] <-   sd(x[,j])
                        for ( i in 1:Ntotal ) {
                                zx[i,j] <- ( x[i,j] - xm[j] ) / xsd[j]
                        }
                }
        }
        
        # Specify the model for standardized data:
        model {
                for ( i in 1:Ntotal ) {
                        # In JAGS, ilogit is logistic:
                        y[i] ~ dbern( ilogit( zbeta0 + sum( zbeta[1:Nx] * zx[i,1:Nx] ) ) )
                }
        # Priors vague on standardized scale:
                zbeta0 ~ dnorm( 0 , 1/2^2 )  
                for ( j in 1:Nx ) {
                        zbeta[j] ~ dnorm( 0 , 1/2^2 )
                }
        
        # Transform to original scale:
                beta[1:Nx] <- zbeta[1:Nx] / xsd[1:Nx] 
                beta0 <- zbeta0 - sum( zbeta[1:Nx] * xm[1:Nx] / xsd[1:Nx] )
        }"
# close quote for modelString
# Write out modelString to a text file
writeLines( modelString , con="TEMPmodel" )

# Generate the MCMC chain:
startTime = proc.time()
runJagsOut = run.jags(method='parallel',
                        model="TEMPmodel" , 
                        monitor=parameters , 
                        data=dataList , 
                        n.chains=nChains ,
                        adapt=adaptSteps ,
                        burnin=burnInSteps , 
                        sample=ceiling(numSavedSteps/nChains) ,
                        thin=thinSteps ,
                        summarise=FALSE ,
                        plots=FALSE )
stopTime = proc.time()
(duration = stopTime - startTime)

##############################################################
# MCMC Chain Diagnostics 
##############################################################

# Load the relevant scripts
#source("Jags-Ydich-XmetMulti-Mlogistic.R")
source("DBDA2E-utilities.R")

# save MCMC chain as variable
mcmcCoda = as.mcmc.list( runJagsOut )

# obtain parameter names
parameterNames = varnames(mcmcCoda) 
# analyze the chains for the first 10 parameters
parameterNames = parameterNames[1:12]
for ( parName in parameterNames ) {
        diagMCMC( codaObject=mcmcCoda , parName=parName)
        saveGraph( file=as.character(parName) , type="jpeg")
}

##############################################################
# GLM Logistic Regression with MCMC
############################################################## 

# get summary statistics of chain
bal_summaryInfo = smryMCMC(mcmcCoda, saveName=fileNameRoot)
bal_summaryInfo = data.frame(bal_summaryInfo)

# mcmc summary info and coefficient estimates
bal_logreg_coeffs = bal_summaryInfo[1:31,'Median']
#bal_logreg_coeffs = bal_summaryInfo[1:31,'Mode']
(round(bal_logreg_coeffs, 4))

# implement logistic regression classifier
# start by adding a dummy column for the y intercept coefficient
bal_sample_test$X_0 = 1
# reorder columns and drop response variable
X_bal_sample_test = bal_sample_test[,c("X_0","Time","V1","V2","V3","V4","V5","V6","V7","V8","V9"
                                           ,"V10","V11","V12","V13","V14","V15","V16","V17","V18"
                                           ,"V19","V20","V21","V22","V23","V24","V25","V26","V27"
                                           ,"V28","Amount")]
# convert dataframe to a matrix for linear algebra
X_bal_test_matrix =  data.matrix(X_bal_sample_test)
# logistic regression model
predictions = c()
for(i in 1:nrow(X_bal_test_matrix)){
        # mean function
        temp_result = (-1 * X_bal_test_matrix[i,]) %*% bal_logreg_coeffs
        # link function
        mean_func_result = 1 / (1 + exp(temp_result))
        # append predictions
        predictions = append(predictions, log(1 / (1 - mean_func_result)))
}
# analyze accuracy
bal_predictions = data.frame(logreg_output=predictions)
bal_predictions$predictions = ifelse(bal_predictions$logreg_output >= .5, 1, 0)
#bal_predictions$predictions = ifelse(bal_predictions$logreg_output >= .25, 1, 0)
bal_predictions$actual = bal_sample_test$Class
bal_predictions$correct = ifelse(bal_predictions$actual==bal_predictions$predictions, 1, 0)
sum(bal_predictions$correct) / nrow(bal_predictions)

# confusion matrix
library(caret)
bm_cmatrix = confusionMatrix(bal_predictions$predictions,bal_predictions$actual)
bm_cmatrix$table
bm_cmatrix$byClass

# ROC curve and AUC
library(pROC)
roc_obj = roc(bal_predictions$actual,bal_predictions$predictions)
auc(roc_obj)

##############################################################
# regular GLM logistic regression
############################################################## 

# create model
m1 = glm(as.factor(Class) ~ ., data=bal_sample_train, family="binomial")
round(m1$coefficients, 4)
# predictions
fitted.results.m1 = predict(m1,bal_sample_test,type='response')

# analyze accuracy
bal_predictions_lr = data.frame(logreg_output=fitted.results.m1)
bal_predictions_lr$predictions = ifelse(bal_predictions_lr$logreg_output >= .5, 1, 0)
bal_predictions_lr$actual = bal_sample_test$Class
bal_predictions_lr$correct = ifelse(bal_predictions_lr$actual==bal_predictions_lr$predictions, 1, 0)
sum(bal_predictions_lr$correct) / nrow(bal_predictions_lr)

# confusion matrix
m1_cmatrix = confusionMatrix(bal_predictions_lr$predictions,bal_predictions_lr$actual)
m1_cmatrix$table
m1_cmatrix$byClass

# ROC curve and AUC
roc_obj = roc(bal_predictions_lr$actual,bal_predictions_lr$predictions)
auc(roc_obj)

##############################################################
# support vector machine
############################################################## 

# create model
library(e1071)
m2 = svm(as.factor(Class) ~ ., data=bal_sample_train,probability=TRUE)
round(m1$coefficients, 4)
fitted.results.m2 = predict(m2,bal_sample_test,probability=TRUE)

# analyze accuracy
bal_predictions_svm = data.frame(svm_output=fitted.results.m2)
bal_predictions_svm$predictions = as.numeric(as.character(bal_predictions_svm$svm_output))
bal_predictions_svm$actual = bal_sample_test$Class
bal_predictions_svm$correct = ifelse(bal_predictions_svm$actual==bal_predictions_svm$predictions, 1, 0)
sum(bal_predictions_svm$correct) / nrow(bal_predictions_svm)

# confusion matrix
m2_cmatrix = confusionMatrix(bal_predictions_svm$predictions,bal_predictions_svm$actual)
m2_cmatrix$table
m2_cmatrix$byClass

# AUC
roc_obj = roc(bal_predictions_svm$actual,bal_predictions_svm$predictions)
auc(roc_obj)

##############################################################
# random forest
############################################################## 

# create model
require(randomForest)
require(MASS)
m3 = randomForest(as.factor(Class) ~ .,data=bal_sample_train,ntree=500)
fitted.results.m3 = predict(m3,bal_sample_test,type="prob")
fitted.results.m3 = data.frame(fitted.results.m3)

# analyze accuracy
bal_predictions_rf = data.frame(rf_output=fitted.results.m3$X1)
bal_predictions_rf$predictions = ifelse(bal_predictions_rf$rf_output >= .5, 1, 0)
bal_predictions_rf$actual = bal_sample_test$Class
bal_predictions_rf$correct = ifelse(bal_predictions_rf$actual==bal_predictions_rf$predictions, 1, 0)
sum(bal_predictions_rf$correct) / nrow(bal_predictions_rf)

# confusion matrix
m3_cmatrix = confusionMatrix(bal_predictions_rf$predictions,bal_predictions_rf$actual)
m3_cmatrix$table
m3_cmatrix$byClass

# AUC
roc_obj = roc(bal_predictions_rf$actual,bal_predictions_rf$predictions)
auc(roc_obj)

##################################################################
# significance test between model accuracy and type II error rate
##################################################################

# mcnemar's chi-squared test on overall accuracy
# bayesian LR GLM vs SVM
# SVM had next highest overall accuracy
acc_results = matrix(c(198, 9, 193, 14),
                 nrow = 2,
                 dimnames = list("Bayesian LR GLM" = c("Correct", "Incorrect"),
                                 "SVM" = c("Correct", "Incorrect")))
mcnemar.test(acc_results)

# mcnemar's chi-squared test on false negative rate
# bayesian LR GLM vs Random Forest
# Random Forest had next lowest false negative rate
fnr_results = matrix(c(7, 96, 9, 94),
                     nrow = 2,
                     dimnames = list("Bayesian LR GLM" = c("FNR", "TPR"),
                     "RF" = c("FNR", "TPR")))
mcnemar.test(fnr_results)

