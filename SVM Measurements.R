# Machine Learning Code for Processing Measurements
#
# Code written by:
#
# Lloyd A. Courtenay - ladc1995@gmail.com       Universidad Rovira i Virgili [URV]
#                                               Institut de Paleocologia Humana i Evolució Social [IPHES]
#                                               Universidad Complutense de Madrid [UCM]
#
# Code for Yravedra et al. (In Prep)
#
#
# Written - 15/04/2019
# Last Update - 04/05/2019

# Load packages and set up workspace -----------------------------------------------------------------------------------------------------

library(e1071)
library(caret)

# Measurements ---------------------------------------------------------------------------------------------------------------------------

svm_data<-read.csv(file.choose(), header = TRUE) # Load comma delimited csv file containing measurements

# Prepare data to train classification model---------------------------------------------------------------------------------------------------

set.seed(1000) # Set seed for random number generator

boot<-svm_data[sample(nrow(svm_data), size = 1000, replace = TRUE),] # Bootstrap sample for training

# Create function to split data into test and training sets

split.data = function(data,p = 0.7, s = 666) {
  set.seed(2)
  index = sample (1:dim(data)[1])
  train = data [index[1: floor(dim(data)[1]*p)],]
  test = data [index[((ceiling(dim(data)[1]*p)) + 1) :dim(data)[1]],]
  return(list(train = train, test = test))}

# Split into training and test sets

allset<-split.data(boot, p = 0.7) # split data 70% for training
trainset<-allset$train # Allocate 70% to training set
testset<-allset$test # Allocate 30% to test set
ctrl<-trainControl(method = "repeatedcv", repeats = 10) # Set the control funtion to k = 10 fold Cross Validation

# If classification is performed without OA execute following lines of code
trainset<-trainset[-c(8)]
testset<-testset[-c(8)]

# Model optimization ----------------------------------------------------------------------------------------------------------------

# Create loop function to carry out search for optimal hyperparameters

best_score<-0  # Set initial score to 0 - this will be updated during tuning untill reaching best model performance
C<-c(0); gamma<-c(0); best_params<-data.frame(C,gamma) # Set initial Values to 0 - these will be updated as before

for (i in 50) { # Run the tuning process for 50 iterations
  svm_tune<-svm(Sample~., data = trainset, kernel = "radial", # SVM model using a radial kernel
                cost = runif(1, min = 0, max = 500), # run a random number per iteration with cost values between 0 and 100
                gamma = runif(1, min = 0, max = 500), # run a random number per iteration with gamma values between 0 and 100
                trControl = ctrl, probability = TRUE) # Use k fold cross validation and allow for probability calculations
  svm.fit<-predict(svm_tune, testset[, !names(testset) %in% c("Sample")]) # Fit tuned model on testset
  conmax<-confusionMatrix(table(svm.fit,testset$Sample)) # Evaluate model performance
  score<-conmax$overall["Accuracy"] # Extract the accuracy of the tested model
  if (score > best_score) { # Update previously defined values using the best values obtained during tuning
    best_score <- score # Save the best accuracy value obtained
    best_cost <- svm_tune$cost # Save the optimal cost value used to obtain the best accuracy score
    best_gamma <- svm_tune$gamma # Save the optimal gamma value used to obtain the best accuracy score
  }
  optimal_model <- data.frame(best_score,best_cost,best_gamma) # Create table with best values
}; optimal_model # Print the optimal model hyperparameters

# Create final model ---------------------------------------------------------------------------------------------------------------

svm.model<-svm(Sample~., data = trainset, kernel = "radial",
               cost = optimal_model$best_cost, # Choosing the best cost values found in tuning
               gamma = optimal_model$best_gamma, # Choosing the best gamma values found in tuning
               trControl = ctrl, probability = TRUE) # Use k fold cross validation and allow for probability calculations
summary(svm.model) # View final model constructed

# Evaluate final model performance

svm.predict<-predict(svm.model, testset[, !names(testset) %in% c("Sample")]) # Predict labels for test data
svm.table<-table(svm.predict, testset$Sample) # Create a table with final predicted data
confusionMatrix(table(svm.predict, testset$Sample)) # Generate confusion matrix to evaluate model performance

# Check final loss score from test data

svm.prob<-predict(svm.model, testset[, !names(testset) %in% c("Sample")], decision.values = TRUE, probability = TRUE)
svm.classprob<-attr(svm.prob,"probabilities") # Class probabilities
svm.results<-data.frame(True = testset$Sample, SVM_Class = svm.predict, Probability = svm.classprob)
View(svm.results) # This will display a table with the precision of each classification
