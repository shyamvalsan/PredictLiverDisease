# Load libraries
library(caret)
library(dplyr)
library(reshape2)
library(ggplot2)
library(xgboost)
library(DMwR)

# Convert string to integer
str2int <- function(df) {
  strings=sort(unique(df))
  numbers=1:length(strings)
  names(numbers)=strings
  return(numbers[df])
}

# Read the data
data_raw <- read.csv('indian_liver_patient.csv')
data = data_raw[complete.cases(data_raw),]

# Convert string features to numeric
data$Gender <- str2int(data$Gender)

# Remove any features with correlation greater than 80%
tmp <- cor(data)
tmp[!lower.tri(tmp)] <- 0
data.new <- data[,!apply(tmp,2,function(x) any(x > 0.8))]
data = data.new

# Generate histogram plot for all features(attributes)
d <- melt(data)
ggplot(d,aes(x = value)) + 
  facet_wrap(~variable,scales = "free_x") + 
  geom_histogram()
summary(data)

# Test-Train split
data$Dataset <- data$Dataset - 1
data_label <- as.numeric(data$Dataset)  
sample_size <- floor(0.75 * nrow(data))
set.seed(123)
train_ind <- sample(seq_len(nrow(data)), size = sample_size)
train <- data[train_ind, ]
test <- data[-train_ind, ]
train$Dataset <- as.factor(train$Dataset)

train_label <- as.numeric(train$Dataset)
train <- as(as.matrix(train[ , -which(names(train) %in% c("Dataset"))]), "dgCMatrix")
test_label <- as.numeric(test$Dataset)
test <- as(as.matrix(test[ , -which(names(test) %in% c("Dataset"))]), "dgCMatrix")

fulldata <- as(as.matrix(data[ , -which(names(data) %in% c("Dataset"))]), "dgCMatrix")

# Create watchlist for training
dtrain <- xgb.DMatrix(data = train, label=train_label)
dtest <- xgb.DMatrix(data = test, label=test_label)    
watchlist <- list(train=dtrain, test=dtest)

# Train xgbmodel
xgbModel <- xgb.train(data = dtrain, max.depth = 100, eta = 0.001, 
                      nthread = 2,  nround = 10000, 
                      watchlist=watchlist, objective = "binary:logistic", early_stopping_rounds = 500)

# Make prediction and generate confusion matrix
test_pred <- predict(xgbModel, newdata = fulldata)
cMatrix <- confusionMatrix(round(test_pred), data_label)
cMatrix
cMatrix$overall

# Generate importance plot
xgb.importance(colnames(fulldata), model = xgbModel)