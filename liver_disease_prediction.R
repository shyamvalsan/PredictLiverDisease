# Load libraries
library(caret)
library(xgboost)
library(methods)

data <- read.csv('indian_liver_patient.csv')
data$Albumin_and_Globulin_Ratio[is.na(data$Albumin_and_Globulin_Ratio)] <- mean(data$Albumin_and_Globulin_Ratio, na.rm = TRUE)

# Convert string to integer
str2int <- function(df) {
  strings=sort(unique(df))
  numbers=1:length(strings)
  names(numbers)=strings
  return(numbers[df])
}
data$Gender <- str2int(data$Gender)

tmp <- cor(data)
tmp[!lower.tri(tmp)] <- 0
data.new <- data[,!apply(tmp,2,function(x) any(x > 0.8))]
data = data.new

data$Dataset <- data$Dataset - 1

sample_size <- floor(0.75 * nrow(data))
set.seed(123)
train_ind <- sample(seq_len(nrow(data)), size = sample_size)
train <- data[train_ind, ]
test <- data[-train_ind, ]
train_label <- as.numeric(train$Dataset)
test_label <- as.numeric(test$Dataset)

train <- as(as.matrix(train[ , -which(names(train) %in% c("Dataset"))]), "dgCMatrix")
test <- as(as.matrix(test[ , -which(names(test) %in% c("Dataset"))]), "dgCMatrix")
dtrain <- xgb.DMatrix(data = train, label=train_label)
dtest <- xgb.DMatrix(data = test, label=test_label)    
watchlist <- list(train=dtrain, test=dtest)

xgbModel <- xgb.train(data = dtrain, max.depth = 100, eta = 0.001, 
                      nthread = 2,  nround = 10000, 
                      watchlist=watchlist, objective = "binary:logistic", early_stopping_rounds = 500)

fulldata <- as(as.matrix(data[ , -which(names(data) %in% c("Dataset"))]), "dgCMatrix")
test_pred <- predict(xgbModel, newdata = fulldata)

confusionMatrix(round(test_pred), data$Dataset)
xgb.importance(colnames(fulldata), model = xgbModel)

