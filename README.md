# Predicting Liver Disease using Machine Learning

This document describes how we can use machine learning, and the XGBoost (eXtreme Gradient Boosting) library in particular, to predict liver disease risk in patients.

The data that we will be using has been sourced from https://www.kaggle.com/uciml/indian-liver-patient-records. This data set contains 416 liver patient records and 167 non liver patient records collected from the state of Andhra Pradesh in India.

The final model achieved an accuracy of 88.51%.

The dataset we used was indeed limited, and to truly have a model which generalizes well we would need to collect much more data but the results we achieved are very promising indeed. And hospitals and health authorities would clearly have more of the data we require to make our model achieve (or even surpass) human-level diagnosis accuracy.

## Acknowledgements
This dataset was downloaded from the UCI ML Repository:

Lichman, M. (2013). UCI Machine Learning Repository [http://archive.ics.uci.edu/ml]. Irvine, CA: University of California, School of Information and Computer Science.

The XGBoost library is used for training the prediction model [https://github.com/dmlc/xgboost]
