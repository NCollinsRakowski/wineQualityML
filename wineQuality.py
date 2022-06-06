import numpy as pd
import pandas as pd
from sklearn import pipeline

from sklearn.model_selection import train_test_split # splits the data for testing and training.
# this step is important because you do not want to overfit the test data
from sklearn import preprocessing #scaling, transforming, and wranlging data

from sklearn.ensemble import RandomForestRegressor # import random forest model

# import cross-validation pipeline
# Cross-validation is a technique for evaluating ML models by training several ML models on subsets of the available input data 
# and evaluating them on the complementary subset of the data. 
# Use cross-validation to detect overfitting, ie, failing to generalize a pattern.
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV

# import evaluation metrics
from sklearn.metrics import mean_squared_error, r2_score

# import module for savinf scikit-learn models
#from sklearn.externals import joblib # alternae pickle package

df = pd.read_csv(r'/Users/cr185200/repos/wineQualityML/data/winequality-red.csv', sep=';')

# usually need to explore data. This is a means of start 
# print(df.head())
# print(df.shape)
# print(df.describe())

# because x = y. y is what we are solving for
y = df.quality #what we are aiming for
x = df.drop('quality', axis = 1) # drops the quality column of data

# Split into Test and Train Data
# test size is set to 20% of the data
# random state is the seed
# stratify preserve the proportion of target as in original dataset, in the train and test
xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=.2, random_state=123, stratify=y) 

# most data needs to be standardized like around zero. here is scaling
#Lazy Way of Scaling
#xTrainedScaled = preprocessing.scale(xTrain)
#print(xTrainedScaled)
#confirm using these commands
# print(xTrainedScaled.mean(axis=0))
# print(xTrainedScaled.std(axis=0))

#Scikit-Learn Tranformer API - allows you ti fit preprocessing step using the training data the same way you'd fit the model
#1. fit the tranformer on the training set (saving the means and standard deciations)
#2. Apply the transformer to the training set (scaling the training data)
#3. Apply the transformer to the test set (using the same means and standard deviations)
scaler = preprocessing.StandardScaler().fit(xTrain)
xTrainScaled = scaler.transform(xTrain)
xTestScaled = scaler.transform(xTest)

#print(xTrainScaled.mean(axis=0))
#print(xTrainScaled.std(axis=0))
#print(xTestScaled.mean(axis=0))
#print(xTestScaled.std(axis=0))

#pipeline with preprocessing and model
# first transforms data using standardscaler, then fits model using randomforegregressor
# Randomforestregressor - is a supervised learning algorithm that uses ensemble learning method for regression. 
# Ensemble learning method is a technique that combines predictions from multiple machine learning algorithms to make a more accurate prediction than a single model
pipeline = make_pipeline(preprocessing.StandardScaler(), RandomForestRegressor(n_estimators=100))

#hyperparameters - list tunable hyperparameters
#print(pipeline.get_params())
# http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html

hyperparameters = {'randomforestregressor__max_features' : ['auto', 'sqrt', 'log2'],
                    'randomforestregressor__max_depth' : [None, 5, 3 , 1]
                    }

#kfold cross validation - we want to train the random forest regressor
#Sklearn cross-validation with pipeline
clf = GridSearchCV(pipeline, hyperparameters, cv = 10) # performs cross validation across entire grid
clf.fit(xTrain, yTrain)
print(clf.best_params_) # now you can find best set of params found using CV
# print(clf.refit) # Check refit
yPred = clf.predict(xTest)
print(r2_score(yTest, yPred))
print(mean_squared_error(yTest, yPred))

#not working 
# joblib.dump(clf, 'rf_regressor.pkl')
# clf2 = joblib.load('rf_regressor.pkl')
# clf2.predict(xTest)