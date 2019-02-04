# -*- coding: utf-8 -*-
"""
Created on Tue Jul 31 14:43:31 2018

@author: mshahbaz
"""

import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from math import sqrt

df = pd.read_excel("Concrete_Data.xls")
#Extracting Labels (Target) and features (Independent Variables)
X = df.iloc[:, :-1].values
y = df.iloc[:,-1].values      

#Splitting into test and train
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=1/3, random_state=0)


#Using Poly kernal with SVM
reg = SVR(kernel ='poly', degree=1)
reg.fit(X_train,y_train)

y_pred = reg.predict(X_test)

score = reg.score(X_test,y_test)
mse = mean_squared_error(y_test,y_pred)
rmse = sqrt(mse)
print("R-square: ", score)
print("Mean Square Error: ", mse)
print("Root Mean Square Error:", rmse)

#K-Fold Cross Validation
from sklearn.model_selection import cross_val_score
scores = cross_val_score(reg,X,y,cv=10,scoring='r2')
print("K-Fold R-Square Mean: ", scores.mean())











