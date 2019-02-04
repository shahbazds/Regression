# -*- coding: utf-8 -*-
"""
Created on Sat Aug  4 23:32:44 2018

@author: Muhammad Shahbaz
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_excel("Folds5x2_pp.xlsx")
#Extracting Labels (Target) and features (Independent Variables)
X = df.iloc[:, :-1].values
y = df.iloc[:,-1].values           



#Splitting into test and train
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=1/3, random_state=0)

#Scaling 
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train_sc = sc_X.fit_transform(X_train)
X_test_sc = sc_X.transform(X_test)

#Prediction
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X_train_sc,y_train)
y_pred = lin_reg.predict(X_test_sc)

#RMSE
from sklearn.metrics import mean_squared_error
from math import sqrt
mse = mean_squared_error(y_test,y_pred)
rmse = sqrt(mse)
score = lin_reg.score(X_test_sc, y_test)

#K-Fold Cross Validation
from sklearn.model_selection import cross_val_score
scores = cross_val_score(lin_reg,X,y,cv=10,scoring='r2')

print("Linear Regressor Score : ", score)
print("Linear Regressor : ", rmse)
print("Linear Regressor MSE : ",mse)
print("Linear Regressor K-Fold R-Square Mean: ", scores.mean())
