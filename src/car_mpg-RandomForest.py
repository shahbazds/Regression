# -*- coding: utf-8 -*-
"""
Created on Sat Aug  4 23:02:00 2018

@author: Muhammad Shahbaz
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from math import sqrt
from sklearn.metrics import mean_squared_error 


df = pd.read_csv("auto-mpg-nameless.csv")
#Extracting Labels (Target) and features (Independent Variables)
X = df.iloc[:, 1:].values
y = df.iloc[:,0].values      
           
#Splitting
#splitting the data Training and Test
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.25, random_state=0)

#Feature Scaling
sc = StandardScaler()
X_train_sc = sc.fit_transform(X_train)
X_test_sc = sc.transform(X_test)

#Using Random Forest Regressor
reg = RandomForestRegressor()
reg.fit(X_train_sc, y_train)
y_pred = reg.predict(X_test_sc)
print("Random Forest Regressor Score : ", reg.score(X_test_sc, y_test))
print("Random Forest RMSE : ", sqrt(mean_squared_error(y_test,y_pred)))
print("Random Forest MSE : ",mean_squared_error(y_test,y_pred))

#K-Fold Cross Validation
from sklearn.model_selection import cross_val_score
scores = cross_val_score(reg,X,y,cv=10,scoring='r2')
print("Random Forest K-Fold R-Square Mean: ", scores.mean())