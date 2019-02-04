# -*- coding: utf-8 -*-
"""
Created on Thu Aug  2 23:08:20 2018

@author: Muhammad Shahbaz
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.preprocessing import PolynomialFeatures

df = pd.read_csv("auto-mpg-nameless.csv")
#Extracting Labels (Target) and features (Independent Variables)
X = df.iloc[:, 1:].values
y = df.iloc[:,0].values      

#Splitting into test and train
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=1/3, random_state=0)

#Transformation and Regressioin with Degree-1
poly = PolynomialFeatures(degree = 1)
X_test_poly = poly.fit_transform(X_test)
X_train_poly = poly.fit_transform(X_train)

lin_reg = LinearRegression()
lin_reg.fit(X_train_poly,y_train)
y_pred1 = lin_reg.predict(X_test_poly)

print("Polynomial Regression Score with Degree-1 : ",lin_reg.score(X_test_poly, y_test))
print("Polynomial Regression MSE with Degree-1 : ",mean_squared_error(y_test,y_pred1))
print("Polynomial Regression RMSE with Degree-1: ", sqrt(mean_squared_error(y_test,y_pred1)))



#Transformation and Regressioin with Degree-2

poly = PolynomialFeatures(degree = 2)
X_test_poly = poly.fit_transform(X_test)
X_train_poly = poly.fit_transform(X_train)
lin_reg = LinearRegression()
lin_reg.fit(X_train_poly,y_train)
y_pred1 = lin_reg.predict(X_test_poly)
print("Polynomial Regression Score with Degree-2 : ",lin_reg.score(X_test_poly, y_test))
print("Polynomial Regression MSE with Degree-2 : ",mean_squared_error(y_test,y_pred1))
print("Polynomial Regression RMSE with Degree-2: ", sqrt(mean_squared_error(y_test,y_pred1)))



#Transformation and Regressioin with Degree-3


poly = PolynomialFeatures(degree = 3)
X_test_poly = poly.fit_transform(X_test)
X_train_poly = poly.fit_transform(X_train)
lin_reg = LinearRegression()
lin_reg.fit(X_train_poly,y_train)
y_pred1 = lin_reg.predict(X_test_poly)
print("Polynomial Regression Score with Degree-3 : ",lin_reg.score(X_test_poly, y_test))
print("Polynomial Regression MSE with Degree-3 : ",mean_squared_error(y_test,y_pred1))
print("Polynomial Regression RMSE with Degree-3: ", sqrt(mean_squared_error(y_test,y_pred1)))


#Checking degree accuracy through k-fold
from sklearn.model_selection import cross_val_score
scores=[]
for n in range(2,4):
    
    polyn = PolynomialFeatures(degree =n )
    X_polyn = polyn.fit_transform(X)
    lin_reg = LinearRegression()
    score = cross_val_score(lin_reg,X_polyn,y,cv=10,scoring='r2')
    print("With Degree "+str(n)+":",score.mean())
    scores.append(score.mean())
print("K-fold Scores: ",np.array(scores).mean())

