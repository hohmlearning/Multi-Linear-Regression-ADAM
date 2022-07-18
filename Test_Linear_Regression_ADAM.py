# -*- coding: utf-8 -*-
"""
Created on Sun Jul 17 21:41:55 2022

@author: Manue
"""
import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from Linear_Regression_ADAM import ADAM, ADAM_learning_rate_decay, ADAM_learning_rate_decay_full_train
from Evaluation_Metric import Metric_regression

#%%
diabetes = load_diabetes()
X = diabetes.data
y = diabetes.target
#%%
X = X - X.mean(axis=0)
X = X / X.var(axis=0)**0.5
#%%
X_train = X[:400,:]
y_train = y[:400]
X_test =  X[400:,:]
y_test = y[400:]
#%%
print('1. Fit direct Linear Regression (Sklearn):')
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
y_test_sklearn = np.dot(X_test, lin_reg.coef_) + lin_reg.intercept_
#%%
max_epoch = 1E4
model_list = []
#%%
Gradient_descent_1 = ADAM(max_epoch=max_epoch, eta=.1, batch_size=1)
print('2. {} on train:'.format(Gradient_descent_1.name))
Gradient_descent_1(X_train, y_train) 
model_list.append(Gradient_descent_1)
#%%
Gradient_descent_2 = ADAM_learning_rate_decay(max_epoch=max_epoch, eta=.1, batch_size=1, patience=1E2)
print('3. {} on train:'.format(Gradient_descent_2.name))
Gradient_descent_2(X_train, y_train)
model_list.append(Gradient_descent_2)
#%%
Gradient_descent_2_2 = ADAM_learning_rate_decay(max_epoch=max_epoch, eta=.1, batch_size=1, patience=1E2)#
print('4. {} on subtrain:'.format(Gradient_descent_2_2.name))
split = int(X_train.shape[0]*0.8)
Gradient_descent_2_2(X_train[:split], y_train[:split],
                     X_train[split:], y_train[split:]
                     )
model_list.append(Gradient_descent_2_2)
#%%
Gradient_descent_3 = ADAM_learning_rate_decay_full_train(max_epoch=max_epoch, eta=.1, batch_size=1, patience=1E2)
print('6. {} on subtrain and train, respectively:'.format(Gradient_descent_3.name))
Gradient_descent_3(X_train, y_train)
model_list.append(Gradient_descent_3)
#%%
pd.set_option('display.max_columns', None)
compare_weights = pd.DataFrame()
compare_weights['Sklearn'] = np.append(lin_reg.intercept_ , lin_reg.coef_ )

for SGD_model in model_list:
    column_name = SGD_model.name
    if column_name == 'SGD with Adam':
        w = SGD_model.theta
        w_0 = SGD_model.theta_0
    
    else:
        w = SGD_model.best_theta
        w_0 = SGD_model.best_theta_0
    compare_weights[column_name] = np.append(w_0, w)
pd.set_option('display.precision', 1)
print(compare_weights.T)   
#%%
MSE_compare = pd.Series(dtype=float)
ARD_compare = pd.Series(dtype=float) 
MSE_sklearn = Metric_regression().fun_MSE(y_test, y_test_sklearn)
MSE_compare['Sklearn'] = MSE_sklearn
ARD_compare['Sklearn'] = 0
for SGD_model in model_list:
    column_name = SGD_model.name
    MSE_test = SGD_model.MSE(X_test, y_test)
    MSE_compare[column_name] = MSE_test
    ARD_compare[column_name] = -(MSE_test-MSE_sklearn)/MSE_sklearn*100

MSE_DF = MSE_compare.to_frame(name='MSE')
ARD_DF = ARD_compare.to_frame(name='Deviation / %')
performance_DF = pd.concat([ARD_DF, MSE_DF], axis=1)
print(performance_DF)


