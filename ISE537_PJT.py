import pandas as pd
import matplotlib.pyplot as plt
import math 
import numpy as np
import pickle #pickle is used to save the model(python ver.)
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn import model_selection
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import Ridge


class ISE(object):
    
    def __init__(self, data):
        self.data = data
    
    def data_split(self):
        x_data = self.data.iloc[:, 1:-1]
        y_data = self.data.iloc[:, [-1]]
        x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=0)
        return x_train, x_test, y_train, y_test

    def train(self, x_train, y_train, x_test, y_test, model):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        model.fit(self.x_train, self.y_train)
        y_pred = model.predict(self.x_test)
        return y_pred
    
    def evaluation(self, name, y_pred):
        temp_y_test = self.y_test['Thermostat_Temperature']
        ###r2###
        #r2_score_ = r2_score(self.x_test, self.y_test)
        #print(name, "'s r2 socre is", r2_score_)
        ##MBE###
        
        ##MNBE###
        
        ###Coefficient of Variation of the Root Mean Square Error(CVRMSE)###
        mse = mean_squared_error(self.y_test, y_pred)
        rmse = math.sqrt(mse)
        Avg_target = sum(temp_y_test)/len(temp_y_test)
        CVRMSE = rmse/Avg_target * 100
        print(name, "'s CVRMSE is", CVRMSE)
        return


data = pd.read_csv('C:/Practice/ISE537_data_preprocessed_final.csv')
models = []
models.append(('Logistic Regression', LogisticRegression()))
models.append(('multiple linear regression', linear_model.LinearRegression()))
models.append(('Ridge', Ridge()))
models.append(('LASSO', linear_model.Lasso()))
models.append(('Elastic Net', ElasticNet()))
models.append(('LARS', linear_model.Lars()))

# Data Split
ISE = ISE(data)
(x_train, x_test, y_train, y_test) = ISE.data_split()
print(len(x_train), len(y_test))
for name, model in models:
    y_pred = ISE.train(x_train, y_train, x_test, y_test, model)
    ISE.evaluation(name, y_pred)
