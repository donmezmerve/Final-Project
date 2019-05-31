# Merve Donmez 2148914
# Büşra Aydoğdu 2148633
# Sedef Ece Akansel 2148500
# Utku Haluk Bayram 2220325
# Utku Erkan 2148989


import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.metrics import mean_squared_error
from math import sqrt
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt
import statsmodels.api as sm
import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv("cafeteria.csv", sep=',', usecols = ["lunch"])
print(df.axes)

seriesname = 'lunch'
series = df[seriesname]
sarray = np.asarray(series)
print("Estimating", seriesname)
print("Series Data:", sarray)

#seriesname = 'dinner'
#series = df[seriesname]
#sarray = np.asarray(series)
#print("Estimating", seriesname)
#print("Series Data:", sarray)

size = len(series)
train = series[0:size-5]
trainarray= np.asarray(train)
test = series[size-5:]
testarray = np.asarray(test)
print("Training data:", trainarray, "Test data:", testarray)

#df.DATE = pd.to_datetime(df.DATE,format="%m-%d-%Y")
#df.index = df.DATE 
#train.DATE = pd.to_datetime(train.DATE,format="%m-%d-%Y") 
#train.index = train.DATE 
#test.DATE = pd.to_datetime(train.DATE,format="%m-%d-%Y") 
#test.index = test.DATE 

#Naive approach
print("Naive")
dd= np.asarray(train.values)
y_hat = test.copy()
y_hat['naive'] = [dd[-1]] * len(test)
rms = sqrt(mean_squared_error(test.values, y_hat.naive))
print("RMSE: ",rms)

#Simple average approach
print("Simple Average")
y_hat_avg = test.copy()
y_hat_avg['avg_forecast'] = [train.values.mean()] * len(test)
rms = sqrt(mean_squared_error(test.values, y_hat_avg.avg_forecast))
print("RMSE: ",rms)

#Moving average approach
print("Moving Average")
windowsize = 5
y_hat_avg = test.copy()
y_hat_avg['moving_avg_forecast'] = [train.rolling(windowsize).mean().iloc[-1]]*len(test)
rms = sqrt(mean_squared_error(test.values, y_hat_avg.moving_avg_forecast))
print("RMSE: ",rms)

# Simple Exponential Smoothing
print("Simple Exponential Smoothing")
y_hat_avg = test.copy()
alpha = 0.2
fit2 = SimpleExpSmoothing(np.asarray(train.values)).fit(smoothing_level=alpha,optimized=False)
y_hat_avg['SES'] = fit2.forecast(len(test))
rms = sqrt(mean_squared_error(test.values, y_hat_avg.SES))
print("RMSE: ",rms)

# Holt
print("Holt")
date_range = pd.date_range(start = '10/01/2018', freq='D', periods = len(train))
train = pd.DataFrame({'values': train.values}, index=date_range)
sm.tsa.seasonal_decompose(train).plot()
result = sm.tsa.stattools.adfuller(train.iloc[:,0].values)
# plt.show()

y_hat_avg = test.copy()
alpha = 0.4
fit1 = Holt(np.asarray(train.values)).fit(smoothing_level = alpha,smoothing_slope = 0.1)
y_hat_avg['Holt_linear'] = fit1.forecast(len(test))
rms = sqrt(mean_squared_error(test.values, y_hat_avg.Holt_linear))
print("RMSE: ",rms)
