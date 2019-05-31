import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.metrics import mean_squared_error
from math import sqrt
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt
import statsmodels.api as sm
import warnings
warnings.filterwarnings("ignore")
#we will use the yearly dataset to forecast the next years total of students that will use the cafeteria
#we also see that seasonality plays an important role in the amount of students that use the cafeteria
#so different from the monthly estimation, we also will use Holt-Winters to consider the seasonality

df = pd.read_csv("yearlycafeteria.csv", sep=',', usecols = ["student"])
print(df.axes)

seriesname = 'student'
series = df
sarray = np.asarray(series)
print("Estimating", "student")
print("Series Data:", sarray)

size = len(series)
train = series[0:size-5]
trainarray= np.asarray(train)
test = series[size-5:]
testarray = np.asarray(test)
print("Training data:", trainarray, "Test data:", testarray)

#Naive approach
print("Naive")
dd= np.asarray(train.values)
y_hat = test.copy()
y_hat['naive'] = [dd[len(dd)-1]]*len(test)
rms = sqrt(mean_squared_error(test.values, y_hat.naive))
print("RMSE: ",rms)

#Simple average approach
print("Simple Average")
y_hat_avg = test.copy()
y_hat_avg['avg_forecast'] = train.values.mean()
rms = sqrt(mean_squared_error(test.values, y_hat_avg.avg_forecast))
print("RMSE: ",rms)

#Moving average approach
print("Moving Average")
windowsize = 1
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
date_range = pd.date_range(start = '01-01-2018', freq='MS', periods = len(train))
train = pd.DataFrame({'values': train.values[:,0]}, index=date_range)
sm.tsa.seasonal_decompose(train).plot()
result = sm.tsa.stattools.adfuller(train.iloc[:,0].values)
# plt.show()

y_hat_avg = test.copy()
alpha = 0.4
fit1 = Holt(np.asarray(train.values)).fit(smoothing_level = alpha,smoothing_slope = 0.1)
y_hat_avg['Holt_linear'] = fit1.forecast(len(test))
rms = sqrt(mean_squared_error(test.values, y_hat_avg.Holt_linear))
print("RMSE: ",rms)
#because there are some seasonal differences among the months, we also use Holt-Winters for this dataset
# Holt-Winters
print("Holt-Winters")
y_hat_avg = test.copy()
seasons = 12
fit1 = ExponentialSmoothing(np.asarray(train.values) ,seasonal_periods=seasons ,trend='add', seasonal='add',).fit()
y_hat_avg['Holt_Winter'] = fit1.forecast(len(test))
rms = sqrt(mean_squared_error(test.values, y_hat_avg.Holt_Winter))
print("RMSE: ",rms)



# Holtlari data eksigi yuzunden calistiramadik hocam :(