#The observations in a stationary time series are not dependent on time.

#import some libraries

import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller

#import data

data_return = pd.read_csv('return.csv', index_col = 'date', parse_dates=True)

#We check at stationarity with three different ways :

#1st : Look at the plot.

plt.style.use('ggplot')
data_return['closing_price'].plot(label='CLOSE', title='Daily Return')

#We can notice the COVID crash in March 2020.It does not seem to have a trend or seasonal effects, which is a good news for stationarity. For the other variables :

for i in range(len(data_return.columns) - 1):
    x = data_return.iloc[:,i]
    x.plot(title=data_return.columns[i])
    plt.show()
    
#2nd : Statistical approach = dividing dataset in 4 and check mean/variance

for i in range(len(data_return.columns) - 1):
    X = data_return.values[i]
    split = round(len(X) / 4)
    X1, X2, X3, X4 = X[0:split], X[split:(2*split)], X[(2*split):(3*split)], X[split:]
    mean1, mean2, mean3, mean4 = X1.mean(), X2.mean(), X3.mean(), X4.mean()
    var1, var2, var3, var4 = X1.var(), X2.var(), X3.var(), X4.var()
    print("Mean & Variance of", data_return.columns[i])
    print('mean1 = %f, mean2 = %f, mean3 = %f, mean4 = %f' % (mean1, mean2, mean3, mean4))
    print('variance1 = %f, variance2 = %f, variance3 = %f, variance4 = %f' % (var1, var2, var3, var4))
    print(80*"-")

#3rd : Augmented Dickey-Fuller test
#p-value > 0.05: non-stationary.
#p-value <= 0.05: stationary.

for i in range(len(data_return.columns)):
    X = data_return.values[i]
    result = adfuller(X)
    print("Results of the test for :", data_return.columns[i])
    print('ADF Statistic: %f' % result[0])
    print('p-value: %f' % result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t%s: %.3f' % (key, value))