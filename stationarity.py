#Stationarity Test of our data.

#import some libraries

import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller

#import data from a csv file and convert to dataframe.

data_return = pd.read_csv('dataset.csv', index_col = 'date', parse_dates=True)

#We check at stationarity with three different ways : visually, check mean/variance & Dicky-Fuller test.

#1st : Look at the plot.



for i in range(len(data_return.columns)):
    x = data_return.iloc[:,i]
    x.plot(title=data_return.columns[i])
    plt.show()
    
#Regarding, the daily returns, we can notice the COVID crash in March 2020.It does not seem to have a trend or seasonal effects, which is a good news for stationarity.

#2nd : Statistical approach = dividing dataset in 4 and check mean/variance. Ideally, each "sub-dataset" has the same mean and the same variance. 
#We will do this for each variable.

for i in range(len(data_return.columns) - 1):
    data = data_return.values[i]
    
    #Split the data in 4 "sub-datasets"
    split = round(len(data) / 4)
    data_1, data_2, data_3, data_4 = data[0:split], data[split:(2*split)], data[(2*split):(3*split)], data[split:] 
    
    #Compute the mean and the variance of each "sub-dataset".
    mean1, mean2, mean3, mean4 = data_1.mean(), data_2.mean(), data_3.mean(), data_4.mean()
    var1, var2, var3, var4 = data_1.var(), data_2.var(), data_3.var(), data_4.var()
    print("Mean & Variance of", data_return.columns[i])
    print('mean1 = %f, mean2 = %f, mean3 = %f, mean4 = %f' % (mean1, mean2, mean3, mean4))
    print('variance1 = %f, variance2 = %f, variance3 = %f, variance4 = %f' % (var1, var2, var3, var4))
    print(80*"-")

#3rd : Augmented Dickey-Fuller test
#p-value > 0.05: non-stationary.
#p-value <= 0.05: stationary.

for i in range(len(data_return.columns)):
    variables = data_return.values[i]
    result = adfuller(variables)
    print("Results of the test for :", data_return.columns[i])
    print('p-value: %f' % result[1])
    print(40*"-")
