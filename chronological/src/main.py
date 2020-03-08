#coding:utf-8
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import rcParams
import os
import sys
import statsmodels.api as sm

sys.path.append('../')
rcParams['figure.figsize'] = 15,6 # change graph size(hide, width)

dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m')
## compare to with basic data
# data = pd.read_csv('input/AirPassengers.csv', index_col='Month')

# print(data.head())

data = pd.read_csv('input/AirPassengers.csv', index_col='Month', date_parser=dateparse, dtype='float')

print('> get Head Data:')
print(data.head())
print('> get Tail Data:')
print(data.tail())

data.to_csv('./output/table.csv',encoding='utf-8')

res = sm.tsa.seasonal_decompose(data) #make seasonal adjuste model
original = data 
trend = res.trend #sampling trend
seasonal = res.seasonal #sampling seasonal data
residual = res.resid #sampling resid data
plt.figure(figsize=(8,8))

#plot origin
plt.subplot(411)
plt.plot(original)
plt.ylabel('Original')

#plot trend data
plt.subplot(412)
plt.plot(trend)
plt.ylabel('Trend')

#plot seasonal data
plt.subplot(413)
plt.plot(seasonal)
plt.ylabel('Seasolnal')

#plot residual
plt.subplot(414)
plt.plot(residual)
plt.ylabel('Residual')

plt.tight_layout() # adjust graph's interval
plt.savefig('./output/glaph.png')

#plot ACF
sum_three_data = trend + seasonal + residual
plt.figure(figsize=(8, 4))
plt.plot(original, label="original")
plt.plot(sum_three_data, label="trend+seasonal+residual", linestyle="--")
plt.legend(loc="best")

plt.savefig('./output/sum_three_graph.png')

#plot PACF
fig = plt.figure(figsize=(8,8))
ax1 = fig.add_subplot(211)
sm.graphics.tsa.plot_acf(data, lags=40, ax=ax1)

ax2 = fig.add_subplot(212)
sm.graphics.tsa.plot_pacf(data, lags=40, ax=ax2)
plt.tight_layout()
plt.savefig('./output/acf_pacf_graph.png')

#split data
test_data = data[data.index >= pd.Timestamp('1959-12-01')]
train_data = data[data.index < pd.Timestamp('1959-12-01')]
print('train data: ', train_data.head())
print('test data: ', test_data.head())
train_data.to_csv('./output/train_data.csv')
test_data.to_csv('./output/test_data.csv')

#SARIMA(Seasonal AutoRegressive Integrated Moving Average)
sarimax_train = sm.tsa.SARIMAX(train_data, 
    order=(3,1,3), #autoregression degree, diff degree, moving average degree
    seasonal_order=(0,1,1,12), #
    enforce_stationarlity=False,
    enforce_invertibility=False
    ).fit()

# predict（TODO：require data fix!)
sarimax_train2_pred = sarimax_train.predict('1959-12', '1960-12')
print('predict>> :' + sarimax_train2_pred)
plt.plot(data, c='r', label='actual')
plt.plot(sarimax_train2_pred, c='b', label='model-pred', alpha=0.7)
plt.legend(loc='best')

plt.savefig('./output/predict.png')