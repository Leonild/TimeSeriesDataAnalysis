import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
# functions for the autocorrelation calc and partial autocorrelation calc
import statsmodels.api as sm
#to work shloud to use pip3 install statsmodels==0.13.0
# ignore not crucial warnings
import warnings
warnings.filterwarnings("ignore")
from matplotlib import pyplot
from datetime import datetime
from math import sin, cos, sqrt, atan2, radians
from pandas.plotting import lag_plot
from pmdarima import auto_arima # to determinate ARIMA order
from statsmodels.tsa.arima.model import ARIMA


class Agregation:

	def __init__(self):
		#self.dataset = self.initialize(dataset)
		self.dataset = pd.DataFrame(columns = ["Date", "AQI", "latitude","longitude"])

	def arima(self):
		# Dickey-Fuller test to confirm stationarity
		# fonte: https://machinelearningmastery.com/time-series-data-stationary-python/
		result = sm.tsa.stattools.adfuller(self.dataset['AQI'], autolag='AIC')
		print('ADF Statistic: %f' % result[0])
		print('p-value: %f' % result[1])
		print('Critical Values:')
		for key, value in result[4].items():
			print('\t%s: %.3f' % (key, value))
		
		#now considering an ARIMA(p,q) model - auto_arima
		stepwise_fit = auto_arima(self.dataset['AQI'], start_p=0, start_q=0,
                          max_p=6, max_q=3, m=0,
                          seasonal=False,
                          d=0, trace=True,
                          error_action='ignore',   # we don't want to know if an order does not work
                          suppress_warnings=True,  # we don't want convergence warnings
                          stepwise=True)           # set to stepwise

		stepwise_fit.summary()

		division = len(self.dataset)
		trainning = self.dataset[:int(division*0.8)]
		test = self.dataset[int(division*0.8):]
		model = ARIMA(trainning['AQI'],order=(1,0,1))
		result = model.fit()
		result.summary()

		# seen prediction
		start=len(trainning)
		end=len(trainning)+len(test)-1
		prediction = result.predict(start=start, end=end).rename('Prediction ARMA(1,1)')
		title = 'AQI ARIMA prediction'
		ylabel='AQI'
		xlabel='' # we don't really need a label here

		ax = test['AQI'].plot(legend=True,figsize=(12,6),title=title)
		prediction.plot(legend=True)
		ax.autoscale(axis='x',tight=True)
		ax.set(xlabel=xlabel, ylabel=ylabel)
		plt.show()



	# add others datasets
	def appendDataset(self, dataset):
		newDataset = self.initialize(dataset)
		self.dataset = self.dataset.append(newDataset)
		self.dataset = self.dataset.sort_index()

	# initialize the dataset by removing the undesire columns, assigning date to index, and grouping by month (commented)
	def initialize(self, dataset):
		#removing undesired columns
		dataset = dataset.drop(['Source','Site ID','UNITS','POC','STATE','STATE_CODE','COUNTY_CODE','COUNTY','Daily Max 8-hour Ozone Concentration','Site Name','DAILY_OBS_COUNT','PERCENT_COMPLETE','AQS_PARAMETER_CODE','AQS_PARAMETER_DESC','CBSA_CODE','CBSA_NAME'],axis=1)
		dataset = dataset.dropna() 
		# Convert the date to datetime64
		dataset['Date'] = pd.to_datetime(dataset['Date'], format='%m/%d/%Y')
		dataset.index = dataset['Date'] # to work the next line
		#grouping by month
		dataset = dataset.groupby(pd.Grouper(freq='M')).mean() #grouping data by month
		dataset = dataset.rename(columns={"Date": "Date", "DAILY_AQI_VALUE": "AQI", "SITE_LATITUDE": "latitude", "SITE_LONGITUDE": "longitude"})
		return dataset

	#correlogram: autocorrelation representation
	def correlogram(self):
		#returns ndarray with a partial autocorrelations for lags 0, 1, …, nlags. Shape (nlags+1,) | ols : regression of time series on lags of it and on constant
		correlation = sm.tsa.stattools.pacf_ols(self.dataset['AQI'])
		#lag_plot(correlation)#.imshow()
		title = 'Autocorrelation: AQI'
		lags = 10
		#plot_acf(correlation,title=title,lags=lags)
		sm.graphics.tsa.plot_pacf(correlation,title=title,lags=lags)
		plt.show()

	# plot data in a simple time series description
	def plotTimeSeries(self):
		self.dataset = self.dataset.sort_index()
		plt.rc('font', size=10)
		#moving average by month
		self.dataset['Moving-Average'] = self.dataset['AQI'].rolling(window=12).median()
		#self.dataset[['DAILY_AQI_VALUE','Moving-Average']].plot(figsize=(12,9))

		plt.plot(self.dataset.index,self.dataset['AQI'],label="Ozone",color="red") # Plota os dados
		plt.plot(self.dataset.index,self.dataset['Moving-Average'],label="Moving Average",color="blue") # Plota os dados
		plt.title('Ozone Polution')
		plt.xlabel("Date")
		plt.ylabel("O3")
		plt.ylim()
		plt.show()
		#plt.savefig("./"+graphName)
		plt.clf()

	#salve the dataframe into a csv
	def saveData(self, file_name):
		self.dataset.to_csv(file_name, index=False)

	# Plot a seasonal decompose
	def seasonalDecompose(self):
		self.dataset = self.dataset.sort_index()
		pollutant = np.array(self.dataset['AQI'])
		result = seasonal_decompose(pollutant, model='additive', period=12)
		result.plot()
		pyplot.show()

	# to split data set per season
	def seasonFilter(self, df):	
		spring = df[(df['Date'] >= '2019-03-01') & (df['Date'] <= '2019-05-31')]
		summer = df[(df['Date'] >= '2019-06-01') & (df['Date'] <= '2019-08-31')]
		fall = df[(df['Date'] >= '2019-09-01') & (df['Date'] <= '2019-11-30')]
		winter = df[(df['Date'] >= '2019-12-01')]# & (df['Date'] <= '2019-02-28')]
		winter = winter.append(df[(df['Date'] <= '2019-02-28')])

if __name__ == "__main__":
	
	path = "/home/leonildo/Downloads/california-data-ozone"
	a = Agregation()

	dir_list = os.listdir(path)
	for file in dir_list:
		readPath = path + '/' + file
		print("Processing dataset ", readPath)
		dataSet = pd.read_csv(readPath) #O3
		a.appendDataset(dataSet)

	#a.seasonalDecompose()
	#a.plotTimeSeries()
	#a.correlogram()
	a.arima()

