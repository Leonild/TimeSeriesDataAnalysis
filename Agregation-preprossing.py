import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from matplotlib import pyplot

from datetime import datetime
from math import sin, cos, sqrt, atan2, radians


class Agregation:

	def __init__(self):
		#self.dataset = self.initialize(dataset)
		self.dataset = pd.DataFrame(columns = ["Date", "DAILY_AQI_VALUE", "SITE_LATITUDE","SITE_LONGITUDE"])
		

	# initialize the dataset by removing the undesire columns, assigning date to index, and grouping by month (commented)
	def initialize(self, dataset):
		#removing undesired columns
		dataset = dataset.drop(['Source','Site ID','UNITS','POC','STATE','STATE_CODE','COUNTY_CODE','COUNTY','Daily Max 8-hour Ozone Concentration','Site Name','DAILY_OBS_COUNT','PERCENT_COMPLETE','AQS_PARAMETER_CODE','AQS_PARAMETER_DESC','CBSA_CODE','CBSA_NAME'],axis=1)
		dataset = dataset.dropna() 
		# Convert the date to datetime64
		dataset['Date'] = pd.to_datetime(dataset['Date'], format='%m/%d/%Y')
		dataset.index = dataset['Date'] # to work the next line
		dataset = dataset.groupby(pd.Grouper(freq='M')).mean() #grouping data by month
		return dataset

	# add others datasets
	def appendDataset(self, dataset):
		newDataset = self.initialize(dataset)
		self.dataset = self.dataset.append(newDataset)

	# to split data set per season
	def seasonFilter(self, df):	
		spring = df[(df['Date Local'] >= '2019-03-01') & (df['Date Local'] <= '2019-05-31')]
		summer = df[(df['Date Local'] >= '2019-06-01') & (df['Date Local'] <= '2019-08-31')]
		fall = df[(df['Date Local'] >= '2019-09-01') & (df['Date Local'] <= '2019-11-30')]
		winter = df[(df['Date Local'] >= '2019-12-01')]# & (df['Date Local'] <= '2019-02-28')]
		winter = winter.append(df[(df['Date Local'] <= '2019-02-28')])


	# plot data in a simple time series description
	def plotTimeSeries(self):
		self.dataset = self.dataset.sort_index()
		plt.rc('font', size=10)
		#moving average by month
		self.dataset['Moving-Average'] = self.dataset['DAILY_AQI_VALUE'].rolling(window=12).median()
		#self.dataset[['DAILY_AQI_VALUE','Moving-Average']].plot(figsize=(12,9))

		plt.plot(self.dataset.index,self.dataset['DAILY_AQI_VALUE'],label="Ozone",color="red") # Plota os dados
		plt.plot(self.dataset.index,self.dataset['Moving-Average'],label="Moving Average",color="blue") # Plota os dados
		plt.title('Ozone Polution')
		plt.xlabel("Date")
		plt.ylabel("O3")
		plt.ylim()
		plt.show()
		#plt.savefig("./"+graphName)
		plt.clf()

	# Plot a seasonal decompose
	def seasonalDecompose(self):
		self.dataset = self.dataset.sort_index()
		pollutant = np.array(self.dataset['DAILY_AQI_VALUE'])

		result = seasonal_decompose(pollutant, model='additive', period=12)
		result.plot()

		pyplot.show()

	#salve the dataframe into a csv
	def saveData(self, file_name):
		self.dataset.to_csv(file_name, index=False)

if __name__ == "__main__":
	
	path = "/home/leonildo/Downloads/california-data-ozone"
	a = Agregation()

	dir_list = os.listdir(path)
	for file in dir_list:
		readPath = path + '/' + file
		print("Processing dataset ", readPath)
		dataSet = pd.read_csv(readPath) #O3
		a.appendDataset(dataSet)

	a.seasonalDecompose()
	#a.plotTimeSeries()

