import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from math import sin, cos, sqrt, atan2, radians


class Agregation:

	def __init__(self, dataset):
		self.dataset = dataset

	def initialize(self):
		#removing undesired columns
		self.dataset = self.dataset.drop(['Source','Site ID','UNITS','POC','STATE','STATE_CODE','COUNTY_CODE','COUNTY','Daily Max 8-hour Ozone Concentration','Site Name','DAILY_OBS_COUNT','PERCENT_COMPLETE','AQS_PARAMETER_CODE','AQS_PARAMETER_DESC','CBSA_CODE','CBSA_NAME'],axis=1)
		self.dataset = self.dataset.dropna() 
		# Convert the date to datetime64
		self.dataset['Date'] = pd.to_datetime(self.dataset['Date'], format='%m/%d/%Y')
		self.dataset.index = self.dataset['Date'] # to work the next line
		self.dataset = self.dataset.groupby(pd.Grouper(freq='M')).mean() #grouping data by month
		print(self.dataset.head())
		print(self.dataset.tail())

	def plotTimeSeries(self, graphName):
		plt.rc('font', size=10)
		plt.plot(self.dataset.index,self.dataset['DAILY_AQI_VALUE'],label="Ozone",color="red") # Plota os dados
		plt.title('Ozone Polution')
		plt.xlabel("Date")
		plt.ylabel("O3")
		plt.ylim()
		#plt.show()
		plt.savefig("./"+graphName)
		plt.clf()

	#salve the dataframe into a csv
	def saveData(self, file_name):
		self.dataset.to_csv(file_name, index=False)

if __name__ == "__main__":
	
	print('Iniciando pre processamento!')
	path = "/home/leonildo/Downloads/california-data-ozone"
	dir_list = os.listdir(path)
	for file in dir_list:
		readPath = path + '/' + file
		dataSet = pd.read_csv(readPath) #O3
		a = Agregation(dataSet)
		a.initialize()
		a.plotTimeSeries(file.replace('.csv',''))
