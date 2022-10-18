import pandas as pd
import numpy as np
from datetime import datetime
from math import sin, cos, sqrt, atan2, radians


class Agregation:

	def __init__(self, dataset):
		self.dataset = dataset
		#removendo colunas indesejaveis
		self.dataset = self.dataset.drop(['Source','Site ID','UNITS','POC','STATE','STATE_CODE','COUNTY_CODE','COUNTY','Daily Max 8-hour Ozone Concentration','Site Name','DAILY_OBS_COUNT','PERCENT_COMPLETE','AQS_PARAMETER_CODE','AQS_PARAMETER_DESC','CBSA_CODE','CBSA_NAME'],axis=1)
		#filtrando pelo ano desejado
		# self.dataset = self.dataset[self.dataset['DateLocal'].str.contains("2011")]
		#dataset = dataset.replace(0, np.nan) #substituindo 0 com NaN
		self.dataset = self.dataset.dropna() # eliminando valores Nulos
		#grouped = dataset.groupby(["Latitude","Longitude"]).mean()#count()[['DateLocal']]  # testendo a funcao contar
		# grouped.to_csv('coletas-poluentes.csv')
		self.dataset = self.dataset.reset_index()
		print(self.dataset.head())
		print(self.dataset.tail())

	#salve the dataframe into a csv
	def saveData(self, file_name):
		self.dataset.to_csv(file_name, index=False)

if __name__ == "__main__":
	
	print('Iniciando pre processamento 1')
	dataSet = pd.read_csv("/home/leonildo/Downloads/ad_viz_plotval_data.csv") #O3
	print('Agrupando')
	a = Agregation(dataSet)
	
	print('identificando dados nos poligonos')
