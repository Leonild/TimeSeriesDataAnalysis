from geopy.geocoders import Nominatim
import geopy.distance
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from math import sin, cos, sqrt, atan2, radians

#trabalhando com as coordenadas
import geopandas as gpd
from pyproj import Proj, transform

import shapefile
import fiona
from shapely.geometry import shape,mapping, Point, Polygon, MultiPolygon
import sys


class Agregation:

	def __init__(self, dataset):
		self.dataset = self.initialize(dataset)
		print(self.dataset.head())
		#exit()

	# initialize the dataset by removing the undesire columns, assigning date to index, and grouping by month (commented)
	def initialize(self, dataset):
		#removing undesired columns
		dataset = dataset.drop(['Source','Site ID','UNITS','POC','STATE','STATE_CODE','COUNTY_CODE','COUNTY','Daily Max 8-hour Ozone Concentration','Site Name','DAILY_OBS_COUNT','PERCENT_COMPLETE','AQS_PARAMETER_CODE','AQS_PARAMETER_DESC','CBSA_CODE','CBSA_NAME'],axis=1)
		dataset = dataset.dropna() 
		# Convert the date to datetime64
		dataset['Date'] = pd.to_datetime(dataset['Date'], format='%m/%d/%Y')
		#dataset.index = dataset['Date'] # to work the next line
		#grouping by month
		#dataset = dataset.groupby(pd.Grouper(freq='M')).mean() #grouping data by month
		dataset = dataset.rename(columns={"Date": "Date", "DAILY_AQI_VALUE": "AQI", "SITE_LATITUDE": "latitude", "SITE_LONGITUDE": "longitude"})
		dataset.insert(loc=0, column='w', value=np.nan)
		return dataset

	#fill w field, identify what data belongs to the specific polygon
	def geopointWithinShape(self, grid):
		shapes = grid
		soa_shape_map_geo = shapes.to_crs(epsg=4326)  # EPSG 4326 = WGS84 = https://epsg.io/4326
		print("Iniciando verificacao")
		cities = self.dataset
		# reading each shape polygon
		i = 1 # polygon index
		aux = pd.DataFrame()
		isNotInto = True
		for multi in shapes['geometry']:
			# checar se eh POLYGON ou MULTIPOLYGON
			if multi.geom_type == 'MultiPolygon':
				for poly in multi:
					isNotInto = True
					poly = shape(poly)
					for index, row in cities.iterrows():
						pt = Point((row['longitude'], row['latitude']))
						#print("Lendo poligono: ", poly)
						if pt.within(poly):
							cities.at[index, 'w'] = i #atribuindo o id do arranjo
							isNotInto = False
							#row['w'] = i
							#aux = aux.append(cities.ix[index]) # ix is deprecated
							#aux = aux.append(cities.iloc[index]) #The frame.append method is deprecat
							#aux = pd.concat([aux,cities.iloc[index]])#,axis=1)
							print(pt, " está no poligono ", i)
					# if not data into plygon appending a row
					if(isNotInto):
						# Appending a row to csv with missing entries
						#row_contents = pd.DataFrame({'w': [i], 'Address': ['0'], 'DateLocalCount': 0.0, 'longitude': 0.0, 'latitude': 0.0,
						#'NO2AQI': 0.0, 'O3AQI': 0.0, 'SO2AQI': 0.0, 'COAQI': 0.0})
						row_contents = pd.DataFrame({'w': [i], 'Date': np.nan,'AQI': np.nan, 'latitude': np.nan, 'longitude': np.nan})
						#cities = cities.append(row_contents, ignore_index=True) #The frame.append method is deprecat
						cities = pd.concat([cities,row_contents])#, axis=1)
						#aux = aux.append(row_contents) #The frame.append method is deprecat
						#aux = pd.concat([aux,row_contents], ignore_index=True, sort=False)
						print("Adicionou a linha ", i)
					i+=1

			elif multi.geom_type == 'Polygon':
				poly = shape(multi)
				for index, row in cities.iterrows():
					pt = Point((row['longitude'], row['latitude']))
					#print("Lendo poligono: ", poly)
					if pt.within(poly):
						cities.at[index, 'w'] = i #atribuindo o id do arranjo
						#row['w'] = i
						isNotInto = False
						#aux = aux.append(cities.ix[index]) # ix is deprecated
						#aux = aux.append(cities.iloc[index]) #The frame.append method is deprecat
						#aux = pd.concat([aux,cities.iloc[index]])#, axis=1)
						print(pt, " está no poligono ", i)
				# if not data into plygon appending a row
				if(isNotInto):
					# Appending a row to csv with missing entries
					##row_contents = pd.DataFrame({'w': [i], 'Address': ['0'], 'DateLocalCount': 0.0, 'longitude': 0.0, 'latitude': 0.0,
					#	'NO2AQI': 0.0, 'O3AQI': 0.0, 'SO2AQI': 0.0, 'COAQI': 0.0})
					row_contents = pd.DataFrame({'w': [i], 'latitude': np.nan, 'longitude': np.nan,'AQI': np.nan})
					#cities = cities.append(row_contents, ignore_index=True) #The frame.append method is deprecat
					cities = pd.concat([cities,row_contents], ignore_index=True, sort=False)
					#append_list_as_row('./pre-final-file.csv', row_contents)
					#aux = aux.append(row_contents) #The frame.append method is deprecat
					#aux = pd.concat([aux,row_contents])#, axis=1)

			else:
				print("Nao identificado")
			i+=1
			isNotInto = True
			
		# aux.to_csv('./new-final-file.csv', index=False)
		#print(aux)
		self.dataset = cities.reset_index()#aux.reset_index()
		return cities#aux

	#grouping and save data by data location inside a polygon. This method will return the final file for the Autocorrelatin
	def groupByPolygon(self):
		grouped = self.dataset#[['w','AQI']]
		#grouped = grouped.dropna(axis='rows')  # Delete rows with NAs
		grouped = grouped.groupby(['w']).mean()#count()[['DateLocal']]  # testendo a funcao contar
		# grouped.to_csv('final-file.csv')
		self.dataset = grouped.reset_index()

	#salve the dataframe into a csv
	def saveData(self, file_name):
		self.dataset.to_csv(file_name, index=False)

if __name__ == "__main__":
	
	year = sys.argv[2]
	originalPath = sys.argv[1]
	path = originalPath + "/" + year + ".csv"
	print('Stagint spatioal agregation for ', path)
	#dataSet = pd.read_csv("./Los-Angeles/california-2019-O3.csv") #O3
	dataSet = pd.read_csv(path) #O3
	a = Agregation(dataSet)
	
	print('identificando dados nos poligonos')
	pathPoligon = originalPath + "/hex-grid.shp"
	grid = gpd.read_file(pathPoligon)
	a.geopointWithinShape(grid)
	print('Agrupando por poligono e salvando')
	#a.groupByPolygon()
	# print('salvando arquivo')
	pathTo = originalPath + "/" + year + "-final-file.csv"
	a.saveData(pathTo)
