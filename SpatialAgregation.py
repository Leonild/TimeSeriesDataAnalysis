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
		# self.dataset = dataset
		#removendo colunas indesejaveis
		# self.dataset = self.dataset.drop(['Index','StateCode','CountyCode','SiteNum','NO2Units','O3Units','SO2Units','COUnits'],axis=1)
		#filtrando pelo ano desejado
		# self.dataset = self.dataset[self.dataset['DateLocal'].str.contains("2011")]

		#Filtrando pelo estado de enteresse
		#dataset = dataset[dataset['State Name'].str.contains("California")]

		#dataset = dataset.drop(["State Code","County Code","Site Num","Parameter Code","POC","Datum","Parameter Name","Sample Duration","Pollutant Standard","Date Local","Units of Measure","Event Type","Observation Count","Observation Percent","Arithmetic Mean","1st Max Value","1st Max Hour","Method Code","Method Name","Local Site Name","Address","State Name","County Name","City Name","CBSA Name","Date of Last Change"],axis=1)
		#dataset = dataset.replace(0, np.nan) #substituindo 0 com NaN
		#dataset = dataset.dropna() # eliminando valores Nulos
		#grouped = dataset.groupby(["Latitude","Longitude"]).mean()#count()[['DateLocal']]  # testendo a funcao contar
		# grouped.to_csv('coletas-poluentes.csv')
		#self.dataset = grouped.reset_index()
		self.dataset = dataset
		print(self.dataset)

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
						if pt.within(poly):
							# cities.ix[index, 'w'] = i #atribuindo o id do arranjo
							isNotInto = False
							row['w'] = i
							aux = aux.append(cities.ix[index])
							#print(pt, " está no poligono ", i)
					# if not data into plygon appending a row
					if(isNotInto):
						# Appending a row to csv with missing entries
						#row_contents = pd.DataFrame({'w': [i], 'Address': ['0'], 'DateLocalCount': 0.0, 'longitude': 0.0, 'latitude': 0.0,
						#'NO2AQI': 0.0, 'O3AQI': 0.0, 'SO2AQI': 0.0, 'COAQI': 0.0})
						row_contents = pd.DataFrame({'w': [i], 'latitude': np.nan, 'longitude': np.nan,'AQI': np.nan})
						cities = cities.append(row_contents, ignore_index=True)
						aux = aux.append(row_contents)
						print("Adicionou a linha ", i)
					i+=1

			elif multi.geom_type == 'Polygon':
				poly = shape(multi)
				for index, row in cities.iterrows():
					pt = Point((row['longitude'], row['latitude']))
					if pt.within(poly):
						#cities.ix[index, 'w'] = i #atribuindo o id do arranjo
						row['w'] = i
						isNotInto = False
						aux = aux.append(cities.ix[index])
						# print(pt, " está no poligono ", i)
				# if not data into plygon appending a row
				if(isNotInto):
					# Appending a row to csv with missing entries
					##row_contents = pd.DataFrame({'w': [i], 'Address': ['0'], 'DateLocalCount': 0.0, 'longitude': 0.0, 'latitude': 0.0,
					#	'NO2AQI': 0.0, 'O3AQI': 0.0, 'SO2AQI': 0.0, 'COAQI': 0.0})
					row_contents = pd.DataFrame({'w': [i], 'latitude': np.nan, 'longitude': np.nan,'AQI': np.nan})
					cities = cities.append(row_contents, ignore_index=True)
					#append_list_as_row('./pre-final-file.csv', row_contents)
					aux = aux.append(row_contents)

			else:
				print("Nao identificado")
			i+=1
			isNotInto = True
			
		# aux.to_csv('./new-final-file.csv', index=False)
		#print(aux)
		self.dataset = aux.reset_index()
		return aux

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
	
	path = sys.argv[1]
	print('Stagint spatioal agregation for ', path)
	#dataSet = pd.read_csv("./Los-Angeles/california-2019-O3.csv") #O3
	dataSet = pd.read_csv(path) #O3
	a = Agregation(dataSet)
	
	print('identificando dados nos poligonos')
	pathPoligon = path + "/grid/hex-grid.shp"
	grid = gpd.read_file(pathPoligon)
	a.geopointWithinShape(grid)
	print('Agrupando por poligono e salvando')
	a.groupByPolygon()
	# print('salvando arquivo')
	pathTo = path + "final-file.csv"
	a.saveData(pathTo)