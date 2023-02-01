# TimeSeriesDataAnalysis
Part of the studies of Spatial and Temporal Data Granularity Analysis in the Internet of Things

This project is a sequence of the work "Analysis of Spatially Distributed Data in Internet of Things in the Environmental Context", published in the [Sensors journal](https://www.mdpi.com/1424-8220/22/5/1693).

*Abstract:*
The Internet of Things consists of ``things'' made up of small sensors and actuators capable of interacting with the environment. The combination of devices with sensor networks and Internet access enables communication between the physical world and cyber space, providing the development of solutions to many real world problems. However, most of the existing applications are dedicated to solving a specific problem using only private sensor networks, which limits the actual capacity of the Internet of things. In addition, these applications do not worry about the quality of service offered by the sensor network or the sensors that compose them and can collect a large amount of inaccurate or irrelevant data, which can cause significant harm to companies. In this context, this proposal aims to develop a metaheuristic-based solution to precise analyses of the quality of data and the infrastructure for IoT environments.

## Making shapefiles for data agregation

MakeTheShape.R makes a grid of hexagons into a specific shapefile
To run the script is necessary to provide de dimensions of the hexagon side and the directory path to the original shapefile

Input: Hexagon's dimension side (in meters) AND directory path to the original shapefile

Command: ```Rscript MakeTheShape.R <dimension> <path>```

Example: ```Rscript MakeTheShape.R 500000 ./USA-shape```

Output: the shape file with a grid in the directory <path>/<dimension>

It's also possible to execute it automatically by running run-MakeTheShape.sh followed by the initial size of the shape grid, the final size of the shape, and then the number of the shape files in that range. However, check the shape path in the script before execution.

```./run-MakeTheShape.sh <First shape size> <Last shape size> <Number of shapes in this range>```

After generating the shapefiles it's possible to execute the SpatialAgregation.py script.

## Data Agregation

Based on the geographical point of the data available in the CSV files (sensors' data), the SpatialAgregation.py script aggregates the sensors' data point by point described in the shapefile. It's crucial to check the directories in the code file before execution.

```python3 SpatialAgregation.py```

## Spatial Analysis

The spatial data analysis by Moran's method and Pareto domination is possible by using the code available in the [Spatial Data Analysis](https://github.com/Leonild/SpatialDataAnalysis) repository. Those analyses were published in [Sensors journal](https://www.mdpi.com/1424-8220/22/5/1693).


## Time Series Analysis

The second part of Leonildo's Ph.D. is to realize the Temporal Data Granularity Analysis. For this, we implemented some time series methods to execute that analysis.

### Implemented methods

*ARMA:* In the statistical analysis of time series, autoregressive–moving-average (ARMA) models provide a parsimonious description of a (weakly) stationary stochastic process in terms of two polynomials, one for the autoregression (AR) and the second for the moving average (MA). 

*ARIMA:* In statistics and econometrics, and in particular in time series analysis, an autoregressive integrated moving average (ARIMA) model is a generalization of an autoregressive moving average (ARMA) model. Both of these models are fitted to time series data either to better understand the data or to predict future points in the series (forecasting). ARIMA models are applied in some cases where data show evidence of non-stationarity in the sense of mean (but not variance/autocovariance), where an initial differencing step (corresponding to the "integrated" part of the model) can be applied one or more times to eliminate the non-stationarity of the mean function.

*Correlogram:* In the analysis of data, a correlogram is a chart of correlation statistics.

*SARIMA:* Seasonal Autoregressive Integrated Moving Average, SARIMA or Seasonal ARIMA, is an extension of ARIMA that explicitly supports univariate time series data with a seasonal component. It adds three new hyperparameters to specify the autoregression (AR), differencing (I) and moving average (MA) for the seasonal component of the series, as well as an additional parameter for the period of the seasonality. "A seasonal ARIMA model is formed by including additional seasonal terms in the ARIMA […] The seasonal part of the model consists of terms that are very similar to the non-seasonal components of the model, but they involve backshifts of the seasonal period." Page 242, Forecasting: principles and practice, 2013."