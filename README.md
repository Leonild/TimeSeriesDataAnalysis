# TimeSeriesDataAnalysis
Part of the studies of Spatial and Temporal Data Granularity Analysis in the Internet of Things

This project is a sequence of the work "Analysis of Spatially Distributed Data in Internet of Things in the Environmental Context", published on Sensors journal <https://www.mdpi.com/1424-8220/22/5/1693>

*Abstract:*
The Internet of Things consists of ``things'' made up of small sensors and actuators capable of interacting with the environment. The combination of devices with sensor networks and Internet access enables communication between the physical world and cyber space, providing the development of solutions to many real world problems. However, most of the existing applications are dedicated to solving a specific problem using only private sensor networks, which limits the actual capacity of the Internet of things. In addition, these applications do not worry about the quality of service offered by the sensor network or the sensors that compose them and can collect a large amount of inaccurate or irrelevant data, which can cause significant harm to companies. In this context, this proposal aims to develop a metaheuristic-based solution to precise analyses of the quality of data and the infrastructure for IoT environments.


# Implemented methods

*ARMA:* In the statistical analysis of time series, autoregressive–moving-average (ARMA) models provide a parsimonious description of a (weakly) stationary stochastic process in terms of two polynomials, one for the autoregression (AR) and the second for the moving average (MA). 

*ARIMA:* In statistics and econometrics, and in particular in time series analysis, an autoregressive integrated moving average (ARIMA) model is a generalization of an autoregressive moving average (ARMA) model. Both of these models are fitted to time series data either to better understand the data or to predict future points in the series (forecasting). ARIMA models are applied in some cases where data show evidence of non-stationarity in the sense of mean (but not variance/autocovariance), where an initial differencing step (corresponding to the "integrated" part of the model) can be applied one or more times to eliminate the non-stationarity of the mean function.

*Correlogram:* In the analysis of data, a correlogram is a chart of correlation statistics.

*SARIMA:* Seasonal Autoregressive Integrated Moving Average, SARIMA or Seasonal ARIMA, is an extension of ARIMA that explicitly supports univariate time series data with a seasonal component. It adds three new hyperparameters to specify the autoregression (AR), differencing (I) and moving average (MA) for the seasonal component of the series, as well as an additional parameter for the period of the seasonality. "A seasonal ARIMA model is formed by including additional seasonal terms in the ARIMA […] The seasonal part of the model consists of terms that are very similar to the non-seasonal components of the model, but they involve backshifts of the seasonal period." Page 242, Forecasting: principles and practice, 2013."