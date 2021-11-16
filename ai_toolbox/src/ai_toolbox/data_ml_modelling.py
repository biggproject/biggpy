#!/usr/bin/python3
# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.tsa.api as smt
import statsmodels.api as sm


def test_stationarity_acf_pacf (data, sample, maxLag):
  """
  This function tests the stationarity and plot the autocorrelation and partial autocorrelation of the time series.
  Test stationarity by:
    - running Augmented Dickey-Fuller test wiht 95%
    In statistics, the Dickey–Fuller test tests the null hypothesis that a unit root is present in an autoregressive model. 
    The alternative hypothesis is different depending on which version of the test is used, but is usually stationarity or trend-stationarity. 
    - plotting mean and variance of a sample from data
    - plottig autocorrelation and partial autocorrelation

    p-value > 0.05: Fail to reject the null hypothesis (H0), the data has a unit root and is non-stationary.
    p-value <= 0.05: Reject the null hypothesis (H0), the data does not have a unit root and is stationary.
    This function is used to verify stationarity so that suitable forecasting methodes can be applied.
    
   Partial autocorrelation is a summary of the relationship between an observation in a time series with observations at prior time steps 
   with the relationships of intervening observations removed. The partial autocorrelation at lag k is the correlation that results 
   after removing the effect of any correlations due to the terms at shorter lags.

   The autocorrelation for an observation and an observation at a prior time step is comprised of both the direct correlation and indirect correlations. 
   These indirect correlations are a linear function of the correlation of the observation, with observations at intervening time steps.

   These correlations are used to define the parameters of the forecasting methods (lag).
   
   :param data: timeSeries for which the stationarity as to be evaluated.
   :param sample: Sample (float) of the data that will be evaluated.
   :param maxLag: Maximum number of lag which included in test. The default value is 12*(nobs/100)^{1/4}.
   :return: plot of the mean and variance of the sample with the p-value and plot of the autocorrelation and partial autocorrelation.
  """
  if data.empty:
        raise ValueError("Input series must be not empty.")
      
  elif 0.0 > sample or sample > 1.0: 
        raise ValueError ("Sample value should be between 0 and 1")
      
  fig = plt.figure(figsize=(15,10))
  ts_ax = plt.subplot2grid(shape=(2,2), loc=(0,0), colspan=2)
  pacf_ax = plt.subplot2grid(shape=(2,2), loc=(1,0))
  acf_ax = plt.subplot2grid(shape=(2,2), loc=(1,1))   
  dtf_ts = data.to_frame(name="ts")
  sample_size = int(len(data)*sample)
  dtf_ts["mean"] = dtf_ts["ts"].head(sample_size).mean()
  dtf_ts["lower"] = dtf_ts["ts"].head(sample_size).mean() + dtf_ts["ts"].head(sample_size).std()
  dtf_ts["upper"] = dtf_ts["ts"].head(sample_size).mean()  - dtf_ts["ts"].head(sample_size).std()
  dtf_ts["ts"].plot(ax=ts_ax, color="black", legend=False)
  dtf_ts["mean"].plot(ax=ts_ax, legend=False, color="red", linestyle="--", linewidth=0.7)
  ts_ax.fill_between(x=dtf_ts.index, y1=dtf_ts['lower'], y2=dtf_ts['upper'], color='lightskyblue', alpha=0.4)
  dtf_ts["mean"].head(sample_size).plot(ax=ts_ax, legend=False, color="red", linewidth=0.9)
  ts_ax.fill_between(x=dtf_ts.index, y1=dtf_ts['lower'], y2=dtf_ts['upper'], color='lightskyblue', alpha=0.4)
  dtf_ts["mean"].head(sample_size).plot(ax=ts_ax, legend=False, color="red", linewidth=0.9)
  ts_ax.fill_between(x=dtf_ts.head(sample_size).index, 
                            y1=dtf_ts['lower'].head(sample_size), 
                            y2=dtf_ts['upper'].head(sample_size),
                            color='lightskyblue')
  adfuller_test = sm.tsa.stattools.adfuller(data, maxlag=maxlag,
                                                    autolag="AIC")
  adf, p, critical_value = adfuller_test[0], adfuller_test[1], adfuller_test[4]["5%"]
  p = round(p, 3)
  conclusion = "Stationary" if p < 0.05 else "Non-Stationary"
  ts_ax.set_title('Dickey-Fuller Test 95%: '+conclusion+
                          '(p value: '+str(p)+')')
          
  ## pacf (for AR) and acf (for MA) 
  smt.graphics.plot_pacf(data, lags=30, ax=pacf_ax, 
                  title="Partial Autocorrelation (for AR component)")
  smt.graphics.plot_acf(data, lags=30, ax=acf_ax,
                  title="Autocorrelation (for MA component)")
  plt.tight_layout()
  
  
  
  
if __name__ == '__main__':
    """
    This module is not supposed to run as a stand-alone module.
    This part below is only for testing purposes. 
    """
