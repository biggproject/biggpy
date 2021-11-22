#!/usr/bin/python3
# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.tsa.api as smt
import statsmodels.api as sm
#!pip install pmdarima
import pmdarima as pm


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

  
  
  
def split_train_test(data, test, plot):
  '''
  Split train/test from any given data point.
  :parameter
    :param ts: pandas Series
    :param test: num or str - test size    or index position
                 (ex. "yyyy-mm-dd", 1000)
  :return
    ts_train, ts_test
  '''
   if data.empty:
        raise ValueError("Input series must be not empty.")
   
   ## define splitting point
   if type(test) is float:
        split = int(len(ts)*(1-test))
        perc = test
   elif type(test) is str:
        split = ts.reset_index()[ 
                      ts.reset_index().iloc[:,0]==test].index[0]
        perc = round(len(ts[split:])/len(ts), 2)
   else:
        split = test
        perc = round(len(ts[split:])/len(ts), 2)
   print("--- splitting at index: ", split, "|", 
          ts.index[split], "| test size:", perc, " ---")
    
   ## split ts
   ts_train = ts.head(split)
   ts_test = ts.tail(len(ts)-split)
   if plot is True:
        fig, ax = plt.subplots(nrows=1, ncols=2, sharex=False, 
                               sharey=True, figsize=(15,5))
        ts_train.plot(ax=ax[0], grid=True, title="Train", 
                      color="black")
        ts_test.plot(ax=ax[1], grid=True, title="Test", 
                     color="blue")
        ax[0].set(xlabel=None)
        ax[1].set(xlabel=None)
        plt.show()
        
   return ts_train, ts_test

def evaluate_forecast(dtf, title, plot=True, figsize=(20,13)):
  '''
  Evaluation metrics for predictions.
  
  :param dtf: DataFrame with columns raw values, fitted training  
                 values, predicted test values
  :return: DataFrame with raw ts and forecast
  '''
  if dtf.empty:
        raise ValueError("Input series must be not empty.")
        
  try:
        ## residuals
        dtf["residuals"] = dtf["ts"] - dtf["model"]
        dtf["error"] = dtf["ts"] - dtf["forecast"]
        dtf["error_pct"] = dtf["error"] / dtf["ts"]
        
        ## kpi
        residuals_mean = dtf["residuals"].mean()
        residuals_std = dtf["residuals"].std()
        error_mean = dtf["error"].mean()
        error_std = dtf["error"].std()
        mae = dtf["error"].apply(lambda x: np.abs(x)).mean()
        mse = dtf["error"].apply(lambda x: x**2).mean()
        rmse = np.sqrt(mse)  #root mean squared error
        
        ## intervals
        dtf["conf_int_low"] = dtf["forecast"] - 1.96*residuals_std
        dtf["conf_int_up"] = dtf["forecast"] + 1.96*residuals_std
        dtf["pred_int_low"] = dtf["forecast"] - 1.96*error_std
        dtf["pred_int_up"] = dtf["forecast"] + 1.96*error_std
        
        ## plot
        if plot==True:
            fig = plt.figure(figsize=figsize)
            fig.suptitle(title, fontsize=20)   
            ax1 = fig.add_subplot(2,2, 1)
            ax2 = fig.add_subplot(2,2, 2, sharey=ax1)
            ax3 = fig.add_subplot(2,2, 3)
            ax4 = fig.add_subplot(2,2, 4)
            ### training
            dtf[pd.notnull(dtf["model"])][["ts","model"]].plot(color=["black","green"], title="Model", grid=True, ax=ax1)      
            ax1.set(xlabel=None)
            ### test
            dtf[pd.isnull(dtf["model"])][["ts","forecast"]].plot(color=["black","red"], title="Forecast", grid=True, ax=ax2)
            ax2.fill_between(x=dtf.index, y1=dtf['pred_int_low'], y2=dtf['pred_int_up'], color='b', alpha=0.2)
            ax2.fill_between(x=dtf.index, y1=dtf['conf_int_low'], y2=dtf['conf_int_up'], color='b', alpha=0.3)     
            ax2.set(xlabel=None)
            ### residuals
            dtf[["residuals","error"]].plot(ax=ax3, color=["green","red"], title="Residuals", grid=True)
            ax3.set(xlabel=None)
            ### residuals distribution
            dtf[["residuals","error"]].plot(ax=ax4, color=["green","red"], kind='kde', title="Residuals Distribution", grid=True)
            ax4.set(ylabel=None)
            plt.show()
            print("Training --> Residuals mean:", np.round(residuals_mean), " | std:", np.round(residuals_std))
            print("Test --> Error mean:", np.round(error_mean), " | std:", np.round(error_std),
                  " | mae:",np.round(mae), " | mape:",np.round(mape*100), "%  | mse:",np.round(mse), " | rmse:",np.round(rmse))
        
        return dtf[["ts","model","residuals","conf_int_low","conf_int_up", 
                    "forecast","error","pred_int_low","pred_int_up"]]
    
  except Exception as e:
        print("--- got error ---")
        print(e)
        

def param_tuning_sarimax (data, m, information_criterion='aic', max_oder):
  """
  Automatically discover the optimal order for a SARIMAX model. 
  The function works by conducting differencing tests to determine the order of differencing, d, 
  and then fitting models within ranges of defined start_p, max_p, start_q, max_q ranges.
  If the seasonal optional is enabled(allowing SARIMAX over ARIMA), it also seeks to identify 
  the optimal P and Q hyper- parameters after conducting the Canova-Hansen to determine the optimal order of seasonal differencing, D.

  In order to find the best model, it optimizes for a given information_criterion, one of (‘aic’, ‘aicc’, ‘bic’, ‘hqic’, ‘oob’) 
  and returns the ARIMA which minimizes the value.
  """
  if data.empty:
        raise ValueError("Input series must be not empty.")
      
  best_model = pm.auto_arima(data, exogenous=None,                                    
                                  seasonal=True, stationary=True, 
                                  m=m, information_criterion='aic', 
                                  max_order=max_order,                               
                                  max_p=2, max_d=1, max_q=2,                                     
                                  max_P=1, max_D=1, max_Q=2,                                   
                                  error_action='ignore')
  return best_model


def fit_sarimax(ts_train, order, seasonal_order, exog_train=None):
    '''
    Fit SARIMAX (Seasonal ARIMA with External Regressors):  
    y[t+1] = (c + a0*y[t] + a1*y[t-1] +...+ ap*y[t-p]) + (e[t] + 
                b1*e[t-1] + b2*e[t-2] +...+ bq*e[t-q]) + (B*X[t])
    :param ts_train: pandas timeseries
    :param order: tuple - ARIMA(p,d,q) --> p: lag order (AR), d: 
    degree of differencing (to remove trend), q: order 
                    of moving average (MA)
    :param seasonal_order: tuple - (P,D,Q,s) --> s: number of 
                    observations per seasonal (ex. 7 for weekly 
                    seasonality with daily data, 12 for yearly 
                    seasonality with monthly data)
    :param exog_train: pandas dataframe or numpy array
    :return Model and dtf with the fitted values
    '''
    ## train
    if data.empty:
        raise ValueError("Input series must be not empty.")
        
    model = smt.SARIMAX(data, order=order, 
                          seasonal_order=seasonal_order, 
                          exog=exog_train, enforce_stationarity=False, 
                          enforce_invertibility=False)
    model=model.fit()  
    dtf_train = data.to_frame(name="ts")
    dtf_train["model"] = model.fittedvalues
         
    return dtf_train
  
def test_sarimax (ts_train, ts_test, exog_test=None, p):
  """
  """
  dtf_test = ts_test[:p].to_frame(name="ts")

  if exog_test is None:
    dtf_test["forecast"] = model.predict(start=len(ts_train)+1, 
                            end=len(ts_train)+len(ts_test[:(p)-1]), 
                            exog=exog_test)
  else:
    dtf_test["forecast"] = model.predict(start=len(ts_train)+1, 
                            end=len(ts_train)+len(ts_test[:(p)-1]), 
                            exog=exog_test[:p])
  
if __name__ == '__main__':
    """
    This module is not supposed to run as a stand-alone module.
    This part below is only for testing purposes. 
    """
