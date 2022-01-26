#!/usr/bin/python3
# -*- coding: utf-8 -*-

import random

import holidays
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pmdarima as pm
import statsmodels.api as sm
import statsmodels.tsa.api as smt
from bokeh.plotting import figure, show
from bokeh.resources import INLINE
from bokeh.io import output_notebook
from bokeh.models import ColumnDataSource
from fbprophet import Prophet
from sklearn.model_selection import ParameterGrid


def test_stationarity_acf_pacf(data, sample, max_lag):
    """
    This function tests the stationarity and plot the autocorrelation
    and partial autocorrelation of the time series.
    Test stationarity by:
    - running Augmented Dickey-Fuller test with 95%
    In statistics, the Dickey–Fuller test tests the null hypothesis
    that a unit root is present in an autoregressive model.
    The alternative hypothesis is different depending
    on which version of the test is used,
    but is usually stationarity or trend-stationarity.
    - plotting mean and variance of a sample from data
    - plotting autocorrelation and partial autocorrelation

    p-value > 0.05: Fail to reject the null hypothesis (H0),
    the data has a unit root and is non-stationary.
    p-value <= 0.05: Reject the null hypothesis (H0),
    the data does not have a unit root and is stationary.
    This function is used to verify stationarity
    so that suitable forecasting methods can be applied.

    Partial autocorrelation is a summary of the relationship
    between an observation in a time series with observations
    at prior time steps with the relationships of intervening
    observations removed. The partial autocorrelation at lag k
    is the correlation that results after removing the effect
    of any correlations due to the terms at shorter lags.

    The autocorrelation for an observation and an observation
    at a prior time step comprises both the direct correlation
    and indirect correlations. These indirect correlations are
    a linear function of the correlation of the observation,
    with observations at intervening time steps.

    These correlations are used to define the parameters
    of the forecasting methods (lag).

    :parameter
     :param data: timeSeries for which the stationarity as to be evaluated.
     :param sample: Sample (float) of the data that will be evaluated.
     :param max_lag: Maximum number of lag which included in test.
                    The default value is 12*(nobs/100)^{1/4}.
    :return:
      plot of the mean and variance of the sample with the p-value
      and plot of the autocorrelation and partial autocorrelation.
    """
    if data.empty:
        raise ValueError("Input series must be not empty.")

    elif 0.0 > sample or sample > 1.0:
        raise ValueError("Sample value should be between 0 and 1")

    for col in data:
        ts_ax = plt.subplot2grid(shape=(2, 2), loc=(0, 0), colspan=2)
        pacf_ax = plt.subplot2grid(shape=(2, 2), loc=(1, 0))
        acf_ax = plt.subplot2grid(shape=(2, 2), loc=(1, 1))
        dtf_ts = data[col].to_frame(name="ts")
        sample_size = int(len(data) * sample)
        dtf_ts["mean"] = dtf_ts["ts"].head(sample_size).mean()
        dtf_ts["lower"] = dtf_ts["ts"].head(sample_size).mean() + dtf_ts["ts"].head(sample_size).std()
        dtf_ts["upper"] = dtf_ts["ts"].head(sample_size).mean() - dtf_ts["ts"].head(sample_size).std()
        dtf_ts["ts"].plot(ax=ts_ax, color="black", legend=False)
        dtf_ts["mean"].plot(
            ax=ts_ax, legend=False, color="red",
            linestyle="--", linewidth=0.7)
        ts_ax.fill_between(
            x=dtf_ts.index, y1=dtf_ts['lower'],
            y2=dtf_ts['upper'], color='lightskyblue', alpha=0.4)
        dtf_ts["mean"].head(sample_size).plot(
            ax=ts_ax, legend=False,
            color="red", linewidth=0.9)
        ts_ax.fill_between(
            x=dtf_ts.index, y1=dtf_ts['lower'],
            y2=dtf_ts['upper'], color='lightskyblue', alpha=0.4)
        dtf_ts["mean"].head(sample_size).plot(
            ax=ts_ax, legend=False,
            color="red", linewidth=0.9)
        ts_ax.fill_between(
            x=dtf_ts.head(sample_size).index,
            y1=dtf_ts['lower'].head(sample_size),
            y2=dtf_ts['upper'].head(sample_size), color='lightskyblue')
        adfuller_test = sm.tsa.stattools.adfuller(
            data, maxlag=max_lag, autolag="AIC")
        adf, p, critical_value = adfuller_test[0], adfuller_test[1], adfuller_test[4]["5%"]
        p = round(p, 3)
        conclusion = "Stationary" if p < 0.05 else "Non-Stationary"
        ts_ax.set_title(
            'Dickey-Fuller Test 95%: ' + conclusion +
            '(p value: ' + str(p) + ')')

        # pacf (for AR) and acf (for MA)
        smt.graphics.plot_pacf(
            data, lags=30, ax=pacf_ax,
            title="Partial Autocorrelation (for AR component)")
        smt.graphics.plot_acf(
            data, lags=30, ax=acf_ax,
            title="Autocorrelation (for MA component)")
        plt.tight_layout()


def split_train_test(ts, test, plot):
    """
    Split train/test from any given data point.
    :parameter
      :param ts: pandas Series
      :param test: num or str - test size    or index position
                   (ex. "yyyy-mm-dd", 1000)
      :param plot: bool to decide if the 2 new time series have to be plotted
    :return
      ts_train, ts_test
    """
    if ts.empty:
        raise ValueError("Input series must be not empty.")

    # define splitting point
    if type(test) is float:
        split = int(len(ts) * (1 - test))
        perc = test
    elif type(test) is str:
        split = ts.reset_index()[
            ts.reset_index().iloc[:, 0] == test].index[0]
        perc = round(len(ts[split:]) / len(ts), 2)
    else:
        split = test
        perc = round(len(ts[split:]) / len(ts), 2)
    print("--- splitting at index: ", split, "|",
          ts.index[split], "| test size:", perc, " ---")

    # split ts
    ts_train = ts.head(split)
    ts_test = ts.tail(len(ts) - split)
    if plot is True:
        fig, ax = plt.subplots(nrows=1, figsize=(15, 5), ncols=2, sharex=False, sharey=True)
        ts_train.plot(ax=ax[0], grid=True, title="Train",
                      color="black")
        ts_test.plot(ax=ax[1], grid=True, title="Test",
                     color="blue")
        ax[0].set(xlabel=None)
        ax[1].set(xlabel=None)
        plt.show()

    return ts_train, ts_test


def evaluate_forecast(dtf, plot=True):
    """
    Evaluation metrics for predictions.

    :parameter
      :param dtf: DataFrame with columns raw values, fitted training
                   values, predicted test values.
      :param plot:  Bool to visualize on a plot. Default is True.

    :return:
      DataFrame with raw ts and forecast.
    """
    try:
        # residuals
        dtf["error"] = dtf["ts"] - dtf["forecast"]
        dtf["error_pct"] = dtf["error"] / dtf["ts"]

        # kpi
        error_mean = dtf["error"].mean()
        error_std = dtf["error"].std()
        # mape = dtf["error_pct"].apply(lambda x: np.abs(x)).mean()
        mae = dtf["error"].apply(lambda x: np.abs(x)).mean()
        mse = dtf["error"].apply(lambda x: x ** 2).mean()
        rmse = np.sqrt(mse)  # root mean squared error
        r2 = 1 - (sum(dtf["error"] ** 2) / sum((dtf["ts"] - np.mean(dtf["ts"])) ** 2))

        # intervals
        dtf["pred_int_low"] = dtf["forecast"] - 1.96 * error_std
        dtf["pred_int_up"] = dtf["forecast"] + 1.96 * error_std

        # plot
        if plot is True:
            dtf = dtf.reset_index()
            dtf['time'] = pd.to_datetime(dtf['time'])
            dtf = dtf.set_index('time')
            dtf = dtf.resample('1H').mean().fillna(0)
            output_notebook(INLINE)

            p = figure(x_axis_type="datetime", x_axis_label='Dates', y_axis_label='Consumption (Wh)')
            data_source = ColumnDataSource(dtf)
            p.line(x='time', y='ts', source=data_source, legend_label='Value')
            p.line(x='time', y='forecast', source=data_source, color='red', legend_label='Pred')

            show(p)
        print("Test --> Error mean:", np.round(error_mean), " | r2:", np.round(r2, 2),
              " | mae:", np.round(mae), " | rmse:", np.round(rmse))

        return dtf

    except Exception as e:
        print("--- got error ---")
        print(e)


def param_tuning_sarimax(data, m, max_order, information_criterion='aic'):
    """
    Automatically discover the optimal order for a SARIMAX model.
    The function works by conducting differencing tests to determine
    the order of differencing, d, and then fitting models within ranges
    of defined start_p, max_p, start_q, max_q ranges.

    If the seasonal optional is enabled(allowing SARIMAX over ARIMA),
    it also seeks to identify the optimal P and Q hyperparameters
    after conducting the Canova-Hansen
    to determine the optimal order of seasonal differencing, D.

    In order to find the best model, it optimizes for
    a given information_criterion
    and returns the ARIMA which minimizes the value.

    :parameter
      :param data: timeSeries used to fit the sarimax estimator.
      :param m: refers to the number of periods in each season.
                      For example, m is 4 for quarterly data,
                      12 for monthly data, or 1 for annual data.
      :param max_order: maximum value of p+q+P+Q.
                      If p+q >= max_order, a model will not be
                      fit with those parameters and will progress
                      to the next combination. Default is 5.
      :param information_criterion: used to select the best model.
                    Possibilities are ‘aic’, ‘bic’, ‘hqic’, ‘oob’.
                    Default is 'aic'.

    :return
      best_model: model with the optimal parameters
    """
    if data.empty:
        raise ValueError("Input series must be not empty.")
    elif not isinstance(m, int):
        raise ValueError("m must be an integer.")

    best_model = pm.auto_arima(data, start_p=1, start_q=1, start_P=1, start_Q=1,
                               test='adf',  # use adftest to find optimal 'd'
                               max_p=max_order, max_q=max_order,  # maximum p and q
                               max_P=max_order, max_D=1,  # maximum P and D and Q
                               max_Q=max_order,
                               m=m,  # frequency of series # m=12 Monthly m=24 Hourly
                               d=None,  # let model determine 'd'
                               seasonal=False,  # No Seasonality
                               trace=True,
                               exogenous=None,
                               error_action='ignore',
                               suppress_warnings=True,
                               stepwise=True)

    return best_model


def fit_sarimax(ts_train, order, seasonal_order, exog_train=None):
    """
    Fit SARIMAX (Seasonal ARIMA with External Regressors):
    y[t+1] = (c + a0*y[t] + a1*y[t-1] +...+ ap*y[t-p]) + (e[t] +
                  b1*e[t-1] + b2*e[t-2] +...+ bq*e[t-q]) + (B*X[t])
    :parameter
    :param ts_train: pandas timeseries
    :param order: tuple - ARIMA(p,d,q) --> p: lag order (AR), d:
    degree of differencing (to remove trend), q: order
                        of moving average (MA).
    :param seasonal_order: tuple - (P,D,Q,s) --> s: number of
                        observations per seasonal (ex. 7 for weekly
                        seasonality with daily data, 12 for yearly
                        seasonality with monthly data).
    :param exog_train: pandas dataframe or numpy array.

    :return
    Model and dtf with the fitted values
    """
    # train
    if ts_train.empty:
        raise ValueError("Train series must be not empty.")

    model = smt.SARIMAX(
        ts_train, order=order, seasonal_order=seasonal_order,
        exog=exog_train, enforce_stationarity=False,
        enforce_invertibility=False)
    model = model.fit()
    for col in ts_train:
        dtf_train = ts_train[col].to_frame(name="ts")
        dtf_train["model"] = model.fittedvalues

    return dtf_train, model


def test_sarimax(ts_train, ts_test, exog_test, p, model):
    """
    The function uses the model from the fit_sarimax function
    to make predictions for the future value.

    :parameter
      :param ts_train: timeSeries used to train the model.
      :param ts_test: timeSeries used to test the model.
      :param exog_test: timeSeries containing the exogeneous variables.
      :param p: number of periods to be forcasted.
      :param model: model from the fit_sarimax function.

    :return
      Dataframe containing the true values and the forecasted ones.
    """

    if ts_train.empty:
        raise ValueError("Train series must be not empty.")
    elif ts_test.empty:
        raise ValueError("Test series must be not empty.")
    elif not isinstance(p, int):
        raise ValueError("p must be an integer.")

    for col in range(ts_test.shape[1]):
        dtf_test = ts_test.iloc[:p, col].to_frame(name="ts")

    if exog_test is None:
        dtf_test["forecast"] = model.predict(
            start=len(ts_train),
            end=len(ts_train) + len(ts_test[1:p]),
            exog=exog_test)
    else:
        dtf_test["forecast"] = model.predict(
            start=len(ts_train),
            end=len(ts_train) + len(ts_test[:p]),
            exog=exog_test[:p])
    dtf_test = dtf_test.round()
    return dtf_test


def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def input_prophet(ts_train, ts_test):
    """

    The input to Prophet is always a dataframe with two columns: ds and y.
    The ds (datestamp) column should be of a format expected by Pandas,
    ideally YYYY-MM-DD for a date or YYYY-MM-DD HH:MM:SS for a timestamp.
    The function rename and adapt the format of ds.

    :parameter
      :param ts_train: timeSeries used to train the model.
      :param ts_test: timeSeries used to test the model.

     :return
        dtf_train: pandas Dataframe with the train set
                 with columns 'ds' (dates),
                 values, 'cap' (capacity if growth="logistic"),
                 other additional regressor.
        dtf_test: pandas Dataframe with the test set
                 with columns 'ds' (dates),
                 values, 'cap' (capacity if growth="logistic"),
                 other additional regressor.
    """

    if ts_train.empty:
        raise ValueError("Train series must be not empty.")
    elif ts_test.empty:
        raise ValueError("Test series must be not empty.")

    dtf_train = ts_train.reset_index()
    dtf_train.rename(index=str, columns={'time': 'ds'}, inplace=True)

    # dtf_train= pd.merge(dtf_train, temp, how='left', on='ds')
    dtf_train = dtf_train.fillna(0)

    dtf_test = ts_test.reset_index()
    dtf_test.rename(index=str, columns={'time': 'ds'}, inplace=True)

    # dtf_test= pd.merge(dtf_test, temp, how='left', on='ds')
    dtf_test = dtf_test.fillna(0)

    dtf_train['ds'] = dtf_train['ds'].dt.tz_localize(None)
    dtf_test['ds'] = dtf_test['ds'].dt.tz_localize(None)

    return dtf_train, dtf_test


def param_tuning_prophet(dtf_train, p, seasonality_mode, ts_test,
                         changepoint_prior_scale, holidays_prior_scale,
                         n_changepoints):
    """
    This function performs a search on all the parameters of
    the parameter grid defined and identifies
    the best parameter set for a prophet model, given a MAPE scoring.

    :parameter
      :param dtf_train: pandas Dataframe with columns 'ds' (dates),
                 values, 'cap' (capacity if growth="logistic"),
                 other additional regressor.
      :param p: number of periods to be forecasted.
      :param seasonality_mode: multiplicative and/or
                 additive seasonality.
      :param ts_test: timeSeries used to test the model.
      :param changepoint_prior_scale: list of floats,
                 tests the influence of the changepoints.
      :param holidays_prior_scale: list of floats,
                 tests the influence of the holidays.
      :param n_changepoints: list of int,
                 maximum number of trend changepoints allowed
                 when modelling the trend.

    :return
       Optimal parameters for the prophet model.

    """
    if dtf_train.empty:
        raise ValueError("Input series must be not empty.")

    holiday = pd.DataFrame([])
    for date, name in sorted(
            holidays.Greece(years=[2019, 2020, 2021, 2022]).items()):
        # pd.DatetimeIndex(holiday['ds']).year[-1] in place of 2021
        holiday = holiday.append(pd.DataFrame(
            {'ds': date, 'holiday': "GR-Holidays"},
            index=[0]), ignore_index=True)
    holiday['ds'] = pd.to_datetime(holiday['ds'],
                                   format='%Y-%m-%d', errors='ignore')

    params_grid = {'seasonality_mode': seasonality_mode,
                   'changepoint_prior_scale': changepoint_prior_scale,
                   'holidays_prior_scale': holidays_prior_scale,
                   'n_changepoints': n_changepoints}

    grid = ParameterGrid(params_grid)

    model_parameters = pd.DataFrame(columns=['MAPE', 'Parameters'])

    for g in grid:
        random.seed(0)
        train_model = Prophet(
            changepoint_prior_scale=g['changepoint_prior_scale'],
            holidays_prior_scale=g['holidays_prior_scale'],
            n_changepoints=g['n_changepoints'],
            seasonality_mode=g['seasonality_mode'],
            weekly_seasonality=True,
            daily_seasonality=True,
            yearly_seasonality=True,
            holidays=holiday,
            interval_width=0.95)

        train_model.add_country_holidays(country_name='GR')
        test = dtf_train
        test.columns = ['ds', 'y']
        train_model.fit(test)
        train_forecast = train_model.make_future_dataframe(
            periods=p, freq='15T', include_history=False)
        train_forecast = train_model.predict(train_forecast)
        test = train_forecast[['ds', 'yhat']]
        ts_test = ts_test.reset_index().value
        actual = ts_test.iloc[:p, ]
        mape = mean_absolute_percentage_error(actual, abs(test['yhat']))

        print('Mean Absolute Percentage Error(MAPE)--------', mape)
        model_parameters = model_parameters.append(
            {'MAPE': mape, 'Parameters': g}, ignore_index=True)

    '''optimals = model_parameters.groupby('Data', as_index=False).max()
    optimals = optimals.merge(
        model_parameters, on=['MAPE', 'Data'], how='left')'''

    return model_parameters


def fit_prophet(dtf_train):
    """
    Fits prophet on the Data.
    Prophet makes use of a decomposable time series
    model with three main model components:
        y = trend + seasonality + holidays

    They are combined in this equation: y(t) = g(t) + s(t) + h(t) + e(t)

    - g(t): trend models non-periodic changes; linear or logistic.
    - s(t): seasonality represents periodic changes;
    i.e. weekly, monthly, yearly.
    - h(t): ties in effects of holidays;
    on potentially irregular schedules ≥ 1 day(s).
    - The error term e(t) represents any idiosyncratic changes
    which are not accommodated by the model;

    :parameter
        :param dtf_train: pandas Dataframe with columns 'ds' (dates),
                 values, 'cap' (capacity if growth="logistic"),
                 other additional regressor.

    :return
        trained model.
    """

    if dtf_train.empty:
        raise ValueError("Input series must be not empty.")
    subdf = dtf_train.dropna()
    subdf.columns = ['ds', 'y']

    # Adding the holidays as a parameter:
    holiday = pd.DataFrame([])
    for date, name in sorted(holidays.Greece(
            years=list(
                range(
                    pd.DatetimeIndex(dtf_train['ds']).year[0],
                    pd.DatetimeIndex(dtf_train['ds']).year[-1] + 1))).items()):
        holiday = holiday.append(pd.DataFrame(
            {'ds': date, 'holiday': "GR-Holidays"},
            index=[0]), ignore_index=True)
    holiday['ds'] = pd.to_datetime(
        holiday['ds'], format='%Y-%m-%d', errors='ignore')

    model = Prophet(
        growth="linear",
        n_changepoints=100,
        yearly_seasonality="auto",
        weekly_seasonality="auto",
        daily_seasonality=True,
        holidays=holiday,
        seasonality_mode='multiplicative')

    model.add_country_holidays(country_name='GR')
    model = model.fit(subdf)
    return model


def test_prophet(dtf_test, model, freq, p):
    """
    This function makes the prediction using
    the model created in the fit_prophet function.

    :parameter
        :param dtf_test: pandas Dataframe containing the test set
                 with columns 'ds' (dates),
                 values, 'cap' (capacity if growth="logistic"),
                 other additional regressor.
        :param model: model from the fit_prophet function.
        :param p: number of periods to be forecasted.
        :param freq: str - "D" daily, "M" monthly, "Y" annual, "MS"
                           monthly start ...
    :return
        DataFrame containing the true and forecasted values.
    """

    if dtf_test.empty:
        raise ValueError("Test series must be not empty.")
    elif not isinstance(p, int):
        raise ValueError("p must be an integer.")

    dtf_prophet = model.make_future_dataframe(
        periods=p, freq=freq, include_history=True)

    dtf_prophet = model.predict(dtf_prophet)
    dtf_prophet = dtf_prophet.round()
    dtf_forecast = dtf_test.merge(
        dtf_prophet[["ds", "yhat"]],
        how="left").rename(
        columns={'yhat': 'forecast',
                 'y': 'ts', 'ds': 'time'}).set_index("time")

    return dtf_forecast


if __name__ == '__main__':
    """
    This module is not supposed to run as a stand-alone module.
    This part below is only for testing purposes.
    """
