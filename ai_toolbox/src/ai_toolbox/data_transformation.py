#!/usr/bin/python3
# -*- coding: utf-8 -*-

from datetime import datetime
from datetime import timedelta
from itertools import chain
from typing import Union

import holidays
import numpy as np
import pandas as pd
import pytz
from pandas.tseries.frequencies import to_offset
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer
from sklearn.utils.validation import check_is_fitted, check_X_y

from ai_toolbox.data_preparation import detect_time_step


def yearly_profile_detection(data, exclude_days=None):
    """
    The function returns the yearly profile of the input time series.
    It aggregates values of the input data over multiple years using
    the median and return a time series at daily resolution, aligned
    with the last year of the series. The frequency of the input
    data should not lower than 'D' (the maximum time step should be 'D').
    The input series should cover at least two years of data.

    :param data: Input time series whose yearly profile has to be detected.
    :param exclude_days: Time series of days to exclude from the input time series.
    :return: Time series representing the yearly profile, e.g. yearly electricity consumption
            pattern of the input time series at daily frequency.
    """

    def to_timestamp(day, month, year, hour=0, minute=0, second=0, tz=pytz.utc):
        try:
            return datetime(
                day=int(day),
                month=int(month),
                year=int(year),
                hour=int(hour),
                minute=int(minute),
                second=int(second),
                tzinfo=tz)
        except ValueError:
            return None

    # Check input data before computing the profile

    if data.empty or (isinstance(data, pd.DataFrame) and data.shape[1] > 1):
        raise ValueError("Input series must be not empty and have exactly one column (if DataFrame),"
                         " i.e. shape = (n, 1).")

    elif detect_time_step(data)[0] is None:
        raise ValueError("Impossible to determine the frequency of the input time series.")

    elif data.index.year.nunique() < 2:
        raise ValueError("Input time series must cover at least two years to get a yearly profile.")

    try:
        frq_cmp = to_offset(detect_time_step(data)[0]) > to_offset('1D')
        if frq_cmp:
            raise ValueError("The frequency of the input time series must be not lower than '1D'.")
    except TypeError:
        raise ValueError("The frequency of the input time series must be not lower than '1D'.")

    # Filter out holidays or other days from the input data
    if exclude_days is not None:
        if isinstance(exclude_days, pd.Series):
            # Assuming exclude_days is a boolean series where True means filter out the day
            # Get the datetimeindex of True values
            exclude_days = exclude_days.index[exclude_days].normalize()
            data = data[~data.index.normalize().isin(exclude_days)]
        elif isinstance(exclude_days, list):
            # If it is a list convert to utc datetimeindex before filtering
            exclude_days = pd.to_datetime(exclude_days, utc=True)
            data = data[~data.index.normalize().isin(exclude_days)]
        else:
            raise TypeError("exclude_days argument must be a list of a boolean series of days to exclude.")

    # Group data first by month and then by day and aggregate by median
    df_group = data.groupby(by=[data.index.month, data.index.day]).median()

    # Set names of multiindex columns to identify month and day
    df_group.index.set_names(["month", "day"], inplace=True)

    # Add column year and set it constant to last year for alignment
    df_group.reset_index(inplace=True)

    # Get datetime object from columns year, month and day applying to timestamp row wise
    df_group["timestamp"] = df_group.apply(lambda x: to_timestamp(
        day=x["day"],
        month=x["month"],
        year=data.index.year[-1]),
                                           axis=1)

    # Drop rows with at least one NaT (number of days per month can be different over the years)
    df_group.dropna(axis=0, how='any', inplace=True)

    # Reindex with the new "synthetic" DatetimeIndex and drop columns day, month, year
    df_group.set_index("timestamp", drop=True, inplace=True)
    df_group.drop(["month", "day"], axis=1, inplace=True)

    return df_group


def weekly_profile_detection(
        data: pd.DataFrame, aggregation: str = 'median', exclude_days: Union[pd.Series, list] = None):
    """
    The function returns the weekly profile of the input time series.
    It aggregates values of the input data over multiple years using
    the median and return a time series at hourly resolution, aligned
    with the last week of the series. The frequency of the input
    data should not lower than 'H' (the maximum time step should be 'D').
    The input series should cover at least two weeks of data.

    :param data: Input time series whose weekly profile has to be detected.
    :param aggregation: Aggregation function to use for the profile.
    :param exclude_days: Time series of days to exclude from the input time series.
    :return: Time series representing the weekly profile, e.g. weekly electricity consumption
            pattern of the input time series at hourly frequency.
    """

    def to_timestamp(day, month, year, hour=0, minute=0, second=0, tz=pytz.utc):
        try:
            return datetime(
                day=int(day),
                month=int(month),
                year=int(year),
                hour=int(hour),
                minute=int(minute),
                second=int(second),
                tzinfo=tz)
        except ValueError:
            return None

    # Check input data before computing the profile

    if data.empty or (isinstance(data, pd.DataFrame) and data.shape[1] > 1):
        raise ValueError("Input series must be not empty and have exactly one column (if DataFrame),"
                         " i.e. shape = (n, 1).")

    elif detect_time_step(data)[0] is None:
        raise ValueError("Impossible to determine the frequency of the input time series.")

    elif data.index.isocalendar().week.nunique() < 2:
        raise ValueError("Input time series must cover at least two weeks to get a weekly profile.")

    try:
        frq_cmp = to_offset(detect_time_step(data)[0]) > to_offset('1H')
        if frq_cmp:
            raise ValueError("The frequency of the input time series must be not lower than '1H'.")
    except TypeError:
        raise ValueError("The frequency of the input time series must be not lower than '1H'.")

    # Filter out holidays or other days from the input data
    if exclude_days is not None:
        if isinstance(exclude_days, pd.Series):
            # Assuming exclude_days is a boolean series where True means filter out the day
            # Get the datetimeindex of True values
            exclude_days = exclude_days.index[exclude_days].normalize()
            mask = pd.Series(data.index.normalize().isin(exclude_days), index=data.index)
            data = data.mask(mask)
        elif isinstance(exclude_days, list):
            # If it is a list convert to utc datetimeindex before filtering
            exclude_days = pd.to_datetime(exclude_days, utc=True)
            mask = pd.Series(data.index.normalize().isin(exclude_days), index=data.index)
            data = data.mask(mask)
        else:
            raise TypeError("exclude_days argument must be a list or a boolean series of days to exclude.")

    # Group data first by month and then by day and aggregate by median
    df_group = data.groupby(by=[data.index.dayofweek, data.index.hour]).agg(aggregation)

    # Set names of multiindex columns to identify month and day
    df_group.index.set_names(["dayofweek", "hour"], inplace=True)

    # Get last Sunday in the original series and generate "synthetic" datetimeindex for alignment
    last_sunday = data.index[data.index.dayofweek == 6][-1].replace(hour=23, minute=0, second=0, tzinfo=pytz.utc)
    days = pd.date_range(end=last_sunday, freq='1H', periods=24 * 7)

    # Add columns year, month, day and set it constant to last year, month for alignment and last week days for day
    df_group["year"] = data.index.year[-1]
    df_group["month"] = data.index.month[-1]
    df_group["day"] = days.day

    df_group.reset_index(inplace=True)

    # Get datetime object from columns year, month and day applying to timestamp row wise
    df_group["timestamp"] = df_group.apply(lambda x: to_timestamp(
        hour=x["hour"],
        day=x["day"],
        month=x["month"],
        year=x["year"]),
                                           axis=1)

    # Drop rows with at least one NaT (number of days per month can be different over the years)
    df_group.dropna(axis=0, how='any', inplace=True)

    # Reindex with the new "synthetic" DatetimeIndex and drop columns day, month, year
    df_group.set_index("timestamp", drop=True, inplace=True)
    df_group.drop(["dayofweek", "hour", "year", "month", "day"], axis=1, inplace=True)

    return df_group


class HolidayTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer version of the function add_holiday_component
    to be used with sklearn Pipelines.
    """

    def __init__(self, country: str, prov: str = None, state: str = None, switch_on: bool = True):
        """
        Adds the holiday feature to the input DataFrame based on the country.
        The computation of the holidays is based on the package
        available in pypi: https://pypi.org/project/holidays/ .
        Check the documentation for more information about country and region codes.

        :param country: string identifying the country based on ISO 3166-1 alpha-2 code.
        :param prov: The Province (see documentation of what is supported; not
           implemented for all countries).
        :param state: The State (see documentation for what is supported; not
           implemented for all countries).
        :param switch_on: Can be used to enable or disable the transformation.
            Useful in the hyperparameter optimization. If False, the input data
            will be passed through.
        :return: new DataFrame with the added holiday component.
        """

        self.country = country
        self.switch_on = switch_on
        self.prov = prov
        self.state = state

    def fit(self, X, y=None):
        return self

    def transform(self, X) -> pd.DataFrame:
        """

        :param X:  input DataFrame with a DateTimeIndex.

        :return:
        """

        # Retrieve the country holidays in the time range of the input data

        if self.switch_on is True:
            country_holidays = holidays.country_holidays(
                country=self.country,
                prov=self.prov,
                state=self.state)[X.index.min():X.index.max() + timedelta(days=1)]
            col_holidays = pd.DatetimeIndex(X.index.date).isin(country_holidays).astype(int)
            return X.assign(holiday=col_holidays)
        else:
            return X


class ScalerTransformer(BaseEstimator, TransformerMixin):
    """
    Wrapper around some preprocessing functions in sklearn.
    Generally, sklearn transformers return numpy matrices,
    which means that only the values of the DataFrame are
    kept while the header and the index are stripped off.
    The index in time series is an important feature we
    do not want to lose in the transformation process.
    """

    def __init__(self, scaler: TransformerMixin, switch_on: bool = True):
        """
        Adds calendar components quarter, month, week, day, hour
        to the input DataFrame.

        :param scaler: a sklearn scaler, such as StandardScaler or MinMaxScaler.
        :param switch_on: Can be used to enable or disable the transformation.
            Useful in the hyperparameter optimization. If False, the input data
            will be passed through.
        :return: new DataFrame with the added calendar components.
        """
        self.scaler = scaler
        self.switch_on = switch_on

    def fit(self, X, y=None):
        return self

    def transform(self, X) -> pd.DataFrame:
        """
        Applies the transformation done by the input scaler
        to the X date while keeping the DataFrame format.
        It assumes that the input scaler will not change
        the column order of X.

        :param X:  input DataFrame with a DateTimeIndex.
        :return: new Dataframe with the scaled features.
        """

        if self.switch_on is True:
            return pd.DataFrame(
                data=self.scaler.fit_transform(X),
                index=X.index,
                columns=X.columns)
        else:
            return X


class CalendarComponentTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer version of the functions add_calendar_components
    and trigonometric_encode_calendar_components, to be used with
    sklearn Pipelines.
    """

    def __init__(self, components: list = None, encode: bool = False, switch_on: bool = True):
        """
        Adds calendar components quarter, month, week, day, hour
        to the input DataFrame.

        :param components: List of strings, specifying the calendar components you want to add in
            ["season", "quarter", "month", "week", "weekday", "hour", "day", "dayofyear"].
        :param encode: If True, encodes the calendar features into cyclic sin/cosine components.
            For each calendar components, the transformer will generate one sin and one cosine
            component.
        :param switch_on: Can be used to enable or disable the transformation.
            Useful in the hyperparameter optimization. If False, the input data
            will be passed through.
        """

        self.component_period = {
            "season": 4,
            "quarter": 4,
            "month": 12,
            "week": 53,
            "weekday": 7,
            "hour": 24,
            "day": 31,
            "dayofyear": 365
        }
        if components is None:
            self.components = self.component_period.keys()
        elif set(components).issubset(self.component_period.keys()):
            self.components = components
        else:
            raise ValueError("Argument 'calendar_components' must be a subset of: {}".format(
                self.component_period.keys()))

        self.encode = encode
        self.switch_on = switch_on

    def fit(self, X, y=None):
        return self

    def transform(self, X) -> pd.DataFrame:
        """
        :param X:  input DataFrame with a DateTimeIndex.
        :return: new DataFrame with the added calendar components.
        """

        if self.switch_on is True:
            if self.encode is True:
                # Here we generate a sin and a cosine component for each calendar feature

                encoded_components = {
                    k: v for component in self.components for k, v in
                    (
                        ('{}_sin'.format(component), np.sin(get_calendar_component(X, component) /
                                                            self.component_period[component] * 2 * np.pi)),
                        ('{}_cos'.format(component), np.cos(get_calendar_component(X, component) /
                                                            self.component_period[component] * 2 * np.pi))
                    )
                }
                return X.assign(**encoded_components)

            else:
                return X.assign(**{
                    '{}'.format(component): get_calendar_component(X, component)
                    for component in self.components
                })
        else:
            return X


class WeeklyProfileTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer version of the function add_weekly_profile, to be used with
    sklearn Pipelines.
    """

    def __init__(self, aggregation: str = "median", switch_on: bool = True):
        """
        Adds calendar components quarter, month, week, day, hour
        to the input DataFrame.

        :param aggregation: aggregation function to use for the profile
        :param switch_on: Can be used to enable or disable the transformation.
            Useful in the hyperparameter optimization. If False, the input data
            will be passed through.
        """

        self.aggregation = aggregation
        self.feature_name = ""
        self.switch_on = switch_on
        self.profile_ = None

    def fit(self, X, y=None):

        check_X_y(X, y, ensure_2d=False)

        # Check input data before computing profile

        if isinstance(y, pd.Series):
            y = y.to_frame()

        if not isinstance(X.index, pd.DatetimeIndex) or not isinstance(y.index, pd.DatetimeIndex):
            raise ValueError("Input must be a pandas DataFrame or Series with a DateTimeIndex.")

        elif y.index.isocalendar().week.nunique() < 2:
            raise ValueError("Input time series must cover at least two weeks to get a weekly profile.")

        # Create weekly profile and align it with the input dataframe
        self.profile_ = y.groupby([y.index.dayofweek, y.index.hour]).agg(self.aggregation)
        self.feature_name = "{}_weekly_profile".format(self.profile_.columns[0])
        self.profile_.rename(columns={self.profile_.columns[0]: self.feature_name}, inplace=True)
        self.profile_ = self.profile_.assign(dayofweek=self.profile_.index.get_level_values(0),
                                             hour=self.profile_.index.get_level_values(1))
        return self

    def transform(self, X) -> pd.DataFrame:
        """
        :param X:  input DataFrame with a DateTimeIndex.
        :return: new DataFrame with the added weekly profile.
        """

        check_is_fitted(self)
        if self.switch_on is True:
            if isinstance(X, pd.Series):
                X = X.to_frame()
            X_temp = X.assign(dayofweek=X.index.dayofweek, hour=X.index.hour)
            merged_df = X_temp.reset_index().merge(self.profile_, on=["dayofweek", "hour"], how="left")
            X_temp[self.feature_name] = merged_df[self.feature_name].values
            X_temp.drop(columns=["dayofweek", "hour"], inplace=True)
            return X_temp

        else:
            return X


def add_lag_components(data: pd.DataFrame, columns: list = None, max_lag: int = 1) -> pd.DataFrame:
    """
    Returns a DataFrame with the lag components of the columns as new columns.
    If the argument 'columns' is not None, only the lag components of the
    specified columns will be generated.

    :param data: DataFrame with at least one column.
    :param columns: List of strings specifying the column to use.
        Must be a subset of the input data columns.
    :param max_lag: Maximum lag to be generated.
    :return: New DataFrame with the lag components included as new columns
    """

    if data.empty or not isinstance(data, pd.DataFrame):
        raise ValueError("Input data must be not empty and must be a pandas DataFrame.")

    if columns is not None:
        if not set(columns).issubset(data.columns.to_list()):
            raise ValueError("Argument 'columns' must be a subset of the input DataFrame columns.")
        return data.assign(**{
            '{}(-{})'.format(col, lag): data[col].shift(lag)
            for lag in range(1, max_lag + 1)
            for col in columns
        })

    return data.assign(**{
        '{}(-{})'.format(col, lag): data[col].shift(lag)
        for lag in range(1, max_lag + 1)
        for col in data
    })


def add_calendar_components(data: pd.DataFrame,
                            calendar_components: list = None,
                            drop_constant_columns: bool = True) -> pd.DataFrame:
    """
    Add calendar components year, quarter, month, week, day, hour
    to the input DataFrame.

    :param data: input DataFrame with a DateTimeIndex and at least one column.
    :param calendar_components: List of strings, specifying the calendar components you want to add in
        ["season", "quarter", "month", "week", "weekday", "hour", "day", "dayofyear"].
    :param drop_constant_columns: If True, drops constant calendar components.
    :return: new DataFrame with the added calendar components.
    """

    if data.empty or not isinstance(data, pd.DataFrame) or not isinstance(data.index, pd.DatetimeIndex):
        raise ValueError("Input must be a non-empty pandas DataFrame with a DateTimeIndex.")

    default_components = ["season", "quarter", "month", "week", "weekday", "hour", "day", "dayofyear"]

    if calendar_components is not None:
        if not set(calendar_components).issubset(default_components):
            raise ValueError("Argument 'calendar_components' must be a subset of: {}".format(default_components))
    else:
        calendar_components = default_components

    df_new = data.assign(**{
        '{}'.format(component): get_calendar_component(data, component)
        for component in calendar_components
    })

    if drop_constant_columns:
        varying_columns = [e for e in df_new.columns if df_new[e].nunique() != 1]
        return df_new.loc[:, varying_columns]

    return df_new


def trigonometric_encode_calendar_components(data, calendar_components=None, remainder='passthrough', drop=True):
    """
    This function returns a sklearn transformer to encode all the calendar components
    added to a Dataframe into trigonometric cyclic components, sin and cosine. This type
    of encoding is beneficial for some models like LinearRegression, PolynomialRegression,
    SVM, etc. but generally not for DecisionTree or ensemble methods like Random Forest.

    :param data: DataFrame containing the calendar components added with
        the function add_calendar_components.
    :param calendar_components: Calendar components to be transformed to cyclic sin and
        cosine components. Default is None, which means to transform all the components.
    :param remainder: {‘drop’, ‘passthrough’} or estimator, default=‘passthrough’.
        By default, only the specified calendar components are transformed and combined
        in the output, and the non-specified columns are passed through.
        This subset of columns is concatenated with the output of the transformers.
        By setting remainder to be an estimator, for example a scaling transformer like
        StandardScaler, the remaining non-specified columns will use the remainder estimator.
    :param drop: Whether to drop or not the original calendar components.
    :return: Transformer to be used in a sklearn Pipeline to perform the encoding.
    """

    def sin_transformer(period):
        return FunctionTransformer(lambda x: np.sin(x / period * 2 * np.pi))

    def cos_transformer(period):
        return FunctionTransformer(lambda x: np.cos(x / period * 2 * np.pi))

    if data.empty or not isinstance(data, pd.DataFrame) or not isinstance(data.index, pd.DatetimeIndex):
        raise ValueError("Input must be a non-empty pandas DataFrame with a DateTimeIndex.")

    component_period = {
        "season": 4,
        "quarter": 4,
        "month": 12,
        "week": 53,
        "weekday": 7,
        "hour": 24,
        "day": 31,
        "dayofyear": 365
    }

    if calendar_components is not None:
        if not set(calendar_components).issubset(component_period.keys()):
            raise ValueError("Argument 'calendar_components' must be a subset of: {}".format(component_period.keys()))
    else:
        calendar_components = component_period.keys()

    # Create list of transformers for each column
    transformers = list(
        chain.from_iterable(
            (
                ("{}_sin".format(component), sin_transformer(component_period[component]), [component]),
                ("{}_cos".format(component), cos_transformer(component_period[component]), [component])
            )
            for component in calendar_components)
    )
    # Append to the list a transformer to drop the original components after the transformation
    if drop:
        transformers.append(('drop', 'drop', calendar_components))

    return ColumnTransformer(
        transformers=transformers,
        remainder=remainder)


def add_holiday_component(data: pd.DataFrame, country: str, prov: str = None, state: str = None) -> pd.DataFrame:
    """
    Adds the holiday feature to the input DataFrame based on the country.
    The computation of the holidays is based on the package
    available in pypi: https://pypi.org/project/holidays/ .
    Check the documentation for more information about country and region codes.

    :param data: input DataFrame with a DateTimeIndex and at least one column.
    :param country: string identifying the country based on ISO 3166-1 alpha-2 code.
    :param prov: The Province (see documentation of what is supported; not
       implemented for all countries).
    :param state: The State (see documentation for what is supported; not
       implemented for all countries).
    :return: new DataFrame with the added holiday component.
    """

    if data.empty or not isinstance(data, pd.DataFrame) or not isinstance(data.index, pd.DatetimeIndex):
        raise ValueError("Input must be a non-empty pandas DataFrame with a DateTimeIndex.")

    country_holidays = holidays.country_holidays(
        country=country,
        prov=prov,
        state=state)[data.index.min():data.index.max() + timedelta(days=1)]

    return data.assign(holiday=pd.DatetimeIndex(data.index.date).isin(country_holidays).astype(int))


def get_calendar_component(data: pd.DataFrame, component: str) -> pd.Series:
    """
    Method to avoid the Deprecation Warning when getting some calendar
    components like week and to compute season component.

    :param data: Dataframe with a DatetimeIndex
    :param component: calendar component
    :return: calendar component
    """

    if component in ['week', 'weekofyear']:
        return getattr(data.index.isocalendar(), component)
    elif component == 'season':
        return data.index.month % 12 // 3 + 1
    else:
        return getattr(data.index, component)


def add_weekly_profile(data: pd.DataFrame, target: str, aggregation: str = "median") -> pd.DataFrame:
    """
    Detects the weekly profile of the target feature, aligning it with the
    the other data and repeating it for the entire time range.
    :param data: input Dataframe
    :param target: target feature for the weekly profile
    :param aggregation: aggregation function to use for the profile
    :return: DataFrame with the weekly profile included
    """

    # Check input data before computing profile
    if target not in data.columns:
        raise KeyError("Feature '{}' not found in columns.".format(target))
    elif data.empty or not isinstance(data, pd.DataFrame) or not isinstance(data.index, pd.DatetimeIndex):
        raise ValueError("Input must be a non-empty pandas DataFrame with a DateTimeIndex.")

    elif detect_time_step(data[[target]])[0] is None:
        raise ValueError("Impossible to determine the frequency of the input time series.")

    elif data.index.isocalendar().week.nunique() < 2:
        raise ValueError("Input time series must cover at least two weeks to get a weekly profile.")

    try:
        frq_cmp = to_offset(detect_time_step(data[[target]])[0]) > to_offset('1H')
        if frq_cmp:
            raise ValueError("The frequency of the input time series must be not lower than '1H'.")
    except TypeError:
        raise ValueError("The frequency of the input time series must be not lower than '1H'.")

    # Create weekly profile and align it with the input dataframe
    feature_name = "{}_weekly_profile".format(target)
    df_weekly = data[[target]].groupby([data.index.dayofweek, data.index.hour]).agg(aggregation)
    df_weekly = df_weekly.assign(dayofweek=df_weekly.index.get_level_values(0),
                                 hour=df_weekly.index.get_level_values(1))
    df_weekly.rename(columns={target: feature_name}, inplace=True)
    data_temp = data.assign(dayofweek=data.index.dayofweek, hour=data.index.hour)
    merged_df = data_temp.reset_index().merge(df_weekly, on=["dayofweek", "hour"], how="left")
    data_temp[feature_name] = merged_df[feature_name].values
    data_temp.drop(columns=["dayofweek", "hour"], inplace=True)
    return data_temp


if __name__ == '__main__':
    """
    This module is not supposed to run as a stand-alone module.
    This part below is only for testing purposes. 
    """

