#!/usr/bin/python3
# -*- coding: utf-8 -*-

from datetime import datetime

import pandas as pd
import pytz
from pandas.tseries.frequencies import to_offset

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


def weekly_profile_detection(data, exclude_days=None):
    """
    The function returns the weekly profile of the input time series.
    It aggregates values of the input data over multiple years using
    the median and return a time series at hourly resolution, aligned
    with the last week of the series. The frequency of the input
    data should not lower than 'H' (the maximum time step should be 'D').
    The input series should cover at least two weeks of data.

    :param data: Input time series whose weekly profile has to be detected.
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
            raise TypeError("exclude_days argument must be a list of a boolean series of days to exclude.")

    # Group data first by month and then by day and aggregate by median
    df_group = data.groupby(by=[data.index.dayofweek, data.index.hour]).median()

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


def add_lag_components(data, columns=None, max_lag=1):
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


def add_calendar_components(data, calendar_components=None, drop_constant_columns=True):
    """
    Add calendar components year, quarter, month, week, day, hour
    to the input DataFrame.

    :param data: input DataFrame with a DateTimeIndex and at least one column.
    :param calendar_components: List of strings, specifying the calendar components you want to add in
        ["year", "quarter", "month", "week", "day", "hour"].
    :param drop_constant_columns: If True, drops constant calendar components.
    :return: new DataFrame with the added calendar components.
    """

    if data.empty or not isinstance(data, pd.DataFrame) or not isinstance(data.index, pd.DatetimeIndex):
        raise ValueError("Input must be a non-empty pandas DataFrame with a DateTimeIndex.")

    default_components = ["year", "quarter", "month", "week", "day", "hour"]

    if calendar_components is not None:
        if not set(calendar_components).issubset(default_components):
            raise ValueError("Argument 'calendar_components' must be a subset of "
                             "['year', 'quarter', 'month', 'week', 'day', 'hour'].")
    else:
        calendar_components = default_components

    df_new = data.assign(**{
        '{}'.format(component): getattr(data.index, component)
        for component in calendar_components
    })

    if drop_constant_columns:
        df_new = df_new.loc[:, (df_new[calendar_components] != df_new.iloc[0]).any()]

    return df_new


if __name__ == '__main__':
    """
    This module is not supposed to run as a stand-alone module.
    This part below is only for testing purposes. 
    """
    from os.path import dirname, join, realpath, pardir
    import matplotlib.pyplot as plt

    # Load time series for profile (dataframe) from csv
    dir_path = dirname(realpath(__file__))
    filename = join(dir_path, pardir, pardir, "tests", "fixtures", "df_weekly_profile.csv")
    profile_data = pd.read_csv(
        filename,
        sep=',',
        parse_dates=True,
        infer_datetime_format=True,
        index_col=0)

    profile = weekly_profile_detection(profile_data)
    profile.plot(ylim=(15, 30), figsize=(10, 10)).grid(which='major', axis='both', linestyle='--')
    plt.show()
