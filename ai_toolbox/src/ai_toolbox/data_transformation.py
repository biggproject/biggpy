#!/usr/bin/python3
# -*- coding: utf-8 -*-

from datetime import datetime

import pandas as pd
import pytz
from ai_toolbox.data_preparation import detect_time_step
from pandas.tseries.frequencies import to_offset


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

    def to_timestamp(day, month, year):
        try:
            return datetime(day=int(day), month=int(month), year=int(year), hour=0, minute=0, second=0, tzinfo=pytz.utc)
        except ValueError as e:
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
            exclude_days = exclude_days.index[exclude_days]
            data = data[~data.index.normalize().isin(exclude_days)]
        elif isinstance(exclude_days, list):
            # If it is a list convert to utc datetimeindex before filtering
            exclude_days = pd.to_datetime(exclude_days, utc=True)
            data = data[~data.index.normalize().isin(exclude_days)]
        else:
            raise TypeError("exclude_days argument must be a list of a boolean series of days to exclude.")

    # Group data first by month and then by day and aggregate by median
    df_group = data.groupby(by=[data.index.month, data.index.day]).median()

    # Set names of multiindex columns to indentify month and day
    df_group.index.set_names(["month", "day"], inplace=True)

    # Add column year and set it constant to last year for alignment
    df_group["year"] = data.index.year[-1]
    df_group.reset_index(inplace=True)

    # Get datetime object from columns year, month and day applying to timestamp row wise
    df_group["timestamp"] = df_group.apply(lambda x: to_timestamp(
            day=x["day"],
            month=x["month"],
            year=data.index.year[-1]),
        axis=1)

    # Drop rows with at least one NaT (number of days per month can be different over the years)
    df_group.dropna(axis=0, how='any', inplace=True)

    # Reindex with the new "syntethic" DatetimeIndex and drop columns day, month, year
    df_group.set_index("timestamp", drop=True, inplace=True)
    df_group.drop(["month", "day", "year"], axis=1, inplace=True)

    return df_group


if __name__ == '__main__':
    """
    This module is not supposed to run as a stand-alone module.
    This part below is only for testing purposes. 
    """