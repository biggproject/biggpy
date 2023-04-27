#!/usr/bin/python3
# -*- coding: utf-8 -*-

import pandas as pd
from pandas.tseries.frequencies import to_offset


def detect_time_step(data):
    """
    The function infers, i.e. automatically deduce from the input data,
     the minimum time step (frequency) that can be used for
    the input time series.
    For regular (uniform) time series, the function will return directly
     the frequency detected and None.
    For irregular (non-uniform) time series, the function will still try to
    guess the best frequency to use, analysing the time deltas between consecutive
    observations (samples). The most frequent time delta in the series is the
    best frequency.
    In this case, the function will return, together with the
    best frequency, also a DataFrame with all the time steps detected and their
    count in descending order.
    If there is more than one "best frequency", the function returns the minimum.

    :param data: Input DataFrame of shape equal to (n, 1) or Series whose time step has to be detected.
    :return: Tuple with the "best" frequency and a DataFrame specifying the count of all the
    time deltas detected in the series, in descending order.
    """

    if data.empty or (isinstance(data, pd.DataFrame) and data.shape[1] > 1):
        raise ValueError("Input series must be not empty and have exactly one column (if DataFrame),"
                         " i.e. shape = (n, 1).")

    frequencies = None
    best_frequency = None

    # Clean-up the data by dropping NaN values and sorting the index (irregular time series) .
    # Otherwise, the function will be misled when guessing the frequency by the DateTimeIndex
    df_clean = data.dropna().sort_index()

    # Check if the series frequency is set and return it
    try:
        best_frequency = df_clean.index.freq.freqstr
    except AttributeError:
        # If time series frequency attribute is not set, try to infer it
        # or guess the "best" frequency

        try:
            # First case: "easiest" case, uniform time series
            # (The try block is because infer_freq raises ValueError if there
            # are fewer than 3 datapoints in the series)
            best_frequency = pd.infer_freq(df_clean.index)
        except ValueError:
            pass
        # Second case: "worst" case, non-uniform time series
        # the "best" frequency is the most frequent time delta between
        # two consecutive samples
        if best_frequency is None:
            frequencies = pd.DataFrame(data={"freq_count": df_clean.index.to_series().diff().value_counts()})
            frequencies.index.name = "timedelta"
            frequencies["freqstr"] = frequencies.apply(lambda x: to_offset(x.name).freqstr, axis=1)
            best_frequency = frequencies.sort_values(by=["freq_count", "timedelta"], ascending=[False, True]).iloc[0, 1]

    finally:
        return best_frequency, frequencies


def align_time_grid(data, output_frequency, aggregation_function, closed=None):
    """
    The function aligns the frequency of the input time series with the output
    frequency given as an argument using the specified aggregation function.
    If the measurement_reading_type of the series is not instantaneous, the data
    must be converted first using the function clean_ts_integrate.
    This is a wrapper around the function 'resample' of pandas.

    :param data:  The time series that has to be aligned with an output time step, i.e. with a specific frequency.
    :param output_frequency: The frequency used to resample the input time series for the alignment.
            It must be a string in ISO 8601 format representing the time step (e.g. "15T","1H", "M", "D",...).
    :param aggregation_function: The aggregation function to use when resampling the series. Possible values are
            mean, sum, min, max, median.
    :param closed: {‘right’, ‘left’}, default None. Which side of bin interval is closed. The default is ‘left’
            for all frequency offsets except for ‘M’, ‘A’, ‘Q’, ‘BM’, ‘BA’, ‘BQ’, and ‘W’ which all have
            a default of ‘right’.
    :return: The time series resampled with the specified period and aggregation function.
    """

    if data.empty:
        raise ValueError("Input series must be not empty.")
    elif aggregation_function not in ['mean', 'median', 'max', 'min', 'sum']:
        raise ValueError("Aggregation function must be in ['mean', 'median', 'max', 'min', 'sum'].")

    resampler = data.resample(rule=output_frequency, closed=closed)
    return getattr(resampler, aggregation_function)()


def clean_ts_integrate(data, measurement_reading_type):
    """
    The function converts a cumulative (counter) or onChange (delta) measurement to instantaneous.
    If 'cumulative' the function will interpret each observation as a counter value.
    It will replace each value with its difference from the previous value if the result is not negative,
    otherwise will keep the same value (counter rollover).
    If 'delta' the function will interpret each observation as the difference between the current value
    and the previous value and return the cumulative sum of the values.

    :param data: The cumulative or on-change time series that has to be converted to instantaneous.
            An instantaneous measurement is a gauge metric, in which the value can increase or decrease
            and measures a specific instant in time.
    :param measurement_reading_type: Defines the type of the measurement of the input data.
            Possible values:
                - 'on_change' or 'delta': representing a delta metric, in which the value measures the change since it
                    was last recorded.
                - 'cumulative' or 'counter': representing a cumulative metric, in which the value can only increase over
                    time or be reset to zero on restart.
    :return: The cumulative or onChange time series with the measurements converted to the instantaneous type.

    """

    if data.empty or (isinstance(data, pd.DataFrame) and data.shape[1] > 1):
        raise ValueError("Input series must be not empty and have exactly one column (if DataFrame),"
                         " i.e. shape = (n, 1).")

    if measurement_reading_type in ["cumulative", "counter"]:
        # Sort the time series and compute the difference with the previous values
        df_clean = data.sort_index().diff().fillna(data)

        # Keep the same value from data if the difference is negative (counter rollover)
        df_clean.where(df_clean.gt(0), data, inplace=True)

    elif measurement_reading_type in ["delta", "on_change"]:
        # Compute the cumulative sum for delta metrics
        df_clean = data.sort_index().cumsum()

    else:
        raise ValueError("Measurement reading type must be in ['on_change', 'delta', 'cumulative', 'counter'].")

    return df_clean


if __name__ == '__main__':
    """
    This module is not supposed to run as a stand-alone module.
    This part below is only for testing purposes. 
    """
