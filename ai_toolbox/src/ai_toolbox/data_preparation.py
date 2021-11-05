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

    # Clean-up the data by dropping NaN values (irregular time series)
    # and sorting the index.
    # Otherwise the function will be misled when guessing the frequency
    # by the datetimeindex
    data.dropna(inplace=True)
    data.sort_index(inplace=True)

    # Check if the series frequency is set and return it
    try:
        best_frequency = data.index.freq.freqstr
    except AttributeError:
        # If time series frequency attribute is not set, try to infer it
        # or guess the "best" frequency

        try:
            # First case: "easiest" case, uniform time series
            # (The try block is because infer_freq raises ValueError if there
            # are fewer than 3 datapoints in the series)
            best_frequency = pd.infer_freq(data.index, warn=True)
        except ValueError:
            pass
        # Second case: "worst" case, non-uniform time series
        # the "best" frequency is the most frequent time delta between
        # two consecutive samples
        if best_frequency is None:
            frequencies = pd.DataFrame(data=data.index.to_series().diff().value_counts())
            frequencies.rename(columns={frequencies.columns[0]: "freq_count"}, inplace=True)
            frequencies["freqstr"] = frequencies.apply(lambda x: to_offset(x.name).freqstr, axis=1)
            best_frequency = frequencies[
                frequencies["freq_count"] == frequencies.freq_count.max()].sort_index().freqstr[0]

    finally:
        return best_frequency, frequencies


if __name__ == '__main__':
    """
    This module is not supposed to run as a stand-alone module.
    This part below is only for testing purposes. 
    """

