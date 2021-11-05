# !/usr/bin/python3
# -*- coding: utf-8 -*-

import unittest
from unittest.mock import patch
import pandas as pd
import numpy as np
from ai_toolbox import data_preparation


class DataPreparation(unittest.TestCase):
    """Class to test the data_preparation module"""

    @classmethod
    def setUpClass(cls):
        """ Set up dfs and series for the entire test case """

        from os.path import dirname, join, realpath
        import pytz
        
        super().setUpClass()

        # Load irregular time series (dataframe) from csv
        dir_path = dirname(realpath(__file__))
        filename = join(dir_path, "fixtures", "irregular_timeseries.csv")
        cls.df_irregular = pd.read_csv(
            filename,
            sep=';',
            parse_dates=True,
            infer_datetime_format=True,
            index_col=0)

        # Generate regular time series (dataframe)
        idx = pd.date_range(start='2021/10/01', end='2021/11/01', tz=pytz.utc, freq='10T')
        cls.df_regular = pd.DataFrame(data=np.random.randint(0, 100, size=(len(idx))), index=idx, columns=["rand_int"])

        # Generate regular time series (pandas series)
        cls.series_regular = pd.Series(data=np.random.randint(0, 100, size=(len(idx))), index=idx)

        # Generate short series of randint
        idx = pd.to_datetime(["2021-11-05 11:48:00", "2021-11-05 11:33:00"])
        cls. df_short = pd.DataFrame(data=np.random.randint(0, 100, size=(len(idx))), index=idx, columns=["rand_int"])

        # Generate short series of randint with two equal frequencies
        idx = pd.to_datetime(["2021-11-05 11:01:00", "2021-11-05 11:33:00", "2021-11-05 11:48:00",
                              "2021-11-05 12:03:00", "2021-11-05 12:35:00", "2021-11-05 14:03:00"])

        cls.df_two_freq = pd.DataFrame(data=np.random.randint(0, 100, size=(len(idx))), index=idx, columns=["rand_int"])

    def test_detect_time_step_for_irregular_time_series(self):
        """ Test that detect_time_step works with an irregular time series """

        self.df_irregular = self.df_irregular.resample('30T').mean()
        self.assertEqual(data_preparation.detect_time_step(self.df_irregular)[0], '30T')

    def test_detect_time_step_regular_time_series_with_df(self):
        """ Test that detect_time_step works with a regular time series represented by DataFrame """

        self.assertEqual(data_preparation.detect_time_step(self.df_regular)[0], '10T')

    def test_detect_time_step_of_regular_time_series_with_series(self):
        """ Test that detect_time_step works with a time series represented by Series """

        self.assertEqual(data_preparation.detect_time_step(self.series_regular)[0], '10T')

    def test_detect_time_step_unsorted_series(self):
        """ Test that detect_time_step still works with an unsorted time series """

        self.assertEqual(data_preparation.detect_time_step(self.df_regular.sample(frac=1))[0], '10T')

    def test_detect_time_step_two_values(self):
        """ Test that detect_time_step still works with a time series of just two values """

        self.assertEqual(data_preparation.detect_time_step(self.df_short)[0], '15T')

    def test_detect_time_step_two_frequencies(self):
        """
        Test that detect_time_step returns the minimum frequency if there is
         more than one occurrence of the most frequent time delta.
        """

        self.assertEqual(data_preparation.detect_time_step(self.df_two_freq)[0], '15T')

    def test_detect_time_step_raises_if_empty_series(self):
        """ Test that detect_time_step raises ValueError in case of empty time series """

        self.assertRaises(ValueError, data_preparation.detect_time_step, pd.DataFrame(data=[]))

    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()


if __name__ == '__main__':
    unittest.main()
