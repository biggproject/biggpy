# !/usr/bin/python3
# -*- coding: utf-8 -*-

import unittest

import numpy as np
import pandas as pd
from ai_toolbox import data_preparation
from pandas.testing import assert_frame_equal, assert_series_equal


class TestDataPreparation(unittest.TestCase):
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
            index_col=0)

        # Generate regular time series (dataframe)
        idx = pd.date_range(start='2021/10/01', end='2021/11/01', tz=pytz.utc, freq='10min')
        cls.df_regular = pd.DataFrame(data=np.random.randint(0, 100, size=(len(idx))), index=idx, columns=["rand_int"])

        # Generate regular time series (pandas series)
        cls.series_regular = pd.Series(data=np.random.randint(0, 100, size=(len(idx))), index=idx)

        # Generate short series of randint
        idx = pd.to_datetime(["2021-11-05 11:48:00", "2021-11-05 11:33:00"])
        cls.df_short = pd.DataFrame(data=np.random.randint(0, 100, size=(len(idx))), index=idx, columns=["rand_int"])

        # Generate short series of randint with two equal frequencies
        idx = pd.to_datetime(["2021-11-05 11:01:00", "2021-11-05 11:33:00", "2021-11-05 11:48:00",
                              "2021-11-05 12:03:00", "2021-11-05 12:35:00", "2021-11-05 14:03:00"])

        cls.df_two_freq = pd.DataFrame(data=np.random.randint(0, 100, size=(len(idx))), index=idx, columns=["rand_int"])

        # Generate regular time series (dataframe) with 2 columns
        idx = pd.date_range(start='2021/10/01', end='2021/11/01', tz=pytz.utc, freq='10min')
        cls.df_two_columns = pd.DataFrame(
            data=np.random.randint(0, 100, size=(len(idx), 2)),
            index=idx,
            columns=["n1", "n2"]
        )

        counter = [30, 1, 20, 28, 44, 0, 2, 11, 56, 0, 23, 89, 10, 32, 45, 19]
        cls.df_counter = pd.Series(data=counter, index=pd.date_range('12/11/2021', freq='15min', periods=len(counter)))

        delta = [30.0, -29.0, 19.0, 8.0, 16.0, -44.0, 2.0, 9.0, 45.0, -56.0, 23.0, 66.0, -79.0, 22.0, 13.0, -26.0]
        cls.df_delta = pd.Series(data=delta, index=pd.date_range('12/11/2021', freq='15min', periods=len(delta)))

    # detect_time_step tests below

    def test_detect_time_step_for_irregular_time_series(self):
        """ Test that detect_time_step works with an irregular time series """

        self.df_irregular = self.df_irregular.resample('30min').mean()
        self.assertIn(data_preparation.detect_time_step(self.df_irregular)[0], ['30T', '30min'])

    def test_detect_time_step_regular_time_series_with_df(self):
        """ Test that detect_time_step works with a regular time series represented by DataFrame """

        self.assertIn(data_preparation.detect_time_step(self.df_regular)[0], ['10T', '10min'])

    def test_detect_time_step_of_regular_time_series_with_series(self):
        """ Test that detect_time_step works with a time series represented by Series """

        self.assertIn(data_preparation.detect_time_step(self.series_regular)[0], ['10T', '10min'])

    def test_detect_time_step_unsorted_series(self):
        """ Test that detect_time_step still works with an unsorted time series """

        self.assertIn(data_preparation.detect_time_step(self.df_regular.sample(frac=1))[0], ['10T', '10min'])

    def test_detect_time_step_two_values(self):
        """ Test that detect_time_step still works with a time series of just two values """

        self.assertIn(data_preparation.detect_time_step(self.df_short)[0], ['15T', '15min'])

    def test_detect_time_step_two_frequencies(self):
        """
        Test that detect_time_step returns the minimum frequency if there is
         more than one occurrence of the most frequent time delta
        """

        self.assertIn(data_preparation.detect_time_step(self.df_two_freq)[0], ['15T', '15min'])

    def test_detect_time_step_raises_if_empty_series(self):
        """ Test that detect_time_step raises ValueError in case of empty time series """

        self.assertRaises(ValueError, data_preparation.detect_time_step, pd.DataFrame(data=[]))

    def test_detect_time_step_raises_if_more_columns(self):
        """
        Test that detect_time_step raises ValueError in case of DataFrame with
        more than one column
        """

        self.assertRaises(ValueError, data_preparation.detect_time_step, self.df_two_columns)

    # align_time_grid tests below

    def test_align_time_grid_raises_if_empty_series(self):
        """ Test that align_time_grid raises ValueError in case of empty time series """

        self.assertRaises(ValueError, data_preparation.align_time_grid, pd.DataFrame(data=[]), "30T", "mean")

    def test_align_time_grid_raises_if_wrong_aggr_function(self):
        """ Test that align_time_grid raises ValueError in case of empty time series """

        self.assertRaises(ValueError, data_preparation.align_time_grid, self.df_irregular, "30T", "meh")

    @staticmethod
    def test_align_time_grid_ok():
        """ Test that test_align_time_grid returns expected result for all the aggregations """

        idx = pd.date_range(start='2021/10/01', periods=9, freq='min')
        df = pd.DataFrame(data=range(len(idx)), index=idx, columns=['numbers'])
        result_list = [[1, 4, 7], [1, 4, 7], [3, 12, 21], [0, 3, 6], [2, 5, 8]]
        for i, aggr_function in enumerate(['mean', 'median', 'sum', 'min', 'max']):
            df_aligned = data_preparation.align_time_grid(
                data=df, output_frequency='3min', aggregation_function=aggr_function)
            expected_result = pd.DataFrame(
                data=result_list[i], index=pd.date_range(start='2021/10/01', periods=3, freq='3min'), columns=['numbers'])
            assert_frame_equal(df_aligned, expected_result, check_exact=False, check_dtype=False)

    # clean_ts_integrate tests below

    def test_clean_ts_integrate_raises_if_wrong_type(self):
        """ Test that clean_ts_integrate raises returns if incorrect measurement type """

        self.assertRaises(ValueError, data_preparation.clean_ts_integrate, self.series_regular, "del")

    def test_clean_ts_integrate_raises_if_empty_series(self):
        """ Test that clean_ts_integrate raises ValueError in case of empty time series """

        self.assertRaises(ValueError, data_preparation.clean_ts_integrate, pd.DataFrame(data=[]), "delta")

    def test_clean_ts_integrate_raises_if_more_columns(self):
        """
        Test that clean_ts_integrate raises ValueError in case of DataFrame with
        more than one column
        """

        self.assertRaises(ValueError, data_preparation.clean_ts_integrate, self.df_two_columns, "delta")

    def test_clean_ts_integrate_ok_cumulative(self):
        """ Test that clean_ts_integrate returns expected result for cumulative metrics """

        series_converted = data_preparation.clean_ts_integrate(self.df_counter, "counter")
        expected_result = [30.0, 1.0, 19.0, 8.0, 16.0, 0.0, 2.0, 9.0, 45.0, 0.0, 23.0, 66.0, 10.0, 22.0, 13.0, 19.0]
        expected_result = pd.Series(
            data=expected_result,
            index=pd.date_range('12/11/2021', freq='15min', periods=len(expected_result)))
        assert_series_equal(series_converted, expected_result, check_exact=False, check_dtype=False)

    def test_clean_ts_integrate_ok_delta(self):
        """ Test that clean_ts_integrate returns expected result for delta metrics """

        series_converted = data_preparation.clean_ts_integrate(self.df_delta, "delta")
        assert_series_equal(series_converted, self.df_counter, check_exact=False, check_dtype=False)

    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()


if __name__ == '__main__':
    unittest.main()
