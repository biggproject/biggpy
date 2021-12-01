# !/usr/bin/python3
# -*- coding: utf-8 -*-

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytz
from pandas.testing import assert_frame_equal

from ai_toolbox import data_transformation


class TestDataTransformation(unittest.TestCase):
    """Class to test the data_transformation module"""

    @classmethod
    def setUpClass(cls):
        """ Set up dfs and series for the entire test case """

        from os.path import dirname, join, realpath
        import pytz

        super().setUpClass()

        # Load time series for profile (dataframe) from csv
        dir_path = dirname(realpath(__file__))

        filename = join(dir_path, "fixtures", "df_yearly_profile.csv")
        cls.yearly_profile = pd.read_csv(
            filename,
            sep=',',
            parse_dates=True,
            infer_datetime_format=True,
            index_col=0)

        filename = join(dir_path, "fixtures", "df_weekly_profile.csv")
        cls.weekly_profile = pd.read_csv(
            filename,
            sep=',',
            parse_dates=True,
            infer_datetime_format=True,
            index_col=0)

        # Load resulting time series for profile (dataframe) from csv
        filename = join(dir_path, "fixtures", "df_yearly_profile_result.csv")
        cls.result_yearly_profile = pd.read_csv(
            filename,
            sep=',',
            parse_dates=True,
            infer_datetime_format=True,
            index_col=0)

        filename = join(dir_path, "fixtures", "df_weekly_profile_result.csv")
        cls.result_weekly_profile = pd.read_csv(
            filename,
            sep=',',
            parse_dates=True,
            infer_datetime_format=True,
            index_col=0)

        # Generate regular time series (dataframe) with 2 columns
        idx = pd.date_range(start='2021/10/01', end='2021/11/01', tz=pytz.utc, freq='10T')
        cls.df_two_columns = pd.DataFrame(
            data=np.random.randint(0, 100, size=(len(idx), 2)),
            index=idx,
            columns=["n1", "n2"]
        )

    # yearly_profile_detection tests below

    def test_yearly_profile_detection_raises_if_empty_series(self):
        """ Test that yearly_profile_detection raises ValueError in case of empty time series """

        self.assertRaises(ValueError, data_transformation.yearly_profile_detection, pd.DataFrame(data=[]))

    def test_yearly_profile_detection_raises_if_more_columns(self):
        """
        Test that yearly_profile_detection raises ValueError in case of DataFrame with
        more than one column
        """

        self.assertRaises(ValueError, data_transformation.yearly_profile_detection, self.df_two_columns)

    @patch("ai_toolbox.data_transformation.detect_time_step")
    def test_yearly_profile_detection_raises_if_freq_none(self, mock_detect_time_step):
        """ Test that yearly_profile_detection raises if the frequency cannot be determined """

        mock_detect_time_step.return_value = None, None
        self.assertRaises(ValueError, data_transformation.yearly_profile_detection, self.yearly_profile)

    @patch("ai_toolbox.data_transformation.detect_time_step")
    def test_yearly_profile_detection_raises_if_freq_lower_than_daily(self, mock_detect_time_step):
        """ Test that yearly_profile_detection raises if the frequency is lower than daily """

        # Try with yearly frequency
        mock_detect_time_step.return_value = 'Y', 'None'
        self.assertRaises(ValueError, data_transformation.yearly_profile_detection, self.yearly_profile)

    @patch("ai_toolbox.data_transformation.detect_time_step")
    def test_yearly_profile_detection_ok_if_freq_higher_than_daily(self, mock_detect_time_step):
        """
        Test that yearly_profile_detection gives expected result for each frequency higher than 1D
        and does not raise exception
        """

        for frequency in ['60S', '1T', '30min', '15H', '1D']:
            mock_detect_time_step.return_value = frequency, 'None'
            assert_frame_equal(
                data_transformation.yearly_profile_detection(self.yearly_profile),
                self.result_yearly_profile,
                check_exact=False,
                check_dtype=False)

    def test_yearly_profile_detection_raises_if_exclude_days_wrong_type(self):
        """ Test that yearly_profile_detection raises if exclude_days is of the wrong type """

        self.assertRaises(TypeError, data_transformation.yearly_profile_detection, self.yearly_profile, "wrong_type")

    def test_yearly_profile_detection_ok_if_exclude_days_list(self):
        """
        Test that yearly_profile_detection gives expected result if exclude days is a list
        and does not raise exception
        """

        df_result_exclude = self.result_yearly_profile.drop(pd.Timestamp("2021-03-17 00:00:00+00:00"), axis="index")
        assert_frame_equal(
            data_transformation.yearly_profile_detection(
                self.yearly_profile, ["2019-03-17", "2020-03-17", "2021-03-17"]),
            df_result_exclude,
            check_exact=False,
            check_dtype=False)

    def test_yearly_profile_detection_ok_if_exclude_days_series(self):
        """
        Test that yearly_profile_detection gives expected result if exclude days is a series
        and does not raise exception
        """

        idx = pd.date_range(start='2019/01/01', end='2022/01/01', tz=pytz.utc, freq='D')
        series_exclude = pd.Series(data=((idx.day == 17) & (idx.month == 3)), index=idx)
        df_result_exclude = self.result_yearly_profile.drop(pd.Timestamp("2021-03-17 00:00:00+00:00"), axis="index")
        assert_frame_equal(
            data_transformation.yearly_profile_detection(self.yearly_profile, series_exclude),
            df_result_exclude,
            check_exact=False,
            check_dtype=False)

    # weekly_profile_detection tests below
    
    def test_weekly_profile_detection_raises_if_empty_series(self):
        """ Test that weekly_profile_detection raises ValueError in case of empty time series """

        self.assertRaises(ValueError, data_transformation.weekly_profile_detection, pd.DataFrame(data=[]))

    def test_weekly_profile_detection_raises_if_more_columns(self):
        """
        Test that weekly_profile_detection raises ValueError in case of DataFrame with
        more than one column
        """

        self.assertRaises(ValueError, data_transformation.weekly_profile_detection, self.df_two_columns)

    @patch("ai_toolbox.data_transformation.detect_time_step")
    def test_weekly_profile_detection_raises_if_freq_none(self, mock_detect_time_step):
        """ Test that weekly_profile_detection raises if the frequency cannot be determined """

        mock_detect_time_step.return_value = None, None
        self.assertRaises(ValueError, data_transformation.weekly_profile_detection, self.weekly_profile)

    @patch("ai_toolbox.data_transformation.detect_time_step")
    def test_weekly_profile_detection_raises_if_freq_lower_than_hourly(self, mock_detect_time_step):
        """ Test that weekly_profile_detection raises if the frequency is lower than hourly """

        # Try with daily frequency
        mock_detect_time_step.return_value = '1D', 'None'
        self.assertRaises(ValueError, data_transformation.weekly_profile_detection, self.weekly_profile)

    @patch("ai_toolbox.data_transformation.detect_time_step")
    def test_weekly_profile_detection_ok_if_freq_higher_than_hourly(self, mock_detect_time_step):
        """
        Test that weekly_profile_detection gives expected result for each frequency higher than 1H
        and does not raise exception
        """

        for frequency in ['60S', '1T', '30min', '1H']:
            mock_detect_time_step.return_value = frequency, 'None'
            assert_frame_equal(
                data_transformation.weekly_profile_detection(self.weekly_profile),
                self.result_weekly_profile,
                check_exact=False,
                check_dtype=False)

    def test_weekly_profile_detection_raises_if_exclude_days_wrong_type(self):
        """ Test that weekly_profile_detection raises if exclude_days is of the wrong type """

        self.assertRaises(TypeError, data_transformation.weekly_profile_detection, self.weekly_profile, "wrong_type")

    def test_weekly_profile_detection_ok_if_exclude_days_list(self):
        """
        Test that weekly_profile_detection gives expected result if exclude days is a list
        and does not raise exception
        """

        # Exclude all the Sundays
        exclude_days = \
            ['2020-07-05 00:00:00+00:00', '2020-07-12 00:00:00+00:00', '2020-07-19 00:00:00+00:00',
             '2020-07-26 00:00:00+00:00', '2020-08-02 00:00:00+00:00', '2020-08-09 00:00:00+00:00',
             '2020-08-16 00:00:00+00:00', '2020-08-23 00:00:00+00:00', '2020-08-30 00:00:00+00:00',
             '2020-09-06 00:00:00+00:00', '2020-09-13 00:00:00+00:00', '2020-09-20 00:00:00+00:00',
             '2020-09-27 00:00:00+00:00']
        df_result_exclude = self.result_weekly_profile.drop(
            pd.date_range(start="2020-09-27", freq='1H', tz=pytz.utc, periods=24))
        assert_frame_equal(
            data_transformation.weekly_profile_detection(
                self.weekly_profile, exclude_days=exclude_days),
            df_result_exclude,
            check_exact=False,
            check_dtype=False)

    def test_weekly_profile_detection_ok_if_exclude_days_series(self):
        """
        Test that weekly_profile_detection gives expected result if exclude days is a series
        and does not raise exception
        """

        # Exclude all the Sundays
        idx = pd.date_range(start='2020-07-01', end='2020-09-30', tz=pytz.utc, freq='H')
        series_exclude = pd.Series(data=(idx.dayofweek == 6), index=idx)
        df_result_exclude = self.result_weekly_profile.drop(
            pd.date_range(start="2020-09-27", freq='1H', tz=pytz.utc, periods=24))
        assert_frame_equal(
            data_transformation.weekly_profile_detection(self.weekly_profile, series_exclude),
            df_result_exclude,
            check_exact=False,
            check_dtype=False)

    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()


if __name__ == '__main__':
    unittest.main()
