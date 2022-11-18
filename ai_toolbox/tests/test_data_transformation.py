# !/usr/bin/python3
# -*- coding: utf-8 -*-

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytz
from ai_toolbox import data_transformation
from pandas.testing import assert_frame_equal, assert_series_equal


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
        idx = pd.date_range(start='2021/10/01', end='2021/10/31', tz=pytz.utc, freq='10T')
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

    def test_add_lag_components_raises_if_empty_dataframe(self):
        """ Test that add_lag_components raises ValueError in case of empty DataFrame """

        self.assertRaises(ValueError, data_transformation.add_lag_components, pd.DataFrame(data=[]))

    def test_add_lag_components_raises_if_not_dataframe(self):
        """ Test that add_lag_components raises ValueError in case data is not a DataFrame """

        self.assertRaises(ValueError, data_transformation.add_lag_components, pd.Series(data=[2]))

    def test_add_lag_components_raises_if_columns_not_subset_of_dataframe_columns(self):
        """
        Test that add_lag_components raises ValueError in case the specified columns are
        not a subset of the input Dataframe columns
        """

        self.assertRaises(
            ValueError, data_transformation.add_lag_components, self.df_two_columns, columns=['n1', 'wrong_column'])

    def test_add_lag_components_returns_expected_result(self):
        """ Test that add_lag_components returns expected result """

        df_with_lags = data_transformation.add_lag_components(self.df_two_columns, columns=['n1'])
        self.assertEqual(df_with_lags.columns.to_list(), ['n1', 'n2', 'n1(-1)'])
        assert_series_equal(
            df_with_lags['n1(-1)'],
            self.df_two_columns['n1'].shift(1),
            check_exact=False,
            check_dtype=False,
            check_names=False)

    def test_add_calendar_components_raises_if_empty_dataframe(self):
        """ Test that add_calendar_components raises ValueError in case of empty DataFrame """

        self.assertRaises(ValueError, data_transformation.add_calendar_components, pd.DataFrame(data=[]))

    def test_add_calendar_components_raises_if_not_dataframe(self):
        """ Test that add_calendar_components raises ValueError in case data is not a DataFrame """

        self.assertRaises(ValueError, data_transformation.add_calendar_components, pd.Series(data=[2]))

    def test_add_calendar_components_raises_if_calendar_components_not_subset_of_dataframe_columns(self):
        """
        Test that add_calendar_components raises ValueError in case the specified calendar components are
        not a subset of the input Dataframe columns
        """

        self.assertRaises(
            ValueError,
            data_transformation.add_calendar_components,
            self.df_two_columns,
            calendar_components=['year', 'date'])

    def test_add_calendar_components_returns_expected_result(self):
        """ Test that add_calendar_components returns expected result """

        df_with_calendar = data_transformation.add_calendar_components(
            self.df_two_columns, calendar_components=['month', 'day'], drop_constant_columns=True)
        self.assertEqual(df_with_calendar.columns.to_list(), ['n1', 'n2', 'day'])

    def test_add_calendar_components_returns_expected_result_if_drop_constant_columns_is_false(self):
        """ Test that add_calendar_components returns expected result if drop_constant_columns is False """

        df_with_calendar = data_transformation.add_calendar_components(
            self.df_two_columns, calendar_components=['month', 'day'], drop_constant_columns=False)
        self.assertEqual(df_with_calendar.columns.to_list(), ['n1', 'n2', 'month', 'day'])

    def test_add_calendar_components_transformer_returns_same_results_as_function(self):
        """ Test that calendar component transformer and add_calendar_components return same results """

        calendar_components = ['month', 'day']

        assert_frame_equal(
            data_transformation.add_calendar_components(
                self.df_two_columns, calendar_components=calendar_components, drop_constant_columns=False),
            data_transformation.CalendarComponentTransformer(
                components=calendar_components).fit_transform(self.df_two_columns),
            check_exact=False,
            check_dtype=False)

    def test_add_calendar_components_transformer_returns_same_results_as_function_with_encoding(self):
        """ Test that calendar component transformer and add_calendar_components return same results """

        calendar_components = ['month', 'day']
        df_calendar = data_transformation.add_calendar_components(
                self.df_two_columns, calendar_components=calendar_components, drop_constant_columns=False)
        df_transformed1 = data_transformation.CalendarComponentTransformer(
            components=calendar_components, encode=True).fit_transform(self.df_two_columns)
        column_transformer = data_transformation.trigonometric_encode_calendar_components(
            data=df_calendar, calendar_components=calendar_components)
        df_transformed2 = column_transformer.fit_transform(df_calendar)
        df_transformed2 = pd.DataFrame(
            data=df_transformed2,
            index=df_calendar.index,
            columns=['month_sin', 'month_cos', 'day_sin', 'day_cos', 'n1', 'n2'])
        # Sort columns in the same order
        df_transformed2 = df_transformed2[df_transformed1.columns.to_list()]
        assert_frame_equal(
            df_transformed1,
            df_transformed2,
            check_exact=False,
            check_dtype=False)

    def test_crange_returns_expected_result_if_start_lower_than_end(self):
        """ Test that crange returns expected result if start lower than end """

        self.assertEqual(list(data_transformation.crange(3, 9, 12, include_zero=False)), [*range(3, 9)])
        self.assertEqual(list(data_transformation.crange(0, 5, 12, include_zero=True)), [*range(0, 5)])

    def test_crange_returns_expected_result_if_start_greater_than_end(self):
        """ Test that crange returns expected result if start greater than end"""

        self.assertEqual(list(data_transformation.crange(10, 4, 12, include_zero=False)), [10, 11, 12, 1, 2, 3])
        self.assertEqual(list(data_transformation.crange(20, 3, 24, include_zero=True)), [*range(20, 24)] + [0, 1, 2])

    def test_crange_returns_expected_result_if_start_equal_end(self):
        """ Test that crange returns expected result if start is equal to end """

        self.assertEqual(list(data_transformation.crange(4, 4, 7, include_zero=False)), [4, 5, 6, 0, 1, 2, 3])

    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()


if __name__ == '__main__':
    unittest.main()
