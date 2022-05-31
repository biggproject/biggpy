# !/usr/bin/python3
# -*- coding: utf-8 -*-

import unittest

import numpy as np
from ai_toolbox import perfomance_metrics


class TestPerformanceMetrics(unittest.TestCase):
    """Class to test the performance_metrics module"""

    def test_cv_rmse_raises_if_arrays_inconsistent_length(self):
        """
        Test that cv_rmse raises if the y_true and y_pred arrays have
        inconsistent length.
        """

        y_true = np.array([-2, -1, 0, 1])
        y_pred = np.array([-3, -2, 1, 3, 2])

        self.assertRaises(
            ValueError,
            perfomance_metrics.cv_rmse,
            y_true=y_true,
            y_pred=y_pred
        )

    def test_cv_rmse_raises_if_y_true_empty(self):
        """ Test that cv_rmse raises if y_true is an empty array. """

        y_true = np.array([])
        y_pred = np.array([])

        self.assertRaises(
            ValueError,
            perfomance_metrics.cv_rmse,
            y_true=y_true,
            y_pred=y_pred
        )

    @unittest.skip("To be completed")
    def test_cv_rmse_returns_expected_result_with_multi_output(self):
        """  Test that cv_rmse returns the expected result if multi output. """

        y_true = np.array([[-2, -1], [0, 2]])
        y_pred = np.array([[-1, -1], [0, 2]])
        self.assertEqual(
            perfomance_metrics.cv_rmse(y_true=y_true, y_pred=y_pred),
            0
        )


if __name__ == '__main__':
    unittest.main()
