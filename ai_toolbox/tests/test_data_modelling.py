# !/usr/bin/python3
# -*- coding: utf-8 -*-

import unittest

import numpy as np
from ai_toolbox import data_modelling
from numpy.testing import assert_array_equal
from sklearn import svm
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import KFold
from sklearn.utils.validation import NotFittedError


class TestDataModelling(unittest.TestCase):
    """Class to test the data_modelling module"""

    full_path = "/tmp/test"
    full_path_ext = "{}.joblib".format(full_path)
    df = load_breast_cancer(as_frame=True).frame
    X = df.iloc[:, :-1]
    y = df.target
    clf = svm.SVC()
    fitted_model = svm.SVC().fit(X, y)

    @classmethod
    def setUpClass(cls):
        """ Set up dfs and series for the entire test case """

        super().setUpClass()

    # serialize_model tests below

    def test_serialize_model_raises_if_model_format_wrong(self):
        """ Test that serialize_model raises if the specified model format is wrong """

        self.assertRaises(
            ValueError,
            data_modelling.serialize_model,
            self.fitted_model,
            self.full_path,
            "wrong_format")

    def test_serialize_model_raises_if_not_full_path(self):
        """ Test that serialize_model raises if the specified path is not a full path """

        self.assertRaises(
            ValueError,
            data_modelling.serialize_model,
            self.fitted_model,
            "not_full_path")

    def test_serialize_model_raises_if_dir_not_exists(self):
        """ Test that serialize_model raises if the specified dir does not exist """

        self.assertRaises(
            ValueError,
            data_modelling.serialize_model,
            self.fitted_model,
            "/tmp/some_dir_not_existing/my_model")

    def test_serialize_model_raises_if_not_model_not_fitted(self):
        """ Test that serialize_model raises if the model to serialize is not fitted """

        self.assertRaises(
            NotFittedError,
            data_modelling.serialize_model,
            self.clf,
            self.full_path)

    def test_serialize_model_returns_expected_prediction_with_joblib(self):
        """ Test that serialize_model returns expected prediction """
        from joblib import load

        my_model = data_modelling.serialize_model(self.fitted_model, self.full_path)
        my_model = load(my_model)
        assert_array_equal(self.fitted_model.predict(self.X[-8:]), my_model.predict(self.X[-8:]))

    def test_serialize_model_returns_expected_prediction_with_pickle(self):
        """ Test that serialize_model returns expected prediction with pickle"""
        from pickle import load

        my_model = data_modelling.serialize_model(self.fitted_model, self.full_path, "pickle")
        with open(my_model, 'rb') as f:
            my_model = load(f)
        assert_array_equal(self.fitted_model.predict(self.X[-8:]), my_model.predict(self.X[-8:]))

    # deserialize_and_predict tests below

    def test_deserialize_and_predict_raises_if_model_ext_wrong(self):
        """ Test that deserialize_and_predict raises if the specified model format is wrong """

        self.assertRaises(
            ValueError,
            data_modelling.deserialize_and_predict,
            "test.job",
            self.X)

    def test_deserialize_and_predict_raises_if_not_full_path(self):
        """ Test that deserialize_and_predict raises if the specified path is not a full path """

        self.assertRaises(
            ValueError,
            data_modelling.deserialize_and_predict,
            "not_full_path.joblib",
            self.X)

    def test_deserialize_and_predict_raises_if_dir_not_exists(self):
        """ Test that deserialize_and_predict raises if the specified dir does not exist """

        self.assertRaises(
            ValueError,
            data_modelling.deserialize_and_predict,
            "/tmp/some_dir_not_existing/my_model.joblib",
            self.X)

    def test_deserialize_and_predict_raises_if_model_not_fitted(self):
        """ Test that deserialize_and_predict raises if the model to serialize is not fitted """

        from joblib import dump

        dump(self.clf, self.full_path_ext)
        self.assertRaises(
            NotFittedError,
            data_modelling.deserialize_and_predict,
            self.full_path_ext,
            self.X)

    def test_deserialize_and_predict_returns_expected_prediction_with_pickle(self):
        """ Test that deserialize_and_predict returns expected prediction with pickle """
        
        my_model = data_modelling.serialize_model(self.fitted_model, self.full_path, "pickle")
        predictions = data_modelling.deserialize_and_predict(my_model, self.X[-8:])
        assert_array_equal(self.fitted_model.predict(self.X[-8:]), predictions)

    def test_deserialize_and_predict_returns_expected_prediction_with_joblib(self):
        """ Test that deserialize_and_predict returns expected prediction with pickle """

        my_model = data_modelling.serialize_model(self.fitted_model, self.full_path)
        predictions = data_modelling.deserialize_and_predict(my_model, self.X[-8:])
        assert_array_equal(self.fitted_model.predict(self.X[-8:]), predictions)

    # evaluate_model_cv_with_tuning tests below

    def test_evaluate_model_cv_with_tuning_raises_if_not_cv_objects(self):
        """
        Test that evaluate_model_cv_with_tuning raises in case cv_inner and
        cv_outer are not cross validator objects.
        """

        self.assertRaises(TypeError,
                          data_modelling.evaluate_model_cv_with_tuning,
                          model_family=self.clf,
                          parameter_grid={},
                          X_data=self.X,
                          y_data=self.y,
                          cv_outer=2,
                          cv_inner=2,
                          scoring=None)

    def test_evaluate_model_cv_with_tuning_parallel_raises_if_not_cv_objects(self):
        """
        Test that evaluate_model_cv_with_tuning_parallel raises in case cv_inner and
        cv_outer are not cross validator objects.
        """

        self.assertRaises(TypeError,
                          data_modelling.evaluate_model_cv_with_tuning_parallel,
                          model_family=self.clf,
                          parameter_grid={},
                          X_data=self.X,
                          y_data=self.y,
                          cv_outer=2,
                          cv_inner=2,
                          scoring=None)

    def test_evaluate_model_cv_with_tuning_raises_if_wrong_scoring_type(self):
        """
        Test that evaluate_model_cv_with_tuning raises in case scoring is not
        str, list, tuple or None.
        """

        cv_splitter = KFold(n_splits=5, shuffle=True, random_state=1)
        self.assertRaises(TypeError,
                          data_modelling.evaluate_model_cv_with_tuning,
                          model_family=self.clf,
                          parameter_grid={},
                          X_data=self.X,
                          y_data=self.y,
                          cv_outer=cv_splitter,
                          cv_inner=cv_splitter,
                          scoring=4)

    def test_evaluate_model_cv_with_tuning_parallel_raises_if_wrong_scoring_type(self):
        """
        Test that evaluate_model_cv_with_tuning_parallel raises in case scoring is not
        str, list, tuple or None.
        """

        cv_splitter = KFold(n_splits=5, shuffle=True, random_state=1)
        self.assertRaises(TypeError,
                          data_modelling.evaluate_model_cv_with_tuning_parallel,
                          model_family=self.clf,
                          parameter_grid={},
                          X_data=self.X,
                          y_data=self.y,
                          cv_outer=cv_splitter,
                          cv_inner=cv_splitter,
                          scoring=4)

    def test_identify_best_model_raises_if_compare_with_not_in_scoring(self):
        """
        Test that evaluate_model_cv_with_tuning_parallel raises in case scoring is not
        str, list, tuple or None.
        """

        cv_splitter = KFold(n_splits=5, shuffle=True, random_state=1)
        self.assertRaises(ValueError,
                          data_modelling.identify_best_model,
                          X_data=self.X,
                          y_data=self.y,
                          model_families_parameter_grid={},
                          cv_outer=cv_splitter,
                          cv_inner=cv_splitter,
                          scoring=['precision', 'recall', 'accuracy'],
                          compare_with='not_in_scoring'
                          )

    def test_blocking_time_series_split_returns_expected_result(self):
        """ Test that BlockingTimeSeriesSplit returns the expected results """

        X = np.linspace(start=(1, 2), stop=(10, 20), num=13, dtype=int)
        splitter = data_modelling.BlockingTimeSeriesSplit(n_splits=3, gap=1)
        indices = [(train_index.tolist(), test_index.tolist()) for train_index, test_index in splitter.split(X)]
        self.assertEqual(indices, [([0, 1], [3, 4]), ([5, 6], [8]), ([9, 10], [12])])

    def test_blocking_time_series_split_raises_if_nsplits_lower_than_2(self):
        """ Test that BlockingTimeSeriesSplit raises in case n_splits is lower than 2 """

        self.assertRaises(ValueError, data_modelling.BlockingTimeSeriesSplit, 1, 0)

    def test_blocking_time_series_split_raises_if_nsplits_greater_than_nsamples(self):
        """ Test that BlockingTimeSeriesSplit raises in case n_splits is greater than n_samples """

        X = np.linspace(start=(1, 2), stop=(10, 20), num=13, dtype=int)
        splitter = data_modelling.BlockingTimeSeriesSplit(n_splits=14, gap=0)
        with self.assertRaises(ValueError):
            next(splitter.split(X))

    @classmethod
    def tearDownClass(cls):
        """ Clean-up actions """

        from os import remove

        try:
            remove("{}.joblib".format(cls.full_path))
        except FileNotFoundError:
            pass
        try:
            remove("{}.pickle".format(cls.full_path))
        except FileNotFoundError:
            pass
        finally:
            super().tearDownClass()


if __name__ == '__main__':
    unittest.main()
