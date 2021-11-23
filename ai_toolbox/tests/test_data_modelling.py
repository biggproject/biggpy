# !/usr/bin/python3
# -*- coding: utf-8 -*-

import unittest

from ai_toolbox import data_modelling
from numpy.testing import assert_array_equal
from sklearn import svm
from sklearn.datasets import load_iris
from sklearn.utils.validation import NotFittedError


class TestDataModelling(unittest.TestCase):
    """Class to test the data_modelling module"""

    full_path = "/tmp/test"
    full_path_ext = "{}.joblib".format(full_path)
    X, y = load_iris(return_X_y=True)
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
