#!/usr/bin/python3
# -*- coding: utf-8 -*-

from os.path import isabs, splitext, dirname, isdir

from sklearn.utils.validation import check_is_fitted


def evaluate_model_cv_with_tuning():
    pass


def identify_best_model():
    pass


def serialize_model(model_instance, model_full_path, model_format='joblib'):
    """
    This function serializes and saves a model instance, according to the specified file format,
     to the specified full path on the file system.

    :param model_instance: Model instance which has been already fitted on X data.
    :param model_full_path: String identifying the full path (not relative and with no file extensions) of the file
        where the model should be saved. The extension will be added by the function based on the format chosen.
    :param model_format: Format of the model to serialize and persist.
        By default it will use the 'joblib' pickle format.
    :return: String identifying the filename in which the data is stored. The function will add the extension
        according to the format chosen.
    """

    if model_format not in ['joblib', 'pickle']:
        raise ValueError("Serialization format must be in ['joblib', 'pickle'].")
    elif not isabs(model_full_path) or not isdir(dirname(model_full_path)):
        raise ValueError("The model path is not an absolute path or the directory does not exist.")

    # Check that the model is fitted before serializing
    check_is_fitted(model_instance)

    filename = "{}.{}".format(model_full_path, model_format)
    if model_format == 'joblib':
        from joblib import dump
        dump(model_instance, filename)
    elif model_format == 'pickle':
        from pickle import dump
        with open(filename, 'wb') as f:
            dump(model_instance, f)

    return filename


def deserialize_and_predict(model_full_path, x_data):
    """
    This function deserializes a model, inferring the file format from the file name,
     applies the model on the X_data and returns the predicted values in the form of a time series.

    :param model_full_path: String identifying the full path (not relative and with the extension), on the file system
        of the file where the model should be loaded.
    :param x_data: Vector of predictors of shape (n_samples, n_features), where n_samples is the number
        of samples and n_features is the number of features or predictors.
    :return: Y time series with the predicted target values.
    """

    path, ext = splitext(model_full_path)

    if ext not in ['.joblib', '.pickle']:
        raise ValueError("Model extension must be in ['.joblib', '.pickle'].")
    elif not isabs(model_full_path) or not isdir(dirname(model_full_path)):
        raise ValueError("The model path is not an absolute path or the directory does not exist.")

    if ext == '.joblib':
        from joblib import load
        model_instance = load(model_full_path)
    elif ext == '.pickle':
        from pickle import load
        with open(model_full_path, 'rb') as f:
            model_instance = load(f)

    # Check that the model is fitted after loading it and before predicting
    check_is_fitted(model_instance)

    return model_instance.predict(x_data)
