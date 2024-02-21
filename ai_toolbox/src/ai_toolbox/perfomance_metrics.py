#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy as np

from sklearn.metrics import make_scorer
from sklearn.utils.validation import check_consistent_length


def cv_rmse(y_true, y_pred, multioutput='uniform_average', number_of_parameters=0):
    """
    Coefficient of variation of the RMSE.

    :param y_true: array-like of shape (n_samples,) or (n_samples, n_outputs)
        Ground truth (correct) target values.
    :param y_pred: array-like of shape (n_samples,) or (n_samples, n_outputs)
        Estimated target values.
    :param multioutput: {'raw_values', 'uniform_average'} or array-like
        Defines aggregating of multiple output values.
        Array-like value defines weights used to average errors.
        If input is list then the shape must be (n_outputs,).

        'raw_values' :
            Returns a full set of errors in case of multioutput input.

        'uniform_average' :
            Errors of all outputs are averaged with uniform weight.
    :param number_of_parameters: number of adjustable model parameters used to
        compute the degrees of freedom of the estimate.
    :return: loss : float or ndarray of floats
        A non-negative floating point value (the best value is 0.0), or an
        array of floating point values, one for each individual target.
    """

    check_consistent_length(y_true, y_pred)
    if len(y_true) == 0:
        raise ValueError("Input arrays must have length greater than zero.")

    sum_squared_errors = np.sum((y_true - y_pred) ** 2, axis=0)

    # Replace the mean of y_true with the epsilon if it is zero
    # to avoid zero division errors
    mean_y_true = np.where(np.mean(y_true, axis=0) == 0, np.finfo(np.float64).eps, np.mean(y_true, axis=0))
    output = (100 / mean_y_true) * np.sqrt(sum_squared_errors / (len(y_true) - number_of_parameters))

    if isinstance(multioutput, str):
        if multioutput == 'raw_values':
            return output
        elif multioutput == 'uniform_average':
            # pass None as weights to np.average: uniform mean
            multioutput = None

    return np.average(output, weights=multioutput)


def mean_bias_error(y_true, y_pred, normalized=False, multioutput='uniform_average', number_of_parameters=0):
    """
    Mean bias error (MBE) and Normalized Mean Bias Error (NMBE)
    It is an indicator of the overall behavior of the simulated data with regards to the
    regression line. Generally, positive values indicate the the model under-predicts
    measured data, while negative value that it over-predicts.


    :param y_true: array-like of shape (n_samples,) or (n_samples, n_outputs)
        Ground truth (correct) target values.
    :param y_pred: array-like of shape (n_samples,) or (n_samples, n_outputs)
        Estimated target values.
    :param normalized: normalizes by the mean if normalized is True.
    :param multioutput: {'raw_values', 'uniform_average'} or array-like
        Defines aggregating of multiple output values.
        Array-like value defines weights used to average errors.
        If input is list then the shape must be (n_outputs,).

        'raw_values' :
            Returns a full set of errors in case of multioutput input.

        'uniform_average' :
            Errors of all outputs are averaged with uniform weight.
    :param number_of_parameters: number of adjustable model parameters used to
        compute the degrees of freedom of the estimate.
    :return: loss : float or ndarray of floats
        A non-negative floating point value (the best value is 0.0), or an
        array of floating point values, one for each individual target.
    """

    check_consistent_length(y_true, y_pred)
    if len(y_true) == 0:
        raise ValueError("Input arrays must have length greater than zero.")

    output_errors = np.mean(y_true - y_pred, axis=0)

    if normalized:
        # Replace the mean of y_true with the epsilon if it is zero
        # to avoid zero division errors
        mean_y_true = np.where(np.mean(y_true, axis=0) == 0, np.finfo(np.float64).eps, np.mean(y_true, axis=0))
        return (100 / mean_y_true) * np.sum(y_true - y_pred, axis=0) / (len(y_true) - number_of_parameters)

    if isinstance(multioutput, str):
        if multioutput == 'raw_values':
            return output_errors
        elif multioutput == 'uniform_average':
            # pass None as weights to np.average: uniform mean
            multioutput = None

    return np.average(output_errors, weights=multioutput)


def r2_ipmvp(y_true, y_pred, multioutput='uniform_average'):
    """
    R2 (coefficient of determination) indicates how close simulated values are to the regression line
    of the measured values. The formula used here guarantees that the result will be
    limited between 0.0 and 1.0, differently from the scikit-learn version, which could
    also produce a negative value.

    :param y_true: array-like of shape (n_samples,) or (n_samples, n_outputs)
        Ground truth (correct) target values.
    :param y_pred: array-like of shape (n_samples,) or (n_samples, n_outputs)
        Estimated target values.
    :param multioutput: {'raw_values', 'uniform_average'} or array-like
        Defines aggregating of multiple output values.
        Array-like value defines weights used to average errors.
        If input is list then the shape must be (n_outputs,).

        'raw_values' :
            Returns a full set of errors in case of multioutput input.

        'uniform_average' :
            Errors of all outputs are averaged with uniform weight.
    :return: The R^2 score (non-negative).
    """

    check_consistent_length(y_true, y_pred)
    if len(y_true) == 0:
        raise ValueError("Input arrays must have length greater than zero.")

    num_r2 = (len(y_true) * np.sum(y_true * y_pred) - np.sum(y_true) * np.sum(y_pred)) ** 2
    den_r2 = ((len(y_true) * np.sum(y_true ** 2) - np.sum(y_true) ** 2) *
              (len(y_pred) * np.sum(y_pred ** 2) - np.sum(y_pred) ** 2))
    output = num_r2 / den_r2

    if isinstance(multioutput, str):
        if multioutput == 'raw_values':
            return output
        elif multioutput == 'uniform_average':
            # pass None as weights to np.average: uniform mean
            multioutput = None

    return np.average(output, weights=multioutput)


mean_bias_error_scorer = make_scorer(
    score_func=mean_bias_error,
    greater_is_better=False,
    normalized=False,
    multioutput='uniform_average',
    number_of_parameters=0)

normalized_mean_bias_error_scorer = make_scorer(
    score_func=mean_bias_error,
    greater_is_better=False,
    normalized=True,
    multioutput='uniform_average',
    number_of_parameters=0)

cv_rmse_scorer = make_scorer(
    score_func=cv_rmse,
    greater_is_better=False,
    multioutput='uniform_average',
    number_of_parameters=1)

r2_ipmvp_scorer = make_scorer(
    score_func=r2_ipmvp,
    greater_is_better=True,
    multioutput='uniform_average')

custom_scorers = {
    'mean_bias_error': mean_bias_error_scorer,
    'normalized_mean_bias_error': normalized_mean_bias_error_scorer,
    'cv_rmse': cv_rmse_scorer,
    'r2_ipmvp': r2_ipmvp_scorer
}

if __name__ == '__main__':
    """
    This module is not supposed to run as a stand-alone module.
    This part below is only for testing purposes. 
    """
