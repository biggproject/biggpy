#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy as np

from sklearn.metrics import make_scorer, SCORERS
from sklearn.utils.validation import check_consistent_length


def cv_rmse(y_true, y_pred, multioutput='uniform_average', number_of_parameters=0):
    """ Mean bias error for regression.

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
    """ Mean bias error for regression.

    :param y_true: array-like of shape (n_samples,) or (n_samples, n_outputs)
        Ground truth (correct) target values.
    :param y_pred: array-like of shape (n_samples,) or (n_samples, n_outputs)
        Estimated target values.
    :param normalized: computes the normalized mean bias error if normalized is True.
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

SCORERS.update({
    'mean_bias_error': mean_bias_error_scorer,
    'normalized_mean_bias_error': normalized_mean_bias_error_scorer,
    'cv_rmse': cv_rmse_scorer
})

if __name__ == '__main__':
    """
    This module is not supposed to run as a stand-alone module.
    This part below is only for testing purposes. 
    """

    y_true = np.array([-2, -1, 0, 1, 2])
    y_pred = np.array([-3, -2, 1, 3, 2])

    print(mean_bias_error(y_true=y_true, y_pred=y_pred, normalized=True))
