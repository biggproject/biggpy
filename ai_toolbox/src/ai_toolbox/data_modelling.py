#!/usr/bin/python3
# -*- coding: utf-8 -*-
from abc import ABC
from os.path import isabs, splitext, dirname, isdir
from typing import Union

from numpy import arange, maximum, mean, std, full
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression
from sklearn.metrics import get_scorer
from sklearn.model_selection import GridSearchCV, BaseCrossValidator, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

from ai_toolbox.perfomance_metrics import custom_scorers


class BlockingTimeSeriesSplit(BaseCrossValidator, ABC):
    """
    This class is a splitter performing a special type of time series partitioning
    to be used in the cross-validation framework. Differently from TimeSeriesSplit,
    this method will generate disjoint partitions of the dataset in each iteration.
    """

    def __init__(self, n_splits=5, gap=0):
        """
        Constructor of the splitter.

        :param n_splits: Number of splits. Must be at least 2. Default value is 5.
        :param gap: Number of samples to exclude from the end of each train set before the test set.
            Default value is 0.
        """
        if n_splits <= 1:
            raise ValueError(
                "BlockingTimeSeriesSplit requires at least one"
                " train/test split by setting n_splits=2 or more,"
                " got n_splits={0}.".format(n_splits))
        self.n_splits = n_splits
        self.gap = gap

    def get_n_splits(self, X=None, y=None, groups=None):
        """
        Returns the number of splits.

        :param X: X data
        :param y: y data
        :param groups: Left for backward compatibility
        :return: number of split
        """
        return self.n_splits

    def split(self, X, y=None, groups=None):
        """
        Yields indices to split data into training and test set.

        :param X: X data
        :param y: y data
        :param groups: Left for backward compatibility
        :return: Yields the split indices
        """
        n_samples = len(X)
        if self.n_splits > n_samples:
            raise ValueError(
                ("Cannot have number of splits n_splits={0} greater"
                 " than the number of samples: n_samples={1}.")
                .format(self.n_splits, n_samples))
        indices = arange(n_samples)
        fold_sizes = full(self.n_splits, n_samples // self.n_splits, dtype=int)
        fold_sizes[:n_samples % self.n_splits] += 1
        stop = 0
        for i in range(self.n_splits):
            start = stop
            stop = start + (fold_sizes[i])
            mid = int(0.5 * (stop - start)) + start
            yield indices[start: mid], indices[mid + self.gap: stop]


class NNWrapper(BaseEstimator, RegressorMixin):
    """
    Non-Negative Wrapper wraps other estimators to generate only non-negative predictions.
    This can be useful with polynomial regression when modeling on data that cannot
    assume negative values.
    """

    def __init__(self, estimator, **kwargs):
        self.estimator = estimator
        self.estimator.__init__(**kwargs)
        self.X_ = None
        self.y_ = None
        self.coef_ = None
        self.intercept_ = None

    def fit(self, X, y):

        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        self.X_ = X
        self.y_ = y

        # Fit the model and store the coefficients and intercept internally
        fitted_model = self.estimator.fit(X=X, y=y)
        self.coef_ = self.estimator.coef_
        self.intercept_ = self.estimator.intercept_

        return fitted_model

    def predict(self, X):

        # Check is fit had been called
        check_is_fitted(self)
        # Input validation
        X = check_array(X)
        return maximum(self.estimator.predict(X=X), 0)

    def set_params(self, **parameters):
        """
        Redefine 'set_params', used by optimization frameworks, e.g. GridSearchCV,
        to override the parameters set in the _init_ method.
        """

        self.estimator.set_params(**parameters)
        return self


class PolynomialRegression(BaseEstimator, RegressorMixin):
    """
    Polynomial regression estimator, created to offer a uniform interface
    to the other functions of this module.
    This class transforms the features into 'PolynomialFeatures' before
    fitting the data with the 'LinearRegression' estimator.
    """

    def __init__(self, degree=4, fit_intercept=True, normalize=False, copy_X=True,
                 n_jobs=None, positive=False):
        self.degree = degree
        self.fit_intercept = fit_intercept
        self.normalize = normalize
        self.copy_X = copy_X
        self.n_jobs = n_jobs
        self.positive = positive
        self.model = Pipeline([
            ("poly", PolynomialFeatures(
                degree=self.degree,
                include_bias=False
            )),
            ('linear', LinearRegression(
                fit_intercept=True,
                normalize=normalize,
                copy_X=copy_X,
                n_jobs=n_jobs,
                positive=positive))
        ])
        self.X_ = None
        self.y_ = None
        self.coef_ = None
        self.intercept_ = None

    def fit(self, X, y):

        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        self.X_ = X
        self.y_ = y

        # Fit the model and store the coefficients and intercept internally
        fitted_model = self.model.fit(X=X, y=y)
        self.coef_ = self.model.named_steps["linear"].coef_
        self.intercept_ = self.model.named_steps["linear"].intercept_

        return fitted_model

    def predict(self, X):

        # Check is fit had been called
        check_is_fitted(self)
        # Input validation
        X = check_array(X)
        return self.model.predict(X=X)

    def set_params(self, **parameters):
        """
        Redefine 'set_params', used by optimization frameworks, e.g. GridSearchCV,
        to override the parameters set in the _init_ method.
        """

        super(PolynomialRegression, self).set_params(**parameters)
        self.model = Pipeline([
            ("poly", PolynomialFeatures(
                degree=self.degree,
                include_bias=False
            )),
            ('linear', LinearRegression(
                fit_intercept=self.fit_intercept,
                normalize=self.normalize,
                copy_X=self.copy_X,
                n_jobs=self.n_jobs,
                positive=self.positive))
        ])
        return self


def evaluate_model_cv_with_tuning(model_family, X_data, y_data, parameter_grid, cv_outer, cv_inner, scoring=None):
    """
    This function performs a nested cross-validation (double cross-validation),
    which includes an internal hyper-parameter tuning, to reduce the bias when combining
    the two tasks of model selection and generalization error estimation.
    However, the purpose of this function is not to select the best model instance of a model family
    but instead to provide a less biased estimate of a tuned model’s performance on the dataset.

    :param model_family: identifies the class representing the model family
        (the ML algorithm to use, e.g. 'SVC()', 'DecisionTreeClassifier()', etc.)
    :param X_data: Training vector of shape (n_samples, n_features),
        where n_samples is the number of samples and n_features is the number of features.
    :param y_data: Y time series. Target relative to X for classification or regression; None for unsupervised learning.
    :param parameter_grid: Dictionary containing the set of parameters to explore.
    :param scoring: A string representing the scoring function to use. Default is the accuracy.
    :param cv_outer: This parameter is a generator coming from a partitioning function of the library
        which yields couples of k training sets and test sets indices, each couple representing one split.
        This splitter is related to the outer loop of cross-validation and generally has a k lower than or equal to the
        inner. The default value is 10.
    :param cv_inner: This parameter is a generator coming from a partitioning function of the library which
        yields couples of k training sets and test sets indices, each couple representing one split. This splitter is
        related to the inner loop of cross-validation for the hyper-parameter tuning. The default value is 5.
    :return:
        - scores: Dict with the mean cross-validated score and standard deviation for all the model instances for each
                scoring function specified.
        - cv_results: A list of dictionaries where each element represents the results obtained
            on a specific model instance in terms of performance evaluation and selected hyper-parameters.
            Can be imported into a DataFrame.
    """

    if not all(isinstance(i, BaseCrossValidator) for i in [cv_outer, cv_inner]):
        raise TypeError("Parameters 'cv_outer' and 'cv_inner' must be cross validator objects.")
    if scoring is not None and not isinstance(scoring, (str, list, tuple)):
        raise TypeError("Parameter 'scoring' must be a 'str', 'list' or 'tuple'.")

    scores = []
    cv_results = []
    dict_scores = {}

    sanitized_scorers = sanitize_scorers(scoring)

    # Initialize dict of aggregated scores in case of multiple scoring functions
    if isinstance(scoring, (list, tuple)):
        dict_scores = {"test_{}".format(score): list() for score in scoring}

    # Configure the internal hyper-parameter tuner
    hp_tuner = GridSearchCV(
        estimator=model_family,
        param_grid=parameter_grid,
        n_jobs=-1,
        cv=cv_inner,
        refit=True
    )

    for train_idx, test_idx in cv_outer.split(X_data):
        # Get train and test slices using the indices returned by the splitter
        X_train, X_test = X_data.iloc[train_idx], X_data.iloc[test_idx]
        y_train, y_test = y_data[train_idx], y_data[test_idx]

        # For each iteration of the outer cv run the GridSearchCV on the train-test slices
        fitted_model = hp_tuner.fit(X_train, y_train)
        cv_results.append(fitted_model.cv_results_)

        # Fill in the iterators with the scores obtained from the evaluation of best_estimator_
        if isinstance(scoring, (list, tuple)):
            # In case of multiple scoring metrics, build dict of scores per scoring function
            for score in sanitized_scorers.keys():
                score_name = "test_{}".format(score)
                dict_scores[score_name].append(
                    get_scorer(sanitized_scorers[score])(fitted_model.best_estimator_, X_test, y_test))

        elif isinstance(scoring, str) or scoring is None:
            if scoring is None:
                # Use default scorer for the estimator (accuracy for classifiers and r2 for regressors)
                scores.append(fitted_model.best_estimator_.score(X_test, y_test))
            else:
                scores.append(get_scorer(sanitized_scorers[scoring])(fitted_model.best_estimator_, X_test, y_test))

    # Aggregate values and return the results
    if isinstance(scoring, (list, tuple)):
        # In case of multiple scoring metrics, build dict of aggregated scores (mean and std) and return it
        aggr_scores = {}
        for score in scoring:
            score_name = "test_{}".format(score)
            aggr_scores[score_name] = {
                'mean': mean(dict_scores.get(score_name)),
                'std': std(dict_scores.get(score_name))}
        return aggr_scores, cv_results
    elif isinstance(scoring, str):
        score_name = "test_{}".format(scoring)
        return {score_name: {"mean": mean(scores), "std": std(scores)}}, cv_results
    elif scoring is None:
        return {"test_score": {"mean": mean(scores), "std": std(scores)}}, cv_results


def evaluate_model_cv_with_tuning_parallel(model_family, X_data, y_data, parameter_grid, cv_outer, cv_inner,
                                           scoring=None):
    """
    This is a simplified version of the function 'evaluate_model_cv_with_tuning' which uses directly cross_val_score
    of scikit-learn for the outer loop of the double cross-validation. However, in this version it
    is impossible to export the "cv_results" parameter for each iteration of the inner cv loop.
    This is the reason why the other version has been created.

    :param model_family: identifies the class representing the model family
        (the ML algorithm to use, e.g. 'SVC()', 'DecisionTreeClassifier()', etc.)
    :param X_data: Training vector of shape (n_samples, n_features),
        where n_samples is the number of samples and n_features is the number of features.
    :param y_data: Y time series. Target relative to X for classification or regression; None for unsupervised learning.
    :param parameter_grid: Dictionary containing the set of parameters to explore.
    :param scoring: A string representing the scoring function to use. Default is the accuracy.
    :param cv_outer: This parameter is a generator coming from a partitioning function of the library
        which yields couples of k training sets and test sets indices, each couple representing one split.
        This splitter is related to the outer loop of cross-validation and generally has a k lower than or equal to the
        inner. The default value is 10.
    :param cv_inner: This parameter is a generator coming from a partitioning function of the library which
        yields couples of k training sets and test sets indices, each couple representing one split. This splitter is
        related to the inner loop of cross-validation for the hyper-parameter tuning. The default value is 5.
    :return:
        - scores: Dict with the mean cross-validated score and standard deviation for all the model instances for each
                scoring function specified.
        - cv_results: A list of dictionaries where each element represents the results obtained
            on a specific model instance in terms of performance evaluation and selected hyper-parameters.
            Can be imported into a DataFrame.
    """

    if not all(isinstance(i, BaseCrossValidator) for i in [cv_outer, cv_inner]):
        raise TypeError("Parameters 'cv_outer' and 'cv_inner' must be cross validator objects.")
    if scoring is not None and not isinstance(scoring, (str, list, tuple)):
        raise TypeError("Parameter 'scoring' must be a 'str', 'list' or 'tuple'.")

    # Sanitize scorers and get the scoring function in case of custom scorer
    sanitized_scorers = sanitize_scorers(scoring)

    # Configure the internal hyper-parameter tuner
    hp_tuner = GridSearchCV(
        estimator=model_family,
        param_grid=parameter_grid,
        n_jobs=1,
        cv=cv_inner,
        refit=True
    )
    # Compute the score for each outer cv iteration
    scores = cross_validate(
        estimator=hp_tuner,
        X=X_data,
        y=y_data,
        scoring=sanitized_scorers,
        cv=cv_outer,
        n_jobs=-1
    )
    if isinstance(scoring, (list, tuple)):
        # In case of multiple scoring metrics, build dict of aggregated scores (mean and std) and return it
        aggr_scores = {}
        for score in scoring:
            score_name = "test_{}".format(score)
            aggr_scores[score_name] = {'mean': mean(scores.get(score_name)), 'std': std(scores.get(score_name))}
        return aggr_scores
    elif isinstance(scoring, str):
        score_name = "test_{}".format(scoring)
        scores = scores.get("test_score")
        return {score_name: {"mean": mean(scores), "std": std(scores)}}
    elif scoring is None:
        scores = scores.get("test_score")
        return {"test_score": {"mean": mean(scores), "std": std(scores)}}


def identify_best_model(X_data, y_data, model_families_parameter_grid, cv_outer, cv_inner, scoring=None,
                        compare_with=None):
    """
    This function implement a complete generalized pipeline to find the best model among different model families,
    each one associated with a specific parameter grid, given a input time series and a scoring function.

    :param X_data: Training vector of shape (n_samples, n_features), where n_samples is the number of samples and
        n_features is the number of features.
    :param y_data: Target relative to X for classification or regression; None for unsupervised learning.
    :param model_families_parameter_grid: Dictionary of key:values pairs, where the key is an object identifying the
        model_family (e.g. 'SVC', 'DecisionTreeClassifier', etc.) and the value is a dictionary identifying the
        parameter grid (subset of parameters to test) for that specific model family.
    :param cv_outer:  This parameter is a generator coming from a partitioning function of the library which yields
        couples of k training sets and test sets indices, each couple representing one split. This splitter is related
        to the outer loop of cross-validation and generally has a k lower than or equal to the inner.
    :param cv_inner: This parameter is a generator coming from a partitioning function of the library which yields
        couples of k training sets and test sets indices, each couple representing one split. This splitter is related
        to the inner loop of cross-validation for the hyper-parameter tuning and for the final model tuning required by
        the model selection procedure.
    :param scoring: A string representing the scoring function to use.
    :param compare_with: Parameter used to specify which one of the scoring functions specified in 'scoring' should
        be used to decide the best model family. This parameter will be ignored if 'scoring' is not a list or a tuple.
    :return:
        - best_model_instance: Best model instance of the model families found by the exhaustive search
            and retrained on the whole dataset.
        - best_params: Dictionary with a key:value pair, where the key identifies the best model family and the value
            the best configuration, i.e. the best set of hyper-parameters.
        - mean_score: float. Mean cross-validated score coming from the double cv for the best_model_instance.
        - mean_std: float. Standard deviation from the mean score coming from the double cv for the best_model_instance.
        _ cv_results_final: dict. A dict with keys as column headers and values as columns representing the test score
            for each split, each parameter combination, the rank of each set of parameters and the mean test score and
            standard deviation. Can be imported into a DataFrame.
        - cv_results_evaluation: dict. A dictionary containing the results of the performance evaluation obtained
            with the nested cross-validation, i.e. the mean cross-validated score and standard deviation for each
            model family and each scoring function specified.
    """

    if not all(isinstance(i, BaseCrossValidator) for i in [cv_outer, cv_inner]):
        raise TypeError("Parameters 'cv_outer' and 'cv_inner' must be cross validator objects.")
    if scoring is not None and not isinstance(scoring, (str, list, tuple)):
        raise TypeError("Parameter 'scoring' must be a 'str', 'list' or 'tuple'.")
    if isinstance(scoring, (list, tuple)) and compare_with not in scoring:
        raise ValueError("Parameter 'compare_with' must be a scoring function defined in 'scoring'.")

    double_cv_results = {}
    best_model_family, mean_score, std_score = None, None, None

    for model_family, parameter_grid in model_families_parameter_grid.items():
        double_cv_results[model_family] = evaluate_model_cv_with_tuning_parallel(
            model_family=model_family,
            X_data=X_data,
            y_data=y_data,
            parameter_grid=parameter_grid,
            cv_outer=cv_outer,
            cv_inner=cv_inner,
            scoring=scoring
            )

    # If multiple metrics in scoring, use the one in compare_with to choose the best family
    if isinstance(scoring, (list, tuple)):
        best_model_family = max(
            double_cv_results,
            key=lambda x: double_cv_results.get(x).get("test_{}".format(compare_with)).get("mean")
        )
        mean_score = double_cv_results.get(best_model_family).get("test_{}".format(compare_with)).get("mean")
        std_score = double_cv_results.get(best_model_family).get("test_{}".format(compare_with)).get("std")
    elif isinstance(scoring, str):
        best_model_family = max(
            double_cv_results,
            key=lambda x: double_cv_results.get(x).get("test_{}".format(scoring)).get("mean")
        )
        mean_score = double_cv_results.get(best_model_family).get("test_{}".format(scoring)).get("mean")
        std_score = double_cv_results.get(best_model_family).get("test_{}".format(scoring)).get("std")
    elif scoring is None:
        best_model_family = max(
            double_cv_results,
            key=lambda x: double_cv_results.get(x).get("test_score").get("mean")
        )
        mean_score = double_cv_results.get(best_model_family).get("test_score").get("mean")
        std_score = double_cv_results.get(best_model_family).get("test_score").get("std")

    search = GridSearchCV(
        estimator=best_model_family,
        param_grid=model_families_parameter_grid[best_model_family],
        n_jobs=-1,
        cv=cv_inner,
        refit=True
        ).fit(X=X_data, y=y_data)

    # Stringify double_cv_results keys before returning it
    double_cv_results = {stringify_estimator_name(key): value for key, value in double_cv_results.items()}

    return search.best_estimator_, search.best_params_, mean_score, std_score, search.cv_results_, double_cv_results


def serialize_model(model_instance, model_full_path, model_format='joblib'):
    """
    This function serializes and saves a model instance or a complete pipeline,
    according to the specified file format, to the specified full path on the file system.

    :param model_instance: Model instance which has been already fitted on X data.
    :param model_full_path: String identifying the full path (not relative and with no file extensions) of the file
        where the model should be saved. The extension will be added by the function based on the format chosen.
    :param model_format: Format of the model to serialize and persist.
        By default it will use the 'joblib' pickle format.
    :return: String identifying the filename in which the data is stored. The function will add the extension
        according to the format chosen.
    """

    if model_format not in ['joblib', 'pickle', 'cloudpickle']:
        raise ValueError("Serialization format must be in ['joblib', 'pickle', 'cloudpickle'].")
    elif not isabs(model_full_path) or not isdir(dirname(model_full_path)):
        raise ValueError("The model path is not an absolute path or the directory does not exist.")

    filename = "{}.{}".format(model_full_path, model_format)
    if model_format == 'joblib':
        from joblib import dump
        dump(model_instance, filename)
    elif model_format == 'pickle':
        from pickle import dump
        with open(filename, 'wb') as f:
            dump(model_instance, f)
    elif model_format == 'cloudpickle':
        from cloudpickle import dump
        with open(filename, 'wb') as f:
            dump(model_instance, f)

    return filename


def deserialize_and_predict(model_full_path, x_data):
    """
    This function deserializes a model or a pipeline, inferring the file format from the file name,
     applies the model on the X_data and returns the predicted values in the form of a time series.

    :param model_full_path: String identifying the full path (not relative and with the extension), on the file system
        of the file where the model should be loaded.
    :param x_data: Vector of predictors of shape (n_samples, n_features), where n_samples is the number
        of samples and n_features is the number of features or predictors.
    :return: Y time series with the predicted target values.
    """

    path, ext = splitext(model_full_path)

    if ext not in ['.joblib', '.pickle', '.cloudpickle']:
        raise ValueError("Model extension must be in ['.joblib', '.pickle', '.cloudpickle'].")
    elif not isabs(model_full_path) or not isdir(dirname(model_full_path)):
        raise ValueError("The model path is not an absolute path or the directory does not exist.")

    if ext == '.joblib':
        from joblib import load
        model_instance = load(model_full_path)
    elif ext == '.pickle':
        from pickle import load
        with open(model_full_path, 'rb') as f:
            model_instance = load(f)
    elif ext == '.cloudpickle':
        from cloudpickle import load
        with open(model_full_path, 'rb') as f:
            model_instance = load(f)

    return model_instance.predict(x_data)


def sanitize_scorers(scorers: Union[str, list, tuple]) -> dict:
    """
    Returns a dictionary of key-value pairs where the key is a string
    identifying the scorer and the value is the same string if the scorer
    is a predefined scikit-learn scorer or the scorer object if it is a
    custom ai-toolbox scorer.

    :param scorers: string or list/tuple of strings identifying a scorer to use
    :return: dictionary of 'sanitized scorers' to be used in crossvalidate
    """

    if isinstance(scorers, str):
        if scorers in custom_scorers.keys():
            return {scorers: custom_scorers[scorers]}
        else:
            return {scorers: scorers}
    elif isinstance(scorers, (list, tuple)):
        return {scorer: (custom_scorers[scorer] if scorer in custom_scorers.keys() else scorer) for scorer in scorers}
    else:
        raise ValueError("'{}' is not a valid scoring value.".format(scorers))


def stringify_estimator_name(estimator: Union[Pipeline, BaseEstimator]) -> str:
    """
    Converts the name of an estimator to string.
    In case the estimator is a Pipeline, this
    function will return only the string
    representation of the last step.
    """
    if isinstance(estimator, Pipeline):
        return str(estimator.steps[-1][1])
    else:
        return str(estimator)


if __name__ == '__main__':
    """
    This module is not supposed to run as a stand-alone module.
    This part below is only for testing purposes. 
    """

    import pandas as pd

    from sklearn.datasets import load_diabetes
    from sklearn.model_selection import KFold
    from sklearn.linear_model import Lasso
    from time import time

    start_time = time()
    df = load_diabetes(as_frame=True).frame
    df_X = df.iloc[:, :-1]
    df_y = df.target

    cv_splitter_outer = KFold(n_splits=5, shuffle=True, random_state=1)
    cv_splitter_inner = KFold(n_splits=3, shuffle=True, random_state=1)
    grid = {
        Pipeline([
            ('poly', PolynomialFeatures(include_bias=False)),
            ('model', NNWrapper(estimator=Lasso()))
        ]):
            {
                'poly__degree': list(range(3, 5)),
                'model__alpha': [0.1, 1, 10]
            }
    }
    results = identify_best_model(
        X_data=df_X,
        y_data=df_y,
        model_families_parameter_grid=grid,
        cv_inner=cv_splitter_inner,
        cv_outer=cv_splitter_outer,
        scoring=['mean_bias_error', 'normalized_mean_bias_error', 'r2', 'neg_root_mean_squared_error', 'cv_rmse'],
        compare_with='r2'
    )

    print("Best model: {}".format(results[0]))
    print("Best parameters: {}".format(results[1]))
    print("Mean score: {}".format(results[2]))
    print("Std score: {}".format(results[3]))
    print("{}".format(pd.DataFrame.from_dict(results[4]).to_markdown()))
    print("Evaluation results: {}".format(results[5]))

    print("Time to identify best model: {} seconds.".format(time() - start_time))
