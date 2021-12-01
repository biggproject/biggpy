#!/usr/bin/python3
# -*- coding: utf-8 -*-

from os.path import isabs, splitext, dirname, isdir

from numpy import mean, std
from sklearn.metrics import get_scorer
from sklearn.model_selection import GridSearchCV, BaseCrossValidator, cross_validate
from sklearn.utils.validation import check_is_fitted


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
            for score in scoring:
                score_name = "test_{}".format(score)
                dict_scores[score_name].append(get_scorer(score)(fitted_model.best_estimator_, X_test, y_test))

        elif isinstance(scoring, str) or scoring is None:
            if scoring is None:
                # Use default scorer for the estimator (accuracy for classifiers and r2 for regressors)
                scores.append(fitted_model.best_estimator_.score(X_test, y_test))
            else:
                scores.append(get_scorer(scoring)(fitted_model.best_estimator_, X_test, y_test))

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
        scoring=scoring,
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
        - cv_results_evaluation: list of dict. A list of dictionaries where each element represents the results obtained
            on a specific model instance in terms of performance evaluation and selected hyper-parameters.
            Can be imported into a DataFrame.
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

    return search.best_estimator_, search.best_params_, mean_score, std_score, search.cv_results_, double_cv_results


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


if __name__ == '__main__':
    """
    This module is not supposed to run as a stand-alone module.
    This part below is only for testing purposes. 
    """

    from sklearn.datasets import load_breast_cancer
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    from sklearn.model_selection import KFold
    from time import time

    start_time = time()

    df = load_breast_cancer(as_frame=True).frame
    X = df.iloc[:, :-1]
    y = df.target

    cv_splitter_outer = KFold(n_splits=5, shuffle=True, random_state=1)
    cv_splitter_inner = KFold(n_splits=3, shuffle=True, random_state=1)
    grid = {
        RandomForestClassifier(random_state=1): {
            'n_estimators': [10, 100, 500],
            'max_features': [2, 4, 6]
        },
        SVC(random_state=1): {'C': [1, 10, 100]}
    }
    results = identify_best_model(
        X_data=X,
        y_data=y,
        model_families_parameter_grid=grid,
        cv_inner=cv_splitter_inner,
        cv_outer=cv_splitter_outer,
        scoring=['precision', 'recall', 'accuracy'],
        compare_with='recall'
    )

    print("Best model: {}".format(results[0]))
    print("Best parameters: {}".format(results[1]))
    print("Mean score: {}".format(results[2]))
    print("Std score: {}".format(results[3]))
    print("Evaluation results: {}".format(results[5]))

    print("Time to identify best model: {} seconds.".format(time()-start_time))
