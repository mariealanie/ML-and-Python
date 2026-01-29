import numpy as np
import typing
from collections import defaultdict


def kfold_split(num_objects: int,
                num_folds: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Split [0, 1, ..., num_objects - 1] into equal num_folds folds
       (last fold can be longer) and returns num_folds train-val
       pairs of indexes.

    Parameters:
    num_objects: number of objects in train set
    num_folds: number of folds for cross-validation split

    Returns:
    list of length num_folds, where i-th element of list
    contains tuple of 2 numpy arrays, he 1st numpy array
    contains all indexes without i-th fold while the 2nd
    one contains i-th fold
    """
    folds = []
    ln = num_objects // num_folds
    rmn = ln * (num_folds - 1)
    for el in range(num_folds - 1):
        val = [i for i in range(el * ln, (el + 1) * ln)]
        tr = [j for j in range(num_objects) if j not in val]
        folds.append((tr, val))
    val = [i for i in range(rmn, num_objects)]
    tr = [j for j in range(num_objects) if j not in val]
    folds.append((tr, val))
    return folds


def knn_cv_score(X: np.ndarray, y: np.ndarray, parameters: dict[str, list],
                 score_function: callable,
                 folds: list[tuple[np.ndarray, np.ndarray]],
                 knn_class: object) -> dict[str, float]:
    """Takes train data, counts cross-validation score over
    grid of parameters (all possible parameters combinations)

    Parameters:
    X: train set
    y: train labels
    parameters: dict with keys from
        {n_neighbors, metrics, weights, normalizers}, values of type list,
        parameters['normalizers'] contains tuples (normalizer, normalizer_name)
        see parameters example in your jupyter notebook

    score_function: function with input (y_true, y_predict)
        which outputs score metric
    folds: output of kfold_split
    knn_class: class of knn model to fit

    Returns:
    dict: key - tuple of (normalizer_name, n_neighbors, metric, weight),
    value - mean score over all folds
    """
    tbl = {}
    for n_neighbors in parameters['n_neighbors']:
        for metric in parameters['metrics']:
            for weight in parameters['weights']:
                for normalizer, normalizer_name in parameters['normalizers']:
                    scr = []
                    for train, val in folds:
                        x_train, x_val = X[train], X[val]
                        y_train, y_val = y[train], y[val]
                        if normalizer is not None:
                            Rel = normalizer
                            Rel.fit(x_train)
                            x_train_scaled = Rel.transform(x_train)
                            x_val_scaled = Rel.transform(x_val)
                        else:
                            x_train_scaled = x_train
                            x_val_scaled = x_val
                        model = knn_class(
                            n_neighbors=n_neighbors, metric=metric, weights=weight)
                        model.fit(x_train_scaled, y_train)
                        err = score_function(
                            y_val, model.predict(x_val_scaled))
                        scr.append(err)
                    tbl[(normalizer_name, n_neighbors,
                         metric, weight)] = np.mean(scr)
    return tbl
