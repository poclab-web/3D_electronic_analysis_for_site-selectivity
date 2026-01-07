from __future__ import annotations

from itertools import product, combinations
from multiprocessing import Pool
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import (
    ElasticNet,
    Lasso,
    OrthogonalMatchingPursuit,
    Ridge,
)
from sklearn.model_selection import KFold


def regression(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y: np.ndarray,
    method: str,
) -> Tuple[np.ndarray, float, np.ndarray]:
    """
    Perform linear regression with various models specified by a method string.

    Supported models
    ----------------
    The `method` string must begin with one of the following keywords:

    - ``"Ridge alpha"``  
      Ridge regression with L2 regularization.
      * alpha: float (regularization strength)

    - ``"Lasso alpha"``  
      Lasso regression with L1 regularization.
      * alpha: float (regularization strength)

    - ``"ElasticNet alpha l1_ratio"``  
      Elastic Net regression with a mix of L1/L2 regularization.
      * alpha: float (overall regularization strength)
      * l1_ratio: float in [0, 1] (balance between L1 and L2)

    - ``"PLS n_components"``  
      Partial Least Squares regression.
      * n_components: int (number of latent components)

    - ``"OMP n_components"``  
      Orthogonal Matching Pursuit regression.
      * n_components: int (maximum number of non-zero coefficients)

    Parameters
    ----------
    X_train : numpy.ndarray, shape (n_train, n_features)
        Training feature matrix.
    X_test : numpy.ndarray, shape (n_test, n_features)
        Test feature matrix on which predictions are made.
    y_train : numpy.ndarray, shape (n_train,)
        Training target values corresponding to `X_train`.
    y : numpy.ndarray, shape (n_samples,)
        Full target array (train + test) used only for possible clipping logic
        in some variants (kept for backward compatibility).
    method : str
        Method specification string as described above.

    Returns
    -------
    coef : numpy.ndarray, shape (n_features,)
        Fitted regression coefficients.
    intercept : float
        Intercept term of the fitted model.
    predict : numpy.ndarray, shape (n_test,)
        Predicted values for `X_test`.

    Notes
    -----
    The function does not perform any feature scaling or preprocessing; this is
    expected to be done upstream if necessary.

    Examples
    --------
    >>> coef, intercept, pred = regression(
    ...     X_train, X_test, y_train, y, "ElasticNet 0.1 0.5"
    ... )
    >>> coef, intercept, pred = regression(
    ...     X_train, X_test, y_train, y, "PLS 3"
    ... )
    """
    if "Ridge" in method:
        alpha = float(method.split()[1])
        model = Ridge(alpha=alpha, fit_intercept=True, max_iter=10000)
        model.fit(X_train, y_train)
        coef = model.coef_
        intercept = model.intercept_
        predict = model.predict(X_test)

    elif "Lasso" in method and "ElasticNet" not in method:
        # Avoid accidental match with "ElasticNet" (although it does not contain "Lasso")
        alpha = float(method.split()[1])
        model = Lasso(alpha=alpha, fit_intercept=True, max_iter=10000)
        model.fit(X_train, y_train)
        coef = model.coef_
        intercept = model.intercept_
        predict = model.predict(X_test)

    elif "ElasticNet" in method:
        alpha, l1ratio = map(float, method.split()[1:3])
        model = ElasticNet(
            alpha=alpha,
            l1_ratio=l1ratio,
            fit_intercept=True,
            max_iter=10000,
        )
        model.fit(X_train, y_train)
        coef = model.coef_
        intercept = model.intercept_
        predict = model.predict(X_test)

    elif "PLS" in method:
        n_components = int(method.split()[1])
        model = PLSRegression(n_components=n_components, scale=False)
        model.fit(X_train, y_train)
        coef = model.coef_[0]
        intercept = model.intercept_[0]
        predict = model.predict(X_test)  # shape (n_test, 1)

    elif "OMP" in method:
        n_components = int(method.split()[1])
        model = OrthogonalMatchingPursuit(
            n_nonzero_coefs=int(n_components),
            fit_intercept=True,
        )
        model.fit(X_train, y_train)
        coef = model.coef_
        intercept = model.intercept_
        predict = model.predict(X_test)

    else:
        raise ValueError(f"Unsupported method string: {method}")

    return coef, intercept, predict


def regression_parallel(
    args: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]
) -> Tuple[str, np.ndarray, float, np.ndarray, np.ndarray]:
    """
    Wrapper function to run a single regression model with LOOCV in parallel.

    This function is designed to be used with ``multiprocessing.Pool``.
    It runs one regression model specified by `method`, computes predictions
    on the full feature matrix `X`, and performs leave-one-out cross-validation
    (LOOCV) on the training subset.

    Parameters
    ----------
    args : tuple
        A tuple containing:
        - X_train : numpy.ndarray, shape (n_train, n_features)
            Feature matrix for training.
        - X : numpy.ndarray, shape (n_samples, n_features)
            Full feature matrix (train + test) on which `predict` is evaluated.
        - y_train : numpy.ndarray, shape (n_train,)
            Training target values.
        - y : numpy.ndarray, shape (n_samples,)
            Full target array (train + test); passed to `regression` for
            compatibility.
        - method : str
            Method specification string (see :func:`regression`).

    Returns
    -------
    method : str
        The method string corresponding to this regression run.
    coef : numpy.ndarray, shape (n_features,)
        Regression coefficients.
    intercept : float
        Intercept term.
    predict : numpy.ndarray, shape (n_samples,)
        Predictions for the full feature matrix `X`.
    cv_values : numpy.ndarray, shape (n_train,)
        LOOCV predictions for the training subset, ordered to match `y_train`.
    """
    X_train, X, y_train, y, method = args
    coef, intercept, predict = regression(X_train, X, y_train, y, method)

    cvs: List[float] = []
    sort_index: List[int] = []

    # LOOCV: KFold with n_splits = n_train (no shuffling to preserve order)
    kf = KFold(n_splits=len(y_train), shuffle=False)
    for train_index, test_index in kf.split(y_train):
        if np.count_nonzero(coef) == 0:
            cv = [0.0] * len(test_index)
        else:
            # Use only non-zero coefficients for the CV model
            _, _, cv = regression(
                X_train[train_index][:, coef != 0],
                X_train[test_index][:, coef != 0],
                y_train[train_index],
                y,
                method,
            )
            # _, cv = regression(X_train[train_index], X_train[test_index], y_train[train_index], y, method)  # noqa: E501
        cvs.extend(cv)
        sort_index.extend(test_index)

    original_array = np.empty_like(cvs, dtype=float)
    original_array[sort_index] = cvs
    return method, coef, intercept, predict, original_array


def regression_(path: str, names: Sequence[str]) -> None:
    """
    Perform regression on grid-based descriptors and save model results.

    This function:
      1. Loads a preprocessed dataset from a pickle file.
      2. Extracts folded grid-based features (e.g., electronic/electrostatic).
      3. Standardizes each feature block by its standard deviation.
      4. Builds a combined feature matrix for training and full dataset.
      5. Evaluates multiple linear models (Lasso, Ridge, ElasticNet, PLS, OMP)
         in parallel with LOOCV.
      6. Stores regression coefficients and predictions to disk.

    Parameters
    ----------
    path : str
        Path to the input pickle file. The file must contain columns:
        - ``ΔΔG.expt.`` : experimental target values [kcal/mol]
        - ``test`` : indicator (0 for training, 1 for test)
        - Feature columns like ``"{name}_fold ..."`` for each `name` in `names`.
    names : sequence of str
        List of feature prefixes (e.g., ``["electronic", "electrostatic", "lumo"]``)
        whose folded grid columns (``"{name}_fold ..."``) will be used as features.

    Returns
    -------
    None
        Results are saved to:
        - ``path.replace(".pkl", f"_{feature_names}_regression.pkl")``:  
          pickle file with predictions, CV results and meta-data.
        - ``path.replace(".pkl", f"_{feature_names}_regression.csv")``:  
          CSV file containing coefficients on the original grid.

    Notes
    -----
    - LOOCV is implemented via :class:`sklearn.model_selection.KFold`
      with ``n_splits=len(y_train)``.
    - Each model's coefficients are rescaled by the standard deviation
      used in feature normalization.
    """
    print(path)
    df = pd.read_pickle(path)
    df_train = df[df["test"] == 0]

    y_train = df_train["ΔΔG.expt."].values
    y = df["ΔΔG.expt."].values

    trains: List[np.ndarray] = []
    train_tests: List[np.ndarray] = []
    stds: List[float] = []

    # --- build feature blocks ---
    for name in names:
        train = df_train.filter(like=f"{name}_fold").to_numpy()
        std = np.std(train)
        # std = np.linalg.norm(train)  # /np.size(train)
        train_test = df.filter(like=f"{name}_fold").to_numpy()
        # train -= average
        # train_test -= average

        train /= std
        train_test /= std

        trains.append(train)
        train_tests.append(train_test)
        stds.append(std)

    # --- define methods ---
    methods: List[str] = []

    for alpha in np.logspace(-14, 0, 15, base=2):
        methods.append(f"Lasso {alpha}")
    for alpha in np.logspace(-9, 5, 15, base=2):
        methods.append(f"Ridge {alpha}")
    for alpha, l1ratio in product(np.logspace(-14, 0, 15, base=2), [0.5]):  # np.round(np.linspace(0.1, 0.9, 9),decimals=10)  # noqa: E501
        methods.append(f"ElasticNet {alpha} {l1ratio}")
    for n_components in range(1, 15):
        methods.append(f"PLS {n_components}")
    for n_components in range(1, 15):
        methods.append(f"OMP {n_components}")

    # index for grid coefficients (x y z)
    grid = pd.DataFrame(
        index=[
            col.replace("electronic_fold ", "")
            for col in df.filter(like="electronic_fold ").columns
        ]
    )

    # --- run all regressions in parallel ---
    X_train_full = np.concatenate(trains, axis=1)
    X_full = np.concatenate(train_tests, axis=1)

    with Pool(24) as pool:
        results = list(
            pool.imap_unordered(
                regression_parallel,
                [
                    (X_train_full, X_full, y_train, y, method)
                    for method in methods
                ],
            )
        )

    # --- collect results ---
    for result in results:
        method, coef, intercept, predict, original_array = result
        print(method)

        # split coefficient vector back into blocks for each name
        coef_blocks = np.split(coef, len(names))
        for name, std, coef_block in zip(names, stds, coef_blocks):
            grid[f"{method} {name}_coef"] = coef_block / std

        df[f"{method} intercept"] = intercept
        df[f"{method} regression"] = np.where(df["test"] == 0, predict, np.nan)
        df[f"{method} prediction"] = np.where(df["test"] == 1, predict, np.nan)
        df.loc[df["test"] == 0, f"{method} cv"] = original_array

    feature_names = "_".join(names)
    df.to_pickle(path.replace(".pkl", f"_{feature_names}_regression.pkl"))
    grid.to_csv(path.replace(".pkl", f"_{feature_names}_regression.csv"))


def generate_combinations(elements: Iterable[str]) -> List[List[str]]:
    """
    Generate all non-empty combinations of the given elements.

    Parameters
    ----------
    elements : iterable of str
        A sequence (or any iterable) of feature names.

    Returns
    -------
    list of list of str
        List of combinations. Each element is a list of strings representing
        one subset of the input `elements`, with length ranging from 1 to
        ``len(elements)``.

    Examples
    --------
    >>> generate_combinations(["a", "b"])
    [['a'], ['b'], ['a', 'b']]
    """
    elems = list(elements)
    result: List[List[str]] = []
    for r in range(1, len(elems) + 1):
        result.extend([list(c) for c in combinations(elems, r)])
    return result


if __name__ == "__main__":
    regression_("data/data.pkl", ["electronic", "electrostatic", "lumo"])
    # for feat, path in product(
    #     generate_combinations(["electronic", "electrostatic", "lumo"]),
    #     [
    #         "data/data.pkl",
    #     ],
    # ):
    #     regression_(path, feat)
