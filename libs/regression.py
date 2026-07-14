from __future__ import annotations

import os
import re
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

INPUT_DATA_PATH = "data/data.pkl"
PREFERRED_REGRESSION_METHOD = os.getenv(
    "PREFERRED_REGRESSION_METHOD",
    "Lasso 0.006",
)

# Manuscript model selected from the 2026-06-29 search. H1 is treated as an
# additional training substrate, while the remaining holdout monoketones stay
# outside the training set.
EXTRA_TRAIN_ENTRIES = tuple(
    entry.strip()
    for entry in os.getenv("REGRESSION_EXTRA_TRAIN_ENTRIES", "H1").split(",")
    if entry.strip()
)

# Spatial prefilter + variance preselection selected for the current manuscript
# model: electronic/electrostatic box plus compact LUMO box, followed by
# block-wise variance top-k selection.
REGRESSION_USE_GRID_PREFILTER = os.getenv(
    "REGRESSION_USE_GRID_PREFILTER",
    "1",
).strip().lower() not in {"0", "false", "no", "off"}
GRID_BOUNDS = {
    "electronic": (-6, 2, 1, 6, -4, 3),
    "electrostatic": (-6, 2, 1, 6, -4, 3),
    "lumo": (-4, 1, 1, 4, -1, 2),
}
BLOCK_VARIANCE_TOP = {
    "electronic": 240,
    "electrostatic": 200,
    "lumo": 50,
}


def _preferred_regression_methods() -> List[str]:
    preferred = PREFERRED_REGRESSION_METHOD.strip()
    if preferred and preferred.lower() not in {"none", "auto"}:
        return [preferred]
    return []


def _positive_int_from_env(name: str, default: int) -> int:
    """Read a positive integer from environment; fallback to default."""
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value > 0 else default


NUM_WORKERS = _positive_int_from_env("REGRESSION_NUM_WORKERS", os.cpu_count() or 1)


def _parse_coord(column: str) -> tuple[int, int, int]:
    return tuple(map(int, re.findall(r"[+-]?\d+", column)))  # type: ignore[return-value]


def _in_bounds(coords: np.ndarray, bounds: tuple[int, int, int, int, int, int]) -> np.ndarray:
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    return (
        (coords[:, 0] >= xmin)
        & (coords[:, 0] <= xmax)
        & (coords[:, 1] >= ymin)
        & (coords[:, 1] <= ymax)
        & (coords[:, 2] >= zmin)
        & (coords[:, 2] <= zmax)
    )


def _select_feature_indices(
    name: str,
    columns: Sequence[str],
    train_scaled: np.ndarray,
) -> np.ndarray:
    """Select feature columns for one descriptor block."""
    if not REGRESSION_USE_GRID_PREFILTER:
        return np.arange(len(columns), dtype=int)

    coords = np.asarray([_parse_coord(column) for column in columns], dtype=int)
    bounds = GRID_BOUNDS[name]
    candidate_indices = np.where(_in_bounds(coords, bounds))[0]

    keep_n = BLOCK_VARIANCE_TOP.get(name)
    if keep_n is None or keep_n >= len(candidate_indices):
        return candidate_indices

    variances = np.var(train_scaled[:, candidate_indices], axis=0)
    order = np.argsort(variances)[::-1]
    keep_local = np.sort(order[:keep_n])
    return candidate_indices[keep_local]


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
    - Multiprocessing worker count is controlled by ``NUM_WORKERS``
      (override via ``REGRESSION_NUM_WORKERS``).
    """
    print(path)
    df = pd.read_pickle(path)
    df = df.copy()
    if EXTRA_TRAIN_ENTRIES:
        extra_train_mask = df["entry"].astype(str).isin(EXTRA_TRAIN_ENTRIES)
        df.loc[extra_train_mask, "test"] = 0

    has_y = pd.to_numeric(df["ΔΔG.expt."], errors="coerce").notna()
    train_mask = (df["test"] == 0) & has_y
    df_train = df[train_mask]

    y_train = df_train["ΔΔG.expt."].values
    y = df["ΔΔG.expt."].values

    trains: List[np.ndarray] = []
    train_tests: List[np.ndarray] = []
    stds: List[float] = []
    selected_indices_by_name: List[np.ndarray] = []
    block_columns_by_name: List[List[str]] = []

    # --- build feature blocks ---
    for name in names:
        columns = df.filter(like=f"{name}_fold").columns.tolist()
        train = df_train[columns].to_numpy()
        std = np.std(train)
        # std = np.linalg.norm(train)  # /np.size(train)
        train_test = df[columns].to_numpy()
        # train -= average
        # train_test -= average

        train /= std
        train_test /= std

        selected_indices = _select_feature_indices(name, columns, train)

        trains.append(train[:, selected_indices])
        train_tests.append(train_test[:, selected_indices])
        stds.append(std)
        selected_indices_by_name.append(selected_indices)
        block_columns_by_name.append(columns)

    # --- define methods ---
    methods: List[str] = _preferred_regression_methods()
    if not methods:
        lasso_alphas = np.unique(
            np.r_[
                np.logspace(np.log2(5e-4), np.log2(1.2e-2), 18, base=2),
                [
                    0.0009765625,
                    0.001381067932,
                    0.001953125,
                    0.00225932721534,
                    0.00390625,
                    0.006,
                    0.0078125,
                ],
            ]
        )
        for alpha in lasso_alphas:
            methods.append(f"Lasso {alpha}")

        for alpha in [0.1, 0.3, 1.0, 3.0, 10.0]:
            methods.append(f"Ridge {alpha}")

        elastic_alphas = [
            0.0008,
            0.00114505,
            0.0016,
            0.00225932721534,
            0.00390625,
            0.006,
            0.0078125,
        ]
        for alpha, l1ratio in product(elastic_alphas, [0.5, 0.7, 0.9]):
            methods.append(f"ElasticNet {alpha} {l1ratio}")

        for n_components in range(2, 7):
            methods.append(f"PLS {n_components}")

        for n_nonzero in [5, 7, 9, 11, 13, 15, 17]:
            methods.append(f"OMP {n_nonzero}")
    methods = list(dict.fromkeys(methods))

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

    regression_args = [
        (X_train_full, X_full, y_train, y, method)
        for method in methods
    ]
    if NUM_WORKERS <= 1:
        results = [regression_parallel(args) for args in regression_args]
    else:
        with Pool(NUM_WORKERS) as pool:
            results = list(pool.imap_unordered(regression_parallel, regression_args))

    # --- collect results ---
    for result in results:
        method, coef, intercept, predict, original_array = result
        print(method)

        # split coefficient vector back into blocks for each name and place
        # selected coefficients back on the full coordinate grid. Unselected
        # features are kept as zero coefficients so downstream contribution
        # plotting can continue to use the original full grid.
        coef_blocks: List[np.ndarray] = []
        start = 0
        for selected_indices in selected_indices_by_name:
            stop = start + len(selected_indices)
            coef_blocks.append(coef[start:stop])
            start = stop
        for name, std, coef_block, selected_indices, columns in zip(
            names,
            stds,
            coef_blocks,
            selected_indices_by_name,
            block_columns_by_name,
        ):
            full_coef = np.zeros(len(columns), dtype=float)
            full_coef[selected_indices] = coef_block / std
            grid[f"{method} {name}_coef"] = full_coef

        df[f"{method} intercept"] = intercept
        df[f"{method} regression"] = np.where(train_mask, predict, np.nan)
        df[f"{method} prediction"] = np.where(~train_mask, predict, np.nan)
        df.loc[train_mask, f"{method} cv"] = original_array

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
    regression_(INPUT_DATA_PATH, ["electronic", "electrostatic", "lumo"])
