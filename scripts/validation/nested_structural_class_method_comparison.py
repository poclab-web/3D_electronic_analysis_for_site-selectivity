#!/usr/bin/env python3
"""Compare regression methods using strict nested structural-class CV.

For every outer fold, one complete A-E structural class is withheld. Feature
construction, scaling, and method-specific hyperparameter selection are then
repeated using only the remaining four classes. Hyperparameters are selected
by observation-level inner LOOCV. The finalized Lasso grid is used so its
pooled predictions match the structural-class validation reported in the SI.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn import __version__ as sklearn_version
from sklearn.cross_decomposition import PLSRegression
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import ElasticNet, Lasso, OrthogonalMatchingPursuit, Ridge
from sklearn.metrics import mean_squared_error, r2_score


CLASSES = tuple("ABCDE")
METHODS = ("PLS", "Ridge", "Elastic Net", "Lasso", "OMP")


@dataclass(frozen=True)
class Candidate:
    family: str
    label: str
    parameters: tuple[float | int, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--lasso-reference",
        type=Path,
        help="Optional structural-class Lasso outer_predictions.csv to verify.",
    )
    return parser.parse_args()


def candidate_grid(final_lasso_alphas: tuple[float, ...]) -> list[Candidate]:
    candidates = [Candidate("Lasso", f"Lasso {alpha:g}", (alpha,)) for alpha in final_lasso_alphas]
    candidates += [Candidate("Ridge", f"Ridge {alpha:g}", (alpha,)) for alpha in (0.1, 0.3, 1.0, 3.0, 10.0)]
    candidates += [
        Candidate("Elastic Net", f"ElasticNet {alpha:g} {ratio:g}", (alpha, ratio))
        for alpha in (0.0008, 0.00114505, 0.0016, 0.00225932721534, 0.00390625, 0.006, 0.0078125)
        for ratio in (0.5, 0.7, 0.9)
    ]
    candidates += [Candidate("PLS", f"PLS {components}", (components,)) for components in range(2, 7)]
    candidates += [Candidate("OMP", f"OMP {count}", (count,)) for count in (5, 7, 9, 11, 13, 15, 17)]
    return candidates


def fit_predict(
    candidate: Candidate,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    *,
    lasso_max_iter: int = 10000,
    lasso_tol: float = 1.0e-4,
) -> tuple[np.ndarray, int, int]:
    if candidate.family == "PLS":
        model = PLSRegression(n_components=int(candidate.parameters[0]), scale=False)
    elif candidate.family == "Ridge":
        model = Ridge(alpha=float(candidate.parameters[0]), fit_intercept=True, max_iter=10000)
    elif candidate.family == "Lasso":
        model = Lasso(
            alpha=float(candidate.parameters[0]),
            fit_intercept=True,
            max_iter=lasso_max_iter,
            tol=lasso_tol,
        )
    elif candidate.family == "Elastic Net":
        model = ElasticNet(
            alpha=float(candidate.parameters[0]),
            l1_ratio=float(candidate.parameters[1]),
            fit_intercept=True,
            max_iter=10000,
        )
    elif candidate.family == "OMP":
        model = OrthogonalMatchingPursuit(
            n_nonzero_coefs=int(candidate.parameters[0]),
            fit_intercept=True,
        )
    else:
        raise ValueError(candidate.family)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(x_train, y_train)
    warning_count = sum(issubclass(item.category, ConvergenceWarning) for item in caught)
    nonzero = int(np.count_nonzero(np.asarray(model.coef_))) if hasattr(model, "coef_") else 0
    predicted = np.asarray(model.predict(x_test), dtype=float).reshape(-1)
    return predicted, warning_count, nonzero


def verify_lasso_reference(predictions: pd.DataFrame, reference_path: Path) -> None:
    reference = pd.read_csv(reference_path).sort_values("row_index")
    lasso = predictions.loc[predictions["method"].eq("Lasso")].sort_values("row_index")
    if len(reference) != 83 or len(lasso) != 83:
        raise RuntimeError("Expected 83 finalized Lasso predictions.")
    if not np.array_equal(reference["row_index"].to_numpy(), lasso["row_index"].to_numpy()):
        raise RuntimeError("Lasso row coverage does not match the supplied reference.")
    if not np.allclose(
        reference["predicted_ddg_kcal_mol"].to_numpy(dtype=float),
        lasso["predicted_ddg_kcal_mol"].to_numpy(dtype=float),
        rtol=0,
        atol=1e-12,
    ):
        raise RuntimeError("Nested method-comparison Lasso predictions do not match Panel B.")


def main() -> None:
    args = parse_args()
    repository_root = args.repository_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(repository_root / "libs"))
    import current_model as cm

    cm.verify_input_manifest()
    payload = cm.ensure_frozen_inputs()
    metadata = payload["meta"].copy()
    raw = {block: np.asarray(payload["raw_blocks"][block], dtype=float) for block in cm.BLOCKS}
    coordinates = {block: np.asarray(payload["coords"][block], dtype=int) for block in cm.BLOCKS}
    response = pd.to_numeric(metadata[cm.TARGET], errors="coerce").to_numpy(dtype=float)
    train = pd.read_csv(cm.TRAIN_ROWS_PATH)["row_index"].to_numpy(dtype=int)
    class_labels = metadata.loc[train, "entry"].astype(str).str.extract(r"^([A-E])")[0].to_numpy()
    if pd.isna(class_labels).any() or tuple(sorted(set(class_labels))) != CLASSES:
        raise RuntimeError("Training entries do not define the expected A-E classes.")

    final_lasso_alphas = tuple(float(value) for value in cm.ALPHAS)
    candidates = candidate_grid(final_lasso_alphas)
    candidates_by_method = {
        method: [candidate for candidate in candidates if candidate.family == method]
        for method in METHODS
    }
    prediction_rows: list[dict[str, object]] = []
    selection_rows: list[dict[str, object]] = []

    for heldout_class in CLASSES:
        outer_train = train[class_labels != heldout_class]
        outer_test = train[class_labels == heldout_class]
        inner_predictions = {
            candidate.label: np.empty(len(outer_train), dtype=float)
            for candidate in candidates
        }
        inner_warning_counts = {candidate.label: 0 for candidate in candidates}

        for local_index, inner_holdout in enumerate(outer_train):
            inner_train = outer_train[outer_train != inner_holdout]
            x_inner, _, _ = cm.build_features(raw, coordinates, inner_train)
            for candidate in candidates:
                if candidate.family == "Lasso":
                    continue
                estimate, warning_count, _ = fit_predict(
                    candidate,
                    x_inner[inner_train],
                    response[inner_train],
                    x_inner[[inner_holdout]],
                )
                inner_predictions[candidate.label][local_index] = estimate[0]
                inner_warning_counts[candidate.label] += warning_count
            lasso_estimates = cm.alpha_path_predictions(
                x_inner[inner_train],
                response[inner_train],
                x_inner[[inner_holdout]],
            ).ravel()
            for alpha, estimate in zip(final_lasso_alphas, lasso_estimates):
                inner_predictions[f"Lasso {alpha:g}"][local_index] = estimate

        x_outer, _, _ = cm.build_features(raw, coordinates, outer_train)
        for method in METHODS:
            scored = []
            for candidate in candidates_by_method[method]:
                inner_rmse = math.sqrt(
                    mean_squared_error(response[outer_train], inner_predictions[candidate.label])
                )
                scored.append((inner_rmse, candidate.label, candidate))
            inner_rmse, _, selected = min(scored, key=lambda row: (row[0], row[1]))
            fit_options = {}
            if method == "Lasso":
                fit_options = {
                    "lasso_max_iter": cm.LASSO_FIT_MAX_ITER,
                    "lasso_tol": cm.LASSO_FIT_TOL,
                }
            outer_prediction, outer_warning_count, nonzero = fit_predict(
                selected,
                x_outer[outer_train],
                response[outer_train],
                x_outer[outer_test],
                **fit_options,
            )
            selection_rows.append(
                {
                    "heldout_class": heldout_class,
                    "method": method,
                    "selected_candidate": selected.label,
                    "inner_rmse_kcal_mol": float(inner_rmse),
                    "nonzero_features": nonzero,
                    "inner_convergence_warning_count": inner_warning_counts[selected.label],
                    "outer_convergence_warning_count": outer_warning_count,
                    "train_n": int(len(outer_train)),
                    "test_n": int(len(outer_test)),
                }
            )
            for row_index, estimate in zip(outer_test, outer_prediction):
                observed = float(response[row_index])
                prediction_rows.append(
                    {
                        "heldout_class": heldout_class,
                        "method": method,
                        "row_index": int(row_index),
                        "entry": str(metadata.at[row_index, "entry"]),
                        "name": str(metadata.at[row_index, "name"]),
                        "observed_ddg_kcal_mol": observed,
                        "predicted_ddg_kcal_mol": float(estimate),
                        "residual_kcal_mol": float(estimate - observed),
                    }
                )
        print(f"Completed outer class {heldout_class}", flush=True)

    predictions = pd.DataFrame(prediction_rows)
    selections = pd.DataFrame(selection_rows)
    metrics_rows = []
    for method in METHODS:
        subset = predictions.loc[predictions["method"].eq(method)].sort_values("row_index")
        if len(subset) != 83 or subset["row_index"].nunique() != 83:
            raise RuntimeError(f"{method}: incomplete out-of-fold coverage.")
        metrics_rows.append(
            {
                "method": method,
                "n": int(len(subset)),
                "nested_oof_r2": float(
                    r2_score(subset["observed_ddg_kcal_mol"], subset["predicted_ddg_kcal_mol"])
                ),
                "nested_oof_rmse_kcal_mol": float(
                    math.sqrt(
                        mean_squared_error(
                            subset["observed_ddg_kcal_mol"],
                            subset["predicted_ddg_kcal_mol"],
                        )
                    )
                ),
            }
        )
    metrics = pd.DataFrame(metrics_rows)

    if args.lasso_reference:
        verify_lasso_reference(predictions, args.lasso_reference.resolve())

    predictions.to_csv(output_dir / "nested_method_predictions.csv", index=False)
    selections.to_csv(output_dir / "nested_method_selections.csv", index=False)
    metrics.to_csv(output_dir / "nested_method_metrics.csv", index=False)
    audit = {
        "analysis": "strict nested leave-one-structural-class-out method comparison",
        "outer_classes": {name: int(np.sum(class_labels == name)) for name in CLASSES},
        "inner_selection": "observation-level LOOCV within each outer training fold",
        "lasso_alphas": list(final_lasso_alphas),
        "method_candidates": {
            method: [candidate.label for candidate in candidates_by_method[method]]
            for method in METHODS
        },
        "lasso_reference_verified": bool(args.lasso_reference),
        "software": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn_version,
        },
        "metrics": metrics.to_dict(orient="records"),
    }
    (output_dir / "calculation_design_and_audit.json").write_text(
        json.dumps(audit, indent=2) + "\n",
        encoding="utf-8",
    )
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
