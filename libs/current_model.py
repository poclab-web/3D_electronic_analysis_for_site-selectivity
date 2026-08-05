"""Reproduce the current three-block site-selectivity reference model.

The model uses electronic, electrostatic, and HOMO-gap damped projected C=O
pi* grids.  Every block is aligned to the same 4,927 pre-cut coordinates,
scaled by one training-fold SD before spatial selection, and represented by
105 compact-grid values plus max/min summaries (321 features in total).  Lasso
alpha is selected independently inside every outer LOOCV fold.

Run from the repository root::

    OMP_NUM_THREADS=1 python libs/current_model.py --workers 20 \
        --no-excel-refresh --skip-contribution-cubes
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, lasso_path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "current_model"
INPUT_DIR = DATA_DIR / "inputs"
VALIDATION_DIR = ROOT / "data" / "validation" / "current_model"
RESULTS_DIR = DATA_DIR / "results"
MODEL_RESULTS_DIR = RESULTS_DIR / "model"
DIKETONE_RESULTS_DIR = RESULTS_DIR / "diketones"
PUBLICATION_TABLES_DIR = RESULTS_DIR / "publication_tables"
AUDIT_DIR = DATA_DIR / "audits"
MODEL_FIGURE_DIR = VALIDATION_DIR / "model"
DIKETONE_FIGURE_DIR = VALIDATION_DIR / "diketones"
CONTRIBUTION_FIGURE_DIR = VALIDATION_DIR / "contribution_series"
COMPARATOR_FIGURE_DIR = VALIDATION_DIR / "comparators"
MODEL_ARRAYS_PATH = INPUT_DIR / "model_arrays.npz"
MODEL_METADATA_PATH = INPUT_DIR / "model_metadata.csv"
MODEL_PROVENANCE_PATH = INPUT_DIR / "model_provenance.json"
INPUT_MANIFEST_PATH = INPUT_DIR / "input_manifest.csv"
DISPLAY_GEOMETRIES_PATH = INPUT_DIR / "display_geometries.json"
TRAIN_ROWS_PATH = INPUT_DIR / "train_rows.csv"
ACTIVE_EXCEL = ROOT / "data" / "Details_of_experimental_results.xlsx"
ORBITAL_CACHE_PATH = INPUT_DIR / "projected_orbital_fullgrid_2bohr.npz"
ORBITAL_MANIFEST_PATH = INPUT_DIR / "projected_orbital_manifest.csv"
ORBITAL_FREE_SUMMARY_PATH = DATA_DIR / "comparators" / "orbital_free_summary.csv"
TARGET = "ΔΔG.expt."
BLOCKS = ("electronic", "electrostatic", "orbital")
EE_BOUNDS = (-5, 2, 1, 3, -2, 3)
ALPHAS = (1.0, 0.1, 0.01, 0.001)
EXPECTED_FULL_GRID_N = 4927
EXPECTED_SELECTED_GRID_N = 105
EXPECTED_FEATURE_N = 321
LASSO_FIT_MAX_ITER = 200000
LASSO_FIT_TOL = 1.0e-6
LASSO_PATH_MAX_ITER = 200000
LASSO_PATH_TOL = 1.0e-4
DESCRIPTOR_VERSION = "projected_co_pi_star_fullgrid_scaled_es_zero_pad_v1"
DIKETONE_ENTRY_SUFFIXES = (
    "1", "2", "3", "4", "13", "14", "23", "24", "31", "32", "41", "42"
)


def _sha256(path: Path) -> str:
    """Return the SHA-256 digest of a frozen input file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_input_manifest() -> pd.DataFrame:
    """Verify byte sizes and SHA-256 hashes for every listed frozen input.

    Manifest paths must be relative to :data:`INPUT_DIR` and may not traverse
    outside it.  A mismatch is fatal because even a subtle descriptor change
    can alter feature scaling, Lasso selection, and the reported validation
    metrics.
    """
    if not INPUT_MANIFEST_PATH.exists():
        raise FileNotFoundError(f"Frozen-input manifest is missing: {INPUT_MANIFEST_PATH}")
    manifest = pd.read_csv(INPUT_MANIFEST_PATH)
    expected_columns = ["path", "size_bytes", "sha256"]
    if manifest.columns.tolist() != expected_columns:
        raise ValueError(
            f"Unexpected input-manifest columns {manifest.columns.tolist()}; "
            f"expected {expected_columns}."
        )
    if manifest["path"].duplicated().any():
        duplicates = manifest.loc[manifest["path"].duplicated(), "path"].tolist()
        raise ValueError(f"Duplicate frozen-input manifest paths: {duplicates}")
    errors: list[str] = []
    for record in manifest.to_dict("records"):
        relative = Path(str(record["path"]))
        if relative.is_absolute() or ".." in relative.parts:
            errors.append(f"unsafe path: {relative}")
            continue
        path = INPUT_DIR / relative
        if not path.is_file():
            errors.append(f"missing: {relative}")
            continue
        actual_size = path.stat().st_size
        expected_size = int(record["size_bytes"])
        if actual_size != expected_size:
            errors.append(
                f"size mismatch: {relative} ({actual_size} != {expected_size})"
            )
            continue
        actual_digest = _sha256(path)
        expected_digest = str(record["sha256"]).lower()
        if actual_digest != expected_digest:
            errors.append(f"SHA-256 mismatch: {relative}")
    if errors:
        raise ValueError("Frozen-input verification failed:\n- " + "\n- ".join(errors))
    return manifest


def load_frozen_inputs() -> dict[str, object]:
    """Load the portable metadata, descriptor arrays, and provenance record."""
    meta = pd.read_csv(MODEL_METADATA_PATH)
    with np.load(MODEL_ARRAYS_PATH, allow_pickle=False) as archive:
        expected_keys = {
            *(f"raw_{block}" for block in BLOCKS),
            *(f"coords_{block}" for block in BLOCKS),
        }
        if set(archive.files) != expected_keys:
            raise ValueError(
                f"Unexpected keys in {MODEL_ARRAYS_PATH}: {sorted(archive.files)}"
            )
        raw = {
            block: np.asarray(archive[f"raw_{block}"], dtype=float)
            for block in BLOCKS
        }
        coords = {
            block: np.asarray(archive[f"coords_{block}"], dtype=int)
            for block in BLOCKS
        }
    provenance = json.loads(MODEL_PROVENANCE_PATH.read_text(encoding="utf-8"))
    return {
        "meta": meta,
        "raw_blocks": raw,
        "coords": coords,
        "descriptor_version": provenance.get("descriptor_version"),
        "provenance": provenance,
    }


def ensure_frozen_inputs() -> dict[str, object]:
    """Validate and return the self-contained current-model input package."""
    required = (
        MODEL_ARRAYS_PATH,
        MODEL_METADATA_PATH,
        MODEL_PROVENANCE_PATH,
        INPUT_MANIFEST_PATH,
        DISPLAY_GEOMETRIES_PATH,
        TRAIN_ROWS_PATH,
        ORBITAL_CACHE_PATH,
        ORBITAL_MANIFEST_PATH,
    )
    missing = [path for path in required if not path.exists()]
    if missing:
        joined = "\n".join(f"- {path}" for path in missing)
        raise FileNotFoundError(f"Current-model inputs are incomplete:\n{joined}")
    verify_input_manifest()
    train_rows = pd.read_csv(TRAIN_ROWS_PATH)
    if len(train_rows) != 83:
        raise ValueError(f"Expected 83 training rows, found {len(train_rows)}.")
    required_train_columns = {"row_index", "entry", "name", "InChIKey", "ddg"}
    if not required_train_columns.issubset(train_rows.columns):
        raise ValueError(
            f"Training manifest is missing columns: "
            f"{sorted(required_train_columns - set(train_rows.columns))}"
        )
    payload = load_frozen_inputs()
    meta = payload["meta"]
    row_indices = train_rows["row_index"].to_numpy(dtype=int)
    if np.any(row_indices < 0) or np.any(row_indices >= len(meta)):
        raise ValueError("Training row indices fall outside the frozen metadata table.")
    frozen_train_keys = meta.loc[row_indices, "InChIKey"].astype(str).to_numpy()
    manifest_train_keys = train_rows["InChIKey"].astype(str).to_numpy()
    if not np.array_equal(frozen_train_keys, manifest_train_keys):
        raise ValueError("Training identities do not align with frozen metadata rows.")
    if len(set(manifest_train_keys)) != len(manifest_train_keys):
        raise ValueError("Training InChIKeys must be unique.")
    bundle_ok = (
        payload.get("descriptor_version") == DESCRIPTOR_VERSION
        and set(payload.get("raw_blocks", {})) == set(BLOCKS)
        and all(
            np.asarray(payload["raw_blocks"][block]).shape
            == (len(meta), EXPECTED_FULL_GRID_N)
            for block in BLOCKS
        )
        and all(
            np.asarray(payload["coords"][block]).shape == (EXPECTED_FULL_GRID_N, 3)
            for block in BLOCKS
        )
        and all(
            np.isfinite(np.asarray(payload["raw_blocks"][block], dtype=float)).all()
            for block in BLOCKS
        )
    )
    if not bundle_ok:
        raise ValueError(f"Current-model descriptor inputs are invalid: {INPUT_DIR}")
    return payload


def refresh_inputs_from_excel(
    payload: dict[str, object],
    train_rows: pd.DataFrame,
) -> tuple[dict[str, object], np.ndarray, dict[str, object]]:
    """Refresh labels and responses by molecular identity, never by row number.

    The expensive grid arrays remain attached to their InChIKeys.  This makes
    entry renumbering harmless while still applying corrected experimental
    values from the active workbook.  Rows removed from Excel are dropped only
    when they are not members of the frozen training set.  All changes occur
    in memory; immutable files below :data:`INPUT_DIR` are never rewritten.
    """
    from current_model_support.workbook import (  # noqa: PLC0415
        load_experimental_dataset,
    )

    old_meta = payload["meta"].copy().reset_index(drop=True)
    old_train_rows = train_rows["row_index"].to_numpy(dtype=int)
    train_keys = old_meta.loc[old_train_rows, "InChIKey"].astype(str).tolist()
    if len(train_keys) != len(set(train_keys)):
        raise ValueError("Training InChIKeys must be unique before Excel refresh.")

    # The active workbook is the authority for current entry labels. Historical
    # holdout aliases such as H1 and Dxx must not leak into manuscript figures.
    source = load_experimental_dataset(
        ACTIVE_EXCEL,
        apply_overrides=False,
    ).copy().reset_index(drop=True)
    source["InChIKey"] = source["InChIKey"].astype(str)
    source_by_key = source.drop_duplicates("InChIKey", keep="first").set_index("InChIKey")
    old_keys = old_meta["InChIKey"].astype(str)
    matched = old_keys.isin(source_by_key.index).to_numpy()
    missing_training = sorted(set(train_keys) - set(source_by_key.index))
    if missing_training:
        raise ValueError(f"Active Excel is missing training molecules: {missing_training}")

    retained_old_indices = np.flatnonzero(matched)
    refreshed_meta = old_meta.iloc[retained_old_indices].copy().reset_index(drop=True)
    metadata_columns = (
        "entry", "name", "SMILES", "InChIKey", "temperature", TARGET, "test"
    )
    changed_rows: list[dict[str, object]] = []
    for new_index, old_index in enumerate(retained_old_indices):
        key = str(old_meta.at[old_index, "InChIKey"])
        source_row = source_by_key.loc[key]
        before_entry = str(old_meta.at[old_index, "entry"])
        before_name = str(old_meta.at[old_index, "name"])
        before_target = float(old_meta.at[old_index, TARGET])
        after_target = source_row[TARGET]
        for column in metadata_columns:
            refreshed_meta.at[new_index, column] = (
                key if column == "InChIKey" else source_row[column]
            )
        target_changed = (
            pd.notna(after_target)
            and (not np.isclose(before_target, float(after_target), rtol=0.0, atol=1e-12))
        )
        if (
            before_entry != str(source_row["entry"])
            or before_name != str(source_row["name"])
            or target_changed
        ):
            changed_rows.append(
                {
                    "InChIKey": key,
                    "old_entry": before_entry,
                    "new_entry": source_row["entry"],
                    "old_name": before_name,
                    "new_name": source_row["name"],
                    "old_ddg": before_target,
                    "new_ddg": after_target,
                    "ddg_changed": bool(target_changed),
                }
            )

    refreshed_raw = {
        block: np.asarray(payload["raw_blocks"][block], dtype=float)[retained_old_indices]
        for block in BLOCKS
    }
    refreshed_coords = {
        block: np.asarray(payload["coords"][block], dtype=int) for block in BLOCKS
    }
    key_to_new_index = {
        str(key): index for index, key in enumerate(refreshed_meta["InChIKey"])
    }
    refreshed_train = np.asarray([key_to_new_index[key] for key in train_keys], dtype=int)
    if refreshed_meta.loc[refreshed_train, TARGET].isna().any():
        raise ValueError("A refreshed training row has no experimental response.")

    refreshed_payload = {
        **payload,
        "meta": refreshed_meta,
        "raw_blocks": refreshed_raw,
        "coords": refreshed_coords,
    }
    change_columns = (
        "InChIKey",
        "old_entry",
        "new_entry",
        "old_name",
        "new_name",
        "old_ddg",
        "new_ddg",
        "ddg_changed",
    )
    pd.DataFrame(changed_rows, columns=change_columns).to_csv(
        AUDIT_DIR / "excel_refresh_changes.csv", index=False
    )

    removed = old_meta.loc[~matched, ["entry", "name", "InChIKey"]].copy()
    removed.to_csv(AUDIT_DIR / "excel_refresh_removed_rows.csv", index=False)
    source_keys = set(source["InChIKey"].astype(str))
    new_external = source.loc[
        ~source["InChIKey"].astype(str).isin(set(old_keys)),
        ["entry", "name", "InChIKey", "SMILES", "temperature", TARGET, "test"],
    ].copy()
    new_external.to_csv(AUDIT_DIR / "excel_refresh_new_external_rows.csv", index=False)
    audit = {
        "excel": str(ACTIVE_EXCEL.relative_to(ROOT)),
        "old_bundle_n": int(len(old_meta)),
        "refreshed_bundle_n": int(len(refreshed_meta)),
        "training_n": int(len(refreshed_train)),
        "removed_nontraining_n": int(len(removed)),
        "changed_metadata_n": int(len(changed_rows)),
        "new_external_rows_n": int(len(new_external)),
        "new_external_unique_molecules_n": int(
            len(source_keys - set(refreshed_meta["InChIKey"].astype(str)))
        ),
        "feature_blocks": list(BLOCKS),
    }
    (AUDIT_DIR / "excel_refresh_audit.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8"
    )
    return refreshed_payload, refreshed_train, audit


def in_bounds(coords: np.ndarray) -> np.ndarray:
    """Return the mask for the adopted 105-cell compact spatial domain."""
    xmin, xmax, ymin, ymax, zmin, zmax = EE_BOUNDS
    return (
        (coords[:, 0] >= xmin)
        & (coords[:, 0] <= xmax)
        & (coords[:, 1] >= ymin)
        & (coords[:, 1] <= ymax)
        & (coords[:, 2] >= zmin)
        & (coords[:, 2] <= zmax)
    )


def build_features(
    raw_blocks: dict[str, np.ndarray],
    coords_by_block: dict[str, np.ndarray],
    train: np.ndarray,
) -> tuple[np.ndarray, list[str], dict[str, np.ndarray]]:
    """Build 315 compact grids and six max/min summary features."""
    arrays: list[np.ndarray] = []
    names: list[str] = []
    masks: dict[str, np.ndarray] = {}
    # The grid blocks precede both summary blocks, matching the validated
    # reference feature order exactly.
    for block in BLOCKS:
        raw = raw_blocks[block]
        coords = coords_by_block[block]
        keep = in_bounds(coords)
        masks[block] = keep
        scale = float(np.std(raw[train]))
        if not np.isfinite(scale) or scale == 0.0:
            scale = 1.0
        arrays.append(raw[:, keep] / scale)
        names.extend(f"{block}:grid:{tuple(map(int, coord))}" for coord in coords[keep])
    for block in BLOCKS:
        raw = raw_blocks[block]
        keep = masks[block]
        summary = np.column_stack((np.max(raw[:, keep], axis=1), np.min(raw[:, keep], axis=1)))
        summary_scale = np.std(summary[train], axis=0)
        summary_scale[~np.isfinite(summary_scale) | (summary_scale == 0.0)] = 1.0
        arrays.append(summary / summary_scale)
        names.extend((f"{block}:summary:max", f"{block}:summary:min"))
    return np.concatenate(arrays, axis=1), names, masks


def alpha_path_predictions(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_pred: np.ndarray,
) -> np.ndarray:
    """Predict all systematic absolute-alpha candidates in one Lasso path."""
    x_mean = np.mean(x_train, axis=0)
    y_mean = float(np.mean(y_train))
    _, coefficients, _ = lasso_path(
        x_train - x_mean,
        y_train - y_mean,
        alphas=np.asarray(ALPHAS, dtype=float),
        max_iter=LASSO_PATH_MAX_ITER,
        tol=LASSO_PATH_TOL,
    )
    return (x_pred - x_mean) @ coefficients + y_mean


def inner_path_scores(
    raw: dict[str, np.ndarray],
    coords: dict[str, np.ndarray],
    y: np.ndarray,
    train: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return leakage-free inner-LOOCV predictions and RMSE for every alpha."""
    predictions = np.empty((len(ALPHAS), len(train)), dtype=float)
    for local, holdout in enumerate(train):
        fold_train = train[train != holdout]
        x, _, _ = build_features(raw, coords, fold_train)
        predictions[:, local] = alpha_path_predictions(
            x[fold_train], y[fold_train], x[[holdout]]
        ).ravel()
    rmse = np.sqrt(np.mean((predictions - y[train][None, :]) ** 2, axis=1))
    return predictions, rmse


def inner_best(
    raw: dict[str, np.ndarray],
    coords: dict[str, np.ndarray],
    y: np.ndarray,
    train: np.ndarray,
) -> tuple[float, float, float]:
    """Select alpha by leakage-free inner LOOCV and return alpha/RMSE/R2."""
    predictions, rmse = inner_path_scores(raw, coords, y, train)
    best = int(np.argmin(rmse))
    return float(ALPHAS[best]), float(rmse[best]), float(r2_score(y[train], predictions[best]))


def outer_task(
    fold_id: int,
    raw: dict[str, np.ndarray],
    coords: dict[str, np.ndarray],
    y: np.ndarray,
    train: np.ndarray,
    diketone: np.ndarray,
    meta: pd.DataFrame,
) -> dict[str, object]:
    """Fit one strict outer-LOOCV model and predict its held-out substrate."""
    holdout = int(train[fold_id])
    fold_train = train[train != holdout]
    alpha, inner_rmse, inner_r2 = inner_best(raw, coords, y, fold_train)
    x, names, _ = build_features(raw, coords, fold_train)
    model = Lasso(
        alpha=alpha,
        fit_intercept=True,
        max_iter=LASSO_FIT_MAX_ITER,
        tol=LASSO_FIT_TOL,
    )
    model.fit(x[fold_train], y[fold_train])
    return {
        "fold_id": fold_id,
        "holdout_index": holdout,
        "entry": str(meta.loc[holdout, "entry"]),
        "name": str(meta.loc[holdout, "name"]),
        "y_true": float(y[holdout]),
        "y_pred": float(model.predict(x[[holdout]])[0]),
        "selected_alpha": alpha,
        "inner_rmse": inner_rmse,
        "inner_r2": inner_r2,
        "nonzero_features": int(np.count_nonzero(model.coef_)),
        "orbital_nonzero_features": int(
            np.count_nonzero(
                model.coef_[np.asarray([name.startswith("orbital:") for name in names], dtype=bool)]
            )
        ),
        "diketone_prediction": model.predict(x[diketone]),
    }


def plot_yy(
    outer: pd.DataFrame,
    regression: pd.DataFrame,
    external: pd.DataFrame,
    path: Path,
    *,
    show_external: bool = True,
    show_legend: bool = True,
    transparent: bool = False,
) -> None:
    """Plot full-fit, nested-LOOCV, and excluded-substrate predictions."""
    values = np.r_[
        outer["y_true"],
        outer["y_pred"],
        regression[TARGET],
        regression["fulltrain_prediction"],
        external[TARGET],
        external["prediction"],
    ]
    low, high = float(np.nanmin(values) - 0.25), float(np.nanmax(values) + 0.25)
    fig, ax = plt.subplots(figsize=(4.3, 4.15))
    ax.scatter(
        regression[TARGET],
        regression["fulltrain_prediction"],
        color="black",
        s=22,
        alpha=0.72,
        linewidth=0,
        zorder=3,
        label=f"Regression (N={len(regression)})",
    )
    ax.scatter(
        outer["y_true"],
        outer["y_pred"],
        facecolors="none",
        edgecolors="#3d6f8e",
        s=46,
        linewidth=0.9,
        zorder=4,
        label=f"Nested LOOCV (N={len(outer)})",
    )
    if show_external and not external.empty:
        ax.scatter(
            external[TARGET], external["prediction"], color="#c53d3d",
            marker="X", s=58, edgecolor="black", linewidth=0.45, zorder=5,
            label=f"Excluded (N={len(external)})",
        )
    ax.plot([low, high], [low, high], color="0.55", linewidth=1.0, zorder=0)
    ax.set(
        xlim=(low, high),
        ylim=(low, high),
        aspect="equal",
        xlabel=r"Experimental $\Delta\Delta G^\ddagger$ [kcal/mol]",
        ylabel=r"Predicted $\Delta\Delta G^\ddagger$ [kcal/mol]",
    )
    ax.xaxis.label.set_size(12)
    ax.yaxis.label.set_size(12)
    ax.tick_params(axis="both", labelsize=10.5)
    regression_r2 = float(r2_score(regression[TARGET], regression["fulltrain_prediction"]))
    cv_r2 = float(r2_score(outer["y_true"], outer["y_pred"]))
    ax.text(
        0.04,
        0.96,
        rf"Regression $R^2$ = {regression_r2:.2f}" + "\n" + rf"Nested LOOCV $R^2$ = {cv_r2:.2f}",
        transform=ax.transAxes,
        va="top",
        fontsize=11.5,
    )
    if show_legend:
        ax.legend(frameon=False, fontsize=9.2, loc="lower right")
    ax.grid(False)
    fig.tight_layout()
    fig.savefig(path, dpi=500, transparent=transparent)
    plt.close(fig)


PRIMARY_DIKETONE_CHECKS = (
    ("a", "initial", "2"),
    ("b", "initial", "1"),
    ("c", "initial", "2"),
    ("d", "initial", "1"),
    ("e", "initial", "2"),
    ("f", "initial", "1"),
    ("a", "final", "2-4"),
    ("e", "final", "2-3"),
)


def save_outer_diketone_uncertainty(
    dike_matrix: np.ndarray,
    dike_entries: list[str],
    full_dike: pd.DataFrame,
    outer: pd.DataFrame,
    semiquant_exp8: object,
) -> dict[str, float]:
    """Summarize selectivity variation across the 83 outer-fold models."""

    def primary_value(prediction: dict[str, float], group: str, stage: str) -> tuple[float, bool]:
        """Return the requested semiquantitative percentage and top-match flag."""
        simulation = semiquant_exp8.simulate_full(prediction, group)
        if stage == "initial":
            row = semiquant_exp8.max_metrics(simulation, group)[1]
        else:
            row = semiquant_exp8.final_metrics(simulation, group)[0]
        return float(row["predicted_percent"]), bool(row["top_match"])

    full_prediction = dict(
        zip(full_dike["entry"].astype(str), full_dike["prediction"].astype(float))
    )
    model_rows: list[dict[str, object]] = []
    primary_values: list[dict[str, float]] = []
    for model_id, values in enumerate(dike_matrix):
        prediction = dict(zip(dike_entries, np.asarray(values, dtype=float)))
        _, details = semiquant_exp8.evaluate_predictions(f"outer_{model_id}", prediction)
        quantified = pd.DataFrame(details).dropna(subset=["observed_percent"])
        errors = quantified["abs_error_percent"].to_numpy(dtype=float)
        checks: list[bool] = []
        failed: list[str] = []
        row_values: dict[str, float] = {}
        for group, stage, expected in PRIMARY_DIKETONE_CHECKS:
            percent, match = primary_value(prediction, group, stage)
            checks.append(match)
            if not match:
                failed.append(f"{group}:{stage}:{expected}")
            row_values[f"{group}_{stage}"] = percent
        primary_values.append(row_values)
        model_rows.append(
            {
                "outer_model": model_id,
                "holdout_entry": str(outer.iloc[model_id]["entry"]),
                "holdout_name": str(outer.iloc[model_id]["name"]),
                "checks_passed": int(sum(checks)),
                "all_8_correct": bool(all(checks)),
                "failed_checks": ";".join(failed),
                "semiquant_rmse_percent": float(np.sqrt(np.mean(errors**2))),
                "semiquant_mae_percent": float(np.mean(errors)),
            }
        )

    primary_frame = pd.DataFrame(primary_values)
    primary_rows: list[dict[str, object]] = []
    for group, stage, expected in PRIMARY_DIKETONE_CHECKS:
        key = f"{group}_{stage}"
        values = primary_frame[key].to_numpy(dtype=float)
        matches = [
            primary_value(dict(zip(dike_entries, row)), group, stage)[1]
            for row in dike_matrix
        ]
        full_percent, full_match = primary_value(full_prediction, group, stage)
        observed = (
            semiquant_exp8.MAX_TARGETS[group]["dr_percent"] if stage == "initial" else np.nan
        )
        primary_rows.append(
            {
                "group": group,
                "stage": stage,
                "expected": expected,
                "observed_percent": observed,
                "fulltrain_percent": full_percent,
                "fulltrain_top_match": full_match,
                "outer83_top_match_fraction": float(np.mean(matches)),
                "outer83_mean_percent": float(np.mean(values)),
                "outer83_median_percent": float(np.median(values)),
                "outer83_sd_percent": float(np.std(values, ddof=1)),
                "outer83_p16_percent": float(np.percentile(values, 16)),
                "outer83_p84_percent": float(np.percentile(values, 84)),
                "outer83_min_percent": float(np.min(values)),
                "outer83_max_percent": float(np.max(values)),
            }
        )
    primary = pd.DataFrame(primary_rows)
    model_metrics = pd.DataFrame(model_rows)
    metric_summary = {
        "outer_model_n": int(len(model_metrics)),
        "models_8_of_8": int(model_metrics["all_8_correct"].sum()),
        "models_8_of_8_fraction": float(model_metrics["all_8_correct"].mean()),
        "semiquant_rmse_median_percent": float(model_metrics["semiquant_rmse_percent"].median()),
        "semiquant_rmse_p16_percent": float(model_metrics["semiquant_rmse_percent"].quantile(0.16)),
        "semiquant_rmse_p84_percent": float(model_metrics["semiquant_rmse_percent"].quantile(0.84)),
        "semiquant_rmse_min_percent": float(model_metrics["semiquant_rmse_percent"].min()),
        "semiquant_rmse_max_percent": float(model_metrics["semiquant_rmse_percent"].max()),
    }
    primary.to_csv(
        DIKETONE_RESULTS_DIR / "diketone_primary8_outer83_68_interval.csv",
        index=False,
    )
    model_metrics.to_csv(
        DIKETONE_RESULTS_DIR / "diketone_outer83_model_metrics.csv", index=False
    )
    pd.DataFrame([metric_summary]).to_csv(
        DIKETONE_RESULTS_DIR / "diketone_outer83_uncertainty_summary.csv",
        index=False,
    )

    labels = [f"{row.group} {row.stage}" for row in primary.itertuples(index=False)]
    medians = primary["outer83_median_percent"].to_numpy(dtype=float)
    lower = medians - primary["outer83_p16_percent"].to_numpy(dtype=float)
    upper = primary["outer83_p84_percent"].to_numpy(dtype=float) - medians
    positions = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8.4, 4.5))
    ax.errorbar(
        positions,
        medians,
        yerr=np.vstack((lower, upper)),
        fmt="o",
        color="#3d6f8e",
        ecolor="#7294aa",
        capsize=4,
        label="Outer 83 median and 68% interval",
    )
    ax.scatter(
        positions,
        primary["fulltrain_percent"],
        marker="D",
        color="#b44b3f",
        s=30,
        label="Full-train model",
    )
    observed = primary["observed_percent"].to_numpy(dtype=float)
    has_observed = np.isfinite(observed)
    ax.scatter(
        positions[has_observed], observed[has_observed], marker="x", color="#222222", s=42,
        label="Experiment",
    )
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel("Reported selectivity / monoalcohol fraction (%)")
    ax.set_ylim(0, 106)
    ax.grid(axis="y", color="#dddddd", linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(
        DIKETONE_FIGURE_DIR / "diketone_primary8_outer83_68_interval.png",
        dpi=350,
    )
    plt.close(fig)
    return metric_summary


def save_model_comparison(current_summary: dict[str, object]) -> None:
    """Keep the adopted model and archived orbital-free comparator side by side."""
    archived_summary_path = ORBITAL_FREE_SUMMARY_PATH
    rows = [
        {
            "model": "Projected C=O pi*",
            "feature_n": int(current_summary["feature_n"]),
            "nested_outer_r2": float(current_summary["nested_outer_r2"]),
            "nested_outer_rmse_kcal_mol": float(current_summary["nested_outer_rmse"]),
            "diketone_top_checks": int(current_summary["diketone_top_checks"]),
            "diketone_check_n": int(current_summary["diketone_check_n"]),
            "diketone_semiquant_rmse_percent": float(
                current_summary["diketone_semiquant_rmse_percent"]
            ),
            "diketone_evaluation_series": "a-f (matched comparator scope)",
        }
    ]
    if archived_summary_path.exists():
        archived = pd.read_csv(archived_summary_path).iloc[0]
        rows.append(
            {
                "model": "Orbital-free EE/ES",
                "feature_n": int(archived["feature_n"]),
                "nested_outer_r2": float(archived["nested_outer_r2"]),
                "nested_outer_rmse_kcal_mol": float(archived["nested_outer_rmse"]),
                "diketone_top_checks": int(archived["diketone_top_checks"]),
                "diketone_check_n": int(archived["diketone_check_n"]),
                "diketone_semiquant_rmse_percent": float(
                    archived["diketone_semiquant_rmse_percent"]
                ),
                "diketone_evaluation_series": "a-f (matched comparator scope)",
            }
        )
    comparison = pd.DataFrame(rows)
    comparison.to_csv(
        RESULTS_DIR / "model_comparison_current_vs_orbital_free.csv", index=False
    )
    if len(comparison) != 2:
        return
    colors = ["#3d6f8e", "#8f8f8f"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4))
    axes[0].bar(comparison["model"], comparison["nested_outer_r2"], color=colors)
    axes[0].set_ylabel("Nested outer LOOCV $R^2$")
    axes[0].set_ylim(0, 1)
    axes[1].bar(
        comparison["model"], comparison["diketone_semiquant_rmse_percent"], color=colors
    )
    axes[1].set_ylabel("Diketone a-f RMSE [percentage points]")
    for ax in axes:
        ax.tick_params(axis="x", labelrotation=18)
        for label in ax.get_xticklabels():
            label.set_ha("right")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#dddddd", linewidth=0.6)
    fig.tight_layout()
    fig.savefig(COMPARATOR_FIGURE_DIR / "current_vs_orbital_free.png", dpi=350)
    plt.close(fig)


def main() -> None:
    """Validate frozen inputs, fit the model, and regenerate reported outputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workers",
        type=int,
        default=20,
        help="Parallel strict outer folds (1-20; default: 20).",
    )
    parser.add_argument(
        "--skip-nested",
        action="store_true",
        help="Reuse existing outer prediction files (not a full reproduction check).",
    )
    parser.add_argument(
        "--no-excel-refresh",
        action="store_true",
        help="Use installed metadata without synchronizing the active Excel file.",
    )
    parser.add_argument(
        "--skip-contribution-cubes",
        action="store_true",
        help="Skip the optional 498 display-cube exports.",
    )
    parser.add_argument(
        "--verify-inputs-only",
        action="store_true",
        help="Verify all frozen hashes and array schemas, then exit.",
    )
    args = parser.parse_args()
    workers = min(max(args.workers, 1), 20)

    payload = ensure_frozen_inputs()
    if args.verify_inputs_only:
        print(f"Verified {len(verify_input_manifest())} frozen input files.")
        return
    for directory in (
        DATA_DIR,
        VALIDATION_DIR,
        MODEL_RESULTS_DIR,
        DIKETONE_RESULTS_DIR,
        PUBLICATION_TABLES_DIR,
        AUDIT_DIR,
        MODEL_FIGURE_DIR,
        DIKETONE_FIGURE_DIR,
        CONTRIBUTION_FIGURE_DIR,
        COMPARATOR_FIGURE_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)
    train_manifest = pd.read_csv(TRAIN_ROWS_PATH)
    train = train_manifest["row_index"].to_numpy(dtype=int)
    refresh_audit = None
    if not args.no_excel_refresh:
        payload, train, refresh_audit = refresh_inputs_from_excel(
            payload, train_manifest
        )
    if refresh_audit is not None:
        print(f"Excel refresh: {json.dumps(refresh_audit, sort_keys=True)}", flush=True)
    meta = payload["meta"].copy()
    raw = {block: np.asarray(payload["raw_blocks"][block], dtype=float) for block in BLOCKS}
    coords = {block: np.asarray(payload["coords"][block], dtype=int) for block in BLOCKS}
    y = pd.to_numeric(meta[TARGET], errors="coerce").to_numpy(dtype=float)
    diketone_pattern = rf"[a-f](?:{'|'.join(DIKETONE_ENTRY_SUFFIXES)})"
    diketone = np.flatnonzero(
        meta["entry"].astype(str).str.fullmatch(diketone_pattern).to_numpy()
    )
    if len(diketone) != 72:
        raise ValueError(
            f"Expected 72 a-f diketone pathways, got {len(diketone)}."
        )
    if len(train) != 83:
        raise ValueError(f"Expected the frozen 83-point training manifest, got {len(train)} rows.")
    for block in BLOCKS:
        if raw[block].shape[1] != EXPECTED_FULL_GRID_N:
            raise ValueError(
                f"Expected {EXPECTED_FULL_GRID_N} {block} full-grid values, got {raw[block].shape[1]}."
            )
        selected_n = int(in_bounds(coords[block]).sum())
        if selected_n != EXPECTED_SELECTED_GRID_N:
            raise ValueError(
                f"Expected {EXPECTED_SELECTED_GRID_N} selected {block} grids, got {selected_n}."
            )

    x_full, names, _ = build_features(raw, coords, train)
    if x_full.shape[1] != EXPECTED_FEATURE_N:
        raise ValueError(f"Expected {EXPECTED_FEATURE_N} features, got {x_full.shape[1]}.")
    full_inner_predictions, full_candidate_rmse = inner_path_scores(raw, coords, y, train)
    full_best = int(np.argmin(full_candidate_rmse))
    full_alpha = float(ALPHAS[full_best])
    full_inner_rmse = float(full_candidate_rmse[full_best])
    full_inner_r2 = float(r2_score(y[train], full_inner_predictions[full_best]))
    model = Lasso(
        alpha=full_alpha,
        fit_intercept=True,
        max_iter=LASSO_FIT_MAX_ITER,
        tol=LASSO_FIT_TOL,
    ).fit(x_full[train], y[train])
    prediction = model.predict(x_full)
    pd.DataFrame(
        {
            "alpha": ALPHAS,
            "current_style_loocv_rmse": full_candidate_rmse,
            "current_style_loocv_r2": [
                r2_score(y[train], candidate) for candidate in full_inner_predictions
            ],
        }
    ).to_csv(MODEL_RESULTS_DIR / "fulltrain_inner_alpha_path.csv", index=False)

    outer_path = MODEL_RESULTS_DIR / "outer_predictions.csv"
    dike_matrix_path = DIKETONE_RESULTS_DIR / "diketone_predictions_by_outer_model.csv"
    if args.skip_nested and not (outer_path.exists() and dike_matrix_path.exists()):
        raise FileNotFoundError("--skip-nested requires current outer prediction files.")
    if args.skip_nested:
        outer = pd.read_csv(outer_path)
        # Predictions are reusable after entry renumbering, but their display
        # metadata must always follow the active workbook.
        holdout_indices = outer["holdout_index"].to_numpy(dtype=int)
        outer["entry"] = meta.loc[holdout_indices, "entry"].astype(str).to_numpy()
        outer["name"] = meta.loc[holdout_indices, "name"].astype(str).to_numpy()
        outer.to_csv(outer_path, index=False)
        dike_frame = pd.read_csv(dike_matrix_path)
        expected_columns = meta.loc[diketone, "entry"].astype(str).tolist()
        if dike_frame.columns.tolist() != expected_columns:
            raise ValueError(
                "Stored outer-model diketone predictions do not match the current "
                "a-f evaluation scope; rerun without --skip-nested."
            )
        dike_matrix = dike_frame.to_numpy(dtype=float)
    else:
        rows: list[dict[str, object]] = []
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(outer_task, fold, raw, coords, y, train, diketone, meta) for fold in range(len(train))]
            for count, future in enumerate(as_completed(futures), start=1):
                rows.append(future.result())
                print(f"nested outer fold {count}/{len(train)}", flush=True)
        rows.sort(key=lambda row: int(row["fold_id"]))
        outer = pd.DataFrame([{key: value for key, value in row.items() if key != "diketone_prediction"} for row in rows])
        dike_matrix = np.vstack([row["diketone_prediction"] for row in rows])
        outer.to_csv(outer_path, index=False)
        pd.DataFrame(dike_matrix, columns=meta.loc[diketone, "entry"].astype(str)).to_csv(dike_matrix_path, index=False)

    pd.DataFrame({"feature": names, "coefficient": model.coef_}).loc[
        lambda frame: frame["coefficient"].ne(0)
    ].to_csv(MODEL_RESULTS_DIR / "nonzero_coefficients.csv", index=False)

    contribution = pd.DataFrame({"entry": meta["entry"].astype(str), "name": meta["name"].astype(str), "InChIKey": meta["InChIKey"].astype(str), TARGET: y, "test": meta["test"].to_numpy(), "fulltrain_prediction": prediction})
    for block in BLOCKS:
        mask = np.asarray([name.startswith(f"{block}:") for name in names], dtype=bool)
        contribution[f"{block}_contribution"] = x_full[:, mask] @ model.coef_[mask]
    contribution["intercept"] = float(model.intercept_)
    contribution["role"] = np.where(np.isin(np.arange(len(meta)), train), "training", np.where(np.isin(np.arange(len(meta)), diketone), "diketone_test", "excluded_monoketone"))
    contribution.to_csv(
        MODEL_RESULTS_DIR / "fulltrain_predictions_and_contributions.csv", index=False
    )

    if not args.skip_contribution_cubes:
        # These display cubes are regenerated from the frozen atom-geometry
        # cache, so the external Gaussian archive is not needed.
        from export_current_model_contribution_cubes import (  # noqa: PLC0415
            A24_OUTPUT,
            export_contribution_cubes,
        )

        cube_manifest = export_contribution_cubes(
            meta,
            raw,
            coords,
            train,
            x_full,
            names,
            model,
            rows=train,
        )
        print(
            f"Contribution cubes: {len(cube_manifest)} substrates / "
            f"{len(cube_manifest) * len(BLOCKS)} files",
            flush=True,
        )
        a24_rows = train[meta.loc[train, "entry"].astype(str).eq("A24")]
        if len(a24_rows) != 1:
            raise ValueError("A24 must identify exactly one current training row.")
        a24_cube_manifest = export_contribution_cubes(
            meta,
            raw,
            coords,
            train,
            x_full,
            names,
            model,
            output_dir=A24_OUTPUT,
            rows=train,
            reference_row=int(a24_rows[0]),
        )
        print(
            f"A24-relative contribution cubes: {len(a24_cube_manifest)} substrates / "
            f"{len(a24_cube_manifest) * len(BLOCKS)} files",
            flush=True,
        )

    dike = contribution.iloc[diketone].copy()
    dike = dike.rename(columns={"fulltrain_prediction": "prediction"})
    dike.to_csv(
        DIKETONE_RESULTS_DIR / "fulltrain_diketone_predictions_and_contributions.csv",
        index=False,
    )
    external = contribution.loc[(contribution["role"] == "excluded_monoketone") & contribution[TARGET].notna()].copy()
    external = external.rename(columns={"fulltrain_prediction": "prediction"})
    external["error"] = external["prediction"] - external[TARGET]
    external.to_csv(
        MODEL_RESULTS_DIR / "excluded_monoketone_predictions.csv", index=False
    )

    regression = contribution.loc[contribution["role"].eq("training")].copy()
    plot_yy(
        outer,
        regression,
        external,
        MODEL_FIGURE_DIR / "yy_nested_outer_and_excluded.png",
    )
    plot_yy(
        outer,
        regression,
        external,
        MODEL_FIGURE_DIR / "yy_nested_outer_graphical_abstract.png",
        show_external=False,
        show_legend=False,
        transparent=True,
    )
    # Regenerate the report-ready contribution distribution in the normal
    # current-model run, rather than requiring a separate spatial-analysis run.
    from analyze_current_model_spatial_contributions import (  # noqa: PLC0415
        plot_block_contributions,
    )

    spatial_figure_dir = VALIDATION_DIR / "spatial_analysis"
    spatial_figure_dir.mkdir(parents=True, exist_ok=True)
    plot_block_contributions(
        x_full,
        names,
        model.coef_,
        train,
        spatial_figure_dir / "centered_block_contribution_violins.png",
    )
    from current_model_support.diketone_plots import (  # noqa: PLC0415
        reaction_concentration_plot_complex,
    )
    from current_model_support.model_figures import (  # noqa: PLC0415
        plot_component_contribution_series,
    )
    import diketone_metrics as semiquant_exp8  # noqa: PLC0415

    selectivity = semiquant_exp8.save_selectivity_identity_summary(
        dike,
        DIKETONE_RESULTS_DIR / "diketone_selectivity_summary.csv",
    )

    semiquant_summary, semiquant_detail = semiquant_exp8.evaluate_predictions(
        "current_projected_orbital_model",
        dict(zip(dike["entry"].astype(str), dike["prediction"].astype(float))),
    )
    semiquant_detail = pd.DataFrame(semiquant_detail)
    semiquant_detail.to_csv(
        DIKETONE_RESULTS_DIR / "diketone_semiquant_detail.csv", index=False
    )
    quantified = semiquant_detail.dropna(subset=["observed_percent"])
    semiquant_rmse = float(
        np.sqrt(np.mean(quantified["abs_error_percent"].to_numpy(dtype=float) ** 2))
    )
    uncertainty = save_outer_diketone_uncertainty(
        dike_matrix,
        meta.loc[diketone, "entry"].astype(str).tolist(),
        dike,
        outer.sort_values("fold_id").reset_index(drop=True),
        semiquant_exp8,
    )
    for group, temperature in {
        "a": 273.15,
        "b": 298.15,
        "c": 298.15,
        "d": 298.15,
        "e": 298.15,
        "f": 298.15,
    }.items():
        ordered = [f"{group}{suffix}" for suffix in ("1", "2", "3", "4", "13", "14", "23", "24", "31", "32", "41", "42")]
        values = dike.set_index("entry").loc[ordered, "prediction"].to_numpy(dtype=float)
        reaction_concentration_plot_complex(
            values,
            T=temperature,
            a0=1,
            save_path=DIKETONE_FIGURE_DIR / f"diketone_{group}_progress.png",
        )

    def numbered_entries(prefix: str, lower: int, upper: int) -> list[str]:
        """Return numerically sorted current entry labels within a series."""
        candidates: list[tuple[int, str]] = []
        for entry in contribution["entry"].astype(str):
            match = re.fullmatch(rf"{re.escape(prefix)}(\d+)(?:\([^)]*\))?", entry)
            if match and lower <= int(match.group(1)) <= upper:
                candidates.append((int(match.group(1)), entry))
        return [entry for _, entry in sorted(set(candidates))]

    a_entries = numbered_entries("A", 1, 13)
    plot_component_contribution_series(
        contribution,
        "electrostatic",
        a_entries,
        "A1",
        CONTRIBUTION_FIGURE_DIR / "A1_A13_electrostatic_vs_experiment.png",
        "A1--A13 series",
    )
    e_entries = numbered_entries("E", 1, 99)
    plot_component_contribution_series(
        contribution,
        "electronic",
        list(e_entries),
        "E4",
        CONTRIBUTION_FIGURE_DIR / "E_series_electronic_vs_experiment.png",
        "E series",
    )
    a_all = numbered_entries("A", 1, 99)
    plot_component_contribution_series(
        contribution,
        "electrostatic",
        [*a_entries, "A24"],
        "A24",
        CONTRIBUTION_FIGURE_DIR / "A_series_electrostatic_vs_experiment.png",
        "A series relative to A24",
        xlabel="electrostatic descriptor contribution [kcal/mol]",
        ylabel=r"$\Delta\Delta G^\ddagger_{\mathrm{expt.}}$ [kcal/mol]",
        show_grid=False,
        avoid_label_overlap=True,
        reference_label="A24",
        label_fontsize=8.5,
        figure_size=(4.2, 5.6),
        regression_line_color="0.55",
        equal_axis_scale=True,
        x_tick_step=1.0,
        label_fontweight="bold",
    )
    a_late = numbered_entries("A", 14, 99)
    c_entries = numbered_entries("C", 1, 99)
    d_entries = numbered_entries("D", 1, 99)
    a_late_reference = a_late[0]
    plot_component_contribution_series(
        contribution, "electronic", a_late, a_late_reference,
        CONTRIBUTION_FIGURE_DIR / "A_late_series_electronic_vs_experiment.png",
        f"{a_late[0]}--{a_late[-1]} series",
    )
    plot_component_contribution_series(
        contribution, "electronic", c_entries, "A24",
        CONTRIBUTION_FIGURE_DIR / "C_series_electronic_vs_experiment.png",
        "C series relative to A24",
        xlabel="electronic descriptor contribution [kcal/mol]",
        ylabel=r"$\Delta\Delta G^\ddagger_{\mathrm{expt.}}$ [kcal/mol]",
        show_grid=False,
        avoid_label_overlap=True,
        reference_label="A24",
        regression_excluded_entries=("C16", "C11"),
        series_label="Included in linear fit",
        excluded_label="Excluded from linear fit",
        label_fontsize=8.5,
        figure_size=(4.2, 5.2),
        regression_line_color="0.55",
        equal_axis_scale=True,
        x_tick_step=1.0,
        label_fontweight="bold",
    )
    combined_series_path = (
        CONTRIBUTION_FIGURE_DIR / "A_C_series_contribution_comparison.png"
    )
    combined_fig, combined_axes = plt.subplots(1, 2, figsize=(8.8, 4.4))
    plot_component_contribution_series(
        contribution,
        "electrostatic",
        [*a_entries, "A24"],
        "A24",
        combined_series_path,
        "A series relative to A24",
        xlabel="electrostatic descriptor contribution [kcal/mol]",
        ylabel=r"$\Delta\Delta G^\ddagger_{\mathrm{expt.}}$ [kcal/mol]",
        show_grid=False,
        avoid_label_overlap=True,
        reference_label="A24",
        label_fontsize=8.5,
        regression_line_color="0.55",
        square_axes=True,
        x_tick_step=1.0,
        y_tick_step=1.0,
        label_fontweight="bold",
        ax=combined_axes[0],
    )
    plot_component_contribution_series(
        contribution,
        "electronic",
        c_entries,
        "A24",
        combined_series_path,
        "C series relative to A24",
        xlabel="electronic descriptor contribution [kcal/mol]",
        ylabel=r"$\Delta\Delta G^\ddagger_{\mathrm{expt.}}$ [kcal/mol]",
        show_grid=False,
        avoid_label_overlap=True,
        reference_label="A24",
        regression_excluded_entries=("C16", "C11"),
        series_label="Included in linear fit",
        excluded_label="Excluded from linear fit",
        label_fontsize=8.5,
        regression_line_color="0.55",
        square_axes=True,
        x_tick_step=1.0,
        y_tick_step=1.0,
        label_fontweight="bold",
        ax=combined_axes[1],
    )
    combined_fig.tight_layout(w_pad=1.2)
    combined_fig.savefig(combined_series_path, dpi=500)
    plt.close(combined_fig)
    c_added_names = (
        "5-methyl-3-heptanone",
        "dicyclopropyl ketone",
        "6-methyl-5-hepten-2-one",
        "5-Chloro-2-pentanone",
    )
    c_added_labels = {
        name: str(contribution.loc[contribution["name"].eq(name), "entry"].iloc[0])
        for name in c_added_names
        if contribution["name"].eq(name).any()
    }
    c_augmented = plot_component_contribution_series(
        contribution, "electronic", c_entries, "C1",
        CONTRIBUTION_FIGURE_DIR / "C_series_highlighted_electronic_vs_experiment.png",
        f"{c_entries[0]}--{c_entries[-1]} with highlighted aliphatic ketones",
        highlighted_names=c_added_names,
        highlighted_labels=c_added_labels,
        series_label="C series",
        highlight_label="Added aliphatic ketones",
    )
    c_augmented.to_csv(
        PUBLICATION_TABLES_DIR / "C_series_highlighted_electronic_values.csv",
        index=False,
    )
    plot_component_contribution_series(
        contribution, "electronic", list(d_entries), "D4",
        CONTRIBUTION_FIGURE_DIR / "D_series_electronic_vs_experiment.png",
        f"{d_entries[0]}--{d_entries[-1]} series",
    )
    for legacy_path in (
        CONTRIBUTION_FIGURE_DIR / "A1_A13_H1_electrostatic_vs_experiment.png",
        CONTRIBUTION_FIGURE_DIR / "A13_A24_electronic_vs_experiment.png",
        CONTRIBUTION_FIGURE_DIR / "C1_C11_electronic_vs_experiment.png",
        CONTRIBUTION_FIGURE_DIR / "C1_C11_plus_added_electronic_vs_experiment.png",
        CONTRIBUTION_FIGURE_DIR / "D1_D15_electronic_vs_experiment.png",
        CONTRIBUTION_FIGURE_DIR / "A_series_electronic_vs_experiment.png",
        PUBLICATION_TABLES_DIR / "C1_C11_plus_added_electronic_values.csv",
    ):
        legacy_path.unlink(missing_ok=True)

    summary = {
        "training_n": int(len(train)), "feature_n": int(len(names)), "fulltrain_selected_alpha": full_alpha,
        "fulltrain_fit_r2": float(r2_score(y[train], prediction[train])),
        "fulltrain_fit_rmse": float(
            math.sqrt(mean_squared_error(y[train], prediction[train]))
        ),
        "fulltrain_intercept": float(model.intercept_),
        "fulltrain_inner_loocv_r2": full_inner_r2, "fulltrain_inner_loocv_rmse": full_inner_rmse,
        "fulltrain_nonzero_features": int(np.count_nonzero(model.coef_)),
        "fulltrain_orbital_nonzero": int(
            np.count_nonzero(
                model.coef_[
                    np.asarray([name.startswith("orbital:") for name in names], dtype=bool)
                ]
            )
        ),
        "nested_outer_r2": float(r2_score(outer["y_true"], outer["y_pred"])),
        "nested_outer_rmse": float(math.sqrt(mean_squared_error(outer["y_true"], outer["y_pred"]))),
        "nested_outer_mae": float(mean_absolute_error(outer["y_true"], outer["y_pred"])),
        "diketone_top_checks": int(selectivity["ok"].sum()), "diketone_check_n": int(len(selectivity)),
        "diketone_semiquant_top_checks": int(semiquant_summary["top_checks_passed"]),
        "diketone_semiquant_check_n": int(semiquant_summary["top_checks_total"]),
        "diketone_semiquant_metric_n": int(len(quantified)),
        "diketone_semiquant_mae_percent": float(semiquant_summary["semiquant_mae_percent"]),
        "diketone_semiquant_rmse_percent": semiquant_rmse,
        "diketone_evaluation_series": "a,b,c,d,e,f",
        "outer_models_8_of_8": int(uncertainty["models_8_of_8"]),
        "outer_models_8_of_8_fraction": float(uncertainty["models_8_of_8_fraction"]),
        **{
            f"{block}_fullgrid_scale": float(np.std(raw[block][train])) for block in BLOCKS
        },
    }
    pd.DataFrame([summary]).to_csv(MODEL_RESULTS_DIR / "summary.csv", index=False)
    save_model_comparison(summary)
    (DATA_DIR / "model_specification.json").write_text(
        json.dumps(
            {
                "status": "current_model",
                "descriptor_version": DESCRIPTOR_VERSION,
                "descriptor_blocks": list(BLOCKS),
                "orbital_definition": (
                    "normalize(sum_i <psi_i|pi*_C=O>/(epsilon_i-epsilon_HOMO) psi_i), then square"
                ),
                "precut_full_grid_n_per_block": EXPECTED_FULL_GRID_N,
                "electrostatic_coordinate_alignment": "electronic coordinate order",
                "electrostatic_zero_padded_coordinates": [[-7, 5, -12]],
                "grid_block_scaling": (
                    "one scalar SD over every pre-cut raw grid value in the current training fold"
                ),
                "bounds_grid_units": EE_BOUNDS,
                "selected_grid_n_per_block": EXPECTED_SELECTED_GRID_N,
                "summary_features_per_block": ["max", "min"],
                "summary_scaling": "one training-fold SD per summary feature",
                "grid_spacing_bohr": 2,
                "model": "Lasso",
                "fit_solver": {
                    "max_iter": LASSO_FIT_MAX_ITER,
                    "tolerance": LASSO_FIT_TOL,
                },
                "path_solver": {
                    "max_iter": LASSO_PATH_MAX_ITER,
                    "tolerance": LASSO_PATH_TOL,
                },
                "alpha_candidates": ALPHAS,
                "alpha_selection": "minimum inner LOOCV RMSE",
                "nested_validation": "strict outer LOOCV; held-out row excluded from all scaling and selection",
                "diketone_used_for_model_selection": False,
                "diketone_evaluation_series": list("abcdef"),
                "input_arrays": str(MODEL_ARRAYS_PATH.relative_to(ROOT)),
                "input_metadata": str(MODEL_METADATA_PATH.relative_to(ROOT)),
                "input_provenance": str(MODEL_PROVENANCE_PATH.relative_to(ROOT)),
                "input_manifest": str(INPUT_MANIFEST_PATH.relative_to(ROOT)),
                "orbital_cache": str(ORBITAL_CACHE_PATH.relative_to(ROOT)),
                "training_manifest": str(TRAIN_ROWS_PATH.relative_to(ROOT)),
                "metadata_source": str(ACTIVE_EXCEL.relative_to(ROOT)),
                "metadata_join_key": "InChIKey",
                "orbital_free_comparator_summary": str(
                    ORBITAL_FREE_SUMMARY_PATH.relative_to(ROOT)
                ),
                "contribution_cube_export": {
                    "directory": "data/validation/current_model/contribution_cubes",
                    "scope": "all 83 full-training substrates",
                    "blocks": list(BLOCKS),
                    "effect_definition": "(x_substrate - mean(x_training)) * beta",
                    "grid_spacing_bohr": 2,
                    "model_coordinate_domain": "positive-y folded compact grid",
                    "cube_coordinate_domain": "full-y symmetric display grid",
                    "cell_center_definition": "(i - sign(i)/2) * 2 Bohr",
                    "y_expansion": "half of each folded effect at +y and half at -y",
                    "summary_features": "max/min effects are listed in the manifest CSV",
                    "additional_reference_export": {
                        "directory": "data/validation/current_model/contribution_cubes_relative_to_A24",
                        "reference": "A24 benzophenone",
                        "effect_definition": "(x_substrate - x_A24) * beta",
                    },
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(pd.DataFrame([summary]).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
