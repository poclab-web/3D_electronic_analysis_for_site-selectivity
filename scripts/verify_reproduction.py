#!/usr/bin/env python3
"""Validate the frozen inputs and saved metrics of the adopted model.

This is a read-only, lightweight verification command.  It checks every file
listed in the frozen-input manifest, validates the portable array package and
training identities, constructs the 321-feature matrix, and independently
recalculates metrics from ``outer_predictions.csv``.  It does not run Gaussian
or refit the nested models.

Exit status is zero only when all checks pass; validation failures return one.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BLOCKS = ("electronic", "electrostatic", "orbital")
TARGET = "ΔΔG.expt."


class VerificationError(AssertionError):
    """Raised when a reproducibility invariant is not satisfied."""


def _require(condition: bool, message: str) -> None:
    """Raise :class:`VerificationError` when a required invariant is false."""
    if not condition:
        raise VerificationError(message)


def _sha256(path: Path) -> str:
    """Return the streaming SHA-256 digest of one frozen input file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_expected(root: Path = ROOT) -> dict[str, Any]:
    """Load the versioned reference values used by all checks."""
    path = root / "data" / "current_model" / "expected_metrics.json"
    _require(path.is_file(), f"Expected-metrics file is missing: {path}")
    with path.open(encoding="utf-8") as handle:
        expected = json.load(handle)
    _require(expected.get("schema_version") == 1, "Unsupported expected-metrics schema.")
    return expected


def verify_input_manifest(root: Path = ROOT) -> dict[str, int]:
    """Verify byte size and SHA-256 for every declared frozen input."""
    input_dir = root / "data" / "current_model" / "inputs"
    manifest_path = input_dir / "input_manifest.csv"
    _require(manifest_path.is_file(), f"Input manifest is missing: {manifest_path}")
    manifest = pd.read_csv(manifest_path)
    required_columns = {"path", "size_bytes", "sha256"}
    _require(
        required_columns.issubset(manifest.columns),
        f"Input manifest must contain columns {sorted(required_columns)}.",
    )
    _require(len(manifest) > 0, "Input manifest is empty.")
    _require(not manifest["path"].duplicated().any(), "Input manifest contains duplicate paths.")

    input_root = input_dir.resolve()
    total_bytes = 0
    for row in manifest.itertuples(index=False):
        relative = Path(str(row.path))
        _require(not relative.is_absolute(), f"Manifest path must be relative: {relative}")
        path = (input_dir / relative).resolve()
        _require(
            path == input_root or input_root in path.parents,
            f"Manifest path escapes the input directory: {relative}",
        )
        _require(path.is_file(), f"Frozen input is missing: {relative}")
        actual_size = path.stat().st_size
        expected_size = int(row.size_bytes)
        _require(
            actual_size == expected_size,
            f"Size mismatch for {relative}: expected {expected_size}, found {actual_size}",
        )
        actual_hash = _sha256(path)
        expected_hash = str(row.sha256).lower()
        _require(
            actual_hash == expected_hash,
            f"SHA-256 mismatch for {relative}: expected {expected_hash}, found {actual_hash}",
        )
        total_bytes += actual_size
    return {"manifest_files": int(len(manifest)), "manifest_bytes": total_bytes}


def _load_portable_inputs(root: Path) -> tuple[
    pd.DataFrame,
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, Any],
]:
    """Load metadata, numeric descriptor blocks, coordinates, and provenance."""
    input_dir = root / "data" / "current_model" / "inputs"
    arrays_path = input_dir / "model_arrays.npz"
    metadata_path = input_dir / "model_metadata.csv"
    provenance_path = input_dir / "model_provenance.json"
    for path in (arrays_path, metadata_path, provenance_path):
        _require(path.is_file(), f"Portable model input is missing: {path}")

    metadata = pd.read_csv(metadata_path)
    with provenance_path.open(encoding="utf-8") as handle:
        provenance = json.load(handle)
    with np.load(arrays_path, allow_pickle=False) as archive:
        required_keys = {
            *(f"raw_{block}" for block in BLOCKS),
            *(f"coords_{block}" for block in BLOCKS),
        }
        _require(
            required_keys.issubset(archive.files),
            f"model_arrays.npz is missing keys: {sorted(required_keys - set(archive.files))}",
        )
        raw = {
            block: np.asarray(archive[f"raw_{block}"], dtype=float).copy()
            for block in BLOCKS
        }
        coords = {
            block: np.asarray(archive[f"coords_{block}"], dtype=int).copy()
            for block in BLOCKS
        }
    return metadata, raw, coords, provenance


def verify_portable_inputs(root: Path = ROOT, expected: dict[str, Any] | None = None) -> dict[str, int]:
    """Validate portable arrays, training identities, and model features."""
    expected = load_expected(root) if expected is None else expected
    limits = expected["inputs"]
    metadata, raw, coords, provenance = _load_portable_inputs(root)
    metadata_rows = int(limits["metadata_rows"])
    grid_n = int(limits["full_grid_points_per_block"])
    selected_n = int(limits["selected_grid_points_per_block"])

    expected_columns = ["entry", "name", "SMILES", "InChIKey", "temperature", TARGET, "test"]
    _require(metadata.columns.tolist() == expected_columns, "Portable metadata columns or order changed.")
    _require(len(metadata) == metadata_rows, f"Expected {metadata_rows} metadata rows, found {len(metadata)}.")
    _require(metadata["InChIKey"].notna().all(), "Portable metadata contains a missing InChIKey.")
    _require(metadata["InChIKey"].is_unique, "Portable metadata InChIKeys are not unique.")

    _require(provenance.get("descriptor_version") == limits["descriptor_version"], "Descriptor version changed.")
    _require(tuple(provenance.get("blocks", ())) == BLOCKS, "Descriptor block order changed.")
    _require(int(provenance.get("row_count", -1)) == metadata_rows, "Provenance row count is incorrect.")
    conformer = provenance.get("conformer_generation", {})
    quantum = provenance.get("quantum_chemistry", {})
    _require(
        int(conformer.get("random_seed", -1)) == int(limits["conformer_random_seed"]),
        "Conformer-generation random seed changed.",
    )
    for key, provenance_key in (
        ("geometry_method", "optimization_frequency"),
        ("single_point_method", "single_point"),
        ("solvation", "solvation"),
        ("reference_program", "reference_program"),
    ):
        _require(
            quantum.get(provenance_key) == limits[key],
            f"Quantum-chemistry provenance changed for {provenance_key}.",
        )

    for block in BLOCKS:
        _require(raw[block].shape == (metadata_rows, grid_n), f"Unexpected {block} raw shape: {raw[block].shape}")
        _require(coords[block].shape == (grid_n, 3), f"Unexpected {block} coordinate shape: {coords[block].shape}")
        _require(np.isfinite(raw[block]).all(), f"{block} raw values contain NaN or infinity.")
        _require(len(np.unique(coords[block], axis=0)) == grid_n, f"{block} coordinates are not unique.")
        provenance_grid_n = int(provenance.get("grid_count_per_block", {}).get(block, -1))
        _require(provenance_grid_n == grid_n, f"Provenance grid count is wrong for {block}.")
    for block in BLOCKS[1:]:
        _require(np.array_equal(coords[BLOCKS[0]], coords[block]), f"{block} coordinate order is not aligned.")

    input_dir = root / "data" / "current_model" / "inputs"
    train_path = input_dir / "train_rows.csv"
    _require(train_path.is_file(), f"Training manifest is missing: {train_path}")
    train_rows = pd.read_csv(train_path)
    required_train_columns = {"row_index", "entry", "name", "InChIKey", "ddg"}
    _require(required_train_columns.issubset(train_rows.columns), "Training manifest columns are incomplete.")
    training_n = int(limits["training_rows"])
    _require(len(train_rows) == training_n, f"Expected {training_n} training rows, found {len(train_rows)}.")
    train = pd.to_numeric(train_rows["row_index"], errors="raise").to_numpy(dtype=int)
    _require(len(np.unique(train)) == training_n, "Training row indices are not unique.")
    _require(((0 <= train) & (train < metadata_rows)).all(), "A training row index is out of range.")
    selected_meta = metadata.iloc[train].reset_index(drop=True)
    _require(
        np.array_equal(selected_meta["InChIKey"].astype(str), train_rows["InChIKey"].astype(str)),
        "Training identities do not match portable metadata.",
    )
    _require(selected_meta["InChIKey"].is_unique, "Training InChIKeys are not unique.")
    training_targets = pd.to_numeric(selected_meta[TARGET], errors="coerce").to_numpy(dtype=float)
    _require(np.isfinite(training_targets).all(), "A training target is missing or non-finite.")
    _require(
        np.allclose(
            training_targets,
            pd.to_numeric(train_rows["ddg"], errors="coerce").to_numpy(dtype=float),
            rtol=0.0,
            atol=1e-12,
        ),
        "Training target values do not match the training manifest.",
    )

    combined_raw = raw
    combined_n = int(limits["combined_rows"])
    _require(all(values.shape[0] == combined_n for values in combined_raw.values()), "Combined row count is incorrect.")

    libs_dir = root / "libs"
    sys.path.insert(0, str(libs_dir))
    try:
        import current_model  # noqa: PLC0415
    finally:
        try:
            sys.path.remove(str(libs_dir))
        except ValueError:
            pass
    features, names, masks = current_model.build_features(combined_raw, coords, train)
    feature_n = int(limits["feature_count"])
    _require(features.shape == (combined_n, feature_n), f"Unexpected feature shape: {features.shape}")
    _require(len(names) == feature_n and len(set(names)) == feature_n, "Feature names are incomplete or duplicated.")
    _require(np.isfinite(features).all(), "Feature matrix contains NaN or infinity.")
    for block in BLOCKS:
        _require(int(np.count_nonzero(masks[block])) == selected_n, f"Expected {selected_n} selected {block} grids.")
    return {
        "metadata_rows": metadata_rows,
        "training_rows": training_n,
        "combined_rows": combined_n,
        "features": feature_n,
    }


def _close(actual: float, expected: float, *, atol: float, rtol: float) -> bool:
    """Compare two finite scalar results with the versioned tolerances."""
    return math.isclose(float(actual), float(expected), abs_tol=atol, rel_tol=rtol)


def verify_saved_metrics(root: Path = ROOT, expected: dict[str, Any] | None = None) -> dict[str, float]:
    """Validate summary values and recalculate outer-prediction metrics."""
    expected = load_expected(root) if expected is None else expected
    atol = float(expected["absolute_tolerance"])
    rtol = float(expected["relative_tolerance"])
    data_dir = root / "data" / "current_model"

    model_results = data_dir / "results" / "model"
    summary_path = model_results / "summary.csv"
    outer_path = model_results / "outer_predictions.csv"
    _require(summary_path.is_file(), f"Saved summary is missing: {summary_path}")
    _require(outer_path.is_file(), f"Saved outer predictions are missing: {outer_path}")
    summary_frame = pd.read_csv(summary_path)
    _require(len(summary_frame) == 1, "summary.csv must contain exactly one result row.")
    summary = summary_frame.iloc[0]
    for key, expected_value in expected["summary"].items():
        _require(key in summary.index, f"summary.csv is missing metric {key!r}.")
        actual = summary[key]
        if isinstance(expected_value, str):
            _require(str(actual) == expected_value, f"Summary mismatch for {key}: expected {expected_value!r}, found {actual!r}")
        else:
            _require(
                _close(float(actual), float(expected_value), atol=atol, rtol=rtol),
                f"Summary mismatch for {key}: expected {expected_value}, found {actual}",
            )

    outer = pd.read_csv(outer_path).sort_values("fold_id").reset_index(drop=True)
    outer_expected = expected["outer_predictions"]
    outer_n = int(outer_expected["rows"])
    _require(len(outer) == outer_n, f"Expected {outer_n} outer predictions, found {len(outer)}.")
    _require(outer["fold_id"].tolist() == list(range(outer_n)), "Outer fold IDs are incomplete or out of order.")

    train_rows = pd.read_csv(data_dir / "inputs" / "train_rows.csv")
    _require(
        np.array_equal(outer["holdout_index"].to_numpy(dtype=int), train_rows["row_index"].to_numpy(dtype=int)),
        "Outer holdout indices do not match the training manifest.",
    )
    _require(
        np.allclose(
            outer["y_true"].to_numpy(dtype=float),
            train_rows["ddg"].to_numpy(dtype=float),
            rtol=0.0,
            atol=1e-12,
        ),
        "Outer true responses do not match the training manifest.",
    )

    y_true = outer["y_true"].to_numpy(dtype=float)
    y_pred = outer["y_pred"].to_numpy(dtype=float)
    residual = y_pred - y_true
    r2 = 1.0 - float(np.sum(residual**2) / np.sum((y_true - np.mean(y_true)) ** 2))
    rmse = float(np.sqrt(np.mean(residual**2)))
    mae = float(np.mean(np.abs(residual)))
    for label, actual, key in (
        ("R2", r2, "r2"),
        ("RMSE", rmse, "rmse_kcal_mol"),
        ("MAE", mae, "mae_kcal_mol"),
    ):
        _require(
            _close(actual, float(outer_expected[key]), atol=atol, rtol=rtol),
            f"Outer {label} mismatch: expected {outer_expected[key]}, found {actual}",
        )
    selected_alpha = float(outer_expected["selected_alpha"])
    selected_count = int(np.count_nonzero(np.isclose(outer["selected_alpha"], selected_alpha, rtol=0.0, atol=atol)))
    _require(
        selected_count == int(outer_expected["selected_alpha_count"]),
        f"Expected {outer_expected['selected_alpha_count']} outer folds with alpha={selected_alpha}, found {selected_count}.",
    )
    return {"outer_r2": r2, "outer_rmse": rmse, "outer_mae": mae}


def verify_spatial_analysis(
    root: Path = ROOT,
    expected: dict[str, Any] | None = None,
) -> dict[str, int]:
    """Validate portable spatial-analysis matrices, tables, and final figures."""
    expected = load_expected(root) if expected is None else expected
    limits = expected["spatial_analysis"]
    analysis_dir = root / "data" / "current_model" / "spatial_analysis"
    figure_dir = root / "data" / "validation" / "current_model" / "spatial_analysis"

    specification_path = analysis_dir / "analysis_specification.json"
    _require(
        specification_path.is_file(),
        f"Spatial-analysis specification is missing: {specification_path}",
    )
    with specification_path.open(encoding="utf-8") as handle:
        specification = json.load(handle)
    for key in ("training_n", "feature_n", "spatial_grid_n", "outer_model_n"):
        _require(
            int(specification.get(key, -1)) == int(limits[key]),
            f"Spatial-analysis {key} mismatch.",
        )
    _require(
        _close(
            float(specification.get("fulltrain_alpha", math.nan)),
            float(limits["fulltrain_alpha"]),
            atol=float(expected["absolute_tolerance"]),
            rtol=float(expected["relative_tolerance"]),
        ),
        "Spatial-analysis full-training alpha mismatch.",
    )
    _require(specification.get("model_changed") is False, "Spatial analysis changed the model.")

    feature_table = pd.read_csv(
        analysis_dir / "feature_coefficient_and_effect_statistics.csv"
    )
    grid_table = pd.read_csv(analysis_dir / "spatial_grid_statistics.csv")
    feature_n = int(limits["feature_n"])
    grid_n = int(limits["spatial_grid_n"])
    _require(len(feature_table) == feature_n, "Spatial feature-table row count changed.")
    _require(len(grid_table) == grid_n, "Spatial grid-table row count changed.")
    _require(feature_table["feature"].is_unique, "Spatial feature names are duplicated.")
    finite_columns = (
        "fulltrain_coefficient",
        "outer_selection_frequency",
        "outer_coefficient_mean",
        "outer_coefficient_median_all",
        "outer_coefficient_p16",
        "outer_coefficient_p84",
        "training_centered_effect_rms_kcal_mol",
        "training_centered_effect_mean_abs_kcal_mol",
        "training_centered_effect_max_abs_kcal_mol",
        "stability_weighted_abs_coefficient",
    )
    for column in finite_columns:
        _require(column in feature_table, f"Spatial feature table is missing {column}.")
        _require(
            np.isfinite(pd.to_numeric(feature_table[column], errors="coerce")).all(),
            f"Spatial feature column {column} contains a missing or non-finite value.",
        )

    feature_names = feature_table["feature"].astype(str).to_numpy()
    outer_predictions = pd.read_csv(
        root
        / "data"
        / "current_model"
        / "results"
        / "model"
        / "outer_predictions.csv"
    )
    outer_predictions = outer_predictions.sort_values("fold_id").reset_index(drop=True)

    with np.load(
        analysis_dir / "outer83_feature_coefficients.npz", allow_pickle=False
    ) as archive:
        required = {"coefficients", "feature_names", "fold_id", "holdout_entry"}
        _require(required.issubset(archive.files), "Outer-coefficient NPZ is incomplete.")
        _require(
            archive["coefficients"].shape == (int(limits["outer_model_n"]), feature_n),
            "Outer-coefficient matrix shape changed.",
        )
        coefficients = np.asarray(archive["coefficients"], dtype=float)
        _require(np.isfinite(coefficients).all(), "Outer coefficients contain NaN or infinity.")
        _require(
            np.array_equal(archive["feature_names"].astype(str), feature_names),
            "Outer NPZ feature names do not match the feature table.",
        )
        _require(
            archive["fold_id"].astype(int).tolist()
            == list(range(int(limits["outer_model_n"]))),
            "Outer coefficient fold IDs are incomplete or out of order.",
        )
        _require(
            archive["holdout_entry"].astype(str).tolist()
            == outer_predictions["entry"].astype(str).tolist(),
            "Outer coefficient holdout identities do not match outer_predictions.csv.",
        )
        outer_coefficient_frobenius = float(np.linalg.norm(coefficients))
    with np.load(
        analysis_dir / "training_centered_feature_effects.npz", allow_pickle=False
    ) as archive:
        required = {"effects", "feature_names", "entry", "name", "InChIKey"}
        _require(required.issubset(archive.files), "Training-effect NPZ is incomplete.")
        _require(
            archive["effects"].shape == (int(limits["training_n"]), feature_n),
            "Training-effect matrix shape changed.",
        )
        effects = np.asarray(archive["effects"], dtype=float)
        _require(np.isfinite(effects).all(), "Training effects contain NaN or infinity.")
        _require(
            np.array_equal(archive["feature_names"].astype(str), feature_names),
            "Training-effect NPZ feature names do not match the feature table.",
        )
        train_rows = pd.read_csv(root / "data" / "current_model" / "inputs" / "train_rows.csv")
        _require(
            archive["entry"].astype(str).tolist() == train_rows["entry"].astype(str).tolist(),
            "Training-effect entry identities do not match train_rows.csv.",
        )
        _require(
            archive["InChIKey"].astype(str).tolist()
            == train_rows["InChIKey"].astype(str).tolist(),
            "Training-effect InChIKeys do not match train_rows.csv.",
        )
        training_effect_rms = float(np.sqrt(np.mean(effects**2)))

    atol = float(expected["absolute_tolerance"])
    rtol = float(expected["relative_tolerance"])
    numeric_checks = {
        "fulltrain_coefficient_l1": float(
            np.abs(feature_table["fulltrain_coefficient"].to_numpy(dtype=float)).sum()
        ),
        "fulltrain_coefficient_l2": float(
            np.linalg.norm(feature_table["fulltrain_coefficient"].to_numpy(dtype=float))
        ),
        "fulltrain_nonzero_features": float(
            np.count_nonzero(
                np.abs(feature_table["fulltrain_coefficient"].to_numpy(dtype=float)) > 1.0e-12
            )
        ),
        "outer_coefficient_frobenius": outer_coefficient_frobenius,
        "training_effect_rms": training_effect_rms,
        "grid_outer_selection_frequency_sum": float(
            grid_table["outer_selection_frequency"].to_numpy(dtype=float).sum()
        ),
    }
    for key, actual in numeric_checks.items():
        _require(key in limits, f"Expected spatial metric {key} is not versioned.")
        _require(
            _close(actual, float(limits[key]), atol=atol, rtol=rtol),
            f"Spatial metric {key} mismatch: expected {limits[key]}, found {actual}.",
        )

    figures = (
        "centered_block_contribution_violins.png",
        "spatial_grid_effect_by_distance.png",
        "spatial_grid_fulltrain_coefficients_3d.png",
        "spatial_grid_fulltrain_coefficients_by_y.png",
        "spatial_grid_outer_selection_frequency_by_y.png",
        "spatial_grid_realized_effect_by_y.png",
    )
    _require(len(figures) == int(limits["figure_count"]), "Expected spatial figure count changed.")
    for name in figures:
        path = figure_dir / name
        _require(path.is_file() and path.stat().st_size > 0, f"Spatial figure is missing or empty: {path}")
    return {"features": feature_n, "spatial_grids": grid_n, "figures": len(figures)}


def verify_repository(root: Path = ROOT) -> dict[str, Any]:
    """Run every lightweight reproducibility check and return a summary."""
    root = root.resolve()
    expected = load_expected(root)
    return {
        "manifest": verify_input_manifest(root),
        "inputs": verify_portable_inputs(root, expected),
        "metrics": verify_saved_metrics(root, expected),
        "spatial": verify_spatial_analysis(root, expected),
    }


def main(argv: list[str] | None = None) -> int:
    """Run the command-line verifier and return a process exit status."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=ROOT,
        help="Repository root to verify (default: inferred from this script).",
    )
    parser.add_argument("--quiet", action="store_true", help="Print only failures.")
    args = parser.parse_args(argv)
    try:
        result = verify_repository(args.root)
    except (VerificationError, FileNotFoundError, KeyError, ValueError, OSError) as exc:
        print(f"REPRODUCIBILITY CHECK FAILED: {exc}", file=sys.stderr)
        return 1
    if not args.quiet:
        manifest = result["manifest"]
        inputs = result["inputs"]
        metrics = result["metrics"]
        spatial = result["spatial"]
        print(
            "REPRODUCIBILITY CHECK PASSED\n"
            f"  frozen inputs: {manifest['manifest_files']} files, {manifest['manifest_bytes']} bytes\n"
            f"  model matrix: {inputs['combined_rows']} rows x {inputs['features']} features\n"
            f"  training rows: {inputs['training_rows']}\n"
            f"  nested outer LOOCV: R2={metrics['outer_r2']:.9f}, "
            f"RMSE={metrics['outer_rmse']:.9f}, MAE={metrics['outer_mae']:.9f} kcal/mol\n"
            f"  spatial analysis: {spatial['spatial_grids']} grids, "
            f"{spatial['figures']} final figures"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
