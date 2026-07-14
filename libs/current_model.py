"""Reproducible runner for the current site-selectivity reference model.

The model uses the frozen 83-observation descriptor bundle generated from the
current experimental data.  It retains only the compact electronic and
electrostatic grids plus max/min summaries (214 features) and fits Lasso with
alpha selected by inner LOOCV in every outer fold.

Run from the repository root::

    OMP_NUM_THREADS=1 python libs/current_model.py --workers 20
"""
from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "current_model"
VALIDATION_DIR = ROOT / "data" / "validation" / "current_model"
BUNDLE_PATH = DATA_DIR / "model_input_bundle.pkl"
TRAIN_ROWS_PATH = DATA_DIR / "train_rows.csv"
ARCHIVED_BUNDLE = (
    ROOT
    / "analysis_runs/homo_damped_projected_orbital_nested_20260709"
    / "current_model_20260712/full85_bundle_20260713/full85_bundle.pkl"
)
ARCHIVED_TRAIN_ROWS = (
    ROOT
    / "analysis_runs/homo_damped_projected_orbital_nested_20260709"
    / "current_model_20260712/strict_nested_lasso_plus_D100_C101_C102_20260713/train_rows.csv"
)
ARCHIVED_VALIDATION_DIR = (
    ROOT
    / "analysis_runs/homo_damped_projected_orbital_nested_20260709"
    / "current_model_20260712/summary_max_min_only_ablation_20260714"
)
ARCHIVED_SEMIQUANT_DETAIL = ARCHIVED_VALIDATION_DIR / "fulltrain_diketone_semiquant_detail.csv"

TARGET = "ΔΔG.expt."
BLOCKS = ("electronic", "electrostatic")
EE_BOUNDS = (-5, 2, 1, 3, -2, 3)
ALPHAS = (0.010, 0.011, 0.012, 0.013, 0.014, 0.015, 0.016)
FULLTRAIN_ALPHA = 0.010


def ensure_frozen_inputs() -> None:
    """Install the descriptor cache and 83-point training manifest once."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not BUNDLE_PATH.exists():
        if not ARCHIVED_BUNDLE.exists():
            raise FileNotFoundError(f"Reference descriptor bundle is missing: {ARCHIVED_BUNDLE}")
        shutil.copy2(ARCHIVED_BUNDLE, BUNDLE_PATH)
    manifest_ok = TRAIN_ROWS_PATH.exists() and len(pd.read_csv(TRAIN_ROWS_PATH)) == 83
    if not manifest_ok:
        if not ARCHIVED_TRAIN_ROWS.exists():
            raise FileNotFoundError(f"Reference training manifest is missing: {ARCHIVED_TRAIN_ROWS}")
        shutil.copy2(ARCHIVED_TRAIN_ROWS, TRAIN_ROWS_PATH)


def in_bounds(coords: np.ndarray) -> np.ndarray:
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
    """Build the 210 compact grid and four max/min summary features."""
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
        names.extend(f"{block}:grid:{tuple(coord)}" for coord in coords[keep])
    for block in BLOCKS:
        raw = raw_blocks[block]
        keep = masks[block]
        summary = np.column_stack((np.max(raw[:, keep], axis=1), np.min(raw[:, keep], axis=1)))
        summary_scale = np.std(summary[train], axis=0)
        summary_scale[~np.isfinite(summary_scale) | (summary_scale == 0.0)] = 1.0
        arrays.append(summary / summary_scale)
        names.extend((f"{block}:summary:max", f"{block}:summary:min"))
    return np.concatenate(arrays, axis=1), names, masks


def fit_predict(x: np.ndarray, y: np.ndarray, train: np.ndarray, pred: np.ndarray, alpha: float) -> np.ndarray:
    model = Lasso(alpha=alpha, fit_intercept=True, max_iter=50000, tol=1e-5)
    model.fit(x[train], y[train])
    return model.predict(x[pred])


def inner_best(raw: dict[str, np.ndarray], coords: dict[str, np.ndarray], y: np.ndarray, train: np.ndarray) -> tuple[float, float, float]:
    predictions = np.empty((len(ALPHAS), len(train)), dtype=float)
    for local, holdout in enumerate(train):
        fold_train = train[train != holdout]
        x, _, _ = build_features(raw, coords, fold_train)
        for candidate, alpha in enumerate(ALPHAS):
            predictions[candidate, local] = fit_predict(x, y, fold_train, np.asarray([holdout]), alpha)[0]
    rmse = np.sqrt(np.mean((predictions - y[train][None, :]) ** 2, axis=1))
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
    holdout = int(train[fold_id])
    fold_train = train[train != holdout]
    alpha, inner_rmse, inner_r2 = inner_best(raw, coords, y, fold_train)
    x, _, _ = build_features(raw, coords, fold_train)
    return {
        "fold_id": fold_id,
        "holdout_index": holdout,
        "entry": str(meta.loc[holdout, "entry"]),
        "name": str(meta.loc[holdout, "name"]),
        "y_true": float(y[holdout]),
        "y_pred": float(fit_predict(x, y, fold_train, np.asarray([holdout]), alpha)[0]),
        "selected_alpha": alpha,
        "inner_rmse": inner_rmse,
        "inner_r2": inner_r2,
        "diketone_prediction": fit_predict(x, y, fold_train, diketone, alpha),
    }


def plot_yy(outer: pd.DataFrame, external: pd.DataFrame, path: Path) -> None:
    values = np.r_[outer["y_true"], outer["y_pred"], external[TARGET], external["prediction"]]
    low, high = float(np.nanmin(values) - 0.25), float(np.nanmax(values) + 0.25)
    fig, ax = plt.subplots(figsize=(5.0, 4.8))
    ax.scatter(outer["y_true"], outer["y_pred"], color="#3d6f8e", s=36, edgecolor="white", linewidth=0.45, label=f"Nested outer LOOCV (n={len(outer)})")
    if not external.empty:
        ax.scatter(external[TARGET], external["prediction"], color="#c53d3d", s=58, edgecolor="white", linewidth=0.55, label=f"Excluded monoketones (n={len(external)})")
    ax.plot([low, high], [low, high], color="0.2", linewidth=1.0)
    ax.set(xlim=(low, high), ylim=(low, high), aspect="equal", xlabel=r"Experimental $\Delta\Delta G^\ddagger$ [kcal/mol]", ylabel=r"Predicted $\Delta\Delta G^\ddagger$ [kcal/mol]")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.4)
    fig.tight_layout()
    fig.savefig(path, dpi=500)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--skip-nested", action="store_true", help="Reuse an existing outer_predictions.csv if available.")
    args = parser.parse_args()
    workers = min(max(args.workers, 1), 20)

    ensure_frozen_inputs()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
    payload = pd.read_pickle(BUNDLE_PATH)
    meta = payload["meta"].copy()
    raw = {block: np.asarray(payload["raw_blocks"][block], dtype=float) for block in BLOCKS}
    coords = {block: np.asarray(payload["coords"][block], dtype=int) for block in BLOCKS}
    y = meta[TARGET].astype(float).to_numpy()
    train = pd.read_csv(TRAIN_ROWS_PATH)["row_index"].to_numpy(dtype=int)
    diketone = np.flatnonzero(meta["entry"].astype(str).str.match(r"^[a-f]").to_numpy())
    if len(train) != 83:
        raise ValueError(f"Expected the frozen 83-point training manifest, got {len(train)} rows.")

    x_full, names, _ = build_features(raw, coords, train)
    if x_full.shape[1] != 214:
        raise ValueError(f"Expected 214 features, got {x_full.shape[1]}.")
    full_alpha, full_inner_rmse, full_inner_r2 = inner_best(raw, coords, y, train)
    model = Lasso(alpha=full_alpha, fit_intercept=True, max_iter=50000, tol=1e-5).fit(x_full[train], y[train])
    prediction = model.predict(x_full)

    outer_path = DATA_DIR / "outer_predictions.csv"
    dike_matrix_path = DATA_DIR / "diketone_predictions_by_outer_model.csv"
    if args.skip_nested and not (outer_path.exists() and dike_matrix_path.exists()):
        for source_name, destination in (
            ("outer_predictions.csv", outer_path),
            ("diketone_predictions_by_outer_model.csv", dike_matrix_path),
        ):
            source = ARCHIVED_VALIDATION_DIR / source_name
            if not source.exists():
                raise FileNotFoundError(f"Validated nested result is missing: {source}")
            shutil.copy2(source, destination)
    if args.skip_nested and ARCHIVED_SEMIQUANT_DETAIL.exists():
        shutil.copy2(ARCHIVED_SEMIQUANT_DETAIL, DATA_DIR / "diketone_semiquant_validated_detail.csv")
    if args.skip_nested and outer_path.exists() and dike_matrix_path.exists():
        outer = pd.read_csv(outer_path)
        dike_matrix = pd.read_csv(dike_matrix_path).to_numpy(dtype=float)
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

    contribution = pd.DataFrame({"entry": meta["entry"].astype(str), "name": meta["name"].astype(str), "InChIKey": meta["InChIKey"].astype(str), TARGET: y, "test": meta["test"].to_numpy(), "fulltrain_prediction": prediction})
    for block in BLOCKS:
        mask = np.asarray([name.startswith(f"{block}:") for name in names], dtype=bool)
        contribution[f"{block}_contribution"] = x_full[:, mask] @ model.coef_[mask]
    contribution["intercept"] = float(model.intercept_)
    contribution["role"] = np.where(np.isin(np.arange(len(meta)), train), "training", np.where(np.isin(np.arange(len(meta)), diketone), "diketone_test", "excluded_monoketone"))
    contribution.to_csv(DATA_DIR / "fulltrain_predictions_and_contributions.csv", index=False)

    dike = contribution.iloc[diketone].copy()
    dike = dike.rename(columns={"fulltrain_prediction": "prediction"})
    dike.to_csv(DATA_DIR / "fulltrain_diketone_predictions_and_contributions.csv", index=False)
    external = contribution.loc[(contribution["role"] == "excluded_monoketone") & contribution[TARGET].notna()].copy()
    external = external.rename(columns={"fulltrain_prediction": "prediction"})
    external["error"] = external["prediction"] - external[TARGET]
    external.to_csv(DATA_DIR / "excluded_monoketone_predictions.csv", index=False)

    plot_yy(outer, external, VALIDATION_DIR / "yy_nested_outer_and_excluded.png")
    from graph import (  # noqa: PLC0415
        plot_component_contribution_series,
        reaction_concentration_plot_complex,
        save_diketone_selectivity_summary,
    )

    selectivity = save_diketone_selectivity_summary(dike, DATA_DIR / "diketone_selectivity_summary.csv")
    for group, temperature in {"a": 273.15, "b": 298.15, "c": 298.15, "d": 298.15, "e": 298.15, "f": 298.15}.items():
        ordered = [f"{group}{suffix}" for suffix in ("1", "2", "3", "4", "13", "14", "23", "24", "31", "32", "41", "42")]
        values = dike.set_index("entry").loc[ordered, "prediction"].to_numpy(dtype=float)
        reaction_concentration_plot_complex(values, T=temperature, a0=1, save_path=VALIDATION_DIR / f"diketone_{group}_progress.png")

    a_entries = [f"A{i}" for i in range(1, 14)] + ["H1"]
    plot_component_contribution_series(contribution, "electrostatic", a_entries, "A1", VALIDATION_DIR / "A1_A13_H1_electrostatic_vs_experiment.png", "A1--A13 + Benzoylacetonitrile")
    e_entries = [entry for entry in contribution["entry"] if str(entry).startswith("E")]
    plot_component_contribution_series(contribution, "electronic", list(e_entries), "E4", VALIDATION_DIR / "E_series_electronic_vs_experiment.png", "E series")
    a_late = [f"A{i}" for i in range(13, 25)]
    c_entries = [f"C{i}" for i in range(1, 12)]
    d_entries = [entry for entry in contribution["entry"] if re.match(r"^D(?:[1-9]|1[0-5])", str(entry))]
    plot_component_contribution_series(contribution, "electronic", a_late, "A13", VALIDATION_DIR / "A13_A24_electronic_vs_experiment.png", "A13--A24 series")
    plot_component_contribution_series(contribution, "electronic", c_entries, "C1", VALIDATION_DIR / "C1_C11_electronic_vs_experiment.png", "C1--C11 series")
    c_added_names = (
        "5-methyl-3-heptanone",
        "dicyclopropyl ketone",
        "6-methyl-5-hepten-2-one",
        "5-Chloro-2-pentanone",
    )
    c_added_labels = {
        "5-methyl-3-heptanone": "C102",
        "dicyclopropyl ketone": "dicyclopropyl ketone",
        "6-methyl-5-hepten-2-one": "6-methyl-5-hepten-2-one",
        "5-Chloro-2-pentanone": "C101 (5-chloro-2-pentanone)",
    }
    c_augmented = plot_component_contribution_series(
        contribution, "electronic", c_entries, "C1",
        VALIDATION_DIR / "C1_C11_plus_added_electronic_vs_experiment.png",
        "C1--C11 + added aliphatic ketones",
        highlighted_names=c_added_names,
        highlighted_labels=c_added_labels,
        series_label="C series",
        highlight_label="Added aliphatic ketones",
    )
    c_augmented.to_csv(DATA_DIR / "C1_C11_plus_added_electronic_values.csv", index=False)
    plot_component_contribution_series(contribution, "electronic", list(d_entries), "D4", VALIDATION_DIR / "D1_D15_electronic_vs_experiment.png", "D1--D15 series")

    validated_detail_path = DATA_DIR / "diketone_semiquant_validated_detail.csv"
    semiquant_mae = math.nan
    if validated_detail_path.exists():
        validated_detail = pd.read_csv(validated_detail_path)
        semiquant_mae = float(validated_detail["abs_error_percent"].dropna().mean())
    summary = {
        "training_n": int(len(train)), "feature_n": int(len(names)), "fulltrain_selected_alpha": full_alpha,
        "fulltrain_inner_loocv_r2": full_inner_r2, "fulltrain_inner_loocv_rmse": full_inner_rmse,
        "fulltrain_nonzero_features": int(np.count_nonzero(model.coef_)),
        "nested_outer_r2": float(r2_score(outer["y_true"], outer["y_pred"])),
        "nested_outer_rmse": float(math.sqrt(mean_squared_error(outer["y_true"], outer["y_pred"]))),
        "nested_outer_mae": float(mean_absolute_error(outer["y_true"], outer["y_pred"])),
        "diketone_top_checks": int(selectivity["ok"].sum()), "diketone_check_n": int(len(selectivity)),
        "validated_diketone_semiquant_mae_percent": semiquant_mae,
    }
    pd.DataFrame([summary]).to_csv(DATA_DIR / "summary.csv", index=False)
    (DATA_DIR / "model_specification.json").write_text(json.dumps({"descriptor": "electronic/electrostatic compact grid plus max/min summaries", "bounds_grid_units": EE_BOUNDS, "grid_spacing_bohr": 2, "alpha_candidates": ALPHAS, "frozen_input_bundle": str(BUNDLE_PATH.relative_to(ROOT)), "training_manifest": str(TRAIN_ROWS_PATH.relative_to(ROOT))}, indent=2) + "\n", encoding="utf-8")
    print(pd.DataFrame([summary]).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
