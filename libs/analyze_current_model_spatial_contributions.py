"""Analyze spatial coefficient and realized-effect distributions of the current model.

The script refits the frozen 321-feature Lasso model and its 83 outer-LOOCV
models.  It maps the 105 retained cells in each electronic, electrostatic, and
projected-orbital block back to integer 2-Bohr grid coordinates.  Coefficients
are reported per block-scaled descriptor unit; realized effects are centered
training descriptors multiplied by those coefficients and are in kcal/mol.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize, TwoSlopeNorm
from sklearn.linear_model import Lasso

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "libs") not in sys.path:
    sys.path.insert(0, str(ROOT / "libs"))
import current_model  # noqa: E402


OUT = ROOT / "data/current_model/spatial_analysis"
FIGURES = ROOT / "data/validation/current_model/spatial_analysis"
BOHR_TO_ANGSTROM = 0.529177210903
GRID_SPACING_BOHR = 2.0
GRID_UNIT_ANGSTROM = BOHR_TO_ANGSTROM * GRID_SPACING_BOHR
NONZERO_TOLERANCE = 1.0e-12
BLOCK_COLORS = {
    "electronic": "#2f6f8f",
    "electrostatic": "#b44b3f",
    "orbital": "#4f7d56",
}


def load_model_data(*, refresh_excel: bool = True):
    """Load verified inputs, optional Excel labels, and saved outer predictions.

    When ``refresh_excel`` is false, the canonical frozen metadata and response
    values are used without consulting the editable workbook.
    """
    payload = current_model.ensure_frozen_inputs()
    train_manifest = pd.read_csv(current_model.TRAIN_ROWS_PATH)
    if refresh_excel:
        payload, train, _ = current_model.refresh_inputs_from_excel(
            payload, train_manifest
        )
    else:
        train = train_manifest["row_index"].to_numpy(dtype=int)
    meta = payload["meta"].reset_index(drop=True)
    raw = {
        block: np.asarray(payload["raw_blocks"][block], dtype=float)
        for block in current_model.BLOCKS
    }
    coords = {
        block: np.asarray(payload["coords"][block], dtype=int)
        for block in current_model.BLOCKS
    }
    y = meta[current_model.TARGET].astype(float).to_numpy()
    outer = pd.read_csv(current_model.DATA_DIR / "outer_predictions.csv").sort_values("fold_id")
    if len(outer) != len(train):
        raise ValueError("Current nested outer predictions do not match the training manifest.")
    return payload, meta, raw, coords, train, y, outer


def fit_outer_coefficient(
    fold_id: int,
    holdout: int,
    alpha: float,
    raw: dict[str, np.ndarray],
    coords: dict[str, np.ndarray],
    train: np.ndarray,
    y: np.ndarray,
    reference_names: list[str],
) -> tuple[int, np.ndarray]:
    """Refit one saved outer fold and return its 321 Lasso coefficients.

    ``holdout`` and ``train`` contain zero-based rows of the frozen descriptor
    matrices.  Scaling is recomputed from ``train`` without the held-out row,
    using the saved dimensionless Lasso ``alpha`` for that fold.
    """
    fold_train = train[train != holdout]
    x, names, _ = current_model.build_features(raw, coords, fold_train)
    if names != reference_names:
        raise ValueError(f"Feature order changed in outer fold {fold_id}.")
    model = Lasso(
        alpha=float(alpha), fit_intercept=True, max_iter=200000, tol=1.0e-6
    ).fit(x[fold_train], y[fold_train])
    return fold_id, model.coef_.copy()


def fit_outer_models(
    raw: dict[str, np.ndarray],
    coords: dict[str, np.ndarray],
    train: np.ndarray,
    y: np.ndarray,
    outer: pd.DataFrame,
    feature_names: list[str],
    workers: int,
) -> np.ndarray:
    """Refit all outer folds and return ``(n_folds, n_features)`` coefficients."""
    coefficients = np.zeros((len(outer), len(feature_names)), dtype=float)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(
                fit_outer_coefficient,
                int(row.fold_id),
                int(row.holdout_index),
                float(row.selected_alpha),
                raw,
                coords,
                train,
                y,
                feature_names,
            )
            for row in outer.itertuples(index=False)
        ]
        for count, future in enumerate(as_completed(futures), start=1):
            fold_id, values = future.result()
            coefficients[fold_id] = values
            if count % 20 == 0:
                print(f"outer coefficient models {count}/{len(outer)}", flush=True)
    return coefficients


def feature_metadata(
    feature_names: list[str],
    coords: dict[str, np.ndarray],
) -> pd.DataFrame:
    """Map ordered model features to block, kind, and spatial coordinates.

    Grid coordinates are integer bin labels spaced by 2 Bohr; the derived
    ``*_angstrom`` columns contain their origin-relative display positions.
    Max/min summary features have no unique spatial coordinate and receive
    missing coordinate values.
    """
    rows: list[dict[str, object]] = []
    feature_index = 0
    for block in current_model.BLOCKS:
        selected_coords = coords[block][current_model.in_bounds(coords[block])]
        for point in selected_coords:
            x_grid, y_grid, z_grid = map(int, point)
            rows.append(
                {
                    "feature_index": feature_index,
                    "feature": feature_names[feature_index],
                    "block": block,
                    "kind": "grid",
                    "x_grid": x_grid,
                    "y_grid": y_grid,
                    "z_grid": z_grid,
                    "x_angstrom": x_grid * GRID_UNIT_ANGSTROM,
                    "y_angstrom": y_grid * GRID_UNIT_ANGSTROM,
                    "z_angstrom": z_grid * GRID_UNIT_ANGSTROM,
                    "radius_angstrom": math.sqrt(x_grid**2 + y_grid**2 + z_grid**2)
                    * GRID_UNIT_ANGSTROM,
                }
            )
            feature_index += 1
    for block in current_model.BLOCKS:
        for summary in ("max", "min"):
            rows.append(
                {
                    "feature_index": feature_index,
                    "feature": feature_names[feature_index],
                    "block": block,
                    "kind": f"summary_{summary}",
                    "x_grid": np.nan,
                    "y_grid": np.nan,
                    "z_grid": np.nan,
                    "x_angstrom": np.nan,
                    "y_angstrom": np.nan,
                    "z_angstrom": np.nan,
                    "radius_angstrom": np.nan,
                }
            )
            feature_index += 1
    if feature_index != len(feature_names):
        raise ValueError(f"Mapped {feature_index} of {len(feature_names)} features.")
    return pd.DataFrame(rows)


def add_coefficient_statistics(
    metadata: pd.DataFrame,
    full_coefficients: np.ndarray,
    outer_coefficients: np.ndarray,
    x_full: np.ndarray,
    train: np.ndarray,
) -> pd.DataFrame:
    """Attach full-fit, outer-fold stability, and realized-effect statistics.

    ``outer_coefficients`` has one row per outer model.  Realized effects use
    ``(x - mean(x_train)) * beta`` and therefore have units of kcal/mol.
    """
    table = metadata.copy()
    selected = np.abs(outer_coefficients) > NONZERO_TOLERANCE
    table["fulltrain_coefficient"] = full_coefficients
    table["fulltrain_nonzero"] = np.abs(full_coefficients) > NONZERO_TOLERANCE
    table["outer_selection_frequency"] = selected.mean(axis=0)
    table["outer_coefficient_mean"] = outer_coefficients.mean(axis=0)
    table["outer_coefficient_median_all"] = np.median(outer_coefficients, axis=0)
    table["outer_coefficient_p16"] = np.percentile(outer_coefficients, 16, axis=0)
    table["outer_coefficient_p84"] = np.percentile(outer_coefficients, 84, axis=0)
    selected_medians = []
    sign_consistency = []
    dominant_sign = []
    for index in range(outer_coefficients.shape[1]):
        values = outer_coefficients[selected[:, index], index]
        if len(values):
            selected_medians.append(float(np.median(values)))
            sign_consistency.append(float(abs(np.mean(np.sign(values)))))
            dominant_sign.append("positive" if np.median(values) > 0 else "negative")
        else:
            selected_medians.append(0.0)
            sign_consistency.append(0.0)
            dominant_sign.append("not_selected")
    table["outer_coefficient_median_when_selected"] = selected_medians
    table["outer_sign_consistency_when_selected"] = sign_consistency
    table["outer_dominant_sign"] = dominant_sign

    centered_x = x_full[train] - np.mean(x_full[train], axis=0)
    centered_effect = centered_x * full_coefficients[None, :]
    table["training_centered_effect_rms_kcal_mol"] = np.sqrt(
        np.mean(centered_effect**2, axis=0)
    )
    table["training_centered_effect_mean_abs_kcal_mol"] = np.mean(
        np.abs(centered_effect), axis=0
    )
    table["training_centered_effect_max_abs_kcal_mol"] = np.max(
        np.abs(centered_effect), axis=0
    )
    table["stability_weighted_abs_coefficient"] = (
        table["outer_selection_frequency"] * np.abs(table["fulltrain_coefficient"])
    )
    return table


def save_outer_coefficient_matrix(
    coefficients: np.ndarray,
    feature_names: list[str],
    outer: pd.DataFrame,
) -> None:
    """Save outer-fold coefficients and fold identities as a compressed NPZ."""
    np.savez_compressed(
        OUT / "outer83_feature_coefficients.npz",
        coefficients=coefficients,
        feature_names=np.asarray(feature_names, dtype=str),
        fold_id=outer["fold_id"].to_numpy(dtype=int),
        holdout_entry=outer["entry"].to_numpy(dtype=str),
    )


def save_training_effect_matrix(
    x_full: np.ndarray,
    feature_names: list[str],
    coefficients: np.ndarray,
    train: np.ndarray,
    meta: pd.DataFrame,
) -> None:
    """Save centered per-feature and per-block training effects in kcal/mol."""
    centered_x = x_full[train] - np.mean(x_full[train], axis=0)
    effects = centered_x * coefficients[None, :]
    np.savez_compressed(
        OUT / "training_centered_feature_effects.npz",
        effects=effects,
        feature_names=np.asarray(feature_names, dtype=str),
        entry=meta.loc[train, "entry"].to_numpy(dtype=str),
        name=meta.loc[train, "name"].to_numpy(dtype=str),
        InChIKey=meta.loc[train, "InChIKey"].to_numpy(dtype=str),
    )
    block_rows = meta.loc[train, ["entry", "name", "InChIKey"]].reset_index(drop=True)
    for block in current_model.BLOCKS:
        grid_mask = np.asarray(
            [name.startswith(f"{block}:grid:") for name in feature_names], dtype=bool
        )
        all_mask = np.asarray(
            [name.startswith(f"{block}:") for name in feature_names], dtype=bool
        )
        block_rows[f"{block}_centered_grid_contribution"] = effects[:, grid_mask].sum(axis=1)
        block_rows[f"{block}_centered_total_contribution"] = effects[:, all_mask].sum(axis=1)
    block_rows.to_csv(OUT / "training_centered_block_contributions.csv", index=False)


def summarize_blocks(
    table: pd.DataFrame,
    x_full: np.ndarray,
    feature_names: list[str],
    coefficients: np.ndarray,
    train: np.ndarray,
) -> pd.DataFrame:
    """Summarize coefficient norms and centered contributions by descriptor block."""
    centered_x = x_full[train] - np.mean(x_full[train], axis=0)
    rows = []
    for block in current_model.BLOCKS:
        grid_mask = np.asarray(
            [name.startswith(f"{block}:grid:") for name in feature_names], dtype=bool
        )
        all_mask = np.asarray(
            [name.startswith(f"{block}:") for name in feature_names], dtype=bool
        )
        summary_mask = all_mask & ~grid_mask
        grid_contribution = centered_x[:, grid_mask] @ coefficients[grid_mask]
        total_contribution = centered_x[:, all_mask] @ coefficients[all_mask]
        block_table = table.loc[table["block"].eq(block)]
        grid_table = block_table.loc[block_table["kind"].eq("grid")]
        rows.append(
            {
                "block": block,
                "candidate_grid_n": int(grid_mask.sum()),
                "fulltrain_nonzero_grid_n": int(np.count_nonzero(coefficients[grid_mask])),
                "fulltrain_nonzero_summary_n": int(np.count_nonzero(coefficients[summary_mask])),
                "outer_mean_selected_grid_n": float(
                    grid_table["outer_selection_frequency"].sum()
                ),
                "outer_stable_grid_n_frequency_ge_0_5": int(
                    grid_table["outer_selection_frequency"].ge(0.5).sum()
                ),
                "grid_coefficient_l1": float(np.sum(np.abs(coefficients[grid_mask]))),
                "grid_coefficient_l2": float(np.linalg.norm(coefficients[grid_mask])),
                "centered_grid_contribution_sd_kcal_mol": float(
                    np.std(grid_contribution, ddof=1)
                ),
                "centered_grid_contribution_mean_abs_kcal_mol": float(
                    np.mean(np.abs(grid_contribution))
                ),
                "centered_grid_contribution_min_kcal_mol": float(np.min(grid_contribution)),
                "centered_grid_contribution_max_kcal_mol": float(np.max(grid_contribution)),
                "centered_total_block_contribution_sd_kcal_mol": float(
                    np.std(total_contribution, ddof=1)
                ),
                "centered_total_block_contribution_mean_abs_kcal_mol": float(
                    np.mean(np.abs(total_contribution))
                ),
            }
        )
    result = pd.DataFrame(rows)
    result["grid_coefficient_l1_fraction"] = (
        result["grid_coefficient_l1"] / result["grid_coefficient_l1"].sum()
    )
    return result


def summarize_spatial_groups(grid_table: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate grid statistics into radial-Angstrom shells and folded-y layers."""
    shell_edges = [0.0, 2.0, 3.0, 4.0, 5.0, np.inf]
    shell_labels = ["0-2", "2-3", "3-4", "4-5", ">=5"]
    data = grid_table.copy()
    data["radial_shell_angstrom"] = pd.cut(
        data["radius_angstrom"], bins=shell_edges, labels=shell_labels, right=False
    )

    def aggregate(group: pd.DataFrame) -> pd.Series:
        """Reduce one descriptor shell or layer to stability/effect summaries."""
        return pd.Series(
            {
                "candidate_grid_n": len(group),
                "fulltrain_nonzero_grid_n": int(group["fulltrain_nonzero"].sum()),
                "sum_abs_coefficient": float(np.abs(group["fulltrain_coefficient"]).sum()),
                "coefficient_l2": float(np.linalg.norm(group["fulltrain_coefficient"])),
                "quadrature_centered_effect_rms_kcal_mol": float(
                    np.linalg.norm(group["training_centered_effect_rms_kcal_mol"])
                ),
                "mean_outer_selection_frequency": float(
                    group["outer_selection_frequency"].mean()
                ),
                "stable_grid_n_frequency_ge_0_5": int(
                    group["outer_selection_frequency"].ge(0.5).sum()
                ),
            }
        )

    shell = (
        data.groupby(["block", "radial_shell_angstrom"], observed=False)
        .apply(aggregate, include_groups=False)
        .reset_index()
    )
    layer = (
        data.groupby(["block", "y_grid"], observed=False)
        .apply(aggregate, include_groups=False)
        .reset_index()
    )
    return shell, layer


def summarize_xz_quadrants(grid_table: pd.DataFrame) -> pd.DataFrame:
    """Aggregate spatial statistics by signs of the aligned x and z coordinates."""
    data = grid_table.copy()
    data["x_side"] = np.where(data["x_grid"].lt(0), "x<0", "x>0")
    data["z_side"] = np.where(data["z_grid"].lt(0), "z<0", "z>0")

    def aggregate(group: pd.DataFrame) -> pd.Series:
        """Reduce one block/quadrant group to coefficient and effect summaries."""
        return pd.Series(
            {
                "candidate_grid_n": len(group),
                "fulltrain_nonzero_grid_n": int(group["fulltrain_nonzero"].sum()),
                "coefficient_sum": float(group["fulltrain_coefficient"].sum()),
                "sum_abs_coefficient": float(np.abs(group["fulltrain_coefficient"]).sum()),
                "quadrature_centered_effect_rms_kcal_mol": float(
                    np.linalg.norm(group["training_centered_effect_rms_kcal_mol"])
                ),
                "mean_outer_selection_frequency": float(
                    group["outer_selection_frequency"].mean()
                ),
            }
        )

    return (
        data.groupby(["block", "x_side", "z_side"], observed=False)
        .apply(aggregate, include_groups=False)
        .reset_index()
    )


def marker_sizes(values: np.ndarray, minimum: float = 18.0, maximum: float = 260.0) -> np.ndarray:
    """Map non-negative magnitudes to square-root-scaled marker areas."""
    values = np.asarray(values, dtype=float)
    peak = float(np.max(values)) if len(values) else 0.0
    if peak <= 0:
        return np.full(len(values), minimum)
    return minimum + (maximum - minimum) * np.sqrt(values / peak)


def plot_coefficient_layers(grid_table: pd.DataFrame, path: Path) -> None:
    """Plot signed full-training coefficients in each folded-y grid layer."""
    coefficient_limit = float(np.abs(grid_table["fulltrain_coefficient"]).max())
    norm = TwoSlopeNorm(vmin=-coefficient_limit, vcenter=0.0, vmax=coefficient_limit)
    y_layers = sorted(grid_table["y_grid"].dropna().astype(int).unique())
    fig, axes = plt.subplots(
        len(current_model.BLOCKS), len(y_layers), figsize=(10.0, 10.0), sharex=True, sharey=True
    )
    scatter = None
    for row, block in enumerate(current_model.BLOCKS):
        for column, y_layer in enumerate(y_layers):
            ax = axes[row, column]
            subset = grid_table.loc[
                grid_table["block"].eq(block) & grid_table["y_grid"].eq(y_layer)
            ]
            ax.scatter(
                subset["z_angstrom"], subset["x_angstrom"], s=14, color="#dddddd", zorder=1
            )
            active = subset.loc[subset["fulltrain_nonzero"]]
            if not active.empty:
                scatter = ax.scatter(
                    active["z_angstrom"],
                    active["x_angstrom"],
                    c=active["fulltrain_coefficient"],
                    s=marker_sizes(np.abs(active["fulltrain_coefficient"])),
                    cmap="coolwarm",
                    norm=norm,
                    edgecolor="#333333",
                    linewidth=0.35,
                    zorder=2,
                )
            if row == 0:
                ax.set_title(f"y = {y_layer} ({y_layer * GRID_UNIT_ANGSTROM:.2f} A)")
            if column == 0:
                ax.set_ylabel(f"{block}\nx [A]")
            if row == len(current_model.BLOCKS) - 1:
                ax.set_xlabel("z [A]")
            ax.axhline(0, color="#bbbbbb", linewidth=0.5)
            ax.axvline(0, color="#bbbbbb", linewidth=0.5)
            ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.35)
            ax.set_aspect("equal")
    if scatter is not None:
        colorbar = fig.colorbar(scatter, ax=axes, fraction=0.025, pad=0.02)
        colorbar.set_label("Lasso coefficient [kcal/mol per block-scaled unit]")
    fig.suptitle("Spatial full-training coefficients; x is vertical", y=0.995)
    fig.subplots_adjust(left=0.10, right=0.88, bottom=0.07, top=0.95, wspace=0.16, hspace=0.20)
    fig.savefig(path, dpi=350)
    plt.close(fig)


def plot_stability_layers(grid_table: pd.DataFrame, path: Path) -> None:
    """Plot outer-fold selection frequency and dominant coefficient sign by cell."""
    y_layers = sorted(grid_table["y_grid"].dropna().astype(int).unique())
    fig, axes = plt.subplots(
        len(current_model.BLOCKS), len(y_layers), figsize=(10.0, 10.0), sharex=True, sharey=True
    )
    norm = Normalize(vmin=0.0, vmax=1.0)
    scatter = None
    for row, block in enumerate(current_model.BLOCKS):
        for column, y_layer in enumerate(y_layers):
            ax = axes[row, column]
            subset = grid_table.loc[
                grid_table["block"].eq(block) & grid_table["y_grid"].eq(y_layer)
            ]
            scatter = ax.scatter(
                subset["z_angstrom"],
                subset["x_angstrom"],
                c=subset["outer_selection_frequency"],
                s=marker_sizes(subset["outer_selection_frequency"], 14, 240),
                cmap="viridis",
                norm=norm,
                edgecolor=np.where(
                    subset["outer_dominant_sign"].eq("positive"), "#b44b3f",
                    np.where(subset["outer_dominant_sign"].eq("negative"), "#2f6f8f", "#bbbbbb"),
                ),
                linewidth=0.8,
            )
            if row == 0:
                ax.set_title(f"y = {y_layer} ({y_layer * GRID_UNIT_ANGSTROM:.2f} A)")
            if column == 0:
                ax.set_ylabel(f"{block}\nx [A]")
            if row == len(current_model.BLOCKS) - 1:
                ax.set_xlabel("z [A]")
            ax.axhline(0, color="#bbbbbb", linewidth=0.5)
            ax.axvline(0, color="#bbbbbb", linewidth=0.5)
            ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.35)
            ax.set_aspect("equal")
    colorbar = fig.colorbar(scatter, ax=axes, fraction=0.025, pad=0.02)
    colorbar.set_label("Selection frequency across 83 outer models")
    fig.suptitle("Outer-model spatial stability; red edge positive, blue edge negative", y=0.995)
    fig.subplots_adjust(left=0.10, right=0.88, bottom=0.07, top=0.95, wspace=0.16, hspace=0.20)
    fig.savefig(path, dpi=350)
    plt.close(fig)


def plot_realized_effect_layers(grid_table: pd.DataFrame, path: Path) -> None:
    """Plot coefficient sign with marker area proportional to training RMS effect."""
    y_layers = sorted(grid_table["y_grid"].dropna().astype(int).unique())
    fig, axes = plt.subplots(
        len(current_model.BLOCKS), len(y_layers), figsize=(10.0, 10.0), sharex=True, sharey=True
    )
    active_all = grid_table.loc[grid_table["fulltrain_nonzero"]]
    coefficient_limit = float(np.abs(active_all["fulltrain_coefficient"]).max())
    norm = TwoSlopeNorm(vmin=-coefficient_limit, vcenter=0.0, vmax=coefficient_limit)
    scatter = None
    for row, block in enumerate(current_model.BLOCKS):
        for column, y_layer in enumerate(y_layers):
            ax = axes[row, column]
            subset = grid_table.loc[
                grid_table["block"].eq(block) & grid_table["y_grid"].eq(y_layer)
            ]
            ax.scatter(subset["z_angstrom"], subset["x_angstrom"], s=14, color="#dddddd")
            active = subset.loc[subset["fulltrain_nonzero"]]
            if not active.empty:
                scatter = ax.scatter(
                    active["z_angstrom"],
                    active["x_angstrom"],
                    c=active["fulltrain_coefficient"],
                    s=marker_sizes(active["training_centered_effect_rms_kcal_mol"]),
                    cmap="coolwarm",
                    norm=norm,
                    edgecolor="#333333",
                    linewidth=0.35,
                )
            if row == 0:
                ax.set_title(f"y = {y_layer} ({y_layer * GRID_UNIT_ANGSTROM:.2f} A)")
            if column == 0:
                ax.set_ylabel(f"{block}\nx [A]")
            if row == len(current_model.BLOCKS) - 1:
                ax.set_xlabel("z [A]")
            ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.35)
            ax.set_aspect("equal")
    if scatter is not None:
        colorbar = fig.colorbar(scatter, ax=axes, fraction=0.025, pad=0.02)
        colorbar.set_label("Coefficient sign and magnitude; marker size = training RMS effect")
    fig.suptitle("Realized spatial effects on centered training descriptors", y=0.995)
    fig.subplots_adjust(left=0.10, right=0.88, bottom=0.07, top=0.95, wspace=0.16, hspace=0.20)
    fig.savefig(path, dpi=350)
    plt.close(fig)


def plot_coefficients_3d(grid_table: pd.DataFrame, path: Path) -> None:
    """Plot nonzero coefficients in the aligned three-dimensional model frame."""
    coefficient_limit = float(np.abs(grid_table["fulltrain_coefficient"]).max())
    norm = TwoSlopeNorm(vmin=-coefficient_limit, vcenter=0.0, vmax=coefficient_limit)
    fig = plt.figure(figsize=(12.0, 4.2))
    scatter = None
    for index, block in enumerate(current_model.BLOCKS, start=1):
        ax = fig.add_subplot(1, 3, index, projection="3d")
        subset = grid_table.loc[grid_table["block"].eq(block)]
        active = subset.loc[subset["fulltrain_nonzero"]]
        ax.scatter(
            subset["z_angstrom"], subset["y_angstrom"], subset["x_angstrom"],
            color="#dddddd", s=7, alpha=0.35,
        )
        if not active.empty:
            scatter = ax.scatter(
                active["z_angstrom"], active["y_angstrom"], active["x_angstrom"],
                c=active["fulltrain_coefficient"],
                s=marker_sizes(np.abs(active["fulltrain_coefficient"]), 18, 180),
                cmap="coolwarm", norm=norm, edgecolor="#333333", linewidth=0.3,
            )
        ax.set_title(block)
        ax.set_xlabel("z [A]")
        ax.set_ylabel("y [A]")
        ax.set_zlabel("x [A]")
        ax.view_init(elev=20, azim=-55)
    if scatter is not None:
        colorbar = fig.colorbar(scatter, ax=fig.axes, fraction=0.025, pad=0.06)
        colorbar.set_label("Lasso coefficient")
    fig.suptitle("3D spatial coefficient distribution; x axis is vertical")
    fig.subplots_adjust(left=0.02, right=0.88, bottom=0.03, top=0.86, wspace=0.10)
    fig.savefig(path, dpi=350)
    plt.close(fig)


def plot_distance_summary(shell: pd.DataFrame, path: Path) -> None:
    """Plot coefficient L1 mass and quadrature RMS effect by radial shell."""
    labels = ["0-2", "2-3", "3-4", "4-5", ">=5"]
    positions = np.arange(len(labels), dtype=float)
    width = 0.24
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8))
    for offset, block in enumerate(current_model.BLOCKS):
        subset = shell.loc[shell["block"].eq(block)].set_index("radial_shell_angstrom")
        subset = subset.reindex(labels)
        axes[0].bar(
            positions + (offset - 1) * width,
            subset["sum_abs_coefficient"],
            width=width,
            color=BLOCK_COLORS[block],
            label=block,
        )
        axes[1].bar(
            positions + (offset - 1) * width,
            subset["quadrature_centered_effect_rms_kcal_mol"],
            width=width,
            color=BLOCK_COLORS[block],
            label=block,
        )
    axes[0].set_ylabel("Sum of absolute grid coefficients")
    axes[1].set_ylabel("Quadrature RMS grid effect [kcal/mol]")
    for ax in axes:
        ax.set_xticks(positions)
        ax.set_xticklabels(labels)
        ax.set_xlabel("Distance from origin [A]")
        ax.grid(axis="y", color="#dddddd", linewidth=0.6)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=350)
    plt.close(fig)


def plot_block_contributions(
    x_full: np.ndarray,
    feature_names: list[str],
    coefficients: np.ndarray,
    train: np.ndarray,
    path: Path,
) -> None:
    """Plot centered training contributions, in kcal/mol, for each model block."""
    centered_x = x_full[train] - np.mean(x_full[train], axis=0)
    distributions = []
    for block in current_model.BLOCKS:
        mask = np.asarray([name.startswith(f"{block}:") for name in feature_names])
        distributions.append(centered_x[:, mask] @ coefficients[mask])
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    violin = ax.violinplot(distributions, showmeans=False, showmedians=True, widths=0.72)
    for body, block in zip(violin["bodies"], current_model.BLOCKS):
        body.set_facecolor(BLOCK_COLORS[block])
        body.set_edgecolor("#333333")
        body.set_alpha(0.75)
    violin["cmedians"].set_color("#222222")
    ax.axhline(0, color="#555555", linewidth=0.8)
    ax.set_xticks(np.arange(1, len(current_model.BLOCKS) + 1))
    ax.set_xticklabels(current_model.BLOCKS)
    ax.set_ylabel("Centered block contribution [kcal/mol]")
    ax.tick_params(axis="x", labelsize=12)
    ax.tick_params(axis="y", labelsize=10.5)
    ax.yaxis.label.set_size(12)
    ax.grid(axis="y", color="#dddddd", linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=350)
    plt.close(fig)


def write_readme(block_summary: pd.DataFrame, grid_table: pd.DataFrame) -> None:
    """Write interpretation notes and key block/grid counts for derived outputs."""
    lines = [
        "# Current-model spatial contribution analysis",
        "",
        "The current 321-feature model was refit without changing its inputs or alpha.",
        "Grid coordinates are reported in model grid units and Angstrom; one grid unit",
        f"is 2 Bohr = {GRID_UNIT_ANGSTROM:.6f} Angstrom. The y coordinate is the model's",
        "folded coordinate. Spatial plots place x vertically.",
        "",
        "## Full-training spatial grid counts",
        "",
        "| Block | Nonzero grids | Stable grids (outer frequency >= 0.5) | Grid L1 fraction | Centered total-block SD |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in block_summary.itertuples(index=False):
        lines.append(
            f"| {row.block} | {row.fulltrain_nonzero_grid_n} | "
            f"{row.outer_stable_grid_n_frequency_ge_0_5} | "
            f"{row.grid_coefficient_l1_fraction:.3f} | "
            f"{row.centered_total_block_contribution_sd_kcal_mol:.3f} kcal/mol |"
        )
    stable = grid_table.loc[
        grid_table["outer_selection_frequency"].ge(0.5)
    ].sort_values(["block", "outer_selection_frequency"], ascending=[True, False])
    lines.extend(
        [
            "",
            "## Interpretation rules",
            "",
            "- A positive coefficient raises predicted DeltaDeltaG when that scaled grid value increases; a negative coefficient lowers it.",
            "- Electrostatic grid values are signed, so coefficient sign alone is not a potential-sign assignment.",
            "- Marker size in the realized-effect map uses the RMS of beta_j times the centered training feature.",
            "- Selection frequency is more reliable than one full-fit coefficient for correlated neighboring grids.",
            "- Block contribution spreads are correlated and must not be read as additive percentages of explained variance.",
            "",
            f"Stable spatial grids with outer selection frequency >= 0.5: {len(stable)}.",
            "The compressed training effect matrix permits later substrate-specific spatial maps without refitting.",
        ]
    )
    (OUT / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    """Refit spatial models and regenerate all tabular and graphical analyses."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument(
        "--no-excel-refresh",
        action="store_true",
        help="Use frozen metadata without synchronizing the editable workbook.",
    )
    args = parser.parse_args()
    workers = min(max(args.workers, 1), 20)
    OUT.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    _, meta, raw, coords, train, y, outer = load_model_data(
        refresh_excel=not args.no_excel_refresh
    )
    x_full, feature_names, _ = current_model.build_features(raw, coords, train)
    full_alpha = float(pd.read_csv(current_model.DATA_DIR / "summary.csv").iloc[0]["fulltrain_selected_alpha"])
    full_model = Lasso(
        alpha=full_alpha, fit_intercept=True, max_iter=200000, tol=1.0e-6
    ).fit(x_full[train], y[train])
    outer_coefficients = fit_outer_models(
        raw, coords, train, y, outer, feature_names, workers
    )
    save_outer_coefficient_matrix(outer_coefficients, feature_names, outer)
    save_training_effect_matrix(
        x_full, feature_names, full_model.coef_, train, meta
    )

    metadata = feature_metadata(feature_names, coords)
    table = add_coefficient_statistics(
        metadata, full_model.coef_, outer_coefficients, x_full, train
    )
    table.to_csv(OUT / "feature_coefficient_and_effect_statistics.csv", index=False)
    grid_table = table.loc[table["kind"].eq("grid")].copy()
    summary_table = table.loc[~table["kind"].eq("grid")].copy()
    grid_table.to_csv(OUT / "spatial_grid_statistics.csv", index=False)
    summary_table.to_csv(OUT / "summary_feature_statistics.csv", index=False)

    block_summary = summarize_blocks(
        table, x_full, feature_names, full_model.coef_, train
    )
    block_summary.to_csv(OUT / "block_contribution_summary.csv", index=False)
    shell, layer = summarize_spatial_groups(grid_table)
    shell.to_csv(OUT / "radial_shell_contribution_summary.csv", index=False)
    layer.to_csv(OUT / "y_layer_contribution_summary.csv", index=False)
    summarize_xz_quadrants(grid_table).to_csv(
        OUT / "xz_quadrant_contribution_summary.csv", index=False
    )
    grid_table.sort_values(
        ["stability_weighted_abs_coefficient", "training_centered_effect_rms_kcal_mol"],
        ascending=False,
    ).head(30).to_csv(OUT / "top_30_spatial_grids.csv", index=False)

    plot_coefficient_layers(grid_table, FIGURES / "spatial_grid_fulltrain_coefficients_by_y.png")
    plot_stability_layers(grid_table, FIGURES / "spatial_grid_outer_selection_frequency_by_y.png")
    plot_realized_effect_layers(grid_table, FIGURES / "spatial_grid_realized_effect_by_y.png")
    plot_coefficients_3d(grid_table, FIGURES / "spatial_grid_fulltrain_coefficients_3d.png")
    plot_distance_summary(shell, FIGURES / "spatial_grid_effect_by_distance.png")
    plot_block_contributions(
        x_full, feature_names, full_model.coef_, train,
        FIGURES / "centered_block_contribution_violins.png",
    )
    write_readme(block_summary, grid_table)

    audit = {
        "training_n": int(len(train)),
        "feature_n": int(len(feature_names)),
        "spatial_grid_n": int(len(grid_table)),
        "outer_model_n": int(len(outer_coefficients)),
        "grid_spacing_bohr": GRID_SPACING_BOHR,
        "grid_unit_angstrom": GRID_UNIT_ANGSTROM,
        "fulltrain_alpha": full_alpha,
        "coefficient_nonzero_tolerance": NONZERO_TOLERANCE,
        "model_changed": False,
        "metadata_join_key": "InChIKey",
    }
    (OUT / "analysis_specification.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8"
    )
    print(block_summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
