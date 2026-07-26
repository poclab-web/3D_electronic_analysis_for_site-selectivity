"""Export substrate-specific spatial contributions from the current model.

The exported scalar at each model grid point is

    (x_substrate - mean(x_training)) * beta

in kcal/mol. The model folds the molecular grid onto positive y. For display,
each folded effect is divided equally between its +y and -y cells; this is the
unique symmetry-preserving expansion that does not invent unavailable facial
information. Max/min summary-feature effects are reported in CSV rather than
placed in the cubes.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "libs") not in sys.path:
    sys.path.insert(0, str(ROOT / "libs"))

import current_model  # noqa: E402


DEFAULT_OUTPUT = ROOT / "data" / "validation" / "current_model" / "contribution_cubes"
A24_OUTPUT = (
    ROOT
    / "data"
    / "validation"
    / "current_model"
    / "contribution_cubes_relative_to_A24"
)
GRID_SPACING_BOHR = 2.0
GAUSSVIEW_LAUNCHER = ROOT / "scripts" / "viewers" / "open_gaussview_surface.sh"
GAUSSVIEW_ISOVALUE = 0.020


def _safe_label(value: object) -> str:
    """Return a filesystem-safe display label without altering identities."""
    label = re.sub(r"[^A-Za-z0-9_.()-]+", "_", str(value).strip())
    return label.strip("_") or "substrate"


def _load_display_geometries() -> dict[str, dict[str, object]]:
    """Load frozen display atoms keyed by InChIKey.

    Coordinates are in Bohr and are used only as a visual reference inside
    derived contribution cubes.  Descriptor values never depend on this
    geometry cache.
    """
    document = json.loads(
        current_model.DISPLAY_GEOMETRIES_PATH.read_text(encoding="utf-8")
    )
    if document.get("units") != "bohr":
        raise ValueError("Display-geometry coordinates must be stored in Bohr.")
    geometries = document.get("geometries")
    if not isinstance(geometries, dict) or len(geometries) != 83:
        raise ValueError("Expected display geometries for all 83 training substrates.")
    return geometries


def _write_cube(
    path: Path,
    title: str,
    coords: np.ndarray,
    values: np.ndarray,
    atoms: list[tuple[int, float, float, float, float]],
    effect_description: str,
) -> None:
    """Write contributions at the physical centers of the coarse-grid cells."""
    labels = [np.unique(coords[:, axis]).astype(int) for axis in range(3)]
    if any(np.any(axis_labels == 0) for axis_labels in labels):
        raise ValueError("Current-model contribution coordinates must exclude grid 0.")

    # Binning uses ceil(r/step) for r > 0 and floor(r/step) for r < 0.
    # Therefore bin i is centered at (i - sign(i)/2) * step. Removing label 0
    # makes the centers a continuous regular lattice across the molecular plane.
    centers = [
        (axis_labels - 0.5 * np.sign(axis_labels)) * GRID_SPACING_BOHR
        for axis_labels in labels
    ]
    for axis_centers in centers:
        if len(axis_centers) > 1 and not np.allclose(
            np.diff(axis_centers), GRID_SPACING_BOHR, rtol=0.0, atol=1.0e-12
        ):
            raise ValueError(f"Grid labels do not form a regular 2-Bohr axis: {axis_centers}")

    shape = tuple(len(axis_labels) for axis_labels in labels)
    lookups = [
        {int(label): index for index, label in enumerate(axis_labels)}
        for axis_labels in labels
    ]
    volume = np.zeros(shape, dtype=float)
    for coord, value in zip(coords.astype(int), values):
        index = tuple(lookups[axis][int(coord[axis])] for axis in range(3))
        volume[index] += float(value)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii", newline="\n") as handle:
        handle.write(f"{title}\n")
        handle.write(f"{effect_description}; units kcal/mol per model grid\n")
        origin = np.asarray([axis_centers[0] for axis_centers in centers], dtype=float)
        handle.write(
            f"{len(atoms):5d}{origin[0]:13.6f}{origin[1]:13.6f}{origin[2]:13.6f}\n"
        )
        handle.write(f"{shape[0]:5d}{GRID_SPACING_BOHR:13.6f}{0.0:13.6f}{0.0:13.6f}\n")
        handle.write(f"{shape[1]:5d}{0.0:13.6f}{GRID_SPACING_BOHR:13.6f}{0.0:13.6f}\n")
        handle.write(f"{shape[2]:5d}{0.0:13.6f}{0.0:13.6f}{GRID_SPACING_BOHR:13.6f}\n")
        for atomic_number, charge, x, y, z in atoms:
            handle.write(
                f"{atomic_number:5d}{charge:13.6f}{x:13.6f}{y:13.6f}{z:13.6f}\n"
            )
        flattened = volume.ravel(order="C")
        for start in range(0, len(flattened), 6):
            handle.write("".join(f"{value:16.8E}" for value in flattened[start:start + 6]))
            handle.write("\n")


def _write_gaussview_launcher(cube_path: Path, block: str) -> Path:
    """Write a Finder-double-clickable launcher beside a contribution cube."""
    launcher_path = cube_path.with_name(f"VIEW_{block}.command")
    relative_launcher = os.path.relpath(GAUSSVIEW_LAUNCHER, launcher_path.parent)
    launcher_path.write_text(
        "\n".join(
            (
                "#!/bin/zsh",
                "set -eu",
                f'exec "${{0:A:h}}/{relative_launcher}" "${{0:A:h}}/{cube_path.name}" "{block}"',
                "",
            )
        ),
        encoding="utf-8",
    )
    launcher_path.chmod(0o755)
    return launcher_path


def _expand_folded_y(
    coords: np.ndarray,
    values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Expand |y| bins symmetrically while preserving the folded contribution."""
    coords = np.asarray(coords, dtype=int)
    values = np.asarray(values, dtype=float)
    if np.any(coords[:, 1] < 0):
        raise ValueError("Expected non-negative folded y coordinates.")
    zero = coords[:, 1] == 0
    positive = ~zero
    negative_coords = coords[positive].copy()
    negative_coords[:, 1] *= -1
    expanded_coords = np.vstack((negative_coords, coords[zero], coords[positive]))
    expanded_values = np.concatenate(
        (values[positive] * 0.5, values[zero], values[positive] * 0.5)
    )
    if not np.isclose(expanded_values.sum(), values.sum(), rtol=0.0, atol=1.0e-12):
        raise ValueError("Symmetric y expansion did not preserve the grid contribution.")
    return expanded_coords, expanded_values


def export_contribution_cubes(
    meta: pd.DataFrame,
    raw: dict[str, np.ndarray],
    coords: dict[str, np.ndarray],
    train: np.ndarray,
    x_full: np.ndarray,
    feature_names: list[str],
    model: Lasso,
    output_dir: Path = DEFAULT_OUTPUT,
    rows: np.ndarray | None = None,
    reference_row: int | None = None,
) -> pd.DataFrame:
    """Export current-model spatial effects and return a substrate manifest."""
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = np.asarray(train if rows is None else rows, dtype=int)
    predictions = model.predict(x_full)
    if reference_row is None:
        reference_vector = np.mean(x_full[train], axis=0)
        reference_prediction = float(np.mean(predictions[train]))
        reference_entry = "training_mean"
        reference_name = "83-point training mean"
        cube_suffix = "centered_contribution"
        effect_formula = "(x_substrate - mean(x_training)) * beta"
        effect_description = "Training-mean-centered current-model contribution"
    else:
        reference_row = int(reference_row)
        reference_vector = x_full[reference_row]
        reference_prediction = float(predictions[reference_row])
        reference_entry = str(meta.at[reference_row, "entry"])
        reference_name = str(meta.at[reference_row, "name"])
        cube_suffix = f"relative_to_{_safe_label(reference_entry)}"
        effect_formula = f"(x_substrate - x_{reference_entry}) * beta"
        effect_description = f"Current-model contribution relative to {reference_entry}"
    effects = (x_full - reference_vector) * np.asarray(model.coef_, dtype=float)[None, :]
    display_geometries = _load_display_geometries()
    manifest_rows: list[dict[str, object]] = []

    grid_masks = {
        block: np.asarray(
            [name.startswith(f"{block}:grid:") for name in feature_names], dtype=bool
        )
        for block in current_model.BLOCKS
    }
    summary_masks = {
        block: np.asarray(
            [name.startswith(f"{block}:summary:") for name in feature_names], dtype=bool
        )
        for block in current_model.BLOCKS
    }

    for row_index in rows:
        entry = str(meta.at[row_index, "entry"])
        name = str(meta.at[row_index, "name"])
        inchikey = str(meta.at[row_index, "InChIKey"])
        substrate_dir = output_dir / f"{_safe_label(entry)}_{inchikey}"
        if inchikey not in display_geometries:
            raise ValueError(f"No frozen display geometry for {entry} ({inchikey}).")
        geometry = display_geometries[inchikey]
        conf_id = str(geometry["dominant_conformer"])
        atoms = [
            (
                int(atom[0]),
                float(atom[1]),
                float(atom[2]),
                float(atom[3]),
                float(atom[4]),
            )
            for atom in geometry["atoms"]
        ]
        record: dict[str, object] = {
            "row_index": int(row_index),
            "entry": entry,
            "name": name,
            "InChIKey": inchikey,
            "dominant_conformer": conf_id,
            "atom_geometry_source": str(
                current_model.DISPLAY_GEOMETRIES_PATH.relative_to(ROOT)
            ),
            "atom_count": len(atoms),
            "effect_reference_entry": reference_entry,
            "effect_reference_name": reference_name,
            "prediction_difference_from_reference_kcal_mol": float(
                predictions[row_index] - reference_prediction
            ),
        }
        if reference_row is None:
            record["centered_prediction_kcal_mol"] = record[
                "prediction_difference_from_reference_kcal_mol"
            ]
        for block in current_model.BLOCKS:
            grid_mask = grid_masks[block]
            summary_mask = summary_masks[block]
            selected_coords = np.asarray(coords[block])[current_model.in_bounds(coords[block])]
            grid_effects = effects[row_index, grid_mask]
            display_coords, display_effects = _expand_folded_y(
                selected_coords, grid_effects
            )
            cube_path = substrate_dir / f"{block}_{cube_suffix}.cube"
            _write_cube(
                cube_path,
                f"{entry} {name}: {block} contribution vs {reference_entry}",
                display_coords,
                display_effects,
                atoms,
                effect_description,
            )
            launcher_path = _write_gaussview_launcher(cube_path, block)
            record[f"{block}_cube"] = str(cube_path.relative_to(ROOT))
            record[f"{block}_gaussview_launcher"] = str(
                launcher_path.relative_to(ROOT)
            )
            record[f"{block}_grid_contribution_kcal_mol"] = float(grid_effects.sum())
            record[f"{block}_cube_value_sum_kcal_mol"] = float(display_effects.sum())
            record[f"{block}_summary_contribution_kcal_mol"] = float(
                effects[row_index, summary_mask].sum()
            )
            record[f"{block}_total_contribution_kcal_mol"] = float(
                effects[row_index, grid_mask | summary_mask].sum()
            )
        reconstructed = sum(
            float(record[f"{block}_total_contribution_kcal_mol"])
            for block in current_model.BLOCKS
        )
        record["reconstructed_prediction_difference_kcal_mol"] = reconstructed
        if reference_row is None:
            record["reconstructed_centered_prediction_kcal_mol"] = reconstructed
        if not np.isclose(
            reconstructed,
            float(record["prediction_difference_from_reference_kcal_mol"]),
            rtol=0.0,
            atol=1.0e-8,
        ):
            raise ValueError(f"Contribution reconstruction failed for {entry}.")
        manifest_rows.append(record)

    exported = pd.DataFrame(manifest_rows)
    manifest_path = output_dir / "contribution_cube_manifest.csv"
    manifest = exported
    if len(rows) < len(train) and manifest_path.exists():
        existing = pd.read_csv(manifest_path)
        retained = existing.loc[
            ~existing["InChIKey"].astype(str).isin(
                set(exported["InChIKey"].astype(str))
            )
        ]
        manifest = pd.concat((retained, exported), ignore_index=True, sort=False)
    manifest = manifest.sort_values("row_index").reset_index(drop=True)
    manifest.to_csv(manifest_path, index=False)
    reference_sentence = (
        "the 83-point training mean"
        if reference_row is None
        else f"{reference_entry} ({reference_name})"
    )
    (output_dir / "README.md").write_text(
        "\n".join(
            (
                "# Current-model contribution cubes",
                "",
                f"Reference: {reference_sentence}.",
                "These cubes contain substrate-specific spatial contribution differences",
                f"in kcal/mol per model grid: `{effect_formula}`.",
                "They are model-effect maps, not electron-density or orbital-amplitude cubes.",
                "",
                "- Grid spacing: 2 Bohr (1.05835 Angstrom).",
                "- Cube coordinates and atom coordinates are both written in Bohr.",
                "- Each bin is placed at its physical cell center: (i - sign(i)/2) * 2 Bohr.",
                "- Coordinate frame: the aligned model frame; +x follows carbonyl C to O.",
                "- The folded |y| effect is split equally over +y and -y for display.",
                "- This symmetric y expansion preserves each block's contribution sum exactly.",
                "- Grid label 0 is absent by construction and is not emitted as an empty slice.",
                "- Atomic coordinates use the highest-Boltzmann-weight conformer when available.",
                "- The descriptor is Boltzmann averaged; the atom geometry is only a display reference.",
                "- Max/min summary-feature effects are not spatial and are listed in the manifest CSV.",
                "- Positive and negative values respectively raise and lower predicted DeltaDeltaG",
                f"  relative to {reference_sentence}.",
                "- Double-click `VIEW_electronic.command`, `VIEW_electrostatic.command`, or",
                "  `VIEW_orbital.command` to open GaussView and create the signed surface.",
                f"- The automated absolute isovalue is {GAUSSVIEW_ISOVALUE:.3f} kcal/mol per grid.",
                "- macOS may request Accessibility permission for Terminal on the first launch.",
                "- In an A24-relative export, A24 itself is identically zero and has no surface.",
                "",
            )
        ),
        encoding="utf-8",
    )
    with (output_dir / "cube_format_audit.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "format": "Gaussian cube scalar field",
                "grid_spacing_bohr": GRID_SPACING_BOHR,
                "blocks": list(current_model.BLOCKS),
                "substrate_count": int(len(manifest)),
                "cube_count": int(len(manifest) * len(current_model.BLOCKS)),
                "effect_definition": effect_formula,
                "effect_reference_entry": reference_entry,
                "effect_reference_name": reference_name,
                "model_spatial_domain": "positive-y folded compact grid",
                "cube_spatial_domain": "full-y symmetric display grid",
                "cell_center_definition": "(i - sign(i)/2) * 2 Bohr",
                "y_expansion": "half of each folded effect at +y and half at -y",
                "gaussview_quick_surface_isovalue_kcal_mol_per_grid": GAUSSVIEW_ISOVALUE,
                "gaussview_quick_surface_launcher": str(
                    GAUSSVIEW_LAUNCHER.relative_to(ROOT)
                ),
            },
            handle,
            indent=2,
        )
        handle.write("\n")
    return exported


def main() -> None:
    """Fit the full model and export selected contribution-cube displays."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--entries",
        default="",
        help="Comma-separated current entry labels; default exports all 83 training rows.",
    )
    parser.add_argument(
        "--reference-entry",
        default="training_mean",
        help="Reference entry label, for example A24; default uses the training mean.",
    )
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    payload = current_model.ensure_frozen_inputs()
    train_manifest = pd.read_csv(current_model.TRAIN_ROWS_PATH)
    payload, train, _ = current_model.refresh_inputs_from_excel(
        payload, train_manifest
    )
    meta = payload["meta"].copy().reset_index(drop=True)
    raw = {
        block: np.asarray(payload["raw_blocks"][block], dtype=float)
        for block in current_model.BLOCKS
    }
    coords = {
        block: np.asarray(payload["coords"][block], dtype=int)
        for block in current_model.BLOCKS
    }
    x_full, feature_names, _ = current_model.build_features(raw, coords, train)
    alpha_path = pd.read_csv(
        current_model.MODEL_RESULTS_DIR / "fulltrain_inner_alpha_path.csv"
    )
    alpha = float(alpha_path.loc[alpha_path["current_style_loocv_rmse"].idxmin(), "alpha"])
    y = meta[current_model.TARGET].astype(float).to_numpy()
    model = Lasso(alpha=alpha, fit_intercept=True, max_iter=200000, tol=1.0e-6).fit(
        x_full[train], y[train]
    )
    rows = train
    if args.entries:
        requested = [value.strip() for value in args.entries.split(",") if value.strip()]
        entry_to_row = {str(meta.at[index, "entry"]): int(index) for index in train}
        missing = [entry for entry in requested if entry not in entry_to_row]
        if missing:
            raise ValueError(f"Entries are not in current training data: {missing}")
        rows = np.asarray([entry_to_row[entry] for entry in requested], dtype=int)
    reference_row = None
    output_dir = args.output_dir.resolve() if args.output_dir else DEFAULT_OUTPUT
    if args.reference_entry != "training_mean":
        matches = train[meta.loc[train, "entry"].astype(str).eq(args.reference_entry)]
        if len(matches) != 1:
            raise ValueError(
                f"Reference entry must identify exactly one training row: {args.reference_entry}"
            )
        reference_row = int(matches[0])
        if args.output_dir is None:
            output_dir = (
                A24_OUTPUT
                if args.reference_entry == "A24"
                else DEFAULT_OUTPUT.with_name(
                    f"contribution_cubes_relative_to_{_safe_label(args.reference_entry)}"
                )
            )
    manifest = export_contribution_cubes(
        meta,
        raw,
        coords,
        train,
        x_full,
        feature_names,
        model,
        output_dir=output_dir,
        rows=rows,
        reference_row=reference_row,
    )
    print(
        f"Exported {len(manifest)} substrates and {len(manifest) * len(current_model.BLOCKS)} cubes "
        f"to {output_dir}"
    )


if __name__ == "__main__":
    main()
