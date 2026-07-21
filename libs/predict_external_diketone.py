"""Predict a newly added diketone series with the current model.

The input rows are read from ``data/Details_of_experimental_results.xlsx`` at
runtime.  Thus, the calculation follows the entry labels in the current Excel
file rather than any historical row number.  Quantum-chemical files are made
only for the distinct structures required by the requested series.

Example
-------
``python libs/predict_external_diketone.py --series x --run-quantum --molecule-root <external>/x_series``

When ``--orbital-npz`` is omitted, the projected-orbital block is generated
from the completed single-point fchk files with the same Boltzmann weighting
and 2-Bohr full-grid binning as the current model.

By default, reusable metadata and descriptor arrays remain in the ignored
``data/validation/external_diketones/<series>_series/portable_cache`` together
with prediction tables and plots. The explicit ``--promote-inputs`` option may
write reviewed x/y arrays to ``data/current_model/inputs/external_diketones``.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXCEL = ROOT / "data" / "Details_of_experimental_results.xlsx"
ENTRY_ORDER = ("1", "2", "3", "4", "13", "14", "23", "24", "31", "32", "41", "42")
EE_ES_BLOCKS = ("electronic", "electrostatic")

sys.path.insert(0, str(ROOT))
from libs import calc_mol, current_model, dataset, diketone_metrics  # noqa: E402
from libs.current_model_support import (  # noqa: E402
    conformer_helpers,
    ecomfa_transform,
    homo_projected_orbital,
    orbital_grid,
)


def normalize_entry(value: object) -> str:
    """Return a lowercase entry label without Excel's trailing ``.0``."""
    return str(value).strip().lower().removesuffix(".0")


def select_series_rows(excel_path: Path, series: str) -> pd.DataFrame:
    """Load one complete 12-pathway diketone network from the active Excel."""
    rows = dataset.common(str(excel_path)).copy()
    rows["entry_normalized"] = rows["entry"].map(normalize_entry)
    expected = [f"{series}{suffix}" for suffix in ENTRY_ORDER]
    selected = rows.loc[rows["entry_normalized"].isin(expected)].copy()
    selected["suffix"] = selected["entry_normalized"].str[len(series) :]
    selected = selected.set_index("suffix").reindex(ENTRY_ORDER).reset_index()
    missing = selected.loc[selected["SMILES"].isna(), "suffix"].tolist()
    if missing:
        raise ValueError(
            f"The {series.upper()} series is incomplete in {excel_path.name}; missing {missing}."
        )
    selected["entry"] = [f"{series}{suffix}" for suffix in ENTRY_ORDER]
    selected["test"] = 1
    return selected.loc[:, list(current_model.EXTERNAL_METADATA_COLUMNS)].copy()


def ensure_quantum_files(
    rows: pd.DataFrame,
    molecule_root: Path,
    run_quantum: bool,
    workers: int,
) -> None:
    """Generate the Gaussian/cube inputs once per distinct InChIKey."""
    unique = rows.drop_duplicates("InChIKey")
    missing: list[str] = []
    pending: list[dict[str, object]] = []
    for row in unique.itertuples(index=False):
        molecule_dir = molecule_root / str(row.InChIKey)
        # A local interactive runner can be interrupted after cube generation
        # but before calc_mol writes ``done``.  The eCoMFA descriptor only
        # needs a normal opt log plus its density and ESP cubes, so recognise
        # that completed state and avoid rerunning an expensive calculation.
        opt_logs = conformer_helpers.discover_conformer_logs(molecule_dir)
        has_ee_cubes = bool(opt_logs) and all(
            calc_mol.normal_termination(log)
            and log.with_name(log.name.replace("opt", "Dt", 1)).with_suffix(".cube").exists()
            and log.with_name(log.name.replace("opt", "ESP", 1)).with_suffix(".cube").exists()
            and log.with_name(log.name.replace("opt", "sp", 1)).with_suffix(".fchk").exists()
            for log in opt_logs
        )
        if has_ee_cubes:
            continue
        (molecule_dir / "done").unlink(missing_ok=True)
        if not run_quantum:
            missing.append(f"{row.entry} ({row.InChIKey})")
            continue
        pending.append(
            {
                "entry": row.entry,
                "name": row.name,
                "InChIKey": row.InChIKey,
                "SMILES": row.SMILES,
                "molecule_dir": molecule_dir,
            }
        )
    if missing:
        raise FileNotFoundError(
            "Missing Gaussian cube files for "
            + ", ".join(missing)
            + ". Re-run with --run-quantum."
        )
    if not pending:
        return

    def calculate(record: dict[str, object]) -> str:
        """Run the validated Gaussian workflow for one distinct structure."""
        print(f"quantum calculation: {record['entry']} {record['name']}", flush=True)
        calc_mol.calc_ket(
            str(record["molecule_dir"]),
            str(record["SMILES"]),
            calc_mol.GAUSSIAN_RUN_COMMAND,
        )
        return str(record["entry"])

    with ThreadPoolExecutor(max_workers=min(max(workers, 1), len(pending))) as executor:
        futures = {
            executor.submit(calculate, record): str(record["entry"])
            for record in pending
        }
        failures: list[tuple[str, Exception]] = []
        for count, future in enumerate(as_completed(futures), start=1):
            entry = futures[future]
            try:
                future.result()
            except Exception as exc:  # report every failed pathway before aborting
                failures.append((entry, exc))
                print(
                    f"quantum failed {count}/{len(pending)}: {entry}: {exc}",
                    flush=True,
                )
            else:
                print(f"quantum completed {count}/{len(pending)}: {entry}", flush=True)
        if failures:
            details = "; ".join(f"{entry}: {exc}" for entry, exc in failures)
            raise RuntimeError(f"Quantum calculations failed for {details}")


def build_external_orbital_block(
    rows: pd.DataFrame,
    molecule_root: Path,
    target_coords: np.ndarray,
    output: Path,
    workers: int,
) -> np.ndarray:
    """Build the current HOMO-gap projected orbital descriptor for new rows."""
    nbo = conformer_helpers
    pipeline = homo_projected_orbital
    full_grid = orbital_grid
    nbo.MOLECULE_ROOT = molecule_root
    temporary_root = molecule_root / "projected_orbital_tmp"
    cache_root = molecule_root / "projected_orbital_conformer_cache"
    temporary_root.mkdir(parents=True, exist_ok=True)
    cache_root.mkdir(parents=True, exist_ok=True)
    pipeline.TMP_ROOT = temporary_root
    pipeline.GRID_STEP_BOHR = 2.0
    pipeline.folded_bins_from_cube_stream = full_grid.cube_reader(target_coords)

    unique = rows.drop_duplicates("InChIKey").reset_index(drop=True)
    jobs: list[dict[str, object]] = []
    for row_index, row in unique.iterrows():
        conformers = nbo.conformer_records(row)
        for conformer in conformers:
            if conformer["skip_reason"]:
                continue
            jobs.append(
                {
                    "row_index": row_index,
                    "entry": str(row["entry"]),
                    "name": str(row["name"]),
                    "InChIKey": str(row["InChIKey"]),
                    "conf_id": int(conformer["conf_id"]),
                    "boltzmann_weight": float(conformer["boltzmann_weight"]),
                    "target_c_index": int(conformer["target_c_index"]),
                    "target_o_index": int(conformer["target_o_index"]),
                    "sp_chk": str(conformer["sp_chk"]),
                }
            )
    if not jobs:
        raise RuntimeError("No completed conformers were available for projected-orbital generation.")

    def cache_path(job: dict[str, object]) -> Path:
        """Return the per-conformer projected-orbital cache path."""
        return cache_root / f"{job['InChIKey']}_conf{job['conf_id']}.npy"

    def build_one(job: dict[str, object]) -> tuple[dict[str, object], np.ndarray]:
        """Load or generate one conformer's projected-orbital grid vector."""
        cache = cache_path(job)
        if cache.exists():
            values = np.load(cache)
            if values.shape != (len(target_coords),):
                raise ValueError(f"Unexpected projected-orbital cache shape: {cache}")
            return job, values
        workdir = Path(
            tempfile.mkdtemp(
                prefix=f"{int(job['row_index']):03d}_{int(job['conf_id'])}_",
                dir=temporary_root,
            )
        )
        try:
            result = pipeline.build_homo_damped_cube(pd.Series(job), workdir, stream_cube=True)
            values = np.asarray(result["streamed_bins"], dtype=float)
            temporary = cache.with_suffix(".tmp.npy")
            np.save(temporary, values)
            temporary.replace(cache)
            return job, values
        finally:
            shutil.rmtree(workdir, ignore_errors=True)

    orbital_unique = np.zeros((len(unique), len(target_coords)), dtype=float)
    weight_sums = np.zeros(len(unique), dtype=float)
    manifest_rows: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=min(max(workers, 1), len(jobs))) as executor:
        futures = [executor.submit(build_one, job) for job in jobs]
        for count, future in enumerate(as_completed(futures), start=1):
            job, values = future.result()
            row_index = int(job["row_index"])
            weight = float(job["boltzmann_weight"])
            orbital_unique[row_index] += weight * values
            weight_sums[row_index] += weight
            manifest_rows.append({**job, "cache": str(cache_path(job))})
            if count % 5 == 0 or count == len(jobs):
                print(f"projected orbital conformers {count}/{len(jobs)}", flush=True)
    if np.any(np.abs(weight_sums - 1.0) > 1.0e-6):
        raise RuntimeError(f"Incomplete projected-orbital conformer weights: {weight_sums}")

    key_to_row = dict(zip(unique["InChIKey"].astype(str), range(len(unique))))
    orbital = orbital_unique[[key_to_row[key] for key in rows["InChIKey"].astype(str)]]
    np.savez_compressed(
        output,
        orbital=orbital,
        coords=target_coords,
        inchikeys=rows["InChIKey"].to_numpy(dtype=str),
        weight_sums=weight_sums,
    )
    pd.DataFrame(manifest_rows).sort_values(["entry", "conf_id"]).to_csv(
        output.with_suffix(".manifest.csv"), index=False
    )
    return orbital


def transformed_external_blocks(
    rows: pd.DataFrame,
    molecule_root: Path,
    coords: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Apply the frozen eCoMFA transform and align it to bundle coordinates."""
    ecomfa_transform.MOLECULE_ROOT = molecule_root
    label = "ecomfa_kernel_1e-2_v1"
    by_key: dict[str, pd.Series] = {}
    for row in rows.drop_duplicates("InChIKey").itertuples(index=False):
        print(f"grid transform: {row.entry} {row.name}", flush=True)
        values = ecomfa_transform.calc_transform_features_for_row(pd.Series(row._asdict()))[label]
        if values.empty:
            raise RuntimeError(f"No transformed grid values were generated for {row.entry}.")
        by_key[str(row.InChIKey)] = values

    blocks: dict[str, np.ndarray] = {}
    for block in EE_ES_BLOCKS:
        columns = [f"{block}_fold {x} {y} {z}" for x, y, z in coords[block]]
        blocks[block] = np.asarray(
            [[by_key[str(row.InChIKey)].get(column, 0.0) for column in columns] for row in rows.itertuples(index=False)],
            dtype=float,
        )
    return blocks


def load_external_orbital_block(
    path: Path,
    rows: pd.DataFrame,
    target_coords: np.ndarray,
) -> np.ndarray:
    """Load an explicitly generated projected-orbital cache by InChIKey."""
    cache = np.load(path, allow_pickle=False)
    required = {"orbital", "coords", "inchikeys"}
    missing = required - set(cache.files)
    if missing:
        raise ValueError(f"External orbital NPZ lacks arrays: {sorted(missing)}")
    coords = np.asarray(cache["coords"], dtype=int)
    if not np.array_equal(coords, target_coords):
        raise ValueError("External projected-orbital coordinates do not match the current model.")
    values = np.asarray(cache["orbital"], dtype=float)
    keys = [str(key) for key in cache["inchikeys"]]
    if values.shape != (len(keys), len(target_coords)):
        raise ValueError(f"Unexpected external projected-orbital shape: {values.shape}")
    key_to_row = {key: index for index, key in enumerate(keys)}
    missing_keys = sorted(set(rows["InChIKey"].astype(str)) - set(key_to_row))
    if missing_keys:
        raise ValueError(f"External orbital NPZ lacks molecules: {missing_keys}")
    return values[[key_to_row[key] for key in rows["InChIKey"].astype(str)]]


def simulate_network(prediction: pd.Series, temperature: float) -> tuple[dict[str, float], dict[str, float]]:
    """Return canonical peak monoalcohol and endpoint diol percentages."""
    barrier = prediction.reindex(ENTRY_ORDER).to_numpy(dtype=float)
    return diketone_metrics.simulate_barrier_network(barrier, temperature)


def predict_outer_model_selectivity(
    raw_extended: dict[str, np.ndarray],
    coords: dict[str, np.ndarray],
    train: np.ndarray,
    y: np.ndarray,
    external_index: np.ndarray,
    series: str,
    temperature: float,
    out_dir: Path,
    workers: int,
) -> pd.DataFrame:
    """Propagate the 83 nested outer models to an external diketone series."""
    outer = pd.read_csv(current_model.DATA_DIR / "outer_predictions.csv").sort_values(
        "fold_id"
    )

    def predict_fold(row: pd.Series) -> list[dict[str, object]]:
        """Refit one stored outer-fold model and predict both network stages."""
        holdout = int(row["holdout_index"])
        fold_train = train[train != holdout]
        x_fold, _, _ = current_model.build_features(
            raw_extended, coords, fold_train
        )
        model = Lasso(
            alpha=float(row["selected_alpha"]),
            fit_intercept=True,
            max_iter=current_model.LASSO_FIT_MAX_ITER,
            tol=current_model.LASSO_FIT_TOL,
        ).fit(x_fold[fold_train], y[fold_train])
        barriers = pd.Series(
            model.predict(x_fold[external_index]), index=ENTRY_ORDER
        )
        intermediate, final = simulate_network(barriers, temperature)
        common = {
            "fold_id": int(row["fold_id"]),
            "holdout_entry": str(row["entry"]),
            "selected_alpha": float(row["selected_alpha"]),
        }
        return [
            *(
                {
                    **common,
                    "stage": "peak monoalcohol",
                    "product": key,
                    "predicted_percent": value,
                }
                for key, value in intermediate.items()
            ),
            *(
                {
                    **common,
                    "stage": "endpoint diol",
                    "product": key,
                    "predicted_percent": value,
                }
                for key, value in final.items()
            ),
        ]

    rows: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=min(max(workers, 1), 20)) as executor:
        futures = [
            executor.submit(predict_fold, row)
            for _, row in outer.iterrows()
        ]
        for count, future in enumerate(as_completed(futures), start=1):
            rows.extend(future.result())
            if count % 20 == 0 or count == len(futures):
                print(f"outer-model selectivity {count}/{len(futures)}", flush=True)

    predictions = pd.DataFrame(rows).sort_values(
        ["fold_id", "stage", "product"]
    )
    predictions.to_csv(out_dir / "outer83_selectivity_predictions.csv", index=False)

    summary_rows = []
    for (stage, product), group in predictions.groupby(["stage", "product"], sort=False):
        values = group["predicted_percent"].to_numpy(dtype=float)
        summary_rows.append(
            {
                "stage": stage,
                "product": product,
                "mean_percent": float(np.mean(values)),
                "median_percent": float(np.median(values)),
                "p16_percent": float(np.percentile(values, 16)),
                "p84_percent": float(np.percentile(values, 84)),
                "min_percent": float(np.min(values)),
                "max_percent": float(np.max(values)),
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out_dir / "outer83_selectivity_summary.csv", index=False)

    initial = predictions.loc[predictions["stage"].eq("peak monoalcohol")]
    major = initial.loc[
        initial.groupby("fold_id")["predicted_percent"].idxmax(),
        ["fold_id", "holdout_entry", "product", "predicted_percent"],
    ].sort_values("fold_id")
    major.to_csv(out_dir / "outer83_major_monoalcohol.csv", index=False)
    return summary


def main() -> None:
    """Generate descriptors, fit the current model, and predict one 12-pathway series."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--series", default="x", help="Diketone series prefix, e.g. x")
    parser.add_argument("--excel", type=Path, default=DEFAULT_EXCEL)
    parser.add_argument("--run-quantum", action="store_true", help="Run missing Gaussian/cube calculations.")
    parser.add_argument("--quantum-workers", type=int, default=1)
    parser.add_argument("--orbital-workers", type=int, default=4)
    parser.add_argument("--outer-workers", type=int, default=20)
    parser.add_argument(
        "--molecule-root",
        type=Path,
        help=(
            "External location for large Gaussian files. If omitted, MOLECULES_ROOT "
            "must be set; Gaussian files are never created inside the repository."
        ),
    )
    parser.add_argument(
        "--promote-inputs",
        action="store_true",
        help=(
            "Write reviewed x/y descriptor arrays to the frozen-input package. "
            "Without this flag, portable caches remain under ignored validation output."
        ),
    )
    parser.add_argument(
        "--orbital-npz",
        type=Path,
        help=(
            "Projected C=O pi* cache containing orbital, coords, and inchikeys arrays. "
            "If omitted, it is built from the series Gaussian files."
        ),
    )
    args = parser.parse_args()

    series = normalize_entry(args.series)
    if len(series) != 1 or not series.isalpha():
        raise ValueError("--series must be a single letter, for example x.")
    excel_path = args.excel.resolve()
    out_dir = ROOT / "data" / "validation" / "external_diketones" / f"{series}_series"
    if args.promote_inputs:
        if series not in current_model.EXTERNAL_DIKETONE_SERIES:
            raise ValueError(
                "Only the reviewed x/y series may be promoted. Update the model specification "
                "and Git policy explicitly before adopting another series."
            )
        input_dir = current_model.EXTERNAL_DIKETONE_INPUT_DIR / f"{series}_series"
    else:
        input_dir = out_dir / "portable_cache"

    configured_root = args.molecule_root or (
        Path(os.environ["MOLECULES_ROOT"]) if os.environ.get("MOLECULES_ROOT") else None
    )
    if configured_root is None:
        raise ValueError(
            "Set --molecule-root or MOLECULES_ROOT to a repository-external Gaussian directory."
        )
    molecule_root = configured_root.expanduser().resolve()
    if molecule_root == ROOT or ROOT in molecule_root.parents:
        raise ValueError(
            f"Gaussian molecule root must be outside the repository: {molecule_root}"
        )
    scratch_dir = Path(
        os.environ.get("GAUSS_SCRDIR", str(molecule_root / "gaussian_scratch"))
    ).expanduser().resolve()
    if scratch_dir == ROOT or ROOT in scratch_dir.parents:
        raise ValueError(f"Gaussian scratch directory must be outside the repository: {scratch_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    input_dir.mkdir(parents=True, exist_ok=True)
    molecule_root.mkdir(parents=True, exist_ok=True)
    scratch_dir.mkdir(parents=True, exist_ok=True)
    os.environ["GAUSS_SCRDIR"] = str(scratch_dir)
    os.environ.setdefault("CALC_MOL_SKIP_FRONTIER_CUBES", "1")

    rows = select_series_rows(excel_path, series)
    rows.to_csv(out_dir / "input_rows.csv", index=False)
    rows.to_csv(input_dir / "input_rows.csv", index=False)
    ensure_quantum_files(rows, molecule_root, args.run_quantum, args.quantum_workers)

    payload = current_model.ensure_frozen_inputs()
    train_manifest = pd.read_csv(current_model.TRAIN_ROWS_PATH)
    payload, train, _ = current_model.refresh_inputs_from_excel(
        payload, train_manifest
    )
    meta = payload["meta"].copy()
    raw = {block: np.asarray(payload["raw_blocks"][block], dtype=float) for block in current_model.BLOCKS}
    coords = {block: np.asarray(payload["coords"][block], dtype=int) for block in current_model.BLOCKS}
    external = transformed_external_blocks(rows, molecule_root, coords)
    if "orbital" in current_model.BLOCKS:
        orbital_path = (
            args.orbital_npz.resolve()
            if args.orbital_npz is not None
            else out_dir / "projected_orbital_fullgrid_2bohr.npz"
        )
        if orbital_path.exists():
            external["orbital"] = load_external_orbital_block(
                orbital_path, rows, coords["orbital"]
            )
        else:
            external["orbital"] = build_external_orbital_block(
                rows,
                molecule_root,
                coords["orbital"],
                orbital_path,
                args.orbital_workers,
            )
    np.savez_compressed(
        input_dir / "external_raw_blocks_2bohr.npz",
        entries=rows["entry"].to_numpy(dtype=str),
        inchikeys=rows["InChIKey"].to_numpy(dtype=str),
        **{block: external[block] for block in current_model.BLOCKS},
        **{f"coords_{block}": coords[block] for block in current_model.BLOCKS},
    )
    raw_extended = {block: np.vstack((raw[block], external[block])) for block in current_model.BLOCKS}
    x, feature_names, _ = current_model.build_features(raw_extended, coords, train)
    y = meta[current_model.TARGET].astype(float).to_numpy()
    selected_alpha, _, _ = current_model.inner_best(raw, coords, y, train)
    model = Lasso(
        alpha=selected_alpha,
        fit_intercept=True,
        max_iter=current_model.LASSO_FIT_MAX_ITER,
        tol=current_model.LASSO_FIT_TOL,
    )
    model.fit(x[train], y[train])
    external_index = np.arange(len(meta), len(meta) + len(rows), dtype=int)
    prediction = model.predict(x[external_index])

    result = rows[["entry", "name", "InChIKey", "SMILES", "temperature"]].copy()
    result["prediction"] = prediction
    for block in current_model.BLOCKS:
        mask = np.asarray([name.startswith(f"{block}:") for name in feature_names])
        result[f"{block}_contribution"] = x[external_index][:, mask] @ model.coef_[mask]
    result["intercept"] = float(model.intercept_)
    result.to_csv(out_dir / "predicted_barriers_and_contributions.csv", index=False)

    by_suffix = result.assign(suffix=result["entry"].str[len(series) :]).set_index("suffix")["prediction"]
    temperature = float(rows["temperature"].iloc[0])
    intermediate, final = simulate_network(by_suffix, temperature)
    selectivity = pd.DataFrame(
        [
            *({"stage": "peak monoalcohol", "product": key, "predicted_percent": value} for key, value in intermediate.items()),
            *({"stage": "endpoint diol", "product": key, "predicted_percent": value} for key, value in final.items()),
        ]
    )
    if "isolated yield after reduction to monoketone" in rows.columns:
        observed = pd.to_numeric(
            rows.set_index("suffix")["isolated yield after reduction to monoketone"],
            errors="coerce",
        )
        selectivity["observed_percent"] = np.nan
        initial_mask = selectivity["stage"].eq("peak monoalcohol")
        selectivity.loc[initial_mask, "observed_percent"] = [
            float(observed.get(product, np.nan) * 100.0)
            for product in selectivity.loc[initial_mask, "product"]
        ]
        selectivity["absolute_error_percent"] = (
            selectivity["predicted_percent"] - selectivity["observed_percent"]
        ).abs()
    selectivity.to_csv(out_dir / "predicted_selectivity.csv", index=False)

    outer_summary = predict_outer_model_selectivity(
        raw_extended,
        coords,
        train,
        y,
        external_index,
        series,
        temperature,
        out_dir,
        args.outer_workers,
    )

    from libs.graph import reaction_concentration_plot_complex  # noqa: PLC0415

    reaction_concentration_plot_complex(
        by_suffix.reindex(ENTRY_ORDER).to_numpy(dtype=float),
        T=temperature,
        a0=100,
        save_path=out_dir / "diketone_progress.png",
    )
    (out_dir / "model_and_input_specification.json").write_text(
        json.dumps(
            {
                "excel": str(excel_path),
                "series": series,
                "series_entries": result["entry"].tolist(),
                "descriptor_transform": "ecomfa_kernel_1e-2_v1",
                "model": "current projected C=O pi* Lasso",
                "alpha": selected_alpha,
                "training_n": int(len(train)),
                "molecule_root": str(molecule_root),
                "orbital_npz": str(orbital_path),
                "grid_bounds": current_model.EE_BOUNDS,
                "grid_spacing_bohr": 2,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    contribution_columns = [f"{block}_contribution" for block in current_model.BLOCKS]
    print(result[["entry", "prediction", *contribution_columns]].to_string(index=False))
    print(selectivity.to_string(index=False))
    print(outer_summary.to_string(index=False))


if __name__ == "__main__":
    main()
