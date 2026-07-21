"""Build HOMO-gap-damped carbonyl virtual-orbital grid descriptors.

For every conformer, this pipeline constructs a local carbonyl pi-star seed
from C/O p orbitals, projects it into the canonical virtual-MO space, divides
each virtual contribution by its energy separation from the HOMO, and
normalizes the resulting AO vector in the overlap metric.  ``cubegen`` samples
that orbital; squared amplitudes are summed into 2-bohr bins after folding the
molecular-frame ``y`` axis to ``abs(y)``.  Conformer vectors are multiplied by
the ``boltzmann_weight`` column and accumulated by dataset ``row_index``.

Input rows come from an explicitly supplied external full conformer manifest
and require at least
``row_index``, ``entry``, ``name``, ``InChIKey``, ``conf_id``, ``sp_chk``,
``status``, ``target_c_index``, ``target_o_index``, and
``boltzmann_weight``.  The repository's sanitized seven-column manifest is
provenance-only and is not a calculation input. Atom indices are Gaussian-style
one-based indices. FCHK coordinates and cube grid vectors are in bohr; orbital
and Fock energies are in hartree, with reported gaps also converted to eV.
Output feature columns use the schema
``orbital_fold <x> <y> <z>``, where the coordinates are integer 2-bohr bin
indices and values are Boltzmann-weighted sums of sampled orbital amplitude
squared (without a voxel-volume normalization).
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import tempfile
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "data" / "current_model" / "work" / "homo_projected_orbital"
REQUIRED_DATASET_COLUMNS = frozenset({"entry", "name", "InChIKey"})
REQUIRED_MANIFEST_COLUMNS = frozenset(
    {
        "row_index",
        "entry",
        "name",
        "InChIKey",
        "conf_id",
        "status",
        "boltzmann_weight",
        "sp_chk",
        "target_c_index",
        "target_o_index",
    }
)
TMP_ROOT = Path(
    os.environ.get(
        "HOMO_PROJECTED_TMP_ROOT",
        str(Path(tempfile.gettempdir()) / "homo_damped_projected_pi_star"),
    )
)
GRID_STEP_BOHR = 2.0
DEFAULT_ORBITAL_BUILD_BOUNDS = (-5, 2, 1, 4, -2, 3)
CUBEGEN = Path(
    os.environ.get("CUBEGEN_EXECUTABLE", shutil.which("cubegen") or "cubegen")
)
CUBEGEN_NPROC = os.environ.get("HOMO_PROJECTED_CUBEGEN_NPROC", "2")
GRID_SPEC = "-3 h"
HARTREE_TO_EV = 27.211386245988

from .projected_pi_star import (
    carbonyl_normal,
    coordinates_and_numbers,
    flatten_mo_coefficients,
    local_p_vector,
    normalized,
    p_shell_indices,
    parse_multiwfn_matrix,
    read_array,
    read_fchk_lines,
    read_scalar,
    replace_float_array,
    run_multiwfn_matrix,
)


def safe_token(value: object, max_len: int = 80) -> str:
    """Convert a manifest value to a bounded filesystem-safe label.

    Characters outside ASCII letters, digits, ``_.()+-`` become underscores;
    the stripped result is truncated to ``max_len`` characters.
    """
    text = re.sub(r"[^A-Za-z0-9_.()+-]+", "_", str(value)).strip("_")
    return text[:max_len] if len(text) > max_len else text


def grid_keys_from_bounds(bounds: tuple[int, int, int, int, int, int]) -> list[tuple[int, int, int]]:
    """Enumerate inclusive grid-bin bounds in feature-column order.

    ``bounds`` has schema ``(xmin, xmax, ymin, ymax, zmin, zmax)`` in integer
    2-bohr bin units.  Returned ``(x, y, z)`` keys are ordered with ``z``
    varying fastest, then ``y``, then ``x``.
    """
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    return [
        (x, y, z)
        for x in range(xmin, xmax + 1)
        for y in range(ymin, ymax + 1)
        for z in range(zmin, zmax + 1)
    ]


def read_mo_cube(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read coordinates and scalar amplitudes from a Gaussian MO cube.

    Returns an ``(n_voxels, 3)`` coordinate array and an ``(n_voxels,)``
    orbital-amplitude array in cube traversal order.  Coordinates retain the
    cube header's units (Gaussian ``cubegen`` output used here is in bohr); the
    orbital values are not squared or volume-normalized.
    """
    with path.open(encoding="UTF-8", errors="ignore") as handle:
        handle.readline()
        handle.readline()
        atom_line = handle.readline().split()
        n_atom_signed = int(atom_line[0])
        n_atom = abs(n_atom_signed)
        origin = np.array(atom_line[1:4], dtype=float)
        sizes: list[int] = []
        axes: list[list[float]] = []
        for _ in range(3):
            line = handle.readline().split()
            sizes.append(int(line[0]))
            axes.append([float(value) for value in line[1:4]])
        for _ in range(n_atom):
            handle.readline()
        if n_atom_signed < 0:
            handle.readline()
        values = np.fromstring(handle.read(), dtype=float, sep=" ")
    sizes_array = np.asarray(sizes, dtype=int)
    axes_array = np.asarray(axes, dtype=float)
    coords = np.indices(sizes_array, dtype=float).reshape(3, -1).T @ np.asarray(axes, dtype=float) + origin
    if values.size != coords.shape[0]:
        raise ValueError(f"Cube size mismatch for {path}: {values.size} != {coords.shape[0]}")
    return coords, values


def read_mo_coefficients_relaxed(lines: list[str], overlap: np.ndarray) -> tuple[np.ndarray, str, float]:
    """Decode alpha-MO coefficients while tolerating text-matrix precision.

    ``lines`` are complete FCHK lines and ``overlap`` is the ``(nbasis,
    nbasis)`` AO overlap matrix.  Both possible FCHK flattening orders are
    tested against ``C.T @ S @ C``.  The return value is the basis-by-MO
    coefficient matrix, the selected order name, and the maximum absolute
    orthonormality error.  Coefficients and overlap elements are dimensionless.
    """
    nbasis = int(read_scalar(lines, "Number of basis functions", int))
    nmo = int(read_scalar(lines, "Number of independent functions", int))
    flat, _, _ = read_array(lines, "Alpha MO coefficients", float)
    candidates = {
        "mo_major": flat.reshape((nmo, nbasis)).T,
        "basis_major": flat.reshape((nbasis, nmo)),
    }
    best_name = ""
    best_matrix = None
    best_error = float("inf")
    identity = np.eye(nmo)
    for name, matrix in candidates.items():
        gram = matrix.T @ overlap @ matrix
        error = float(np.max(np.abs(gram - identity)))
        if error < best_error:
            best_error = error
            best_name = name
            best_matrix = matrix
    # Multiwfn integral matrices are text files with limited precision. Fluorenone
    # can land near 0.03 in this check, while the coefficient ordering is still clear.
    if best_matrix is None or best_error > 5.0e-2:
        raise ValueError(f"Could not identify MO coefficient order; best {best_name} err={best_error:g}")
    return best_matrix, best_name, best_error


def folded_bins_from_cube(
    path: Path,
    weight: float,
    bounds: tuple[int, int, int, int, int, int] = DEFAULT_ORBITAL_BUILD_BOUNDS,
) -> np.ndarray:
    """Aggregate one materialized MO cube into weighted folded grid features.

    ``bounds`` is inclusive ``(xmin, xmax, ymin, ymax, zmin, zmax)`` in 2-bohr
    bin indices.  Coordinates are rounded away from zero and ``y`` is replaced
    by ``abs(y)``.  The returned flat vector follows
    :func:`grid_keys_from_bounds` and contains ``weight * sum(amplitude**2)``;
    no voxel-volume factor is applied.
    """
    coords, mo_values = read_mo_cube(path)
    values = mo_values * mo_values
    scaled = coords / GRID_STEP_BOHR
    binned = np.where(scaled > 0, np.ceil(scaled), np.floor(scaled)).astype(np.int16)
    binned[:, 1] = np.abs(binned[:, 1])
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    mask = (
        (binned[:, 0] >= xmin)
        & (binned[:, 0] <= xmax)
        & (binned[:, 1] >= ymin)
        & (binned[:, 1] <= ymax)
        & (binned[:, 2] >= zmin)
        & (binned[:, 2] <= zmax)
    )
    nx = xmax - xmin + 1
    ny = ymax - ymin + 1
    nz = zmax - zmin + 1
    linear = (
        (binned[mask, 0].astype(int) - xmin) * ny * nz
        + (binned[mask, 1].astype(int) - ymin) * nz
        + (binned[mask, 2].astype(int) - zmin)
    )
    return np.bincount(linear, weights=values[mask], minlength=nx * ny * nz) * float(weight)


def folded_bins_from_cube_stream(
    path: Path,
    bounds: tuple[int, int, int, int, int, int] = DEFAULT_ORBITAL_BUILD_BOUNDS,
) -> np.ndarray:
    """Stream an MO cube and return unweighted folded squared-amplitude bins.

    ``path`` may be a regular file or FIFO.  ``bounds`` uses inclusive integer
    2-bohr indices in ``xmin, xmax, ymin, ymax, zmin, zmax`` order.  The output
    is ordered as :func:`grid_keys_from_bounds` and omits voxel-volume scaling.
    """
    with path.open(encoding="UTF-8", errors="ignore") as handle:
        handle.readline()
        handle.readline()
        atom_line = handle.readline().split()
        n_atom_signed = int(atom_line[0])
        n_atom = abs(n_atom_signed)
        origin = np.array(atom_line[1:4], dtype=float)
        sizes: list[int] = []
        axes: list[list[float]] = []
        for _ in range(3):
            line = handle.readline().split()
            sizes.append(int(line[0]))
            axes.append([float(value) for value in line[1:4]])
        for _ in range(n_atom):
            handle.readline()
        if n_atom_signed < 0:
            handle.readline()

        sizes_array = np.asarray(sizes, dtype=int)
        coords = np.indices(sizes_array, dtype=float).reshape(3, -1).T @ np.asarray(axes, dtype=float) + origin
        scaled = coords / GRID_STEP_BOHR
        binned = np.where(scaled > 0, np.ceil(scaled), np.floor(scaled)).astype(np.int16)
        binned[:, 1] = np.abs(binned[:, 1])
        xmin, xmax, ymin, ymax, zmin, zmax = bounds
        mask = (
            (binned[:, 0] >= xmin)
            & (binned[:, 0] <= xmax)
            & (binned[:, 1] >= ymin)
            & (binned[:, 1] <= ymax)
            & (binned[:, 2] >= zmin)
            & (binned[:, 2] <= zmax)
        )
        nx = xmax - xmin + 1
        ny = ymax - ymin + 1
        nz = zmax - zmin + 1
        linear_full = np.full(coords.shape[0], -1, dtype=np.int32)
        linear_full[mask] = (
            (binned[mask, 0].astype(int) - xmin) * ny * nz
            + (binned[mask, 1].astype(int) - ymin) * nz
            + (binned[mask, 2].astype(int) - zmin)
        )
        accum = np.zeros(nx * ny * nz, dtype=float)
        offset = 0
        for line in handle:
            values = np.fromstring(line, dtype=float, sep=" ")
            if values.size == 0:
                continue
            local = linear_full[offset : offset + values.size]
            keep = local >= 0
            if np.any(keep):
                np.add.at(accum, local[keep], values[keep] * values[keep])
            offset += values.size
        if offset != linear_full.size:
            raise ValueError(f"Cube stream size mismatch for {path}: {offset} != {linear_full.size}")
    return accum


def write_text_fifo(path: Path, text: str, errors: list[BaseException]) -> None:
    """Write FCHK text to a FIFO, appending any exception to ``errors``.

    This thread target communicates failures through the caller-owned mutable
    list rather than raising across the thread boundary.
    """
    try:
        with path.open("w", encoding="utf-8") as handle:
            handle.write(text)
    except BaseException as exc:  # noqa: BLE001
        errors.append(exc)


def read_cube_bins_thread(path: Path, result: dict[str, np.ndarray], errors: list[BaseException]) -> None:
    """Consume a cube FIFO into ``result['bins']`` for a worker thread.

    The bin vector is the unweighted, folded sum of orbital amplitude squared;
    exceptions are appended to the shared ``errors`` list.
    """
    try:
        result["bins"] = folded_bins_from_cube_stream(path)
    except BaseException as exc:  # noqa: BLE001
        errors.append(exc)


def build_homo_damped_cube(row: pd.Series, workdir: Path, stream_cube: bool = True) -> dict[str, object]:
    """Construct and sample one conformer's gap-damped projected orbital.

    ``row`` must provide ``sp_chk``, one-based ``target_c_index`` and
    ``target_o_index``, plus ``row_index``, ``entry``, and ``conf_id`` for file
    naming.  The corresponding ``.fchk`` supplies AO/MO data; Multiwfn supplies
    AO overlap and Fock matrices.  Virtual coefficients are weighted by
    ``1 / (epsilon_virtual - epsilon_HOMO)`` before overlap normalization.

    When ``stream_cube`` is true, the cube is passed through a FIFO and the
    return dictionary includes ``streamed_bins``; otherwise ``cube`` names the
    retained cube file.  Diagnostic keys ending in ``_energy_au`` are hartree,
    ``homo_lumo_gap_ev`` is eV, atom indices are one-based, and overlaps and MO
    coefficients are dimensionless.
    """
    fchk = Path(str(row["sp_chk"])).with_suffix(".fchk")
    if not fchk.exists():
        raise FileNotFoundError(f"missing fchk: {fchk}")

    lines = read_fchk_lines(fchk)
    nbasis = int(read_scalar(lines, "Number of basis functions", int))
    nocc = int(read_scalar(lines, "Number of alpha electrons", int))

    overlap = parse_multiwfn_matrix(run_multiwfn_matrix(fchk, workdir, 1, "overlap_intmat.txt"), nbasis)
    fock = parse_multiwfn_matrix(run_multiwfn_matrix(fchk, workdir, 0, "fock_intmat.txt"), nbasis)

    target_c = int(float(row["target_c_index"]))
    target_o = int(float(row["target_o_index"]))
    numbers, coords = coordinates_and_numbers(lines)
    normal, alpha_neighbors = carbonyl_normal(numbers, coords, target_c, target_o)
    p_shells = p_shell_indices(lines)
    p_c = normalized(local_p_vector(nbasis, p_shells, target_c, normal), overlap)
    p_o = normalized(local_p_vector(nbasis, p_shells, target_o, normal), overlap)

    local = np.column_stack([p_c, p_o])
    gram = local.T @ overlap @ local
    eigvals, eigvecs = np.linalg.eigh(gram)
    local_orth = local @ eigvecs @ np.diag(1.0 / np.sqrt(np.maximum(eigvals, 1.0e-14)))
    fock_local = local_orth.T @ fock @ local_orth
    local_energies, local_mix = np.linalg.eigh(fock_local)
    seed = normalized(local_orth @ local_mix[:, -1], overlap)

    mo_coeff, coeff_order, mo_orth_error = read_mo_coefficients_relaxed(lines, overlap)
    energies, _, _ = read_array(lines, "Alpha Orbital Energies", float)
    homo_energy = float(energies[nocc - 1])
    gaps = energies[nocc:] - homo_energy
    if np.any(gaps <= 0.0):
        raise ValueError("non-positive virtual-HOMO gap")

    virt = mo_coeff[:, nocc:]
    coeff = virt.T @ overlap @ seed
    damped = normalized(virt @ (coeff / gaps), overlap)
    if float(damped @ overlap @ seed) < 0.0:
        damped = -damped
    occ = mo_coeff[:, :nocc]
    occ_leak = float(np.max(np.abs(occ.T @ overlap @ damped)))
    damped_energy = float(damped @ fock @ damped)
    normalized_coeff = virt.T @ overlap @ damped
    top_rel = np.argsort(np.abs(normalized_coeff))[::-1][:8]

    modified = list(lines)
    new_coeff = mo_coeff.copy()
    new_coeff[:, nocc] = damped
    replace_float_array(modified, "Alpha MO coefficients", flatten_mo_coefficients(new_coeff, coeff_order))
    new_energies = energies.copy()
    new_energies[nocc] = damped_energy
    replace_float_array(modified, "Alpha Orbital Energies", new_energies)

    label = f"row{int(row['row_index']):03d}_{safe_token(row['entry'], 24)}_conf{int(row['conf_id'])}"
    out_fchk = workdir / f"{label}_homo_damped_projected_pi_star.fchk"
    out_cube = workdir / f"{label}_homo_damped_projected_pi_star.cube"
    modified_text = "\n".join(modified) + "\n"

    env = os.environ.copy()
    env["GAUSS_SCRDIR"] = str(workdir)
    cubegen_log = workdir / "cubegen_homo_damped_projected_pi_star.log"
    streamed_bins: np.ndarray | None = None
    if stream_cube:
        if out_cube.exists():
            out_cube.unlink()
        if out_fchk.exists():
            out_fchk.unlink()
        out_fchk.write_text(modified_text, encoding="utf-8")
        os.mkfifo(out_cube)
        reader_result: dict[str, np.ndarray] = {}
        reader_errors: list[BaseException] = []
        reader = threading.Thread(
            target=read_cube_bins_thread,
            args=(out_cube, reader_result, reader_errors),
            daemon=True,
        )
        with cubegen_log.open("w", encoding="utf-8") as log_handle:
            reader.start()
            process = subprocess.Popen(
                [str(CUBEGEN), CUBEGEN_NPROC, f"MO={nocc + 1}", str(out_fchk), str(out_cube), *GRID_SPEC.split()],
                cwd=workdir,
                env=env,
                text=True,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
            )
            try:
                returncode = process.wait()
            finally:
                if process.poll() is None:
                    process.kill()
                if reader.is_alive():
                    try:
                        fd = os.open(out_cube, os.O_WRONLY | os.O_NONBLOCK)
                        os.close(fd)
                    except OSError:
                        pass
                reader.join(timeout=30)
                try:
                    out_cube.unlink()
                except OSError:
                    pass
                try:
                    out_fchk.unlink()
                except OSError:
                    pass
        if returncode != 0:
            tail = cubegen_log.read_text(encoding="utf-8", errors="ignore")[-3000:] if cubegen_log.exists() else ""
            raise RuntimeError(f"cubegen failed returncode={returncode}\n{tail}")
        if reader_errors:
            raise RuntimeError(f"cube FIFO reader failed: {reader_errors[0]}")
        if "bins" not in reader_result:
            raise RuntimeError("cube FIFO reader did not return bins")
        streamed_bins = reader_result["bins"]
    else:
        out_fchk.write_text(modified_text, encoding="utf-8")
        with cubegen_log.open("w", encoding="utf-8") as log_handle:
            completed = subprocess.run(
                [str(CUBEGEN), CUBEGEN_NPROC, f"MO={nocc + 1}", str(out_fchk), str(out_cube), *GRID_SPEC.split()],
                cwd=workdir,
                env=env,
                text=True,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if completed.returncode != 0 or not out_cube.exists() or out_cube.stat().st_size < 10000:
            tail = cubegen_log.read_text(encoding="utf-8", errors="ignore")[-3000:] if cubegen_log.exists() else ""
            raise RuntimeError(f"cubegen failed returncode={completed.returncode}\n{tail}")

    result = {
        "cube": out_cube,
        "target_c": target_c,
        "target_o": target_o,
        "alpha_neighbors": ";".join(str(i) for i in alpha_neighbors),
        "homo_energy_au": homo_energy,
        "lumo_energy_au": float(energies[nocc]),
        "homo_lumo_gap_ev": float((energies[nocc] - homo_energy) * HARTREE_TO_EV),
        "damped_projected_energy_au": damped_energy,
        "seed_overlap": float(seed @ overlap @ damped),
        "occ_leak_max": occ_leak,
        "mo_orth_error_max": float(mo_orth_error),
        "top_virtual_mos": ";".join(str(nocc + 1 + int(i)) for i in top_rel),
        "top_gap_damped_coefficients": ";".join(f"{float(normalized_coeff[i]):.6g}" for i in top_rel),
    }
    if streamed_bins is not None:
        result["streamed_bins"] = streamed_bins
    return result


def run_orbital_feature_job(row_dict: dict[str, object]) -> tuple[int, int, np.ndarray | None, dict[str, object]]:
    """Run and clean up one manifest-row feature job.

    ``row_dict`` follows the manifest schema documented at module level.  The
    returned tuple is ``(row_index, conf_id, weighted_bins, status_record)``;
    ``weighted_bins`` is ``None`` on failure and otherwise already multiplied
    by the row's dimensionless Boltzmann weight.
    """
    key = (int(row_dict["row_index"]), int(row_dict["conf_id"]))
    workdir = Path(tempfile.mkdtemp(prefix=f"{key[0]:03d}_{key[1]}_", dir=TMP_ROOT))
    record = {
        "row_index": key[0],
        "entry": row_dict["entry"],
        "name": row_dict["name"],
        "InChIKey": row_dict["InChIKey"],
        "conf_id": key[1],
        "boltzmann_weight": float(row_dict["boltzmann_weight"]),
    }
    bins: np.ndarray | None = None
    try:
        result = build_homo_damped_cube(pd.Series(row_dict), workdir, stream_cube=True)
        if "streamed_bins" in result:
            bins = np.asarray(result.pop("streamed_bins"), dtype=float) * float(row_dict["boltzmann_weight"])
        else:
            bins = folded_bins_from_cube(Path(result["cube"]), float(row_dict["boltzmann_weight"]))
        record.update({k: v for k, v in result.items() if k != "cube"})
        record.update(status="done", skip_reason="")
    except Exception as exc:  # noqa: BLE001
        record.update(status="failed", skip_reason=str(exc)[-2000:])
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
    return key[0], key[1], bins, record


def required_row_indices(data: pd.DataFrame, needed_only: bool) -> np.ndarray:
    """Return dataset row positions that must receive orbital features.

    The current reproducibility pipeline requires every positional row of
    ``data``.  ``needed_only`` is retained for CLI compatibility but does not
    change that completeness requirement.
    """
    del needed_only
    return np.arange(len(data), dtype=int)


def read_source_dataset(dataset_pickle: Path) -> pd.DataFrame:
    """Load and validate the external descriptor dataset pickle.

    The dataset must contain identity metadata plus at least one folded
    electronic and electrostatic feature. Rows are interpreted positionally by
    the manifest's zero-based ``row_index`` values.
    """
    data = pd.read_pickle(dataset_pickle)
    if not isinstance(data, pd.DataFrame):
        raise TypeError(f"Dataset pickle must contain a DataFrame: {dataset_pickle}")
    if data.empty:
        raise ValueError(f"Dataset is empty: {dataset_pickle}")
    missing = sorted(REQUIRED_DATASET_COLUMNS - set(data.columns))
    if missing:
        raise ValueError(
            f"Dataset {dataset_pickle} is missing required columns: "
            f"{', '.join(missing)}"
        )
    if data.columns.duplicated().any():
        duplicates = sorted(set(data.columns[data.columns.duplicated()].astype(str)))
        raise ValueError(f"Dataset {dataset_pickle} has duplicate columns: {duplicates}")
    for prefix in ("electronic_fold ", "electrostatic_fold "):
        if not any(str(column).startswith(prefix) for column in data.columns):
            raise ValueError(
                f"Dataset {dataset_pickle} has no {prefix.strip()} feature columns"
            )
    return data


def read_full_manifest(manifest_path: Path, dataset_size: int) -> pd.DataFrame:
    """Load and validate an external full projected-orbital manifest.

    Required column names are checked before filtering. Completed rows must
    have numeric conformer, weight, target-atom, and positional dataset-index
    values; their ``row_index`` values must fall within ``dataset_size``.
    """
    manifest = pd.read_csv(manifest_path)
    missing = sorted(REQUIRED_MANIFEST_COLUMNS - set(manifest.columns))
    if missing:
        raise ValueError(
            f"Full projected-orbital manifest {manifest_path} is missing "
            f"required columns: {', '.join(missing)}"
        )
    completed = manifest[manifest["status"].eq("done")]
    numeric_columns = (
        "row_index",
        "conf_id",
        "boltzmann_weight",
        "target_c_index",
        "target_o_index",
    )
    numeric_values: dict[str, pd.Series] = {}
    for column in numeric_columns:
        try:
            values = pd.to_numeric(completed[column], errors="raise")
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Completed rows in {manifest_path} have nonnumeric {column} values"
            ) from exc
        if values.isna().any() or not np.isfinite(values.to_numpy(dtype=float)).all():
            raise ValueError(
                f"Completed rows in {manifest_path} have missing or non-finite "
                f"{column} values"
            )
        numeric_values[column] = values
    integer_columns = ("row_index", "conf_id", "target_c_index", "target_o_index")
    for column in integer_columns:
        values = numeric_values[column].to_numpy(dtype=float)
        if not np.equal(values, np.floor(values)).all():
            raise ValueError(
                f"Completed rows in {manifest_path} have non-integer {column} values"
            )
    if (
        completed["sp_chk"].isna().any()
        or completed["sp_chk"].astype(str).str.strip().eq("").any()
    ):
        raise ValueError(f"Completed rows in {manifest_path} have empty sp_chk paths")
    row_indices = numeric_values["row_index"].to_numpy(dtype=int)
    if np.any(row_indices < 0) or np.any(row_indices >= dataset_size):
        raise ValueError(
            f"Completed manifest row_index values must be between 0 and {dataset_size - 1}"
        )
    return manifest


def output_feature_dataset(
    data: pd.DataFrame,
    orbital: np.ndarray,
    weight_sums: np.ndarray,
    required_rows: np.ndarray,
) -> pd.DataFrame:
    """Combine orbital bins with metadata and electronic/electrostatic fields.

    ``data`` is the already validated external source dataset. ``orbital`` must
    have shape ``(len(data), n_grid_bins)`` in :func:`grid_keys_from_bounds`
    order. ``weight_sums`` is one completed conformer-weight sum per dataset row
    and ``required_rows`` contains positional row indices. The function writes
    the combined pickle and a CSV completeness summary, then returns it.
    """
    meta_cols = [
        col
        for col in data.columns
        if not (
            col.startswith("electronic_fold ")
            or col.startswith("electrostatic_fold ")
            or col.startswith("lumo_fold ")
            or col.startswith("electronic_unfold ")
            or col.startswith("electrostatic_unfold ")
            or col.startswith("lumo_unfold ")
        )
    ]
    ee_cols = [
        col
        for col in data.columns
        if col.startswith("electronic_fold ") or col.startswith("electrostatic_fold ")
    ]
    orbital_cols = [f"orbital_fold {x} {y} {z}" for x, y, z in grid_keys_from_bounds(DEFAULT_ORBITAL_BUILD_BOUNDS)]
    out = pd.concat(
        [
            data[meta_cols + ee_cols].reset_index(drop=True),
            pd.DataFrame(orbital, columns=orbital_cols),
        ],
        axis=1,
    )
    out_path = OUT_DIR / "data_electronic_electrostatic_orbital.pkl"
    out.to_pickle(out_path)
    pd.DataFrame(
        {
            "n_rows": [len(out)],
            "n_orbital_features": [len(orbital_cols)],
            "n_electronic_features": [sum(col.startswith("electronic_fold ") for col in out.columns)],
            "n_electrostatic_features": [sum(col.startswith("electrostatic_fold ") for col in out.columns)],
            "n_required_rows": [int(len(required_rows))],
            "min_required_completed_weight_sum": [float(np.min(weight_sums[required_rows]))],
            "max_required_completed_weight_sum": [float(np.max(weight_sums[required_rows]))],
        }
    ).to_csv(OUT_DIR / "orbital_feature_summary.csv", index=False)
    return out


def build_features(
    dataset_pickle: Path,
    manifest_path: Path,
    limit: int | None = None,
    resume: bool = True,
    allow_partial: bool = False,
    needed_only: bool = False,
    workers: int = 1,
) -> pd.DataFrame:
    """Build or resume the full Boltzmann-averaged orbital feature dataset.

    Parameters
    ----------
    dataset_pickle
        External pickle containing the source descriptor DataFrame.
    manifest_path
        External full conformer manifest containing calculation paths and
        target-atom columns; the sanitized Git manifest is provenance-only.
    limit
        Optional maximum number of conformer jobs after manifest filtering.
    resume
        Reuse the NPZ accumulator and CSV status manifest when both exist.
    allow_partial
        Permit required dataset rows with zero completed conformer weight.
    needed_only
        Restrict submitted manifest jobs to required rows; currently every
        dataset row is required.
    workers
        Number of concurrent external-calculation worker threads.

    Returns
    -------
    pandas.DataFrame
        Metadata plus existing folded electronic/electrostatic columns and
        ``orbital_fold x y z`` columns containing weighted ``amplitude**2``
        sums on the fixed 2-bohr grid.
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TMP_ROOT.mkdir(parents=True, exist_ok=True)
    data = read_source_dataset(dataset_pickle)
    manifest = read_full_manifest(manifest_path, len(data))
    required_rows = required_row_indices(data, needed_only)
    required_set = set(map(int, required_rows))
    keys = grid_keys_from_bounds(DEFAULT_ORBITAL_BUILD_BOUNDS)
    orbital = np.zeros((len(data), len(keys)), dtype=float)
    weight_sums = np.zeros(len(data), dtype=float)
    progress_path = OUT_DIR / "homo_projected_orbital_progress.npz"
    status_path = OUT_DIR / "homo_projected_orbital_manifest.csv"

    status = pd.DataFrame()
    done_keys: set[tuple[int, int]] = set()
    if resume and progress_path.exists() and status_path.exists():
        progress = np.load(progress_path)
        orbital = progress["orbital"]
        weight_sums = progress["weight_sums"]
        if orbital.shape != (len(data), len(keys)) or weight_sums.shape != (len(data),):
            raise ValueError(
                "Saved progress shape does not match the supplied dataset; "
                "rerun with --no-resume"
            )
        status = pd.read_csv(status_path)
        status = status.drop_duplicates(["row_index", "conf_id"], keep="last")
        done = status[status["status"].eq("done")]
        done_keys = {(int(r.row_index), int(r.conf_id)) for r in done.itertuples(index=False)}

    jobs = manifest[manifest["status"].eq("done")].copy()
    if needed_only:
        jobs = jobs[jobs["row_index"].astype(int).isin(required_set)].copy()
    jobs = jobs[jobs["sp_chk"].map(lambda p: Path(str(p)).with_suffix(".fchk").exists())].copy()
    jobs = jobs.sort_values(["row_index", "boltzmann_weight"], ascending=[True, False])
    if limit is not None:
        jobs = jobs.head(limit)

    status_rows = [] if status.empty else status.to_dict("records")
    pending = []
    for row in jobs.itertuples(index=False):
        row_dict = row._asdict()
        key = (int(row_dict["row_index"]), int(row_dict["conf_id"]))
        if key not in done_keys:
            pending.append(row_dict)

    total = len(pending)
    workers = max(1, int(workers))

    def store_result(
        count: int,
        row_index: int,
        conf_id: int,
        bins: np.ndarray | None,
        record: dict[str, object],
    ) -> None:
        """Accumulate one worker result and periodically checkpoint progress."""
        if record["status"] == "done":
            if bins is None:
                record.update(status="failed", skip_reason="worker returned no bins")
            else:
                orbital[row_index, :] += bins
                weight_sums[row_index] += float(record["boltzmann_weight"])
        status_rows.append(record)
        if count % 5 == 0 or record["status"] != "done":
            status_current = pd.DataFrame(status_rows).drop_duplicates(["row_index", "conf_id"], keep="last")
            status_current.to_csv(status_path, index=False)
            np.savez_compressed(progress_path, orbital=orbital, weight_sums=weight_sums)
            print(
                f"processed {count}/{total}: row {row_index} {record['entry']} conf{conf_id} {record['status']}",
                flush=True,
            )

    if workers == 1:
        for count, row_dict in enumerate(pending, start=1):
            store_result(count, *run_orbital_feature_job(row_dict))
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(run_orbital_feature_job, row_dict) for row_dict in pending]
            for count, future in enumerate(as_completed(futures), start=1):
                store_result(count, *future.result())

    status_final = pd.DataFrame(status_rows).drop_duplicates(["row_index", "conf_id"], keep="last")
    status_final.to_csv(status_path, index=False)
    np.savez_compressed(progress_path, orbital=orbital, weight_sums=weight_sums)
    failed = status_final[status_final["row_index"].astype(int).isin(required_set)]
    if not failed.empty and failed["status"].ne("done").any():
        bad = failed[failed["status"].ne("done")]
        bad.to_csv(OUT_DIR / "homo_projected_orbital_failed_jobs.csv", index=False)
        raise RuntimeError(f"{len(bad)} projected orbital jobs failed; see homo_projected_orbital_failed_jobs.csv")

    incomplete = required_rows[weight_sums[required_rows] <= 0.0]
    if len(incomplete) and not allow_partial:
        raise RuntimeError(f"{len(incomplete)} data rows have zero completed conformer weight: {incomplete[:20].tolist()}")
    return output_feature_dataset(data, orbital, weight_sums, required_rows)


def main() -> None:
    """Run the ``build-features`` or ``summary`` command-line workflow."""
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    build = sub.add_parser("build-features")
    build.add_argument(
        "--dataset-pickle",
        type=Path,
        required=True,
        help="external source descriptor DataFrame pickle",
    )
    build.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="external full conformer manifest (not the sanitized Git manifest)",
    )
    build.add_argument("--limit", type=int, default=None)
    build.add_argument("--no-resume", action="store_true")
    build.add_argument("--allow-partial", action="store_true")
    build.add_argument("--needed-only", action="store_true")
    build.add_argument("--workers", type=int, default=1)
    sub.add_parser("summary")
    args = parser.parse_args()

    if args.cmd == "build-features":
        out = build_features(
            dataset_pickle=args.dataset_pickle,
            manifest_path=args.manifest,
            limit=args.limit,
            resume=not args.no_resume,
            allow_partial=args.allow_partial,
            needed_only=args.needed_only,
            workers=args.workers,
        )
        print(f"wrote feature dataset shape={out.shape}")
    elif args.cmd == "summary":
        for path in [
            OUT_DIR / "homo_projected_orbital_manifest.csv",
            OUT_DIR / "orbital_feature_summary.csv",
        ]:
            if path.exists():
                print(f"\n== {path.name}")
                print(pd.read_csv(path).head(20).to_string(index=False))


if __name__ == "__main__":
    main()
