"""Build frozen eCoMFA-like electronic and electrostatic grid descriptors.

For each substrate row, Gaussian density and electrostatic-potential cubes are
loaded from ``MOLECULE_ROOT/<InChIKey>`` and grouped onto a folded 2-bohr grid.
Electron density is transformed by a log-normal kernel centred at ``1e-2``;
the electrostatic field is the potential multiplied by the same kernel.
Conformers are averaged with thermochemical Boltzmann weights derived at the
row's ``temperature`` in kelvin. Coordinates and Gaussian thermochemistry use
atomic units; returned feature keys are dimensionless integer grid labels.
"""
from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path

import cclib
import numpy as np
import pandas as pd

from .conformer_helpers import conformer_id_from_path, discover_conformer_logs


MOLECULE_ROOT = Path.home() / "molecules"
LABEL = "ecomfa_kernel_1e-2_v1"
BOLTZMANN_KT_AU = 3.1668114e-6


def read_cube_values(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Read coordinates and scalar values from a Gaussian cube file.

    Coordinates are returned in the cube's native unit (bohr for project
    ``cubegen`` outputs) with shape ``(n_grid, 3)``. Scalar values are returned
    as a flat array in Gaussian cube traversal order. The routine is intended
    for single-valued density and electrostatic-potential cubes.
    """
    with open(path, encoding="UTF-8") as handle:
        handle.readline()
        handle.readline()
        atom_line = handle.readline().split()
        atom_n = abs(int(atom_line[0]))
        origin = np.asarray(atom_line[1:4], dtype=float)
        sizes: list[int] = []
        axes: list[list[float]] = []
        for _ in range(3):
            line = handle.readline().split()
            sizes.append(int(line[0]))
            axes.append([float(value) for value in line[1:4]])
        for _ in range(atom_n):
            handle.readline()
        values = np.fromstring(handle.read(), dtype=float, sep=" ")
    coordinates = np.indices(np.asarray(sizes), dtype=float).reshape(3, -1).T @ np.asarray(axes) + origin
    return coordinates, values


def read_conformer(log_path: str, temperature: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Load one conformer's density, potential, coordinates, and free energy.

    ``log_path`` names ``opt<id>.log``; matching ``Dt<id>.cube`` and
    ``ESP<id>.cube`` paths are inferred. ``temperature`` is in kelvin and the
    returned ``H - T*S`` value is in hartree. Cube coordinates are in bohr,
    while the two scalar arrays retain their Gaussian cube units.
    """
    data = cclib.io.ccread(log_path)
    if data is None or not hasattr(data, "enthalpy") or not hasattr(data, "entropy"):
        raise ValueError(f"Could not read thermochemistry from {log_path}")
    gibbs = float(data.enthalpy - data.entropy * temperature)
    path = Path(log_path)
    conf_id = conformer_id_from_path(path)
    density_path = path.with_name(f"Dt{conf_id}.cube")
    esp_path = path.with_name(f"ESP{conf_id}.cube")
    coordinates, electronic = read_cube_values(str(density_path))
    _, electrostatic = read_cube_values(str(esp_path))
    if not (len(coordinates) == len(electronic) == len(electrostatic)):
        raise ValueError(f"Cube size mismatch for {log_path}")
    return coordinates, electronic, electrostatic, gibbs


def normal_kernel(values: np.ndarray, center: float = 1e-2, variance: float = 1.0) -> np.ndarray:
    """Apply the adopted Gaussian kernel in log-density space.

    ``values`` and ``center`` are electron-density-like positive quantities in
    the same units. ``variance`` is dimensionless variance of ``log(values)``.
    Values at or below zero are floored at ``1e-300`` before taking a logarithm.
    """
    log_values = np.log(np.maximum(values, 1e-300))
    return np.exp(-((log_values - math.log(center)) ** 2) / (2.0 * variance)) / np.sqrt(
        2.0 * np.pi * variance
    )


def bin_coordinates(coordinates: np.ndarray, grid_step: float = 2.0) -> np.ndarray:
    """Map Cartesian cube coordinates to folded integer grid labels.

    ``coordinates`` and ``grid_step`` must share units (bohr in this project).
    Nonzero positions are rounded away from zero and the y label is replaced by
    its absolute value, encoding the model's two-face symmetry. The result has
    shape ``(n_grid, 3)`` and integer dtype.
    """
    scaled = coordinates / grid_step
    binned = np.where(scaled > 0, np.ceil(scaled), np.floor(scaled)).astype(np.int16)
    binned[:, 1] = np.abs(binned[:, 1])
    return binned


def aggregate_conformer(
    coordinates: np.ndarray,
    electronic_raw: np.ndarray,
    electrostatic_raw: np.ndarray,
) -> dict[str, dict[tuple[int, int, int], float]]:
    """Aggregate one conformer's transformed fields on the folded 2-bohr grid.

    All three input arrays describe the same cube traversal order. The returned
    mapping has ``electronic`` and ``electrostatic`` blocks, each keyed by an
    integer ``(x, |y|, z)`` grid coordinate. Values within the same coarse cell
    are summed; no Boltzmann weight is applied at this stage.
    """
    kernel = normal_kernel(electronic_raw)
    transformed = {"electronic": kernel, "electrostatic": electrostatic_raw * kernel}
    unique, inverse = np.unique(bin_coordinates(coordinates), axis=0, return_inverse=True)
    output: dict[str, dict[tuple[int, int, int], float]] = {}
    for block, values in transformed.items():
        sums = np.bincount(inverse, weights=values)
        output[block] = {
            tuple(map(int, xyz)): float(value)
            for xyz, value in zip(unique, sums)
            if value != 0
        }
    return output


def calc_transform_features_for_row(row: pd.Series) -> dict[str, pd.Series]:
    """Build Boltzmann-weighted folded descriptors for one substrate row.

    Required fields are ``InChIKey`` and ``temperature`` (kelvin). All readable
    ``opt<id>.log``/cube sets under the molecule directory are transformed and
    averaged. The return mapping contains the fixed transform ``LABEL`` and a
    Series whose indices follow ``"<block>_fold x y z"``. If no conformer can
    be read, the Series is empty rather than populated with zeros.
    """
    molecule_dir = MOLECULE_ROOT / str(row["InChIKey"])
    conformers: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    gibbs_values: list[float] = []
    for log_path_obj in discover_conformer_logs(molecule_dir):
        log_path = str(log_path_obj)
        try:
            coordinates, electronic, electrostatic, gibbs = read_conformer(
                log_path, float(row["temperature"])
            )
        except Exception as exc:  # noqa: BLE001
            print(f"PARSING FAILURE {log_path}: {exc}", flush=True)
            continue
        conformers.append((coordinates, electronic, electrostatic))
        gibbs_values.append(gibbs)
    if not conformers:
        return {LABEL: pd.Series(dtype=float)}

    gibbs = np.asarray(gibbs_values)
    exponent = np.clip(
        -(gibbs - np.min(gibbs)) / (BOLTZMANN_KT_AU * float(row["temperature"])),
        -700.0,
        0.0,
    )
    weights = np.exp(exponent)
    weights /= weights.sum()
    accumulators = {
        "electronic": defaultdict(float),
        "electrostatic": defaultdict(float),
    }
    for conformer, weight in zip(conformers, weights):
        for block, values in aggregate_conformer(*conformer).items():
            for xyz, value in values.items():
                accumulators[block][xyz] += value * float(weight)
    series = {
        f"{block}_fold {x} {y} {z}": value
        for block, grid in accumulators.items()
        for (x, y, z), value in grid.items()
    }
    return {LABEL: pd.Series(series, dtype=float)}
