"""
Grid-based electronic and electrostatic descriptor generation.

This module provides utilities to:
- Extract grid data (density/electrostatics/LUMO) from Gaussian cube files
- Thermodynamically weight conformers using cclib thermochemistry
- Aggregate/fold grid values onto a coarse 3D lattice
- Batch-process multiple molecules listed in an Excel file
"""

from __future__ import annotations

from itertools import product
from multiprocessing import Pool
from pathlib import Path
import os
import re

import numpy as np
import pandas as pd
import cclib

try:
    from libs.current_model_support.conformer_helpers import (
        conformer_id_from_path,
        discover_conformer_logs,
    )
except ImportError:  # Support direct execution as ``python libs/legacy/calc_grid.py``.
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from libs.current_model_support.conformer_helpers import (  # type: ignore[no-redef]
        conformer_id_from_path,
        discover_conformer_logs,
    )


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


OUTPUT_ROOT = os.getenv(
    "MOLECULES_ROOT",
    os.path.join(os.path.expanduser("~"), "molecules"),
)
NUM_WORKERS = _positive_int_from_env("GRID_NUM_WORKERS", os.cpu_count() or 1)

# Descriptor settings selected from the 2026-06 model search:
# no radial cutoff/taper, electronic density clipped at 1e-3, ESP mask at 1e-2.
GRID_RADIUS = np.inf
GRID_STEP = 2.0
ELECTRONIC_DENSITY_CLIP = 1e-3
ESP_DENSITY_MASK_THRESHOLD = 1e-2
APPLY_RADIAL_TAPER = False
TAPER_POWER = 1.0

# Manuscript model selected from the 2026-06-29 search:
# for initial diketone reductions, augment the LUMO descriptor with an
# energy-weighted LUMO+1 term computed conformer-by-conformer.
USE_LUMO_PLUS1_FOR_INITIAL_DIKETONES = os.getenv(
    "GRID_USE_LUMO_PLUS1_INITIAL_DIKETONES",
    "1",
).strip().lower() not in {"0", "false", "no", "off"}
LUMO_PLUS1_TAU_EV = float(os.getenv("GRID_LUMO_PLUS1_TAU_EV", "0.025"))
LUMO_PLUS1_COMBINE_MODE = os.getenv("GRID_LUMO_PLUS1_COMBINE_MODE", "additive").strip()
DIKETONE_INITIAL_ENTRY_RE = re.compile(r"^[a-f][1-4]$")


def _cube_path(log: str, prefix: str) -> str:
    """Return ``<prefix><conformer-id>.cube`` beside an ``opt<id>.log`` file."""
    log_path = Path(log)
    conf_id = conformer_id_from_path(log_path)
    return str(log_path.with_name(f"{prefix}{conf_id}.cube"))


def _lumo_plus1_path(log: str) -> str:
    """Return the LUMO+1 cube path matching an ``opt<id>.log`` file."""
    log_path = Path(log)
    conf_id = conformer_id_from_path(log_path)
    return str(log_path.with_name(f"LUMO+1_{conf_id}.cube"))


def _sp_log_path(log: str) -> str:
    """Return the single-point log path matching an ``opt<id>.log`` file."""
    log_path = Path(log)
    conf_id = conformer_id_from_path(log_path)
    return str(log_path.with_name(f"sp{conf_id}.log"))


def _read_mo_cube_values(cube_path: str, n_atom: int, n_values: int) -> np.ndarray:
    """Read MO cube values robustly across Gaussian cube header variants."""
    # Some MO cubes contain an extra orbital-index line after the atom block,
    # while others do not. Read after the atom block and take the final grid
    # values so both variants are handled consistently.
    with open(cube_path, "r", encoding="UTF-8") as f:
        for _ in range(6 + abs(n_atom)):
            f.readline()
        raw_values = np.fromstring(f.read(), dtype=float, sep=" ")

    if raw_values.size < n_values:
        raise ValueError(
            f"{cube_path} has {raw_values.size} numeric values, expected at least {n_values}"
        )
    return raw_values[-n_values:].reshape(-1, 1)


def _lumo_plus1_energy_weight(log: str) -> float:
    """Compute the conformer-specific LUMO+1 mixing weight from SP MO energies."""
    sp_data = cclib.io.ccread(_sp_log_path(log))
    if sp_data is None or not hasattr(sp_data, "homos") or not hasattr(sp_data, "moenergies"):
        raise ValueError(f"Failed to read MO energies from {_sp_log_path(log)}")
    homo_index = int(sp_data.homos[0])
    mo_energies = np.asarray(sp_data.moenergies[0], dtype=float)
    delta_e = float(mo_energies[homo_index + 2] - mo_energies[homo_index + 1])
    exponent = np.clip(delta_e / LUMO_PLUS1_TAU_EV, -700, 700)
    return float(1.0 / (1.0 + np.exp(exponent)))


def calc_grid__(
    log: str,
    T: float,
    use_lumo_plus1: bool = False,
) -> tuple[pd.DataFrame, float, float]:
    """Extract grid data (density, ESP, LUMO) and a thermodynamic weight from a single log/cube set.

    This function reads a Gaussian log file with thermochemical data (via cclib) and
    the corresponding cube files (`Dt`, `ESP`, `LUMO`) and returns:

    - A DataFrame with:
        - x, y, z: grid point coordinates
        - electronic: scalar field from the Dt cube (e.g., density-like quantity)
        - electrostatic: ESP values from the ESP cube
        - lumo: LUMO MO amplitude values from the LUMO cube
    - A scalar "weight" derived from enthalpy and entropy at temperature T.

    Parameters
    ----------
    log : str
        Path to the Gaussian log file (optimization log). The corresponding cube files
        are inferred by:
        - Dt cube: ``Dt<id>.cube``
        - ESP cube: ``ESP<id>.cube``
        - LUMO cube: ``LUMO<id>.cube``
    T : float
        Temperature [K]. Used to compute a Gibbs-like weight from the cclib thermochemistry.
    use_lumo_plus1 : bool
        Also read ``LUMO+1_<id>.cube`` and compute its SP-energy mixing weight.

    Returns
    -------
    tuple[pandas.DataFrame, float, float]
        df : pandas.DataFrame
            Columns:
                - "x", "y", "z": Cartesian coordinates of grid points (float)
                - "electronic": values from the Dt cube (float)
                - "electrostatic": values from the ESP cube (float)
                - "lumo": values from the LUMO cube (float)
        weight : float
            Gibbs-like quantity `G = enthalpy - T * entropy` extracted from the log file.
        lumo_plus1_weight : float
            Conformer-specific energy weight for LUMO+1. Zero when LUMO+1 is disabled.

    Notes
    -----
    - The cube header is assumed to follow the standard Gaussian format:
      line 3: number of atoms and origin
      lines 4–6: grid sizes and axis vectors.
    """
    # Parse thermochemistry from log
    data = cclib.io.ccread(log)
    if data is None or not hasattr(data, "enthalpy") or not hasattr(data, "entropy"):
        raise ValueError(f"Failed to read thermochemistry from {log}")
    weight = data.enthalpy - data.entropy * T

    # Infer cube file paths
    dt_path = _cube_path(log, "Dt")
    esp_path = _cube_path(log, "ESP")
    lumo_path = _cube_path(log, "LUMO")

    # ----- Read Dt cube (reference for grid geometry) -----
    with open(dt_path, "r", encoding="UTF-8") as f:
        # Skip title/comment lines
        f.readline()
        f.readline()

        # Number of atoms and origin
        n_atom_str, x0_str, y0_str, z0_str, _ = f.readline().split()
        n1_str, x1_str, y1_str, z1_str = f.readline().split()
        n2_str, x2_str, y2_str, z2_str = f.readline().split()
        n3_str, x3_str, y3_str, z3_str = f.readline().split()

        n_atom = int(n_atom_str)
        origin = np.array([x0_str, y0_str, z0_str], dtype=float)
        size = np.array([n1_str, n2_str, n3_str], dtype=int)
        axis = np.array(
            [
                [x1_str, y1_str, z1_str],
                [x2_str, y2_str, z2_str],
                [x3_str, y3_str, z3_str],
            ],
            dtype=float,
        )

        # Generate Cartesian coordinates for all grid points
        # (i, j, k) indices multiplied by axis vectors, then shifted by origin
        ijk = np.array(list(product(range(size[0]), range(size[1]), range(size[2]))))
        coord = ijk @ axis + origin

        # Skip atomic lines｀
        for _ in range(abs(n_atom)):
            f.readline()

        # Read Dt values
        dt_values = np.fromstring(f.read(), dtype=float, sep=" ").reshape(-1, 1)

    # ----- Read ESP cube -----
    with open(esp_path, "r", encoding="UTF-8") as f:
        # Skip header + atomic lines
        for _ in range(6 + abs(n_atom)):
            f.readline()
        esp_values = np.fromstring(f.read(), dtype=float, sep=" ").reshape(-1, 1)

    n_values = int(np.prod(size))

    # ----- Read LUMO cube -----
    lumo_values = _read_mo_cube_values(lumo_path, n_atom, n_values)

    if use_lumo_plus1:
        lumo_plus1_values = _read_mo_cube_values(_lumo_plus1_path(log), n_atom, n_values)
        lumo_plus1_weight = _lumo_plus1_energy_weight(log)
    else:
        lumo_plus1_values = np.zeros_like(lumo_values)
        lumo_plus1_weight = 0.0

    # Build DataFrame
    df = pd.DataFrame(
        data=np.hstack((coord, dt_values, esp_values, lumo_values, lumo_plus1_values)),
        columns=["x", "y", "z", "electronic", "electrostatic", "lumo", "lumo_plus1"],
    )

    return df, weight, lumo_plus1_weight


def calc_grid(
    path: str,
    T: float,
    folded: int,
    use_lumo_plus1: bool = False,
) -> pd.Series:
    """Aggregate weighted grid values for all ``opt<digits>.log`` files.

    For each log/cube set in `path`, this function:
    1. Extracts grid data and Gibbs-like weights via :func:`calc_grid__`.
    2. Applies various filters and radial weighting to electronic/ESP/LUMO fields.
    3. Coarse-grains the grid to integer coordinates (after scaling by 1/2).
    4. Thermodynamically weights conformers using a Boltzmann-like factor.
    5. Aggregates:
        - Unfolded grid data (`*_unfold x y z`)
        - Folded grid data (mirroring y to |y|; `*_fold x y z`)

    Parameters
    ----------
    path : str
        Directory containing Gaussian log files named exactly
        ``opt<digits>.log``. Corresponding cubes must be in the same directory.
    T : float
        Temperature [K], used in the Boltzmann factors `exp(-ΔG / (3.1668114e-6 * T))`.
    folded : int
        Factor for the z-coordinate (e.g., 1 or -1) applied before folding.
    use_lumo_plus1 : bool
        Include the configured energy-weighted LUMO+1 term for initial
        diketone reductions.

    Returns
    -------
    pandas.Series
        A concatenated Series containing:
        - Unfolded grid:
            - indices of the form "electronic_unfold x y z"
            - indices of the form "electrostatic_unfold x y z"
            - indices of the form "lumo_unfold x y z"
        - Folded grid:
            - "electronic_fold x y z"
            - "electrostatic_fold x y z"
            - "lumo_fold x y z"

        Values are the aggregated, thermodynamically weighted grid quantities.

    Notes
    -----
    - Grid points are optionally restricted to within ``GRID_RADIUS`` from origin.
    - Electronic density is clipped at ``ELECTRONIC_DENSITY_CLIP``.
    - ESP is masked by the original electronic density with
      ``ESP_DENSITY_MASK_THRESHOLD``.
    - Fields are optionally tapered to zero at ``GRID_RADIUS``.
    - Coordinates are scaled by ``GRID_STEP``, then rounded away from zero
      (ceil for positive, floor for negative).
    - Folding is applied by taking the absolute value of y (mirror in y) and
      multiplying z by `folded`.
    """
    grids = []
    weights = []
    failures: list[tuple[str, str]] = []

    # Loop over optimization logs
    for log_path in discover_conformer_logs(path):
        log = str(log_path)
        try:
            df, weight, lumo_plus1_weight = calc_grid__(log, T, use_lumo_plus1)
            print(f"PARSING SUCCESS {log}")
        except Exception as e:  # noqa: BLE001
            print(f"PARSING FAILURE {log}")
            print(e)
            failures.append((log, str(e)))
            continue

        # Restrict to points within the selected radial cutoff, if enabled.
        r2 = df["x"] ** 2 + df["y"] ** 2 + df["z"] ** 2
        if np.isfinite(GRID_RADIUS):
            df = df[r2 < GRID_RADIUS**2].copy()

        # Clamp / transform electronic & electrostatic fields
        df["electrostatic"] = df["electrostatic"] * np.where(
            df["electronic"] < ESP_DENSITY_MASK_THRESHOLD,
            ESP_DENSITY_MASK_THRESHOLD - df["electronic"],
            0.0,
        )
        df["electronic"] = np.where(
            df["electronic"] < ELECTRONIC_DENSITY_CLIP,
            df["electronic"],
            ELECTRONIC_DENSITY_CLIP,
        )

        # LUMO: square amplitude. For initial diketone reductions, add an
        # energy-weighted LUMO+1 square term using the conformer's SP MO gap.
        lumo_sq = df["lumo"] ** 2
        if use_lumo_plus1:
            lumo_plus1_sq = df["lumo_plus1"] ** 2
            if LUMO_PLUS1_COMBINE_MODE == "additive":
                lumo_sq = lumo_sq + lumo_plus1_weight * lumo_plus1_sq
            elif LUMO_PLUS1_COMBINE_MODE == "convex":
                lumo_sq = (1.0 - lumo_plus1_weight) * lumo_sq + (
                    lumo_plus1_weight * lumo_plus1_sq
                )
            else:
                raise ValueError(f"Unsupported LUMO+1 combine mode: {LUMO_PLUS1_COMBINE_MODE}")
        df["lumo"] = lumo_sq

        # Radial taper to zero at the selected cutoff radius, if enabled.
        if APPLY_RADIAL_TAPER and np.isfinite(GRID_RADIUS):
            r = np.linalg.norm(df[["x", "y", "z"]], axis=1).reshape(-1, 1)
            taper = np.where(r < GRID_RADIUS, 1.0 - r / GRID_RADIUS, 0.0)
            taper = taper**TAPER_POWER
            df[["electronic", "electrostatic", "lumo"]] *= taper

        # Coarse grid: scale and round
        df[["x", "y", "z"]] /= GRID_STEP
        df[["x", "y", "z"]] = np.where(
            df[["x", "y", "z"]] > 0,
            np.ceil(df[["x", "y", "z"]]),
            np.floor(df[["x", "y", "z"]]),
        ).astype(int)

        # Group by coarse grid and sum fields
        df = (
            df.groupby(["x", "y", "z"], as_index=False)[
                ["electronic", "electrostatic", "lumo"]
            ]
            .sum()
        )

        grids.append(df.copy())
        weights.append(weight)

    if failures:
        details = "\n".join(f"- {log}: {error}" for log, error in failures)
        raise RuntimeError(
            "One or more conformers failed descriptor parsing; refusing to "
            f"renormalize the remaining conformers:\n{details}"
        )
    if not grids:
        raise FileNotFoundError(f"No valid Gaussian conformer logs found under {path}")

    # Compute one Boltzmann weight per conformer and apply it globally.
    # Missing voxels in a conformer then contribute zero rather than
    # renormalizing the conformer weights locally at each grid point.
    gibbs = np.asarray(weights, dtype=float)
    delta = gibbs - np.min(gibbs)
    expo = -delta / (3.1668114e-6 * T)
    expo = np.clip(expo, -700, 0)
    boltz = np.exp(expo)
    boltz /= np.sum(boltz)

    weighted_grids = []
    for grid, weight in zip(grids, boltz):
        weighted = grid.copy()
        weighted[["electronic", "electrostatic", "lumo"]] *= weight
        weighted_grids.append(weighted)

    wgrids = pd.concat(weighted_grids, ignore_index=True)
    wgrids = (
        wgrids.groupby(["x", "y", "z"], as_index=False)[
            ["electronic", "electrostatic", "lumo"]
        ]
        .sum()
        .astype({"x": int, "y": int, "z": int})
    )

    # Apply z-fold factor
    wgrids[["z"]] = wgrids[["z"]] * folded

    # Unfolded series
    electronic_unfold = pd.Series(
        {
            f"electronic_unfold {int(row.x)} {int(row.y)} {int(row.z)}": row.electronic
            for _, row in wgrids.iterrows()
        }
    )
    electrostatic_unfold = pd.Series(
        {
            f"electrostatic_unfold {int(row.x)} {int(row.y)} {int(row.z)}": row.electrostatic
            for _, row in wgrids.iterrows()
        }
    )
    lumo_unfold = pd.Series(
        {
            f"lumo_unfold {int(row.x)} {int(row.y)} {int(row.z)}": row.lumo
            for _, row in wgrids.iterrows()
        }
    )

    # Fold in y (mirror) and re-aggregate
    wgrids_fold = wgrids.copy()
    wgrids_fold[["y"]] = wgrids_fold[["y"]].abs()
    wgrids_fold = (
        wgrids_fold.groupby(["x", "y", "z"], as_index=False)[
            ["electronic", "electrostatic", "lumo"]
        ]
        .sum()
    )

    electronic_fold = pd.Series(
        {
            f"electronic_fold {int(row.x)} {int(row.y)} {int(row.z)}": row.electronic
            for _, row in wgrids_fold.iterrows()
        }
    )
    electrostatic_fold = pd.Series(
        {
            f"electrostatic_fold {int(row.x)} {int(row.y)} {int(row.z)}": row.electrostatic
            for _, row in wgrids_fold.iterrows()
        }
    )
    lumo_fold = pd.Series(
        {
            f"lumo_fold {int(row.x)} {int(row.y)} {int(row.z)}": row.lumo
            for _, row in wgrids_fold.iterrows()
        }
    )

    return pd.concat(
        [
            electronic_unfold,
            electrostatic_unfold,
            lumo_unfold,
            electronic_fold,
            electrostatic_fold,
            lumo_fold,
        ]
    )


def process_row(row: pd.Series) -> pd.Series:
    """Wrapper for multiprocessing: compute grid features for a single dataframe row.

    Parameters
    ----------
    row : pandas.Series
        A row from the input Excel DataFrame. It must contain:
        - "InChIKey": used to locate the molecule directory under
          OUTPUT_ROOT/<InChIKey>
        - "temperature": temperature [K] for this molecule.

    Returns
    -------
    pandas.Series
        The Series returned by :func:`calc_grid`, i.e. aggregated grid descriptors for
        this molecule. The index is a set of feature names; the single row corresponds
        to one molecule.
    """
    target_dir = Path(OUTPUT_ROOT) / row["InChIKey"]
    entry = str(row.get("entry", ""))
    use_lumo_plus1 = (
        USE_LUMO_PLUS1_FOR_INITIAL_DIKETONES
        and DIKETONE_INITIAL_ENTRY_RE.match(entry) is not None
    )
    return calc_grid(
        str(target_dir),
        row["temperature"],
        folded=1,
        use_lumo_plus1=use_lumo_plus1,
    )


def calc_grid_(path: str) -> None:
    """Batch-process grid features for all molecules listed in an Excel file.

    This function:
    1. Reads molecular data from an Excel file.
    2. For each row (molecule), locates the corresponding directory under
       `OUTPUT_ROOT/<InChIKey>` and computes grid descriptors via :func:`calc_grid`.
    3. Combines the grid descriptors with the original DataFrame.
    4. Saves the result as:
        - A pickle file with the same basename (``.pkl``)
        - A CSV file with the suffix ``feat.csv``

    Parameters
    ----------
    path : str
        Path to the Excel (.xlsx) file containing molecular data.
        Required columns:
            - "InChIKey"
            - "temperature"

    Returns
    -------
    None
        Results are written to disk as pickle and CSV files.

    Example
    -------
    >>> calc_grid_("data/data.xlsx")
    """
    print(f"START PARSING {path}")
    df = pd.read_excel(path)

    rows = [row for _, row in df.iterrows()]
    if NUM_WORKERS <= 1:
        results = [process_row(row) for row in rows]
    else:
        # Multiprocessing over rows (worker count is configurable via NUM_WORKERS).
        with Pool(NUM_WORKERS) as pool:
            results = pool.map(process_row, rows)

    # Missing spatial cells mean zero contribution, whereas missing response
    # values and metadata must remain missing rather than becoming false zeros.
    features = pd.DataFrame(results).fillna(0)
    df_out = pd.concat([df, features], axis=1)

    # Save as pickle and CSV
    pkl_path = path.replace(".xlsx", ".pkl")
    csv_path = path.replace(".xlsx", "feat.csv")

    df_out.to_pickle(pkl_path)
    df_out.to_csv(csv_path, index=False)


if __name__ == "__main__":
    calc_grid_("data/data.xlsx")
