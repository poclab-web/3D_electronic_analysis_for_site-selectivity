"""
Grid-based electronic and electrostatic descriptor generation.

This module provides utilities to:
- Extract grid data (density/electrostatics/LUMO) from Gaussian cube files
- Thermodynamically weight conformers using cclib thermochemistry
- Aggregate/fold grid values onto a coarse 3D lattice
- Batch-process multiple molecules listed in an Excel file
"""

from itertools import product
from multiprocessing import Pool
from pathlib import Path
import glob

import numpy as np
import pandas as pd
import cclib


def calc_grid__(log: str, T: float):
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
        - Dt cube : log.replace("opt", "Dt").replace(".log", ".cube")
        - ESP cube: log.replace("opt", "ESP").replace(".log", ".cube")
        - LUMO cube: log.replace("opt", "LUMO").replace(".log", ".cube")
    T : float
        Temperature [K]. Used to compute a Gibbs-like weight from the cclib thermochemistry.

    Returns
    -------
    tuple[pandas.DataFrame, float]
        df : pandas.DataFrame
            Columns:
                - "x", "y", "z": Cartesian coordinates of grid points (float)
                - "electronic": values from the Dt cube (float)
                - "electrostatic": values from the ESP cube (float)
                - "lumo": values from the LUMO cube (float)
        weight : float
            Gibbs-like quantity `G = enthalpy - T * entropy` extracted from the log file.

    Notes
    -----
    - The cube header is assumed to follow the standard Gaussian format:
      line 3: number of atoms and origin
      lines 4–6: grid sizes and axis vectors.
    """
    # Parse thermochemistry from log
    data = cclib.io.ccread(log)
    weight = data.enthalpy - data.entropy * T

    # Infer cube file paths
    dt_path = log.replace("opt", "Dt").replace(".log", ".cube")
    esp_path = log.replace("opt", "ESP").replace(".log", ".cube")
    lumo_path = log.replace("opt", "LUMO").replace(".log", ".cube")

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

        # Skip atomic lines
        for _ in range(n_atom):
            f.readline()

        # Read Dt values
        dt_values = np.fromstring(f.read(), dtype=float, sep=" ").reshape(-1, 1)

    # ----- Read ESP cube -----
    with open(esp_path, "r", encoding="UTF-8") as f:
        # Skip header + atomic lines
        for _ in range(6 + n_atom):
            f.readline()
        esp_values = np.fromstring(f.read(), dtype=float, sep=" ").reshape(-1, 1)

    # ----- Read LUMO cube -----
    # Note: one extra line after atoms (MO index line)
    with open(lumo_path, "r", encoding="UTF-8") as f:
        for _ in range(6 + n_atom + 1):
            f.readline()
        lumo_values = np.fromstring(f.read(), dtype=float, sep=" ").reshape(-1, 1)

    # Build DataFrame
    df = pd.DataFrame(
        data=np.hstack((coord, dt_values, esp_values, lumo_values)),
        columns=["x", "y", "z", "electronic", "electrostatic", "lumo"],
    )

    return df, weight


def calc_grid(path: str, T: float, folded: int) -> pd.Series:
    """Aggregate weighted grid values for all opt*.log files under a directory.

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
        Directory containing Gaussian log files named like `opt*.log`. For each log,
        corresponding cube files are assumed to exist in the same directory.
    T : float
        Temperature [K], used in the Boltzmann factors `exp(-ΔG / (3.1668114e-6 * T))`.
    folded : int
        Factor for the z-coordinate (e.g., 1 or -1) applied before folding.

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
    - Grid points are first restricted to within radius 8 (in Å) from origin.
    - Fields are tapered to zero at radius 8.
    - Coordinates are scaled by 1/2, then rounded to the nearest integer
      (ceil for positive, floor for negative).
    - Folding is applied by taking the absolute value of y (mirror in y) and
      multiplying z by `folded`.
    """
    grids = []
    weights = []

    # Loop over optimization logs
    for log in glob.glob(f"{path}/opt*.log"):
        try:
            df, weight = calc_grid__(log, T)
            print(f"PARSING SUCCESS {log}")
        except Exception as e:  # noqa: BLE001
            print(f"PARSING FAILURE {log}")
            print(e)
            continue

        # Restrict to points within radius 8
        r2 = df["x"] ** 2 + df["y"] ** 2 + df["z"] ** 2
        df = df[r2 < 8**2].copy()

        # Clamp / transform electronic & electrostatic fields
        df["electrostatic"] = df["electrostatic"] * np.where(
            df["electronic"] < 1e-2, 1e-2 - df["electronic"], 0.0
        )
        df["electronic"] = np.where(
            df["electronic"] < 1e-2, df["electronic"], 1e-2
        )

        # LUMO: square amplitude
        df["lumo"] = df["lumo"] ** 2

        # Radial taper to zero at r = 8
        r = np.linalg.norm(df[["x", "y", "z"]], axis=1).reshape(-1, 1)
        taper = np.where(r < 8.0, 1.0 - r / 8.0, 0.0)
        df[["electronic", "electrostatic", "lumo"]] *= taper

        # Coarse grid: scale and round
        df[["x", "y", "z"]] /= 2.0
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

        # Attach Gibbs-like weight per conformer
        df["gibbs"] = weight
        grids.append(df.copy())
        weights.append(weight)

    if not grids:
        # No valid grids found; return empty Series
        return pd.Series(dtype=float)

    def _total_keepnoindex(d: pd.DataFrame) -> pd.DataFrame:
        """Thermodynamically weight all rows in group `d` (same x,y,z)."""
        weights_arr = d.gibbs.values
        # Shift to minimum
        delta = weights_arr - np.min(weights_arr)

        # Basic Boltzmann factor on Gibbs-like weight
        boltz = np.exp(-delta / (3.1668114e-6 * T))
        boltz /= np.sum(boltz)

        return pd.DataFrame(
            {
                "x": d.x.mean(),
                "y": d.y.mean(),
                "z": d.z.mean(),
                "electronic": (d.electronic * boltz).sum(),
                "electrostatic": (d.electrostatic * boltz).sum(),
                "lumo": (d.lumo * boltz).sum(),
            },
            index=["_"],
        )

    # Concatenate all conformer grids and apply weighting on each coarse grid point
    grids_all = pd.concat(grids, ignore_index=True)
    wgrids = (
        grids_all.groupby(["x", "y", "z"], as_index=False)
        .apply(_total_keepnoindex)
        .reset_index(drop=True)
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
        - "InChIKey": used to locate the molecule directory under ~/molecules/<InChIKey>
        - "temperature": temperature [K] for this molecule.

    Returns
    -------
    pandas.Series
        The Series returned by :func:`calc_grid`, i.e. aggregated grid descriptors for
        this molecule. The index is a set of feature names; the single row corresponds
        to one molecule.
    """
    home = Path.home()
    target_dir = home / "molecules" / row["InChIKey"]
    return calc_grid(str(target_dir), row["temperature"], folded=1)


def calc_grid_(path: str) -> None:
    """Batch-process grid features for all molecules listed in an Excel file.

    This function:
    1. Reads molecular data from an Excel file.
    2. For each row (molecule), locates the corresponding directory under
       `~/molecules/<InChIKey>` and computes grid descriptors via :func:`calc_grid`.
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

    # Multiprocessing over rows
    with Pool(24) as pool:
        results = pool.map(process_row, [row for _, row in df.iterrows()])

    features = pd.DataFrame(results)
    df_out = pd.concat([df, features], axis=1).fillna(0)

    # Save as pickle and CSV
    pkl_path = path.replace(".xlsx", ".pkl")
    csv_path = path.replace(".xlsx", "feat.csv")

    df_out.to_pickle(pkl_path)
    df_out.to_csv(csv_path, index=False)


if __name__ == "__main__":
    calc_grid_("data/data.xlsx")
