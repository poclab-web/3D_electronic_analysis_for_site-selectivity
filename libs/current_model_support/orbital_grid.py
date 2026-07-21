"""Stream molecular-orbital cubes into the model's folded spatial grid.

Gaussian cube coordinates are interpreted in bohr.  Each voxel is assigned to a
2-bohr integer bin by rounding away from zero, after which the molecular-frame
``y`` coordinate is folded to ``abs(y)``.  The returned features are sums of
the squared orbital amplitude at requested ``(x, y, z)`` bin
indices; they are not normalized probability integrals because no voxel-volume
factor is applied.

The public factory accepts an ``(n_bins, 3)`` integer coordinate array and
returns a reader for Gaussian/``cubegen`` MO cube files.  Cube atom and grid
coordinates follow the cube-file unit convention (normally bohr for Gaussian
output), while requested coordinates are dimensionless indices of 2-bohr bins.
"""
from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np


GRID_STEP_BOHR = 2.0


def make_lookup(coordinates: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a dense lookup from integer grid coordinates to feature columns.

    Parameters
    ----------
    coordinates
        ``(n_bins, 3)`` integer array of folded 2-bohr bin indices in ``x, y,
        z`` order.  Coordinates should be unique.

    Returns
    -------
    lookup, minimum, maximum
        ``lookup`` maps coordinates within the inclusive bounding box to their
        row in ``coordinates`` (``-1`` denotes an unrequested bin); ``minimum``
        and ``maximum`` are the three-element bounds in bin-index units.
    """
    minimum = np.min(coordinates, axis=0)
    maximum = np.max(coordinates, axis=0)
    lookup = np.full(tuple(maximum - minimum + 1), -1, dtype=np.int32)
    shifted = coordinates - minimum
    lookup[shifted[:, 0], shifted[:, 1], shifted[:, 2]] = np.arange(len(coordinates))
    return lookup, minimum, maximum


def cube_reader(target_coordinates: np.ndarray):
    """Create a streaming cube-to-feature reader for a fixed target grid.

    Parameters
    ----------
    target_coordinates
        ``(n_bins, 3)`` integer array in ``x, abs(y), z`` order.  Each integer
        denotes a 2-bohr bin boundary under the project's away-from-zero rule.

    Returns
    -------
    callable
        A function accepting a :class:`~pathlib.Path` to a Gaussian MO cube
        (and an ignored compatibility ``bounds`` argument) and returning an
        ``(n_bins,)`` float array of summed squared orbital amplitudes.
    """
    lookup, minimum, maximum = make_lookup(target_coordinates)

    def folded_bins_from_cube_stream(path: Path, bounds=None) -> np.ndarray:
        """Read one cube sequentially and accumulate squared amplitudes."""
        del bounds
        with path.open("rb") as handle:
            handle.readline()
            handle.readline()
            atom_line = handle.readline().split()
            atom_n_signed = int(atom_line[0])
            atom_n = abs(atom_n_signed)
            origin = np.asarray(atom_line[1:4], dtype=float)
            sizes: list[int] = []
            axes: list[list[float]] = []
            for _ in range(3):
                line = handle.readline().split()
                sizes.append(int(line[0]))
                axes.append([float(value) for value in line[1:4]])
            for _ in range(atom_n):
                handle.readline()
            if atom_n_signed < 0:
                handle.readline()

            cube_coordinates = (
                np.indices(np.asarray(sizes), dtype=float).reshape(3, -1).T @ np.asarray(axes)
                + origin
            )
            scaled = cube_coordinates / GRID_STEP_BOHR
            binned = np.where(scaled > 0, np.ceil(scaled), np.floor(scaled)).astype(np.int16)
            binned[:, 1] = np.abs(binned[:, 1])
            inside = np.all((binned >= minimum) & (binned <= maximum), axis=1)
            linear = np.full(len(cube_coordinates), -1, dtype=np.int32)
            shifted = binned[inside] - minimum
            linear[inside] = lookup[shifted[:, 0], shifted[:, 1], shifted[:, 2]]

            result = np.zeros(len(target_coordinates), dtype=float)
            offset = 0
            while lines := list(itertools.islice(handle, 8192)):
                values = np.fromstring(b" ".join(lines), dtype=float, sep=" ")
                local = linear[offset : offset + values.size]
                keep = local >= 0
                if np.any(keep):
                    result += np.bincount(
                        local[keep], weights=values[keep] ** 2, minlength=len(target_coordinates)
                    )
                offset += values.size
            if offset != len(linear):
                raise ValueError(f"Cube size mismatch: {offset} != {len(linear)}")
            return result

    return folded_bins_from_cube_stream
