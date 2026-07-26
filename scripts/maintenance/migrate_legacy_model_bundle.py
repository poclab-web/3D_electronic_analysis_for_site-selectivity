#!/usr/bin/env python3
"""Convert the legacy current-model pickle into portable frozen inputs.

This maintainer utility is intentionally separate from the reproduction
runner.  It removes thousands of duplicated legacy descriptor columns from
the pickle, writes numeric arrays as a compressed NumPy archive, stores the
seven required metadata columns as CSV, and records provenance as JSON.  It
can also freeze the small atom geometries used only to display contribution
cubes and refresh the SHA-256 input manifest.

The legacy pickle is needed only while performing this one-time migration;
normal model reproduction never reads it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def find_repo_root() -> Path:
    start = Path(__file__).resolve().parent
    for candidate in (start, *start.parents):
        if (
            (candidate / "README.md").is_file()
            and (candidate / "data" / "current_model").is_dir()
        ):
            return candidate
    raise RuntimeError("Repository root could not be located")


ROOT = find_repo_root()
INPUT_DIR = ROOT / "data" / "current_model" / "inputs"
DEFAULT_SOURCE = INPUT_DIR / "model_input_bundle.pkl"
DEFAULT_CUBE_MANIFEST = (
    ROOT
    / "data"
    / "validation"
    / "current_model"
    / "contribution_cubes"
    / "contribution_cube_manifest.csv"
)
BLOCKS = ("electronic", "electrostatic", "orbital")
METADATA_COLUMNS = (
    "entry",
    "name",
    "SMILES",
    "InChIKey",
    "temperature",
    "ΔΔG.expt.",
    "test",
)


def sha256(path: Path) -> str:
    """Return the hexadecimal SHA-256 digest of *path*."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_legacy_bundle(path: Path) -> dict[str, Any]:
    """Load and validate the fields required from the historical pickle."""
    payload = pd.read_pickle(path)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a dictionary in {path}, got {type(payload)!r}.")
    missing = {"meta", "raw_blocks", "coords", "descriptor_version"} - set(payload)
    if missing:
        raise ValueError(f"Legacy bundle is missing keys: {sorted(missing)}")
    metadata = payload["meta"]
    missing_columns = set(METADATA_COLUMNS) - set(metadata.columns)
    if missing_columns:
        raise ValueError(f"Legacy metadata is missing columns: {sorted(missing_columns)}")
    if set(payload["raw_blocks"]) != set(BLOCKS):
        raise ValueError("Legacy raw-block names do not match the adopted model.")
    if set(payload["coords"]) != set(BLOCKS):
        raise ValueError("Legacy coordinate-block names do not match the adopted model.")
    return payload


def write_portable_bundle(payload: dict[str, Any], source: Path) -> None:
    """Write canonical model arrays, metadata, and provenance files."""
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    metadata = payload["meta"].loc[:, METADATA_COLUMNS].copy().reset_index(drop=True)
    arrays: dict[str, np.ndarray] = {}
    for block in BLOCKS:
        raw = np.asarray(payload["raw_blocks"][block], dtype=np.float64)
        coords = np.asarray(payload["coords"][block], dtype=np.int64)
        if raw.ndim != 2 or raw.shape[0] != len(metadata):
            raise ValueError(f"Unexpected {block} raw shape: {raw.shape}")
        if coords.shape != (raw.shape[1], 3):
            raise ValueError(f"Unexpected {block} coordinate shape: {coords.shape}")
        if not np.isfinite(raw).all():
            raise ValueError(f"The {block} block contains non-finite values.")
        arrays[f"raw_{block}"] = raw
        arrays[f"coords_{block}"] = coords

    np.savez_compressed(INPUT_DIR / "model_arrays.npz", **arrays)
    metadata.to_csv(INPUT_DIR / "model_metadata.csv", index=False)
    provenance = {
        "format_version": 1,
        "descriptor_version": str(payload["descriptor_version"]),
        "blocks": list(BLOCKS),
        "metadata_columns": list(METADATA_COLUMNS),
        "row_count": int(len(metadata)),
        "grid_count_per_block": {
            block: int(arrays[f"raw_{block}"].shape[1]) for block in BLOCKS
        },
        "conformer_generation": {
            "software": "RDKit",
            "embedding": "EmbedMultipleConfs with ETKDG defaults",
            "random_seed": 1,
            "force_field": "MMFF94",
            "relative_energy_cutoff_kcal_mol": 5.0,
            "rmsd_cutoff_angstrom": 0.5,
            "maximum_conformers": 5,
        },
        "quantum_chemistry": {
            "reference_program": "Gaussian 16 Revision A.03",
            "optimization_frequency": "B3LYP-D3(BJ)/def2-SVP",
            "single_point": "wB97XD/def2-TZVP",
            "solvation": "SMD(methanol)",
            "symmetry": "nosymm",
            "grid_tools": ["formchk", "cubegen", "Multiwfn"],
            "multiwfn_version": (
                "not recorded for historical descriptor generation; "
                "frozen descriptors are canonical"
            ),
        },
        "legacy_bundle_sha256": sha256(source),
        "legacy_bundle_provenance": payload.get("provenance", {}),
    }
    (INPUT_DIR / "model_provenance.json").write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def read_cube_atoms(path: Path) -> list[list[float | int]]:
    """Read atom records from a Gaussian cube without loading its grid."""
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        handle.readline()
        handle.readline()
        header = handle.readline().split()
        if len(header) < 4:
            raise ValueError(f"Malformed cube header: {path}")
        atom_count = abs(int(header[0]))
        for _ in range(3):
            handle.readline()
        atoms: list[list[float | int]] = []
        for _ in range(atom_count):
            fields = handle.readline().split()
            if len(fields) < 5:
                raise ValueError(f"Malformed atom record: {path}")
            atoms.append(
                [
                    int(float(fields[0])),
                    float(fields[1]),
                    float(fields[2]),
                    float(fields[3]),
                    float(fields[4]),
                ]
            )
    return atoms


def write_display_geometries(manifest_path: Path) -> None:
    """Freeze display-only atom geometries from an existing cube manifest."""
    manifest = pd.read_csv(manifest_path)
    geometries: dict[str, dict[str, Any]] = {}
    for row in manifest.to_dict("records"):
        cube_path = ROOT / str(row["electronic_cube"])
        atoms = read_cube_atoms(cube_path)
        if not atoms:
            raise ValueError(f"No display atoms found for {row['entry']}: {cube_path}")
        geometries[str(row["InChIKey"])] = {
            "entry_at_freeze": str(row["entry"]),
            "dominant_conformer": str(row["dominant_conformer"]),
            "atoms": atoms,
        }
    document = {
        "format_version": 1,
        "units": "bohr",
        "purpose": "Display geometry for regenerated model-contribution cubes only",
        "geometry_count": len(geometries),
        "geometries": geometries,
    }
    (INPUT_DIR / "display_geometries.json").write_text(
        json.dumps(document, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def sanitize_orbital_manifest() -> None:
    """Remove machine-specific archive paths from orbital provenance rows."""
    path = INPUT_DIR / "projected_orbital_manifest.csv"
    frame = pd.read_csv(path)
    if "cache" in frame.columns:
        frame["cache_file"] = frame["cache"].map(lambda value: Path(str(value)).name)
        frame = frame.drop(columns="cache")
    frame.to_csv(path, index=False)


def write_input_manifest() -> None:
    """Write sizes and SHA-256 hashes for every immutable runtime input."""
    excluded = {
        "README.md",
        "input_manifest.csv",
        "model_input_bundle.pkl",
    }
    paths = sorted(
        path
        for path in INPUT_DIR.rglob("*")
        if path.is_file() and path.name not in excluded and not path.name.startswith(".")
    )
    rows = [
        {
            "path": str(path.relative_to(INPUT_DIR)),
            "size_bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in paths
    ]
    pd.DataFrame(rows, columns=("path", "size_bytes", "sha256")).to_csv(
        INPUT_DIR / "input_manifest.csv", index=False
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line options for the one-time migration."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--cube-manifest", type=Path, default=DEFAULT_CUBE_MANIFEST)
    parser.add_argument(
        "--skip-geometries",
        action="store_true",
        help="Do not rebuild the display-geometry cache.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the portable-input migration and print the resulting file sizes."""
    args = parse_args()
    source = args.source.resolve()
    payload = load_legacy_bundle(source)
    write_portable_bundle(payload, source)
    if not args.skip_geometries:
        write_display_geometries(args.cube_manifest.resolve())
    sanitize_orbital_manifest()
    write_input_manifest()
    for name in (
        "model_arrays.npz",
        "model_metadata.csv",
        "model_provenance.json",
        "display_geometries.json",
        "input_manifest.csv",
    ):
        path = INPUT_DIR / name
        if path.exists():
            print(f"{path.relative_to(ROOT)}\t{path.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
