#!/usr/bin/env python3
"""Promote a reviewed workbook refresh into the frozen model inputs.

This utility is deliberately conservative.  It permits the active workbook to
remove only non-training molecules already present in the frozen package.  It
refuses changed training identities or responses, new molecules without frozen
descriptors, and any mismatch between the adopted orbital block and its cache.
All row-aligned files and the SHA-256 manifest are then updated together.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


def find_repo_root() -> Path:
    """Locate the repository independently of the caller's working directory."""
    start = Path(__file__).resolve().parent
    for candidate in (start, *start.parents):
        if (
            (candidate / "README.md").is_file()
            and (candidate / "data" / "current_model" / "inputs").is_dir()
        ):
            return candidate
    raise RuntimeError("Repository root could not be located")


ROOT = find_repo_root()
LIBS = ROOT / "libs"
SCRIPTS = ROOT / "scripts"
if str(LIBS) not in sys.path:
    sys.path.insert(0, str(LIBS))
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import current_model  # noqa: E402
from maintenance.migrate_legacy_model_bundle import write_input_manifest  # noqa: E402


INPUT_DIR = ROOT / "data" / "current_model" / "inputs"
TARGET = current_model.TARGET
BLOCKS = current_model.BLOCKS


def parse_args() -> argparse.Namespace:
    """Parse the explicit safety limits for this reviewed promotion."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--expected-removed",
        type=int,
        required=True,
        help="Required number of removed non-training molecules.",
    )
    parser.add_argument(
        "--review-date",
        required=True,
        help="Review date recorded in provenance (YYYY-MM-DD).",
    )
    return parser.parse_args()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def main() -> None:
    """Validate and atomically promote the workbook-aligned 161-row package."""
    args = parse_args()
    frozen = current_model.ensure_frozen_inputs()
    old_meta = frozen["meta"].copy().reset_index(drop=True)
    old_train = pd.read_csv(current_model.TRAIN_ROWS_PATH)
    refreshed, refreshed_train, audit = current_model.refresh_inputs_from_excel(
        frozen, old_train
    )
    new_meta = refreshed["meta"].copy().reset_index(drop=True)

    _require(
        int(audit["removed_nontraining_n"]) == args.expected_removed,
        "The workbook removal count does not match --expected-removed.",
    )
    _require(int(audit["changed_metadata_n"]) == 0, "Metadata or response values changed.")
    _require(int(audit["new_external_rows_n"]) == 0, "Workbook contains new descriptor-free rows.")
    _require(len(refreshed_train) == len(old_train), "Training row count changed.")

    old_train_indices = old_train["row_index"].to_numpy(dtype=int)
    old_train_keys = old_meta.loc[old_train_indices, "InChIKey"].astype(str).to_numpy()
    new_train_keys = new_meta.loc[refreshed_train, "InChIKey"].astype(str).to_numpy()
    _require(np.array_equal(old_train_keys, new_train_keys), "Training identities changed.")
    old_train_ddg = old_meta.loc[old_train_indices, TARGET].to_numpy(dtype=float)
    new_train_ddg = new_meta.loc[refreshed_train, TARGET].to_numpy(dtype=float)
    _require(
        np.allclose(old_train_ddg, new_train_ddg, rtol=0.0, atol=1.0e-12),
        "Training experimental responses changed.",
    )

    old_index_by_key = {
        str(key): index for index, key in enumerate(old_meta["InChIKey"].astype(str))
    }
    retained_old_indices = np.asarray(
        [old_index_by_key[str(key)] for key in new_meta["InChIKey"].astype(str)],
        dtype=int,
    )

    orbital_cache_path = current_model.ORBITAL_CACHE_PATH
    with np.load(orbital_cache_path, allow_pickle=False) as archive:
        _require(
            set(archive.files) == {"orbital", "coords", "weight_sums"},
            "Unexpected projected-orbital cache schema.",
        )
        cache_orbital = np.asarray(archive["orbital"], dtype=np.float64)
        cache_coords = np.asarray(archive["coords"], dtype=np.int64)
        cache_weight_sums = np.asarray(archive["weight_sums"], dtype=np.float64)
    _require(cache_orbital.shape[0] == len(old_meta), "Orbital cache rows are misaligned.")
    _require(cache_weight_sums.shape == (len(old_meta),), "Orbital weight rows are misaligned.")
    promoted_orbital = cache_orbital[retained_old_indices]
    _require(
        np.array_equal(promoted_orbital, refreshed["raw_blocks"]["orbital"]),
        "The orbital cache does not equal the adopted orbital descriptor block.",
    )

    orbital_manifest = pd.read_csv(current_model.ORBITAL_MANIFEST_PATH)
    new_index_by_key = {
        str(key): index for index, key in enumerate(new_meta["InChIKey"].astype(str))
    }
    orbital_manifest = orbital_manifest.loc[
        orbital_manifest["InChIKey"].astype(str).isin(new_index_by_key)
    ].copy()
    orbital_manifest["row_index"] = orbital_manifest["InChIKey"].astype(str).map(
        new_index_by_key
    )
    meta_by_key = new_meta.set_index(new_meta["InChIKey"].astype(str))
    orbital_manifest["entry"] = orbital_manifest["InChIKey"].astype(str).map(
        meta_by_key["entry"]
    )
    orbital_manifest["name"] = orbital_manifest["InChIKey"].astype(str).map(
        meta_by_key["name"]
    )
    orbital_manifest = orbital_manifest.sort_values(
        ["row_index", "conf_id"], kind="stable"
    ).reset_index(drop=True)
    _require(
        set(orbital_manifest["InChIKey"].astype(str)) == set(new_index_by_key),
        "At least one promoted molecule lacks orbital provenance rows.",
    )

    promoted_train = pd.DataFrame(
        {
            "row_index": refreshed_train,
            "entry": new_meta.loc[refreshed_train, "entry"].to_numpy(),
            "name": new_meta.loc[refreshed_train, "name"].to_numpy(),
            "InChIKey": new_train_keys,
            "ddg": new_train_ddg,
        }
    )

    provenance = json.loads(current_model.MODEL_PROVENANCE_PATH.read_text(encoding="utf-8"))
    provenance["row_count"] = int(len(new_meta))
    removed = pd.read_csv(current_model.AUDIT_DIR / "excel_refresh_removed_rows.csv")
    provenance["workbook_refresh"] = {
        "review_date": args.review_date,
        "source": str(current_model.ACTIVE_EXCEL.relative_to(ROOT)),
        "identity_join": "InChIKey",
        "removed_nontraining_count": int(len(removed)),
        "removed_nontraining_inchikeys": removed["InChIKey"].astype(str).tolist(),
        "training_identity_or_response_changes": 0,
    }

    with tempfile.TemporaryDirectory(prefix="promote-refresh-", dir=INPUT_DIR.parent) as tmp:
        staging = Path(tmp)
        arrays: dict[str, np.ndarray] = {}
        for block in BLOCKS:
            arrays[f"raw_{block}"] = np.asarray(
                refreshed["raw_blocks"][block], dtype=np.float64
            )
            arrays[f"coords_{block}"] = np.asarray(
                refreshed["coords"][block], dtype=np.int64
            )
        np.savez_compressed(staging / "model_arrays.npz", **arrays)
        new_meta.to_csv(staging / "model_metadata.csv", index=False)
        (staging / "model_provenance.json").write_text(
            json.dumps(provenance, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        np.savez_compressed(
            staging / "projected_orbital_fullgrid_2bohr.npz",
            orbital=promoted_orbital,
            coords=cache_coords,
            weight_sums=cache_weight_sums[retained_old_indices],
        )
        orbital_manifest.to_csv(staging / "projected_orbital_manifest.csv", index=False)
        promoted_train.to_csv(staging / "train_rows.csv", index=False)

        for name in (
            "model_arrays.npz",
            "model_metadata.csv",
            "model_provenance.json",
            "projected_orbital_fullgrid_2bohr.npz",
            "projected_orbital_manifest.csv",
            "train_rows.csv",
        ):
            (staging / name).replace(INPUT_DIR / name)

    write_input_manifest()
    current_model.ensure_frozen_inputs()
    print(json.dumps(audit, indent=2, sort_keys=True))
    print(f"Promoted {len(new_meta)} rows with {len(promoted_train)} training observations.")


if __name__ == "__main__":
    main()
