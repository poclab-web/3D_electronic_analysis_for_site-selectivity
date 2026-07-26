"""Load and normalize the experimental workbook used by the accepted model."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from rdkit import Chem


HOLDOUT_ENTRY_BY_NAME = {
    "Benzoylacetonitrile": "H1",
    "Bicyclo[2.2.1]hept-5-en-2-one (exo)": "H2(exo)",
    "Bicyclo[2.2.1]hept-5-en-2-one (endo)": "H2(endo)",
    "1,4-Cyclohexanedione Monoethyleneketal": "H3",
    "Isophorone Oxide (cis)": "H4(cis)",
    "Isophorone Oxide (trans)": "H4(trans)",
    "2-methylcyclohexanone(trans)": "Dxx(trans)",
    "2-methylcyclohexanone(cis)": "Dxx(cis)",
}


def apply_dataset_overrides(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply the manuscript holdout labels to a workbook-derived table."""
    frame = frame.copy()
    names = frame["name"].astype(str)
    for substrate_name, entry in HOLDOUT_ENTRY_BY_NAME.items():
        mask = names == substrate_name
        frame.loc[mask, "entry"] = entry
        frame.loc[mask, "test"] = 1
    return frame


def load_experimental_dataset(
    workbook: str | Path,
    *,
    apply_overrides: bool = True,
) -> pd.DataFrame:
    """Return validated molecular rows from the experimental workbook.

    The workbook has a descriptive first row and column headings on the second
    row. Invalid or missing SMILES entries are excluded. Molecular identity is
    represented by the InChIKey of the explicit-hydrogen RDKit molecule, which
    is the stable key used to align workbook rows with frozen descriptors.
    """
    frame = pd.read_excel(workbook, skiprows=1)
    if apply_overrides:
        frame = apply_dataset_overrides(frame)
    frame = frame.dropna(subset=["SMILES"]).copy()
    frame["mol"] = frame["SMILES"].apply(Chem.MolFromSmiles)
    frame = frame.dropna(subset=["mol", "SMILES"])
    frame["InChIKey"] = frame["mol"].apply(
        lambda molecule: Chem.inchi.MolToInchiKey(Chem.AddHs(molecule))
    )
    return frame
