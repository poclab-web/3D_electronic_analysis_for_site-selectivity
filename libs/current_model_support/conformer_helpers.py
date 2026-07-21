"""Identify carbonyl atoms and assign thermodynamic conformer weights.

The helpers bridge the tabular substrate metadata and Gaussian files used by
the adopted projected-orbital descriptor.  Input rows are expected to contain
``InChIKey`` and ``SMILES`` and may contain ``temperature`` in kelvin. Gaussian
``opt<id>.log`` and ``sp<id>.chk`` files are read from
``MOLECULE_ROOT/<InChIKey>``. Atom indices returned by this module are 1-based,
matching Gaussian formatted-checkpoint conventions rather than RDKit's
0-based indices. Energies are in hartree and Boltzmann weights are unitless.
"""
from __future__ import annotations

import re
from pathlib import Path

import cclib
import numpy as np
import pandas as pd
from rdkit import Chem


MOLECULE_ROOT = Path.home() / "molecules"
BOLTZMANN_KT_AU = 3.1668114e-6
OPT_LOG_NAME_RE = re.compile(r"opt(\d+)\.log")


def discover_conformer_logs(directory: str | Path) -> list[Path]:
    """Return strict ``opt<digits>.log`` files in conformer-ID order.

    Files such as ``opt12_optimization_partial.log`` are deliberately excluded:
    they are diagnostic intermediates, not independent conformers.
    A missing directory produces an empty list so callers can provide their own
    context-specific error message.
    """
    molecule_dir = Path(directory)
    if not molecule_dir.is_dir():
        return []
    logs = [
        path
        for path in molecule_dir.glob("opt*.log")
        if path.is_file() and OPT_LOG_NAME_RE.fullmatch(path.name)
    ]
    return sorted(logs, key=conformer_id_from_path)


def carbonyl_pairs(smiles: str) -> tuple[list[tuple[int, int]], list[int], str]:
    """Return carbonyl C/O pairs and isotope-labelled carbons for a SMILES.

    Hydrogens are added before atom indices are reported so that the resulting
    1-based indices follow the atom order written to Gaussian by ``calc_mol``.
    The second return value lists 1-based carbon-13 atom indices, which can
    disambiguate the reactive carbonyl of a diketone. The final string is empty
    when at least one C=O pair exists and ``"no_carbonyl"`` otherwise.
    """
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f"RDKit could not parse SMILES: {smiles}")
    molecule = Chem.AddHs(molecule)
    isotope_carbons = [
        atom.GetIdx() + 1
        for atom in molecule.GetAtoms()
        if atom.GetAtomicNum() == 6 and atom.GetIsotope() == 13
    ]
    pairs: list[tuple[int, int]] = []
    for bond in molecule.GetBonds():
        if bond.GetBondType() != Chem.BondType.DOUBLE:
            continue
        first, second = bond.GetBeginAtom(), bond.GetEndAtom()
        if {first.GetAtomicNum(), second.GetAtomicNum()} != {6, 8}:
            continue
        carbon = first if first.GetAtomicNum() == 6 else second
        oxygen = second if first.GetAtomicNum() == 6 else first
        pairs.append((carbon.GetIdx() + 1, oxygen.GetIdx() + 1))
    return pairs, isotope_carbons, "" if pairs else "no_carbonyl"


def target_carbonyl_pair(smiles: str) -> tuple[int, int, str]:
    """Choose the reactive 1-based carbonyl C/O indices for one structure.

    A unique C=O is selected directly. For multiple C=O groups, a carbon-13
    labelled carbonyl is preferred; unresolved ties fall back to the first
    RDKit bond-order match and are explicitly described by the returned rule.
    A molecule without a carbonyl raises :class:`ValueError`.
    """
    pairs, isotope_carbons, note = carbonyl_pairs(smiles)
    if len(pairs) == 1:
        return *pairs[0], "single_carbonyl"
    isotope_pairs = [pair for pair in pairs if pair[0] in isotope_carbons]
    if isotope_pairs:
        rule = "isotope_carbonyl" if len(isotope_pairs) == 1 else "ambiguous_isotope_carbonyl_first"
        return *isotope_pairs[0], rule
    if pairs:
        return *pairs[0], "ambiguous_first_carbonyl"
    raise ValueError(note or f"No carbonyl pair found in {smiles}")


def conformer_id_from_path(path: Path) -> int:
    """Extract the integer conformer ID from a Gaussian ``opt<id>.log`` path."""
    match = OPT_LOG_NAME_RE.fullmatch(path.name)
    if match is None:
        raise ValueError(f"Could not parse conformer ID from {path}")
    return int(match.group(1))


def conformer_gibbs(log_path: Path, temperature: float) -> float:
    """Return ``H - T*S`` for one Gaussian optimization log in hartree.

    ``temperature`` is expressed in kelvin. ``cclib`` supplies enthalpy in
    hartree and entropy in hartree per kelvin, so the returned value is suitable
    for relative Boltzmann weighting with ``BOLTZMANN_KT_AU``.
    """
    data = cclib.io.ccread(str(log_path))
    if data is None or not hasattr(data, "enthalpy") or not hasattr(data, "entropy"):
        raise ValueError(f"Could not read thermochemistry from {log_path}")
    return float(data.enthalpy - data.entropy * temperature)


def conformer_records(row: pd.Series) -> list[dict[str, object]]:
    """Return Gaussian conformer records with molecule-normalized weights.

    ``row`` must provide ``InChIKey`` and ``SMILES`` and may provide
    ``temperature`` in kelvin (default 298.15 K). Each output mapping records
    the conformer ID, optimization log, single-point checkpoint, relative
    thermochemical population, target carbonyl indices, and diagnostic rule.
    Only conformers having readable thermochemistry participate in the
    Boltzmann normalization; unreadable records remain in the output with zero
    weight and a non-empty ``skip_reason``.
    """
    inchikey = str(row["InChIKey"])
    molecule_dir = MOLECULE_ROOT / inchikey
    logs = discover_conformer_logs(molecule_dir)
    if not logs:
        raise FileNotFoundError(f"No opt<digits>.log files found for {inchikey}")
    temperature = float(row.get("temperature", 298.15))
    carbon, oxygen, target_rule = target_carbonyl_pair(str(row["SMILES"]))
    raw: list[dict[str, object]] = []
    for log_path in logs:
        conf_id = conformer_id_from_path(log_path)
        sp_chk = molecule_dir / f"sp{conf_id}.chk"
        if not sp_chk.exists():
            continue
        try:
            gibbs = conformer_gibbs(log_path, temperature)
            reason = ""
        except Exception as exc:  # noqa: BLE001
            gibbs = np.nan
            reason = f"thermo_failed:{exc}"
        raw.append(
            {
                "conf_id": conf_id,
                "opt_log": str(log_path),
                "sp_chk": str(sp_chk),
                "gibbs_au": gibbs,
                "skip_reason": reason,
            }
        )
    good = [record for record in raw if np.isfinite(float(record["gibbs_au"]))]
    if not good:
        raise RuntimeError(f"No readable conformer thermochemistry for {inchikey}")
    gibbs = np.asarray([float(record["gibbs_au"]) for record in good])
    exponent = np.clip(
        -(gibbs - np.min(gibbs)) / (BOLTZMANN_KT_AU * temperature), -700.0, 0.0
    )
    weights = np.exp(exponent)
    weights /= weights.sum()
    weight_by_conf = {
        int(record["conf_id"]): float(weight) for record, weight in zip(good, weights)
    }
    return [
        {
            **record,
            "boltzmann_weight": weight_by_conf.get(int(record["conf_id"]), 0.0),
            "target_c_index": carbon,
            "target_o_index": oxygen,
            "target_rule": target_rule,
            "temperature": temperature,
        }
        for record in raw
    ]
