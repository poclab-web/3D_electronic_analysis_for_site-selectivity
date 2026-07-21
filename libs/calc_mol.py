"""
Workflow utilities for generating and processing quantum-chemical data
for ketone molecules from SMILES.

Main features
-------------
- Conformer generation and MMFF optimization (RDKit)
- Energy and RMSD based conformer pruning
- Coordinate alignment of carbonyl groups using Rodrigues rotation
- Preparation and execution of Gaussian jobs (opt/freq, SP)
- Generation of cube files (density, ESP, frontier orbitals)
- Batch processing of molecules defined in an Excel sheet
"""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
import cclib


def _total_ram_gb() -> int:
    """Return total physical RAM in GiB (best effort)."""
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        phys_pages = os.sysconf("SC_PHYS_PAGES")
        total_bytes = int(page_size) * int(phys_pages)
        return max(1, total_bytes // (1024**3))
    except (AttributeError, ValueError, OSError):
        return 1


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


# Runtime resources for Gaussian/cubegen.
# Defaults:
# - NUM_THREADS: all detected CPU cores
# - MEMORY_GB : half of detected RAM
# Overrides:
# - GAUSSIAN_NUM_THREADS (positive integer)
# - GAUSSIAN_MEMORY_GB  (positive integer)
NUM_THREADS = _positive_int_from_env("GAUSSIAN_NUM_THREADS", os.cpu_count() or 1)
MEMORY_GB = _positive_int_from_env("GAUSSIAN_MEMORY_GB", max(1, _total_ram_gb() // 2))
CONFORMER_NUM_THREADS = _positive_int_from_env(
    "CALC_MOL_CONFORMER_THREADS", os.cpu_count() or 1
)
OUTPUT_ROOT = os.getenv("MOLECULES_ROOT", str(Path.home() / "molecules"))
GAUSSIAN_RUN_COMMAND = os.getenv(
    "GAUSSIAN_RUN_COMMAND",
    "g16",
)
FORMCHK_EXECUTABLE = os.getenv(
    "FORMCHK_EXECUTABLE", shutil.which("formchk") or "formchk"
)
CUBEGEN_EXECUTABLE = os.getenv(
    "CUBEGEN_EXECUTABLE", shutil.which("cubegen") or "cubegen"
)
GAUSSIAN_BACKUP_ROOT = os.getenv("GAUSSIAN_BACKUP_ROOT")


def _normal_termination(path: str) -> bool:
    """Return whether a Gaussian log ends normally."""
    log = Path(path)
    if not log.exists():
        return False
    with log.open("rb") as handle:
        try:
            handle.seek(-8192, os.SEEK_END)
        except OSError:
            handle.seek(0)
        lines = [line.strip() for line in handle.read().splitlines() if line.strip()]
    return bool(lines and lines[-1].startswith(b"Normal termination"))


def normal_termination(path: str | Path) -> bool:
    """Public path-like wrapper for strict Gaussian normal-termination checks."""
    return _normal_termination(str(path))


def _contains_log_marker(path: str, marker: bytes) -> bool:
    """Return whether an existing Gaussian log contains *marker*."""
    log = Path(path)
    return log.exists() and marker in log.read_bytes()


def _copy_if_distinct(source: str | Path, destination: str | Path) -> None:
    """Copy a reusable calculation file while rejecting a self-copy."""
    source_path = Path(source)
    destination_path = Path(destination)
    if not source_path.is_file():
        raise FileNotFoundError(f"Gaussian backup file is missing: {source_path}")
    if source_path.resolve() == destination_path.resolve():
        raise ValueError(f"Gaussian backup and destination are identical: {source_path}")
    shutil.copy2(source_path, destination_path)


def energy_cut(mol: Chem.Mol, res, max_energy: float) -> None:
    """Filter conformers by relative MMFF energy and convergence flag.

    The function:
    1. Takes the MMFF optimization result list `res`, which is assumed to
       be an iterable of (flag, energy) pairs.
    2. Normalizes the energies such that the lowest energy becomes 0 kcal/mol.
    3. Removes conformers from `mol` if:
       - the flag is interpreted as "not converged", or
       - the relative energy is greater than `max_energy`.

    For conformers that are kept, the relative energy is stored in the
    conformer property `"energy"` as a string.

    Parameters
    ----------
    mol : rdkit.Chem.Mol
        Molecule that already has conformers generated and optimized.
        This object is modified in-place.
    res : iterable
        Iterable of tuples (flag, energy) returned by
        `AllChem.MMFFOptimizeMoleculeConfs`.
        The first element is treated as a boolean-like flag
        (`not_converged` in this workflow), the second element is the
        absolute energy.
    max_energy : float
        Maximum allowed relative energy (in the same units as `res`
        energies, typically kcal/mol). Conformers above this threshold
        are removed.

    Returns
    -------
    None
        The molecule is modified in-place; no value is returned.
    """
    res_arr = np.array(res, dtype=float)
    # Normalize energies so that the minimum is set to 0
    res_arr[:, 1] -= res_arr[:, 1].min()

    to_remove = []
    for conf, res_item in zip(mol.GetConformers(), res_arr):
        not_converged, energy = res_item
        if not_converged or energy > max_energy:
            to_remove.append(conf.GetId())
            continue
        conf.SetProp("energy", str(energy))

    # Remove conformers marked for deletion
    for conf_id in to_remove:
        mol.RemoveConformer(conf_id)


def conformer_cut(
    mol: Chem.Mol,
    min_rmse: float,
    max_num_conformer: int,
) -> None:
    """Prune conformers based on RMSD diversity and energy order.

    Conformers are:
    1. Sorted by conformer energy (stored in the `"energy"` property).
    2. Selected in ascending energy order as long as each candidate has
       an RMSD (BestRMS over heavy-atom skeleton without hydrogens)
       larger than `min_rmse` with respect to all already selected
       conformers.
    3. All non-selected conformers are removed.

    Parameters
    ----------
    mol : rdkit.Chem.Mol
        Molecule with conformers. Each conformer is assumed to have an
        `"energy"` property (e.g., set by `energy_cut`). The molecule is
        modified in-place.
    min_rmse : float
        RMSD threshold between conformers. A candidate conformer is
        discarded if its RMSD to any already selected conformer is
        smaller than this value.
    max_num_conformer : int
        Maximum number of conformers to keep.

    Returns
    -------
    None
        The molecule is modified in-place; no value is returned.
    """
    # A hydrogen-stripped copy is used for RMSD evaluation
    mol_no_h = Chem.RemoveHs(mol)

    # Sort conformers by stored energy
    conformer_list = sorted(
        [
            (float(conf.GetProp("energy")), conf.GetId())
            for conf in mol.GetConformers()
        ],
        key=lambda x: x[0],
    )

    selected_ids = []

    for _, conf1_id in conformer_list:
        if len(selected_ids) >= max_num_conformer:
            break

        keep = True
        for conf2_id in selected_ids:
            rmsd = AllChem.GetBestRMS(mol_no_h, mol_no_h, conf1_id, conf2_id)
            if rmsd < min_rmse:
                keep = False
                break

        if keep:
            selected_ids.append(conf1_id)

    # Remove all conformers that are not in the selected set
    all_ids = [conf.GetId() for conf in mol.GetConformers()]
    for conf_id in all_ids:
        if conf_id not in selected_ids:
            mol.RemoveConformer(conf_id)


def Rodrigues_rotation(n: np.ndarray, sin: float, cos: float) -> np.ndarray:
    """Compute a 3×3 Rodrigues rotation matrix for a given axis and angle.

    Rodrigues' rotation formula rotates a vector by angle θ around an
    axis `n` (assumed to be normalized here), using sin(θ) and cos(θ).

    Parameters
    ----------
    n : numpy.ndarray
        A 1D array of length 3 representing the (unit) rotation axis.
    sin : float
        Sine of the rotation angle θ.
    cos : float
        Cosine of the rotation angle θ.

    Returns
    -------
    numpy.ndarray
        A 3×3 rotation matrix R such that v_rot = R @ v.
    """
    return np.array(
        [
            [
                n[0] ** 2 * (1 - cos) + cos,
                n[0] * n[1] * (1 - cos) - n[2] * sin,
                n[0] * n[2] * (1 - cos) + n[1] * sin,
            ],
            [
                n[0] * n[1] * (1 - cos) + n[2] * sin,
                n[1] ** 2 * (1 - cos) + cos,
                n[1] * n[2] * (1 - cos) - n[0] * sin,
            ],
            [
                n[0] * n[2] * (1 - cos) - n[1] * sin,
                n[1] * n[2] * (1 - cos) + n[0] * sin,
                n[2] ** 2 * (1 - cos) + cos,
            ],
        ]
    )


def transform(conf: np.ndarray, carbonyl_atom) -> np.ndarray:
    """Reorient conformer coordinates by aligning a carbonyl group.

    The transformation is defined such that:
    - The carbonyl C=O bond is aligned with the x-axis.
    - The plane defined by the carbonyl group and neighboring atoms is
      aligned with the xz-plane.

    This transformation is used to bring different conformers and
    different molecules into a common frame, which is useful for field
    or grid-based analyses.

    Parameters
    ----------
    conf : numpy.ndarray
        A 2D array of shape (N_atoms, 3) containing Cartesian coordinates
        of a conformer.
    carbonyl_atom : sequence of int
        A sequence (c, o, c1, c2) of four atom indices:
        - c  : carbonyl carbon atom index.
        - o  : carbonyl oxygen atom index.
        - c1 : a carbon atom directly bonded to c.
        - c2 : another carbon atom bonded to c1 (downstream along the chain).

    Returns
    -------
    numpy.ndarray
        Transformed coordinates with the same shape as `conf`.

    Notes
    -----
    - If the rotation angle is numerically close to 0 or π, a fallback
      axis is used to avoid division-by-zero issues.
    """
    c, o, c1, c2 = carbonyl_atom

    # Translate so that the carbonyl carbon is at the origin
    conf = conf - conf[c]

    # Step 1: align C=O with x-axis
    a = conf[o] - conf[c]
    a = a / np.linalg.norm(a)

    target_x = np.array([1.0, 0.0, 0.0])
    cos1 = np.dot(a, target_x)
    cross1 = np.cross(a, target_x)
    sin1 = np.linalg.norm(cross1)

    if sin1 > 1e-6:
        n1 = cross1 / sin1
    else:
        # If already aligned (or nearly), take an arbitrary axis
        n1 = np.array([0.0, 0.0, 1.0])

    # Rotate a neighboring direction for the next alignment step
    b = conf[c2] - conf[c1]
    b_rot = Rodrigues_rotation(n1, sin1, cos1) @ b

    # Project b_rot onto yz-plane
    b_yz = b_rot * np.array([0.0, 1.0, 1.0])
    b_yz = b_yz / np.linalg.norm(b_yz)

    target_y = np.array([0.0, 1.0, 0.0])
    cos2 = np.dot(b_yz, target_y)
    cross2 = np.cross(b_yz, target_y)
    sin2 = np.linalg.norm(cross2)

    if sin2 > 1e-6:
        n2 = cross2 / sin2
    else:
        # If already aligned, use x-axis as a placeholder
        n2 = np.array([1.0, 0.0, 0.0])

    # Apply the two rotations to all coordinates
    conf_rot = Rodrigues_rotation(n1, sin1, cos1) @ conf.T
    conf_rot = Rodrigues_rotation(n2, sin2, cos2) @ conf_rot

    return conf_rot.T


def run_subprocess_remote(gjf: str, path: str = GAUSSIAN_RUN_COMMAND) -> None:
    """Run Gaussian via a shell/SSH command, feeding the gjf file to stdin.

    Parameters
    ----------
    gjf : str
        Path to the Gaussian input file (.gjf) to be submitted.
    path : str
        Shell command used to execute Gaussian.
        Examples:
            'source ~/.bash_profile && g16'
            'ssh user@host "source ~/.bash_profile && g16"'

    Returns
    -------
    None
        The Gaussian job is executed; no value is returned.
    """
    ssh_cmd = path
    workdir = os.path.dirname(gjf)
    log_path = gjf.replace(".gjf", ".log")

    with open(gjf, "r") as gjf_file, open(log_path, "w") as log_file:
        subprocess.run(
            ssh_cmd,
            shell=True,
            check=True,
            text=True,
            input=gjf_file.read(),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            cwd=workdir,
        )


def calc_ket(
    out_path: str,
    smiles: str,
    run_path: str = GAUSSIAN_RUN_COMMAND,
) -> None:
    """Main workflow for a single ketone: conformers → Gaussian → cube files.

    Given a SMILES string, this function:

    1. Creates an output directory `out_path` (if needed). If a marker file
       ``done`` exists inside, it returns immediately (job already finished).
    2. Builds an RDKit molecule from the SMILES and adds hydrogens.
    3. Identifies the carbonyl substructure for alignment:
       pattern ``[#6](=[#8])([#6])([#6])``.
    4. Generates multiple 3D conformers using ``EmbedMultipleConfs`` and
       optimizes them with MMFF (``MMFFOptimizeMoleculeConfs``).
    5. Filters conformers by energy (``energy_cut``) and RMSD / count
       (``conformer_cut``).
    6. For each remaining conformer:
       - Either copy precomputed Gaussian optimization logs/chk/fchk/cubes
         from the optional ``GAUSSIAN_BACKUP_ROOT``, or
         generate a new optimization gjf and submit it with
         ``run_subprocess_remote``.
       - Parse the optimization log with cclib to obtain final coordinates
         and atomic numbers.
       - If a single-point (SP) calculation already exists in the backup,
         copy its files; otherwise:
           * Align the geometry using ``transform``.
           * Write a SP gjf (WB97XD/def2TZVP, SMD-methanol) and submit it.
           * Run ``formchk`` and ``cubegen`` to produce density and ESP
             cube files.
       - Unless ``CALC_MOL_SKIP_FRONTIER_CUBES=1``, generate additional cube
         files for HOMO, LUMO, LUMO+1, LUMO+2 using ``cubegen``.

    Finally, a marker file called ``done`` is written inside `out_path`.

    Parameters
    ----------
    out_path : str
        Directory path to store all Gaussian input/output and cube files
        for this specific molecule (usually keyed by InChIKey).
    smiles : str
        SMILES representation of the ketone molecule.
    run_path : str
        Shell or SSH command used to execute Gaussian, e.g.
        'source ~/.bash_profile && g16' or a remote SSH command.

    Returns
    -------
    None
        All results are written to `out_path`; no value is returned.
    """
    start = time.time()
    os.makedirs(out_path, exist_ok=True)

    # Skip if this molecule was already processed
    done_flag = os.path.join(out_path, "done")
    if os.path.isfile(done_flag):
        return

    # Build and prepare the molecule
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit could not parse SMILES: {smiles}")
    mol = Chem.AddHs(mol)

    # Identify the carbonyl substructure [C(=O)(C)(C)]
    substruct_pattern = Chem.MolFromSmarts("[#6](=[#8])([#6])([#6])")
    substruct = mol.GetSubstructMatch(substruct_pattern)
    if not substruct:
        raise ValueError(
            "The molecule does not contain the ketone alignment pattern "
            "[#6](=[#8])([#6])([#6])."
        )

    # Reorder substituent atoms so that downstream alignment is deterministic
    if int(substruct[3]) < int(substruct[2]):
        substruct = (substruct[0], substruct[1], substruct[3], substruct[2])

    # Generate and optimize conformers
    AllChem.EmbedMultipleConfs(
        mol,
        numConfs=mol.GetNumAtoms() ** 2,
        randomSeed=1,
        numThreads=CONFORMER_NUM_THREADS,
    )
    if mol.GetNumConformers() == 0:
        raise RuntimeError(f"RDKit failed to embed any conformers for {smiles}")
    res = AllChem.MMFFOptimizeMoleculeConfs(
        mol,
        maxIters=1000,
        numThreads=CONFORMER_NUM_THREADS,
    )

    energy_cut(mol, res, 5)
    print("After energy_cut:", time.time() - start, "sec", len(mol.GetConformers()))
    if mol.GetNumConformers() == 0:
        raise RuntimeError("All conformers failed MMFF convergence or the energy cutoff.")

    conformer_cut(mol, min_rmse=0.5, max_num_conformer=5)
    print("After conformer_cut:", time.time() - start, "sec", len(mol.GetConformers()))
    if mol.GetNumConformers() == 0:
        raise RuntimeError("Conformer pruning removed every conformer.")

    # Process each remaining conformer
    for conf in mol.GetConformers():
        conf_id = conf.GetId()

        # An optional, explicit backup root may contain reusable calculations.
        backup_path = (
            str(Path(GAUSSIAN_BACKUP_ROOT) / Path(out_path).name)
            if GAUSSIAN_BACKUP_ROOT
            else None
        )
        local_path = out_path

        # ---------- Optimization step (opt) ----------
        opt_log_name = f"opt{conf_id}.log"
        opt_log_backup = (
            os.path.join(backup_path, opt_log_name) if backup_path else None
        )
        opt_log_local = os.path.join(local_path, opt_log_name)

        if opt_log_backup and os.path.isfile(opt_log_backup):
            _copy_if_distinct(opt_log_backup, opt_log_local)
            print(f"{opt_log_name} copied to {local_path}.")
        elif _normal_termination(opt_log_local):
            print(f"{opt_log_name} reused from {local_path}.")
        elif (
            _contains_log_marker(opt_log_local, b"Optimization completed.")
            and os.path.isfile(os.path.join(out_path, f"opt{conf_id}.chk"))
        ):
            # Resume only the frequency part when an opt+freq job was
            # interrupted after reaching the stationary point.
            restart_gjf = os.path.join(out_path, f"opt{conf_id}_freq_restart.gjf")
            restart_log = restart_gjf.replace(".gjf", ".log")
            with open(restart_gjf, "w") as f:
                print(
                    f"%nprocshared={NUM_THREADS}\n"
                    f"%mem={MEMORY_GB}GB\n"
                    f"%chk={out_path}/opt{conf_id}.chk\n"
                    "# freq B3LYP/def2SVP EmpiricalDispersion=GD3BJ "
                    "geom=allcheck guess=read\n",
                    file=f,
                )
            run_subprocess_remote(restart_gjf, run_path)
            if not _normal_termination(restart_log):
                raise RuntimeError(f"Frequency restart did not finish normally: {restart_log}")
            archived_opt = os.path.join(out_path, f"opt{conf_id}_optimization_partial.log")
            shutil.move(opt_log_local, archived_opt)
            shutil.move(restart_log, opt_log_local)
        else:
            opt_gjf = os.path.join(out_path, f"opt{conf_id}.gjf")
            xyz_block = Chem.rdmolfiles.MolToXYZBlock(mol, confId=conf_id)
            xyz_lines = "\n".join(xyz_block.split("\n")[2:])  # skip header lines

            # Write optimization/frequency input from the RDKit conformer geometry
            with open(opt_gjf, "w") as f:
                gjf_input = (
                    f"%nprocshared={NUM_THREADS}\n"
                    f"%mem={MEMORY_GB}GB\n"
                    f"%chk={out_path}/opt{conf_id}.chk\n"
                    "# freq opt B3LYP/def2SVP EmpiricalDispersion=GD3BJ "
                    "optcyc=300\n\n"
                    "good luck!\n\n"
                    f"{Chem.GetFormalCharge(mol)} 1\n"
                    f"{xyz_lines}"
                )
                print(gjf_input, file=f)

            run_subprocess_remote(opt_gjf, run_path)

        if not _normal_termination(opt_log_local):
            raise RuntimeError(
                f"Optimization/frequency calculation did not finish normally: {opt_log_local}"
            )

        # Parse optimized geometry from the optimization log
        opt_log_for_cclib = os.path.join(out_path, f"opt{conf_id}.log")
        opt_data = cclib.io.ccread(opt_log_for_cclib)
        if opt_data is None or not hasattr(opt_data, "atomcoords"):
            raise ValueError(f"Failed to parse optimized geometry: {opt_log_for_cclib}")

        # ---------- Single-point step (sp) ----------
        sp_log_name = f"sp{conf_id}.log"
        sp_log_backup = (
            os.path.join(backup_path, sp_log_name) if backup_path else None
        )
        sp_log_local = os.path.join(local_path, sp_log_name)

        chk_path = f"{out_path}/sp{conf_id}.chk"
        fchk_path = f"{out_path}/sp{conf_id}.fchk"

        if sp_log_backup and os.path.isfile(sp_log_backup):
            # Copy SP log, chk, fchk, and cube files from backup
            _copy_if_distinct(sp_log_backup, sp_log_local)
            _copy_if_distinct(
                sp_log_backup.replace(".log", ".chk"),
                sp_log_local.replace(".log", ".chk"),
            )
            _copy_if_distinct(
                sp_log_backup.replace(".log", ".fchk"),
                sp_log_local.replace(".log", ".fchk"),
            )
            _copy_if_distinct(
                sp_log_backup.replace("/sp", "/Dt").replace(".log", ".cube"),
                sp_log_local.replace("/sp", "/Dt").replace(".log", ".cube"),
            )
            _copy_if_distinct(
                sp_log_backup.replace("/sp", "/ESP").replace(".log", ".cube"),
                sp_log_local.replace("/sp", "/ESP").replace(".log", ".cube"),
            )
            print(f"{sp_log_name} copied to {local_path}.")
        elif _normal_termination(sp_log_local) and os.path.isfile(chk_path):
            print(f"{sp_log_name} reused from {local_path}.")
        else:
            # Align the final optimized coordinates to a common molecular frame
            # before single-point evaluation and cube generation.
            coords = opt_data.atomcoords[-1]
            coords = transform(coords, substruct)
            atomic_numbers = opt_data.atomnos

            sp_gjf = os.path.join(out_path, f"sp{conf_id}.gjf")

            coord_lines = ""
            for atomic_num, coord in zip(atomic_numbers, coords):
                coord_lines += (
                    f"{atomic_num} {coord[0]: .6f} {coord[1]: .6f} {coord[2]: .6f}\n"
                )

            # Write single-point input in the aligned coordinate system.
            with open(sp_gjf, "w") as f:
                gjf_input = (
                    f"%nprocshared={NUM_THREADS}\n"
                    f"%mem={MEMORY_GB}GB\n"
                    f"%chk={out_path}/sp{conf_id}.chk\n"
                    "# wb97xd/def2tzvp scrf=(smd,solvent=methanol) nosymm\n\n"
                    "good luck!\n\n"
                    f"{Chem.GetFormalCharge(mol)} 1\n"
                    f"{coord_lines}"
                )
                print(gjf_input, file=f)

            # Run the single-point calculation.
            run_subprocess_remote(sp_gjf, run_path)

        if not _normal_termination(sp_log_local) or not os.path.isfile(chk_path):
            raise RuntimeError(f"Single-point calculation did not finish normally: {sp_log_local}")

        # Use argument lists so paths containing spaces remain intact.
        if not os.path.isfile(fchk_path):
            subprocess.run(
                [FORMCHK_EXECUTABLE, chk_path, fchk_path],
                check=True,
            )

        dt_cube = f"{out_path}/Dt{conf_id}.cube"
        if not os.path.isfile(dt_cube):
            subprocess.run(
                [
                    CUBEGEN_EXECUTABLE,
                    str(NUM_THREADS),
                    "Density=SCF",
                    fchk_path,
                    dt_cube,
                    "-3",
                    "h",
                ],
                check=True,
            )

        esp_cube = f"{out_path}/ESP{conf_id}.cube"
        if not os.path.isfile(esp_cube):
            subprocess.run(
                [
                    CUBEGEN_EXECUTABLE,
                    str(NUM_THREADS),
                    "Potential=SCF",
                    fchk_path,
                    esp_cube,
                    "-3",
                    "h",
                ],
                check=True,
            )

        # ---------- Frontier orbitals (HOMO/LUMO etc.) ----------
        # External prediction jobs can skip these large cubes because the
        # projected-orbital descriptor is generated directly from the fchk.
        if os.getenv("CALC_MOL_SKIP_FRONTIER_CUBES", "0") == "1":
            continue

        # Read the MO indices from the single-point calculation, since the
        # orbital ordering can differ from the optimization/frequency job.
        sp_log_for_cclib = os.path.join(out_path, f"sp{conf_id}.log")
        sp_data = cclib.io.ccread(sp_log_for_cclib)
        if sp_data is None or not hasattr(sp_data, "homos"):
            raise ValueError(f"Failed to read HOMO/LUMO indices from {sp_log_for_cclib}")

        homo_index = sp_data.homos[0]

        orbital_cubes = (
            (f"{out_path}/HOMO{conf_id}.cube", homo_index + 1),
            (f"{out_path}/LUMO{conf_id}.cube", homo_index + 2),
            (f"{out_path}/LUMO+1_{conf_id}.cube", homo_index + 3),
            (f"{out_path}/LUMO+2_{conf_id}.cube", homo_index + 4),
        )
        for cube_path, mo_number in orbital_cubes:
            if not os.path.isfile(cube_path):
                subprocess.run(
                    [
                        CUBEGEN_EXECUTABLE,
                        str(NUM_THREADS),
                        f"MO={mo_number}",
                        fchk_path,
                        cube_path,
                        "-3",
                        "h",
                    ],
                    check=True,
                )

    # Mark completion
    with open(done_flag, "w") as f:
        f.write("")


def main() -> None:
    """Entry point for batch processing defined in data/data.xlsx.

    This function:

    1. Uses the module-level Gaussian run command (`GAUSSIAN_RUN_COMMAND`).
    2. Sets the root output directory from ``OUTPUT_ROOT``.
    3. Reads an Excel file ``data/data.xlsx`` which must contain at least
       the columns ``InChIKey`` and ``SMILES``.
    4. For each row, calls ``calc_ket`` with an output directory named by
       the InChIKey.

    Returns
    -------
    None
    """
    out_root = Path(OUTPUT_ROOT)

    df = pd.read_excel("data/data.xlsx")

    # Iterate over all entries and launch the workflow per molecule
    for _, row in df[["InChIKey", "SMILES"]].iterrows():
        inchi_key = row["InChIKey"]
        smiles = row["SMILES"]
        out_path = out_root / inchi_key
        calc_ket(str(out_path), smiles, GAUSSIAN_RUN_COMMAND)


if __name__ == "__main__":
    main()
