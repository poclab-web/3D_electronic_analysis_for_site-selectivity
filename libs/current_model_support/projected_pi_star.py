"""Construct carbonyl-centered projected pi-star orbitals for reference cases.

The prototype identifies the target C=O from a conformer manifest, builds
local carbon and oxygen p orbitals normal to the carbonyl plane, diagonalizes
their two-dimensional Fock subspace, and projects the higher-energy pi-star
seed onto the canonical virtual-MO space.  The normalized projected vector is
written into the LUMO slot of a copied formatted checkpoint so Gaussian
``cubegen`` can produce a visualization cube without changing the underlying
wavefunction calculation.

Manifest rows require ``row_index``, ``entry``, ``name``, ``InChIKey``,
``conf_id``, ``status``, ``boltzmann_weight``, ``sp_chk``,
``target_c_index``, and ``target_o_index``; an optional ``cube`` field links an
NBO reference.  Atom indices are one-based.  FCHK Cartesian coordinates are in
bohr, orbital/Fock energies are in hartree, AO/MO coefficients and overlaps are
dimensionless, and carbonyl-plane normals are Cartesian unit vectors.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
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
MULTIWFN = Path(
    os.environ.get(
        "MULTIWFN_EXECUTABLE",
        shutil.which("Multiwfn") or shutil.which("multiwfn") or "Multiwfn",
    )
)
CUBEGEN = Path(
    os.environ.get("CUBEGEN_EXECUTABLE", shutil.which("cubegen") or "cubegen")
)
CUBEGEN_NPROC = os.environ.get("PROJECTED_ORBITAL_CUBEGEN_NPROC", "4")
GRID_SPEC = "-3 h"
BOHR_PER_ANGSTROM = 1.8897259886


@dataclass(frozen=True)
class Job:
    """Immutable input schema for one representative projected-orbital job.

    ``row_index`` is the zero-based dataset row, ``conf_id`` identifies a
    conformer, ``fchk`` points to its formatted checkpoint, and ``target_c`` /
    ``target_o`` are one-based atom indices.  Remaining strings preserve the
    human-readable and chemical identifiers used in output tables.
    """

    label: str
    row_index: int
    entry: str
    name: str
    inchikey: str
    conf_id: int
    fchk: Path
    target_c: int
    target_o: int


def read_manifest(manifest_path: Path) -> list[dict[str, str]]:
    """Read the projected-orbital CSV manifest as string-valued row mappings.

    ``manifest_path`` must name the external full conformer manifest.  The
    required columns documented in the module docstring are validated before
    rows are returned; numeric conversion is deferred to the workflow.
    """
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or ())
        missing = sorted(REQUIRED_MANIFEST_COLUMNS - columns)
        if missing:
            raise ValueError(
                f"Full projected-orbital manifest {manifest_path} is missing "
                f"required columns: {', '.join(missing)}"
            )
        return list(reader)


def select_representatives(manifest_path: Path) -> list[Job]:
    """Select the fixed proof-of-concept molecules and conformers.

    ``manifest_path`` is the external full manifest containing Gaussian paths.
    For entries without an explicitly pinned ``conf_id``, the completed row
    with the greatest dimensionless ``boltzmann_weight`` is chosen.  Returned
    jobs require the ``.fchk`` corresponding to each manifest ``sp_chk`` path.
    """
    rows = read_manifest(manifest_path)

    def pick(label: str, entry: str, *, conf_id: int | None = None) -> Job:
        """Choose one completed manifest conformer and construct its job."""
        candidates = [row for row in rows if row["entry"] == entry and row["status"] == "done"]
        if conf_id is not None:
            candidates = [row for row in candidates if int(float(row["conf_id"])) == conf_id]
        if not candidates:
            raise ValueError(f"No manifest row for {entry} conf={conf_id}")
        row = max(candidates, key=lambda item: float(item["boltzmann_weight"]))
        sp_chk = Path(row["sp_chk"])
        fchk = sp_chk.with_suffix(".fchk")
        if not fchk.exists():
            raise FileNotFoundError(f"Missing fchk for {label}: {fchk}")
        return Job(
            label=label,
            row_index=int(row["row_index"]),
            entry=row["entry"],
            name=row["name"],
            inchikey=row["InChIKey"],
            conf_id=int(float(row["conf_id"])),
            fchk=fchk,
            target_c=int(float(row["target_c_index"])),
            target_o=int(float(row["target_o_index"])),
        )

    return [
        pick("A1_acetophenone", "A1"),
        pick("A4_4nitroacetophenone", "A4"),
        pick("A24_triphenylacetophenone", "A24", conf_id=131),
        pick("D11_trans", "D11(trans)"),
        pick("D11_cis", "D11(cis)"),
        pick("a4_diketone_to_trans20ol", "a4"),
        pick("a24_eq3ol_to_trans20ol", "a24"),
    ]


def run_multiwfn_matrix(fchk: Path, workdir: Path, kind: int, out_name: str) -> Path:
    """Generate or reuse an AO matrix exported by Multiwfn.

    Parameters
    ----------
    fchk
        Gaussian formatted checkpoint containing the AO basis and wavefunction.
    workdir
        Directory for Multiwfn's ``intmat.txt`` and the renamed cache file.
    kind
        ``0`` requests the Fock matrix (hartree); ``1`` requests the
        dimensionless AO overlap matrix.
    out_name
        Cache filename within ``workdir``.
    """
    workdir.mkdir(parents=True, exist_ok=True)
    target = workdir / out_name
    if target.exists() and target.stat().st_size > 1024:
        return target
    if kind == 0:
        script = "6\n7\n0\n2\n1\n-1\nq\n"
    elif kind == 1:
        script = "6\n7\n1\n2\n-1\nq\n"
    else:
        raise ValueError(kind)
    completed = subprocess.run(
        [str(MULTIWFN), str(fchk)],
        input=script,
        text=True,
        cwd=workdir,
        capture_output=True,
        check=False,
    )
    intmat = workdir / "intmat.txt"
    if completed.returncode != 0 or not intmat.exists() or intmat.stat().st_size < 1024:
        raise RuntimeError(
            f"Multiwfn matrix generation failed for {fchk}\n"
            f"returncode={completed.returncode}\nSTDOUT tail:\n{completed.stdout[-3000:]}\nSTDERR:\n{completed.stderr[-2000:]}"
        )
    intmat.replace(target)
    return target


def parse_multiwfn_matrix(path: Path, n: int) -> np.ndarray:
    """Parse a blocked, one-based Multiwfn matrix into a symmetric array.

    ``n`` is the AO-basis dimension.  The returned shape is ``(n, n)`` and the
    numeric units are those of the exported matrix (dimensionless for overlap,
    hartree for Fock); lower/upper triangles are mirrored during parsing.
    """
    matrix = np.zeros((n, n), dtype=float)
    current_cols: list[int] = []
    float_re = re.compile(r"[-+]?\d+(?:\.\d*)?(?:[Ee][-+]?\d+)?")
    with path.open(encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            tokens = stripped.split()
            if all(token.isdigit() for token in tokens):
                current_cols = [int(token) for token in tokens]
                continue
            if not current_cols or not tokens[0].isdigit():
                continue
            row = int(tokens[0])
            values = [float(value) for value in float_re.findall(" ".join(tokens[1:]))]
            for col, value in zip(current_cols, values):
                matrix[row - 1, col - 1] = value
                matrix[col - 1, row - 1] = value
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"Non-finite values in {path}")
    return matrix


def read_fchk_lines(path: Path) -> list[str]:
    """Read a Gaussian formatted checkpoint as newline-free text lines."""
    return path.read_text(encoding="utf-8", errors="ignore").splitlines()


def find_field(lines: list[str], name: str) -> int:
    """Return the zero-based line position of an FCHK field prefix."""
    for index, line in enumerate(lines):
        if line.startswith(name):
            return index
    raise KeyError(name)


def read_scalar(lines: list[str], name: str, cast=float):
    """Read and cast the final token of a scalar FCHK field line.

    Units follow the named FCHK field; for example orbital energies are
    hartree and integer count fields are dimensionless.
    """
    line = lines[find_field(lines, name)]
    return cast(line.split()[-1])


def read_array(lines: list[str], name: str, cast=float) -> tuple[np.ndarray, int, int]:
    """Read one ``N=`` array field from FCHK text.

    Returns ``(values, header_index, end_index)`` where ``end_index`` is the
    first line after the array.  Values use the FCHK field's native units and
    are converted elementwise with ``cast``.
    """
    start = find_field(lines, name)
    match = re.search(r"N=\s*(\d+)", lines[start])
    if match is None:
        raise ValueError(f"{name} is not an array field")
    size = int(match.group(1))
    values: list[object] = []
    end = start + 1
    while len(values) < size and end < len(lines):
        values.extend(cast(token) for token in lines[end].split())
        end += 1
    if len(values) != size:
        raise ValueError(f"Expected {size} values for {name}, got {len(values)}")
    return np.asarray(values), start, end


def write_float_array(values: np.ndarray) -> list[str]:
    """Format a one-dimensional float array in Gaussian FCHK field layout.

    Five values are emitted per line using 16-character exponential fields.
    Input units are preserved textually.
    """
    out: list[str] = []
    for start in range(0, len(values), 5):
        chunk = values[start : start + 5]
        out.append("".join(f"{float(value):16.8E}" for value in chunk))
    return out


def replace_float_array(lines: list[str], name: str, values: np.ndarray) -> None:
    """Replace an existing FCHK float-array payload in mutable ``lines``.

    ``values`` must match the field's declared element count and native units;
    the header line itself is retained.
    """
    _, start, end = read_array(lines, name, float)
    lines[start + 1 : end] = write_float_array(values)


def basis_function_count(shell_type: int) -> int:
    """Map a Gaussian FCHK shell-type code to its AO function count.

    Positive codes denote Cartesian shells and negative codes denote pure
    spherical shells; ``-1`` is the combined ``sp`` shell code.
    """
    return {
        0: 1,
        1: 3,
        -1: 4,
        2: 6,
        -2: 5,
        3: 10,
        -3: 7,
        4: 15,
        -4: 9,
    }[shell_type]


def p_shell_indices(lines: list[str]) -> dict[int, list[tuple[int, int, int]]]:
    """Map one-based atom indices to AO indices for each p-shell triplet.

    The returned ``px, py, pz`` indices are zero-based columns in the Gaussian
    AO coefficient vector.  Pure p shells and the p portion of ``sp`` shells
    are included; the reconstructed total basis size is validated.
    """
    shell_types, _, _ = read_array(lines, "Shell types", int)
    shell_atoms, _, _ = read_array(lines, "Shell to atom map", int)
    by_atom: dict[int, list[tuple[int, int, int]]] = {}
    basis_index = 0
    for shell_type, atom in zip(shell_types.astype(int), shell_atoms.astype(int)):
        if shell_type == 1:
            by_atom.setdefault(int(atom), []).append((basis_index, basis_index + 1, basis_index + 2))
        elif shell_type == -1:
            by_atom.setdefault(int(atom), []).append((basis_index + 1, basis_index + 2, basis_index + 3))
        basis_index += basis_function_count(int(shell_type))
    nbasis = int(read_scalar(lines, "Number of basis functions", int))
    if basis_index != nbasis:
        raise ValueError(f"Basis count mismatch from shells: {basis_index} != {nbasis}")
    return by_atom


def coordinates_and_numbers(lines: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Extract atomic numbers and Cartesian coordinates from FCHK lines.

    Returns integer atomic numbers with shape ``(n_atoms,)`` and coordinates
    with shape ``(n_atoms, 3)`` in bohr.
    """
    numbers, _, _ = read_array(lines, "Atomic numbers", int)
    coords, _, _ = read_array(lines, "Current cartesian coordinates", float)
    return numbers.astype(int), coords.reshape((-1, 3))


def bonded_neighbors(numbers: np.ndarray, coords: np.ndarray, atom_index: int) -> list[int]:
    """Infer covalently bonded neighbors from distances and element radii.

    ``numbers`` and ``coords`` have shapes ``(n_atoms,)`` and ``(n_atoms, 3)``;
    coordinates are in bohr and ``atom_index`` is one-based.  Covalent radii
    tabulated in angstrom are converted to bohr, and returned neighbor indices
    are one-based.
    """
    covalent_radius_ang = {
        1: 0.31,
        6: 0.76,
        7: 0.71,
        8: 0.66,
        9: 0.57,
        15: 1.07,
        16: 1.05,
        17: 1.02,
        35: 1.20,
        53: 1.39,
    }
    center = atom_index - 1
    out: list[int] = []
    for idx, atomic_number in enumerate(numbers):
        if idx == center:
            continue
        r1 = covalent_radius_ang.get(int(numbers[center]), 0.77)
        r2 = covalent_radius_ang.get(int(atomic_number), 0.77)
        cutoff = 1.25 * (r1 + r2) * BOHR_PER_ANGSTROM
        distance = float(np.linalg.norm(coords[idx] - coords[center]))
        if distance <= cutoff:
            out.append(idx + 1)
    return out


def carbonyl_normal(numbers: np.ndarray, coords: np.ndarray, target_c: int, target_o: int) -> tuple[np.ndarray, list[int]]:
    """Determine the unit normal to a target carbonyl's local molecular plane.

    ``target_c`` and ``target_o`` are one-based indices and ``coords`` are in
    bohr.  C=O cross products with each non-oxygen carbon neighbor are aligned
    and averaged.  Returns a dimensionless Cartesian unit vector and the
    one-based list of those alpha-neighbor atoms.
    """
    c = coords[target_c - 1]
    o_vec = coords[target_o - 1] - c
    alpha_neighbors = [idx for idx in bonded_neighbors(numbers, coords, target_c) if idx != target_o]
    normals: list[np.ndarray] = []
    for idx in alpha_neighbors:
        candidate = np.cross(o_vec, coords[idx - 1] - c)
        norm = float(np.linalg.norm(candidate))
        if norm > 1.0e-8:
            candidate = candidate / norm
            if normals and float(np.dot(candidate, normals[0])) < 0.0:
                candidate = -candidate
            normals.append(candidate)
    if not normals:
        raise ValueError(f"Could not determine carbonyl plane for C{target_c}=O{target_o}")
    normal = np.sum(normals, axis=0)
    normal /= np.linalg.norm(normal)
    return normal, alpha_neighbors


def normalized(vector: np.ndarray, overlap: np.ndarray) -> np.ndarray:
    """Normalize an AO coefficient vector in the overlap metric.

    ``overlap`` is the dimensionless AO matrix ``S`` and the returned vector
    satisfies ``vector.T @ S @ vector == 1`` within numerical precision.
    """
    norm = math.sqrt(float(vector @ overlap @ vector))
    if norm <= 1.0e-12:
        raise ValueError("Zero norm vector")
    return vector / norm


def local_p_vector(nbasis: int, p_shells: dict[int, list[tuple[int, int, int]]], atom: int, direction: np.ndarray) -> np.ndarray:
    """Build an AO vector for an atom-centered p orbital along ``direction``.

    ``atom`` is one-based; ``p_shells`` contains zero-based ``px, py, pz`` AO
    positions; and ``direction`` is a three-component Cartesian unit vector.
    The dimensionless, unnormalized result has length ``nbasis`` and applies
    the same direction coefficients to every p shell on that atom.
    """
    vec = np.zeros(nbasis, dtype=float)
    shells = p_shells.get(atom, [])
    if not shells:
        raise ValueError(f"No p shells for atom {atom}")
    for px, py, pz in shells:
        vec[px] += direction[0]
        vec[py] += direction[1]
        vec[pz] += direction[2]
    return vec


def read_mo_coefficients(lines: list[str], overlap: np.ndarray) -> tuple[np.ndarray, str, float]:
    """Decode the FCHK alpha-MO coefficient array and determine its ordering.

    MO-major and basis-major interpretations are compared using overlap
    orthonormality.  Returns a dimensionless ``(nbasis, nmo)`` coefficient
    matrix, the chosen order label, and the maximum absolute error in
    ``C.T @ S @ C``.
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
    # Multiwfn prints integral matrices in a compact text format with limited
    # precision. The resulting C.T S C check is therefore not machine-precision.
    if best_matrix is None or best_error > 2.0e-2:
        raise ValueError(f"Could not identify MO coefficient order; best {best_name} err={best_error:g}")
    return best_matrix, best_name, best_error


def flatten_mo_coefficients(matrix: np.ndarray, order_name: str) -> np.ndarray:
    """Flatten a basis-by-MO coefficient matrix in the selected FCHK order.

    ``order_name`` must be ``'mo_major'`` or ``'basis_major'`` as returned by
    :func:`read_mo_coefficients`.
    """
    if order_name == "mo_major":
        return matrix.T.reshape(-1)
    if order_name == "basis_major":
        return matrix.reshape(-1)
    raise ValueError(order_name)


def build_projected_orbital(
    job: Job,
    manifest_path: Path,
    output_dir: Path,
) -> dict[str, object]:
    """Construct, write, and summarize one projected carbonyl pi-star orbital.

    ``manifest_path`` is the external full conformer manifest used to locate an
    optional matching NBO reference cube.  It is intentionally explicit so the
    sanitized Git provenance manifest cannot be mistaken for calculation input.
    The local higher-energy C/O p-perpendicular Fock eigenvector is projected
    onto all canonical virtual MOs and overlap-normalized.  It replaces the
    first virtual (LUMO) coefficient vector in a copied FCHK solely so
    ``cubegen`` can sample it.

    The returned record contains identifiers, one-based atom/MO indices,
    dimensionless normal/overlap diagnostics, energies suffixed ``_au`` in
    hartree, semicolon-delimited leading virtual contributions, and paths to
    the generated FCHK/cube plus any matching NBO pi-star cube.
    """
    workdir = output_dir / job.label
    workdir.mkdir(parents=True, exist_ok=True)
    lines = read_fchk_lines(job.fchk)
    nbasis = int(read_scalar(lines, "Number of basis functions", int))
    nocc = int(read_scalar(lines, "Number of alpha electrons", int))
    overlap_path = run_multiwfn_matrix(job.fchk, workdir, 1, "overlap_intmat.txt")
    fock_path = run_multiwfn_matrix(job.fchk, workdir, 0, "fock_intmat.txt")
    overlap = parse_multiwfn_matrix(overlap_path, nbasis)
    fock = parse_multiwfn_matrix(fock_path, nbasis)

    numbers, coords = coordinates_and_numbers(lines)
    normal, alpha_neighbors = carbonyl_normal(numbers, coords, job.target_c, job.target_o)
    p_shells = p_shell_indices(lines)
    p_c = normalized(local_p_vector(nbasis, p_shells, job.target_c, normal), overlap)
    p_o = normalized(local_p_vector(nbasis, p_shells, job.target_o, normal), overlap)

    local = np.column_stack([p_c, p_o])
    gram = local.T @ overlap @ local
    eigvals, eigvecs = np.linalg.eigh(gram)
    local_orth = local @ eigvecs @ np.diag(1.0 / np.sqrt(np.maximum(eigvals, 1.0e-14)))
    fock_local = local_orth.T @ fock @ local_orth
    local_energies, local_mix = np.linalg.eigh(fock_local)
    seed = local_orth @ local_mix[:, -1]
    seed = normalized(seed, overlap)

    mo_coeff, coeff_order, mo_orth_error = read_mo_coefficients(lines, overlap)
    virt = mo_coeff[:, nocc:]
    occ = mo_coeff[:, :nocc]
    projected = virt @ (virt.T @ overlap @ seed)
    projected = normalized(projected, overlap)
    occ_leak = float(np.max(np.abs(occ.T @ overlap @ projected)))
    coeff_in_virt = virt.T @ overlap @ projected
    top_rel = np.argsort(np.abs(coeff_in_virt))[::-1][:8]
    projected_energy = float(projected @ fock @ projected)

    modified = list(lines)
    new_coeff = mo_coeff.copy()
    new_coeff[:, nocc] = projected
    replace_float_array(modified, "Alpha MO coefficients", flatten_mo_coefficients(new_coeff, coeff_order))
    energies, _, _ = read_array(modified, "Alpha Orbital Energies", float)
    energies[nocc] = projected_energy
    replace_float_array(modified, "Alpha Orbital Energies", energies)

    out_fchk = workdir / f"{job.label}_projected_pi_star.fchk"
    out_fchk.write_text("\n".join(modified) + "\n", encoding="utf-8")
    out_cube = workdir / f"{job.label}_projected_pi_star.cube"
    if not out_cube.exists() or out_cube.stat().st_size < 10000:
        if out_cube.exists():
            out_cube.unlink()
        env = os.environ.copy()
        env["GAUSS_SCRDIR"] = str(workdir)
        cubegen_log = workdir / "cubegen_projected_pi_star.log"
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
            log_tail = cubegen_log.read_text(encoding="utf-8", errors="ignore")[-3000:] if cubegen_log.exists() else ""
            raise RuntimeError(
                f"cubegen failed for {job.label}\nreturncode={completed.returncode}\n"
                f"log tail:\n{log_tail}"
            )

    nbo_cube = ""
    nbo_manifest_rows = [
        row
        for row in read_manifest(manifest_path)
        if row["entry"] == job.entry
        and int(float(row["conf_id"])) == job.conf_id
    ]
    if nbo_manifest_rows:
        cube_value = nbo_manifest_rows[0].get("cube", "")
        if cube_value:
            cube_path = Path(cube_value)
            if cube_path.exists():
                nbo_cube = str(cube_path)

    return {
        "label": job.label,
        "entry": job.entry,
        "name": job.name,
        "InChIKey": job.inchikey,
        "conf_id": job.conf_id,
        "target_c": job.target_c,
        "target_o": job.target_o,
        "alpha_neighbors": ";".join(map(str, alpha_neighbors)),
        "normal_x": normal[0],
        "normal_y": normal[1],
        "normal_z": normal[2],
        "local_pi_energy_au": float(local_energies[0]),
        "local_pi_star_seed_energy_au": float(local_energies[-1]),
        "projected_energy_au": projected_energy,
        "projected_occ_leak_max": occ_leak,
        "mo_coeff_order": coeff_order,
        "mo_orth_error_max": mo_orth_error,
        "top_virtual_mos": ";".join(str(nocc + 1 + int(i)) for i in top_rel),
        "top_virtual_coeffs": ";".join(f"{float(coeff_in_virt[i]):.6g}" for i in top_rel),
        "projected_fchk": str(out_fchk),
        "projected_cube": str(out_cube),
        "nbo_pi_star_cube": nbo_cube,
    }


def main() -> None:
    """Build all selected prototypes and write their CSV/README summary."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help=(
            "external full conformer CSV containing status, sp_chk, target atom "
            "indices, and the other documented required columns"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="repository-external directory for generated fchk, cube, and summaries",
    )
    args = parser.parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir == ROOT or ROOT in output_dir.parents:
        raise ValueError(f"Projected-orbital output must be outside the repository: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    for job in select_representatives(args.manifest):
        print(f"building {job.label} {job.entry} conf{job.conf_id}", flush=True)
        records.append(build_projected_orbital(job, args.manifest, output_dir))
    summary_path = output_dir / "projected_pi_star_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)
    readme = output_dir / "README_projected_pi_star.md"
    readme.write_text(
        "# Projected carbonyl pi-star prototype\n\n"
        "This directory contains prototype carbonyl-centered projected virtual orbital cubes.\n\n"
        "Construction rule:\n\n"
        "1. Determine the target C=O from the NBO manifest.\n"
        "2. Determine the carbonyl p_perp direction from the local C=O plane.\n"
        "3. Build C and O p_perp local AO combinations.\n"
        "4. Orthogonalize the two local AOs using the Multiwfn overlap matrix.\n"
        "5. Diagonalize the 2x2 local Fock matrix and take the higher-energy vector as the pi-star seed.\n"
        "6. Project the seed onto the canonical virtual MO space and normalize it.\n"
        "7. Replace the LUMO coefficient vector in a copy of the fchk and use cubegen to output the cube.\n\n"
        "This is a proof-of-concept descriptor. It is less localized than NBO BD*(2), but still anchored to the target carbonyl.\n",
        encoding="utf-8",
    )
    print(f"wrote {summary_path}", flush=True)


if __name__ == "__main__":
    main()
