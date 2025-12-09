import os
from pathlib import Path
import shutil
import subprocess
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
import cclib
import time

def energy_cut(mol,res,max_energy):
    res=np.array(res)
    res[:, 1] -= res[:, 1].min()
    l=[]
    for conf,res_ in zip(mol.GetConformers(),res):
        not_converged,energy=res_
        if not_converged or energy>max_energy:
            l.append(conf.GetId())
            continue
        conf.SetProp("energy", str(energy))
    for confid in l:
        mol.RemoveConformer(confid)


def conformer_cut(mol, min_rmse, max_num_conformer):#max energy, resの処理もまとめたい。
    """
    min_rmseの閾値を超えるConformerをエネルギー順に最大max_num_conformer個残し、それ以外を削除する。
    
    Args:
        mol (rdkit.Chem.Mol): Conformerを持つ分子
        min_rmse (float): Conformer間のRMSDの閾値
        max_num_conformer (int): 残すConformerの最大数
    """

    new_mol=Chem.RemoveHs(mol)
    # Conformerのエネルギーを取得し、エネルギー順にソート
    conformer_list = sorted(
        [(float(conf.GetProp("energy")), conf.GetId()) for conf in mol.GetConformers()],
        key=lambda x: x[0])
    selected_ids = []  # 選択されたConformer IDを格納するリスト

    for _, conf1_id in conformer_list:
        if len(selected_ids) >= max_num_conformer:
            break
        keep = True  # このConformerを残すかどうか
        for conf2_id in selected_ids:
            rmsd = AllChem.GetBestRMS(new_mol, new_mol, conf1_id, conf2_id)
            if rmsd < min_rmse:  # RMSDが閾値未満なら削除対象
                keep = False
                break

        if keep:
            selected_ids.append(conf1_id)

    # 残すConformer以外を削除
    confids=[conf.GetId() for conf in mol.GetConformers()]
    for confid in confids:
        if confid not in selected_ids:
            mol.RemoveConformer(confid)


def Rodrigues_rotation(n, sin, cos):
    """
    Computes the Rodrigues' rotation matrix for a given axis and rotation angle.

    This function constructs a 3x3 rotation matrix based on the Rodrigues' rotation formula.
    The formula is used to perform a rotation around a given axis in 3D space, specified by a unit vector `n`.
    The sine (`sin`) and cosine (`cos`) of the rotation angle are used directly as inputs.

    Args:
        n (numpy.ndarray): A 3-element array representing the unit vector of the rotation axis.
        sin (float): The sine of the rotation angle.
        cos (float): The cosine of the rotation angle.

    Returns:
        numpy.ndarray: A 3x3 rotation matrix as a NumPy array.

    Formula:
        R = I * cos(θ) + (1 - cos(θ)) * (n ⊗ n) + [n]_x * sin(θ)

        Where:
        - `I` is the identity matrix.
        - `n ⊗ n` is the outer product of `n`.
        - `[n]_x` is the skew-symmetric cross-product matrix of `n`.
    """
    ans = np.array(
        [[n[0] ** 2 * (1 - cos) + cos, n[0] * n[1] * (1 - cos) - n[2] * sin, n[0] * n[2] * (1 - cos) + n[1] * sin],
         [n[0] * n[1] * (1 - cos) + n[2] * sin, n[1] ** 2 * (1 - cos) + cos, n[1] * n[2] * (1 - cos) - n[0] * sin],
         [n[0] * n[2] * (1 - cos) - n[1] * sin, n[1] * n[2] * (1 - cos) + n[0] * sin, n[2] ** 2 * (1 - cos) + cos]
         ])
    return ans

def transform(conf, carbonyl_atom):
    """
    Transforms the coordinates of a molecular conformer to align a carbonyl group with specific axes.

    This function takes a conformer's coordinates and reorients the molecule such that:
    - The carbonyl C=O bond is aligned along the x-axis.
    - The plane defined by the carbonyl group and its neighboring atoms is aligned with the xz-plane.

    The transformation involves translating the carbonyl carbon atom to the origin and applying a series
    of Rodrigues' rotations to align the specified geometric features with the desired axes.

    Args:
        conf (numpy.ndarray): A 2D NumPy array of shape (N, 3) where N is the number of atoms,
                              and each row represents the 3D coordinates (x, y, z) of an atom.
        carbonyl_atom (list or tuple): A list/tuple of four integers [c, o, c1, c2], representing the indices
                                       of the atoms in the carbonyl group:
                                       - `c`: Index of the carbon atom in the carbonyl group.
                                       - `o`: Index of the oxygen atom in the carbonyl group.
                                       - `c1`: Index of a neighboring atom bonded to `c`.
                                       - `c2`: Index of another neighboring atom bonded to `c1`.

    Returns:
        numpy.ndarray: The transformed coordinates of the conformer as a 2D NumPy array of shape (N, 3).

    Steps:
        1. Translate the molecule such that the carbonyl carbon atom (`c`) is at the origin.
        2. Align the C=O bond to the x-axis using Rodrigues' rotation.
        3. Align the neighboring atoms to the xz-plane by further rotating the molecule.

    Notes:
        - The function assumes that the input coordinates (`conf`) and the atom indices (`carbonyl_atom`)
          are consistent with the molecular structure.
        - Proper normalization is performed to avoid numerical instability during rotations.
    """
    c, o, c1, c2 = carbonyl_atom
    conf = conf - conf[c]
    a = conf[o] - conf[c]
    a = a / np.linalg.norm(a)
    cos1 = np.dot(a, np.array([1, 0, 0]))
    cros1 = np.cross(a, np.array([1, 0, 0]))
    sin1 = np.linalg.norm(cros1)
    if sin1 > 1e-6:  # Avoid division by zero
        n1 = cros1 / sin1
    else:
        n1 = np.array([0, 0, 1])  # Arbitrary axis if sin1 is too small
    b = conf[c2] - conf[c1]
    b_ = np.dot(Rodrigues_rotation(n1, sin1, cos1), b)
    byz = b_ * np.array([0, 1, 1])
    byz = byz / np.linalg.norm(byz)
    cos2 = np.dot(byz, np.array([0, 1, 0]))
    cros2 = np.cross(byz, np.array([0, 1, 0]))
    sin2 = np.linalg.norm(cros2)
    if sin2 > 1e-6:  # Avoid division by zero
        n2 = cros2 / sin2
    else:
        n2 = np.array([1, 0, 0])  # Arbitrary axis if sin2 is too small
    conf = np.dot(Rodrigues_rotation(n1, sin1, cos1), conf.T).T
    conf = np.dot(Rodrigues_rotation(n2, sin2, cos2), conf.T).T
    return conf

def run_subprocess(gjf):
    try:
        res=subprocess.run(f'source ~/.bash_profile ; g16 {gjf}', shell=True,check=True, capture_output=True, text=True, executable="/bin/bash")
        print(res)
    except subprocess.CalledProcessError as e:
        print(f"ERROR: {e}")
    return res

def run_subprocess_remote(gjf,path):
    try:
        ssh_cmd = path
        dir = os.path.dirname(gjf)
        _=gjf.replace(".gjf",".log")
        __=gjf.replace(".gjf",".chk")
        with open(gjf, "r") as gjf, open(_, "w") as log:
            subprocess.run(
                ssh_cmd, shell=True, check=True,  text=True, input=gjf.read(), stdout=log,cwd=dir
            )

    except subprocess.CalledProcessError as e:
        print(f"ERROR: {e}")

def calc_ket(out_path,smiles,run_path):
    """
    Performs computational chemistry calculations for a molecule generated from its SMILES representation.

    This function takes a SMILES string as input, generates a 3D molecular structure, optimizes the
    conformers, filters the conformers based on energy and RMSD criteria, and generates Gaussian input
    files for further quantum chemical calculations. It also calculates single-point energy and grid
    properties using Psi4 and processes the output data.

    Args:
        out_path (str): The directory path to store calculation outputs, including Gaussian input files,
                        log files, and Psi4 output files.
        smiles (str): The SMILES string representation of the molecule to be processed.

    Returns:
        None: Outputs are saved in the specified directory. No value is returned.

    Workflow:
        1. Create the output directory.
        2. Generate a molecule object from the SMILES string and add hydrogens.
        3. Assign stereochemistry and identify substructures.
        4. Generate conformers and minimize their energies using MMFF force fields.
        5. Filter conformers based on:
            - Energy difference threshold (`energy_cut`).
            - RMSD threshold (`rmsd_cut`).
            - Maximum number of conformers (`max_n_cut`).
        6. Generate Gaussian input files for the remaining conformers and execute calculations.
        7. Use cclib to extract optimized geometries and transform coordinates for Psi4 calculations.
        8. Run single-point energy calculations with Psi4 and generate ESP grid data.
        9. Rename and save output files for further analysis.

    Notes:
        - The Gaussian software (`g16`) and Psi4 must be installed and properly configured in the environment.
        - Requires RDKit for SMILES parsing, substructure matching, and conformer generation.
        - Assumes that the output directory does not already exist; if it does, the function returns early.

    Example:
        calc("/path/to/output", "CC(=O)C(C)C")

    Error Handling:
        - If the output directory cannot be created, an exception is logged, and the function exits.
        - Gaussian and Psi4 calculations are wrapped in try-except blocks to handle potential errors.
    """
    start=time.time()
    os.makedirs(out_path,exist_ok=True)
    if os.path.isfile(f'{out_path}/done'):
        return
    
    mol=Chem.MolFromSmiles(smiles)
    mol=Chem.AddHs(mol)
    substruct=Chem.MolFromSmarts("[#6](=[#8])([#6])([#6])")
    substruct=mol.GetSubstructMatch(substruct)
    if int(substruct[3])<int(substruct[2]):
        substruct=(substruct[0],substruct[1],substruct[3],substruct[2])
    AllChem.EmbedMultipleConfs(mol, numConfs=mol.GetNumAtoms()**2, randomSeed=1, numThreads=0)
    res=AllChem.MMFFOptimizeMoleculeConfs(mol,maxIters=1000,numThreads=0)
    energy_cut(mol,res,5)
    print("n",time.time()-start,len(mol.GetConformers()))
    conformer_cut(mol, min_rmse=0.5, max_num_conformer=5)
    print("n",time.time()-start,len(mol.GetConformers()))

    for conf in mol.GetConformers():
        path_a = out_path.replace("competitive_ketones","competitive_ketones_20250621")
        path_b = out_path
        log_file = f'opt{conf.GetId()}.log'

        log_path = os.path.join(path_a, log_file)

        if os.path.isfile(log_path):
            # ファイルを移動
            shutil.copy(log_path, os.path.join(path_b, log_file))
            print(f'{log_file} を {path_b} に移動しました。')
        else:
            gjf=f'{out_path}/opt{conf.GetId()}.gjf'
            xyz="\n".join(Chem.rdmolfiles.MolToXYZBlock(mol,confId=conf.GetId()).split("\n")[2:])
            with open(gjf, 'w')as f:
                input=f'%nprocshared=24\n%mem=30GB\n%chk={out_path}/opt{conf.GetId()}.chk\n# freq opt=calcfc B3LYP/def2SVP EmpiricalDispersion=GD3BJ optcyc=300 int=ultrafine\n\ngood luck!\n\n{Chem.GetFormalCharge(mol)} 1\n{xyz}'
                print(input,file=f)
            run_subprocess_remote(gjf,run_path)
        
        log_file = f'sp{conf.GetId()}.log'
        log_path = os.path.join(path_a, log_file)
        data = cclib.io.ccread(f'{out_path}/opt{conf.GetId()}.log')
        chk=f"{out_path}/sp{conf.GetId()}.chk"
        fchk=f"{out_path}/sp{conf.GetId()}.fchk"
        if os.path.isfile(log_path):
            shutil.copy(log_path, os.path.join(path_b, log_file))
            shutil.copy(log_path.replace(".log",".chk"), os.path.join(path_b, log_file).replace(".log",".chk"))
            shutil.copy(log_path.replace(".log",".fchk"), os.path.join(path_b, log_file).replace(".log",".fchk"))
            shutil.copy(log_path.replace("/sp","/Dt").replace(".log",".cube"), os.path.join(path_b, log_file).replace("/sp","/Dt").replace(".log",".cube"))
            shutil.copy(log_path.replace("/sp","/ESP").replace(".log",".cube"), os.path.join(path_b, log_file).replace("/sp","/ESP").replace(".log",".cube"))
            print(f'{log_file} を {path_b} に移動しました。')

        else:
            coords = data.atomcoords[-1]
            coords=transform(coords, substruct)
            nos = data.atomnos
            gjf=f'{out_path}/sp{conf.GetId()}.gjf'
            # xyz="\n".join(Chem.rdmolfiles.MolToXYZBlock(mol,confId=conf.GetId()).split("\n")[2:])
            input=""
            for no,coord in zip(nos,coords):
                input+=f"{no} {coord[0]: .6f} {coord[1]: .6f} {coord[2]: .6f}\n"
            with open(gjf, 'w')as f:
                input=f'%nprocshared=24\n%mem=30GB\n%chk={out_path}/sp{conf.GetId()}.chk\n# wb97xd/def2tzvp scrf=(smd,solvent=methanol) nosymm int=ultrafine\n\ngood luck!\n\n{Chem.GetFormalCharge(mol)} 1\n{input}'
                print(input,file=f)
            run_subprocess_remote(gjf,run_path)
            subprocess.run(["bash", "-c", f"source ~/.bash_profile && formchk {chk} {fchk}"])
            name=f'{out_path}/Dt{conf.GetId()}.cube'
            subprocess.run(["bash", "-c", f"source ~/.bash_profile && cubegen 24 Density=SCF {fchk} {name} -3 h"])
            name=f'{out_path}/ESP{conf.GetId()}.cube'
            subprocess.run(["bash", "-c",f"source ~/.bash_profile && cubegen 24 Potential=SCF {fchk} {name} -3 h"])
        homo_index = data.homos[0]
        name=f'{out_path}/HOMO{conf.GetId()}.cube'
        subprocess.run(["bash", "-c",f"source ~/.bash_profile && cubegen 24 MO={homo_index+1} {fchk} {name} -3 h"])
        name=f'{out_path}/LUMO{conf.GetId()}.cube'
        subprocess.run(["bash", "-c",f"source ~/.bash_profile && cubegen 24 MO={homo_index+2} {fchk} {name} -3 h"])
        name=f'{out_path}/LUMO+1_{conf.GetId()}.cube'
        subprocess.run(["bash", "-c",f"source ~/.bash_profile && cubegen 24 MO={homo_index+3} {fchk} {name} -3 h"])
        name=f'{out_path}/LUMO+2_{conf.GetId()}.cube'
        subprocess.run(["bash", "-c",f"source ~/.bash_profile && cubegen 24 MO={homo_index+4} {fchk} {name} -3 h"])
    with open(f'{out_path}/done', "w") as f:
        f.write("")

if __name__ == '__main__':
    # run_path='ssh macstudio@macstudionoMac-Studio.local "source ~/.bash_profile && g16"'
    run_path='source ~/.bash_profile && g16'
    home = Path.home()
    out_path = home / "competitive_ketones"
    df=pd.read_excel("data/mol_list.xlsx")
    df[["InChIKey","SMILES"]].apply(lambda _:calc_ket(f'{out_path}/{_.iloc[0]}',_.iloc[1],run_path),axis=1)
