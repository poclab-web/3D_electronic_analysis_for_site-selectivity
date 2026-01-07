"""
Utility functions for the competitive ketone reduction dataset.

This module provides:
- Loading & cleaning the original Excel dataset
- Exporting cleaned tables with RDKit molecule thumbnails
- Plotting ΔΔG‡ vs k2/k1 by analogue class
- Hammett and carbonyl-angle correlation plots
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress
from rdkit import Chem
from rdkit.Chem import PandasTools


def common(from_file_path: str) -> pd.DataFrame:
    """Load and preprocess the competitive-reaction Excel data.

    Parameters
    ----------
    from_file_path : str
        Path to the input Excel file.
        The file is expected to contain a 'SMILES' column and a header row
        in the second line (the first row is skipped).

    Returns
    -------
    pandas.DataFrame
        Preprocessed DataFrame, with:
        - 'mol' : RDKit Mol objects converted from 'SMILES'
        - rows with invalid 'SMILES' or missing 'mol' removed
        - 'InChIKey' : InChIKey of the hydrogen-saturated molecule
    """
    # Load
    df = pd.read_excel(from_file_path, skiprows=1)#.iloc[:150]

    # Generate RDKit Mol objects from SMILES
    df["mol"] = df["SMILES"].apply(Chem.MolFromSmiles)

    # Drop rows where SMILES or mol could not be created
    df = df.dropna(subset=["mol", "SMILES"])

    # Generate InChIKey from explicit-hydrogen molecules
    df["InChIKey"] = df["mol"].apply(
        lambda mol: Chem.inchi.MolToInchiKey(Chem.AddHs(mol))
    )

    return df


def output(df: pd.DataFrame, to_file_path: str) -> None:
    """Export a cleaned Excel file with unique molecules and RDKit images.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame. It must contain at least the following columns:
        'entry', 'SMILES', 'InChIKey', 'temperature', 'ΔΔG.expt.', 'test'.
    to_file_path : str
        Output Excel file path.

    Returns
    -------
    None
        The function writes an Excel file to `to_file_path`.
    """
    # Ensure unique molecules by InChIKey
    print(df[df.duplicated(subset="InChIKey")])
    df = df.drop_duplicates(subset="InChIKey").copy()
    print(len(df))
    # Select and sanitize columns for export
    df = df[
        ["entry","name", "SMILES", "InChIKey", "temperature", "ΔΔG.expt.", "test"]
    ].replace([np.nan, None, np.inf, -np.inf], "N/A")

    # Add RDKit molecule column (ROMol) from SMILES
    PandasTools.AddMoleculeColumnToFrame(df, "SMILES")

    # Save as Excel with molecule images
    PandasTools.SaveXlsxFromFrame(df, to_file_path, size=(100, 100))


def plot_ddg_vs_k2k1(df: pd.DataFrame, to_file_path: str) -> None:
    """Plot ΔΔG‡expt. distributions by analogue class and save as PNG.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing at least:
        - 'analogue' : categorical label for each substrate
        - 'ΔΔG.expt.' : experimental ΔΔG‡ values [kcal/mol]
    to_file_path : str
        Output image file path.

    Returns
    -------
    None
        The function saves the plot to `to_file_path`.
    """
    analogue_classes = [
        "acetophenone",
        "arylcyclo",
        "chain",
        "aliphaticcyclo",
        "polycyclic",
    ]

    # RGB colors for each analogue class
    colors = {
        "acetophenone": (233 / 255, 113 / 255, 50 / 255),
        "arylcyclo": (21 / 255, 96 / 255, 130 / 255),
        "chain": (160 / 255, 43 / 255, 147 / 255),
        "aliphaticcyclo": (25 / 255, 107 / 255, 36 / 255),
        "polycyclic": (0 / 255, 0 / 255, 0 / 255),
    }

    fig, ax_top = plt.subplots(figsize=(6, 3), facecolor="none")
    fig.patch.set_alpha(0.0)
    ax_top.set_facecolor("none")

    # y-position assigned per analogue class
    y_positions = {
        analogue: len(analogue_classes) - i
        for i, analogue in enumerate(analogue_classes)
    }

    used_y = []  # 実際に使った y 座標を記録

    for analogue in analogue_classes:
        subset = df[df["analogue"] == analogue]
        if subset.empty:
            # Skip classes not present in the dataset
            continue

        y = y_positions[analogue]
        used_y.append(y)

        x_vals = subset["ΔΔG.expt."]

        # Horizontal line indicating the range of ΔΔG‡ for this class
        ax_top.hlines(
            y=y,
            xmin=x_vals.min(),
            xmax=x_vals.max(),
            color="gray",
            linewidth=2,
        )

        # Individual points
        ax_top.plot(
            x_vals,
            [y] * len(x_vals),
            "o",
            color=colors[analogue],
            alpha=0.7,
            label=analogue,
        )

    # y 範囲に上側の余白を追加
    if used_y:
        y_min = min(used_y)
        y_max = max(used_y)
        ax_top.set_ylim(y_min - 0.2, y_max + 1)

    # Hide frame and y-axis labels
    for spine in ax_top.spines.values():
        spine.set_visible(False)
    ax_top.set_yticks([])

    # ΔΔG‡expt. axis on the top
    ax_top.xaxis.set_ticks_position("top")
    ax_top.xaxis.set_label_position("top")
    ax_top.set_xlabel(r"$\Delta\Delta G^\ddagger_{\rm expt.}$ [kcal/mol]", loc="right")
    ax_top.xaxis.labelpad = 10

    # Top arrow (→)
    ax_top.annotate(
        "",
        xy=(1.02, 1.0),
        xytext=(-0.02, 1.0),
        xycoords="axes fraction",
        arrowprops=dict(arrowstyle="->", color="black", linewidth=1.5),
    )

    # Bottom secondary axis: k2/k1
    ax_bottom = ax_top.secondary_xaxis("bottom")
    ax_bottom.set_facecolor("none")

    # ΔΔG → k2/k1 conversion: k2/k1 = exp(-ΔΔG / (0.273 * 1.99))
    kbT = 0.273 * 1.99
    k2k1_ticks = np.array([100, 10, 1, 0.1, 0.01])
    ddg_ticks = -np.log(k2k1_ticks) * kbT

    ax_bottom.set_xticks(ddg_ticks)
    ax_bottom.set_xticklabels(["100", "10", "1", "0.1", "0.01"])
    ax_bottom.set_xlabel(r"$k_2/k_1$", loc="left")

    # Bottom arrow (←)
    ax_top.annotate(
        "",
        xy=(-0.02, 0.0),
        xytext=(1.02, 0.0),
        xycoords="axes fraction",
        arrowprops=dict(arrowstyle="->", color="black", linewidth=1.5),
    )

    ax_top.set_title("")
    fig.tight_layout()
    fig.savefig(to_file_path, dpi=500, transparent=False)



def list(df: pd.DataFrame, to_file_path: str) -> None:
    """Backward-compatible wrapper for `plot_ddg_vs_k2k1`.

    Parameters
    ----------
    df : pandas.DataFrame
        Same as `plot_ddg_vs_k2k1`.
    to_file_path : str
        Same as `plot_ddg_vs_k2k1`.

    Returns
    -------
    None

    Notes
    -----
    This function is kept for compatibility with existing code that calls
    `list(df, to_file_path)`. New code should call `plot_ddg_vs_k2k1`
    instead.
    """
    plot_ddg_vs_k2k1(df, to_file_path)


def Hammettplot(x, y, save_path: str) -> None:
    """Create a Hammett ρ-plot of ΔΔG‡expt. vs Hammett σ and save it.

    Parameters
    ----------
    x : array-like
        Hammett σ values.
    y : array-like
        Experimental ΔΔG‡ values [kcal/mol] corresponding to `x`.
    save_path : str
        Output image file path.

    Returns
    -------
    None
        The function saves the plot to `save_path`.
    """
    # Linear regression
    slope, intercept, r_value, p_value, std_err = linregress(x, y)

    # Regression line
    x_fit = np.linspace(np.nanmin(x), np.nanmax(x), 100)
    y_fit = slope * x_fit + intercept

    # Compact intercept string
    intercept_str = f"+ {intercept:.1f}" if intercept >= 0 else f"- {abs(intercept):.1f}"

    # Plot
    fig, ax = plt.subplots(figsize=(3, 3), facecolor="none")
    
    ax.plot(x_fit, y_fit, color="midnightblue", linewidth=1.5)
    ax.scatter(
        x,
        y,
        facecolors="white",
        edgecolors="midnightblue",
        s=50,
        linewidth=1.5,
    )

    ax.set_xlabel("Hammett $\\sigma$")
    ax.set_ylabel(r"$\Delta\Delta G^{\ddagger}_{\mathrm{expt.}}$ [kcal/mol]")
    ax.set_xticks(np.arange(-0.5, 1.1, 0.5))
    ax.set_yticks(np.arange(-2, 1.1, 1))
    ax.grid(False)
    fig.patch.set_alpha(0.0)

    # Regression equation and R^2
    text_eq = (
        rf"$\Delta\Delta G^{{\ddagger}}_{{\mathrm{{expt.}}}}$ = {slope:.1f}$\sigma$ {intercept_str}"
        + "\n"
        + rf"$R^2$ = {r_value**2:.2f}"
    )
    ax.text(
        0.95,
        0.95,
        text_eq,
        transform=ax.transAxes,
        fontsize=8,
        verticalalignment="top",
        horizontalalignment="right",
        color="midnightblue",
    )
    ax.set_box_aspect(1)
    #ax.set_aspect("equal", adjustable="box")
    plt.tight_layout()
    plt.savefig(save_path,dpi=500)


def angleplot(x, y, save_path: str) -> None:
    """Plot ΔΔG‡expt. vs carbonyl angle and save the regression plot.

    Parameters
    ----------
    x : array-like
        Carbonyl angles in degrees.
    y : array-like
        Experimental ΔΔG‡ values [kcal/mol] corresponding to `x`.
    save_path : str
        Output image file path.

    Returns
    -------
    None
        The function saves the plot to `save_path`.
    """
    # Linear regression
    slope, intercept, r_value, p_value, std_err = linregress(x, y)

    # Regression line
    x_fit = np.linspace(np.nanmin(x), np.nanmax(x), 100)
    y_fit = slope * x_fit + intercept

    # Compact intercept string
    intercept_str = f"+ {intercept:.1f}" if intercept >= 0 else f"- {abs(intercept):.1f}"

    # Plot
    fig, ax = plt.subplots(figsize=(3, 3), facecolor="none")
    ax.plot(x_fit, y_fit, color="saddlebrown", linewidth=1.5)
    ax.scatter(
        x,
        y,
        facecolors="white",
        edgecolors="saddlebrown",
        s=50,
        linewidth=1.5,
    )

    ax.set_xlabel("carbonyl angle [deg.]")
    ax.set_ylabel(r"$\Delta\Delta G^{\ddagger}_{\mathrm{expt.}}$ [kcal/mol]")
    ax.set_xticks(np.arange(100, 131, 10))
    ax.set_yticks(np.arange(-2, 1.1, 1))
    ax.grid(False)
    fig.patch.set_alpha(0.0)

    # Regression equation and R^2
    text_eq = (
        rf"$\Delta\Delta G^{{\ddagger}}_{{\mathrm{{expt.}}}}$ = {slope:.3f}$\theta$ {intercept_str}"
        + "\n"
        + rf"$R^2$ = {r_value**2:.2f}"
    )
    ax.text(
        0.05,
        0.95,
        text_eq,
        transform=ax.transAxes,
        fontsize=8,
        verticalalignment="top",
        horizontalalignment="left",
        color="saddlebrown",
    )
    ax.set_box_aspect(1)
    plt.tight_layout()
    plt.savefig(save_path,dpi=500)


def main() -> None:
    """Run the full data-processing and plotting workflow.

    Steps
    -----
    1. Load and preprocess the original Excel data.
    2. Export a cleaned Excel file with unique molecules.
    3. Generate the ΔΔG vs k2/k1 analogue plot.
    4. Generate Hammett and carbonyl-angle correlation plots.
    """
    # 1. Load data
    df = common("data/all_experimental_data.xlsx")

    # 2. Cleaned unique-molecule table
    output(df, "data/data.xlsx")

    # 3. ΔΔG vs k2/k1 overview plot
    plot_ddg_vs_k2k1(df, "data/deltaG_k2k1.png")

    # 4. Hammett plot (rows with non-missing Hammett σ)
    mask_hammett = df["Hammett σ"].notna()
    Hammettplot(
        df.loc[mask_hammett, "Hammett σ"].values,
        df.loc[mask_hammett, "ΔΔG.expt."].values,
        "data/hammett.png",
    )

    # 4'. Carbonyl-angle plot (rows with non-missing angle)
    mask_angle = df["carbonyl angle"].notna()
    angleplot(
        df.loc[mask_angle, "carbonyl angle"].values,
        df.loc[mask_angle, "ΔΔG.expt."].values,
        "data/carbonyl angle.png",
    )


if __name__ == "__main__":
    main()
