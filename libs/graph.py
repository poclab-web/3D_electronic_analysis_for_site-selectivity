"""Legacy plotting utilities plus figures shared by the adopted model.

The historical three-field analysis accumulated many plotting and kinetic
helpers in this module.  New paper-model code should use only the explicitly
imported figure helpers; model fitting and validation live in
``libs/current_model.py``.  Diketone summary kinetics delegate to the tested
implementation in ``libs/diketone_metrics.py``.
"""

import glob
import os
import re
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle, Polygon
from matplotlib.ticker import MultipleLocator
from matplotlib.transforms import Bbox
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from rdkit import Chem
from rdkit.Chem import Draw, PandasTools

OUTPUT_ROOT = Path(os.path.expanduser("~")) / "molecules"
CONTRIBUTIONS_ROOT = Path(os.getenv("CONTRIBUTIONS_ROOT", "data/contributions"))
PREFERRED_REGRESSION_METHOD = os.getenv(
    "PREFERRED_REGRESSION_METHOD",
    "Lasso 0.006",
)
BENZOPHENONE_REF_INCHIKEY = "RWCCWEUUXYIKHB-KHWBWMQUSA-N"

CONTRIBUTION_VIOLIN_COMPONENTS = [
    ("Electronic", "electronic_cont", "darkmagenta"),
    ("Electrostatic", "electrostatic_cont", "forestgreen"),
    ("LUMO", "lumo_cont", "saddlebrown"),
]

SKELETON_GROUP_COLORS = {
    "A": "#4c78a8",
    "B": "#f58518",
    "C": "#54a24b",
    "D": "#b279a2",
    "E": "#e45756",
}

SKELETON_HIGHLIGHT_PICKS = [
    ("Electronic", "A", "A24", "A positive outlier"),
    ("Electronic", "B", "B6", "B negative representative"),
    ("Electronic", "C", "C3", "C negative representative"),
    ("Electronic", "D", "D11(trans)", "D large negative"),
    ("Electronic", "E", "E2(exo)", "E negative representative"),
    ("Electrostatic", "A", "A12", "A large negative"),
    ("Electrostatic", "B", "B5", "B large negative"),
    ("Electrostatic", "C", "C10", "C large positive"),
    ("Electrostatic", "D", "D13(cis)", "D large positive"),
    ("Electrostatic", "E", "E3(exo)", "E large positive"),
    ("LUMO", "A", "A16", "A positive representative"),
    ("LUMO", "B", "B5", "B large positive"),
    ("LUMO", "C", "C12", "C large negative"),
    ("LUMO", "D", "D11(trans)", "D large negative"),
    ("LUMO", "E", "E4", "E large negative"),
]

SELECTED_CONTRIBUTION_BREAKDOWN_ENTRIES = ("A24", "A12", "C3", "D11(trans)", "B5")


def _ensure_output_dir(save_path: str | Path) -> Path:
    """Create the parent directory for an output path and return Path object."""
    out_path = Path(save_path)
    if out_path.parent and not out_path.parent.exists():
        out_path.parent.mkdir(parents=True, exist_ok=True)
    return out_path

def nan_rmse(x,y):
    """
    Calculates the Root Mean Square Error (RMSE) while ignoring NaN values.

    This function computes the RMSE between two arrays, where NaN values in the
    first array (`x`) are ignored in the calculation.

    Args:
        x (numpy.ndarray or pandas.Series): Predicted values, which may contain NaN values.
        y (numpy.ndarray or pandas.Series): Actual values, corresponding to `x`.

    Returns:
        float: The RMSE value, calculated as:
               \[
               \text{RMSE} = \sqrt{\frac{1}{N} \sum_{i=1}^{N} (y_i - x_i)^2}
               \]
               where \( N \) is the number of non-NaN values in `x`.
    """
    return np.sqrt(np.nanmean((y-x)**2))

def nan_r2(x,y):
    """
    Calculates the coefficient of determination (R²) while ignoring NaN values.

    This function computes the R² score between two arrays, where NaN values in
    the first array (`x`) are ignored. The R² score indicates the proportion of
    variance in `y` that is predictable from `x`.

    Args:
        x (numpy.ndarray or pandas.Series): Predicted values, which may contain NaN values.
        y (numpy.ndarray or pandas.Series): Actual values, corresponding to `x`.

    Returns:
        float: The R² value, calculated as:
               \[
               R^2 = 1 - \frac{\sum (y_i - x_i)^2}{\sum (y_i - \bar{y})^2}
               \]
               where:
               - \( \bar{y} \) is the mean of the non-NaN `y` values.
               - The summations ignore NaN values in `x`.
    """
    x,y=x[~np.isnan(x)],y[~np.isnan(x)]
    return 1-np.sum((y-x)**2)/np.sum((y-np.mean(y))**2)

def _preferred_cv_column(df_results: pd.DataFrame) -> str:
    """Select the preferred manuscript model if present; otherwise use best CV RMSE."""
    preferred = PREFERRED_REGRESSION_METHOD.strip()
    if preferred and preferred.lower() not in {"none", "auto"}:
        preferred_column = f"{preferred} cv"
        if preferred_column in df_results.index:
            return preferred_column
        print(
            f"Preferred regression column '{preferred_column}' was not found; "
            "falling back to the lowest CV RMSE."
        )
    return df_results["cv_RMSE"].idxmin()


def evaluate_result(path):
    """Score every legacy CV/regression column and write a results CSV.

    ``path`` is the ``*_regression.pkl`` table produced by
    :mod:`libs.regression`; it must contain ``ΔΔG.expt.`` plus paired ``cv``
    and ``regression`` prediction columns in kcal/mol.  The preferred CV column
    name is returned after RMSE and R-squared values are written beside the
    pickle as ``*_results.csv``.
    """
    df=pd.read_pickle(path)
    df_results=pd.DataFrame(index=df.filter(like='cv').columns)
    df_results["cv_RMSE"]=df_results.index.map(lambda column: nan_rmse(df[column].values,df["ΔΔG.expt."].values))
    df_results["cv_r2"]=df_results.index.map(lambda column: nan_r2(df[column].values,df["ΔΔG.expt."].values))
    df_results["regression_RMSE"]=df.filter(like='regression').columns.map(lambda column: nan_rmse(df[column].values,df["ΔΔG.expt."].values))
    df_results["regression_r2"]=df.filter(like='regression').columns.map(lambda column: nan_r2(df[column].values,df["ΔΔG.expt."].values))
    df_results.to_csv(path.replace("_regression.pkl","_results.csv"))
    best_cv_column = _preferred_cv_column(df_results)
    print(best_cv_column, df_results.loc[best_cv_column, ["cv_RMSE", "cv_r2"]].to_dict())
    return best_cv_column

def best_parameter(path):
    """Reconstruct per-grid contributions for the selected legacy model.

    ``path`` names a ``*_results.csv`` file with matching ``*_regression.csv``
    coefficients and ``*_regression.pkl`` predictions.  Folded coefficients
    are applied to unfolded electronic, electrostatic, and LUMO fields; an XLSX
    summary is written and the augmented DataFrame is returned.  Energies and
    contributions follow the source model's kcal/mol target scale.
    """
    results = pd.read_csv(path, index_col=0)
    best_cv_column = _preferred_cv_column(results)
    coef=pd.read_csv(path.replace("_results.csv","_regression.csv"), index_col=0)
    coef = coef[[best_cv_column.replace("cv", "electronic_coef"), best_cv_column.replace("cv", "electrostatic_coef"), best_cv_column.replace("cv", "lumo_coef")]]
    coef.columns = ["electronic_coef", "electrostatic_coef","lumo_coef"]
    df=pd.read_pickle(path.replace("_results.csv","_regression.pkl"))
    columns=df.filter(like='electronic_unfold').columns.tolist()+df.filter(like='electrostatic_unfold').columns.tolist()+df.filter(like='lumo_unfold').columns.tolist()
    def calc_cont(column):
        """Multiply one unfolded descriptor column by its folded coefficient."""
        x,y,z=map(int, re.findall(r'[+-]?\d+', column))
        coef_column=column.replace(f"_unfold {x} {y} {z}","_coef")
        return df[column]*coef.at[f'{x} {abs(y)} {z}',coef_column]#*np.sign(z)
    data = {col.replace("unfold","cont"): calc_cont(col) for col in columns}   
    # data={col.replace("unfold","cont"): calc_cont(col) for col in df.filter(like='electronic_unfold').columns}
    data=pd.DataFrame(data=data)
    data["electronic_cont"],data["electrostatic_cont"],data["lumo_cont"]=data.iloc[:,:len(data.columns)//3].sum(axis=1),data.iloc[:,len(data.columns)//3:len(data.columns)*2//3].sum(axis=1),data.iloc[:,len(data.columns)*2//3:].sum(axis=1)
    df=pd.concat([df,data],axis=1)
    df["intercept"]=df[best_cv_column.replace("cv","intercept")]
    df["cv"]=df[best_cv_column]#<df["ΔΔG.expt."]
    df["prediction"]=df[best_cv_column.replace("cv","prediction")]
    df["regression"]=df[best_cv_column.replace("cv","regression")]
    df["cv_error"]=df["cv"]-df["ΔΔG.expt."]
    df["prediction_error"]=df["prediction"]-df["ΔΔG.expt."]
    #df[["electronic_cont","electrostatic_cont","lumo_cont"]]=df[["electronic_cont","electrostatic_cont","lumo_cont"]]-df[df["InChIKey"]=="RWCCWEUUXYIKHB-KHWBWMQUSA-N"][["electronic_cont","electrostatic_cont","lumo_cont"]].values
    # df = df.reindex(df[["prediction_error","cv_error"]].abs().sort_values(ascending=False).index)
    df_=df[["entry","name","SMILES","InChIKey","ΔΔG.expt.","electronic_cont","electrostatic_cont","lumo_cont","intercept","regression","prediction","cv","prediction_error","cv_error"]].fillna("NAN")#.sort_values(["cv_error","prediction_error"])
    PandasTools.AddMoleculeColumnToFrame(df_, "SMILES")
    path=path.replace(".pkl",".xlsx")
    PandasTools.SaveXlsxFromFrame(df_,path.replace("_results.csv","_regression.xlsx"), size=(100, 100))
    return df#[["ΔΔG.expt.","regression","prediction","cv"]]



def plot_3d_contributions(
    df: pd.DataFrame,
    save_path: str,
    highlight_colors=None,   # {InChIKey: "Label"}
    ref_inchikey: str | None = None,
) -> None:
    """Plot contributions in 3D (electronic / electrostatic / orbital).

    Axes
    ----
    x : electronic_cont    -> electronic [kcal/mol]
    y : electrostatic_cont -> electrostatic [kcal/mol]
    z : lumo_cont          -> orbital [kcal/mol]

    highlight_colors : dict or None
        {InChIKey: "Label"}.
        - Points whose InChIKey is in the dict are highlighted:
          larger, opaque markers with projection lines.
        - "Label" is shown near the corresponding point.

    ref_inchikey : str or None
        If not None, all (x, y, z) values are shifted to
            value(InChIKey) - value(ref_inchikey)
        for each of electronic / electrostatic / lumo.

    Contribution fractions
    ----------------------
    After (optional) shifting by ref_inchikey, the contribution fractions
    are computed as

        S_x = sum_i |x_i|
        S_y = sum_i |y_i|
        S_z = sum_i |z_i|
        S_tot = S_x + S_y + S_z

        frac_x = S_x / S_tot
        frac_y = S_y / S_tot
        frac_z = S_z / S_tot

    (If S_tot == 0, all three fractions are set to 0.)

    These fractions are shown in parentheses under each axis label
    as percentages, e.g. "electronic [kcal/mol]\\n(42.3%)".
    """
    required_cols = [
        "electronic_cont",
        "electrostatic_cont",
        "lumo_cont",
        "regression",
        "InChIKey",
    ]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in DataFrame.")

    # highlight map (InChIKey -> Label)
    if highlight_colors is None:
        highlight_map: dict[str, str] = {}
    elif isinstance(highlight_colors, dict):
        highlight_map = highlight_colors
    else:
        raise TypeError(
            "highlight_colors must be a dict like {InChIKey: 'Label'} or None."
        )

    # use only rows with regression values
    df_reg = df.dropna(
        subset=["electronic_cont", "electrostatic_cont", "lumo_cont", "regression"]
    ).copy()

    # --- shift to differences vs ref_inchikey, if requested ---
    if ref_inchikey is not None:
        if ref_inchikey not in df_reg["InChIKey"].values:
            raise ValueError(
                f"ref_inchikey '{ref_inchikey}' was not found in DataFrame."
            )
        ref_row = df_reg[df_reg["InChIKey"] == ref_inchikey].iloc[0]
        ref_e = float(ref_row["electronic_cont"])
        ref_es = float(ref_row["electrostatic_cont"])
        ref_l = float(ref_row["lumo_cont"])

        df_reg["electronic_cont"] = df_reg["electronic_cont"] - ref_e
        df_reg["electrostatic_cont"] = df_reg["electrostatic_cont"] - ref_es
        df_reg["lumo_cont"] = df_reg["lumo_cont"] - ref_l

    x_reg = df_reg["electronic_cont"].values
    y_reg = df_reg["electrostatic_cont"].values
    z_reg = df_reg["lumo_cont"].values
    inchis = df_reg["InChIKey"].values

    # --- contribution fractions (absolute-sum based) ---
    sum_abs_x = float(np.sum(np.abs(x_reg)))
    sum_abs_y = float(np.sum(np.abs(y_reg)))
    sum_abs_z = float(np.sum(np.abs(z_reg)))
    sum_abs_total = sum_abs_x + sum_abs_y + sum_abs_z

    if sum_abs_total == 0.0:
        frac_x = frac_y = frac_z = 0.0
    else:
        frac_x = sum_abs_x / sum_abs_total
        frac_y = sum_abs_y / sum_abs_total
        frac_z = sum_abs_z / sum_abs_total

    # highlight flags
    is_high = np.array([ik in highlight_map for ik in inchis], dtype=bool)
    is_norm = ~is_high

    # axis ranges (later enforce 1:1:1 aspect ratio)
    x_all = x_reg
    y_all = y_reg
    z_all = z_reg

    xmin, xmax = x_all.min(), x_all.max()
    ymin, ymax = y_all.min(), y_all.max()
    zmin, zmax = z_all.min(), z_all.max()

    margin = 0.05
    dx = (xmax - xmin) or 1.0
    dy = (ymax - ymin) or 1.0
    dz = (zmax - zmin) or 1.0

    x_min_lim = xmin - margin * dx
    x_max_lim = xmax + margin * dx
    y_min_lim = ymin - margin * dy
    y_max_lim = ymax + margin * dy
    z_min_lim = zmin - margin * dz
    z_max_lim = zmax + margin * dz

    fig = plt.figure(figsize=(4, 4), facecolor="white")
    fig.patch.set_alpha(0.0)
    ax = fig.add_subplot(111, projection="3d")

    # axis limits
    ax.set_xlim(x_min_lim, x_max_lim)
    ax.set_ylim(y_min_lim, y_max_lim)
    ax.set_zlim(z_min_lim, z_max_lim)

    # 1:1:1 aspect
    span_x = x_max_lim - x_min_lim
    span_y = y_max_lim - y_min_lim
    span_z = z_max_lim - z_min_lim
    ax.set_box_aspect((span_x, span_y, span_z))

    # integer ticks only
    xticks = np.arange(int(np.floor(x_min_lim)), int(np.ceil(x_max_lim)) + 1)
    yticks = np.arange(int(np.floor(y_min_lim)), int(np.ceil(y_max_lim)) + 1)
    zticks = np.arange(int(np.floor(z_min_lim)), int(np.ceil(z_max_lim)) + 1)
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.set_zticks(zticks)

    # axis colors
    color_x = "darkmagenta"   # electronic
    color_y = "forestgreen"   # electrostatic
    color_z = "saddlebrown"   # orbital

    ax.tick_params(axis="x", colors=color_x, pad=0)
    ax.tick_params(axis="y", colors=color_y, pad=0)
    ax.tick_params(axis="z", colors=color_z, pad=0)

    for lbl in ax.get_xticklabels():
        lbl.set_color(color_x)
    for lbl in ax.get_yticklabels():
        lbl.set_color(color_y)
    for lbl in ax.get_zticklabels():
        lbl.set_color(color_z)

    # try to color the 3D axis lines (version-dependent)
    for axis3d, c in [
        (getattr(ax, "w_xaxis", None), color_x),
        (getattr(ax, "w_yaxis", None), color_y),
        (getattr(ax, "w_zaxis", None), color_z),
    ]:
        if axis3d is not None and hasattr(axis3d, "line"):
            axis3d.line.set_color(c)
            axis3d.line.set_linewidth(1.5)

    # base scatter: normal points (semi-transparent black)
    if is_norm.any():
        ax.scatter(
            x_reg[is_norm],
            y_reg[is_norm],
            z_reg[is_norm],
            c="black",
            marker="o",
            s=10,
            alpha=0.5,
            edgecolor="none",
        )

    # highlighted points: larger and opaque
    if is_high.any():
        ax.scatter(
            x_reg[is_high],
            y_reg[is_high],
            z_reg[is_high],
            c="black",
            marker="o",
            s=40,
            alpha=1.0,
            edgecolor="none",
        )

    # viewpoint (after limits are set)
    ax.view_init(elev=25, azim=45)

    # final limits for planes
    x_min_plane, x_max_plane = ax.get_xlim3d()
    y_min_plane, y_max_plane = ax.get_ylim3d()
    z_min_plane, z_max_plane = ax.get_zlim3d()

    # --- planes (xy, yz, zx) in different greys ---
    xy_verts = [
        [x_min_plane, y_min_plane, z_min_plane],
        [x_max_plane, y_min_plane, z_min_plane],
        [x_max_plane, y_max_plane, z_min_plane],
        [x_min_plane, y_max_plane, z_min_plane],
    ]
    poly_xy = Poly3DCollection(
        [xy_verts],
        facecolors=(0.9, 0.9, 0.9, 0.4),
        edgecolors="none",
    )
    ax.add_collection3d(poly_xy)

    yz_verts = [
        [x_min_plane, y_min_plane, z_min_plane],
        [x_min_plane, y_max_plane, z_min_plane],
        [x_min_plane, y_max_plane, z_max_plane],
        [x_min_plane, y_min_plane, z_max_plane],
    ]
    poly_yz = Poly3DCollection(
        [yz_verts],
        facecolors=(0.8, 0.8, 0.8, 0.4),
        edgecolors="none",
    )
    ax.add_collection3d(poly_yz)

    zx_verts = [
        [x_min_plane, y_min_plane, z_min_plane],
        [x_max_plane, y_min_plane, z_min_plane],
        [x_max_plane, y_min_plane, z_max_plane],
        [x_min_plane, y_min_plane, z_max_plane],
    ]
    poly_zx = Poly3DCollection(
        [zx_verts],
        facecolors=(0.7, 0.7, 0.7, 0.4),
        edgecolors="none",
    )
    ax.add_collection3d(poly_zx)

    def add_projections(
        xs: np.ndarray,
        ys: np.ndarray,
        zs: np.ndarray,
        alpha: float,
        size: float,
        lw: float = 0.5,
    ) -> None:
        """Draw projection lines and points onto xy, xz, yz planes."""
        line_alpha = alpha * 0.6
        for xi, yi, zi in zip(xs, ys, zs):
            # to xy-plane: along -z (z-axis color)
            ax.plot(
                [xi, xi],
                [yi, yi],
                [zi, z_min_plane],
                color=color_z,
                alpha=line_alpha,
                linewidth=lw,
            )
            ax.plot(
                [xi],
                [yi],
                [z_min_plane],
                "o",
                color=color_z,
                alpha=alpha,
                markersize=size,
            )

            # to xz-plane: along -y (y-axis color)
            ax.plot(
                [xi, xi],
                [yi, y_min_plane],
                [zi, zi],
                color=color_y,
                alpha=line_alpha,
                linewidth=lw,
            )
            ax.plot(
                [xi],
                [y_min_plane],
                [zi],
                "o",
                color=color_y,
                alpha=alpha,
                markersize=size,
            )

            # to yz-plane: along -x (x-axis color)
            ax.plot(
                [xi, x_min_plane],
                [yi, yi],
                [zi, zi],
                color=color_x,
                alpha=line_alpha,
                linewidth=lw,
            )
            ax.plot(
                [x_min_plane],
                [yi],
                [zi],
                "o",
                color=color_x,
                alpha=alpha,
                markersize=size,
            )

    if is_norm.any():
        add_projections(
            x_reg[is_norm],
            y_reg[is_norm],
            z_reg[is_norm],
            alpha=0.25,
            size=2.0,
            lw=0.5,
        )

    if is_high.any():
        add_projections(
            x_reg[is_high],
            y_reg[is_high],
            z_reg[is_high],
            alpha=1.0,
            size=4.0,
            lw=0.8,
        )

    # axis labels with contribution fractions
    ax.set_xlabel(
        f"Electronic\n(contribution = {frac_x*100:.1f}%)",# [kcal/mol]
        color=color_x,
    )
    ax.set_ylabel(
        f"Electrostatic\n(contribution = {frac_y*100:.1f}%)",# [kcal/mol]
        color=color_y,
    )
    ax.set_zlabel(
        f"LUMO\n(contribution = {frac_z*100:.1f}%)", # [kcal/mol]
        color=color_z,
    )
    ax.text(
        -2,
        2,
        4,
        "unit: [kcal/mol]",   # always show sign (+X.XX / -X.XX)
        va="center",
        ha="right",      # right-aligned
        # fontsize=8,
    )
    plt.tight_layout()

    save_path = _ensure_output_dir(save_path)

    fig.savefig(
        save_path,
        dpi=500,
        transparent=False,
        bbox_inches="tight",
        pad_inches=1,
    )
    plt.close(fig)





def plot_contribution_bars(
    df: pd.DataFrame,
    inchikeys: list[str],
    labels: list[str],
    save_path: str,
    ref_inchikey: str | None = None,
) -> None:
    """Plot per-molecule electronic/electrostatic/LUMO contributions as grouped bars.

    Three category blocks are shown on the x-axis:
    ``electronic``, ``electrostatic``, and ``lumo``.
    For each block, one bar is drawn per molecule in ``inchikeys``.

    If ``ref_inchikey`` is provided, values are plotted as differences:
    ``target - reference`` for each contribution type.
    """
    required_cols = [
        "InChIKey",
        "electronic_cont",
        "electrostatic_cont",
        "lumo_cont",
    ]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"DataFrame must contain column '{col}'.")

    # Normalize inchikeys to a list.
    if isinstance(inchikeys, str):
        inchikey_list = [inchikeys]
    else:
        inchikey_list = list(inchikeys)

    # Validate that requested InChIKeys exist.
    for ik in inchikey_list:
        if ik not in df["InChIKey"].values:
            raise ValueError(f"InChIKey '{ik}' was not found in DataFrame.")

    # Reference contribution values in difference mode.
    ref_vals = None
    if ref_inchikey is not None:
        if ref_inchikey not in df["InChIKey"].values:
            raise ValueError(
                f"ref_inchikey '{ref_inchikey}' was not found in DataFrame."
            )
        ref_row = df[df["InChIKey"] == ref_inchikey].iloc[0]
        ref_vals = np.array(
            [
                ref_row["electronic_cont"],
                ref_row["electrostatic_cont"],
                ref_row["lumo_cont"],
            ],
            dtype=float,
        )

    # Contribution matrix in InChIKey order, shape: (n_molecules, 3).
    contributions = []
    for ik in inchikey_list:
        row = df[df["InChIKey"] == ik].iloc[0]
        vals = np.array(
            [
                row["electronic_cont"],
                row["electrostatic_cont"],
                row["lumo_cont"],
            ],
            dtype=float,
        )
        if ref_vals is not None:
            vals = vals - ref_vals
        contributions.append(vals)
    contributions = np.vstack(contributions)  # shape: (N, 3)

    n_mol = contributions.shape[0]
    categories = ["electronic", "electrostatic", "lumo"]
    n_cat = len(categories)

    # Centers of the three category blocks on the x-axis.
    x_base = np.arange(n_cat, dtype=float)

    # Offset one bar per InChIKey within each category block.
    total_width = 0.8
    bar_width = total_width / max(n_mol, 1)

    # Create figure.
    fig, ax = plt.subplots(figsize=(5, 3))

    for i, ik in enumerate(labels):
        # Contributions for the i-th molecule: electronic, electrostatic, lumo.
        vals = contributions[i, :]
        # X positions inside the three category blocks.
        x_pos = x_base + (i - (n_mol - 1) / 2) * bar_width

        ax.bar(
            x_pos,
            vals,
            width=bar_width,
            label=ik,
            alpha=0.8,
        )

    # Zero baseline.
    ax.axhline(0, color="black", linewidth=1.0)

    # Tick labels at category block centers.
    ax.set_xticks(x_base)
    ax.set_xticklabels(categories)

    # Y-axis label depends on absolute/difference mode.
    if ref_vals is None:
        ax.set_ylabel("contribution [kcal/mol]")
        title = "Contributions"
    else:
        ax.set_ylabel(
            "contribution difference [kcal/mol]"
            # f"\n(relative to {ref_inchikey})"
        )
        # title = "Contribution differences"

    # ax.set_xlabel("contribution type")
    # ax.set_title(title)

    # Legend for molecules.
    ax.legend(frameon=False, fontsize=8, ncol=1)

    # Add small x-margin.
    ax.margins(x=0.1)

    fig.tight_layout()

    save_path = _ensure_output_dir(save_path)

    fig.savefig(save_path, dpi=400)
    plt.close(fig)


def plot_training_contribution_numberlines(
    df: pd.DataFrame,
    save_path: str,
    label_column: str = "entry",
    train_value: int = 0,
) -> None:
    """Plot training-substrate regression contributions on three number lines.

    The function expects the DataFrame returned by :func:`best_parameter`,
    or another DataFrame with the same columns. Only rows with ``test == 0``
    are used. The plotted values are the fitted-regression contribution terms
    for ``electronic_cont``, ``electrostatic_cont``, and ``lumo_cont``.

    Parameters
    ----------
    df : pandas.DataFrame
        Regression/contribution table containing ``test``,
        ``electronic_cont``, ``electrostatic_cont``, and ``lumo_cont``.
    save_path : str
        Output image path.
    label_column : str, optional
        Column used to label the minimum and maximum point on each number line.
        Defaults to ``"entry"``.
    train_value : int, optional
        Value in the ``test`` column that denotes training data. Defaults to 0.

    Returns
    -------
    None
        The figure is saved to ``save_path``.
    """
    required_cols = [
        "test",
        "electronic_cont",
        "electrostatic_cont",
        "lumo_cont",
    ]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"DataFrame must contain column '{col}'.")

    label_column_available = label_column in df.columns

    test_values = pd.to_numeric(df["test"], errors="coerce")
    train_df = df[test_values == train_value].copy()
    train_df = train_df.dropna(
        subset=["electronic_cont", "electrostatic_cont", "lumo_cont"]
    )
    if train_df.empty:
        raise ValueError(f"No training rows with test == {train_value} were found.")

    components = [
        ("electronic_cont", "Electronic", "darkmagenta"),
        ("electrostatic_cont", "Electrostatic", "forestgreen"),
        ("lumo_cont", "LUMO", "saddlebrown"),
    ]

    all_values = np.concatenate(
        [train_df[col].to_numpy(dtype=float) for col, _, _ in components]
    )
    all_values = all_values[np.isfinite(all_values)]
    if all_values.size == 0:
        raise ValueError("No finite contribution values were found.")

    max_abs = float(np.max(np.abs(all_values)))
    if max_abs == 0.0:
        max_abs = 1.0
    x_lim = max_abs * 1.15

    fig, axes = plt.subplots(
        nrows=3,
        ncols=1,
        figsize=(5.5, 3.4),
        sharex=True,
    )
    fig.patch.set_alpha(0.0)

    rng = np.random.default_rng(0)
    n_train = len(train_df)

    for ax, (col, label, color) in zip(axes, components):
        values = train_df[col].to_numpy(dtype=float)
        finite_mask = np.isfinite(values)
        values = values[finite_mask]
        plot_df = train_df.loc[finite_mask].copy()

        # Small deterministic vertical jitter keeps overlapping substrates visible
        # while preserving the number-line interpretation.
        jitter = rng.uniform(-0.06, 0.06, size=len(values))

        ax.axhline(0.0, color="black", linewidth=1.0, zorder=0)
        ax.axvline(0.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.8)
        ax.scatter(
            values,
            jitter,
            s=18,
            color=color,
            edgecolors="white",
            linewidths=0.4,
            alpha=0.78,
            zorder=2,
        )

        mean_value = float(np.mean(values)) if len(values) else 0.0
        ax.scatter(
            [mean_value],
            [0.0],
            marker="D",
            s=34,
            color="black",
            edgecolors="white",
            linewidths=0.5,
            zorder=3,
            label="mean",
        )

        if label_column_available and len(values):
            for idx in [int(np.argmin(values)), int(np.argmax(values))]:
                point_label = str(plot_df.iloc[idx][label_column])
                ax.annotate(
                    point_label,
                    xy=(values[idx], jitter[idx]),
                    xytext=(4, 5 if values[idx] >= 0 else -11),
                    textcoords="offset points",
                    fontsize=7,
                    color=color,
                    ha="left",
                    va="bottom" if values[idx] >= 0 else "top",
                )

        ax.set_ylabel(
            f"{label}\n[kcal/mol]",
            rotation=0,
            ha="right",
            va="center",
            labelpad=42,
            color=color,
        )
        ax.set_ylim(-0.22, 0.22)
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.grid(True, axis="x", linestyle=":", linewidth=0.6, alpha=0.45)

    axes[-1].set_xlim(-x_lim, x_lim)
    axes[-1].set_xlabel(
        r"Contribution to fitted $\Delta\Delta G^\ddagger_{\mathrm{regression}}$ [kcal/mol]"
    )
    axes[0].set_title(
        f"Training-substrate contribution distributions (N = {n_train})",
        fontsize=10,
    )
    axes[0].legend(loc="upper right", frameon=False, fontsize=8)

    fig.tight_layout()
    save_path = _ensure_output_dir(save_path)
    fig.savefig(save_path, dpi=500, transparent=False)
    plt.close(fig)


def _prepare_skeleton_contribution_delta(
    df: pd.DataFrame,
    ref_inchikey: str = BENZOPHENONE_REF_INCHIKEY,
    train_value: int = 0,
) -> pd.DataFrame:
    """Select A-E training rows and reference-center their contributions.

    The input requires identifiers, ``test``, and the three ``*_cont`` columns
    in kcal/mol.  Returned ``*_cont_delta`` columns are differences from the
    row identified by ``ref_inchikey`` within the selected training subset.
    """
    required_cols = [
        "test",
        "entry",
        "name",
        "InChIKey",
        "electronic_cont",
        "electrostatic_cont",
        "lumo_cont",
    ]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"DataFrame must contain column '{col}'.")

    test_values = pd.to_numeric(df["test"], errors="coerce")
    train_df = df[test_values == train_value].copy()
    train_df["skeleton_group"] = (
        train_df["entry"].astype(str).str.extract(r"^([A-E])", expand=False)
    )
    train_df = train_df.dropna(subset=["skeleton_group"]).copy()
    if train_df.empty:
        raise ValueError("No A-E training rows were found for the violin plot.")

    contribution_cols = [column for _, column, _ in CONTRIBUTION_VIOLIN_COMPONENTS]
    for column in contribution_cols:
        train_df[column] = pd.to_numeric(train_df[column], errors="coerce")
    train_df = train_df.dropna(subset=contribution_cols).copy()

    ref_rows = train_df[train_df["InChIKey"] == ref_inchikey]
    if ref_rows.empty:
        raise ValueError(f"Reference InChIKey '{ref_inchikey}' was not found.")
    ref_row = ref_rows.iloc[0]

    for _, column, _ in CONTRIBUTION_VIOLIN_COMPONENTS:
        train_df[f"{column}_delta"] = (
            train_df[column].astype(float) - float(ref_row[column])
        )
    return train_df


def _skeleton_contribution_summary(delta_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize each contribution delta by skeleton group and component.

    The result has ``group``, ``feature``, ``n``, ``mean``, ``median``,
    ``min``, ``max``, and sample ``std`` columns; energy statistics retain the
    input contribution unit (kcal/mol in this workflow).
    """
    rows: list[dict[str, object]] = []
    for group, group_df in delta_df.groupby("skeleton_group"):
        for feature, column, _ in CONTRIBUTION_VIOLIN_COMPONENTS:
            values = group_df[f"{column}_delta"].to_numpy(dtype=float)
            rows.append(
                {
                    "group": group,
                    "feature": feature,
                    "n": len(values),
                    "mean": float(np.mean(values)),
                    "median": float(np.median(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                    "std": (
                        float(np.std(values, ddof=1))
                        if len(values) > 1
                        else 0.0
                    ),
                }
            )
    return pd.DataFrame(rows)


def _skeleton_highlight_rows(delta_df: pd.DataFrame) -> pd.DataFrame:
    """Extract the configured representative entries for violin annotations."""
    rows: list[dict[str, object]] = []
    by_entry = delta_df.set_index("entry", drop=False)
    column_by_feature = {
        feature: column for feature, column, _ in CONTRIBUTION_VIOLIN_COMPONENTS
    }
    for feature, group, entry, reason in SKELETON_HIGHLIGHT_PICKS:
        if entry not in by_entry.index:
            continue
        row = by_entry.loc[entry]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        column = column_by_feature[feature]
        rows.append(
            {
                "feature": feature,
                "group": group,
                "entry": entry,
                "name": row.get("name", ""),
                "InChIKey": row.get("InChIKey", ""),
                "value": float(row[f"{column}_delta"]),
                "reason": reason,
                "electronic_delta": float(row["electronic_cont_delta"]),
                "electrostatic_delta": float(row["electrostatic_cont_delta"]),
                "lumo_delta": float(row["lumo_cont_delta"]),
                "ddg_expt": (
                    float(row["ΔΔG.expt."])
                    if "ΔΔG.expt." in row.index and pd.notna(row["ΔΔG.expt."])
                    else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def _plot_skeleton_violin_grid(
    delta_df: pd.DataFrame,
    save_path: str | Path,
    title: str,
    highlights: pd.DataFrame | None = None,
) -> None:
    """Render three horizontal skeleton-group contribution violins.

    ``delta_df`` follows :func:`_prepare_skeleton_contribution_delta` and may be
    paired with highlight rows from :func:`_skeleton_highlight_rows`.
    Contributions are plotted in kcal/mol and the figure is written to
    ``save_path``.
    """
    groups = [
        group
        for group in ["A", "B", "C", "D", "E"]
        if (delta_df["skeleton_group"] == group).any()
    ]
    if not groups:
        raise ValueError("No skeleton groups are available for plotting.")

    all_values = np.concatenate(
        [
            delta_df[f"{column}_delta"].to_numpy(dtype=float)
            for _, column, _ in CONTRIBUTION_VIOLIN_COMPONENTS
        ]
    )
    all_values = all_values[np.isfinite(all_values)]
    if all_values.size == 0:
        raise ValueError("No finite contribution-difference values were found.")
    x_lim = np.ceil(max(1.0, float(np.nanmax(np.abs(all_values)))) * 1.15)

    fig_width = 7.4 if highlights is not None and not highlights.empty else 6.8
    fig_height = 6.1 if highlights is not None and not highlights.empty else 5.8
    fig, axes = plt.subplots(
        nrows=3,
        ncols=1,
        figsize=(fig_width, fig_height),
        sharex=True,
        facecolor="white",
    )
    positions = np.arange(len(groups), 0, -1, dtype=float)
    group_y = dict(zip(groups, positions))

    label_offsets = {
        ("Electronic", "A24"): (0.12, 0.20, "left"),
        ("Electronic", "B6"): (-0.12, -0.20, "right"),
        ("Electronic", "C3"): (-0.12, 0.20, "right"),
        ("Electronic", "D11(trans)"): (-0.12, -0.20, "right"),
        ("Electronic", "E2(exo)"): (-0.12, 0.20, "right"),
        ("Electrostatic", "A12"): (-0.12, 0.20, "right"),
        ("Electrostatic", "B5"): (-0.12, 0.20, "right"),
        ("Electrostatic", "C10"): (0.12, -0.20, "left"),
        ("Electrostatic", "D13(cis)"): (0.12, 0.20, "left"),
        ("Electrostatic", "E3(exo)"): (0.12, 0.20, "left"),
        ("LUMO", "A16"): (0.12, 0.20, "left"),
        ("LUMO", "B5"): (0.12, 0.20, "left"),
        ("LUMO", "C12"): (-0.12, 0.20, "right"),
        ("LUMO", "D11(trans)"): (-0.12, -0.20, "right"),
        ("LUMO", "E4"): (-0.12, 0.20, "right"),
    }

    for ax, (feature, column, feature_color) in zip(
        axes,
        CONTRIBUTION_VIOLIN_COMPONENTS,
    ):
        data = [
            delta_df.loc[
                delta_df["skeleton_group"] == group,
                f"{column}_delta",
            ].to_numpy(dtype=float)
            for group in groups
        ]
        parts = ax.violinplot(
            data,
            positions=positions,
            vert=False,
            widths=0.72,
            showmeans=False,
            showmedians=True,
            showextrema=False,
        )
        for body, group in zip(parts["bodies"], groups):
            body.set_facecolor(SKELETON_GROUP_COLORS[group])
            body.set_edgecolor(SKELETON_GROUP_COLORS[group])
            body.set_alpha(0.20 if highlights is not None else 0.25)
            body.set_linewidth(0.8)
        parts["cmedians"].set_color("black")
        parts["cmedians"].set_linewidth(1.1)

        for pos, group in zip(positions, groups):
            values = delta_df.loc[
                delta_df["skeleton_group"] == group,
                f"{column}_delta",
            ].to_numpy(dtype=float)
            if len(values) == 0:
                continue
            jitter = np.linspace(-0.18, 0.18, len(values))
            if len(values) == 1:
                jitter = np.array([0.0])
            rng = np.random.default_rng(seed=100 + ord(group) + len(feature))
            rng.shuffle(jitter)
            ax.scatter(
                values,
                np.full(len(values), pos) + jitter,
                s=13 if highlights is not None else 16,
                color=SKELETON_GROUP_COLORS[group],
                edgecolor="white",
                linewidth=0.3 if highlights is not None else 0.35,
                alpha=0.45 if highlights is not None else 0.78,
                zorder=3,
            )

        if highlights is not None and not highlights.empty:
            highlight_rows = highlights[highlights["feature"] == feature]
            for _, row in highlight_rows.iterrows():
                group = str(row["group"])
                if group not in group_y:
                    continue
                entry = str(row["entry"])
                value = float(row["value"])
                y = group_y[group]
                ax.scatter(
                    value,
                    y,
                    s=42,
                    color=SKELETON_GROUP_COLORS[group],
                    edgecolor="black",
                    linewidth=0.6,
                    zorder=5,
                )
                dx, dy, ha = label_offsets.get(
                    (feature, entry),
                    (0.12 if value >= 0 else -0.12, 0.20, "left" if value >= 0 else "right"),
                )
                label = (
                    entry.replace("(trans)", " trans")
                    .replace("(cis)", " cis")
                    .replace("(exo)", " exo")
                )
                ax.text(
                    value + dx,
                    y + dy,
                    f"{label} {value:+.2f}",
                    ha=ha,
                    va="center",
                    fontsize=6.7,
                    color=SKELETON_GROUP_COLORS[group],
                    zorder=6,
                )

        ax.axvline(0, color="gray", linestyle="--", linewidth=0.9, alpha=0.75)
        ax.grid(True, axis="x", linestyle=":", linewidth=0.65, alpha=0.45)
        ax.set_xlim(-x_lim, x_lim)
        ax.set_yticks(positions)
        ax.set_yticklabels(
            [
                f"{group}\n(n={len(data_i)})"
                for group, data_i in zip(groups, data)
            ]
        )
        ax.set_ylabel(
            feature,
            color=feature_color,
            rotation=0,
            ha="right",
            va="center",
            labelpad=58,
        )
        for spine in ["left", "right", "top"]:
            ax.spines[spine].set_visible(False)

    axes[0].set_title(title, fontsize=11)
    axes[-1].set_xlabel("Contribution difference from benzophenone [kcal/mol]")
    fig.tight_layout(h_pad=1.0)
    save_path = _ensure_output_dir(save_path)
    fig.savefig(save_path, dpi=500, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)


def plot_group_contribution_violins_by_skeleton(
    df: pd.DataFrame,
    save_path: str | Path,
    highlighted_save_path: str | Path | None = None,
    ref_inchikey: str = BENZOPHENONE_REF_INCHIKEY,
    values_csv_path: str | Path | None = None,
    summary_csv_path: str | Path | None = None,
    highlights_csv_path: str | Path | None = None,
    train_value: int = 0,
) -> pd.DataFrame:
    """Plot A-E skeleton-group violin plots for training contribution deltas.

    Values are electronic/electrostatic/LUMO regression contributions shifted by
    the benzophenone reference contribution. The function returns the compact
    contribution-difference table used for plotting.
    """
    delta_df = _prepare_skeleton_contribution_delta(
        df,
        ref_inchikey=ref_inchikey,
        train_value=train_value,
    )
    compact_columns = [
        column
        for column in [
            "entry",
            "name",
            "SMILES",
            "InChIKey",
            "skeleton_group",
            "ΔΔG.expt.",
            "electronic_cont",
            "electrostatic_cont",
            "lumo_cont",
            "electronic_cont_delta",
            "electrostatic_cont_delta",
            "lumo_cont_delta",
        ]
        if column in delta_df.columns
    ]
    compact_df = delta_df[compact_columns].copy()

    if values_csv_path is not None:
        compact_df.to_csv(_ensure_output_dir(values_csv_path), index=False)
    if summary_csv_path is not None:
        _skeleton_contribution_summary(delta_df).to_csv(
            _ensure_output_dir(summary_csv_path),
            index=False,
        )

    _plot_skeleton_violin_grid(
        delta_df,
        save_path,
        "Contribution distributions by substrate-skeleton group",
    )

    highlights = _skeleton_highlight_rows(delta_df)
    if highlights_csv_path is not None:
        highlights.to_csv(_ensure_output_dir(highlights_csv_path), index=False)
    if highlighted_save_path is not None:
        _plot_skeleton_violin_grid(
            delta_df,
            highlighted_save_path,
            "Characteristic points on substrate-skeleton contribution distributions",
            highlights=highlights,
        )

    return compact_df


def _mol_from_smiles_for_grid(smiles: object):
    """Build an RDKit molecule with computed 2D coordinates, or return None."""
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is not None:
        Chem.rdDepictor.Compute2DCoords(mol)
    return mol


def plot_group_contribution_skeleton_highlight_structures(
    df: pd.DataFrame,
    save_path: str | Path,
    ref_inchikey: str = BENZOPHENONE_REF_INCHIKEY,
    train_value: int = 0,
    mols_per_row: int = 5,
    sub_img_size: tuple[int, int] = (520, 430),
    legend_font_size: int = 28,
    legend_fraction: float = 0.24,
) -> pd.DataFrame:
    """Draw structures corresponding to the highlighted skeleton violin points."""
    if "SMILES" not in df.columns:
        raise ValueError("DataFrame must contain column 'SMILES'.")

    delta_df = _prepare_skeleton_contribution_delta(
        df,
        ref_inchikey=ref_inchikey,
        train_value=train_value,
    )
    highlights = _skeleton_highlight_rows(delta_df)
    if highlights.empty:
        raise ValueError("No highlighted skeleton rows were available.")

    smiles_by_entry = delta_df.set_index("entry")["SMILES"].to_dict()
    highlights["SMILES"] = highlights["entry"].map(smiles_by_entry)
    highlights["feature_order"] = highlights["feature"].map(
        {"Electronic": 0, "Electrostatic": 1, "LUMO": 2}
    )
    highlights["group_order"] = highlights["group"].map(
        {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}
    )
    highlights = highlights.sort_values(["feature_order", "group_order"]).copy()

    mols = [_mol_from_smiles_for_grid(smiles) for smiles in highlights["SMILES"]]
    legends = [
        (
            f"{row['feature']} / group {row['group']}\n"
            f"{row['entry']}: {row['value']:+.2f} kcal/mol\n"
            f"{row['name']}"
        )
        for _, row in highlights.iterrows()
    ]
    image = Draw.MolsToGridImage(
        mols,
        molsPerRow=mols_per_row,
        subImgSize=sub_img_size,
        legends=legends,
        useSVG=False,
        legendFontSize=legend_font_size,
        legendFraction=legend_fraction,
    )
    save_path = _ensure_output_dir(save_path)
    image.save(save_path)
    return highlights


def _draw_horizontal_arrow(
    ax,
    base: float,
    val: float,
    y: float,
    height: float,
    color: str,
    alpha: float,
    span: float,
) -> None:
    """Draw a filled horizontal arrow from `base` to `base + val` at y = `y`.

    The maximum thickness of the arrow body is given by `height`.
    A small overlap (≈0.002 * span) is introduced between the body and head
    to remove any gap between them.
    """
    if val == 0:
        return

    x_start = base
    x_end = base + val
    direction = np.sign(val)
    length = abs(val)

    # Fixed arrow-head length in data units
    head_len = 0.1
    body_len = length - head_len

    # Very short arrows: triangle only
    if body_len <= 0:
        base_x = x_start
        head = Polygon(
            [
                (x_end, y),                    # tip
                (base_x, y + height / 2.0),    # upper base
                (base_x, y - height / 2.0),    # lower base
            ],
            closed=True,
            facecolor=color,
            edgecolor="none",
            alpha=alpha,
        )
        ax.add_patch(head)
        return

    # Normal case: rectangle body + triangle head
    if direction > 0:
        body_x0 = x_start
        body_x1 = x_start + body_len
        head_tip_x = x_end
    else:
        body_x0 = x_start - body_len
        body_x1 = x_start
        head_tip_x = x_end

    # Arrow body (rectangle)
    rect_x = min(body_x0, body_x1)
    rect_w = abs(body_x1 - body_x0)
    body = Rectangle(
        (rect_x, y - height / 2.0),
        rect_w,
        height,
        facecolor=color,
        edgecolor="none",
        alpha=alpha,
    )
    ax.add_patch(body)

    # Overlap between body and head to avoid any visual gap
    overlap = min(0.0002 * span, body_len * 0.5, head_len * 0.5)

    if direction > 0:
        base_x = body_x1 - overlap
    else:
        base_x = body_x0 + overlap

    # Arrow head (triangle)
    head = Polygon(
        [
            (head_tip_x, y),                    # tip
            (base_x, y + height / 2.0),         # upper base
            (base_x, y - height / 2.0),         # lower base
        ],
        closed=True,
        facecolor=color,
        edgecolor="none",
        alpha=alpha,
    )
    ax.add_patch(head)


def plot_pair_stacked_contributions(
    df: pd.DataFrame,
    target_inchikey: str,
    ref_inchikey: str | None,
    save_path: str,
    baseline: str = "reference",
    mean_scope: str = "train",
    train_value: int = 0,
) -> None:
    """
    Plot component contribution differences for one target molecule.

    Baseline modes
    --------------
    - ``baseline="reference"``: target - reference InChIKey contribution.
      This is the original behavior and requires ``ref_inchikey``.
    - ``baseline="mean"``: target - mean contribution. By default, the mean is
      computed from training rows (``test == train_value``); set
      ``mean_scope="all"`` to use all rows.

    Arrow structure (from top to bottom):
        - top   : arrow from 0 to electronic
        - middle: arrow from electronic to electronic + electrostatic
        - bottom: arrow from electronic + electrostatic to
                  electronic + electrostatic + lumo

    -> The tip of the lumo arrow corresponds to the total contribution
       (electronic + electrostatic + lumo).

    Spec:
    - Horizontal arrows instead of barh.
    - Each arrow has a thick filled shaft.
    - Contribution values are displayed with explicit signs (+X.XX / -X.XX),
      right-aligned, next to the labels on the left (so the three numbers
      line up vertically).
    - x-axis is roughly symmetric so that 0 is near the center, with slightly
      more space on the left for labels and values.
    - A dashed vertical line is drawn at x = 0.
    - Colors:
        electronic    : darkmagenta
        electrostatic : forestgreen
        lumo          : saddlebrown
    """

    required_cols = [
        "InChIKey",
        "electronic_cont",
        "electrostatic_cont",
        "lumo_cont",
    ]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"DataFrame must contain column '{col}'.")

    if baseline not in {"reference", "mean"}:
        raise ValueError("baseline must be either 'reference' or 'mean'.")
    if mean_scope not in {"train", "all"}:
        raise ValueError("mean_scope must be either 'train' or 'all'.")

    # --- fetch target row and baseline contribution values ---
    if target_inchikey not in df["InChIKey"].values:
        raise ValueError(f"target_inchikey '{target_inchikey}' was not found in DataFrame.")

    row_target = df[df["InChIKey"] == target_inchikey].iloc[0]

    target_vals = np.array(
        [
            float(row_target["electronic_cont"]),
            float(row_target["electrostatic_cont"]),
            float(row_target["lumo_cont"]),
        ]
    )

    if baseline == "reference":
        if ref_inchikey is None:
            raise ValueError("ref_inchikey is required when baseline='reference'.")
        if ref_inchikey not in df["InChIKey"].values:
            raise ValueError(f"ref_inchikey '{ref_inchikey}' was not found in DataFrame.")
        row_ref = df[df["InChIKey"] == ref_inchikey].iloc[0]
        baseline_vals = np.array(
            [
                float(row_ref["electronic_cont"]),
                float(row_ref["electrostatic_cont"]),
                float(row_ref["lumo_cont"]),
            ]
        )
        x_label = "Contribution difference from reference [kcal/mol]"
    else:
        baseline_df = df
        if mean_scope == "train":
            if "test" not in df.columns:
                raise ValueError("DataFrame must contain column 'test' when mean_scope='train'.")
            test_values = pd.to_numeric(df["test"], errors="coerce")
            baseline_df = df[test_values == train_value]
        baseline_df = baseline_df.dropna(
            subset=["electronic_cont", "electrostatic_cont", "lumo_cont"]
        )
        if baseline_df.empty:
            raise ValueError("No rows were available to compute the mean baseline.")
        baseline_vals = baseline_df[
            ["electronic_cont", "electrostatic_cont", "lumo_cont"]
        ].to_numpy(dtype=float).mean(axis=0)
        x_label = "Contribution difference from training mean [kcal/mol]"
        if mean_scope == "all":
            x_label = "Contribution difference from dataset mean [kcal/mol]"

    contrib = target_vals - baseline_vals
    elec, es, lumo = contrib

    # cumulative positions
    s1 = elec
    s2 = elec + es
    s3 = elec + es + lumo  # total

    # --- x-range (start symmetric around 0, then add a little left space) ---
    core = np.max(np.abs([s1, s2, s3]))
    if core == 0:
        core = 1.0  # fallback

    base_margin_ratio = 0.5
    half_width = core * (1.0 + base_margin_ratio)

    # initial symmetric limits
    x_min = -half_width
    x_max = +half_width

    # slightly smaller extra left space than before
    base_span = x_max - x_min
    extra_left = 0.25 * base_span
    x_min = x_min - extra_left

    # final span
    span = x_max - x_min

    # --- figure (landscape) ---
    fig, ax = plt.subplots(figsize=(3.8, 1.4))

    # y positions (top to bottom)
    y_elec = 2.0
    y_es   = 1.0
    y_lumo = 0.0
    y_pos  = [y_elec, y_es, y_lumo]
    y_labels = ["Electronic", "Electrostatic", "LUMO"]

    # colors
    color_elec = "darkmagenta"
    color_es   = "forestgreen"
    color_lumo = "saddlebrown"

    # arrow thickness: revert to original height = 1.0
    arrow_height = 1.0

    # --- arrows ---
    _draw_horizontal_arrow(
        ax,
        base=0.0,
        val=elec,
        y=y_elec,
        height=arrow_height,
        color=color_elec,
        alpha=0.8,
        span=span,
    )

    _draw_horizontal_arrow(
        ax,
        base=s1,
        val=es,
        y=y_es,
        height=arrow_height,
        color=color_es,
        alpha=0.8,
        span=span,
    )

    _draw_horizontal_arrow(
        ax,
        base=s2,
        val=lumo,
        y=y_lumo,
        height=arrow_height,
        color=color_lumo,
        alpha=0.8,
        span=span,
    )

    # --- axes settings ---
    y_min_plot = -0.5
    y_max_plot = 2.5
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min_plot, y_max_plot)

    # remove y-axis but keep labels as text
    ax.yaxis.set_visible(False)
    ax.spines["left"].set_visible(False)

    # label position (left) and value position (slightly to the right, but
    # closer to the axis label than before)
    label_x = x_min + 0.015 * span
    value_right_x = label_x + 0.22 * span  # closer to labels than previous 0.30

    for ypos, label in zip(y_pos, y_labels):
        ax.text(
            label_x,
            ypos,
            label,
            va="center",
            ha="right",
            # fontsize=8,
        )

    # --- annotate contribution values (right-aligned, vertically aligned) ---
    def annotate_value(val: float, y: float) -> None:
        """Draw a signed contribution value at the requested vertical position."""
        ax.text(
            value_right_x,
            y,
            f"{val:+.2f}".replace("-", "−"),   # always show sign (+X.XX / -X.XX)
            va="center",
            ha="right",      # right-aligned
            # fontsize=8,
        )

    annotate_value(elec, y_elec)
    annotate_value(es,   y_es)
    annotate_value(lumo, y_lumo)

    # x-axis label
    ax.set_xlabel(x_label)

    # ticks only at 0 and total (s3)
    xticks = sorted(set([0.0, s3]))
    ax.set_xticks(xticks)
    ax.set_xticklabels([f"{x:.2f}".replace("-", "−") for x in xticks])

    # dashed line at x=0
    ax.axvline(0.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
    # Vertical line at total contribution, up to one-third of axis height.
    ax.axvline(s3, ymax=1/3, color="red", linestyle="-", linewidth=1.0, alpha=0.9)
    # remove box edges (bottom will be replaced by arrow axis)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)

    # --- draw main horizontal axis as a right-pointing arrow ---
    arrow_y = y_min_plot
    ax.annotate(
        "",
        xy=(x_max, arrow_y),
        xytext=(x_min, arrow_y),
        arrowprops=dict(arrowstyle="->", lw=1.0, color="black"),
    )

    # title (comment kept as requested)
    # ax.set_title(
    #     f"Contributions: {target_inchikey} − {ref_inchikey}",
    #     fontsize=9,
    # )

    fig.tight_layout()

    save_path = _ensure_output_dir(save_path)

    fig.savefig(save_path, dpi=400)
    plt.close(fig)


def _safe_entry_filename(entry: str) -> str:
    """Convert an entry label into a stable, portable filename token."""
    name = (
        entry.replace("(trans)", "_trans")
        .replace("(cis)", "_cis")
        .replace("(exo)", "_exo")
    )
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_")


def plot_selected_contribution_breakdowns(
    df: pd.DataFrame,
    output_dir: str | Path = "data/validation",
    entries: tuple[str, ...] = SELECTED_CONTRIBUTION_BREAKDOWN_ENTRIES,
    ref_inchikey: str = BENZOPHENONE_REF_INCHIKEY,
) -> pd.DataFrame:
    """Plot benzophenone-referenced contribution breakdowns for selected entries."""
    required_cols = ["entry", "name", "InChIKey"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"DataFrame must contain column '{col}'.")

    output_dir = Path(output_dir)
    rows: list[dict[str, object]] = []
    by_entry = df.set_index("entry", drop=False)
    for entry in entries:
        if entry not in by_entry.index:
            raise ValueError(f"entry '{entry}' was not found in DataFrame.")
        row = by_entry.loc[entry]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        save_path = output_dir / f"{_safe_entry_filename(entry)}_contribution_breakdown.png"
        plot_pair_stacked_contributions(
            df,
            target_inchikey=str(row["InChIKey"]),
            ref_inchikey=ref_inchikey,
            save_path=str(save_path),
        )
        rows.append(
            {
                "entry": entry,
                "name": row["name"],
                "InChIKey": row["InChIKey"],
                "save_path": str(save_path),
            }
        )
    return pd.DataFrame(rows)


def plot_component_contribution_series(
    df: pd.DataFrame,
    component: str,
    entries: list[str],
    reference_entry: str,
    save_path: str | Path,
    title: str,
    *,
    contribution_column: str | None = None,
    target_column: str = "ΔΔG.expt.",
    highlighted_names: tuple[str, ...] = (),
    highlighted_labels: dict[str, str] | None = None,
    series_label: str = "Series",
    highlight_label: str = "Highlighted substrates",
    xlabel: str | None = None,
    ylabel: str | None = None,
    show_grid: bool = True,
    avoid_label_overlap: bool = False,
    reference_label: str | None = None,
    regression_excluded_entries: tuple[str, ...] = (),
    excluded_label: str = "Excluded from linear fit",
    label_fontsize: float = 7.5,
    figure_size: tuple[float, float] = (5.3, 4.2),
    regression_line_color: str = "0.2",
    square_axes: bool = False,
    equal_axis_scale: bool = False,
    x_tick_step: float | None = None,
    y_tick_step: float | None = None,
    swap_axes: bool = False,
    label_fontweight: str = "normal",
    ax=None,
) -> pd.DataFrame:
    """Plot experimental energy changes against one contribution component.

    Entries are compared with ``reference_entry``.  Named substrates can be
    overlaid with a distinct marker, which is useful when extending a series
    with structurally related training examples that have separate IDs.  Both
    axes are changes in kcal/mol.  ``regression_excluded_entries`` remain
    visible but are omitted from the fitted line and correlation.  When ``ax``
    is supplied the caller owns figure saving/closing; otherwise ``save_path``
    is written directly.  The returned rows include the centered values and a
    ``used_for_linear_fit`` flag.
    """
    contribution_column = contribution_column or f"{component}_contribution"
    required = {"entry", "name", contribution_column, target_column}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"DataFrame is missing required columns: {', '.join(missing)}")

    highlighted_labels = highlighted_labels or {}
    selected = (
        df["entry"].astype(str).isin(entries)
        | df["name"].isin(highlighted_names)
        | df["entry"].astype(str).eq(reference_entry)
    )
    subset = df.loc[selected].copy()
    reference_rows = subset.loc[subset["entry"].astype(str).eq(reference_entry)]
    if reference_rows.empty:
        raise ValueError(f"Reference entry '{reference_entry}' was not found in the selected data.")

    reference = reference_rows.iloc[0]
    subset["contribution_change"] = (
        subset[contribution_column] - float(reference[contribution_column])
    )
    subset["experimental_change"] = (
        subset[target_column] - float(reference[target_column])
    )
    if swap_axes:
        x_column, y_column = "experimental_change", "contribution_change"
    else:
        x_column, y_column = "contribution_change", "experimental_change"
    points = subset.loc[subset["entry"].astype(str).ne(reference_entry)].copy()
    excluded_from_fit = points["entry"].astype(str).isin(regression_excluded_entries)
    fit_points = points.loc[~excluded_from_fit].copy()
    if len(fit_points) < 3:
        raise ValueError("At least three non-reference points are required for a component series plot.")

    slope, intercept = np.polyfit(fit_points[x_column], fit_points[y_column], 1)
    correlation = float(np.corrcoef(fit_points[x_column], fit_points[y_column])[0, 1])
    highlighted = points["name"].isin(highlighted_names) & ~excluded_from_fit
    base_points = points.loc[~highlighted & ~excluded_from_fit]
    highlighted_points = points.loc[highlighted]
    excluded_points = points.loc[excluded_from_fit]

    owns_figure = ax is None
    if owns_figure:
        fig, ax = plt.subplots(figsize=figure_size)
    else:
        fig = ax.figure
    ax.axhline(0, color="0.7", linewidth=0.8)
    ax.axvline(0, color="0.7", linewidth=0.8)
    ax.scatter(
        base_points[x_column], base_points[y_column], color="#4e79a7", s=48,
        edgecolors="black", linewidths=0.5, zorder=3, label=series_label,
    )
    if not highlighted_points.empty:
        ax.scatter(
            highlighted_points[x_column], highlighted_points[y_column],
            color="#f28e2b", marker="D", s=62, edgecolors="black",
            linewidths=0.55, zorder=4, label=highlight_label,
        )
    if not excluded_points.empty:
        ax.scatter(
            excluded_points[x_column], excluded_points[y_column],
            color="#d9534f", marker="X", s=72, edgecolors="black",
            linewidths=0.55, zorder=4, label=excluded_label,
        )
    ax.scatter([0], [0], marker="*", color="black", s=82, zorder=5)

    x_line = np.linspace(
        float(fit_points[x_column].min()) - 0.08,
        float(fit_points[x_column].max()) + 0.08,
        100,
    )
    ax.plot(
        x_line,
        slope * x_line + intercept,
        color=regression_line_color,
        linewidth=1.1,
        zorder=0,
    )
    ax.text(
        0.04, 0.96,
        rf"$R^2$ = {correlation ** 2:.2f}, N = {len(fit_points)}",
        transform=ax.transAxes, va="top", fontsize=11,
    )
    ax.set(
        title=title,
        xlabel=xlabel or f"{component.capitalize()} contribution change vs {reference_entry} [kcal/mol]",
        ylabel=ylabel or rf"Experimental $\Delta\Delta G^\ddagger$ change vs {reference_entry} [kcal/mol]",
    )
    if x_tick_step is not None:
        ax.xaxis.set_major_locator(MultipleLocator(x_tick_step))
    if y_tick_step is not None:
        ax.yaxis.set_major_locator(MultipleLocator(y_tick_step))
    if equal_axis_scale:
        ax.set_aspect("equal", adjustable="box")
    elif square_axes:
        ax.set_box_aspect(1)
    labels = [
        (
            float(row[x_column]),
            float(row[y_column]),
            highlighted_labels.get(str(row["name"]), str(row["entry"])),
        )
        for _, row in points.iterrows()
    ]
    if reference_label:
        labels.append((0.0, 0.0, reference_label))
    if avoid_label_overlap:
        # Greedily choose a nearby offset with the least display-space overlap.
        # Labels are short entry IDs, so this deterministic placement is more
        # stable across machines than a force-based layout.
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        occupied = []
        for x_value, y_value, _ in labels:
            display_x, display_y = ax.transData.transform((x_value, y_value))
            occupied.append(Bbox.from_bounds(display_x - 6, display_y - 6, 12, 12))
        offsets = (
            (5, 5), (5, -12), (-20, 5), (-20, -12),
            (6, 15), (-22, 15), (6, -22), (-22, -22),
            (24, 3), (-38, 3), (24, -13), (-38, -13),
            (24, 16), (-38, 16), (24, -25), (-38, -25),
        )
        for x_value, y_value, label in sorted(labels, key=lambda item: (item[1], item[0])):
            best_offset = offsets[0]
            best_score = float("inf")
            for offset in offsets:
                trial = ax.annotate(
                    label, (x_value, y_value), xytext=offset,
                    textcoords="offset points", fontsize=label_fontsize,
                    fontweight=label_fontweight,
                )
                fig.canvas.draw()
                box = trial.get_window_extent(renderer=renderer).expanded(1.08, 1.18)
                overlap = sum(
                    max(0.0, min(box.x1, other.x1) - max(box.x0, other.x0))
                    * max(0.0, min(box.y1, other.y1) - max(box.y0, other.y0))
                    for other in occupied
                )
                outside = (
                    max(0.0, ax.bbox.x0 - box.x0)
                    + max(0.0, box.x1 - ax.bbox.x1)
                    + max(0.0, ax.bbox.y0 - box.y0)
                    + max(0.0, box.y1 - ax.bbox.y1)
                )
                score = overlap + 1000.0 * outside + 0.01 * (offset[0] ** 2 + offset[1] ** 2)
                trial.remove()
                if score < best_score:
                    best_score = score
                    best_offset = offset
            annotation = ax.annotate(
                label,
                (x_value, y_value),
                xytext=best_offset,
                textcoords="offset points",
                fontsize=label_fontsize,
                fontweight=label_fontweight,
                arrowprops={"arrowstyle": "-", "color": "0.55", "lw": 0.45},
            )
            fig.canvas.draw()
            occupied.append(annotation.get_window_extent(renderer=renderer).expanded(1.08, 1.18))
    else:
        for x_value, y_value, label in labels:
            ax.annotate(
                label, (x_value, y_value), xytext=(4, 3),
                textcoords="offset points", fontsize=label_fontsize,
                fontweight=label_fontweight,
            )
    if not highlighted_points.empty or not excluded_points.empty:
        ax.legend(
            frameon=False,
            fontsize=9,
            loc="upper left",
            bbox_to_anchor=(0.025, 0.885),
            borderaxespad=0,
        )
    ax.title.set_fontsize(13)
    ax.xaxis.label.set_fontsize(11.5)
    ax.yaxis.label.set_fontsize(11.5)
    ax.tick_params(axis="both", labelsize=10.5)
    if show_grid:
        ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.4)
    else:
        ax.grid(False)
    if owns_figure:
        fig.tight_layout()
        fig.savefig(_ensure_output_dir(save_path), dpi=500)
        plt.close(fig)
    subset["used_for_linear_fit"] = (
        subset["entry"].astype(str).ne(reference_entry)
        & ~subset["entry"].astype(str).isin(regression_excluded_entries)
    )
    return subset


def save_non_diketone_test_predictions(
    df: pd.DataFrame,
    save_path: str | Path = "data/validation/non_diketone_test_predictions.csv",
    train_value: int = 0,
) -> pd.DataFrame:
    """Save non-diketone holdout predictions such as Dxx and H-series substrates."""
    required_cols = ["entry", "name", "InChIKey", "ΔΔG.expt.", "test", "prediction"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"DataFrame must contain column '{col}'.")

    entry = df["entry"].astype(str)
    test_values = pd.to_numeric(df["test"], errors="coerce")
    mask = (test_values != train_value) & ~entry.str.match(r"^[a-f]\d+$", na=False)
    columns = [
        "entry",
        "name",
        "InChIKey",
        "ΔΔG.expt.",
        "prediction",
        "prediction_error",
        "electronic_cont",
        "electrostatic_cont",
        "lumo_cont",
    ]
    output = df.loc[mask, [col for col in columns if col in df.columns]].copy()
    output.to_csv(_ensure_output_dir(save_path), index=False)
    return output


DIKETONE_ENTRY_ORDER = ("1", "2", "3", "4", "13", "14", "23", "24", "31", "32", "41", "42")
DIKETONE_TEMPERATURES = {
    "a": 273.15,
    "b": 298.15,
    "c": 298.15,
    "d": 298.15,
    "e": 298.15,
    "f": 298.15,
}
DIKETONE_INITIAL_EXPECTED = {"a": "2", "b": "1", "c": "2", "d": "1", "e": "2", "f": "1"}
DIKETONE_FINAL_EXPECTED = {"a": "2-4", "e": "2-3"}


def _diketone_rate(delta_g: float, temperature: float) -> float:
    """Convert a barrier in kcal/mol to a relative Eyring rate."""
    return float(np.exp(-delta_g / (1.987e-3 * temperature)))


def _simulate_diketone_selectivity(
    pred_by_entry: dict[str, float],
    group: str,
) -> dict[str, dict[str, float]]:
    """Delegate diketone kinetics to the shared, equal-rate-safe implementation."""
    try:
        from .diketone_metrics import simulate_full  # noqa: PLC0415
    except ImportError:
        # ``current_model.py`` also supports legacy execution with ``libs``
        # directly on sys.path, where ``graph`` has no package parent.
        from diketone_metrics import simulate_full  # type: ignore[no-redef]  # noqa: PLC0415

    return simulate_full(pred_by_entry, group)


def save_diketone_selectivity_summary(
    df: pd.DataFrame,
    save_path: str | Path = "data/test/selectivity_summary.csv",
) -> pd.DataFrame:
    """Save the eight manuscript diketone selectivity checks."""
    pred_by_entry = {
        str(entry): float(value)
        for entry, value in zip(df["entry"].astype(str), df["prediction"])
        if str(entry)[:1] in set("abcdef") and pd.notna(value)
    }

    rows: list[dict[str, object]] = []
    for group, expected in DIKETONE_INITIAL_EXPECTED.items():
        values = _simulate_diketone_selectivity(pred_by_entry, group)["intermediate_abs"]
        predicted = max(values, key=values.get)
        rows.append(
            {
                "group": group,
                "stage": "initial",
                "expected": expected,
                "predicted": predicted,
                "ok": predicted == expected,
                "target_percent": values[expected],
            }
        )
    for group, expected in DIKETONE_FINAL_EXPECTED.items():
        values = _simulate_diketone_selectivity(pred_by_entry, group)["final_frac"]
        predicted = max(values, key=values.get)
        rows.append(
            {
                "group": group,
                "stage": "final",
                "expected": expected,
                "predicted": predicted,
                "ok": predicted == expected,
                "target_percent": values[expected],
            }
        )

    summary = pd.DataFrame(rows)
    summary.to_csv(_ensure_output_dir(save_path), index=False)
    return summary


def make_cube(df, path):
    """Write legacy per-substrate contribution cubes for three field blocks.

    ``df`` must contain ``InChIKey``, ``ΔΔG.expt.``, ``temperature``, and
    ``electronic_cont x y z``, ``electrostatic_cont x y z``, and
    ``lumo_cont x y z`` columns.  Atom records are copied from the first
    matching density cube under ``OUTPUT_ROOT`` and three files are written
    below ``path/<InChIKey>``.  This routine preserves the historical cube
    layout; new adopted-model exports use the dedicated export script.
    """
    grid = np.array([re.findall(r'[+-]?\d+', col) for col in df.filter(like='electronic_cont ').columns]).astype(int)
    min=np.min(grid,axis=0).astype(int)
    print("min",min)
    max=np.max(grid,axis=0).astype(int)
    rang=max-min
    
    columns=["ΔΔG.expt.","temperature"]
    for x,y,z in product(range(min[0],max[0]+1),range(min[1],max[1]+1),range(min[2],max[2]+1)):
        if x!=0 and y!=0 and z!=0:
            columns.append(f'electronic_cont {x} {y} {z}')
    for x,y,z in product(range(min[0],max[0]+1),range(min[1],max[1]+1),range(min[2],max[2]+1)):
        if x!=0 and y!=0 and z!=0:
            columns.append(f'electrostatic_cont {x} {y} {z}')
    for x,y,z in product(range(min[0],max[0]+1),range(min[1],max[1]+1),range(min[2],max[2]+1)):
        if x!=0 and y!=0 and z!=0:
            columns.append(f'lumo_cont {x} {y} {z}')
    df=df.set_index("InChIKey").reindex(columns=columns, fill_value=0)
    n=0.52917721092*2
    # print(df.columns)
    out_path = CONTRIBUTIONS_ROOT
    min=' '.join(map(str, (min+np.array([0.5,0.5,-0.5]))*n))
    for inchikey,expt,temp,value in zip(df.index,df["ΔΔG.expt."],df["temperature"],df.iloc[:,2:].values):
        dt = glob.glob(str(OUTPUT_ROOT / inchikey / "Dt*.cube"))[0]
        with open(dt, 'r', encoding='UTF-8') as f:
            f.readline()
            f.readline()
            
            n_atom,x,y,z,_=f.readline().split()
            n_atom=int(n_atom)
            f.readline()
            f.readline()
            f.readline()
            coord=[f.readline() for _ in range(n_atom)]
        coord=''.join(coord)
        # print(len(value)//3)
        electronic='\n'.join([' '.join(f"{x}" for x in value[i:i + 6])for i in range(0, len(value)//3, 6)])
        electrostatic='\n'.join([' '.join(f"{x}" for x in value[i:i + 6])for i in range(len(value)//3, len(value)*2//3, 6)])
        lumo='\n'.join([' '.join(f"{x}" for x in value[i:i + 6])for i in range(len(value)*2//3, len(value), 6)])
        contribution=np.sum(value[:len(value)//3]),np.sum(value[len(value)//3:len(value)*2//3]),np.sum(value[len(value)*2//3:])
        pred=100/(1+np.exp(sum(contribution)/1.99/temp/0.001))
        os.makedirs(f'{path}/{inchikey}',exist_ok=True)
        with open(f'{path}/{inchikey}/electronic.cube','w') as f:
            print(f'contribution Gaussian Cube File.\nProperty: Shielding Density # color electronic {contribution[0]:.2f} predict {sum(contribution):.2f} expt {expt:.2f} pred {pred:.0f}\n{n_atom} {min}\n{rang[0]} {n} 0 0\n{rang[1]} 0 {n} 0\n{rang[2]} 0 0 {n}\n{coord}\n{electronic}',file=f)
        with open(f'{path}/{inchikey}/electrostatic.cube','w') as f:
            print(f'contribution Gaussian Cube File.\nProperty: Shielding Density # color electrostatic {contribution[1]:.2f} predict {sum(contribution):.2f} expt {expt:.2f} pred {pred:.0f}\n{n_atom} {min}\n{rang[0]} {n} 0 0\n{rang[1]} 0 {n} 0\n{rang[2]} 0 0 {n}\n{coord}\n{electrostatic}',file=f)
        with open(f'{path}/{inchikey}/lumo.cube','w') as f:
            print(f'contribution Gaussian Cube File.\nProperty: Shielding Density # color lumo {contribution[2]:.2f} predict {sum(contribution):.2f} expt {expt:.2f} pred {pred:.0f}\n{n_atom} {min}\n{rang[0]} {n} 0 0\n{rang[1]} 0 {n} 0\n{rang[2]} 0 0 {n}\n{coord}\n{lumo}',file=f)


def make_cube_with_sign_markers(df: pd.DataFrame, out_root: str | Path) -> None:
    """Write cube files with sign marker atoms for contribution grids.

    Marker rule:
    - Positive value -> atomic number 0 (X, dummy atom)
    - Negative value -> atomic number 54 (Xe)

    The function copies volumetric data from template cubes under
    ``OUTPUT_ROOT / <InChIKey>`` (Dt/ESP/LUMO) and appends marker atom lines
    computed from contribution columns:
    ``electronic_cont i j k``, ``electrostatic_cont i j k``, ``lumo_cont i j k``.
    """
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    # Collect contribution columns and corresponding grid indices (i, j, k).
    def extract_cont_columns(prefix: str):
        """Return matching contribution columns and their integer grid indices."""
        cols = [c for c in df.columns if c.startswith(prefix + " ")]
        if not cols:
            raise ValueError(f"No columns starting with '{prefix} ' were found in df.")
        # Parse (i, j, k) from column names: "prefix i j k".
        grid_idx = np.array(
            [list(map(int, re.findall(r"[+-]?\d+", c))) for c in cols],
            dtype=int,
        )
        return cols, grid_idx

    ele_cols, grid_idx = extract_cont_columns("electronic_cont")
    es_cols, _ = extract_cont_columns("electrostatic_cont")
    lu_cols, _ = extract_cont_columns("lumo_cont")

    # Assume grid indices are shared across electronic/electrostatic/lumo columns.
    if not (len(ele_cols) == len(es_cols) == len(lu_cols)):
        raise ValueError(
            "The numbers of electronic_cont / electrostatic_cont / lumo_cont "
            "columns do not match."
        )

    # Convert grid indices to Cartesian coordinates using cube origin/axes.
    def make_marker_atoms(
        grid_indices: np.ndarray,
        values: np.ndarray,
        origin: np.ndarray,
        ax: np.ndarray,
        ay: np.ndarray,
        az: np.ndarray,
        pos_Z: int = 0,   # X
        neg_Z: int = 54,  # Xe
        eps: float = 1e-12,
    ) -> list[str]:
        """Build X/Xe atom lines according to the sign of contribution values."""
        atoms: list[str] = []
        for (ix, iy, iz), v in zip(grid_indices, values):
            if np.isnan(v) or abs(v) <= eps:
                continue
            if v > 0:
                Z = pos_Z   # X: dummy atom
            else:
                Z = neg_Z   # Xe

            # Cell-center position (currently fixed at origin by existing logic).
            r = (
                origin
                # + (ix + 0.5) * ax
                # + (iy + 0.5) * ay
                # + (iz + 0.5) * az
            )
            x, y, z = r.tolist()

            # Gaussian cube atom line format:
            #   atomic_number  charge  x  y  z
            # Charge is set equal to atomic number for these marker atoms.
            line = f"{Z:5d}{float(Z):12.6f}{x:12.6f}{y:12.6f}{z:12.6f}"
            atoms.append(line)

        return atoms

    # Iterate rows indexed by InChIKey.
    df_idx = df.set_index("InChIKey")

    for inchikey, row in df_idx.iterrows():
        # Extract contribution values.
        ele_vals = row[ele_cols].to_numpy(dtype=float)
        es_vals  = row[es_cols].to_numpy(dtype=float)
        lu_vals  = row[lu_cols].to_numpy(dtype=float)

        # Find template cubes.
        base_dir = OUTPUT_ROOT / inchikey
        dt_candidates  = glob.glob(str(base_dir / "Dt*.cube"))
        esp_candidates = glob.glob(str(base_dir / "ESP*.cube"))
        lumo_candidates = glob.glob(str(base_dir / "LUMO*.cube"))

        cube_map = {
            "electronic": dt_candidates,
            "electrostatic": esp_candidates,
            "lumo": lumo_candidates,
        }

        for kind, candidates in cube_map.items():
            if not candidates:
                # Skip if template cube is missing.
                continue

            template_path = Path(candidates[0])

            # Read the entire template cube.
            with open(template_path, "r", encoding="utf-8") as f:
                lines = f.readlines()

            if len(lines) < 7:
                raise ValueError(f"Cube file seems too short: {template_path}")

            # Parse cube header.
            title_line   = lines[0].rstrip("\n")
            comment_line = lines[1].rstrip("\n")

            # Line 3: natoms and origin.
            natoms_tokens = lines[2].split()
            natoms = int(natoms_tokens[0])
            origin = np.array(list(map(float, natoms_tokens[1:4])), dtype=float)

            # Lines 4-6: grid definitions.
            gx = lines[3].split()
            gy = lines[4].split()
            gz = lines[5].split()

            nx = int(gx[0])
            ny = int(gy[0])
            nz = int(gz[0])

            ax = np.array(list(map(float, gx[1:4])), dtype=float)
            ay = np.array(list(map(float, gy[1:4])), dtype=float)
            az = np.array(list(map(float, gz[1:4])), dtype=float)

            # Atom block.
            atom_lines = [line.rstrip("\n") for line in lines[6 : 6 + natoms]]

            # Keep volumetric data unchanged.
            data_lines = [line.rstrip("\n") for line in lines[6 + natoms :]]

            # Select the matching contribution vector.
            if kind == "electronic":
                vals = ele_vals
            elif kind == "electrostatic":
                vals = es_vals
            elif kind == "lumo":
                vals = lu_vals
            else:
                continue

            # Build X/Xe marker atoms from contribution signs.
            marker_atoms = make_marker_atoms(
                grid_indices=grid_idx,
                values=vals,
                origin=origin,
                ax=ax,
                ay=ay,
                az=az,
                pos_Z=0,   # X
                neg_Z=54,  # Xe
            )

            # Update atom count in cube header.
            natoms_new = natoms + len(marker_atoms)
            natoms_line_new = (
                f"{natoms_new:5d}"
                f"{origin[0]:12.6f}{origin[1]:12.6f}{origin[2]:12.6f}"
            )

            # Assemble updated cube content.
            new_lines: list[str] = []
            new_lines.append(title_line)
            new_lines.append(comment_line)
            new_lines.append(natoms_line_new)
            new_lines.append(lines[3].rstrip("\n"))
            new_lines.append(lines[4].rstrip("\n"))
            new_lines.append(lines[5].rstrip("\n"))
            new_lines.extend(atom_lines)
            new_lines.extend(marker_atoms)
            new_lines.extend(data_lines)

            # Output path.
            out_dir = out_root / inchikey
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{kind}.cube"

            with open(out_path, "w", encoding="utf-8") as f_out:
                f_out.write("\n".join(new_lines) + "\n")


def plot_expt_vs_pred(df: pd.DataFrame, path: str) -> None:
    """Plot experimental vs predicted ΔΔG‡ (parity plot style)."""
    # figure & axis range
    plt.figure(figsize=(3, 3))
    plt.yticks([-4, 0, 4])
    plt.xticks([-4, 0, 4])
    plt.ylim(-4, 4)
    plt.xlim(-4, 4)

    # regression points
    plt.scatter(
        df["ΔΔG.expt."],
        df["regression"],
        c="black",
        linewidths=0,
        s=10,
        alpha=0.5,
    )
    rmse = nan_rmse(df["regression"].values, df["ΔΔG.expt."].values)
    r2 = nan_r2(df["regression"].values, df["ΔΔG.expt."].values)
    plt.scatter(
        [],
        [],
        label=(
            "regression: $r^2$ = "
            f"{r2:.2f} "#\n
            r"$\mathrm{RMSE}$"
            f" = {rmse:.2f} kcal/mol"
        ),
        c="black",
        linewidths=0,
        alpha=0.5,
        s=10,
    )

    # LOOCV points
    rmse = nan_rmse(df["cv"].values, df["ΔΔG.expt."].values)
    r2 = nan_r2(df["cv"].values, df["ΔΔG.expt."].values)
    plt.scatter(
        [],
        [],
        label=(
            "      LOOCV: $r^2$ = "
            f"{r2:.2f} "#\n
            r"$\mathrm{RMSE}$"
            f" = {rmse:.2f} kcal/mol"
        ),
        c="dodgerblue",
        linewidths=0,
        alpha=0.6,
        s=10,
    )

    plt.scatter(
        df["ΔΔG.expt."],
        df["cv"],
        c="dodgerblue",
        linewidths=0,
        s=10,
        alpha=0.6,
    )

    plt.xlabel(r"$\Delta\Delta G^{\ddagger}_{\mathrm{expt}}$ [kcal/mol]")
    plt.ylabel(r"$\Delta\Delta G^{\ddagger}_{\mathrm{predict}}$ [kcal/mol]")

    # plt.legend(
    #     loc="lower right",
    #     fontsize=6,
    #     ncol=1,
    #     borderpad=0.2,
    #     handletextpad=0.3,
    #     frameon=True,
    #     framealpha=0.8,
    # )

    # plt.text(
    #     -3.6,
    #     3.6,
    #     "$\mathit{N}$" + f' = {len(df[df["test"] == 0])}',
    #     fontsize=10,
    #     verticalalignment="top",
    # )

    # plt.tight_layout()
    # Apply layout first.
    plt.tight_layout()

    # Place legend outside, below x-axis.
    leg=plt.legend(
        loc="upper center",
        bbox_to_anchor=(0.35, -0.25),
        fontsize=8,
        ncol=1,
        borderpad=0.3,
        handletextpad=0.2,
        frameon=True,
        columnspacing=0.2,
        framealpha=0.8,
    )
    # Right-align legend text.
    for txt in leg.get_texts():
        txt.set_ha("right")
        txt.set_multialignment("right")

    # Add bottom margin so the legend is not clipped.
    plt.subplots_adjust(bottom=0.32)

    plt.text(
        -3.6,
        3.6,
        "$\mathit{N}$" + f' = {len(df[df["test"] == 0])}',
        fontsize=10,
        verticalalignment="top",
    )

    png_path = _ensure_output_dir(path.replace(".pkl", ".png"))
    plt.savefig(png_path, dpi=500, transparent=True)

    # df = df.reindex(df["error"].abs().sort_values(ascending=False).index)


def plot_loocv_metrics(csv_path: str, save_path: str) -> None:
    """Plot LOOCV R² and RMSE for several regression models and save as PNG.

    Parameters
    ----------
    csv_path : str
        Path to the CSV file containing the LOOCV results.

    save_path : str
        Output image path, e.g. "folder/file.png".
        Parent folders are created automatically if they do not exist.

    Returns
    -------
    None
    """
    df = pd.read_csv(csv_path, index_col=0)

    models = [
        (r"PLS [+-]?\d+ cv",      "PLS"),
        (r"^Ridge .{0,} cv",      "Ridge"),
        (r"^ElasticNet .{0,} cv", "Elastic Net"),
        (r"^Lasso .{0,} cv",      "Lasso"),
        (r"^OMP .{0,} cv",        "OMP"),
    ]

    fig, ax1 = plt.subplots(figsize=(4, 3))
    ax2 = ax1.twinx()

    color_r2 = "tab:red"
    color_rmse = "tab:blue"

    handles = []  # Reserved for optional custom legend usage.
    labels = []  # Model labels for x-axis.

    best_rmse = np.inf
    best_idx = -1

    for model_idx, (regex, label) in enumerate(models):
        x_pos = model_idx + 1  # Place at 1, 2, 3, ...
        print("x_pos:", x_pos)

        r2_array = np.array([
            df.filter(regex=regex, axis=0).max()["cv_r2"],
        ])
        rmse_array = np.array([
            df.filter(regex=regex, axis=0).min()["cv_RMSE"],
        ])
        print(label, "R²:", r2_array, "RMSE:", rmse_array)

        rmse_val = float(rmse_array[0])
        if rmse_val < best_rmse:
            best_rmse = rmse_val
            best_idx = model_idx  # Zero-based index.

        # RMSE bars on right y-axis.
        b = ax2.bar(
            x_pos,
            rmse_array,
            color=color_rmse,
            alpha=1.0,
            width=0.4,
            label=label + " RMSE",
        )

        # R^2 points on left y-axis.
        s = ax1.scatter(
            x_pos,
            r2_array,
            color=color_r2,
            alpha=1.0,
            label=label + r" $r^2$",
            facecolor="None",
        )

        handles.append(s)
        handles.append(b)
        labels.append(label)

    # Left y-axis: R^2.
    ax1.set_ylabel(r"$r^2_{\mathrm{LOOCV}}$       ", loc="top", color=color_r2)
    ax1.set_yticks(np.arange(0, 1.1, 0.5))
    ax1.tick_params(axis="y", colors=color_r2)
    ax1.set_ylim(-0.5, 1)

    # Right y-axis: RMSE.
    ax2.set_ylabel("RMSE" + r"$_{\mathrm{LOOCV}}$" + " [kcal/mol]", loc="bottom", color=color_rmse)
    ax2.set_ylim(0, 1.5)
    ax2.set_yticks(np.arange(0, 1.1, 0.5))
    ax2.tick_params(axis="y", colors=color_rmse)

    # X-axis: model labels.
    x_ticks = np.arange(1, len(models) + 1)
    ax1.set_xticks(x_ticks)
    ax1.set_xticklabels(labels, rotation=-25, ha="left")

    # Bold only the model with minimum RMSE.
    if best_idx >= 0:
        for i, tick in enumerate(ax1.get_xticklabels()):
            if i == best_idx:
                tick.set_fontweight("bold")

    fig.tight_layout()

    save_path = _ensure_output_dir(save_path)

    fig.savefig(save_path, dpi=500, transparent=False)


def reaction_concentration_plot_complex(
    ΔGs,
    T=298.15,
    a0=100,
    save_path="simulation_complex.png",
):
    """
    Plot concentration profiles for a branched reaction network.

    ΔGs : list[float] of length 12
        Activation free energies [kcal/mol]:
        - ΔG1..ΔG4 for A -> Pi
        - ΔG13p, ΔG14p, ΔG23p, ΔG24p, ΔG31p, ΔG32p, ΔG41p, ΔG42p for Pi -> Pij'
    """
    # Physical constant. The Eyring prefactor is intentionally omitted below:
    # multiplying every rate by the same factor only rescales time and does not
    # change the concentration profile plotted against reaction progress.
    R = 1.987e-3  # [kcal/(mol K)]

    delta_g = np.asarray(ΔGs, dtype=float)
    if delta_g.size != 12:
        raise ValueError("ΔGs must contain 12 activation free energies.")
    if not np.all(np.isfinite(delta_g)):
        raise ValueError("ΔGs contains non-finite values.")

    # Numerically stable relative Eyring rates. Very favorable barriers can
    # otherwise overflow in exp(-ΔG/RT), while only rate ratios matter here.
    log_rates = -delta_g / (R * T)
    log_rates = log_rates - np.max(log_rates)
    rates = np.exp(np.clip(log_rates, -745.0, 0.0))
    (
        k1, k2, k3, k4,
        k13p, k14p, k23p, k24p,
        k31p, k32p, k41p, k42p
    ) = rates

    # Intermediate decomposition rates.
    k1p_sum = k13p + k14p
    k2p_sum = k23p + k24p
    k3p_sum = k31p + k32p
    k4p_sum = k41p + k42p
    ka = k1 + k2 + k3 + k4

    positive_rates = rates[rates > 0]
    if positive_rates.size == 0 or ka <= 0:
        raise ValueError("All kinetic rates are zero after scaling.")

    # Sample around every kinetic timescale so both very fast and very slow
    # follow-up reductions remain visible after the rate scaling.
    base_times = np.logspace(-8, 8, 900, base=10)
    t_values = [np.array([0.0])]
    for rate in positive_rates:
        t_values.append(base_times / rate)
    t = np.unique(np.concatenate(t_values))
    t = t[np.isfinite(t)]
    t.sort()

    def exp_decay(rate):
        """Return a clipped exponential decay for one nonnegative rate."""
        if rate <= 0:
            return np.ones_like(t)
        return np.exp(-np.clip(rate * t, 0.0, 745.0))

    def one_minus_exp_over_rate(rate):
        """Evaluate ``(1-exp(-rate*t))/rate`` including the zero-rate limit."""
        if rate <= 0:
            return t.copy()
        return -np.expm1(-np.clip(rate * t, 0.0, 745.0)) / rate

    def safe_species(values):
        """Replace non-finite concentrations and clip them to physical bounds."""
        values = np.nan_to_num(values, nan=0.0, posinf=a0, neginf=0.0)
        return np.clip(values, 0.0, a0)

    # Concentration of A.
    a = a0 * exp_decay(ka)

    # Intermediate concentrations Pi.
    def p_i(k_i, k_ip_sum):
        """Return one intermediate concentration with its equal-rate limit."""
        if k_i <= 0:
            return np.zeros_like(t)
        scale = max(abs(k_ip_sum), abs(ka), 1.0)
        if abs(k_ip_sum - ka) <= 1e-10 * scale:
            return safe_species(k_i * a0 * t * exp_decay(ka))
        return safe_species(
            (k_i * a0 / (k_ip_sum - ka))
            * (exp_decay(ka) - exp_decay(k_ip_sum))
        )

    p1 = p_i(k1, k1p_sum)
    p2 = p_i(k2, k2p_sum)
    p3 = p_i(k3, k3p_sum)
    p4 = p_i(k4, k4p_sum)

    p_intermediate_total = p1 + p2 + p3 + p4

    # Time where total intermediate concentration is maximal.
    t_max_idx = np.nanargmax(p_intermediate_total)
    t_max = t[t_max_idx]

    # Product concentrations Pij'.
    def pij_total(k_i, k_ijp, k_ip_sum):
        """Return one final-product contribution with stable rate limits."""
        if k_i <= 0 or k_ijp <= 0:
            return np.zeros_like(t)
        scale = max(abs(k_ip_sum), abs(ka), 1.0)
        if abs(k_ip_sum - ka) <= 1e-10 * scale:
            if ka <= 0:
                return np.zeros_like(t)
            integral = (
                1.0
                - exp_decay(ka) * (1.0 + ka * t)
            ) / (ka * ka)
            return safe_species(k_i * k_ijp * a0 * integral)
        term1 = one_minus_exp_over_rate(ka)
        term2 = one_minus_exp_over_rate(k_ip_sum)
        return safe_species(
            (k_i * k_ijp * a0 / (k_ip_sum - ka)) * (term1 - term2)
        )

    # Mapping:
    #  p13 ↔ (1,3), p14 ↔ (1,4), p23 ↔ (2,3), p24 ↔ (2,4)
    p13 = pij_total(k1, k13p, k1p_sum) + pij_total(k3, k31p, k3p_sum)
    p14 = pij_total(k1, k14p, k1p_sum) + pij_total(k4, k41p, k4p_sum)
    p23 = pij_total(k2, k23p, k2p_sum) + pij_total(k3, k32p, k3p_sum)
    p24 = pij_total(k2, k24p, k2p_sum) + pij_total(k4, k42p, k4p_sum)

    pp_total = p13 + p14 + p23 + p24

    # Reaction progress.
    progress = p1 / 2 + p2 / 2 + p3 / 2 + p4 / 2 + pp_total
    progress = np.nan_to_num(progress, nan=0.0, posinf=a0, neginf=0.0)
    progress = np.maximum.accumulate(progress)
    if a0 > 0:
        progress = np.clip(progress / a0, 0.0, 1.0)
        p1, p2, p3, p4, p13, p14, p23, p24 = [
            np.clip(values / a0, 0.0, 1.0)
            for values in (p1, p2, p3, p4, p13, p14, p23, p24)
        ]
        p_intermediate_total = np.clip(p1 + p2 + p3 + p4, 0.0, 1.0)
        pp_total = np.clip(p13 + p14 + p23 + p24, 0.0, 1.0)
        a = np.clip(a / a0, 0.0, 1.0)

    # Report p1-p4 values at t_max.
    print("At t_max (intermediate total maximum):")
    print(f"  t_max = {t_max:.3e}")
    print(f"  p1(t_max) = {p1[t_max_idx]:.6f}")
    print(f"  p2(t_max) = {p2[t_max_idx]:.6f}")
    print(f"  p3(t_max) = {p3[t_max_idx]:.6f}")
    print(f"  p4(t_max) = {p4[t_max_idx]:.6f}")

    # Percent values at t_max.
    total_p_tmax = p_intermediate_total[t_max_idx]
    if total_p_tmax > 0:
        p_fracs = [
            p1[t_max_idx] * 100.0,
            p2[t_max_idx] * 100.0,
            p3[t_max_idx] * 100.0,
            p4[t_max_idx] * 100.0,
        ]
    else:
        p_fracs = [0.0, 0.0, 0.0, 0.0]

    # Plot.
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    fig.patch.set_alpha(0.0)

    # Colors for intermediates.
    c1 = "red"
    c2 = "tab:pink"
    c3 = "blue"
    c4 = "tab:blue"
    base_colors = [c1, c2, c3, c4]

    # Products: two-color style via facecolor + edgecolor + hatch.
    product_facecolors = [c1, c1, c2, c2]      # face
    product_edgecolors = [c3, c4, c3, c4]      # edge
    hatches = ["///", "\\\\", "xx", ".."]  # Hatch styles for visual separation.

    # Legend labels.
    labels = [
        rf"$\bf{{1}}$ {p_fracs[0]:4.1f}%",
        rf"$\bf{{2}}$ {p_fracs[1]:4.1f}%",
        rf"$\bf{{3}}$ {p_fracs[2]:4.1f}%",
        rf"$\bf{{4}}$ {p_fracs[3]:4.1f}%",
        rf"$\bf{{1-3}}$ {p13[-1]*100:4.1f}%",
        rf"$\bf{{1-4}}$ {p14[-1]*100:4.1f}%",
        rf"$\bf{{2-3}}$ {p23[-1]*100:4.1f}%",
        rf"$\bf{{2-4}}$ {p24[-1]*100:4.1f}%",
    ]

    # Stackplot order (bottom -> top): p1, p2, p3, p4, p13, p14, p23, p24.
    all_colors = base_colors + product_facecolors

    polys = ax.stackplot(
        progress,
        p1, p2, p3, p4,
        p13, p14, p23, p24,
        colors=all_colors,
        labels=labels,
    )
    # p1-p4 shown with moderate transparency.
    for poly in polys[:4]:
        poly.set_alpha(0.6)

    # p13-p24 use matching facecolor and differentiated edge/hatch styles.
    for i, poly in enumerate(polys[4:]):
        poly.set_alpha(0.5)
        poly.set_hatch(hatches[i])
        poly.set_edgecolor(product_edgecolors[i])
        poly.set_linewidth(0.6)

    # Intermediate total line.
    ax.plot(progress, p_intermediate_total, color="gray", linestyle="-")
    x0 = progress[t_max_idx]
    y0 = p_intermediate_total[t_max_idx]

    # Vertical guide lines.
    ax.plot([x0, x0], [0, y0], color="green", linestyle="--", linewidth=1.0)
    ax.plot([1, 1], [0, 1], color="purple", linestyle="--", linewidth=1.0)

    # Total dialcohol line.
    ax.plot(
        progress,
        p_intermediate_total + pp_total,
        color="tab:gray",
        linestyle="-",
    )

    ax.set_xlabel("reaction progress [-]")
    ax.set_ylabel("concentration [-]")
    ax.set_xticks([0, 0.5, 1])
    ax.set_yticks([0, 0.5, 1])
    ax.set_ylim(-0.02, 1.01)
    ax.set_xlim(-0.01, 1.01)

    # # Legend
    # handles, legend_labels = ax.get_legend_handles_labels()
    # leg = ax.legend(
    #     handles[::-1],
    #     legend_labels[::-1],
    #     loc="upper left",
    #     ncol=1,
    #     fontsize=9,
    #     borderpad=0.2,
    #     labelspacing=0.2,
    #     handlelength=1.0,
    #     handletextpad=0.3,
    #     borderaxespad=0.2,
    #     frameon=False,
    #     framealpha=0.8,
    #     prop={"family": "monospace", "size": 9},
    # )

    # for text in leg.get_texts():
    #     txt = text.get_text()
    #     if txt.startswith(r"$\bf{1}$") or txt.startswith(r"$\bf{2}$") \
    #        or txt.startswith(r"$\bf{3}$") or txt.startswith(r"$\bf{4}$"):
    #         text.set_color("green")
    #     else:
    #         text.set_color("purple")
    # Base legend handles/labels.
    handles, legend_labels = ax.get_legend_handles_labels()

    # Split legend entries: top four and bottom four.
    handles_1 = handles[:4]
    labels_1  = legend_labels[:4]
    handles_2 = handles[4:]
    labels_2  = legend_labels[4:]

    # First legend (1-4), placed outside left-top.
    leg1 = ax.legend(
        handles_1,
        labels_1,
        loc="upper left",
        bbox_to_anchor=(1.02, 0.5),  # Offset outside axes.
        ncol=1,
        fontsize=9,
        borderpad=0.2,
        labelspacing=0.2,
        handlelength=1.0,
        handletextpad=0.3,
        borderaxespad=0.2,
        frameon=False,
        framealpha=0.8,
        title="Max point",
        title_fontsize=9,
        # prop={"family": "monospace", "size": 9},
    )

    # Second legend (1-3/1-4/2-3/2-4), placed outside upper-right.
    leg2 = ax.legend(
        handles_2,
        labels_2,
        loc="upper left",
        bbox_to_anchor=(1.0, 1.02),  # Offset outside axes.
        ncol=1,
        fontsize=9,
        borderpad=0.2,
        labelspacing=0.2,
        handlelength=1.0,
        handletextpad=0.3,
        frameon=False,
        framealpha=0.8,
        title="Final point",
        title_fontsize=9,
        # prop={"family": "monospace", "size": 9},
    )

    # Add legends explicitly.
    ax.add_artist(leg1)
    ax.add_artist(leg2)
    # Legend title colors.
    leg1.get_title().set_color("green")
    leg2.get_title().set_color("purple")

    # Legend text colors.
    for text in leg1.get_texts():
        text.set_color("green")
    for text in leg2.get_texts():
        text.set_color("purple")
    plt.tight_layout()
    # Add right margin so legends are not clipped.
    plt.subplots_adjust(right=0.7)
    

    save_path = _ensure_output_dir(save_path)

    plt.savefig(save_path, dpi=500, transparent=False)


if __name__ == '__main__':
    evaluate_result(f"data/data_electronic_electrostatic_lumo_regression.pkl")

    df=best_parameter("data/data_electronic_electrostatic_lumo_results.csv")#highlight_colors={"DSSYKIVIOFKYAU-MHPPCMCBSA-N":"1","UMJJFEIKYGFCAT-HOSYLAQJSA-N":"2","YKFKEYKJGVSEIX-KWYDOPHBSA-N":"3"}
    save_non_diketone_test_predictions(df, "data/validation/non_diketone_test_predictions.csv")
    save_diketone_selectivity_summary(df, "data/test/selectivity_summary.csv")
    plot_training_contribution_numberlines(df, "data/validation/training_contribution_numberlines.png")
    plot_group_contribution_violins_by_skeleton(
        df,
        "data/validation/group_contribution_violins_by_skeleton.png",
        highlighted_save_path="data/validation/group_contribution_violins_by_skeleton_highlighted.png",
        values_csv_path="data/validation/group_contribution_distribution_values.csv",
        summary_csv_path="data/validation/group_contribution_distribution_summary.csv",
        highlights_csv_path="data/validation/group_contribution_skeleton_highlights.csv",
    )
    plot_group_contribution_skeleton_highlight_structures(
        df,
        "data/validation/group_contribution_skeleton_highlight_structures.png",
    )
    plot_selected_contribution_breakdowns(df, "data/validation")
    plot_3d_contributions(df, "data/validation/contributions_3d.png", highlight_colors={"RWCCWEUUXYIKHB-KHWBWMQUSA-N":""}, ref_inchikey="RWCCWEUUXYIKHB-KHWBWMQUSA-N")
    plot_contribution_bars(df,inchikeys=["DSSYKIVIOFKYAU-MHPPCMCBSA-N","UMJJFEIKYGFCAT-HOSYLAQJSA-N","YKFKEYKJGVSEIX-KWYDOPHBSA-N"],labels=["1","2","3"],ref_inchikey="RWCCWEUUXYIKHB-KHWBWMQUSA-N", save_path="data/validation/contribution_bars.png")
    plot_pair_stacked_contributions(df, target_inchikey="DSSYKIVIOFKYAU-MHPPCMCBSA-N", ref_inchikey="RWCCWEUUXYIKHB-KHWBWMQUSA-N", save_path="data/validation/DSSYKIVIOFKYAU-MHPPCMCBSA-N.png")
    plot_pair_stacked_contributions(df, target_inchikey="UMJJFEIKYGFCAT-HOSYLAQJSA-N", ref_inchikey="RWCCWEUUXYIKHB-KHWBWMQUSA-N", save_path="data/validation/UMJJFEIKYGFCAT-HOSYLAQJSA-N.png")
    plot_pair_stacked_contributions(df, target_inchikey="YKFKEYKJGVSEIX-KWYDOPHBSA-N", ref_inchikey="RWCCWEUUXYIKHB-KHWBWMQUSA-N", save_path="data/validation/YKFKEYKJGVSEIX-KWYDOPHBSA-N.png")

    plot_loocv_metrics("data/data_electronic_electrostatic_lumo_results.csv", "data/validation/loocv_metrics.png")

    out_path = CONTRIBUTIONS_ROOT
    make_cube(df,out_path)
    # make_cube_with_sign_markers(df,out_path)
    plot_expt_vs_pred(df,"data/validation/regression.png")

    reaction_concentration_plot_complex(
        ΔGs=df.set_index("entry").loc[["a1","a2","a3","a4","a13","a14","a23","a24","a31","a32","a41","a42"], "prediction"].to_numpy(),
        T=273.15,a0=1,save_path="data/test/a.png")
    reaction_concentration_plot_complex(
        ΔGs=df.set_index("entry").loc[["b1","b2","b3","b4","b13","b14","b23","b24","b31","b32","b41","b42"], "prediction"].to_numpy(),
        T=298.15,a0=1,save_path="data/test/b.png")
    reaction_concentration_plot_complex(
        ΔGs=df.set_index("entry").loc[["c1","c2","c3","c4","c13","c14","c23","c24","c31","c32","c41","c42"], "prediction"].to_numpy(),
        T=298.15,a0=1,save_path="data/test/c.png")
    reaction_concentration_plot_complex(
        ΔGs=df.set_index("entry").loc[["d1","d2","d3","d4","d13","d14","d23","d24","d31","d32","d41","d42"], "prediction"].to_numpy(),
        T=298.15,a0=1,save_path="data/test/d.png")
    reaction_concentration_plot_complex(
        ΔGs=df.set_index("entry").loc[["e1","e2","e3","e4","e13","e14","e23","e24","e31","e32","e41","e42"], "prediction"].to_numpy(),
        T=298.15,a0=1,save_path="data/test/e.png")
    reaction_concentration_plot_complex(
        ΔGs=df.set_index("entry").loc[["f1","f2","f3","f4","f13","f14","f23","f24","f31","f32","f41","f42"], "prediction"].to_numpy(),
        T=298.15,a0=1,save_path="data/test/f.png")
