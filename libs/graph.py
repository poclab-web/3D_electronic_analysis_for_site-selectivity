import glob
from itertools import product
import os
from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from rdkit.Chem import PandasTools
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

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

def evaluate_result(path):
    df=pd.read_pickle(path)
    df_results=pd.DataFrame(index=df.filter(like='cv').columns)
    df_results["cv_RMSE"]=df_results.index.map(lambda column: nan_rmse(df[column].values,df["ΔΔG.expt."].values))
    df_results["cv_r2"]=df_results.index.map(lambda column: nan_r2(df[column].values,df["ΔΔG.expt."].values))
    df_results["regression_RMSE"]=df.filter(like='regression').columns.map(lambda column: nan_rmse(df[column].values,df["ΔΔG.expt."].values))
    df_results["regression_r2"]=df.filter(like='regression').columns.map(lambda column: nan_r2(df[column].values,df["ΔΔG.expt."].values))
    df_results.to_csv(path.replace("_regression.pkl","_results.csv"))
    best_cv_column=df_results["cv_RMSE"].idxmin()
    print(best_cv_column,np.log2(float(best_cv_column.split()[1])))
    return df_results["cv_RMSE"].idxmin()

def best_parameter(path):
    best_cv_column=pd.read_csv(path,index_col=0)["cv_RMSE"].idxmin()
    coef=pd.read_csv(path.replace("_results.csv","_regression.csv"), index_col=0)
    coef = coef[[best_cv_column.replace("cv", "electronic_coef"), best_cv_column.replace("cv", "electrostatic_coef"), best_cv_column.replace("cv", "lumo_coef")]]
    coef.columns = ["electronic_coef", "electrostatic_coef","lumo_coef"]
    df=pd.read_pickle(path.replace("_results.csv","_regression.pkl"))
    columns=df.filter(like='electronic_unfold').columns.tolist()+df.filter(like='electrostatic_unfold').columns.tolist()+df.filter(like='lumo_unfold').columns.tolist()
    def calc_cont(column):
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
    highlight_colors=None,  # <- {InChIKey: "Label"} を渡す
) -> None:
    """3D 空間に electronic / electrostatic / orbital の寄与をプロットする。

    Axes
    ----
    x : electronic_cont    -> electronic [kcal/mol]
    y : electrostatic_cont -> electrostatic [kcal/mol]
    z : lumo_cont          -> orbital [kcal/mol]

    highlight_colors : dict or None
        {InChIKey: "Label"} の辞書。
        - 指定された InChIKey の点は、プロットと垂線・射影を
          「大きく・不透明」にしてハイライト。
        - "Label" はその点の近くに図中表示する。
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

    # ハイライト用マップ（実際には InChIKey -> Label の対応）
    if highlight_colors is None:
        highlight_map: dict[str, str] = {}
    elif isinstance(highlight_colors, dict):
        highlight_map = highlight_colors
    else:
        raise TypeError(
            "highlight_colors must be a dict like {InChIKey: 'Label'} or None."
        )

    # regression があるデータだけ使用
    df_reg = df.dropna(
        subset=["electronic_cont", "electrostatic_cont", "lumo_cont", "regression"]
    )

    x_reg = df_reg["electronic_cont"].values
    y_reg = df_reg["electrostatic_cont"].values
    z_reg = df_reg["lumo_cont"].values
    inchis = df_reg["InChIKey"].values

    # 各点がハイライト対象かどうか
    is_high = np.array([ik in highlight_map for ik in inchis], dtype=bool)
    is_norm = ~is_high

    # 軸範囲（後で 1:1:1 にする）
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

    fig = plt.figure(figsize=(5, 4), facecolor="white")
    fig.patch.set_alpha(0.0)
    ax = fig.add_subplot(111, projection="3d")

    # 軸リミットを設定
    ax.set_xlim(x_min_lim, x_max_lim)
    ax.set_ylim(y_min_lim, y_max_lim)
    ax.set_zlim(z_min_lim, z_max_lim)

    # 3軸スケールを揃える
    span_x = x_max_lim - x_min_lim
    span_y = y_max_lim - y_min_lim
    span_z = z_max_lim - z_min_lim
    ax.set_box_aspect((span_x, span_y, span_z))

    # 目盛は整数のみ
    xticks = np.arange(int(np.floor(x_min_lim)), int(np.ceil(x_max_lim)) + 1)
    yticks = np.arange(int(np.floor(y_min_lim)), int(np.ceil(y_max_lim)) + 1)
    zticks = np.arange(int(np.floor(z_min_lim)), int(np.ceil(z_max_lim)) + 1)
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.set_zticks(zticks)

    # 散布図：通常点（半透明黒）
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

    # 散布図：ハイライト点（大きく・不透明）
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

    # 見やすい視点（※ここまでで軸が決まった状態にする）
    ax.view_init(elev=22, azim=35)

    # ★ 最終的な軸リミットを取得して、それを射影面に使う ★
    x_min_plane, x_max_plane = ax.get_xlim3d()
    y_min_plane, y_max_plane = ax.get_ylim3d()
    z_min_plane, z_max_plane = ax.get_zlim3d()

    # --- 各平面（xy, yz, zx）を薄い色で描画 ---
    # xy-plane (z = z_min_plane) -> 薄い赤
    xy_verts = [
        [x_min_plane, y_min_plane, z_min_plane],
        [x_max_plane, y_min_plane, z_min_plane],
        [x_max_plane, y_max_plane, z_min_plane],
        [x_min_plane, y_max_plane, z_min_plane],
    ]
    poly_xy = Poly3DCollection([xy_verts], facecolors=(1, 0, 0, 0.05), edgecolors="none")
    ax.add_collection3d(poly_xy)

    # yz-plane (x = x_min_plane) -> 薄い青
    yz_verts = [
        [x_min_plane, y_min_plane, z_min_plane],
        [x_min_plane, y_max_plane, z_min_plane],
        [x_min_plane, y_max_plane, z_max_plane],
        [x_min_plane, y_min_plane, z_max_plane],
    ]
    poly_yz = Poly3DCollection([yz_verts], facecolors=(0, 0, 1, 0.05), edgecolors="none")
    ax.add_collection3d(poly_yz)

    # zx-plane (y = y_min_plane) -> 薄い緑
    zx_verts = [
        [x_min_plane, y_min_plane, z_min_plane],
        [x_max_plane, y_min_plane, z_min_plane],
        [x_max_plane, y_min_plane, z_max_plane],
        [x_min_plane, y_min_plane, z_max_plane],
    ]
    poly_zx = Poly3DCollection([zx_verts], facecolors=(0, 1, 0, 0.05), edgecolors="none")
    ax.add_collection3d(poly_zx)

    # 射影用の基本色（高透過率）
    base_xy = np.array([1.0, 0.0, 0.0])   # red
    base_yz = np.array([0.0, 0.0, 1.0])   # blue
    base_zx = np.array([0.0, 0.6, 0.0])   # green

    def rgba(base_color, alpha: float):
        return (base_color[0], base_color[1], base_color[2], alpha)

    # --- 垂直線と射影 ---
    def add_projections(
        xs: np.ndarray,
        ys: np.ndarray,
        zs: np.ndarray,
        alpha: float,
        size: float,
        lw: float = 0.5,
    ) -> None:
        """各点から xy, xz, yz 平面への垂線・射影点を描く。

        - xy-plane (z = z_min_plane): 赤
        - yz-plane (x = x_min_plane): 青
        - zx-plane (y = y_min_plane): 緑
        """
        line_alpha = alpha * 0.6
        for xi, yi, zi in zip(xs, ys, zs):
            # xy-plane: z = z_min_plane (red)
            ax.plot(
                [xi, xi],
                [yi, yi],
                [zi, z_min_plane],
                color=rgba(base_xy, line_alpha),
                linewidth=lw,
            )
            ax.plot(
                [xi],
                [yi],
                [z_min_plane],
                "o",
                color=rgba(base_xy, alpha),
                markersize=size,
            )

            # xz-plane: y = y_min_plane (green)
            ax.plot(
                [xi, xi],
                [yi, y_min_plane],
                [zi, zi],
                color=rgba(base_zx, line_alpha),
                linewidth=lw,
            )
            ax.plot(
                [xi],
                [y_min_plane],
                [zi],
                "o",
                color=rgba(base_zx, alpha),
                markersize=size,
            )

            # yz-plane: x = x_min_plane (blue)
            ax.plot(
                [xi, x_min_plane],
                [yi, yi],
                [zi, zi],
                color=rgba(base_yz, line_alpha),
                linewidth=lw,
            )
            ax.plot(
                [x_min_plane],
                [yi],
                [zi],
                "o",
                color=rgba(base_yz, alpha),
                markersize=size,
            )

    # 通常点の垂線・射影（小さく・半透明）
    if is_norm.any():
        add_projections(
            x_reg[is_norm],
            y_reg[is_norm],
            z_reg[is_norm],
            alpha=0.25,
            size=2.0,
            lw=0.5,
        )

    # ハイライト点の垂線・射影（大きく・不透明）
    if is_high.any():
        add_projections(
            x_reg[is_high],
            y_reg[is_high],
            z_reg[is_high],
            alpha=1.0,
            size=4.0,
            lw=0.8,
        )

    # ハイライト点のラベルを図中に表示
    for ik, label in highlight_map.items():
        mask = (inchis == ik)
        if not np.any(mask):
            continue
        xi = x_reg[mask][0]
        yi = y_reg[mask][0]
        zi = z_reg[mask][0]
        # 少しだけオフセットをつけて重なりを避ける
        ax.text(
            xi + 0.02 * span_x,
            yi + 0.02 * span_y,
            zi + 0.02 * span_z,
            label,
            fontsize=8,
            color="black",
            ha="left",
            va="bottom",
        )

    # 軸ラベル
    ax.set_xlabel("electronic [kcal/mol]")
    ax.set_ylabel("electrostatic [kcal/mol]")
    ax.set_zlabel("orbital [kcal/mol]")

    # 凡例なし、見切れ防止
    plt.tight_layout()
    fig.savefig(
        save_path,
        dpi=500,
        transparent=False,
        bbox_inches="tight",
        pad_inches=1,
    )
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_contribution_bars(
    df: pd.DataFrame,
    inchikeys: list[str],
    labels: list[str],
    save_path: str,
    ref_inchikey: str | None = None,
) -> None:
    """
    複数 InChIKey について、electronic / electrostatic / lumo の寄与を
    1 枚の棒グラフにまとめて描画する。

    x 軸方向に
        [ electronic ]  [ electrostatic ]  [ lumo ]
    の 3 ブロックを配置し、
    各ブロックの中に「分子ごとの棒（InChIKey ごと）」を並べる。

    ref_inchikey を指定すると、各寄与は
        寄与値(target) - 寄与値(ref_inchikey)
    の差分として描画される。

    Parameters
    ----------
    df : pandas.DataFrame
        下記カラムを含む DataFrame:
            - "InChIKey"
            - "electronic_cont"
            - "electrostatic_cont"
            - "lumo_cont"

    inchikeys : str or list[str]
        プロット対象とする InChIKey またはそのリスト。

    save_path : str
        出力 PNG などのファイルパス。

    ref_inchikey : str or None, optional
        差分の基準とする InChIKey。
        None の場合は絶対値（各分子そのものの寄与）を描画。

    Returns
    -------
    None
        画像ファイルを save_path に保存する。
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

    # inchikeys をリストに正規化
    if isinstance(inchikeys, str):
        inchikey_list = [inchikeys]
    else:
        inchikey_list = list(inchikeys)

    # 存在チェック
    for ik in inchikey_list:
        if ik not in df["InChIKey"].values:
            raise ValueError(f"InChIKey '{ik}' was not found in DataFrame.")

    # 基準分子の寄与（差分モードの場合）
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

    # 寄与値を InChIKey 順に並べた配列 (N_mol, 3)
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
            vals = vals - ref_vals  # 差分
        contributions.append(vals)
    contributions = np.vstack(contributions)  # shape: (N, 3)

    n_mol = contributions.shape[0]
    categories = ["electronic", "electrostatic", "lumo"]
    n_cat = len(categories)

    # x 軸上で [0,1,2] が 3 ブロックの中心
    x_base = np.arange(n_cat, dtype=float)

    # 各ブロック内で InChIKey ごとの棒を横にずらす
    total_width = 0.8  # ブロックの幅
    bar_width = total_width / max(n_mol, 1)

    # 図の作成
    fig, ax = plt.subplots(figsize=(5, 3))

    for i, ik in enumerate(labels):
        # i 番目の分子の寄与 (3 要素: ele, es, lumo)
        vals = contributions[i, :]
        # 3 つのブロック内での x 位置
        x_pos = x_base + (i - (n_mol - 1) / 2) * bar_width

        ax.bar(
            x_pos,
            vals,
            width=bar_width,
            label=ik,
            alpha=0.8,
        )

    # 0 ライン
    ax.axhline(0, color="black", linewidth=1.0)

    # x 軸の目盛り（ブロック中央）
    ax.set_xticks(x_base)
    ax.set_xticklabels(categories)

    # ラベル
    if ref_vals is None:
        ax.set_ylabel("contribution [kcal/mol]")
        title = "Contributions"
    else:
        ax.set_ylabel(
            "contribution difference [kcal/mol]"
            #f"\n(relative to {ref_inchikey})"
        )
        # title = "Contribution differences"

    # ax.set_xlabel("contribution type")
    # ax.set_title(title)

    # 凡例（InChIKey ごと）
    ax.legend(frameon=False, fontsize=8, ncol=1)

    # x 方向に少し余裕
    ax.margins(x=0.1)

    fig.tight_layout()
    fig.savefig(save_path, dpi=400)
    plt.close(fig)


def plot_pair_stacked_contributions(
    df: pd.DataFrame,
    target_inchikey: str,
    ref_inchikey: str,
    save_path: str,
) -> None:
    """
    2つの InChIKey (target と ref) について、
    electronic / electrostatic / lumo の寄与の差分
        (target - ref)
    を、横向きのずらし積み上げ棒グラフで描画する。

    棒の構成（上から下）:
        - 一番上: 0 から伸びる electronic の横棒
        - 真ん中: electronic の終点から伸びる electrostatic の横棒
        - 一番下: (electronic + electrostatic) の終点から伸びる lumo の横棒
      → lumo の棒の終点が 3 つの寄与の和 (electronic + electrostatic + lumo)

    仕様:
    - 棒の向きは横向き (barh)。
    - 各棒の「すぐ外側」に寄与値をテキスト表示。
    - 横軸に 0 の場所の目盛りと値を表示し、0 の位置に点線を引く。
    - electronic, electronic+electrostatic, electronic+electrostatic+lumo
      の位置に目盛りと値を表示し、総和位置まで点線を引く。
    - 図は横長のアスペクト比。
    - 色:
        electronic    : 赤紫 (darkmagenta)
        electrostatic : 緑   (forestgreen)
        lumo          : 茶色 (saddlebrown)
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

    # --- 対象 & 比較分子のレコード取得 ---
    if target_inchikey not in df["InChIKey"].values:
        raise ValueError(f"target_inchikey '{target_inchikey}' was not found in DataFrame.")
    if ref_inchikey not in df["InChIKey"].values:
        raise ValueError(f"ref_inchikey '{ref_inchikey}' was not found in DataFrame.")

    row_target = df[df["InChIKey"] == target_inchikey].iloc[0]
    row_ref    = df[df["InChIKey"] == ref_inchikey].iloc[0]

    # 寄与 (target - ref)
    target_vals = np.array(
        [
            float(row_target["electronic_cont"]),
            float(row_target["electrostatic_cont"]),
            float(row_target["lumo_cont"]),
        ]
    )
    ref_vals = np.array(
        [
            float(row_ref["electronic_cont"]),
            float(row_ref["electrostatic_cont"]),
            float(row_ref["lumo_cont"]),
        ]
    )
    contrib = target_vals - ref_vals
    elec, es, lumo = contrib

    # 累積位置
    s1 = elec
    s2 = elec + es
    s3 = elec + es + lumo  # 最終的な 3 つの寄与の和

    # --- x 範囲の決定（0 が真ん中に来るように左右対称にする） ---
    # 棒が到達しうる位置の最大絶対値
    core = np.max(np.abs([s1, s2, s3]))
    if core == 0:
        core = 1.0  # 全部 0 のときの保険

    # 少し余裕を持たせた半幅
    margin_ratio = 0.3
    half_width = core * (1.0 + margin_ratio)

    x_min = -half_width
    x_max = +half_width
    span = x_max - x_min  # 注釈位置計算などに使用

    # --- 図の準備（横長） ---
    fig, ax = plt.subplots(figsize=(5, 1.5))

    # 棒の y 位置（上から順に 3 本）
    y_elec = 2.0
    y_es   = 1.0
    y_lumo = 0.0
    y_pos  = [y_elec, y_es, y_lumo]
    y_labels = ["electronic", "electrostatic", "lumo"]

    # 色指定
    color_elec = "darkmagenta"   # 赤紫
    color_es   = "forestgreen"   # 緑
    color_lumo = "saddlebrown"   # 茶色

    # --- 棒グラフ (横向き) ---
    # electronic: 0 → elec
    ax.barh(
        y_elec,
        width=elec,
        left=0.0,
        color=color_elec,
        alpha=0.8,
        height=1,
    )

    # electrostatic: s1 → s2
    ax.barh(
        y_es,
        width=es,
        left=s1,
        color=color_es,
        alpha=0.8,
        height=1,
    )

    # lumo: s2 → s3
    ax.barh(
        y_lumo,
        width=lumo,
        left=s2,
        color=color_lumo,
        alpha=0.8,
        height=1,
    )

    # --- 棒の外側に寄与値を表示 ---
    def annotate_bar(base: float, val: float, y: float) -> None:
        x_end = base + val
        text_offset = 0.02 * span

        if val >= 0:
            x_text = x_end + text_offset
            ha = "left"
        else:
            x_text = x_end - text_offset
            ha = "right"

        ax.text(
            x_text,
            y,
            f"{val:.2f}",
            va="center",
            ha=ha,
            # fontsize=8,
        )

    annotate_bar(0.0, elec, y_elec)
    annotate_bar(s1, es, y_es)
    annotate_bar(s2, lumo, y_lumo)

    # --- 軸まわりの設定 ---
    # y 範囲少し広めにとる（下に矢印を描くため）
    y_min_plot = -0.5
    y_max_plot = 2.5
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min_plot, y_max_plot)

    # 縦軸はなくす
    ax.yaxis.set_visible(False)
    ax.spines["left"].set_visible(False)

    # electronic / electrostatic / lumo のラベルだけ残す（テキストで描画）
    label_x = x_min - 0.02 * span
    for ypos, label in zip(y_pos, y_labels):
        ax.text(
            label_x,
            ypos,
            label,
            va="center",
            ha="right",
            # fontsize=8,
        )

    # 横軸ラベル
    ax.set_xlabel("contribution difference [kcal/mol]")

    # x 方向の目盛り（0 と s3 のみ）
    xticks = sorted(set([0.0, s3]))
    ax.set_xticks(xticks)
    ax.set_xticklabels([f"{x:.2f}" for x in xticks])

    # 0 の位置に縦の点線
    ax.axvline(0.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)

    # 上・右・下の枠を非表示（下は矢印に置き換える）
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)

    # --- 右向き矢印としての横軸を描画（0 が中央になるように左右対称の範囲を使用） ---
    arrow_y = y_min_plot
    ax.annotate(
        "",
        xy=(x_max, arrow_y),
        xytext=(x_min, arrow_y),
        arrowprops=dict(arrowstyle="->", lw=1.0, color="black"),
    )

    # タイトル（コメントはそのまま残す）
    # ax.set_title(
    #     f"Contributions: {target_inchikey} − {ref_inchikey}",
    #     fontsize=9,
    # )

    fig.tight_layout()
    fig.savefig(save_path, dpi=400)
    plt.close(fig)



def make_cube(df,path):
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
    home = Path.home()
    out_path = home / "contributions"
    min=' '.join(map(str, (min+np.array([0.5,0.5,-0.5]))*n))
    for inchikey,expt,temp,value in zip(df.index,df["ΔΔG.expt."],df["temperature"],df.iloc[:,2:].values):
        dt=glob.glob(str(home)+ f"/molecules/{inchikey}/Dt*.cube")[0]
        # dt=f'/Volumes/SSD-PSM960U3-UW/CoMFA_calc/{inchikey}/Dt0.cube'
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
    """
    electronic_cont / electrostatic_cont / lumo_cont の符号に応じて
    X または Xe の原子を追加した Dt / ESP / LUMO の cube ファイルを生成する。

    - electronic_cont > 0 : X (原子番号 0, ダミー原子)
    - electronic_cont < 0 : Xe (原子番号 54)
      （electrostatic_cont, lumo_cont も同様）

    生成されるファイル:
        contributions/<InChIKey>/electronic.cube   (Dt*.cube が元)
        contributions/<InChIKey>/electrostatic.cube (ESP*.cube が元)
        contributions/<InChIKey>/lumo.cube        (LUMO*.cube が元)

    パラメータ
    ----------
    df : pandas.DataFrame
        各行が 1 分子（InChIKey でユニーク）に対応する DataFrame。
        以下のカラムを含んでいる必要がある:

        - "InChIKey"                : 分子 ID（行ごとに一意）
        - "ΔΔG.expt."               : 実験値（ログ出力などに使うだけ）
        - "temperature"             : 温度（同上）
        - "electronic_cont i j k"   : 電子寄与の格子値（任意個）
        - "electrostatic_cont i j k": 静電寄与の格子値（任意個）
        - "lumo_cont i j k"         : 軌道寄与の格子値（任意個）

        electronic_cont / electrostatic_cont / lumo_cont の
        「 i j k 」は、元の Dt / ESP / LUMO cube の格子インデックスに対応する整数。

    out_root : str or pathlib.Path
        出力のルートディレクトリ。通常は `Path.home() / "contributions"` など。
        実際の出力は out_root / InChIKey / {electronic,electrostatic,lumo}.cube

    戻り値
    -------
    None
        cube ファイルをディスクに書き出すだけで、値は返さない。

    注意
    ----
    - テンプレートとして、以下のパターンで cube を探す:
        ~/molecules/<InChIKey>/Dt*.cube
        ~/molecules/<InChIKey>/ESP*.cube
        ~/molecules/<InChIKey>/LUMO*.cube

      必要に応じてパターンは書き換えてください。
    - volumetric データは元の Dt / ESP / LUMO cube からそのままコピーし、
      原子行だけを追加します（グリッドは一切変更しません）。
    """
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    # --- 各 cont カラム名と、そのグリッドインデックス (i,j,k) のペアを取得 ---
    def extract_cont_columns(prefix: str):
        cols = [c for c in df.columns if c.startswith(prefix + " ")]
        if not cols:
            raise ValueError(f"No columns starting with '{prefix} ' were found in df.")
        # "prefix i j k" から (i,j,k) を取り出す
        grid_idx = np.array(
            [list(map(int, re.findall(r"[+-]?\d+", c))) for c in cols],
            dtype=int,
        )
        return cols, grid_idx

    ele_cols, grid_idx = extract_cont_columns("electronic_cont")
    es_cols, _ = extract_cont_columns("electrostatic_cont")
    lu_cols, _ = extract_cont_columns("lumo_cont")

    # グリッドインデックス (i,j,k) は 3 つの cont で共通だと仮定
    if not (len(ele_cols) == len(es_cols) == len(lu_cols)):
        raise ValueError(
            "The numbers of electronic_cont / electrostatic_cont / lumo_cont "
            "columns do not match."
        )

    # --- grid index -> 3D 座標への変換関数（cube の軸ベクトル & 原点を使う） ---
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
        """
        cont 値 `values` の符号に応じて X / Xe の原子行を作る。

        座標は cube の原点 origin と軸ベクトル ax, ay, az から
        「セル中心」を (i+0.5, j+0.5, k+0.5) で求める。
        """
        atoms: list[str] = []
        for (ix, iy, iz), v in zip(grid_indices, values):
            if np.isnan(v) or abs(v) <= eps:
                continue
            if v > 0:
                Z = pos_Z   # X: dummy atom
            else:
                Z = neg_Z   # Xe

            # セル中心の座標 (origin + (i+0.5)*ax + (j+0.5)*ay + (k+0.5)*az)
            r = (
                origin
                # + (ix + 0.5) * ax
                # + (iy + 0.5) * ay
                # + (iz + 0.5) * az
            )
            x, y, z = r.tolist()

            # Gaussian cube の原子行形式:
            #   atomic_number  charge  x  y  z
            # charge はここでは原子番号と同じにしておく
            line = f"{Z:5d}{float(Z):12.6f}{x:12.6f}{y:12.6f}{z:12.6f}"
            atoms.append(line)

        return atoms

    home = Path.home()

    # InChIKey を index にしてループ
    df_idx = df.set_index("InChIKey")

    for inchikey, row in df_idx.iterrows():
        # cont 値を取り出し
        ele_vals = row[ele_cols].to_numpy(dtype=float)
        es_vals  = row[es_cols].to_numpy(dtype=float)
        lu_vals  = row[lu_cols].to_numpy(dtype=float)

        # テンプレ cube を探す
        base_dir = home / "molecules" / inchikey
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
                # 見つからなければスキップ（必要なら warning を print してもよい）
                continue

            template_path = Path(candidates[0])

            # cube を丸ごと読む
            with open(template_path, "r", encoding="utf-8") as f:
                lines = f.readlines()

            if len(lines) < 7:
                raise ValueError(f"Cube file seems too short: {template_path}")

            # ヘッダ部をパース
            title_line   = lines[0].rstrip("\n")
            comment_line = lines[1].rstrip("\n")

            # 3 行目: natoms, origin
            natoms_tokens = lines[2].split()
            natoms = int(natoms_tokens[0])
            origin = np.array(list(map(float, natoms_tokens[1:4])), dtype=float)

            # 4–6 行目: グリッド情報
            gx = lines[3].split()
            gy = lines[4].split()
            gz = lines[5].split()

            nx = int(gx[0])
            ny = int(gy[0])
            nz = int(gz[0])

            ax = np.array(list(map(float, gx[1:4])), dtype=float)
            ay = np.array(list(map(float, gy[1:4])), dtype=float)
            az = np.array(list(map(float, gz[1:4])), dtype=float)

            # 原子行
            atom_lines = [line.rstrip("\n") for line in lines[6 : 6 + natoms]]

            # volumetric データはそのままコピー
            data_lines = [line.rstrip("\n") for line in lines[6 + natoms :]]

            # 対応する cont 値を選ぶ
            if kind == "electronic":
                vals = ele_vals
            elif kind == "electrostatic":
                vals = es_vals
            elif kind == "lumo":
                vals = lu_vals
            else:
                continue  # 念のため

            # cont の符号に応じた X / Xe マーカー原子を作成
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

            # 原子数を更新
            natoms_new = natoms + len(marker_atoms)
            natoms_line_new = (
                f"{natoms_new:5d}"
                f"{origin[0]:12.6f}{origin[1]:12.6f}{origin[2]:12.6f}"
            )

            # 新しい cube を組み立て
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

            # 出力先
            out_dir = out_root / inchikey
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{kind}.cube"

            with open(out_path, "w", encoding="utf-8") as f_out:
                f_out.write("\n".join(new_lines) + "\n")


def graph_(df,path):
    #直線表示
    plt.figure(figsize=(3, 3))
    plt.yticks([-4,0,4])
    plt.xticks([-4,0,4])
    plt.ylim(-4,4)
    plt.xlim(-4,4)
    
    plt.scatter(df["ΔΔG.expt."],df["regression"],c="black",linewidths=0,s=10,alpha=0.5)
    rmse=nan_rmse(df["regression"].values,df["ΔΔG.expt."].values)
    r2=nan_r2(df["regression"].values,df["ΔΔG.expt."].values)
    plt.scatter([],[],label="regression $r^2$ = " + f"{r2:.2f}"+"\n$\mathrm{RMSE}$"+f" = {rmse:.2f} kcal/mol"
                   ,c="black",linewidths=0,  alpha=0.5, s=10)
    
    rmse=nan_rmse(df["cv"].values,df["ΔΔG.expt."].values)
    r2=nan_r2(df["cv"].values,df["ΔΔG.expt."].values)
    plt.scatter([],[],label="LOOCV $r^2$ = " + f"{r2:.2f}"+"\n$\mathrm{RMSE}$"+f" = {rmse:.2f} kcal/mol"
                   ,c="dodgerblue",linewidths=0,  alpha=0.6, s=10)
    
    # rmse=nan_rmse(df["prediction"].values,df["ΔΔG.expt."].values)
    # r2=nan_r2(df["prediction"].values,df["ΔΔG.expt."].values)
    # plt.scatter([],[],label="$\mathrm{RMSE_{test}}$"+f" = {rmse:.2f}"
                #    +"\n$r^2_{\mathrm{test}}$ = " + f"{r2:.2f}",c="red",linewidths=0,  alpha=0.8, s=10)

    plt.scatter(df["ΔΔG.expt."],df["cv"],c="dodgerblue",linewidths=0,s=10,alpha=0.6)
    # plt.scatter(df["ΔΔG.expt."],df["prediction"],c="red",linewidths=0,s=10,alpha=0.8)
    plt.xlabel(r"$\Delta\Delta G^{\ddagger}_{\mathrm{expt}}$ [kcal/mol]")
    plt.ylabel(r"$\Delta\Delta G^{\ddagger}_{\mathrm{predict}}$ [kcal/mol]")
    plt.legend(
    loc='lower right',
    fontsize=6,
    ncol=1,
    borderpad=0.2,        # 枠内の余白
    #labelspacing=0.2,     # ラベル同士の間隔
    handletextpad=0.3,    # マーカーとテキストの間隔
    #borderaxespad=0.2,    # 軸との距離
    #columnspacing=0.5,    # 列間（ncol>1のとき有効）
    frameon=True,
    framealpha=0.8,
    )


    plt.text(-3.6, 3.6, "$\mathit{N}$"+f' = {len(df[df["test"]==0])}',# transform=ax.transAxes, 
                fontsize=10, verticalalignment='top')

    plt.tight_layout()
    plt.savefig(path.replace(".pkl",".png"),dpi=500,transparent=True)
    # df = df.reindex(df["error"].abs().sort_values(ascending=False).index)


def bar(path: str) -> None:
    """Plot LOOCV R² and RMSE for several regression models and save as PNG.

    Parameters
    ----------
    path : str
        Path to the CSV file containing the LOOCV results.
        The figure is saved with the same basename and `.png` extension.

    Returns
    -------
    None
    """
    df = pd.read_csv(path, index_col=0)

    models = [
        (r"PLS [+-]?\d+ cv",   "PLS"),
        (r"^Ridge .{0,} cv",      "Ridge"),
        (r"^ElasticNet .{0,} cv", "Elastic Net"),
        (r"^Lasso .{0,} cv",      "Lasso"),
        (r"^OMP .{0,} cv",        "OMP"),
    ]

    fig, ax1 = plt.subplots(figsize=(4, 2.5))
    ax2 = ax1.twinx()

    color_r2 = "tab:red"
    color_rmse = "tab:blue"

    handles = []  # for custom legend (もし後で使うなら)
    labels = []   # x 軸用のモデル名

    best_rmse = np.inf
    best_idx = -1

    for model_idx, (regex, label) in enumerate(models):
        x_pos = model_idx + 1  # 1,2,3,... に配置
        print("x_pos:", x_pos)

        r2_array = np.array([
            df.filter(regex=regex, axis=0).max()["cv_r2"],
        ])
        rmse_array = np.array([
            df.filter(regex=regex, axis=0).min()["cv_RMSE"],
        ])
        print(label, "R²:", r2_array, "RMSE:", rmse_array)

        # 1モデル1本なので index 0 だけ見ればよい
        rmse_val = float(rmse_array[0])
        if rmse_val < best_rmse:
            best_rmse = rmse_val
            best_idx = model_idx  # 0 始まり

        # RMSE: bar (右軸)
        b = ax2.bar(
            x_pos,
            rmse_array,
            color=color_rmse,
            alpha=1.0,
            width=0.4,
            label=label + " RMSE",
        )

        # R²: scatter (左軸)
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

    # 左 y 軸: R²
    ax1.set_ylabel(r"$r^2_{\mathrm{LOOCV}}$       ", loc="top", color=color_r2)
    ax1.set_yticks(np.arange(0, 1.1, 0.5))
    ax1.tick_params(axis="y", colors=color_r2)
    ax1.set_ylim(-0.5, 1)

    # 右 y 軸: RMSE
    ax2.set_ylabel("RMSE" + r"$_{\mathrm{LOOCV}}$" + " [kcal/mol]", loc="bottom", color=color_rmse)
    ax2.set_ylim(0, 1.5)
    ax2.set_yticks(np.arange(0, 1.1, 0.5))
    ax2.tick_params(axis="y", colors=color_rmse)

    # x 軸: モデル名ラベル
    x_ticks = np.arange(1, len(models) + 1)
    ax1.set_xticks(x_ticks)
    ax1.set_xticklabels(labels, rotation=-20, ha="left")

    # ★ 最小 RMSE のモデルラベルだけ太字にする
    if best_idx >= 0:
        for i, tick in enumerate(ax1.get_xticklabels()):
            if i == best_idx:
                tick.set_fontweight("bold")

    fig.tight_layout()

    base, _ = os.path.splitext(path)
    png_path = base + ".png"
    fig.savefig(png_path, dpi=500, transparent=False)



def reaction_concentration_plot_complex(
    ΔGs, T=298.15, a0=100,
    save_path="simulation_complex.png"
):
    # 定数
    kB = 1.380649e-23  # [J/K]
    h  = 6.62607015e-34  # [J·s]
    R  = 1.987e-3        # [kcal/(mol·K)]
    """
    複雑な分岐反応の濃度時間変化をプロットする。

    ΔGs: list of 12 floats
        ΔG1, ΔG2, ΔG3, ΔG4（A→Pi）[kcal/mol]
        ΔG13p, ΔG14p, ΔG23p, ΔG24p, ΔG31p, ΔG32p, ΔG41p, ΔG42p（Pi→Pij′）[kcal/mol]
    """

    # --- Eyring式 ---
    def k(ΔG):
        return (kB * T / h) * np.exp(-ΔG / (R * T))

    (
        k1, k2, k3, k4,
        k13p, k14p, k23p, k24p,
        k31p, k32p, k41p, k42p
    ) = [k(ΔG) for ΔG in ΔGs]

    # 中間体分解速度
    k1p_sum = k13p + k14p
    k2p_sum = k23p + k24p
    k3p_sum = k31p + k32p
    k4p_sum = k41p + k42p
    ka = k1 + k2 + k3 + k4

    # --- 時間軸 ---
    t = np.logspace(
        -10, 10, 1000, base=10
    ) / np.max([k1, k2, k3, k4, k13p, k14p, k23p, k24p, k31p, k32p, k41p, k42p])

    # A の濃度
    a = a0 * np.exp(-ka * t)

    # --- 中間体 Pi ---
    def p_i(k_i, k_ip_sum):
        return (k_i * a0 / (k_ip_sum - ka)) * (np.exp(-ka * t) - np.exp(-k_ip_sum * t))

    p1 = p_i(k1, k1p_sum)
    p2 = p_i(k2, k2p_sum)
    p3 = p_i(k3, k3p_sum)
    p4 = p_i(k4, k4p_sum)

    p_intermediate_total = p1 + p2 + p3 + p4

    # 中間体合計が最大の時刻
    t_max_idx = np.argmax(p_intermediate_total)
    t_max = t[t_max_idx]

    # --- 生成物 Pij′ ---
    def pij_total(k_i, k_ijp, k_ip_sum):
        term1 = (1 - np.exp(-ka * t)) / ka
        term2 = (1 - np.exp(-k_ip_sum * t)) / k_ip_sum
        return (k_i * k_ijp * a0 / (k_ip_sum - ka)) * (term1 - term2)

    # 対応関係:
    #  p13 ↔ (1,3), p14 ↔ (1,4), p23 ↔ (2,3), p24 ↔ (2,4)
    p13 = pij_total(k1, k13p, k1p_sum) + pij_total(k3, k31p, k3p_sum)
    p14 = pij_total(k1, k14p, k1p_sum) + pij_total(k4, k41p, k4p_sum)
    p23 = pij_total(k2, k23p, k2p_sum) + pij_total(k3, k32p, k3p_sum)
    p24 = pij_total(k2, k24p, k2p_sum) + pij_total(k4, k42p, k4p_sum)

    pp_total = p13 + p14 + p23 + p24

    # --- 反応進行度 ---
    progress = p1 / 2 + p2 / 2 + p3 / 2 + p4 / 2 + pp_total

    # --- t_max における p1〜p4 を print ---
    print("At t_max (intermediate total maximum):")
    print(f"  t_max = {t_max:.3e}")
    print(f"  p1(t_max) = {p1[t_max_idx]:.6f}")
    print(f"  p2(t_max) = {p2[t_max_idx]:.6f}")
    print(f"  p3(t_max) = {p3[t_max_idx]:.6f}")
    print(f"  p4(t_max) = {p4[t_max_idx]:.6f}")

    # t_max における割合（％）
    total_p_tmax = p_intermediate_total[t_max_idx]
    if total_p_tmax > 0:
        p_fracs = [
            p1[t_max_idx]  * 100.0,
            p2[t_max_idx]  * 100.0,
            p3[t_max_idx]  * 100.0,
            p4[t_max_idx]  * 100.0,
        ]
    else:
        p_fracs = [0.0, 0.0, 0.0, 0.0]

    # --- プロット ---
    fig, ax = plt.subplots(figsize=(3, 3))#, facecolor="none")
    fig.patch.set_alpha(0.0)

    # 中間体の色
    c1 = "red"
    c2 = "tab:pink"
    c3 = "blue"
    c4 = "tab:blue"
    base_colors = [c1, c2, c3, c4]

    # 生成物: 対応する 2色を「facecolor×edgecolor+ハッチ」で表現
    product_facecolors = [c1, c1, c2, c2]      # face
    product_edgecolors = [c3, c4, c3, c4]      # edge
    hatches = ["///", "\\\\", "xx", ".."]      # 2色感を出す模様

    # --- 凡例ラベル（1〜4 に％を付ける） ---
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

    # stackplot: 下から p1, p2, p3, p4, p13, p14, p23, p24 の順に積み上げ
    all_colors = base_colors + product_facecolors

    polys = ax.stackplot(
        progress,
        p1, p2, p3, p4,
        p13, p14, p23, p24,
        colors=all_colors,
        labels=labels,
    )
    # 中間体 p1〜p4: 半透明
    for poly in polys[:4]:
        poly.set_alpha(0.6)

    # 生成物 p13〜p24: facecolor は母体に合わせ、edgecolor とハッチで2色表現
    for i, poly in enumerate(polys[4:]):
        poly.set_alpha(0.5)
        poly.set_hatch(hatches[i])
        poly.set_edgecolor(product_edgecolors[i])
        poly.set_linewidth(0.6)

    # 中間体合計線（中間体層の頂部）
    ax.plot(
        progress,
        p_intermediate_total,
        color="gray",
        linestyle="-",
    )
    x0 = progress[t_max_idx]
    y0 = p_intermediate_total[t_max_idx]

    # y = 0 まで垂線を下ろす（緑）
    ax.plot(
        [x0, x0],    # x は固定
        [0, y0],     # y を 0 → y0 に
        color="green",
        linestyle="--",
        linewidth=1.0,
    )
    ax.plot(
        [1, 1],    # x は固定
        [0, 1],     # y を 0 → y0 に
        color="purple",
        linestyle="--",
        linewidth=1.0,
    )
    # dialcohol 合計線（生成物まで含めた「積み上げ頂部」）
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
    ax.set_ylim(-0.02, 1.2)
    ax.set_xlim(-0.01, 1.01)
    # ax.set_aspect("equal", adjustable="box")

    # 凡例作成
    handles, legend_labels = ax.get_legend_handles_labels()

    leg = ax.legend(
        handles[::-1],
        legend_labels[::-1],
        loc="upper left",
        ncol=1,
        fontsize=6,
        borderpad=0.2,
        labelspacing=0.2,
        handlelength=1.0,
        handletextpad=0.3,
        borderaxespad=0.2,
        frameon=False,
        framealpha=0.8,
        prop={"family": "monospace", "size": 8}
    )

    # 「1,2,3,4」ラベルだけ緑色にする
    for text in leg.get_texts():
        txt = text.get_text()
        if txt.startswith(r"$\bf{1}$") or txt.startswith(r"$\bf{2}$") \
           or txt.startswith(r"$\bf{3}$") or txt.startswith(r"$\bf{4}$"):
            text.set_color("green")
        else:
            text.set_color("purple")

    plt.tight_layout()
    plt.savefig(save_path, dpi=500, transparent=False)


if __name__ == '__main__':
    evaluate_result(f"data/data_electronic_electrostatic_lumo_regression.pkl")

    df=best_parameter("data/data_electronic_electrostatic_lumo_results.csv")
    plot_3d_contributions(df, "data/contributions_3d.png", highlight_colors={"DSSYKIVIOFKYAU-MHPPCMCBSA-N":"1","UMJJFEIKYGFCAT-HOSYLAQJSA-N":"2","YKFKEYKJGVSEIX-KWYDOPHBSA-N":"3"})
    plot_contribution_bars(df,inchikeys=["DSSYKIVIOFKYAU-MHPPCMCBSA-N","UMJJFEIKYGFCAT-HOSYLAQJSA-N","YKFKEYKJGVSEIX-KWYDOPHBSA-N"],labels=["1","2","3"],ref_inchikey="RWCCWEUUXYIKHB-KHWBWMQUSA-N", save_path="data/contribution_bars.png")
    plot_pair_stacked_contributions(df, target_inchikey="DSSYKIVIOFKYAU-MHPPCMCBSA-N", ref_inchikey="RWCCWEUUXYIKHB-KHWBWMQUSA-N", save_path="data/DSSYKIVIOFKYAU-MHPPCMCBSA-N.png")
    plot_pair_stacked_contributions(df, target_inchikey="UMJJFEIKYGFCAT-HOSYLAQJSA-N", ref_inchikey="RWCCWEUUXYIKHB-KHWBWMQUSA-N", save_path="data/UMJJFEIKYGFCAT-HOSYLAQJSA-N.png")
    plot_pair_stacked_contributions(df, target_inchikey="YKFKEYKJGVSEIX-KWYDOPHBSA-N", ref_inchikey="RWCCWEUUXYIKHB-KHWBWMQUSA-N", save_path="data/YKFKEYKJGVSEIX-KWYDOPHBSA-N.png")
    bar("data/data_electronic_electrostatic_lumo_results.csv")
    
    home = Path.home()
    out_path = home / "contributions"
    make_cube(df,out_path)
    # make_cube_with_sign_markers(df,out_path)
    graph_(df,"data/regression.png")

    reaction_concentration_plot_complex(
        ΔGs=df.set_index("entry").loc[["a1","a2","a3","a4","a13","a14","a23","a24","a31","a32","a41","a42"], "prediction"].to_numpy(),
        T=273.15,a0=1,save_path="data/a.png")
    reaction_concentration_plot_complex(
        ΔGs=df.set_index("entry").loc[["b1","b2","b3","b4","b13","b14","b23","b24","b31","b32","b41","b42"], "prediction"].to_numpy(),
        T=298.15,a0=1,save_path="data/b.png")
    reaction_concentration_plot_complex(
        ΔGs=df.set_index("entry").loc[["c1","c2","c3","c4","c13","c14","c23","c24","c31","c32","c41","c42"], "prediction"].to_numpy(),
        T=298.15,a0=1,save_path="data/c.png")
    reaction_concentration_plot_complex(
        ΔGs=df.set_index("entry").loc[["d1","d2","d3","d4","d13","d14","d23","d24","d31","d32","d41","d42"], "prediction"].to_numpy(),
        T=298.15,a0=1,save_path="data/d.png")
    reaction_concentration_plot_complex(
        ΔGs=df.set_index("entry").loc[["e1","e2","e3","e4","e13","e14","e23","e24","e31","e32","e41","e42"], "prediction"].to_numpy(),
        T=298.15,a0=1,save_path="data/e.png")
    reaction_concentration_plot_complex(
        ΔGs=df.set_index("entry").loc[["f1","f2","f3","f4","f13","f14","f23","f24","f31","f32","f41","f42"], "prediction"].to_numpy(),
        T=298.15,a0=1,save_path="data/f.png")