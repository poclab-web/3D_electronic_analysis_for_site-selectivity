import glob
from itertools import product
import os
from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from rdkit.Chem import PandasTools
import time

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
    start=time.time()
    df=pd.read_pickle(path)
    print(time.time()-start)
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
    print(path)
    best_cv_column=pd.read_csv(path,index_col=0)["cv_RMSE"].idxmin()
    # print(best_cv_column)
    coef=pd.read_csv(path.replace("_results.csv","_regression.csv"), index_col=0)
    coef = coef[[best_cv_column.replace("cv", "electronic_coef"), best_cv_column.replace("cv", "electrostatic_coef"), best_cv_column.replace("cv", "lumo_coef")]]
    coef.columns = ["electronic_coef", "electrostatic_coef","lumo_coef"]
    df=pd.read_pickle(path.replace("_results.csv","_regression.pkl"))
    start=time.time()
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
    print("time",time.time()-start)

    df["cv"]=df[best_cv_column]#<df["ΔΔG.expt."]
    df["prediction"]=df[best_cv_column.replace("cv","prediction")]
    # df["er.prediction"]=100/(1+np.exp(df["prediction"]/1.99/df["temperature"]/0.001))
    # df["er.cv"]=100/(1+np.exp(df["cv"]/1.99/df["temperature"]/0.001))
    df["regression"]=df[best_cv_column.replace("cv","regression")]
    df["cv_error"]=df["cv"]-df["ΔΔG.expt."]
    df["prediction_error"]=df["prediction"]-df["ΔΔG.expt."]
    #df[["electronic_cont","electrostatic_cont","lumo_cont"]]=df[["electronic_cont","electrostatic_cont","lumo_cont"]]-df[df["InChIKey"]=="RWCCWEUUXYIKHB-KHWBWMQUSA-N"][["electronic_cont","electrostatic_cont","lumo_cont"]].values
    # df = df.reindex(df[["prediction_error","cv_error"]].abs().sort_values(ascending=False).index)
    df_=df[["entry","name","SMILES","InChIKey","ΔΔG.expt.","electronic_cont","electrostatic_cont","lumo_cont","regression","prediction","cv","prediction_error","cv_error"]].fillna("NAN")#.sort_values(["cv_error","prediction_error"])
    PandasTools.AddMoleculeColumnToFrame(df_, "SMILES")
    path=path.replace(".pkl",".xlsx")
    PandasTools.SaveXlsxFromFrame(df_,path.replace("_results.csv","_regression.xlsx"), size=(100, 100))
    return df#[["ΔΔG.expt.","regression","prediction","cv"]]


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
    plt.scatter([],[],label="$\mathrm{RMSE_{regression}}$"+f" = {rmse:.2f}"
                   +"\n$r^2_{\mathrm{regression}}$ = " + f"{r2:.2f}",c="black",linewidths=0,  alpha=0.5, s=10)
    
    rmse=nan_rmse(df["cv"].values,df["ΔΔG.expt."].values)
    r2=nan_r2(df["cv"].values,df["ΔΔG.expt."].values)
    plt.scatter([],[],label="$\mathrm{RMSE_{LOOCV}}$"+f" = {rmse:.2f}"
                   +"\n$r^2_{\mathrm{LOOCV}}$ = " + f"{r2:.2f}",c="dodgerblue",linewidths=0,  alpha=0.6, s=10)
    
    # rmse=nan_rmse(df["prediction"].values,df["ΔΔG.expt."].values)
    # r2=nan_r2(df["prediction"].values,df["ΔΔG.expt."].values)
    # plt.scatter([],[],label="$\mathrm{RMSE_{test}}$"+f" = {rmse:.2f}"
                #    +"\n$r^2_{\mathrm{test}}$ = " + f"{r2:.2f}",c="red",linewidths=0,  alpha=0.8, s=10)

    plt.scatter(df["ΔΔG.expt."],df["cv"],c="dodgerblue",linewidths=0,s=10,alpha=0.6)
    # plt.scatter(df["ΔΔG.expt."],df["prediction"],c="red",linewidths=0,s=10,alpha=0.8)
    plt.xlabel(r"$\Delta\Delta G^{\ddagger}_{\mathrm{expt}}$ [kcal/mol]")
    plt.ylabel(r"$\Delta\Delta G^{\ddagger}_{\mathrm{predict}}$ [kcal/mol]")
    plt.legend(loc='lower right', fontsize=5, ncols=1)

    plt.text(-3.6, 3.6, "$\mathit{N}$"+f' = {len(df[df["test"]==0])}',# transform=ax.transAxes, 
                fontsize=10, verticalalignment='top')

    plt.tight_layout()
    plt.savefig(path.replace(".pkl",".png"),dpi=500,transparent=True)
    # df = df.reindex(df["error"].abs().sort_values(ascending=False).index)

# def bar():
#     path="/Users/mac_poclab/PycharmProjects/CoMFA_model/dataset/"
#     cbs=pd.read_csv(path+"cbs_electronic_electrostatic_results.csv", index_col=0)
#     dip=pd.read_csv(path+"DIP_electronic_electrostatic_results.csv", index_col=0)
#     ru=pd.read_csv(path+"alpine_borane_electronic_electrostatic_results.csv", index_col=0)

#     left=np.arange(3.0)*4
    
#     array=np.array([cbs.filter(regex=r'PLS [+-]?\d+ cv',axis=0).max()["cv_r2"],
#                     dip.filter(regex=r'PLS [+-]?\d+ cv',axis=0).max()["cv_r2"],
#                     ru.filter(regex=r'PLS [+-]?\d+ cv',axis=0).max()["cv_r2"]])
#     plt.figure(figsize=(4.8, 3.2))
#     plt.bar(left,array,color="red",label='PLS',alpha=0.25)
#     for i, v in enumerate(array):
#         plt.text(left[i], v + 0.05, f"{v:.2f}", ha='center', fontsize=8)
#     left+=0.9
#     print(array)
#     array=np.array([cbs.filter(regex=r"^Ridge .{0,} cv",axis=0).max()["cv_r2"],
#                     dip.filter(regex=r"^Ridge .{0,} cv",axis=0).max()["cv_r2"],
#                     ru.filter(regex=r"^Ridge .{0,} cv",axis=0).max()["cv_r2"]])
#     print(array)
#     plt.bar(left,array,color="red",label='Ridge',alpha=0.5)
#     for i, v in enumerate(array):
#         plt.text(left[i], v + 0.05, f"{v:.2f}", ha='center', fontsize=8)
#     left+=0.9

#     array=np.array([cbs.filter(regex=r"^ElasticNet .{0,} cv",axis=0).max()["cv_r2"],
#                     dip.filter(regex=r"^ElasticNet .{0,} cv",axis=0).max()["cv_r2"],
#                     ru.filter(regex=r"^ElasticNet .{0,} cv",axis=0).max()["cv_r2"]])
#     plt.bar(left,array,color="red",label='Elastic Net',alpha=0.75)
#     for i, v in enumerate(array):
#         plt.text(left[i], v + 0.05, f"{v:.2f}", ha='center', fontsize=8)
#     left+=0.9

#     array=np.array([cbs.filter(regex=r"^Lasso .{0,} cv",axis=0).max()["cv_r2"],
#                     dip.filter(regex=r"^Lasso .{0,} cv",axis=0).max()["cv_r2"],
#                     ru.filter(regex=r"^Lasso .{0,} cv",axis=0).max()["cv_r2"]])
#     plt.bar(left,array,color="red",label='Lasso',alpha=1)
#     for i, v in enumerate(array):
#         plt.text(left[i], v + 0.05, f"{v:.2f}", ha='center', fontsize=8)
    
#     label = [r"$\mathit{(S)}$-CBS", r"$\mathit{(+)}$-DIP-Cl", r"$\mathit{(S)}$-alpine borane"]
#     plt.bar(left-1.35, 0, tick_label=label, align="center")
    
#     plt.axhline(0, color='black', linewidth=1.0)  # y=0の枠線
#     plt.gca().spines['top'].set_visible(False)
#     plt.gca().spines['right'].set_visible(False)
#     plt.gca().spines['left'].set_visible(False)
    
#     plt.grid(axis='y', color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
#     plt.gca().xaxis.set_ticks_position('none')  # 横軸の目盛り線を消す
#     plt.gca().yaxis.set_ticks_position('none')  # 横軸の目盛り線を消す
    
#     plt.legend(ncol=4, bbox_to_anchor=(0.5, 1.01), loc='lower center', frameon=True)
#     # plt.xlabel("Dataset")
#     plt.ylabel("$r^2_{\mathrm{LOOCV}}$")
#     plt.yticks(np.arange(0, 1.1, 0.5))
#     plt.tight_layout()
#     plt.savefig(path+"results.png",dpi=500,transparent=False)
def bar():
    path = "data/"
    cbs = pd.read_csv(path + "data_electronic_electrostatic_lumo_results.csv", index_col=0)
    #dip = pd.read_csv(path + "cleaned_electronic_electrostatic_lumo_results.csv", index_col=0)
    #ru = pd.read_csv(path + "cleaned_electronic_electrostatic_lumo_results.csv", index_col=0)

    dataset_labels = [r"$\mathit{(S)}$-CBS"]#, r"$\mathit{(+)}$-DIP-Cl", r"$\mathit{(S)}$-alpine borane"]
    base_x = np.arange(1) * 6  # 基準となるx位置

    models = [
        (r'PLS [+-]?\d+ cv', "tab:red", 'PLS'),
        (r'^Ridge .{0,} cv', "tab:orange", 'Ridge'),
        (r'^ElasticNet .{0,} cv', "tab:green", 'Elastic Net'),
        (r'^Lasso .{0,} cv', "tab:blue", 'Lasso'),
        (r'^LassoLars .{0,} cv', "tab:purple", 'LassoLars'),
        (r'^OMP .{0,} cv', "tab:brown", 'OMP'),
    ]

    fig, ax1 = plt.subplots(figsize=(4.8, 3.2))
    ax2 = ax1.twinx()

    color_r2 = "red"
    color_rmse = "blue"

    handles = []  # for custom legend
    labels = []

    for model_idx, (regex, alpha, label) in enumerate(models):
        x_positions = base_x + model_idx * 0.9

        r2_array = np.array([
            cbs.filter(regex=regex, axis=0).max()["cv_r2"],
            #dip.filter(regex=regex, axis=0).max()["cv_r2"],
            #ru.filter(regex=regex, axis=0).max()["cv_r2"]
        ])
        rmse_array = np.array([
            cbs.filter(regex=regex, axis=0).min()["cv_RMSE"],
            #dip.filter(regex=regex, axis=0).min()["cv_RMSE"],
            #ru.filter(regex=regex, axis=0).min()["cv_RMSE"]
        ])
        print(r2_array)
        b = ax2.bar(x_positions, rmse_array, color=alpha, alpha=1, width=0.4,label=label+" RMSE")

        s = ax1.scatter(x_positions, r2_array, color=alpha, alpha=1,label=label+r" $r^2$",  facecolor='None')
        
        handles.append(s)
        labels.append(label )
        handles.append(b)
        labels.append(label )

    ax1.set_ylabel(r"$r^2_{\mathrm{LOOCV}}$")#, color=color_r2)
    ax1.set_yticks(np.arange(0, 1.1, 0.2))
    ax1.set_ylim(-1, 1)
    ax1.tick_params(axis='y')#, colors=color_r2)

    ax2.set_ylabel("RMSE"+r"$_{\mathrm{LOOCV}}$"+ "[kcal/mol]")#, color=color_rmse)
    ax2.set_ylim(0, 2)
    ax2.tick_params(axis='y')#, colors=color_rmse)

    # Set dataset labels at center of grouped bars
    mid_x = base_x + 1.35  # shift to middle of all bars
    ax1.set_xticks(mid_x)
    ax1.set_xticklabels(dataset_labels)

    #ax1.axhline(0, color='black', linewidth=1.0)
    # ax1.spines['top'].set_visible(False)
    # ax1.spines['right'].set_visible(False)
    # ax1.spines['left'].set_visible(False)

    ax1.xaxis.set_ticks_position('none')
    # ax1.yaxis.set_ticks_position('none')

    plt.legend(handles=handles, ncol=4, bbox_to_anchor=(0.5, 1.02), loc='lower center',frameon=True,fontsize=7.5)
    fig.tight_layout()
    fig.savefig(path + "results_with_rmse.png", dpi=500, transparent=False)

def reaction_concentration_plot_complex(
    ΔGs, T=298.15, a0=100,
    save_path="simulation_complex.png"
):
    # 定数
    kB = 1.380649e-23  # [J/K]
    h = 6.62607015e-34  # [J·s]
    R = 1.987e-3  # [kcal/(mol·K)]
    """
    複雑な分岐反応の濃度時間変化をプロットする。

    ΔGs: list of 4 floats
        ΔG1〜ΔG4（A→Pi）[kcal/mol]
    ΔGs_p: list of 8 floats
        ΔG13p, ΔG14p, ΔG23p, ΔG24p, ΔG31p, ΔG32p, ΔG41p, ΔG42p（Pi→Pij′）[kcal/mol]
    T: float
        温度 [K]
    a0: float
        初期[A]濃度
    save_path: str
        プロット保存先
    """

    # Eyring式
    def k(ΔG):
        return (kB * T / h) * np.exp(-ΔG / (R * T))

    # 速度定数
    k1, k2, k3, k4, k13p, k14p, k23p, k24p, k31p, k32p, k41p, k42p = [k(ΔG) for ΔG in ΔGs]

    # 中間体分解速度
    k1p_sum = k13p + k14p
    k2p_sum = k23p + k24p
    k3p_sum = k31p + k32p
    k4p_sum = k41p + k42p
    ka = k1 + k2 + k3 + k4

    # 時間軸
    # t = np.linspace(0, 10000 / np.max([k1, k2, k3, k4, k13p, k14p, k23p, k24p, k31p, k32p, k41p, k42p]), 100000)
    t=np.logspace(-10,10,1000,base=10)/np.max([k1, k2, k3, k4, k13p, k14p, k23p, k24p, k31p, k32p, k41p, k42p])
    # Aの濃度
    a = a0 * np.exp(-ka * t)

    # Piの濃度（中間体）
    def p_i(k_i, k_ip_sum):
        return (k_i * a0 / (k_ip_sum - ka)) * (np.exp(-ka * t) - np.exp(-k_ip_sum * t))

    p1 = p_i(k1, k1p_sum)
    p2 = p_i(k2, k2p_sum)
    p3 = p_i(k3, k3p_sum)
    p4 = p_i(k4, k4p_sum)

    # 各Piの最大値とその時刻
    def find_max(t, p):
        idx = np.argmax(p)
        return idx, p[idx]

    t1_max, p1_max = find_max(t, p1)
    t2_max, p2_max = find_max(t, p2)
    t3_max, p3_max = find_max(t, p3)
    t4_max, p4_max = find_max(t, p4)
    t_max, p_max = find_max(t, p1 + p2 + p3 + p4)

    # Pij'合計（生成物合算）
    def pij_total(k_i, k_ijp, k_ip_sum):
        term1 = (1 - np.exp(-ka * t)) / ka
        term2 = (1 - np.exp(-k_ip_sum * t)) / k_ip_sum
        return (k_i * k_ijp * a0 / (k_ip_sum - ka)) * (term1 - term2)
    
    p13=pij_total(k1, k13p, k1p_sum)+pij_total(k3, k31p, k3p_sum)
    p14=pij_total(k1, k14p, k1p_sum)+pij_total(k4, k41p, k4p_sum)
    p23=pij_total(k2, k23p, k2p_sum)+pij_total(k3, k32p, k3p_sum)
    p24=pij_total(k2, k24p, k2p_sum)+pij_total(k4, k42p, k4p_sum)
    pp_total = (p13 + p14 + p23 + p24)

    # プロット
    plt.figure(figsize=(4, 4), facecolor="none")
    
    t=p1/2+p2/2+p3/2+p4/2+pp_total
    plt.plot(t, p1, label=r"$\bf{1}$"+f', max={p1_max:.3f}', linestyle='--', color='red')
    # plt.plot(t[t1_max], p1_max, 'o', color='red', label=None)

    plt.plot(t, p2, label=r"$\bf{2}$"+f', max={p2_max:.3f}', linestyle='--', color='tab:pink')
    # plt.plot(t[t2_max], p2_max, 'o', color='tab:pink', label=None)
    plt.plot(t, a, label='diketone', color='black')
    plt.plot(t, p3, label=r"$\bf{3}$"+f', max={p3_max:.3f}', linestyle='--', color='blue')
    # plt.plot(t[t3_max], p3_max, 'o', color='blue', label=None)

    plt.plot(t, p4, label=r"$\bf{4}$"+f', max={p4_max:.3f}', linestyle='--', color='tab:blue')
    # plt.plot(t[t4_max], p4_max, 'o', color='tab:blue', label=None)
    plt.plot(t, p1+p2+p3+p4, label='intermediate', color='gray', linestyle=':')
    plt.plot(t[t_max], p_max, 'o', color='gray', label=None)
    plt.plot(t, pp_total, label='dialcohol', color='tab:gray')
    plt.plot(t,p13, label=r"$\bf{1-3}$", linestyle='--', color='orange')
    plt.plot(t,p14, label=r"$\bf{1-4}$", linestyle='--', color='yellow')
    plt.plot(t,p23, label=r"$\bf{2-3}$", linestyle='--', color='green')
    plt.plot(t,p24, label=r"$\bf{2-4}$", linestyle='--', color='cyan')

    plt.xlabel('reaction progress [-]')
    plt.ylabel('rate [-]')
    plt.xticks([0,0.5,1])
    plt.yticks([0,0.5,1])
    plt.ylim(-0.02, 1 )  # 濃度の範囲を設定
    plt.xlim(-0.01,  1.0)  # 濃度の範囲を設定
    #plt.xlim(-0.01*np.max(t),  )  # 濃度の範囲を設定
    # plt.title('diketone reaction')
    
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=2)
    plt.tight_layout()
    plt.savefig(save_path)
    #plt.show()


def reaction_concentration_plot_complex(
    ΔGs, T=298.15, a0=100,
    save_path="simulation_complex.png"
):
    # 定数
    kB = 1.380649e-23  # [J/K]
    h = 6.62607015e-34  # [J·s]
    R = 1.987e-3        # [kcal/(mol·K)]
    """
    複雑な分岐反応の濃度時間変化をプロットする。

    ΔGs: list of 12 floats
        ΔG1, ΔG2, ΔG3, ΔG4（A→Pi）[kcal/mol]
        ΔG13p, ΔG14p, ΔG23p, ΔG24p, ΔG31p, ΔG32p, ΔG41p, ΔG42p（Pi→Pij′）[kcal/mol]
    T: float
        温度 [K]
    a0: float
        初期[A]濃度
    save_path: str
        プロット保存先
    """

    # --- Eyring式 ---
    def k(ΔG):
        return (kB * T / h) * np.exp(-ΔG / (R * T))

    # 速度定数（ΔGs は長さ 12 を想定）
    k1, k2, k3, k4, k13p, k14p, k23p, k24p, k31p, k32p, k41p, k42p = [k(ΔG) for ΔG in ΔGs]

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

    # Aの濃度
    a = a0 * np.exp(-ka * t)

    # --- 中間体 Pi の濃度 ---
    def p_i(k_i, k_ip_sum):
        return (k_i * a0 / (k_ip_sum - ka)) * (np.exp(-ka * t) - np.exp(-k_ip_sum * t))

    p1 = p_i(k1, k1p_sum)
    p2 = p_i(k2, k2p_sum)
    p3 = p_i(k3, k3p_sum)
    p4 = p_i(k4, k4p_sum)

    # 各Piの最大値（必要ならそのまま利用可能）
    def find_max(t, p):
        idx = np.argmax(p)
        return idx, p[idx]

    t1_max, p1_max = find_max(t, p1)
    t2_max, p2_max = find_max(t, p2)
    t3_max, p3_max = find_max(t, p3)
    t4_max, p4_max = find_max(t, p4)
    t_max, p_max = find_max(t, p1 + p2 + p3 + p4)

    # --- Pij′（生成物）の濃度 ---
    def pij_total(k_i, k_ijp, k_ip_sum):
        term1 = (1 - np.exp(-ka * t)) / ka
        term2 = (1 - np.exp(-k_ip_sum * t)) / k_ip_sum
        return (k_i * k_ijp * a0 / (k_ip_sum - ka)) * (term1 - term2)

    p13 = pij_total(k1, k13p, k1p_sum) + pij_total(k3, k31p, k3p_sum)
    p14 = pij_total(k1, k14p, k1p_sum) + pij_total(k4, k41p, k4p_sum)
    p23 = pij_total(k2, k23p, k2p_sum) + pij_total(k3, k32p, k3p_sum)
    p24 = pij_total(k2, k24p, k2p_sum) + pij_total(k4, k42p, k4p_sum)
    pp_total = p13 + p14 + p23 + p24

    # --- 反応進行度（x 軸に使う量）---
    # 元コードの t = p1/2 + p2/2 + p3/2 + p4/2 + pp_total を
    # 変数名を変えて保存
    progress = p1 / 2 + p2 / 2 + p3 / 2 + p4 / 2 + pp_total

    # --- プロット（2段の積み上げグラフ）---
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(4, 6), sharex=True, facecolor="none"
    )
    fig.patch.set_alpha(0.0)

    # 1段目: p1〜p4 を p1+p2+p3+p4 に対して積み上げ
    ax1.stackplot(
        progress,
        p1, p2, p3, p4,
        labels=[
            r"$\bf{1}$" + f" (max={p1_max:.3f})",
            r"$\bf{2}$" + f" (max={p2_max:.3f})",
            r"$\bf{3}$" + f" (max={p3_max:.3f})",
            r"$\bf{4}$" + f" (max={p4_max:.3f})",
        ],
        colors=["red", "tab:pink", "blue", "tab:blue"],
        alpha=0.7,
    )
    ax1.plot(
        progress,
        p1 + p2 + p3 + p4,
        color="gray",
        linestyle=":",
        label="intermediate total",
    )
    ax1.plot(progress[t_max], (p1 + p2 + p3 + p4)[t_max], "o", color="gray")
    ax1.set_ylabel("intermediate conc. [-]")
    ax1.set_ylim(-0.02, 1.0)
    ax1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2, fontsize=8)

    # 2段目: p13,p14,p23,p24 を pp_total に対して積み上げ
    ax2.stackplot(
        progress,
        p13, p14, p23, p24,
        labels=[
            r"$\bf{1-3}$",
            r"$\bf{1-4}$",
            r"$\bf{2-3}$",
            r"$\bf{2-4}$",
        ],
        colors=["orange", "yellow", "green", "cyan"],
        alpha=0.7,
    )
    ax2.plot(
        progress,
        pp_total,
        color="tab:gray",
        linestyle="-",
        label="dialcohol total",
    )
    # 参考として、Aの減少も同じ軸に重ねる（必要なければ削除可）
    ax2.plot(
        progress,
        a / a0,  # 見やすいように規格化しておく
        color="black",
        linestyle="--",
        label="diketone (A)",
    )

    ax2.set_xlabel("reaction progress [-]")
    ax2.set_ylabel("product conc. [-]")
    ax2.set_xticks([0, 0.5, 1])
    ax2.set_yticks([0, 0.5, 1])
    ax2.set_xlim(-0.01, 1.0)
    ax2.set_ylim(-0.02, 1.0)
    ax2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path)
    # plt.show()

import numpy as np
import matplotlib.pyplot as plt

def reaction_concentration_plot_complex(
    ΔGs, T=298.15, a0=100,
    save_path="simulation_complex.png"
):
    # 定数
    kB = 1.380649e-23  # [J/K]
    h = 6.62607015e-34  # [J·s]
    R = 1.987e-3        # [kcal/(mol·K)]
    """
    複雑な分岐反応の濃度時間変化をプロットする。

    ΔGs: list of 12 floats
        ΔG1, ΔG2, ΔG3, ΔG4（A→Pi）[kcal/mol]
        ΔG13p, ΔG14p, ΔG23p, ΔG24p, ΔG31p, ΔG32p, ΔG41p, ΔG42p（Pi→Pij′）[kcal/mol]
    T: float
        温度 [K]
    a0: float
        初期[A]濃度
    save_path: str
        プロット保存先
    """

    # --- Eyring式 ---
    def k(ΔG):
        return (kB * T / h) * np.exp(-ΔG / (R * T))

    # 速度定数（ΔGs は長さ 12 を想定）
    k1, k2, k3, k4, k13p, k14p, k23p, k24p, k31p, k32p, k41p, k42p = [k(ΔG) for ΔG in ΔGs]

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

    # Aの濃度
    a = a0 * np.exp(-ka * t)

    # --- 中間体 Pi の濃度 ---
    def p_i(k_i, k_ip_sum):
        return (k_i * a0 / (k_ip_sum - ka)) * (np.exp(-ka * t) - np.exp(-k_ip_sum * t))

    p1 = p_i(k1, k1p_sum)
    p2 = p_i(k2, k2p_sum)
    p3 = p_i(k3, k3p_sum)
    p4 = p_i(k4, k4p_sum)

    # 中間体合計とその最大値
    p_intermediate_total = p1 + p2 + p3 + p4

    def find_max(t, p):
        idx = np.argmax(p)
        return idx, p[idx]

    t_max_idx, p_max = find_max(t, p_intermediate_total)

    # --- 生成物 Pij′ の濃度 ---
    def pij_total(k_i, k_ijp, k_ip_sum):
        term1 = (1 - np.exp(-ka * t)) / ka
        term2 = (1 - np.exp(-k_ip_sum * t)) / k_ip_sum
        return (k_i * k_ijp * a0 / (k_ip_sum - ka)) * (term1 - term2)

    p13 = pij_total(k1, k13p, k1p_sum) + pij_total(k3, k31p, k3p_sum)
    p14 = pij_total(k1, k14p, k1p_sum) + pij_total(k4, k41p, k4p_sum)
    p23 = pij_total(k2, k23p, k2p_sum) + pij_total(k3, k32p, k3p_sum)
    p24 = pij_total(k2, k24p, k2p_sum) + pij_total(k4, k42p, k4p_sum)
    pp_total = p13 + p14 + p23 + p24

    # --- 反応進行度（x 軸に使う量）---
    # 元コードの t = p1/2 + ... を踏襲して別変数に
    progress = p1 / 2 + p2 / 2 + p3 / 2 + p4 / 2 + pp_total

    # --- t_max における p1〜p4 の値を表示 ---
    print("At t_max (intermediate total maximum):")
    print(f"  t_max = {t[t_max_idx]:.3e}")
    print(f"  p1(t_max) = {p1[t_max_idx]:.6f}")
    print(f"  p2(t_max) = {p2[t_max_idx]:.6f}")
    print(f"  p3(t_max) = {p3[t_max_idx]:.6f}")
    print(f"  p4(t_max) = {p4[t_max_idx]:.6f}")

    # --- プロット（1枚の軸に重ねた積み上げ）---
    fig, ax = plt.subplots(figsize=(4, 4), facecolor="none")
    fig.patch.set_alpha(0.0)

    # 1つ目の積み上げ: p1〜p4（中間体）
    ax.stackplot(
        progress,
        p1, p2, p3, p4,
        labels=[r"$\bf{1}$", r"$\bf{2}$", r"$\bf{3}$", r"$\bf{4}$"],
        colors=["red", "tab:pink", "blue", "tab:blue"],
        alpha=0.7,
    )

    # 2つ目の積み上げ: p13, p14, p23, p24（生成物）
    ax.stackplot(
        progress,
        p13, p14, p23, p24,
        labels=[r"$\bf{1-3}$", r"$\bf{1-4}$", r"$\bf{2-3}$", r"$\bf{2-4}$"],
        colors=["orange", "yellow", "green", "cyan"],
        alpha=0.5,
    )

    # 合計の線（中間体 & 生成物）
    ax.plot(progress, p_intermediate_total, label="intermediate total",
            color="gray", linestyle=":")
    ax.plot(progress[t_max_idx], p_intermediate_total[t_max_idx],
            "o", color="gray")

    ax.plot(progress, pp_total, label="dialcohol total",
            color="tab:gray", linestyle="-")

    # A の減少も参考として描画（規格化）
    ax.plot(progress, a / a0, label="diketone (A)",
            color="black", linestyle="--")

    ax.set_xlabel("reaction progress [-]")
    ax.set_ylabel("concentration [-]")
    ax.set_xticks([0, 0.5, 1])
    ax.set_yticks([0, 0.5, 1])
    ax.set_ylim(-0.02, 1.0)
    ax.set_xlim(-0.01, 1.0)

    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3, fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path)
    # plt.show()

import numpy as np
import matplotlib.pyplot as plt

def reaction_concentration_plot_complex(
    ΔGs, T=298.15, a0=100,
    save_path="simulation_complex.png"
):
    # 定数
    kB = 1.380649e-23  # [J/K]
    h = 6.62607015e-34  # [J·s]
    R = 1.987e-3        # [kcal/(mol·K)]
    """
    複雑な分岐反応の濃度時間変化をプロットする。

    ΔGs: list of 12 floats
        ΔG1, ΔG2, ΔG3, ΔG4（A→Pi）[kcal/mol]
        ΔG13p, ΔG14p, ΔG23p, ΔG24p, ΔG31p, ΔG32p, ΔG41p, ΔG42p（Pi→Pij′）[kcal/mol]
    T: float
        温度 [K]
    a0: float
        初期[A]濃度
    save_path: str
        プロット保存先
    """

    # --- Eyring式 ---
    def k(ΔG):
        return (kB * T / h) * np.exp(-ΔG / (R * T))

    # 速度定数（ΔGs は長さ 12 を想定）
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

    # Aの濃度
    a = a0 * np.exp(-ka * t)

    # --- 中間体 Pi の濃度 ---
    def p_i(k_i, k_ip_sum):
        return (k_i * a0 / (k_ip_sum - ka)) * (np.exp(-ka * t) - np.exp(-k_ip_sum * t))

    p1 = p_i(k1, k1p_sum)
    p2 = p_i(k2, k2p_sum)
    p3 = p_i(k3, k3p_sum)
    p4 = p_i(k4, k4p_sum)

    p_intermediate_total = p1 + p2 + p3 + p4

    # 中間体合計が最大となる時刻
    t_max_idx = np.argmax(p_intermediate_total)
    t_max = t[t_max_idx]

    # --- 生成物 Pij′ の濃度 ---
    def pij_total(k_i, k_ijp, k_ip_sum):
        term1 = (1 - np.exp(-ka * t)) / ka
        term2 = (1 - np.exp(-k_ip_sum * t)) / k_ip_sum
        return (k_i * k_ijp * a0 / (k_ip_sum - ka)) * (term1 - term2)

    # 対称なペアをまとめた生成物
    p13 = pij_total(k1, k13p, k1p_sum) + pij_total(k3, k31p, k3p_sum)
    p14 = pij_total(k1, k14p, k1p_sum) + pij_total(k4, k41p, k4p_sum)
    p23 = pij_total(k2, k23p, k2p_sum) + pij_total(k3, k32p, k3p_sum)
    p24 = pij_total(k2, k24p, k2p_sum) + pij_total(k4, k42p, k4p_sum)

    pp_total = p13 + p14 + p23 + p24

    # --- 反応進行度（x 軸に使う量）---
    # 元コードの定義を踏襲
    progress = p1 / 2 + p2 / 2 + p3 / 2 + p4 / 2 + pp_total

    # --- t_max における p1〜p4 を print ---
    print("At t_max (intermediate total maximum):")
    print(f"  t_max = {t_max:.3e}")
    print(f"  p1(t_max) = {p1[t_max_idx]:.6f}")
    print(f"  p2(t_max) = {p2[t_max_idx]:.6f}")
    print(f"  p3(t_max) = {p3[t_max_idx]:.6f}")
    print(f"  p4(t_max) = {p4[t_max_idx]:.6f}")

    # --- プロット: 中間体の上に生成物を積み上げ ---
    fig, ax = plt.subplots(figsize=(4, 4), facecolor="none")
    fig.patch.set_alpha(0.0)

    # 色: p1〜p4 と p13〜p24 で対応する色を使う
    base_colors = ["red", "tab:pink", "blue", "tab:blue"]  # p1, p2, p3, p4
    product_colors = base_colors[:]  # p13, p14, p23, p24 も同じ色系

    # stackplot 1回で 8 成分を積み上げ:
    # 下から順に p1, p2, p3, p4, p13, p14, p23, p24
    all_colors = base_colors + product_colors
    labels = [
        r"$\bf{1}$", r"$\bf{2}$", r"$\bf{3}$", r"$\bf{4}$",
        r"$\bf{1-3}$", r"$\bf{1-4}$", r"$\bf{2-3}$", r"$\bf{2-4}$",
    ]

    polys = ax.stackplot(
        progress,
        p1, p2, p3, p4,
        p13, p14, p23, p24,
        colors=all_colors,
        labels=labels,
    )

    # 中間体 (p1〜p4): 半透明
    for poly in polys[:4]:
        poly.set_alpha(0.6)

    # 生成物 (p13〜p24): 同系色＋ハッチで区別
    hatches = ["///", "\\\\", "xx", ".."]  # 1-3, 1-4, 2-3, 2-4 用
    for i, poly in enumerate(polys[4:], start=0):
        poly.set_alpha(0.4)
        poly.set_hatch(hatches[i])
        poly.set_edgecolor(product_colors[i])
        poly.set_linewidth(0.5)

    # 合計の線（中間体 & 生成物）
    ax.plot(
        progress,
        p_intermediate_total,
        label="intermediate total",
        color="gray",
        linestyle=":",
    )
    ax.plot(
        progress[t_max_idx],
        p_intermediate_total[t_max_idx],
        "o",
        color="gray",
    )

    ax.plot(
        progress,
        pp_total,
        label="dialcohol total",
        color="tab:gray",
        linestyle="-",
    )

    # A の減少（規格化）
    ax.plot(
        progress,
        a / a0,
        label="diketone (A)",
        color="black",
        linestyle="--",
    )

    ax.set_xlabel("reaction progress [-]")
    ax.set_ylabel("concentration [-]")
    ax.set_xticks([0, 0.5, 1])
    ax.set_yticks([0, 0.5, 1])
    ax.set_ylim(-0.02, 1.0)
    ax.set_xlim(-0.01, 1.0)

    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3, fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path)
    # plt.show()
import numpy as np
import matplotlib.pyplot as plt

def reaction_concentration_plot_complex(
    ΔGs, T=298.15, a0=100,
    save_path="simulation_complex.png"
):
    # 定数
    kB = 1.380649e-23  # [J/K]
    h = 6.62607015e-34  # [J·s]
    R = 1.987e-3        # [kcal/(mol·K)]
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

    # Aの濃度
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

    # --- プロット ---
    fig, ax = plt.subplots(figsize=(3, 3), facecolor="none")
    fig.patch.set_alpha(0.0)

    # 中間体の色
    c1 = "red"
    c2 = "tab:pink"
    c3 = "blue"
    c4 = "tab:blue"
    base_colors = [c1, c2, c3, c4]

    # 生成物: 対応する 2色を「facecolor×edgecolor+ハッチ」で表現
    #  p13: 1 & 3 → face:red, edge:blue
    #  p14: 1 & 4 → face:red, edge:tab:blue
    #  p23: 2 & 3 → face:tab:pink, edge:blue
    #  p24: 2 & 4 → face:tab:pink, edge:tab:blue
    product_facecolors = [c1, c1, c2, c2]      # face
    product_edgecolors = [c3, c4, c3, c4]      # edge
    hatches = ["///", "\\\\", "xx", ".."]      # 2色感を出す模様

    # stackplot: 下から p1, p2, p3, p4, p13, p14, p23, p24 の順に積み上げ
    all_colors = base_colors + product_facecolors

    labels = [
        r"$\bf{1}$", r"$\bf{2}$", r"$\bf{3}$", r"$\bf{4}$",
        r"$\bf{1-3}$", r"$\bf{1-4}$", r"$\bf{2-3}$", r"$\bf{2-4}$",
    ]

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
    ax.plot(
        progress[t_max_idx],
        p_intermediate_total[t_max_idx],
        "o",
        color="gray",
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
    ax.set_ylim(-0.02, 1.0)
    ax.set_xlim(-0.01, 1.0)

    # 凡例は表示しない
    # ax.legend(...)
    ax.legend(loc="upper center", bbox_to_anchor=(0.4, -0.28), ncol=4, fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path)
    # plt.show()

if __name__ == '__main__':
    start=time.time()
    evaluate_result(f"data/data_electronic_electrostatic_lumo_regression.pkl")
    
    print(time.time()-start)

    df=best_parameter("data/data_electronic_electrostatic_lumo_results.csv")
    # df_dip=best_parameter("/Users/mac_poclab/PycharmProjects/CoMFA_model/dataset/DIP_electronic_electrostatic_results.csv")
    # df_alp=best_parameter("/Users/mac_poclab/PycharmProjects/CoMFA_model/dataset/alpine_borane_electronic_electrostatic_results.csv")
    bar()
    print(df.columns)
    home = Path.home()
    out_path = home / "contributions"
    make_cube(df,out_path)
    # make_cube(df_dip,'/Users/mac_poclab/CoMFA_results/DIP')
    # make_cube(df_alp,'/Users/mac_poclab/CoMFA_results/alp')
    # graph_(df_cbs,"/Users/mac_poclab/PycharmProjects/CoMFA_model/dataset/regression_cbs.png")
    graph_(df,"data/regression.png")
    # graph_(pd.concat([df]),"/Users/mac_poclab/PycharmProjects/3D-electronic_analyzer/data/regression.png")
    print(df[["entry","prediction"]])
    print(df.set_index("entry").loc[["a1","a2","a3","a4","a13","a14","a23","a24","a31","a32","a41","a42"], "prediction"])
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
        ΔGs=df.set_index("entry").loc[["g1","g2","g3","g4","g13","g14","g23","g24","g31","g32","g41","g42"], "prediction"].to_numpy(),
        T=298.15,a0=1,save_path="data/g.png")