# 3D Electronic Analysis for Site-Selectivity

![Python 3.10](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![License: MIT](https://img.shields.io/badge/License-MIT-orange)

本リポジトリは、ケトン還元の位置・立体選択性を三次元電子記述子から解析した論文について、採用モデル（current model）を再現するためのコード、凍結入力、照合用結果をまとめたものです。

正式な再現対象は `libs/current_model.py` と `data/current_model/` です。論文で採用していない旧 three-field pipeline や探索結果は、再現経路には含めません。既存の凍結記述子から再現する場合、Gaussian は不要です。

## 最短の再現手順

Conda または Mamba を用意し、リポジトリのルートで次を実行します。

```bash
conda env create -f environment.yml
conda activate 3d-electronic-analysis
```

### 1. 凍結入力を検証

manifest に記録された相対パス、byte 数、SHA-256 と、portable array の基本構造を確認します。

```bash
python libs/current_model.py --verify-inputs-only
```

### 2. 完全な nested LOOCV を再計算

```bash
OMP_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 \
MKL_NUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 \
python libs/current_model.py \
  --workers 4 \
  --no-excel-refresh \
  --skip-contribution-cubes
```

`--workers` は 1–20 の範囲で計算機に合わせて変更できます。`--no-excel-refresh` は、Git に含まれる凍結 metadata と応答値だけを使う canonical reproduction を指定します。`--skip-contribution-cubes` が省略するのは Git 対象外の表示用 cube だけで、nested LOOCV、予測、要約表、主要な確定 PNG は再生成されます。

寄与 cube の共通 viewer helper は任意機能です。`open_gaussview_surface.sh` と AppleScript 自動操作は macOS + GaussView 専用で、統計モデル・図表の再現には使用しません。

完全再現では `--skip-nested` を使用しないでください。このオプションは保存済み outer prediction を再利用する確認用であり、nested LOOCV の再計算にはなりません。

### 3. 空間解析を再計算

空間係数・寄与の補助表と確定図も更新する場合は、完全 nested LOOCV の後に次を実行します。

```bash
python libs/analyze_current_model_spatial_contributions.py \
  --workers 4 \
  --no-excel-refresh
```

### 4. 保存結果を独立に照合

```bash
python scripts/verify_reproduction.py
```

軽量回帰テストも同じ検証関数を使用します。

```bash
python -m unittest discover -s tests -p 'test_*.py'
```

`run_pipeline.ipynb` は上記と同じ正式 CLI を順に呼ぶ、小さな未実行 notebook です。

## 期待される結果

基準値は `data/current_model/expected_metrics.json` に機械可読形式で固定されています。

| 項目 | 期待値 |
| --- | ---: |
| 凍結 metadata | 167 rows |
| 学習データ | 83 rows |
| x/y 外部 diketone を含む予測行列 | 191 rows |
| full-grid 座標 | 4,927 / block |
| 採用 grid | 105 / block |
| 特徴量 | 321 |
| full-training Lasso alpha | 0.01 |
| strict nested outer-LOOCV R² | 0.8037754478 |
| strict nested outer-LOOCV RMSE | 0.5811724454 kcal/mol |
| strict nested outer-LOOCV MAE | 0.4351459951 kcal/mol |
| diketone a–f/x/y semiquantitative RMSE | 13.04710317 percentage points |

数値許容差は同 JSON に定義されています。検証スクリプトは `summary.csv` の基準値に加え、83件の `outer_predictions.csv` から R²、RMSE、MAE を再計算します。また x/y NPZ と CSV の identity 対応、空間係数・効果行列の有限性、fold/holdout/feature identity、主要な係数 norm も独立に照合します。

## 採用モデルの概要

- descriptor blocks: electronic、electrostatic、projected C=O π* orbital
- 各 block: 4,927点の凍結 full grid から fold 内で105点を選び、max/min を加える
- 合計特徴量: `(105 + 2) × 3 = 321`
- estimator: Lasso
- alpha 候補: 1、0.1、0.01、0.001
- validation: 各 outer holdout を scaling、特徴選択、alpha 選択のすべてから除外した strict nested LOOCV
- x/y diketone: prediction matrix のみに追加し、学習、scaling、alpha 選択には使用しない

詳細は `data/current_model/README.md` と `data/current_model/model_specification.json` を参照してください。

## 主なディレクトリ

- `libs/current_model.py`: 採用モデルの正式 runner
- `scripts/verify_reproduction.py`: 凍結入力と保存結果の独立検証
- `tests/test_reproducibility.py`: 軽量回帰テスト
- `data/current_model/inputs/`: portable な凍結入力と checksum manifest
- `data/current_model/`: compact な数値結果と計算仕様
- `data/validation/current_model/`: 論文確認用の確定 PNG
- `examples/gaussian/`: 新規記述子生成方法を示す最小入力例
- `docs/DATA_POLICY.md`: Git、再生成物、外部 SSD の保管方針

## Gaussian が必要になる場合

採用モデルの再現は凍結 NPZ/CSV/JSON だけで完結します。Gaussian、`formchk`、`cubegen`、Multiwfn が必要なのは、新しい分子・系列の descriptor を生成するときだけです。Gaussian 本体とライセンス、Multiwfn は本リポジトリに含まれません。

新規分子の標準経路は、RDKit ETKDG + MMFF94 conformer生成、B3LYP-D3(BJ)/def2-SVP opt+freq、`wB97XD/def2-TZVP` SMD(MeOH) single point、`formchk` / `cubegen`、Multiwfn による C=O projected orbital 構築です。具体的な小規模入力と期待値は `examples/gaussian/` にあります。

大容量計算ファイルはリポジトリ外の保存領域に置き、その場所を shell 変数 `MOLECULES_ROOT` から `--molecule-root` へ明示的に渡します。低水準の Gaussian helper も同じ環境変数を既定の出力ルートとして認識しますが、正式な外部系列 runner では CLI 引数を記録に残します。

```bash
export MOLECULES_ROOT="<external-storage>/molecules"
SERIES=z

python libs/predict_external_diketone.py \
  --series "$SERIES" \
  --run-quantum \
  --molecule-root "${MOLECULES_ROOT}/${SERIES}_series"
```

新しい系列の descriptor cache と予測は、既定では Git 対象外の `data/validation/external_diketones/` に出力されます。論文入力として採用済みの x/y 系列を意図的に更新する場合だけ `--promote-inputs` を付け、続けて input manifest と model specification を更新・検証します。その他の系列を採用するには、対象系列の定義と Git policy を先に明示的に変更してください。

Gaussian を標準コマンド名以外で起動する場合は `GAUSSIAN_RUN_COMMAND`、補助実行ファイルには `FORMCHK_EXECUTABLE` / `CUBEGEN_EXECUTABLE` / `MULTIWFN_EXECUTABLE` を設定します。再利用可能な計算を別領域に置く場合のみ `GAUSSIAN_BACKUP_ROOT` を使います。CPU・メモリは `GAUSSIAN_NUM_THREADS` と `GAUSSIAN_MEMORY_GB` で明示できます。

歴史的 descriptor 生成時の Multiwfn version は元記録から特定できないため、checksum 付き凍結 descriptor を論文再現の canonical source とします。新規計算では Gaussian と Multiwfn の version を run manifest に記録してください。

入力 workbook に対象系列が存在し、Gaussian 関連コマンドが shell から実行できることが前提です。生成した Gaussian log、checkpoint、formatted checkpoint、cube、scratch は Git に追加しません。採用する descriptor だけを portable 形式へ昇格し、manifest と checksum を更新します。

## Legacy について

`libs/dataset.py`、`libs/calc_mol.py`、`libs/calc_grid.py`、`libs/regression.py` を順に実行する旧 pipeline と、`data/data.pkl`、`data/datafeat.csv`、旧 regression export、EDA/test 出力は、検討過程の superseded artefact です。論文採用モデルの入力・評価値を定義しません。一部の共通描画関数は current model から利用されますが、旧 pipeline 自体を実行する必要はありません。

保管区分、SSD archive、Git 履歴の注意は `docs/DATA_POLICY.md` にまとめています。

## 論文

Daimon Sakaguchi, Taisei Kawasaki, Mayu Itakura, Chihiro Tada, and Hiroaki Gotoh, *Kinetics-Based Framework for Predicting Site- and Facial-Selectivity in Ketone Reductions*, 2026 (submitted).

## License

This project is available under the [MIT License](LICENSE.txt).
