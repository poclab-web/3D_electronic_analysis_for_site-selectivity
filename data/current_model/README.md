# 採用 current model

このディレクトリは、論文で採用した current model の portable input、計算仕様、compact result をまとめた自己完結パッケージです。実行時に旧 `analysis_runs/`、legacy pickle、Gaussian archive を参照しません。

## 再現の流れ

1. `inputs/input_manifest.csv` により凍結入力の byte 数と SHA-256 を検証する。
2. 167件の凍結 metadata・三記述子 block を読み、x/y diketone 24経路を prediction matrix のみに追加する。
3. 凍結した83学習点について、全83 fold の strict nested outer-LOOCV を再計算する。
4. full-training model、diketone 予測、compact table、確定 PNG を再生成する。
5. `scripts/verify_reproduction.py` で保存結果を基準値と照合する。

各 outer fold では holdout を除外したデータだけから raw-grid scale、空間 grid、summary-feature scale、Lasso alpha を決定します。完全再現時に `--skip-nested` は使用しません。

## ディレクトリ構成

- `inputs/`: portable array、metadata、83学習点、x/y 外部入力、provenance、checksum manifest
- `descriptor_coordinate_alignment_audit.csv`: electrostatic grid の欠損1座標を electronic grid に合わせてゼロ埋めした移行時 audit
- このディレクトリ直下: 予測、係数、nested-LOOCV 指標、diketone 評価、計算仕様
- `comparators/`: 採用モデルとの比較に必要な小容量 summary
- `spatial_analysis/`: 係数・空間寄与の compact table と NPZ
- `../validation/current_model/`: 確定 PNG
- `libs/current_model_support/`: 新しい外部分子の descriptor を生成するときに使用する helper

## モデル仕様と期待値

- training observations: 83
- descriptor blocks: electronic、electrostatic、projected C=O π* orbital
- raw coordinates: 4,927 / block
- selected grids: 105 / block
- features: 321（各 block の105 grid + max/min）
- estimator: Lasso
- alpha candidates: 1、0.1、0.01、0.001
- full-training selected alpha: 0.01
- strict nested outer-LOOCV R²: 0.8037754478337535
- strict nested outer-LOOCV RMSE: 0.5811724454040451 kcal/mol
- strict nested outer-LOOCV MAE: 0.43514599510153823 kcal/mol
- diketone evaluation: a–f と外部 x/y。g は評価対象外
- a–f/x/y semiquantitative RMSE: 13.047103174052545 percentage points（18 quantities）

x/y は予測対象としてのみ追加され、学習、fold 内 scaling、特徴選択、alpha 選択には入りません。機械可読な基準値と許容差は `expected_metrics.json`、完全な仕様は `model_specification.json` にあります。

## 正式な再現コマンド

リポジトリのルートで実行します。

```bash
python libs/current_model.py --verify-inputs-only

OMP_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 \
MKL_NUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 \
python libs/current_model.py \
  --workers 4 \
  --no-excel-refresh \
  --skip-contribution-cubes

python libs/analyze_current_model_spatial_contributions.py \
  --workers 4 \
  --no-excel-refresh

python scripts/verify_reproduction.py
```

`--no-excel-refresh` は、この package 内の凍結 metadata と response をそのまま使う canonical reproduction です。これを省略すると、`data/Details_of_experimental_results.xlsx` から InChIKey でラベルと実験値をメモリ上で再同期し、差分 audit を出力します。凍結 input 自体は書き換えません。

`--skip-contribution-cubes` は optional な498個の表示用 cube の生成だけを省略します。nested LOOCV と Git 対象の compact result / final PNG には影響しません。

寄与 cube 用の shell / AppleScript viewer helper は任意の macOS + GaussView 補助機能であり、モデル再計算や確定図の生成には不要です。

## 主な出力

- `summary.csv`: 採用モデルの主要指標
- `outer_predictions.csv`: 83 fold の strict nested outer prediction
- `fulltrain_inner_alpha_path.csv`: full-training inner-LOOCV alpha path
- `nonzero_coefficients.csv`: 非ゼロ係数
- `fulltrain_predictions_and_contributions.csv`: fitted value と block 寄与
- `diketone_predictions_by_outer_model.csv`: 各 outer model の diketone 予測
- `diketone_semiquant_detail.csv`: a–f/x/y の実験値との照合
- `diketone_primary8_xy_outer83_68_interval.csv`: outer-model uncertainty
- `model_comparison_current_vs_orbital_free.csv`: comparator table
- `model_specification.json`: 凍結計算仕様
- `expected_metrics.json`: verify が用いる期待値と許容差

## Gaussian と新規 descriptor

上記の再現では Gaussian は不要です。Gaussian、`formchk`、`cubegen`、Multiwfn が必要なのは、新しい分子・系列に対する記述子を作る場合だけです。大容量計算ファイルはリポジトリ外に置き、外部保存領域を `MOLECULES_ROOT` に設定したうえで、CLI に明示します。標準コマンド名を使わない場合は `FORMCHK_EXECUTABLE`、`CUBEGEN_EXECUTABLE`、`MULTIWFN_EXECUTABLE` を設定します。

```bash
export MOLECULES_ROOT="<external-storage>/molecules"
SERIES=z

python libs/predict_external_diketone.py \
  --series "$SERIES" \
  --run-quantum \
  --molecule-root "${MOLECULES_ROOT}/${SERIES}_series"
```

未採用系列は Git 対象外の validation cache に出力されます。凍結入力への書込みはレビュー済み x/y 系列に `--promote-inputs` を明示した場合だけ許可し、その後に checksum manifest と model specification を再検証します。

Gaussian artefact や一時 cache は Git 対象外です。採用する descriptor を portable NPZ/CSV/JSON に変換してから、manifest の byte 数と SHA-256 を更新してください。

## Legacy

旧 three-field model、過去の benchmark、探索 run は比較検討資料であり、本 package の runtime dependency ではありません。保管区分は `docs/DATA_POLICY.md` に従います。
