# Frozen current-model inputs

このディレクトリには、採用 current model を Gaussian 再計算なしで再現するための portable な凍結入力を置きます。`input_manifest.csv` に列挙されたファイルが正式入力です。

## ファイル

- `model_arrays.npz`: electronic、electrostatic、orbital の raw array と、各 block 4,927点の整数 grid 座標
- `model_metadata.csv`: 167分子の identity、表示名、SMILES、温度、実験応答、test 区分
- `model_provenance.json`: descriptor version、block 順序、shape、conformer/量子化学条件、由来
- `train_rows.csv`: 凍結した83学習点と row index、InChIKey、応答値
- `projected_orbital_fullgrid_2bohr.npz`: projected C=O π* の full-grid cache
- `projected_orbital_manifest.csv`: conformer weight と orbital provenance。パスは package 内の相対表現のみを使用
- `display_geometries.json`: optional contribution cube を再生成するための表示 geometry
- `external_diketones/x_series/`: x 系列12経路の metadata と三 block array
- `external_diketones/y_series/`: y 系列12経路の metadata と三 block array。両系列の `input_rows.csv` は `entry,name,SMILES,InChIKey,temperature,test` の6列だけを持ち、RDKit object や空の旧列は保存しない
- `input_manifest.csv`: 正式入力11ファイルの相対パス、byte 数、SHA-256

`model_input_bundle.pkl` は portable package への移行元となった superseded legacy bundle です。Git 対象でも正式 runtime input でもなく、現行 runner は `model_arrays.npz`、`model_metadata.csv`、`model_provenance.json` を読みます。

## 検証

リポジトリのルートで次を実行します。

```bash
python libs/current_model.py --verify-inputs-only
python scripts/verify_reproduction.py
```

前者は manifest、checksum、必須 shape、83学習点の identity alignment を確認します。後者はさらに portable feature matrix が191 rows × 321 featuresになること、x/y の CSV/NPZ identity、保存済み nested-LOOCV 指標、空間解析の fold/feature/holdout identity と主要な数値 invariant を確認します。

## 更新規則

- ファイルを手作業で部分編集しない。
- identity alignment には entry 番号ではなく InChIKey を使う。
- manifest path はこのディレクトリからの相対パスにし、個人環境の絶対パスを含めない。
- NPZ は `allow_pickle=False` で読める数値・文字列 array に限定する。
- 更新時は全ファイルの byte 数と SHA-256、`expected_metrics.json`、再現結果を同時に検証する。
- x/y descriptor は prediction 用であり、83学習点や model selection に混入させない。

新しい分子の Gaussian 計算はリポジトリ外で行い、採用する descriptor だけを portable input に昇格します。詳細はルート `README.md` と `docs/DATA_POLICY.md` を参照してください。

歴史的 descriptor 生成時の Multiwfn version は元データに記録がなく、事後に特定していません。このため論文値の canonical source は checksum 付き凍結 descriptor です。新規 descriptor 生成では、使用した Multiwfn version を run manifest に必ず記録してください。
