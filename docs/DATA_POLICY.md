# Data and artefact policy

このリポジトリは、論文で採用した current model を第三者が再実行・検証するための最小構成を保持します。探索過程の全成果物や Gaussian の作業ディレクトリを保存する場所ではありません。

## Git に含めるもの

- 再現に必要な Python ソース、設定、依存関係情報、実行手順。
- 実験値の原本 `data/Details_of_experimental_results.xlsx`。
- `data/current_model/inputs/` の portable な凍結入力。対象形式は NPZ、CSV、JSON、Markdown で、pickle は対象外です。
- `data/current_model/` の accepted model に対応する小容量の CSV、JSON、NPZ、README。予測値、係数、評価指標、空間解析の要約など、結果を照合するためのファイルを含みます。
- `data/validation/current_model/` 直下と `data/validation/current_model/spatial_analysis/` の確定 PNG。
- `examples/gaussian/` の最小例。Git に含められるのは Gaussian 入力 `.gjf`、説明 `.md`、期待値・provenance `.json` だけです。計算ログ、checkpoint、formatted checkpoint、cube、scratch は含めません。

凍結入力の manifest には相対パス、サイズ、SHA-256 を記録します。ユーザー名、ホスト名、マウント位置などの個人環境に依存する絶対パスは、入力、結果、manifest、README のいずれにも記録しません。

## Git に含めず再生成するもの

- `data/current_model/work/` 以下の一時出力。
- pickle bundle、conformer cache、Gaussian cube、寄与 cube、cubeごとに生成される viewer launcher、実行ログ。再利用可能な共通 viewer helper source は Git に含めます。
- `data/validation/external_diketones/` の一時 cache と、凍結入力に重複するコピー。
- `data/eda/`、`data/test/`、旧 validation の探索図・中間表。
- 仮想環境、Python cache、テスト cache、Office 一時ファイル。

再現実行は Git にある凍結入力だけを読み、accepted model の小容量結果と確定図を再生成できる形にします。再生成物を更新する場合は、コードと入力の差分、乱数 seed、ソフトウェア版、出力照合結果も同じ変更で更新します。

## 外部 SSD のみに保管するもの

- `examples/gaussian/` を除く Gaussian の全入力・出力・scratch と、元の分子計算ディレクトリ。
- superseded な大容量 tabular export、pickle、過去モデル、探索 run。
- manuscript、Supporting Information、editable figure source、内部 report、memo。
- 第三者論文 PDF。Git には DOI、正式な引用情報、必要に応じて BibTeX のみを置きます。

SSD archive には、元の相対位置、用途、ファイル数、総 byte 数、SHA-256、archive 日時を記した manifest を添付します。ローカル側を整理する前に、ファイル数・byte 数と checksum を照合してください。重要な生データと原稿については、SSD 一台だけを唯一のバックアップにしないでください。

## データの昇格手順

1. Gaussian 計算と探索 run はリポジトリ外で実行する。
2. 論文で採用する入力だけを portable な NPZ、CSV、JSON に変換し、個人絶対パスを除去する。
3. manifest のサイズと SHA-256 を更新し、凍結入力から accepted result を再計算する。
4. 数値照合と図の生成を確認してから、小容量結果と確定 PNG のみを stage する。
5. SSD archive の検証後、superseded な tracked file は Git index から外す。`.gitignore` の追加だけでは、すでに tracked のファイルは untrack されません。

## Git 履歴に関する注意

過去の main 履歴には、commit 済みの Gaussian cube/checkpoint、旧 notebook、仮想環境ファイルなどが残っています。作業ツリーから削除したり `.gitignore` に追加したりしても、その既存履歴は小さくなりません。2026-07-21 時点で main だけを bundle 化した圧縮サイズは約166 MiBです。

この作業環境のローカル `.git` object store は約27 GiBですが、ここには作業支援ツールのローカル snapshot ref と一時履歴も含まれます。したがって27 GiBを GitHub上の main やfresh cloneのサイズとして扱わないでください。ローカル tool refを整理対象に含める場合も、実行中セッションでは削除しません。

公開前に履歴を軽量化する場合は、検証済みの完全 backup を作成した上で `git filter-repo` または同等手段による履歴 rewrite を別作業として実施します。履歴 rewrite は commit ID を変更し、通常は force-push と既存 clone の再取得を伴うため、共同作業者の合意なしには行いません。
