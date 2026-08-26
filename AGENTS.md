# AGENTS.md

このファイルは、`master-thesis` リポジトリで作業する Codex 向けの恒久的な指示です。
リポジトリ全体に適用します。ユーザーの明示的な指示が本ファイルより優先されます。

## 作業の基本方針

- 作業開始時に `git status --short` を確認する。
- 未コミットの変更はユーザーの作業として扱い、無関係な変更を上書き・削除しない。
- 調査だけを依頼された場合は、ファイルを変更しない。
- 実装を依頼された場合は、必要な変更、テスト、差分確認、コミットまで行う。
- コミットメッセージは日本語で、Prefix(feat:, fix:, chore:など)をつける。
- `git push`、PR作成は、明示的に依頼された場合だけ行う。
- `git reset --hard`、強制checkout、大量削除などの破壊的操作は行わない。
- 大容量データや既存の実験結果を削除・上書きする前に、必ず対象と影響を確認する。

## 研究上の正式条件

- RCAEval RE1 の正式データ源は Zenodo record `14590730`、DOI `10.5281/zenodo.14590730`、version `v2` とする。
- 正式対象は `re1_ob`、`re1_ss`、`re1_tt` の各125ケース、合計375ケースとする。
- 正式main設定は `configs/main/rcaeval_re1_zenodo_v2.yaml` とする。
- Hugging Face版RCAEvalを正式実験へ再導入しない。
- 正常・異常windowは、それぞれ利用可能な観測点のうち最大600点とする。
- 正式評価は service-level とし、service aggregation は `mean_top3` とする。
- 正式データパスは次を使用する。
  - raw: `data/raw/rcaeval_zenodo_v2`
  - processed: `data/processed/rcaeval_zenodo_v2`
  - main results: `results/main/rcaeval_re1/`
  - ablation results: `results/ablation/rcaeval_re1/`

研究条件を変更する必要がある場合は、コードへ埋め込まずYAML設定として明示し、正式条件の変更か感度分析かを区別する。

## AMBER実装の保護

- `src/models/amber.py` は提案手法本体として扱い、明示的に依頼されない限りロジックを変更しない。
- データセット対応は、可能な限りbenchmark adapter、前処理、設定、実行スクリプト、評価、テスト側で行う。
- `BenchmarkCase` など既存のbenchmark abstractionを維持する。
- 因果グラフやコールグラフをAMBERのRCA推論入力へ追加しない。
- AR学習、標準化、事前分布推定などに異常期間の情報を混入させない。
- 実験条件の差分は設定ファイルで追跡可能にし、比較対象間で意図しない条件差を作らない。

## アブレーション

- RCAEval正式アブレーション設定は `configs/ablation/rcaeval_re1_zenodo_v2/` に置く。
- 正式mainとの差分は、実験カテゴリ・実験名と、明示されたアブレーション軸だけに限定する。
- 正式variantは次の組合せとする。
  - `no_ar`: `raw` + `bayes_factor`
  - `no_bayes`: `ar` + `glrt`
  - `no_ar_no_bayes`: `raw` + `glrt`
- `configs/ablation/baro/` 配下の3設定はBARO pilot再現用として保持し、削除またはRCAEval用に上書きしない。
- BARO pilotのデータパスは raw `data/raw/baro`、processed `data/processed/baro` とする。

## 検証

- Pythonコードまたは設定を変更したら、原則としてリポジトリルートで次を実行する。

  ```bash
  python3 -m pytest -q
  ```

- shell scriptを変更したら、対象スクリプトを `zsh -n` または適切なshellの構文チェックに通す。
- YAML設定を変更したら、mainとvariantの共通条件、変更軸、結果パスをテストまたは機械的比較で確認する。
- 長時間の正式実験、データダウンロード、375ケース以上の一括実行は、ユーザーが実行まで依頼した場合に行う。
- 長時間実験を実行した場合は、ケース数、summary、失敗ケース、出力先を確認する。プロセスの終了コードだけで成功と判断しない。

## データ・結果・研究記録

- `data/` 配下の大容量データはGit管理へ追加しない。
- 再生成可能な大量の中間結果やデバッグ出力を、明示的な理由なくコミット対象へ追加しない。
- 実験プロトコルや研究判断を変更した場合は、積極的に `notes/decisions/` を更新する。
- 正式実験を実行して結果を確定した場合は、積極的に `notes/experiments/` に条件、コマンド、ケース数、結果、出力先、考察などを記録する。
- 頭の片隅に置いておき、そのうち検証したいアイデアが浮かんだら、積極的に `notes/ideas/` に記録する。
- 論文・README・実験ログに数値を書く前に、summaryや元結果と一致することを確認する。

## 完了報告

作業完了時は、次を簡潔に報告する。

- 変更したファイル
- 変更した研究条件またはロジック
- 実行したテストと結果
- 実行していない長時間処理
- 生成物・実験結果の保存先
- 残っているリスクまたは次の推奨アクション
