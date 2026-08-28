# BARO比較実験のプロトコル

日付: 2026-08-29

## 目的

AMBERのRCAEval RE1 Zenodo v2における精度を、既存手法BAROと同一のデータ分割・評価粒度で比較できるようにする。

## 事実

- BARO論文の完全な処理は、Multivariate Bayesian Online Change Point Detectionによる異常開始時刻の検出と、RobustScorerによるroot-cause rankingからなる。
- 一方、RCAEvalのBARO adapterは既知の障害開始時刻で正常・異常区間を分割し、RobustScorerのみを実行する。
- 公式BARO/RCAEvalのDataFrame実装は、正常区間の中央値とIQRによる `RobustScaler` を各metricにfitし、異常区間の標準化値の最大値をmetric scoreとする。
- service rankingはmetric rankingを上位から読み、metric名の最初の `_` より前をservice名として重複を除いた順序である。
- 論文の数式は絶対偏差を記載しているが、公式DataFrame実装は符号付きの上側最大偏差を使う。

## 解釈と決定

AMBERは現在、障害開始時刻を既知としてnormal/abnormal windowを受け取る。そのため、BAROにだけ障害開始時刻の推定を課すと、RCA ranking以外の難しさが混入する。正式比較では、公式RCAEval adapterと同じ既知開始時刻のRobustScorerを採用する。

ただし、この結果を「変化点検出を含むend-to-end BARO」の結果とは表記しない。実験名は `baro_robust_scorer_known_onset`、モデル識別子は `baro_robust_scorer` とする。また、データセット用の `data/raw/baro` は使わず、RCAEvalの正式パスを使う。

正式設定では公式実装と同じ `max_signed` を使う。論文の絶対偏差に合わせた `max_absolute` はコード上で選択可能にするが、それを使う場合は別の感度分析として明記する。

## 固定する比較条件

- データ: RCAEval RE1 Zenodo record 14590730, version v2
- 対象: `re1_ob`, `re1_ss`, `re1_tt` の各125ケース、合計375ケース
- window: 正常・異常とも利用可能な最大600点
- 評価: service-level, AC@1/3/5とAvg@1/3/5
- BAROのservice化: metric score順でserviceを重複除去する手法固有の処理
- 出力先: `results/baselines/rcaeval_re1/baro_robust_scorer_known_onset/`

## 実行コマンド

```bash
bash scripts/run_baselines.sh service configs/baselines/baro.yaml 4
```

## 次アクション

1. 375ケースを実行し、summaryと失敗ケース数を確認する。
2. AMBERとBARO RobustScorerのケース単位の順位差を、dataset別・fault type別に集計する。
3. 修論では「BARO (RobustScorer, known onset)」と表記し、end-to-end BAROとの区別を注記する。
