# RCAEval RE1 Zenodo v2移行検証と初回service-level実験

作成日: 2026-08-19

作成者: 中川 浩輔

## 1. 目的

AMBERの正式なmain experimentをHugging Face版RCAEvalからZenodo v2原版へ移行し、次を確認する。

1. RE1の375ケースを欠損なく発見できること
2. `inject_time.txt`を基準にnormal/abnormalを正しく分割できること
3. 障害前後をそれぞれ最大600点に制限できること
4. service-level ground truthでAMBERを実行・集計できること
5. Hugging Face版・BARO予備実験と成果物が混在しないこと

## 2. 正式実験条件

### 2.1 データセット

- RCAEval v2
- Zenodo record: 14590730
- DOI: 10.5281/zenodo.14590730
- RE1-OB: 125ケース
- RE1-SS: 125ケース
- RE1-TT: 125ケース
- 合計: 375ケース

### 2.2 前処理

- 入力: 各ケースの`data.csv`
- 障害注入時刻: 各ケースの`inject_time.txt`
- normal: `time < inject_time`
- abnormal: `time >= inject_time`
- normal window: 末尾最大600点
- abnormal window: 先頭最大600点
- latency-50を除外
- latency-90をcanonicalなlatency名へ変更
- 時刻でソートし、重複時刻は最後の値を採用
- 無限値を欠損値へ変換後、forward fillと0埋めを適用

### 2.3 AMBERと評価

- Model: AMBER
- Residualization: AR
- AR order: 3
- Scoring: Bayes Factor
- Service aggregation: mean_top3
- Evaluation granularity: service
- Metrics: AC@1、AC@3、AC@5、Avg@1、Avg@3、Avg@5

設定ファイル:

```text
configs/main/rcaeval_re1_zenodo_v2.yaml
```

## 3. 移行時の実装変更

### 3.1 データ取得

`scripts/download_rcaeval_re1.py`をHugging Faceの`snapshot_download`からZenodo ZIPの取得・MD5検証・展開へ変更した。

### 3.2 Benchmark adapter

`src/benchmarks/rcaeval_re1.py`から`cases.parquet`と`metrics.parquet`への依存を除去した。Zenodo原版の次の階層を直接走査する。

```text
re1_{ob,ss,tt}/RE1-*/<service>_<fault>/<run>/
├── data.csv
└── inject_time.txt
```

case IDはパス区切りを含まない次の形式へ統一した。

```text
<dataset>__<service>_<fault>__<repetition>
```

### 3.3 前処理・設定・結果パス

AMBER本体は変更せず、adapterが返す`BenchmarkCase`と既存のbenchmark共通APIを利用した。raw、processed、resultsを次へ分離した。

```text
data/raw/rcaeval_zenodo_v2/
data/processed/rcaeval_zenodo_v2/
results/main/rcaeval_re1/amber_zenodo_v2/service/
```

`scripts/run_main.sh`のデフォルトconfigをZenodo v2正式設定へ変更した。このため、通常の正式実験は次で実行できる。

```bash
./scripts/run_main.sh
```

データ取得、前処理、テストも含めて最初から再現する場合は次を使用する。

```bash
./scripts/run_rcaeval_re1_zenodo_v2.sh
```

## 4. 検証結果

### 4.1 データとテスト

- 前処理成功: 375/375ケース
- RE1-OB: 125件
- RE1-SS: 125件
- RE1-TT: 125件
- pytest: 18 passed
- service-level結果JSON: 各データセット125件
- Summaryに記録されたケース数: 375

したがって、RCAEval RE1 Zenodo v2へのデータセット移行は完了と判断する。

### 4.2 初回service-level結果

| Dataset | AC@1 | Avg@1 | AC@3 | Avg@3 | AC@5 | Avg@5 |
|---|---:|---:|---:|---:|---:|---:|
| RE1-OB | 0.448 | 0.448 | 0.856 | 0.6827 | 0.944 | 0.7808 |
| RE1-SS | 0.848 | 0.848 | 0.936 | 0.8933 | 0.992 | 0.9312 |
| RE1-TT | 0.544 | 0.544 | 0.784 | 0.6933 | 0.832 | 0.7440 |

実行時間:

- total execution time: 528.2秒
- pure Python execution time: 528.25秒

Summary:

```text
results/main/rcaeval_re1/amber_zenodo_v2/summary_service.json
```

## 5. 移行中に発生した問題と修正

### 5.1 テストファイル名の不一致

一括実行スクリプトが存在しない`tests/test_rcaeval_zenodo_adapter.py`を指定し、前処理後に停止した。特定ファイル名を列挙せず、`python3 -m pytest -q`で全テストを実行するよう修正した。

### 5.2 Slack通知用オプションのparser未登録

`runner.py`の終了処理で`args.defer_success_notification`を参照したが、`argparse`への登録が不足していたため、375ケース完了後に`AttributeError`が発生した。`--defer-success-notification`をparserへ追加して修正した。

この例外は全ケースの結果JSON保存後に発生したため、AMBERの再実行は行わず、既存JSONからSummaryのみを再集計した。

### 5.3 Summary集計のメモリ使用量

旧実装は375件の結果JSONを`all_results`へ保持し、完全な内容をSummaryの`details`へ複製していた。読み込みと同時に指標を集計する方式へ変更し、Summaryには`number_of_cases`と集計値のみを保存するようにした。

これにより、個別ケースのrankingやdiagnosticsは個別JSONに保持しつつ、Summaryの重複とメモリ使用量を削減した。

## 6. 結果の解釈

Sock Shopでは特に高い性能を示し、Online BoutiqueとTrain TicketではAC@1が相対的に低い。一方、いずれのデータセットでもAC@3、AC@5はAC@1より高く、root cause serviceが上位候補には含まれる傾向が確認できる。

ただし、これは移行後の初回結果である。データセット間の性能差について結論を出す前に、fault type別集計、case-level失敗分析、baseline比較、window sensitivityを行う必要がある。

## 7. 残課題

1. Slack成功通知をSummary生成・保存後に一度だけ送る処理を実装する
2. fault type別の性能を集計する
3. Online BoutiqueとTrain Ticketの失敗ケースを分析する
4. baselineを同一375ケース・同一windowで実行する
5. window sensitivityを正式実験とは別設定で実施する
6. READMEの正式実験手順をZenodo v2へ更新する

## 8. 関連成果物

- `configs/main/rcaeval_re1_zenodo_v2.yaml`
- `scripts/download_rcaeval_re1.py`
- `scripts/run_rcaeval_re1_zenodo_v2.sh`
- `scripts/run_main.sh`
- `src/benchmarks/rcaeval_re1.py`
- `src/prepare_rcaeval_re1.py`
- `src/aggregate_results.py`
- `results/main/rcaeval_re1/amber_zenodo_v2/summary_service.json`
- `notes/decisions/2026-08-19_benchmark_strategy.md`
