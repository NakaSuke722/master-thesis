# RUN高速化の同等性・速度確認

## 目的と範囲

RUNの計算を効率化し、推論規則・学習条件の維持を確認する。
正式375ケースの再実行はしない。正常のみを学習に使い、既存case JSONは上書きしない。
変更した方式と採否の理由は`notes/decisions/2026-08-31_run_execution_optimization.md`に記載する。

## 実装・確認対象

- `src/models/baselines/run.py`: 独立層のbank化、teacher encoder計算の再利用、
  不要projectorの解放、相関cacheと二分探索による循環除去、backend・stage時間のdiagnostics。
- `src/main.py`、`configs/baselines/run.yaml`: backend指定の接続。
- `scripts/benchmark_run_optimization.py`: 1ケースの少数targetを既定とした比較スクリプト。
  `--full-case`を明示した場合だけ1ケース全体を比較する。
  参照側は元の循環除去も再現する。`--output`指定時は新規diagnosticのみ作成し上書きしない。
- `tests/test_run_optimization.py`、`tests/test_metric_baseline_configs.py`: 同等性と設定の回帰テスト。
- `README.md`: 設定・使い方・計算量の注意。

## 条件

- macOS、メモリ16 GB、PyTorch 2.12.1、CPU、torch_num_threads=1。
- seed=42、seq_len=32、hidden=128、moving average=25、batch=128、pretrain=1、forecast=1。
- benchmark scriptではoptimizer初回importをタイマーより前に済ませる。
- 最終測定は、他の検証プロセスを終了させてからOB→TTを直列実行した。
  OS上の他アプリ・熱状態まで統制した反復統計ではなく、代表測定である。

## 観測結果

| 対象 | 規模 | 参照版 | 高速版 | 確認 |
|---|---|---:|---:|---|
| OB `re1_ob__adservice_cpu__1` 全体 | 正常600点、45 metrics | 24.461秒 | 15.100秒 | attention完全一致、親集合・最終グラフ・service順位一致 |
| 上記OBの循環除去のみ | 同上 | 5.997秒 | 0.029秒 | 最終グラフ一致 |
| TT `re1_tt__ts-auth-service_cpu__1` の先頭3 targets | 正常480点、802 metrics | 14.898秒 | 12.347秒 | attention最大差1.192e-7、3 targetsの親集合一致 |

OB全体は約38%短縮、TTの部分学習は約17%短縮だった。
TTの全802 targets・グラフ全体は実行しておらず、TTのservice順位一致も未検証。
参考として12.347秒/3 targetsを単純に802倍すると約55分/ケースの学習時間となるが、
target差、case差、熱制限、並列時の競合、グラフ処理を無視した外挿であって実測ではない。
今回の修正だけでTT全125ケースが数分・数十分で終わるとは言えない。

### 出力先

主な評価に用いた最終測定:

- `results/analysis/run_optimization/ob_full_case_isolated.json`
- `results/analysis/run_optimization/tt_three_targets_isolated.json`

初回測定も削除せず保持:

- `results/analysis/run_optimization/ob_full_case.json`
- `results/analysis/run_optimization/tt_three_targets.json`

初回測定の途中には他のテスト・部分計測も実行したため、速度比較には最終測定を用いる。
profileで得た34秒→11秒という探索段階の値も、profileの有無が異なるため正式な速度比較に使わない。

## 再現コマンド

既存JSONを上書きしないよう、再測定時の`--output`は新しい名前を指定するか省略する。

```bash
PYTHONPATH=src:. venv/bin/python scripts/benchmark_run_optimization.py --dataset re1_ob --full-case
PYTHONPATH=src:. venv/bin/python scripts/benchmark_run_optimization.py --dataset re1_tt --targets 3
```

全375ケースを高速版で最初から再実行する場合（今回は未実行）:

```bash
PYTHONPATH=src:. venv/bin/python src/runner.py \
  --config configs/baselines/run.yaml --workers 2 --defer-success-notification \
&& PYTHONPATH=src:. venv/bin/python src/aggregate_results.py \
  --config configs/baselines/run.yaml --require-complete
```

`--resume`なしなので既存RUN case結果は上書きされる。
2 workersは16 GB環境でメモリ競合を抑えるための開始点であり、最速と測定した並列数ではない。
既存結果を残して未完了caseだけ進める場合は`./scripts/run_baselines.sh --run 2`を使うが、
旧版と高速版のcase時間が混ざるため、正式な実行時間比較には使わない。

## テストと次の判断

- 2026-08-31: 全体pytest **202 passed**。
- 初期weight・RNG、forward、gradient、複数Adam更新、teacherのdetach、端数batch、
  0 epoch、完全グラフ・同点相関・定数列・self-loopを含む循環除去を検証。
- 実データの同等性テストはOB1ケースとTT3 targetsに限定。全ケースbit一致は主張しない。
- 本番caseの結果を生成・上書きしていない。計測JSONだけを`results/analysis/`へ保存した。
- OBの代表ケースでは両実装とも最終グラフが空だった。高速化と別に元のRUN adapterの
  親選択・循環除去による空グラフと同点順位の問題を確認する余地がある。
