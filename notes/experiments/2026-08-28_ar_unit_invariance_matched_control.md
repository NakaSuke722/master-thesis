# AR単位不変化のmatched control

## 目的

`stationary_counterfactual_ar_uncertainty_unit_invariant`の精度向上には、AR入力の正常期間metric-wise標準化だけでなく、過去の実験後に導入された`time.1`と完全無情報メトリクの除外が混在している。そのため、現行コードと現行processed dataを共通化し、`ar_input_scaling`の有無だけを変えたmatched controlを行う。

## 比較条件

| Variant | `ar_input_scaling` | 役割 |
|---|---|---|
| control | `none` | 元のmetric単位のままRidge ARを推定 |
| unit-invariant | `normal_standard` | 正常期間だけでmetric-wise標準化してRidge ARを推定 |

上記以外のZenodo v2、375ケース、最大600点window、Counterfactual AR、定常性制約、horizon-aware uncertainty、Bayes Factor、prior、service-level `mean_top3`は同一とする。

過去結果を上書きしないため、controlは専用実験名 `stationary_counterfactual_ar_uncertainty_unit_invariance_control` で保存する。

## 実行手順

正式375ケースのpaired ablation:

```bash
AMBER_WORKERS=4 ./scripts/run_ablation.sh --unit-invariant-ar-matched
```

normal-only pseudo-faultのpaired calibration:

```bash
PYTHONPATH=src:. venv/bin/python \
  scripts/analyze_ar_unit_invariance_matched_control.py
```

## 出力

- 正式control: `results/ablation/rcaeval_re1/stationary_counterfactual_ar_uncertainty_unit_invariance_control/`
- 正式treatment: `results/ablation/rcaeval_re1/stationary_counterfactual_ar_uncertainty_unit_invariant/`
- pseudo-fault比較: `results/analysis/ar_pseudo_fault_calibration_unit_matched/`
  - `control/case_calibration.csv`
  - `unit_invariant/case_calibration.csv`
  - `case_calibration.csv`
  - `case_deltas.csv`
  - `summary.json`
  - `summary.md`

`case_deltas.csv`は`case_id`と実験メタデータで厳密に1対1 joinする。差分は `unit-invariant - control` と定義し、BF校正統計では負の差分が単位不変化による正常時証拠の減少を表す。

## 判定方針

- 正式精度が向上し、pseudo-fault BFも減少するなら、単位不変化を最終候補に採用する根拠とする。
- 正式精度だけが向上し、pseudo-fault校正が改善しない場合は、単位不変化をランキング上の改善として採用しつつ、BFの絶対値校正は未解決と記録する。
- 次のモデル変更は、本matched controlの結果を確定するまで行わない。
