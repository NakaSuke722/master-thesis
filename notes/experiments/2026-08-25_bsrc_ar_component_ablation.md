# BSRC-AR variance/component ablation

実装日: 2026-08-25

## 目的

入力修正後のBSRC-AR v1に対し、variance spike-and-slabと8点積分を入れたv2はmacro AC@1が0.6453から0.5627へ低下した。v1→v2で複数軸を同時に変更したため、低下原因を分解する。

## 2×2 rollback

variance indicatorとquadrature pointsを独立軸とする。

| Variant | Variance model | Quadrature |
|---|---|---:|
| `bsrc_ar_bayes_factor` | slab always active | 4 |
| `bsrc_ar_variance_slab_q8` | slab always active | 8 |
| `bsrc_ar_variance_spike_slab_q4` | spike-and-slab | 4 |
| `bsrc_ar_variance_spike_slab` | spike-and-slab | 8 |

これにより、次を判定できる。

- q4でslab→spike-and-slabを比べた差
- q8でslab→spike-and-slabを比べた差
- slab固定でq4→q8を比べた差
- spike-and-slab固定でq4→q8を比べた差

## Change-type component

| Variant | Coefficient change | Variance change |
|---|---|---|
| `bsrc_ar_coefficient_only` | spike-and-slab | disabled, \(r=1\) |
| `bsrc_ar_variance_only` | disabled | slab active |

`coefficient_only`はintercept/lag regime changeの寄与、`variance_only`はinnovation variance regime changeの寄与を単独で測る。

## 実行

新規4 variantだけを実行する。

```bash
./scripts/run_ablation.sh --bsrc-ar-components
```

取得済みのv1/v2は再実行せず、最後のaggregateで再利用する。出力先は各variantの次のディレクトリである。

```text
results/ablation/rcaeval_re1/<variant>/
```

6 variantの比較summaryは次へ出力する。

```text
results/ablation/rcaeval_re1/summary_service.json
```

## 解釈順序

1. 2×2比較で、spikeと積分点のどちらが低下を生んだか確認する。
2. coefficient-onlyとvariance-onlyで、RCAEvalの順位情報を主に運んでいるchange typeを確認する。
3. dataset/fault type別に効果が異なる場合は、全体macroだけで一方を捨てず、事前に説明可能なhierarchical mixtureの必要性を判断する。
