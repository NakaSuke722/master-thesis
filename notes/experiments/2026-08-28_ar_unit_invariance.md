# Counterfactual ARの単位不変性診断とnormal-only標準化

実施日: 2026-08-28

## 目的

暫定候補 `Stationary Counterfactual AR + diagonal horizon-aware uncertainty + Bayes Factor` が、メトリクスの物理単位を変えても同じ推論を返すか検証する。正のアフィン変換

```text
Y_t = a X_t + b,  a > 0
```

は値の表現だけを変え、障害、時系列形状、root causeを変えない。そのためAR係数、forecast uncertainty、Bayes Factor、service順位は保存されるべきである。

## 修正前診断（事実）

RCAEval RE1 Zenodo v2の375ケースを、元入力、`1000X`、`0.001X`、`0.001X+100`で比較した。

| 変換 | 完全順位一致 | Top-1一致 | root順位一致 | AC@1 | AC@3 | AC@5 |
|---|---:|---:|---:|---:|---:|---:|
| 元入力 | - | - | - | .6640 | .8853 | .9440 |
| `1000X` | 4.8% | 89.9% | 81.1% | .6053 | .8667 | .9333 |
| `0.001X` | 0.0% | 85.3% | 79.2% | .7627 | .8960 | .9627 |
| `0.001X+100` | 0.0% | 59.7% | 57.9% | .6800 | .7387 | .7920 |

全変換でservice score一致とlag係数一致は0ケースだった。`0.001X+100`ではTrain Ticketのroot順位が最大47変化した。したがって差は丸め誤差ではなく、RCA性能へ影響する単位依存である。

## 解釈

生の系列に固定ridge係数 `lambda` を適用すると、`Y=aX`における実効正則化は概ね次になる。

```text
lambda_effective = lambda / a^2
```

縮小時はlag係数が強く0へ縮み、拡大時はOLSに近づく。この差がstationarity projection、impulse response、horizon uncertaintyへ伝播した。また従来の`relative_scale_floor`は`median(abs(normal_y))`へ依存するため、`+100`のような原点変更でも残差標準化が変わった。

`0.001X`でAC@1が上昇したことは、単位変換の採用根拠ではない。単位を隠れハイパーパラメータとしてRidgeの強さを変えた結果であり、性能比較として無効である。

## 実装した修正

既存結果の再現性を残すため、`ar_input_scaling: normal_standard`を新しい明示的な軸として追加した。正常windowのみからmetricごとの平均`m_X`と標準偏差`s_X`を推定し、正常・異常の両方へ同じ変換を適用する。

```text
U_t          = (X_t - m_X) / s_X
U_t_abnormal = (X_t_abnormal - m_X) / s_X
```

Ridge AR、root projection、Counterfactual再帰、horizon uncertainty、残差Bayes Factorをすべてこの無次元空間で計算する。`Y=aX+b`なら、正常側から得た中心と尺度は`am_X+b`と`as_X`になるため、標準化後の系列は理論上同一になる。

正常windowが完全に定数のmetricでは、異常期間を使わずに尺度を定義できない。この場合はscore対象外とする。浮動小数点の標準偏差が偽の微小分散を作らないよう、定数判定はnormalの最小値と最大値の一致で先に行う。

既存診断との意味を保つため、保存するraw系列、予測、残差は元のmetric単位へ戻し、`ar_input_center`、`ar_input_scale`、`normal_scale_original_units`を追加した。AR係数と内部`normal_scale`は無次元モデル空間の値である。

## 新variant

- 設定: `configs/ablation/rcaeval_re1_zenodo_v2/stationary_counterfactual_ar_uncertainty_unit_invariant.yaml`
- 結果予定: `results/ablation/rcaeval_re1/stationary_counterfactual_ar_uncertainty_unit_invariant/service/`
- 単位不変性診断予定: `results/analysis/ar_unit_invariance_normal_standard/`

実行:

```bash
./scripts/run_ablation.sh --unit-invariant-ar

PYTHONPATH=src:. venv/bin/python \
  scripts/analyze_ar_unit_invariance.py \
  --workers 4
```

## 実装時検証

synthetic dataでは`1000X`、`0.001X`、`0.001X+100`に対してservice順位、score、lag係数、forecast uncertaintyの一致を確認した。Zenodo v2の各datasetから1ケースずつを使ったsmoke testでも、3変換すべてで完全service順位、Top-1、root順位、lag係数が一致した。アフィンoffset後のfloat64情報損失を考慮し、score一致許容値は`rtol=1e-5, atol=1e-7`とした。

## 次アクション

1. 全375ケースで単位不変性診断を再実行し、順位一致100%と係数一致100%を確認する。
2. 新variantを全375ケースで実行し、修正前候補とcase-levelで比較する。
3. 標準化はRidgeの意味を全metricで統一するため、従来スコアがそのまま保存されるとは限らない。単位不変性を満たしたうえでAC@k、Avg@k、fault type別変化を評価する。
4. 正式mainへの採用は、上記2つの375ケース検証後に判断する。
