# Stationarity-constrained AR + horizon-aware uncertainty

実施日: 2026-08-23

## 目的

bounded counterfactual ARはRCAEvalで最高性能を得たが、正常期間のmin/max clipが全予測点の21.39%へ作用し、正常のみのpseudo-faultでもBFが膨らんだ。そこで、人工的なhard clipをARの定常性制約へ置き換え、多段先予測ほど大きくなるforecast uncertaintyを残差標準化へ組み込む。

検証したい問いは次の3つである。

1. stationarity constraintだけで長期再帰の発散を防げるか。
2. 同じstationary係数のObservedとCounterfactualを比べると、異常観測のlag feedback除去は有効か。
3. horizon-aware uncertaintyは正常時の偽のBFを抑え、RCA順位を維持または改善するか。

## 実装した方式

正常windowだけで従来どおりridge ARをfitした後、companion rootsのうち絶対値が0.98を超えるものを、偏角を保ったまま半径0.98へ射影する。射影後は正常window平均がARの固定点となるよう切片を調整する。これは完全な制約付き最尤推定ではなく、**normal-only ridge fitへのroot projection**である。

Counterfactual予測ではnormal末尾をseedとし、その後は予測値だけをlagへ戻す。stationary variantではnormal min/max clipを使わない。

ARのMA(infinity)表現におけるimpulse responseを `psi_j` とすると、h-step forecast errorの分散は次である。

```text
Var(e_h) = sigma^2 * sum_{j=0}^{h-1} psi_j^2
```

最終variantでは、異常残差を `sqrt(sum psi_j^2)` で追加除算してから既存のNIG Bayes Factorへ渡す。正常残差は従来どおり1-step innovationである。これは各horizonの周辺分散を揃える補正であり、h-step error間の相関を完全には除かない。

## Variantと固定条件

RCAEval RE1 Zenodo v2全375ケース、normal/abnormal各最大600点、service-level、`mean_top3`、AR(3)、ridge `1e-3`、NIG prior、Bayes Factorを固定した。

| Variant | 異常lag | Stationarity | Hard bound | Horizon uncertainty |
|---|---|---|---|---|
| Stationary Observed | 観測値 | root projection, radius=.98 | なし | なし |
| Stationary Counterfactual | 予測値 | root projection, radius=.98 | なし | なし |
| Stationary Counterfactual + uncertainty | 予測値 | root projection, radius=.98 | なし | あり |

設定:

- `configs/ablation/rcaeval_re1_zenodo_v2/stationary_ar.yaml`
- `configs/ablation/rcaeval_re1_zenodo_v2/stationary_counterfactual_ar.yaml`
- `configs/ablation/rcaeval_re1_zenodo_v2/stationary_counterfactual_ar_uncertainty.yaml`

実行:

```bash
./scripts/run_ablation.sh --ar-redesign
```

診断・paired分析の再生成:

```bash
PYTHONPATH=src:. python3 scripts/analyze_counterfactual_ar.py
PYTHONPATH=src:. python3 scripts/analyze_ar_pseudo_fault_calibration.py
PYTHONPATH=src:. python3 scripts/analyze_ar_pseudo_fault_calibration.py \
  --modes stationary_ar \
          stationary_counterfactual_ar \
          stationary_counterfactual_ar_uncertainty \
  --output-root results/analysis/ar_pseudo_fault_calibration_redesign
PYTHONPATH=src:. python3 scripts/analyze_ar_redesign.py
```

## 正常のみpseudo-fault診断（事実）

各ケースの保存済みnormal windowを時系列順に50%/50%へ分け、前半でfitし、障害のない後半をpseudo-abnormalとして採点した。正常区間にroot causeは存在しないため、AC@kではなくservice BFの大きさと正BF率を校正指標にした。

| Mode | Median max BF | Median service BF | Positive service fraction | Cases with clip |
|---|---:|---:|---:|---:|
| Observed AR | 1299.53 | **31.55** | **69.9%** | 0 |
| Bounded CF | 1539.29 | 266.16 | 91.7% | 361 |
| Stationary Observed | 1299.40 | 34.52 | 71.2% | 0 |
| Stationary CF | 1539.29 | 266.98 | 91.7% | 0 |
| Stationary CF + uncertainty | 1360.49 | **133.66** | **87.6%** | 0 |

uncertainty補正はStationary CFに対し正常時の中央値BFと正BF率を明確に下げた。ただしStationary Observedまでは戻らず、再帰誤差間の相関やモデルずれが残る。

成果物:

- `results/analysis/ar_pseudo_fault_calibration/`
- `results/analysis/ar_pseudo_fault_calibration_redesign/`

## 正式375ケース結果（事実）

| Method | AC@1 macro | AC@3 macro | AC@5 macro | Avg@5 macro |
|---|---:|---:|---:|---:|
| Observed AR+BF | 0.6133 | 0.8587 | 0.9227 | 0.8187 |
| Raw+BF | 0.6400 | 0.8480 | 0.9280 | 0.8187 |
| Bounded CF+BF | **0.6667** | 0.8800 | **0.9440** | 0.8416 |
| Stationary Observed+BF | 0.6107 | 0.8587 | 0.9227 | 0.8181 |
| Stationary CF+BF | 0.6640 | 0.8747 | 0.9387 | 0.8379 |
| **Stationary CF+BF + uncertainty** | 0.6640 | **0.8853** | **0.9440** | **0.8496** |

最終uncertainty variantのdataset別値（AC@1 / AC@3 / AC@5 / Avg@5）は、`re1_ob`が `.536/.896/.976/.8304`、`re1_ss`が `.848/.968/.992/.9408`、`re1_tt`が `.608/.792/.864/.7776` だった。

全variantで375件が完了し、非有限予測による失敗は0件だった。217,251 metric中57,826本（26.62%）でroot projectionが作動し、最大spectral radiusは1.2412から0.9800へ制限された。最終horizon uncertainty倍率は中央値2.47、90%点4.90、最大97.72だった。

## 対応のある比較（事実）

- Observed → Stationary Observed: Top-1獲得0、喪失1、AC@1差-0.0027、McNemar `p=1.0`。stationarity単独の順位影響は小さい。
- Stationary Observed → Stationary CF: 獲得35、喪失15、差+0.0533、`p=0.00660`。同じstationary係数でも異常lag feedbackを除く効果が明確である。
- Stationary CF → +uncertainty: 獲得8、喪失8、AC@1差0、`p=1.0`。AC@1は維持し、macro AC@3とAvg@5は上昇した。
- Raw → 最終variant: 獲得10、喪失1、AC@1差+0.0240、`p=0.01172`。paired bootstrap 95%区間はAC@1 `[+0.0080,+0.0427]`、AC@3 `[+0.0187,+0.0587]`、AC@5 `[+0.0027,+0.0320]`、Avg@5 `[+0.0192,+0.0443]` で、全て0を超えた。
- Bounded CF → 最終variant: 獲得8、喪失9、AC@1差-0.0027、`p=1.0`。Avg@5差は+0.0080だが95%区間 `[-0.0037,+0.0203]` で、優越は確定しない。

fault type別にRawから最終variantへのTop-1変化を見ると、cpu +1、mem +1、disk +2、delay +1、loss +4であり、全fault typeでnet改善した。Bounded CF比ではmem -5、disk -2、delay +3、loss +3（lossは獲得5・喪失2）で、uncertainty補正はresource faultからnetwork faultへ性能を再配分する傾向がある。

成果物:

- `results/ablation/rcaeval_re1/stationary_ar/`
- `results/ablation/rcaeval_re1/stationary_counterfactual_ar/`
- `results/ablation/rcaeval_re1/stationary_counterfactual_ar_uncertainty/`
- `results/analysis/ar_redesign/`

## 解釈

1. Counterfactual ARの改善はhard clipだけの効果ではない。clipなしStationary CFもmacro AC@1 0.6640を保ち、Rawを上回った。
2. persistent shiftを異常lagへ取り込まないことが主要寄与である。同じstationary係数のObservedとCounterfactual間に有意なTop-1改善がある。
3. stationarity constraintは性能をほぼ変えずに全375ケースの再帰を有限に保ち、hard clipの数値安定化役を置換できた。
4. horizon uncertaintyはTop-1を変えず、Top-3とAvg@5、正常時校正を改善した。ただし正常時BFはObservedよりまだ大きく、完全な確率校正ではない。
5. 最終variantはbounded CFと統計的に同等なTop-1を持ち、より高い順位深度と、hard clipを使わない理論的一貫性を持つ。一方、radius 0.98とroot projectionは設計選択であり、最終freeze前に感度を確認する必要がある。

## 次アクション

現時点の最終候補は `Stationary Counterfactual AR + horizon-aware uncertainty` とする。ただし正式mainはまだ変更しない。次に以下を行う。

1. stationarity radius（例: 0.95 / 0.98 / 0.99）の感度を、まず正常pseudo-faultと既存結果に近い小規模subsetで確認する。
2. delay/lossで改善しmem/diskで低下したcaseのAR persistenceとuncertainty倍率を比較し、補正が過大なmetricを特定する。
3. 必要ならh-step errorの周辺分散だけでなく、forecast-error covarianceまたはeffective sample sizeをBFへ組み込む。
4. 感度分析後にAR仕様をfreezeし、正式main configの置換、baseline比較へ進む。
