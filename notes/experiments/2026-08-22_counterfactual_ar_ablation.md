# Counterfactual AR ablation plan

作成日: 2026-08-22

## 目的

observed-lag ARが異常観測値をlagへ取り込み、persistent shiftを予測側へ吸収することがroot-service順位悪化の原因かを介入的に検証する。

## Variant定義

正常期間のAR fitting、AR次数、ridge、正常残差による標準化、NIG prior、Bayes Factor、`mean_top3` service集約は正式mainと同一にする。変更するのは異常期間の予測方法だけである。

- Observed-lag AR: 異常期間の実観測値を次時点のlagへ使用する。
- Counterfactual AR: 正常期間末尾を初期historyとし、その後はAR自身の予測値を次時点のlagへ使用する。異常観測値は予測器へ戻さない。
- Raw: AR residualizationを行わない既存`no_ar`。

Counterfactual ARの長期再帰では、1-step fittingされた非定常ARが発散しうる。障害がなかった場合のbaselineという意味を保ち、異常情報を混入させずに発散を防ぐため、各予測を正常windowの観測min/maxへ制限してから次のlagへ投入する。clip回数と割合はmetric diagnosticsへ保存する。

設定ファイル: `configs/ablation/rcaeval_re1_zenodo_v2/counterfactual_ar.yaml`

## 事前数値監査

正式mainの375ケースに保存された217,251 metricのAR係数で最大600点の再帰予測を監査した。

- 無制約再帰で非有限値となったmetric: 0
- 正常値の100万倍を超えたmetric: 25
- AR companion rootが1以上のmetric: 19,558（9.00%）
- 正常min/max bound適用後の非有限metric: 0
- boundが1回以上作動したmetric: 46,023
- clipされた予測点: 19,959,845 / 93,294,735（21.39%）

boundは単なる例外処理ではなく結果に影響しうるため、最終的な手法名と論文記述では「bounded counterfactual AR」であることを明示する。Observed-lagとの差はlag入力方式と正常support boundの複合差になる点を解釈上の制約とする。

## 実行方法

取得済み3 variantを再実行せず、新variantだけを実行する。

```bash
./scripts/run_ablation.sh --counterfactual-ar
```

既存3 variantを含む正式アブレーション一式を再実行する場合は次を使う。

```bash
./scripts/run_ablation.sh --rcaeval
```

## 主要な比較と判定

1. Counterfactual AR+BF vs observed-lag AR+BF
2. Counterfactual AR+BF vs Raw+BF
3. 全体、dataset別、fault type別のAC@1/3/5とAvg@5
4. Raw/observed-lag/counterfactual間のcase-level Top-1遷移
5. observed-lagで悪化したdiskケースの順位回復数
6. clip率と順位変化の関係

persistent shift吸収が主要因なら、counterfactual ARでは特にdiskでroot signalと順位が回復すると予想する。一方、CPU悪化はsignal attenuationだけでは説明できなかったため、同様の改善を事前には仮定しない。

## 実行状態

375ケースの正式counterfactualアブレーションまで完了した。

## 正式結果（事実）

| Method | AC@1 macro | AC@3 macro | AC@5 macro | Avg@5 macro |
|---|---:|---:|---:|---:|
| Observed-lag AR+BF | 0.6133 | 0.8587 | 0.9227 | 0.8187 |
| Raw+BF | 0.6400 | 0.8480 | 0.9280 | 0.8187 |
| **Bounded counterfactual AR+BF** | **0.6667** | **0.8800** | **0.9440** | **0.8416** |

Dataset別のbounded counterfactual結果（AC@1 / AC@3 / AC@5 / Avg@5）は、`re1_ob`が `.552/.920/.976/.8368`、`re1_ss`が `.832/.968/.992/.9408`、`re1_tt`が `.616/.752/.864/.7472` だった。全375ケースのTop-1正解数はObserved 230、Raw 240、bounded counterfactual 250である。

case-level paired比較では、Rawからbounded counterfactualへのTop-1遷移は獲得16件・喪失6件だった。AC@1差は+0.0267で、exact McNemar検定は `p=0.05248`、paired bootstrap 95%区間は `[+0.0027, +0.0507]` だった。Observedからの遷移は獲得36件・喪失16件で、差+0.0533、`p=0.007787`、95%区間 `[+0.0160, +0.0907]` だった。

clip帰属分析では、RawからTop-1を獲得した16件のうち、counterfactual結果のroot serviceまたは最上位非root competitorの`mean_top3`構成metricでclipが1回以上動いたのは5件だった。全体375件では167件でどちらかにclipが作動した。したがってclipは無視できない構成要素だが、Rawに対するTop-1改善の全てをclipだけでは説明できない。

成果物:

- `results/ablation/rcaeval_re1/counterfactual_ar/`
- `results/analysis/counterfactual_ar/`

## 解釈

Counterfactual化でRawとObservedの両方を上回ったことは、AR過程自体よりも「異常観測をlagへ戻す使い方」が問題だったという仮説を支持する。特にObserved比でdiskのTop-1獲得19件・喪失0件であり、persistent shift吸収仮説と強く整合する。一方、delayとlossではObservedからの喪失が多く、障害種別に最適な時間応答が異なる。

ただし正常window前半50%でfitし、後半50%を障害なしのpseudo-abnormalとして採点すると、bounded counterfactualのservice BF中央値は266.16、正BFとなるservice割合は91.7%で、Observedの31.55、69.9%より大きかった。再帰h-step予測誤差を1-step innovationと同じscaleで扱うため、予測距離による不確実性を障害証拠へ混入している可能性がある。また、hard min/max clipは全予測点の21.39%で作動するため、最終方式としては理論的改善が必要である。

## 次アクション

hard clipをstationarity constraintへ置き換え、ARのh-step forecast-error varianceで異常残差を正規化する。Observed/Counterfactual、stationarity、horizon uncertaintyの寄与を分離するため、`Stationary Observed`、`Stationary Counterfactual`、`Stationary Counterfactual + uncertainty`を順番に比較する。実行記録は `notes/experiments/2026-08-23_stationary_counterfactual_ar.md` に分離する。
