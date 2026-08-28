# 修論に用いる最終AMBERの決定

> **2026-08-29 追記:** AR次数7/9/11/13の拡張感度分析により、本文書の
> AR(3)最終freezeは再検討中とした。後続の事実と現在の判断は
> `notes/decisions/2026-08-29_ar_order_selection.md`を参照する。

決定日: 2026-08-28

## 決定

修論に用いるAMBERを、次の`Unit-invariant Stationary Counterfactual AR + horizon-aware uncertainty + Bayes Factor`としてfreezeする。

| 要素 | 最終値 |
|---|---|
| AR入力 | 正常windowだけによるmetric-wise標準化 |
| residualization | Counterfactual AR |
| AR次数 | 3 |
| stationarity | companion root projection、半径0.98 |
| counterfactual bounds | なし |
| forecast uncertainty | horizon-aware、対角分散 |
| scoring | NIG Bayes Factor |
| service集約 | `mean_top3` |

`configs/main/rcaeval_re1_zenodo_v2.yaml`をこの条件へ更新する。以後、性能向上だけを目的としたモデル要素の追加は停止し、本文執筆と既存比較の整理へ移る。

更新後の正式mainを全375ケースで再実行し、sensitivity基準条件とのfull ranking不一致0件、評価metric不一致0件を確認した。

## 根拠となる事実

RCAEval RE1 Zenodo v2の375ケースで、最終条件のmacroはAC@1 .7440、AC@3 .9280、AC@5 .9680、Avg@5 .8891だった。Raw+BFに対するTop-1は70件獲得・31件喪失、exact McNemar `p=.0001`である。

単位変換1,125比較では、完全ranking、Top-1、root-service rankがすべて100%一致した。残った浮動小数点定数列問題は、正常windowだけを使う8 ULPガードで解消した。

半径.95/.98/.99、AR(1)/(3)/(5)、horizon補正の有無を一軸比較した。半径.99とAR(5)は一部aggregate scoreを僅かに上げたが、paired Top-1差は有意でなく、下位k、pseudo-fault、実行時間のいずれかを悪化させた。no-horizonは全macro指標とnormal-only pseudo-faultの両方で悪化した。

forecast uncertainty倍率10以上は全metric-caseの0.13%で、root serviceのtop-3 metricに含まれるケースは0だった。CPU faultはRaw比でTop-1獲得13・喪失16、`p=.7111`であり、旧Observed-lag ARのような一方向の悪化は残っていない。

## 採用しない変更

- AR(5): 小幅なformal上昇に対し、計算時間とnormal-only P90最大BFが悪化する。
- stationarity半径.99: AC@1は上がるがAC@3/5とpseudo中央値が悪化し、差は少数caseに限られる。
- uncertainty multiplier cap: 極端値はrootの集約上位metricを駆動していない。
- CPUだけRawへ切替: fault type既知情報への事後適合であり、理論を歪める。
- OOS residual scaleへの置換: time-ordered holdoutでin-sample過小評価仮説が反証された。
- BF絶対閾値の追加: 現結果は校正済み障害検定器を支持しない。

## 主張の範囲

最終AMBERのBayes Factorは、同一case内でroot-cause serviceを相対順位付けするevidence scoreとして用いる。BFの絶対値を障害有無の確率や普遍的な検出閾値とは解釈しない。

未知の障害開始時刻、外部benchmarkへの一般化、BFの絶対校正は将来課題とする。この限定を明記することで、未解決事項を隠さず、修論の主張をRCAEvalで実証した範囲に固定する。

詳細な実験条件と結果は`notes/experiments/2026-08-28_final_counterfactual_ar_brushup.md`に記録した。
