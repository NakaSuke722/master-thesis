# Counterfactual ARの最終候補に関する判断

決定日: 2026-08-23

## 事実

RCAEval RE1 Zenodo v2全375ケースにおいて、bounded counterfactual ARはmacro AC@1 0.6667で既存最高だった。hard clipを使わないStationary Counterfactual ARは0.6640、さらにhorizon-aware uncertaintyを加えた方式も0.6640だった。最終方式はRaw+BF 0.6400に対してTop-1を10件獲得・1件喪失し、exact McNemar `p=0.01172`だった。bounded方式との差は獲得8・喪失9、`p=1.0`である。

最終方式はbounded方式に対しmacro AC@3を0.8800から0.8853、Avg@5を0.8416から0.8496へ上げ、AC@5は0.9440で同じだった。正常のみpseudo-faultでは、horizon補正がStationary CFのservice BF中央値を266.98から133.66へ下げたが、Stationary Observedの34.52より高かった。

full forecast-error covarianceでwhiteningした追試はmacro AC@1 0.6107、AC@3 0.8587、AC@5 0.9227、Avg@5 0.8181となり、Stationary observed-lag ARと375ケースすべてのroot-service rankが一致した。対角補正からのAC@1差は-0.0533、exact McNemar `p=0.001193`である。

shared AR対separate ARの直接Bayes Factorは、AR(1)の5種synthetic scenario各200反復で初期検証した。永続平均・AR係数・innovation variance変化は全反復で `log BF > log(10)`、無変化と単発spikeは全反復で `log BF < 0` だった。ただし、これは明確な効果量のGaussian syntheticを用いた探索的sanity checkである。

Direct AR-BFのnormal-only pseudo-faultをRCAEval全375ケース×3分割×3 priorで実行した。strong priorはすべての分割で最も偽証拠を抑えたが、strong service割合は54.4%-56.5%であり、特に `re1_tt` の50/50分割で75.2%に達した。計算失敗は0で、post posterior meanの非定常率は0.5%-1.6%程度だった。

## 解釈

AR過程をRCAから捨てる根拠はない。むしろ、Observed-lag方式で異常観測を予測へfeedbackすることがroot signalを吸収していたという説明が、同じstationary係数を用いた介入的比較で支持された。hard clipは改善の必要条件ではなく、stationarity constraintで数値安定性を置換できる。

horizon-aware uncertaintyは理論的には必要で、正常時校正と順位深度を改善する。ただしforecast-error correlationを無視しており、正常時の偽BFを完全には解消しない。また半径0.98は未感度分析の設計値である。

ただし、forecast-error correlationの完全除去はCounterfactual errorを1-step innovationへ戻し、persistent shiftを再び減衰させる。対角補正の不完全さは、各horizonの不確実性を調整しつつpersistent RCA signalを保持するという実用上の役割を果たしている。

## 判断

`Stationary Counterfactual AR + horizon-aware uncertainty`をAMBERの**暫定最終候補**とする。人工的なnormal min/max clipを用いるbounded方式は比較対象として保持する。半径感度と補正過大caseの診断が終わるまで、`configs/main/rcaeval_re1_zenodo_v2.yaml`は変更せずAMBERをfreezeしない。

full covarianceは最終候補から外し、理論的等価性とRCA目的との衝突を示すnegative ablationとして保持する。次の研究分岐として、ARを前処理ではなく `H0: pre/postでshared AR` 対 `H1: separate AR` のBayes Factorに組み込む方式を検証する。この方式がRCAEvalで検証されるまで、暂定最終候補の判断は変更しない。

normal-only結果を受け、Direct AR-BFの第3段階RCAEvalはstrong priorを用いた探索的順位評価とする。strong priorは「最も保守的な既検証候補」であって校正済みではない。RCAEvalを走らせることは、Direct AR-BFを暫定最終候補に採用する決定ではない。

この判断はAC@1だけの最大値選択ではなく、次の複合基準による。

- Rawに対する対応のある有意な改善
- bounded方式と統計的に同等なTop-1
- AC@3 / Avg@5の改善
- hard clipを除去した理論的一貫性
- 正常時校正の改善

詳細な条件・数値・成果物は `notes/experiments/2026-08-23_stationary_counterfactual_ar.md` に記録する。
