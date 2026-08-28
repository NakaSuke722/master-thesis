# Counterfactual AR次数選択の再開

決定日: 2026-08-29

## 決定

2026-08-28に行ったAR(3)の最終freezeを解除し、次数選択を再開する。AR(7)をRCAEval RE1上の経験的第一候補とするが、本日時点では正式mainへ昇格させず、`configs/main/rcaeval_re1_zenodo_v2.yaml`の`ar_order: 3`は維持する。

AR次数の探索は1/3/5/7/9/11/13で打ち切る。AR(15)以上の追加探索、dataset別次数切替、fault type別次数切替は行わない。

## 根拠

- AR(7)はmacro Avg@5 `.8939`で全候補中最高だった。
- AR(7)のmacro AC@1 `.7493`、AC@3 `.9333`、AC@5 `.9653`はAR(5)と同じで、Avg@5だけ改善した。
- AR(9)はAC@3、AR(11)はAC@1が最高だが、他指標とのtrade-offがあり、case-level差も小さい。
- AR(13)はAR(7)に全macro指標で劣るため、高次化の追加探索に停止根拠がある。
- AR(7)はOnline Boutiqueを改善する一方、Train Ticketを悪化させる。単一の普遍的な最適次数が実証されたとは言えない。
- AR(7) normal-only pseudo-faultはAR(3)比でP90 max log BFが約12.7%増え、re1_ttでは約33.3%増えた。一方、positive service fractionは低下しており、破綻とまでは判断しない。
- AR(7)を現時点で「統計的に最適」と呼ぶ根拠はない。

## BF絶対値校正の位置付け

現在のscoreはmetricごとのlog BFをservice内の上位3 metricで平均した相対順位付けscoreである。normal-only pseudo-faultでも大きな正のscoreが広く出るため、普遍的な障害検出閾値やroot-cause事後確率としては解釈しない。

一方、現在のRCAEvalは障害と開始時刻が既知で、root serviceを必ず順位付けする設定である。この主目的にBF絶対値校正は必須ではない。校正は、障害証拠がないときのabstention、障害検知、未知の開始時刻の探索、case間のscore比較へ拡張する際の将来課題とする。

## 最終freezeへの条件

次回は次のどちらかを選び、それ以降はAR次数を変更しない。

1. **理論優先:** normal-only blocked multi-horizon forecast validationで正解labelを使わずに次数を選択する。
2. **執筆優先:** AR(7)をRE1の探索的感度分析で選んだ経験的候補と明記して採用する。

どちらの場合も、AR(3)/AR(5)/AR(7)/AR(9)/AR(11)/AR(13)の感度を本文または付録に掲載し、AR(7)の有意な優越は主張しない。

詳細は`notes/experiments/2026-08-29_extended_ar_order_sensitivity.md`に記録した。
