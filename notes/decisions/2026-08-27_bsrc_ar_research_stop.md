# BSRC-AR開発を修士論文の現行スコープで終了する

決定日: 2026-08-27

## 決定

Adaptive BSRC-ARをRCAEval正式375ケースへ進めず、AMBERの最終暫定方式にも採用しない。BSRC-ARに対する追加モデル開発は修士論文の現行スコープでは終了する。

BSRC-ARは、ARを前処理ではなくBayesian hypothesisへ直接組み込み、正常時と障害時の生成regime変化を検出する理論的拡張として、提案過程と負の結果を論文へ記録する。

## 何を解決しようとしたか

Observed-lag AR residualizationには、障害後の異常値を次時点のlagへ使うことでpersistent shiftを予測へ吸収し、root-cause signalを弱める問題がある。

Counterfactual ARは正常モデルを障害後へ再帰予測することでこの問題を避けた。BSRC-ARではさらに、ARをresidual前処理として使うのではなく、

\[
H_0:\text{normal/postで同一のAR生成regime}
\]

対

\[
H_1:\text{一部のAR parameterまたはinnovation varianceが変化}
\]

というBayes Factorへ統合することを目指した。

## 判断根拠

### 数値面

固定Gauss--Hermite quadratureでは、q4/q8の最大variance-ratio nodeへMAPが集中し、q4が暗黙のcapとして働いていた。Adaptive Gauss--Hermite quadratureを導入し、syntheticではq11/q15がほぼ機械精度で一致した。したがって、積分近似の収束問題は解消した。

### モデル校正

normal-only pseudo-faultでは、正常データだけにもかかわらず約91%のserviceが正のlog BF、約86--88%がstrong evidenceとなった。これは現在のBSRC仮説が正常期間内の時間変化を障害regime changeから十分に区別できていないことを示す。

### RCA性能

3 dataset × 5 fault type × 5 casesの層化75ケースでは、macro AC@1=.2667、AC@3=.5200、AC@5=.7200、Avg@5=.5200だった。`re1_tt` のAC@1は.0400だった。

75ケースは正式375ケース値ではないが、formal runへ進むためのgateとしては不十分な結果である。数値積分が収束している以上、さらにquadratureを調整して解決する根拠もない。

## 解釈

この結果は「ARをBayesian hypothesisへ入れる考え方が一般に無価値」という意味ではない。今回の具体化である、単一AR(3)、Gaussian innovation、known boundary、metric-wise sparse regime changeという組合せが、RCAEvalの正常変動とservice rankingに適合しなかったという結論である。

BSRC-ARを改善するには、robust innovation、time-varying parameter、normal-regime mixture、service階層化などが考えられる。しかし、これらを追加すると別の研究テーマへ拡大し、スコアを見ながらモデルを太らせ続ける危険がある。

## 修士論文での位置づけ

- BSRC-ARは最終手法ではなく、理論的拡張と検証結果として記載する。
- 固定quadratureの問題、adaptive積分による解消、normal-only校正失敗、75-case ranking結果を事実として分けて示す。
- negative resultとして、数値的に正しい周辺化だけではRCA性能と校正を保証しないことを考察する。
- AMBERの主方式は、より仮定が少なく解釈しやすいCounterfactual AR + Bayes Factorを中心に整理する。
- Adaptive Directは性能比較上の参考とし、障害shape libraryへの依存を限界として明記する。

## 今後の展望

将来BSRC-ARを再開する場合は、次を独立した研究課題として扱う。

1. 正常window内の非定常性を明示するtime-varyingまたはmixture AR。
2. 外れ値・heavy tailに頑健なStudent-t innovation。
3. metric単位ではなくservice単位でchange probabilityを共有するhierarchical model。
4. known fault boundaryを未知change pointへ拡張するBayesian change-point model。

これらは今回の修士論文では実装せず、future workとして残す。
