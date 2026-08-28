# 最終Counterfactual ARブラッシュアップ

実施日: 2026-08-28

## 目的

修論執筆前に、Unit-invariant Stationary Counterfactual AR+BFに残っていた次の不確実性を、一軸感度分析と保存済みartifactのcase-level診断で解消する。

1. stationarity projectionの半径0.98が恣意的でないか。
2. AR次数3が結果を不安定にしていないか。
3. horizon-aware uncertaintyが実際に必要か。
4. 予測不確実性倍率の極端値がroot順位を人工的に押し上げていないか。
5. 旧Observed-lag ARで悪化したCPU faultが最終候補でも系統的に悪化するか。
6. 単位不変性に残った1ケースの浮動小数点誤差を除去できるか。

## 共通条件

- RCAEval RE1 Zenodo v2、375ケース
- dataset: `re1_ob`, `re1_ss`, `re1_tt`
- 正常・異常とも最大600点
- service-level、`mean_top3`
- Counterfactual AR + Bayes Factor
- 正常windowだけでmetric-wise標準化
- root projection、clipなし
- diagonal horizon-aware uncertainty（no-horizon条件だけ無効）

基準条件はAR(3)、stationarity半径0.98、horizon補正ありである。比較では半径、次数、horizon補正のうち一軸だけを変更した。

## 実装上の修正

### ULP定数列ガード

正常windowの値域が8 ULP以下なら、物理的変動ではなくfloat64表現ノイズと判定してAR推定対象外にした。375ケース×3アフィン変換の再診断では、完全順位・Top-1・root順位が全1,125比較で一致した。

### 再現可能な感度分析

`configs/sensitivity/rcaeval_re1_zenodo_v2/`に6条件を追加し、`./scripts/run_ablation.sh --final-ar-sensitivity`で一括実行できるようにした。normal-only pseudo-fault診断には、YAMLの半径・次数・horizon設定を上書きせずそのまま使う`configured`モードと4 worker並列を追加した。

### case-level監査

保存済みfull rankingを`case_id`で厳密にjoinし、Raw+BFおよびno-horizon条件から基準条件へのroot順位変化を全体・dataset・fault type別に集計した。全metricのforecast uncertainty multiplierもCSVへ出力した。

## 事実1: formal 375ケース感度分析

| 条件 | macro AC@1 | AC@3 | AC@5 | Avg@5 | 実時間(s) | stationarity制約率 |
|---|---:|---:|---:|---:|---:|---:|
| **半径.98 / AR(3) / horizon** | .7440 | .9280 | **.9680** | .8891 | 94 | 25.2% |
| 半径.95 / AR(3) / horizon | .7387 | .9147 | .9627 | .8805 | 104 | 51.2% |
| 半径.99 / AR(3) / horizon | .7493 | .9227 | .9653 | .8901 | 117 | 14.7% |
| 半径.98 / AR(1) / horizon | .7227 | .9280 | **.9680** | .8880 | 108 | 25.3% |
| 半径.98 / AR(5) / horizon | .7493 | **.9333** | .9653 | **.8928** | 126 | 26.6% |
| 半径.98 / AR(3) / no horizon | .7227 | .9093 | .9627 | .8784 | 104 | 25.2% |

基準に対するpaired Top-1の獲得/喪失は、半径.99が1/3、AR(5)が4/6、no-horizonが12/4だった。表では「基準の獲得/喪失」なので、半径.99とAR(5)はそれぞれ基準よりTop-1を2件多く取ったにすぎない。exact McNemarは半径.99 `p=.6250`、AR(5) `p=.7539`、horizonあり対なし `p=.0768`である。

## 事実2: normal-only pseudo-fault

正常window前半50%で学習し、後半50%を障害なしの疑似異常期間として評価した。BF絶対値は校正済み閾値ではないため、同一ケースにおける相対比較として用いる。

| 条件 | Median max service BF | P90 max service BF | Median service BF | Positive service fraction |
|---|---:|---:|---:|---:|
| **半径.98 / AR(3) / horizon** | 582.76 | 1083.88 | 156.08 | 89.65% |
| 半径.95 / AR(3) / horizon | 602.46 | 1126.56 | 190.24 | 90.38% |
| 半径.99 / AR(3) / horizon | 601.11 | 1060.75 | 141.85 | 89.64% |
| 半径.98 / AR(1) / horizon | 578.28 | 1011.32 | 158.61 | 89.66% |
| 半径.98 / AR(5) / horizon | 585.66 | 1149.75 | 155.81 | 89.92% |
| 半径.98 / AR(3) / no horizon | 715.54 | 1344.35 | 292.74 | 95.18% |

半径.95はformalとpseudo-faultの両方で悪化した。半径.99は一部統計を改善するが、formal AC@3/5とpseudo median max BFを悪化させた。AR(1)はpseudoの最大BFを僅かに抑える一方、formal AC@1を大きく落とした。AR(5)のformal改善は小さく、実時間が基準比約34%増え、pseudo P90最大BFも悪化した。

## 事実3: horizon補正とfault type

no-horizonから基準条件へ変えると、root順位は33ケースで改善、314件で同一、28件で悪化した。Top-1は12件獲得、4件喪失した。Top-1だけのexact testは5%水準に届かないが、macro AC@1/3/5/Avg@5はすべて改善し、normal-onlyのmedian/P90最大BFも大きく低下した。

Raw+BFから最終基準へは、root順位が107件改善、230件同一、38件悪化した。Top-1は70件獲得、31件喪失し、exact McNemar `p=.0001`だった。

CPU faultだけでは、Rawから13件改善、46件同一、16件悪化、Top-1は13件獲得・16件喪失、`p=.7111`だった。平均rank差は-.1333で僅かにRaw寄りだが、系統的な悪化を示す統計的証拠はない。delayとlossでは最終候補の改善が大きい。fault typeを既知としてCPUだけRawへ切り替える方式は、データセットへの事後適合と説明の複雑化を招くため採用しない。

## 事実4: uncertainty multiplier監査

137,845 metric-caseを監査した。

- 最大倍率: 92.068
- P99: 6.764
- 10以上: 186 metric-case（0.13%）
- 50以上: 12 metric-case（0.01%）
- root serviceの上位3 metricに10以上が含まれるケース: 0
- root serviceの上位3 metricに50以上が含まれるケース: 0

極端な倍率は稀で、root service集約に使われる上位metricには現れなかった。したがって、上限clipを追加する根拠はなく、人工的なcapは導入しない。

## 事実5: 正式mainへの昇格

`configs/main/rcaeval_re1_zenodo_v2.yaml`を基準条件へ更新し、375ケースを再実行した。結果は次の通りで、sensitivity基準条件とcase-levelの全375 rankingおよび全評価metricが完全一致した。

| Dataset | AC@1 | AC@3 | AC@5 | Avg@5 |
|---|---:|---:|---:|---:|
| re1_ob | .752 | .904 | .968 | .8800 |
| re1_ss | .840 | .992 | 1.000 | .9520 |
| re1_tt | .640 | .888 | .936 | .8352 |

正式mainの実時間は111秒、case内Python時間の合計は425.84秒だった。結果は`results/main/rcaeval_re1/amber_zenodo_v2/`へ保存した。

## 解釈

半径0.98は、0.95の過剰なprojectionと0.99の弱い制約の間で、formal順位・正常時挙動・理論上の安定性を最も均衡させる。AR(5)は得点の小幅上昇を示すが、paired差は不確かで、計算量とpseudo P90が悪化する。AR(3)を維持することはスコアの無視ではなく、複数指標・正常時挙動・簡潔性を含む選択である。

horizon-aware uncertaintyはTop-1 exact test単独では決定的でないが、全rank指標、正常時最大BF、forecast horizonに応じて誤差分散が増えるという理論の三者が同じ方向を向くため保持する。

一方、normal-only pseudo-faultでもBFが広く正になる問題は残る。time-ordered holdout診断ではOOS/in-sample residual scale中央値が.981であり、in-sample分散過小評価仮説は支持されなかった。これは最終AMBERを「障害有無を絶対判定する校正済み検定器」ではなく、「同一case内でserviceを相対順位付けするRCA score」と位置付ける理由である。

## 結論と次アクション

最終AMBERは次でfreezeする。

```text
normal-only metric standardization
  -> stationary AR(3), radius 0.98
  -> anomaly-free recursive counterfactual forecast
  -> horizon-wise diagonal forecast uncertainty standardization
  -> normal/post residual distribution Bayes Factor
  -> service mean_top3 ranking
```

clip、fault-type別switch、AR(5)、半径.99、BF threshold校正を最終モデルへ追加しない。修論では本方式を主提案、Raw+BF・Observed-lag AR+BF・旧Counterfactual AR・no-horizonを主要比較として記述する。絶対BF校正、未知障害開始時刻、外部データセットでの再現性は将来課題とする。

## 成果物

- formal sensitivity: `results/sensitivity/rcaeval_re1/`
- pseudo-fault sensitivity: `results/analysis/ar_final_sensitivity_pseudo_fault/`
- paired sensitivity: `results/analysis/final_ar_sensitivity/`
- rank/multiplier audit: `results/analysis/final_ar_diagnostics/`
- affine invariance: `results/analysis/ar_unit_invariance_ulp_guard/`
