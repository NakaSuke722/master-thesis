# In-sample学習残差とtime-ordered holdout残差の診断

## 目的

Unit-invariant Stationary Counterfactual AR+BFのnormal-only pseudo-faultで、障害がないにもかかわらず大きなBFが出る。原因候補として、AR推定に使用した同じ正常データから学習残差を求めているため、正常残差の分散を過小評価している可能性を検証する。

## 診断設計

各RCAEvalケースの保存済み正常windowを時系列順に前半と後半へ分ける。前半だけで正常期間metric-wise標準化とAR推定を行い、係数を固定したまま後半をobserved-lagの1-step predictionで予測する。

- in-sample residual: 前半のAR学習残差
- OOS residual: 後半の観測済みlagを使う1-step予測残差
- scale ratio: `OOS residual scale / in-sample residual scale`
- center shift: OOS残差中央値とin-sample残差中央値の差をin-sample scaleで標準化

さらに、後半をCounterfactual ARではなくobserved-lag 1-step residualで評価したBFも計算する。これにより、問題が学習残差の過小評価か、長期counterfactual再帰か、正常期間内の分布変化かを切り分ける。

## 実行コマンド

```bash
PYTHONPATH=src:. venv/bin/python \
  scripts/analyze_ar_residual_generalization.py \
  --workers 4
```

## 結果

375ケース、131,150 metric-case、13,250 service-caseを分析した。6,695 metric-caseは、前半が定数列でAR係数を推定できないなどの理由で診断対象外となった。AMBER内部のnormal scaleとservice scoreに対する再計算最大誤差はどちらも0であった。

| Scope | Median case scale ratio | 95% bootstrap CI | Ratio > 1 | Ratio > 1.1 | Median absolute center shift |
|---|---:|---:|---:|---:|---:|
| overall | 0.981 | [0.973, 0.989] | 35.2% | 4.5% | 0.259 |
| re1_ob | 0.996 | [0.980, 1.014] | 48.0% | 12.0% | 0.205 |
| re1_ss | 0.979 | [0.957, 0.991] | 34.4% | 1.6% | 0.305 |
| re1_tt | 0.975 | [0.967, 0.987] | 23.2% | 0.0% | 0.258 |

全体のOOS scaleはin-sample scaleより大きくなく、むしろ僅かに小さい。したがって、「in-sample残差が狭すぎるためBFが過大になる」という仮説は支持されない。

| Statistic | Counterfactual | Observed-lag 1-step |
|---|---:|---:|
| Median max service BF | 582.76 | 690.73 |
| Positive service fraction | 89.6% | 71.9% |

Counterfactualを使わない1-step予測でも大きなBFが残り、max BFの中央値はむしろ大い。Counterfactual再帰は正のBFをより広いserviceに生じさせるが、極端なBFの唯一の原因ではない。

service-level Spearman相関は次の通りであった。

- Counterfactual BF vs absolute log-scale ratio: 0.637
- Counterfactual BF vs absolute center shift: 0.474
- Counterfactual BF vs observed-lag 1-step BF: 0.788
- 1-step BF vs absolute log-scale ratio: 0.841
- 1-step BF vs absolute center shift: 0.217

scale ratioの符号付き中央値は1付近だが、scaleの増加と減少の両方を表すabsolute log-scale ratioはBFと強く関連した。このことから、正常window内にも局所的な分散変化や中心移動があり、BFがそれらを分布変化として検出していると解釈できる。ただし、相関は因果を直接証明するものではない。

## 判断

OOS residual scaleを新しい正常基準とする改修は行わない。仮説に反してOOS scaleは大きくなく、より小さいscaleを使うとBFをさらに拡大するおそれがある。

Unit-invariant Stationary Counterfactual AR+BFは、障害有無を絶対判定する校正済み検定器ではなく、service間の相対的なroot-cause ranking scoreとして採用する。修論に向けてモデル要素の追加はここで停止し、残る単位不変性の1ケースに対する数値的定数列ガードだけを実装して最終候補を凍結する。
