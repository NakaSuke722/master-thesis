# Direct AR Bayes Factor normal-only pseudo-fault protocol

実装日: 2026-08-23

## 目的

shared AR対separate ARの直接Bayes Factorは、明確なAR(1) synthetic変化で基本動作を確認した。次に、RCAEvalの実際のnormal windowを時系列順にfit/pseudo-abnormalへ分け、障害がないのに別AR過程を支持する偽の証拠がどの程度生じるかを校正診断する。

この段階ではroot causeが存在しないため、AC@kは計算しない。metric/serviceの `log BF > 0` 割合、`log BF > log(10)` 割合、各caseの最大service BF、posterior meanの非定常率を校正指標とする。

## 固定条件

- 対象: RCAEval RE1 Zenodo v2の保存済みnormal window
- 分割: 40/60、50/50、60/40（先頭がfit）
- AR次数: 3
- service aggregation: `mean_top3`
- strong evidence: `log BF > log(10)`
- 標準化: fit-normalの中央値とrobust scaleのみ
- pseudo-abnormal先頭のlag: fit-normal末尾の実測値

## Prior sensitivity

| Profile | Intercept precision | Lag precision | alpha | beta | Prior mean of variance |
|---|---:|---:|---:|---:|---:|
| weak | 0.001 | 0.1 | 1.5 | 0.5 | 1.0 |
| reference | 0.01 | 1.0 | 2.0 | 1.0 | 1.0 |
| strong | 0.1 | 10.0 | 5.0 | 4.0 | 1.0 |

coefficient priorとInvGamma priorの集中度を同時に変えるが、`beta/(alpha-1)=1` は固定し、innovation varianceのprior meanは揃える。この3 profileは感度分析用であり、最終priorの決定ではない。

## 出力

case×split×priorごとに次を保存する。

- metric/service BFの中央値、90%点、最大値
- positive/strong evidence割合
- posterior meanのspectral radiusが1以上のmetric割合
- 最上service
- 計算失敗metric数

さらに全体とdataset別で集計する。

## 実行方法

全375ケース:

```bash
PYTHONPATH=src:. venv/bin/python \
  scripts/analyze_ar_bayes_factor_pseudo_fault.py \
  --workers 4
```

小規模smoke:

```bash
PYTHONPATH=src:. venv/bin/python \
  scripts/analyze_ar_bayes_factor_pseudo_fault.py \
  --limit 3 \
  --output-root results/debug/ar_bayes_factor_pseudo_fault_smoke
```

## 成果物

- 設定: `configs/sensitivity/direct_ar_bayes_factor_pseudo_fault.yaml`
- スクリプト: `scripts/analyze_ar_bayes_factor_pseudo_fault.py`
- 正式出力先: `results/analysis/ar_bayes_factor_pseudo_fault/`

## 全375ケースの結果（事実）

2026-08-23に375ケース×3分割×3 prior（3,375 case-condition）を実行し、計算失敗metricは全条件で0だった。overallは以下の通りである。

| Fit fraction | Prior | case最大service log BF中央値 | service log BF中央値 | Positive services | Strong services | Positive metrics | Strong metrics | Post unstable |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 0.4 | weak | 2377.4821 | 22.2911 | 61.8% | 60.3% | 17.5% | 16.6% | 1.1% |
| 0.4 | reference | 2331.6240 | 20.4176 | 62.4% | 60.5% | 17.9% | 16.7% | 1.0% |
| 0.4 | strong | **2251.5580** | **11.8276** | **58.3%** | **56.5%** | **15.6%** | **14.3%** | **0.8%** |
| 0.5 | weak | 2615.9417 | 23.4057 | 61.6% | 60.0% | 17.4% | 16.4% | 1.0% |
| 0.5 | reference | 2578.2951 | 20.0727 | 61.9% | 60.0% | 17.5% | 16.3% | 0.8% |
| 0.5 | strong | **2495.3571** | **10.2876** | **57.4%** | **55.3%** | **15.2%** | **14.0%** | **0.6%** |
| 0.6 | weak | 3014.4784 | 21.2929 | 61.0% | 59.4% | 17.3% | 16.2% | 1.4% |
| 0.6 | reference | 2962.9875 | 17.0853 | 61.1% | 59.2% | 17.1% | 16.0% | 1.2% |
| 0.6 | strong | **2840.2189** | **9.6949** | **56.3%** | **54.4%** | **14.6%** | **13.4%** | **0.9%** |

dataset差は大きい。50/50分割のstrong priorでも、strong service割合は `re1_ob` 38.3%、`re1_ss` 52.5%、`re1_tt` 75.2%であった。特にTrain Ticketでは、障害のないnormal windowの前後を「異なるAR過程」と強く判定しやすい。

## 解釈

- strong priorは、3分割すべてでcase最大BF、service/metricのpositive・strong割合、非定常率を概ね最も抑えた。そのため第3段階の保守的候補にはstrong priorを使う。
- ただし、strong priorでもnormal-onlyのstrong service割合が54.4%-56.5%ある。これを「校正済みprior」とは呼べない。
- 問題は数値計算失敗やposterior meanの非定常化が主因ではない。normal window内の局所的なレジーム差、周期性、trend、外れ値、AR(3)+Gaussian innovationのモデル不適合などをseparate modelが拾っている可能性が高い。
- 第3段階で高いAC@kが得られても、それだけでBFの絶対校正が妥当とは言えない。service間の相対順位に利用できるかを探索的に検証する段階である。

## 次アクション

1. strong priorを固定し、RCAEval RE1 Zenodo v2全375ケースでDirect AR-BFを実行する。
2. Raw+BFおよび暫定最終候補 `Stationary Counterfactual AR + horizon-aware uncertainty`とcase-levelで対応付け、AC@k、Avg@5、Top-1得失、dataset/fault別差を評価する。
3. normal-only偽証拠が特に大いTrain Ticketで順位改善が生じた場合は、非定常性をroot-cause signalと取り違えていないかを優先診断する。
