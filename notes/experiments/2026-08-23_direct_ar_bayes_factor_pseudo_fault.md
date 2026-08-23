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

## 小規模smoke（事実）

ソート順の先3ケース（すべて `re1_ob`）で、3分割×3 priorの27条件を実行した。計算失敗metricは0だった。

- positive metric割合: 18.7%-22.7%
- strong metric割合: 17.3%-21.3%
- positive service割合: 48.9%-55.6%
- 各条件のcase最大service log BF中央値: 409.64-798.46
- post posterior meanの非定常metric割合: 0.0%-1.3%

この3ケースだけで校正を判断できないが、障害のないnormal window内でも非常に大きな別AR過程の証拠が出るmetric/serviceがある。priorの強さだけでは解消しておらず、実際のnormal window内の非定常性、非Gaussian性、レベル変化、周期性またはモデル不適合の診断が必要という早期警告である。全375ケースの分布を確認するまで一般化しない。

## 未実施の作業

本ターンでは全375ケースを実行せず、小規模smokeとpytestまでとする。全件実行後に事実・解釈・prior選択の判断を追記する。
