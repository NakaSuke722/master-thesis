# AMBER component ablation: AR residualization and Bayesian scoring

Date: 2026-08-18

## 1. Purpose

AMBER の主要構成要素である

1. AR residualization
2. Bayesian model comparison using NIG marginal likelihood

が RCA 精度にどの程度寄与しているかを検証する。

## 2. Compared methods

| Method | Residualization | Scoring |
|---|---|---|
| AMBER | AR residual | Bayes Factor |
| no_ar | Raw series | Bayes Factor |
| no_bayes | AR residual | Gaussian GLRT |
| no_ar_no_bayes | Raw series | Gaussian GLRT |

評価粒度:
- service
- metric

評価指標:
- AC@1, AC@3, AC@5
- Avg@1, Avg@3, Avg@5

Datasets:
- Online Boutique
- Sock Shop
- Train Ticket

## 3. Results

| 方法 | Service AC@1 | Service Avg@5 | Metric AC@1 | Metric Avg@5 |
|---|---:|---:|---:|---:|
| AMBER = AR + BF | 0.637 | 0.864 | 0.257 | 0.483 |
| **w/o AR = Raw + BF** | **0.643** | **0.883** | **0.273** | **0.549** |
| w/o Bayes = AR + GLRT | 0.423 | 0.734 | 0.143 | 0.373 |
| w/o both = Raw + GLRT | 0.400 | 0.742 | 0.120 | 0.411 |


## 4. Findings directly supported by the experiment

### 4.1 Bayesian scoring is effective

Bayes Factor を GLRT に置換すると、ほぼすべての dataset / granularity で性能が低下した。

特に AMBER と no_bayes の Avg@5 を比較すると ...

したがって現時点では、
NIG marginal likelihood による Bayesian model comparison は
AMBER の性能に実質的に寄与していると考えられる。

### 4.2 Current AR residualization is not supported

AMBER と no_ar を比較すると、AR を除去した no_ar が
多くの条件で同等または高い性能を示した。

特に metric-level Avg@5 は3データセットすべてで no_ar が上回った。

したがって、
「現在実装している observed-lag AR residualization が RCA を改善する」
という仮説は本実験からは支持されなかった。

## 5. Interpretation / hypotheses

AR が persistent shift を吸収している可能性がある。

AR(P)

X_t = c + sum_p phi_p X_{t-p} + epsilon_t

において、障害後に Delta の持続的なlevel shiftが起きると、
異常値自身が lag として使われるため、

r_t ≈ (1 - sum_p phi_p) Delta

となる可能性がある。

特に sum_p phi_p ≈ 1 の場合、
障害によるlevel shiftが residual 上で大幅に減衰する。

この説明は現時点では仮説であり、追加実験による検証が必要。

## 6. What cannot yet be concluded

- ARという考え方そのものが不要とはまだ言えない。
- observed-lag AR が悪いのか、AR modeling 自体が不要なのかは未確定。
- metric-level の低精度については ground truth の観測可能性も監査する必要がある。
- Bayes Factor の優位性については prior sensitivity をまだ確認していない。

## 7. Next actions

1. AMBER vs no_ar の case-level paired analysis
2. dataset × fault type 別の差分分析
3. root metric における AR coefficient sum と rank degradation の関係
4. AR residual による anomaly signal attenuation の直接測定
5. metric ground-truth audit
6. recursive/counterfactual AR の検証

## 8. Related artifacts

- configs/amber.yaml
- configs/ablation/no_ar.yaml
- configs/ablation/no_bayes.yaml
- configs/ablation/no_ar_no_bayes.yaml
- results/main/amber/
- results/ablation/