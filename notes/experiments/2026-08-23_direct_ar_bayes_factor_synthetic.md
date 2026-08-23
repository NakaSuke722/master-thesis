# Direct AR Bayes Factor synthetic validation

実施日: 2026-08-23

## 目的

従来AMBERの `AR -> residual -> distribution-shift Bayes Factor` では、ARは残差を作る前処理である。full forecast-error covarianceでは完全whiteningがObserved-lag residualと等価になり、persistent shiftが再び減衰した。

そこで、AR過程そのものをBayesian hypothesisにする次の方式を導入する。

```text
H0: pre/postで(c, phi, sigma^2)を共有する
H1: pre/postで独立の(c, phi, sigma^2)を持つ
```

本実験の目的はRCAEvalでの精度評価ではなく、直接AR Bayes Factorが理論どおりに永続平均変化、AR係数変化、innovation variance変化を検出し、無変化と単発spikeを構造変化と区別できるか確認することである。

## 方法

ARをBayesian linear regressionとして表す。

```text
y = X beta + epsilon
beta = (c, phi_1, ..., phi_p)
epsilon ~ Normal(0, sigma^2 I)
```

proper conjugate priorは次とした。

```text
beta | sigma^2 ~ Normal(m_0, sigma^2 Lambda_0^-1)
sigma^2 ~ InvGamma(alpha_0, beta_0)
```

pre/postの周辺尤度を `m(D_pre)`、`m(D_post)`、shared modelの周辺尤度を `m(D_pre, D_post)` とすると、

```text
log BF_10
  = log m(D_pre) + log m(D_post) - log m(D_pre, D_post)
```

である。すべてNormal-Inverse-Gammaの共役更新で解析的に計算する。スケール不変性のため、preの中央値とrobust scaleだけを使ってpre/postの両方を標準化した。postの最初のlagにはpre末尾の実測値を使う。

既定priorは、intercept precision `0.01`、lag precision `1.0`、`alpha=2.0`、`beta=1.0` である。これは最終priorではなく、synthetic validation用の初期設定である。

## Synthetic条件

- AR(1)
- pre/post各300点
- burn-in 200点
- 各scenario 200反復
- seed `20260823`
- 基準過程: mean `0.0`、phi `0.65`、innovation SD `1.0`

検証scenario:

1. 変化なし
2. persistent mean shift: post mean `2.0`
3. AR係数変化: phi `0.65 -> -0.20`
4. innovation variance変化: SD `1.0 -> 2.0`
5. 単発spike: post先頭のinnovationに `+6.0`、以後は同じAR過程で減衰

変化scenarioは `log BF > log(10)` の割合80%以上、無変化scenarioは `log BF > 0` の割合10%以下を実装上のPASS条件とした。これは探索的sanity checkであり、事前登録した検証的実験ではない。

## 結果（事実）

| Scenario | Median log BF | 5%-95% | BF>0 | BF>log(10) | 判定 |
|---|---:|---:|---:|---:|---:|
| no change | -8.2485 | [-9.3889, -6.0529] | 0.0% | 0.0% | PASS |
| persistent mean shift | 15.1197 | [7.5127, 24.9465] | 100.0% | 100.0% | PASS |
| AR coefficient change | 55.0641 | [37.1900, 76.7170] | 100.0% | 100.0% | PASS |
| innovation variance change | 57.4011 | [39.3381, 76.0101] | 100.0% | 100.0% | PASS |
| single spike | -7.9076 | [-9.2303, -4.6901] | 0.0% | 0.0% | PASS |

全5scenarioが上記の実装上の判定条件を満たした。

## 解釈

direct AR Bayes Factorは、少なくとも明確なAR(1) synthetic条件において、永続平均、自己回帰係数、innovation varianceの変化をすべて検出できた。特にpersistent mean shiftを正の証拠として検出したことは、Observed-lag residualがshiftを吸収した問題を、segment全体のparameter changeとして扱える可能性を示す。

また、単発spikeはAR状態に一時的に注入された大きなinnovationだが、post全体の別AR過程を支持しなかった。これは「大きな一点」と「持続的な構造変化」を区別する方向の挙動である。

## 限界

- 明確な効果量を持つAR(1) Gaussian simulationであり、RCAEvalの離散的・zero-inflated・non-Gaussian metricを再現していない。
- prior感度は未検証である。
- AR次数は1だけで、正式候補のAR(3)は未検証である。
- stationarityはsimulation側で満たすが、Bayesian regressionのposteriorに定常性制約は課していない。
- 強い時系列変化を検出することと、root serviceを最上位にすることは同じではない。

## 次アクション

1. normal-only pseudo-faultで40/60、50/50、60/40の分割を試す。
2. prior precisionとInvGamma priorの感度を確認する。
3. AR次数1/3/5の感度を確認する。
4. 校正に大きな問題がなければ、RCAEval 375ケース用のAMBER scoring variantとして統合する。
5. RCAEvalで有望な場合だけ、level-only / dynamics-only / variance-only / full changeの仮説分解へ進む。

## 実行コマンドと成果物

```bash
PYTHONPATH=src:. venv/bin/python scripts/validate_ar_bayes_factor_synthetic.py
```

- コア実装: `src/models/ar_bayes_factor.py`
- validation: `scripts/validate_ar_bayes_factor_synthetic.py`
- 結果: `results/analysis/ar_bayes_factor_synthetic/`
