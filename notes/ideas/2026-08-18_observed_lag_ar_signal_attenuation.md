# Hypothesis: observed-lag AR attenuates persistent fault signals

## Motivation

Ablation experiment showed:

Raw + BF >= observed-lag AR + BF

for many cases.

## Hypothesis

正常時:

X_t = c + sum_p phi_p X_{t-p} + epsilon_t

障害による持続的shift:

X_t^F = X_t^0 + Delta

異常期間のobserved valueをlagとして用いると、

r_t ≈ (1 - sum_p phi_p) Delta

となる。

したがって、

rho = sum_p phi_p

が1に近いmetricほど、障害signalがAR residual上で減衰すると予測される。

## Testable predictions

P1.
rho ≈ 1 の root metric ほど
AR版のroot scoreがRaw版より低下する。

P2.
rho ≈ 1 の root metric ほど
AR版でroot rankが悪化する。

P3.
persistent CPU/memory faultsでこの現象が強く、
transient faultsでは弱い可能性がある。

## Required experiment

- case-level AMBER vs no_ar comparison
- root metric AR coefficient sum
- root score difference
- root rank difference
- signal retention ratio

## Possible next model

Counterfactual / recursive AR forecasting:
異常観測値をlagとして使用せず、
正常モデル自身の予測値を再帰入力する。