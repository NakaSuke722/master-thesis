# BSRC-AR v2: innovation variance spike-and-slab

実装日: 2026-08-25

## 目的

BSRC-AR Stage AのH1は、coefficient change maskの種類にかかわらずpost/pre innovation-variance ratioの連続slabを常にactiveにしていた。そのため、以下の基本的な変化を分離できなかった。

- coefficientだけが変化する
- innovation varianceだけが変化する
- coefficientとinnovation varianceが両方変化する

v2ではvariance changeにもBernoulli indicatorを導入する。

## モデル

coefficient change indicatorは従来通り、

\[
\gamma_j\sim\operatorname{Bernoulli}(\pi_\beta)
\]

とする。innovation varianceには独立に、

\[
\gamma_\sigma\sim\operatorname{Bernoulli}(\pi_\sigma)
\]

を置く。variance ratioを \(r=\sigma_A^2/\sigma_N^2\) とすると、

\[
r=
\begin{cases}
1 & \gamma_\sigma=0,\\
\exp(\eta),\quad \eta\sim\mathcal N(0,s_\sigma^2)
& \gamma_\sigma=1
\end{cases}
\]

である。これにより、\(r=1\)という「分散不変」を離散的なspikeとして明示的に表現する。

H1は、

\[
H_1:\quad
\sum_j\gamma_j+\gamma_\sigma\ge 1
\]

という条件付きmodelとする。全coefficient indicatorが0で、かつ\(\gamma_\sigma=0\)のall-spike stateはH0そのものなのでH1から除外し、残ったprior weightを正規化する。

## 数値積分

variance slabの連続積分はGauss--Hermite quadratureで近似する。v2は8点を使う。variance spikeが\(r=1\)を明示的に持つため、slab quadratureに\(r=1\)のnodeを重複して持たせない偶数点を選んだ。

正式variantでは、

\[
\pi_\beta=0.25,\qquad
\pi_\sigma=0.25,\qquad
s_\sigma=0.7
\]

とする。予測分布、AR order、正常期間のprior、service aggregationはStage Aと同一である。

## diagnostics

metricごとに次を追加する。

- MAP modelでvariance slabがactiveか
- posterior variance-change probability
- MAP variance ratio
- coefficientごとのposterior inclusion probability

## 設定と実行

設定:

`configs/ablation/rcaeval_re1_zenodo_v2/bsrc_ar_variance_spike_slab.yaml`

実行:

```bash
./scripts/run_ablation.sh --bsrc-ar-v2
```

出力先:

`results/ablation/rcaeval_re1/bsrc_ar_variance_spike_slab/`

375ケースの実行は本実装ターンでは行わない。Stage Aと同じ再生成済みprocessedデータでユーザーが実行する。
