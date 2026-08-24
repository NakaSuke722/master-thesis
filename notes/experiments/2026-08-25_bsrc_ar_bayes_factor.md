# BSRC-AR Stage A: conjugate sparse regime-change Bayes Factor

実装日: 2026-08-25

## 目的

Adaptive Direct AR-BFはRCAEvalで高い点推定を得たが、障害応答shape、onset候補、half-life、empirical-null補正など、benchmarkを見ながら選べる設計自由度が多い。次の候補では、正常時系列の生成過程が障害後も継続するというH0と、生成parameter自体が疎に別regimeへ変化するというH1だけを比較する。

Stage Aではmetricごとの時系列モデルを検証し、service aggregationは正式比較条件の `mean_top3` に固定する。service-level hierarchical posterior、dynamic factor、Structural VAR-LiNGAMはStage B/Cとして分離し、今回のRCAEvalスコアへ混ぜない。

## 正常時の生成モデル

normalだけからmedian/MADで標準化したmetricを、

\[
X_t=c+\sum_{p=1}^{P}\phi_pX_{t-p}+\epsilon_t,
\qquad
\epsilon_t\sim\mathcal N(0,\sigma^2)
\]

とする。正式設定は \(P=3\) である。係数と分散にはproper Normal-Inverse-Gamma priorを置く。

\[
\boldsymbol\beta\mid\sigma^2
\sim
\mathcal N(\mathbf m_0,\sigma^2\Lambda_0^{-1}),
\qquad
\sigma^2\sim\operatorname{InvGamma}(\alpha_0,\beta_0)
\]

Normal-Inverse-Gammaを積分したpost posterior predictiveはStudent-\(t\)型になる。ただし今回の実装は、各時点のinnovationを固定自由度 \(t_5\) とした非共役モデルではない。この差は結果の解釈で明記する。

## H0: normal AR continuation

normal design/targetを \((X_N,y_N)\)、postを \((X_A,y_A)\) とする。H0はnormal/postで同じ \((\boldsymbol\beta,\sigma^2)\) を共有する。

\[
H_0:\quad
y_t=X_t^\top\boldsymbol\beta+\epsilon_t
\]

評価する量はnormalを条件としたpostのposterior predictiveである。

\[
\log p(y_A\mid y_N,H_0)
=
\log p(y_N,y_A\mid H_0)-\log p(y_N)
\]

この分布は正常ARのcounterfactual trajectoryを一点に固定せず、係数、innovation variance、将来innovationを周辺化する完全Bayesian counterfactualに対応する。

## H1: sparse parameter regime change

既知fault boundary以後で、

\[
y_t
=X_t^\top\boldsymbol\beta
+I(t\ge\tau)X_t^\top\boldsymbol\delta
+\epsilon_t
\]

とする。ここでindicatorはstep状の観測応答を入力するものではなく、intercept/AR coefficientの生成regimeを切り替える。\(\boldsymbol\delta\)の各要素はintercept、lag-1、lag-2、lag-3のparameter changeである。

各change indicatorを、

\[
\gamma_j\sim\operatorname{Bernoulli}(\pi),
\qquad
\delta_j=
\begin{cases}
0 & \gamma_j=0\\
\mathcal N(0,\sigma^2\lambda_j^{-1}) & \gamma_j=1
\end{cases}
\]

とする。正式設定では \(\pi=0.25\) なので、4係数のうち事前に期待するactive change数は1である。intercept changeのprecisionは0.25、lag changeは1.0とする。

これはcontinuous regularized horseshoeそのものではなく、0へのpoint massと有限Gaussian slabを持つspike-and-slab shrinkageである。大多数のparameter changeを0へ縮小し、一部だけを残すというBSRC-ARの中心仮説を、解析的な周辺尤度として実行できるStage A実現である。

AR(3)では \(2^4=16\) 個のchange maskをすべて周辺化する。最良maskだけを選ばないため、parameter subsetの探索にはBayesian multiplicity penaltyが入る。

## Innovation variance regime

post/pre variance ratioを \(r\) として、

\[
\log r\sim\mathcal N(0,0.7^2)
\]

とする。固定したvariance ratio \(r\) の下では、post designとtargetを \(1/\sqrt r\) 倍すれば共通varianceの共役回帰として周辺尤度を計算できる。連続priorの積分は4点Gauss--Hermite quadratureで近似する。

したがって正式H1は、

\[
p(D\mid H_1)
=\sum_{\boldsymbol\gamma}p(\boldsymbol\gamma)
\int p(r)p(D\mid\boldsymbol\gamma,r)\,dr
\]

であり、実装上は16 masks × 4 variance nodes = 64 candidatesをmodel averagingする。

## Bayes Factorとscore

\[
\log BF
=
\log p(y_A\mid y_N,H_1)
-\log p(y_A\mid y_N,H_0)
\]

をmetric scoreとする。shape library、clip、normal-only pseudo-fault quantileの差し引きは行わない。`log BF / post predictive rows` は診断として保存するが、正式順位には生のlog BFを使う。

case artifactには、各metricについて次を保存する。

- H0/H1 posterior predictive log marginal
- MAP changed parameter subset
- MAP variance ratio
- parameterごとのposterior inclusion probability
- pre/post coefficient posterior mean
- pre/post innovation variance
- spectral radiusとlong-run mean

## 今回の実装範囲と未実装部分

実装済み:

- normal-only robust standardization
- known fault boundary
- AR(3) Normal-Inverse-Gamma posterior predictive
- sparse coefficient regime changeのspike-and-slab model averaging
- log-normal variance ratioのGauss--Hermite model averaging
- AMBER diagnostics、RCAEval config、case-level並列実行

未実装であり、今回の方式の成果として主張しないもの:

- PACF parameterizationによるposterior全体のhard stationarity保証
- 固定自由度Student-\(t_5\) innovation likelihood
- continuous regularized horseshoe inference
- unknown \(\tau\) の全時刻周辺化またはBOCPD
- service-level hierarchical change indicator
- dynamic factor / Structural VAR-LiNGAM

まず解析的Stage Aで「正常posterior predictive対疎なregime change」という中心仮説をRCAEvalから独立に検証し、上記拡張を一度に混ぜない。

## 設定と実行

設定:

`configs/ablation/rcaeval_re1_zenodo_v2/bsrc_ar_bayes_factor.yaml`

実行:

```bash
./scripts/run_ablation.sh --bsrc-ar
```

既定は4 workersである。変更する場合は、

```bash
AMBER_WORKERS=8 ./scripts/run_ablation.sh --bsrc-ar
```

とする。出力先は、

`results/ablation/rcaeval_re1/bsrc_ar_bayes_factor/`

である。

## 実装時検証

synthetic AR(1)で次を確認した。

- no change: log BF < 0
- persistent level regime: log BF > 10、intercept inclusion > 0.99
- AR coefficient regime: log BF > 10、lag-1 inclusion > 0.99
- innovation variance regime: log BF > 10、posterior variance ratio > 2

RCAEval smokeをartifact非保存で実行した。`re1_ob__cartservice_loss__1` は49 metrics、13 servicesを0.14秒で処理し、ground-truthの `cartservice` がTop-1だった。大規模な `re1_tt__ts-travel-service_loss__4` は1179 metrics、70 servicesを2.91秒で処理し、ground-truthの `ts-travel-service` がTop-1だった。これらは性能評価ではなく実行経路と計算可能性の確認である。正式375ケースは未実行であり、ユーザー側で実行する。

## 次の判断

正式結果だけでなく、normal-only pseudo-faultで絶対BF calibrationを確認する。偽証拠が多い場合はempirical quantileをスコアから差し引かず、Gaussian innovation、stationarity、AR orderなど生成モデルのmisspecificationとして診断する。

Stage Aが成立した場合、Stage Bでservice-level hierarchical posteriorへ進み、`mean_top3`を廃止する。Stage Cではmetric measurement modelからlatent service stateを推定し、factor-augmented Structural VARのstructural innovation regime changeをroot-cause posteriorとして扱う。
