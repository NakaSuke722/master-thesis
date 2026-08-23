# Adaptive intervention-response Direct AR Bayes Factor

実装日: 2026-08-23

## 目的

Full Direct AR-BFの失敗とIntercept-shift AR-BFの改善を受け、一要因ずつの小幅変更ではなく、Direct AR-BFの可能性を広く試す統合candidateを実装する。このcandidateが改善した場合は、応答形状、onset周辺化、normal-only校正を後続ablationでロールバックし、寄与要因を分解する。

Full Direct、Intercept-shift、既存Counterfactual ARのロジックは変更せず、`adaptive_direct_ar_bayes_factor` として追加する。

## 構造化仮説

baseline AR過程を

\[
X_t=c+\sum_{p=1}^{P}\phi_pX_{t-p}+\epsilon_t,
\qquad \epsilon_t\sim\mathcal N(0,\sigma^2)
\]

とする。帰無仮説はこの過程をpre/postで共有する。対立仮説は、

\[
X_t=c+\sum_{p=1}^{P}\phi_pX_{t-p}
+\mathbf g_m(t;\tau,\eta)^\top\boldsymbol\delta_m
+\epsilon_t
\]

とする。AR係数とinnovation varianceは共有し、障害応答の時間形状を追加回帰列として直接likelihoodに組み込む。

## 応答形状

正式candidateでは次を同時に検討する。

- `step`: 永続的な突然変化
- `ramp`: 線形に進行する変化
- `exp_rise`: 漸近的に飽和する変化
- `exp_decay`: 発生後に減衰する一過性変化
- `step_ramp`: level変化と傾向変化の同時表現

形状を最大尤度で一つ選ぶのではなく、各candidateのanalytic marginal likelihoodを使い、

\[
\log p(D\mid H_1)
=
\log\sum_{m,\tau}p(m,\tau)p(D\mid m,\tau)
\]

とBayesian model averagingする。`step_ramp`の追加parameterは、marginal likelihoodのOccam penaltyを受ける。

## Onset周辺化

現在のRCAEvalではfault boundaryは既知であるが、応答開始の遅延を `0, 5, 15`点として周辺化する。遅いonsetによるdownstream変化の過剰適合を抑えるため、

\[
p(\tau)\propto\exp(-0.15\,\tau)
\]

のearly-onset priorを使う。将来fault timeを未知とする場合は、onset candidateを全windowに広げて同じ周辺化を使える。この実装は将来拡張のためonset候補をAPIとYAMLで与える。

## Normal-only empirical-null calibration

metric固有の非定常性やARモデル不適合が大きなBFを作る問題に対し、normal windowを40/60、50/50、60/40でpseudo pre/post分割し、同じadaptive BFを計算する。

本番とpseudo-faultで観測数が異なるため、生log BFをそのまま引かず、有効conditional-AR行数 \(n\) で割る。

\[
S_{obs}=\frac{\log BF_{obs}}{n_{obs}}
\]

\[
S_{null,q}
=Q_{0.9}\left(
\frac{\log BF_{null,1}}{n_1},
\frac{\log BF_{null,2}}{n_2},
\frac{\log BF_{null,3}}{n_3}
\right)
\]

\[
S_{final}=S_{obs}-S_{null,q}
\]

とし、「そのmetricが正常期間内でも示す構造変化証拠」を超えた分だけをRCA scoreにする。校正は障害期間の情報を使わない。

## 実験条件

- RCAEval RE1 Zenodo v2、375ケース
- 最大600点normal/abnormal window
- AR(3)
- strong AR prior
- service aggregation: `mean_top3`
- winsorizationなし
- 結果先: `results/ablation/rcaeval_re1/adaptive_direct_ar_bayes_factor/`

## 実行

```bash
./scripts/run_ablation.sh --adaptive-direct-ar-bayes-factor
```

実行後の対応比較:

```bash
PYTHONPATH=src:. venv/bin/python \
  scripts/analyze_adaptive_direct_ar_bayes_factor.py
```

比較対象はRaw+BF、暫定最終候補、Full Direct AR-BF、Intercept-shift AR-BFである。

## ロールバックablation候補

統合candidateが改善した場合は、次を順に外して寄与を確認する。

1. normal-only empirical-null calibrationを外す
2. onset候補を0だけに戻す
3. step以外の応答形状を外す
4. `step_ramp`の2係数modelを外す
5. per-row正規化だけを外す

改善しない場合でも、case diagnosticsにraw BF、normal-only BF、MAP形状、onset posteriorを保存しているため、失敗要因を分解できる。

## 現在の状態

正式375ケースの実験まで完了した。macroは `AC@1=0.6880`、`AC@3=0.8800`、`AC@5=0.9520`、`Avg@5=0.8576`である。Full Direct AR-BFに対するAC@1差は+0.2027、Intercept-shift AR-BFに対しては+0.0827だった。統合candidate全体の改善は確認できたが、各構成要素の寄与はrollback ablationで分離する。

## Shapeとonsetの厳密な定義

abnormal windowの長さを \(L\)、window内の0-origin時刻を \(t=0,\ldots,L-1\)、fault boundaryからの応答開始offsetを \(a\)とする。現在は \(a\in\{0,5,15\}\) である。経過時間を \(h=t-a\) とし、すべてのbasisは \(t<a\) で0とする。

H1のconditional meanは、

\[
X_t=c+\sum_{p=1}^{P}\phi_pX_{t-p}
+\mathbf g_m(t;a)^\top\boldsymbol\delta_m+\epsilon_t
\]

である。shapeは観測値 \(X_t\) そのものの形ではなく、AR方程式へ追加するconditional-mean interventionの時間形状である。

### Step

\[
g_{\mathrm{step}}(t;a)=I(t\ge a)
\]

onset後に一定入力が継続する。

### Ramp

実装上は、

\[
g_{\mathrm{ramp}}(t;a)=
\begin{cases}
0 & t<a\\
\dfrac{t-a+1}{L-a} & t\ge a
\end{cases}
\]

である。onset時点で \(1/(L-a)\) から始まり、window末尾で1になる。onset時点を厳密に0とするpure slope-change basisではない点は、実装上の定義として明記する。

### Exponential rise

half-life parameterを \(H=10\) とし、

\[
g_{\mathrm{rise}}(t;a)=
\begin{cases}
0 & t<a\\
1-2^{-(t-a+1)/H} & t\ge a
\end{cases}
\]

とする。入力は0付近から1へ漸近する。

### Exponential decay

\[
g_{\mathrm{decay}}(t;a)=
\begin{cases}
0 & t<a\\
2^{-(t-a)/H} & t\ge a
\end{cases}
\]

onset時点で1、10点後に0.5となる一過性入力である。

### Step + ramp

`step_ramp`は5番目の単一shapeではなく、2本のbasisを持つ複合modelである。

\[
\mathbf g_{\mathrm{step+ramp}}(t;a)=
\begin{bmatrix}
g_{\mathrm{step}}(t;a)\\
g_{\mathrm{ramp}}(t;a)
\end{bmatrix}
\]

\[
u_t=
\delta_{\mathrm{step}}g_{\mathrm{step}}(t;a)
+\delta_{\mathrm{ramp}}g_{\mathrm{ramp}}(t;a)
\]

とし、即時的なlevel changeとその後のslope changeを別係数で表す。`step_ramp`は \(\delta_{\mathrm{ramp}}=0\) でstep、\(\delta_{\mathrm{step}}=0\) でrampを含む。そのため解釈しやすい最小のlevel+slope拡張である一方、step/ramp単体とmodel familyが重複する。この追加は理論的必然ではなく設計選択であり、`adaptive_direct_no_step_ramp` で寄与を検証する。

## Intervention shapeと観測応答の違い

AR(1)で正常状態からのずれを \(Y_h\) とすると、

\[
Y_h=\phi Y_{h-1}+u_h
\]

より、

\[
Y_h=\sum_{j=0}^{h}\phi^{h-j}u_j
\]

である。観測応答はintervention basisをAR dynamicsでfilterした形になる。例えばstep入力 \(u_j=\delta\) に対し、

\[
Y_h=\delta\frac{1-\phi^{h+1}}{1-\phi}
\]

となるため、step入力でも観測上はexponential riseのように見える。`exp_rise`は、AR自体の持続性とは別にinterventionそのものが遅く立ち上がる仮説である。両者は完全に識別可能とは限らず、shape間の相関は現在のmodel libraryの制約である。

## Shape/onsetの周辺化

各shapeの事前確率は等しく、onset priorは、

\[
p(a)\propto e^{-0.15a}
\]

である。H1全体の周辺尤度は、

\[
p(D\mid H_1)
=\sum_m\sum_a p(m)p(a)p(D\mid m,a,H_1)
\]

とし、15候補の最大値ではなく重み付き平均を使う。診断JSONのMAP shape/onsetは、

\[
(m^*,a^*)=\arg\max_{m,a}p(m,a\mid D,H_1)
\]

であり説明用の代表候補である。最終scoreはMAP候補だけではなく15候補の周辺化を使う。

## 計算量と数値等価な高速化

正式candidateは1 metricあたり実faultと1回、normal-only pseudo-faultと3回の計4回、それぞれH0+15候補を評価する。従来実装では375ケースの純Python時間が2803.29秒であった。高速化では仮説、prior、周辺尤度、scoreを変更せず、次を実施した。

- Python loopによるconditional-AR design作成をsliding-windowのNumPy処理へ置換
- 同じ長さ、shape、onset、half-lifeのintervention basisをLRU cacheで再利用
- 15候補で共通する \(X^\top X\)、\(X^\top y\)、\(y^\top y\) を一度だけ計算
- 実faultではMAP候補だけ詳細posteriorを作成し、pseudo-faultでは不要な係数変換・spectral-radius・診断dict生成を省略
- benchmark case単位のprocess並列を追加し、Adaptive Direct専用実行は既定4 workersとする

`re1_tt__ts-auth-service_cpu__1`の1241 metricsを用いた1ケースsmokeで、旧artifactの17.07秒に対し高速化後は2.66秒で、6.42倍の高速化だった。service rankingは完全一致し、metric scoreの最大絶対差は \(2.66\times10^{-13}\) だった。これは小規模smokeであり、375ケースのwall-clock改善率は正式再実行後に確定する。
