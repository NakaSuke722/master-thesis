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

実装、synthetic test、RCAEval 1ケースsmokeまで完了。正式375ケースは未実行である。
