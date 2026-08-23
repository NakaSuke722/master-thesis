# Intercept-shift AR Bayes Factor implementation

実装日: 2026-08-23

## 目的

Full Direct AR-BFは平均level、AR係数、innovation varianceの変化をすべて検出し、RCAEvalでは非root serviceの構造変化へ過剰反応した。そこで、persistent shiftを表現するintercept変化だけを追加自由度とする制約付きAR Bayes Factorを実装する。

## 仮説

AR(P)を次で表す。

\[
X_t=c_s+\sum_{p=1}^{P}\phi_pX_{t-p}+\epsilon_t,
\qquad
\epsilon_t\sim\mathcal N(0,\sigma^2),
\]

`s` はpre/post segmentを表す。比較する仮説は次の通りである。

\[
H_0:c_{pre}=c_{post}=c,
\quad \boldsymbol\phi_{pre}=\boldsymbol\phi_{post},
\quad \sigma^2_{pre}=\sigma^2_{post},
\]

\[
H_1:c_{pre}\neq c_{post},
\quad \boldsymbol\phi_{pre}=\boldsymbol\phi_{post},
\quad \sigma^2_{pre}=\sigma^2_{post}.
\]

H1が追加するのはpost用interceptの1パラメータだけである。レベルシフトをpost interceptの変化として明示的に検定し、AR係数や分散変化を同時に自由化しない。ただし、定常AR過程の長期平均変化を \(\Delta\) とするとintercept変化は \((1-\sum_p\phi_p)\Delta\) である。\(\sum_p\phi_p\) が1に近い場合の減衰まで完全に解消する方法ではない。

## 実装

H0の回帰designは `[1, lag_1, ..., lag_P]`、H1は `[I(pre), I(post), lag_1, ..., lag_P]` とする。H1でもlag列とinnovation varianceは1組だけなので、pre/postで共有される。両仮説のNormal-Inverse-Gamma marginal likelihoodは解析的に計算し、

\[
\log BF=\log p(\mathbf x\mid H_1)-\log p(\mathbf x\mid H_0)
\]

をmetric scoreとする。標準化にはpre期間だけの中央値とrobust scaleを使い、post情報をpriorやscaleに混入させない。

- コア: `src/models/ar_bayes_factor.py`
- AMBER mode: `residualization=ar_model`, `scoring=ar_intercept_bayes_factor`
- 正式config: `configs/ablation/rcaeval_re1_zenodo_v2/intercept_shift_ar_bayes_factor.yaml`
- 結果先: `results/ablation/rcaeval_re1/intercept_shift_ar_bayes_factor/`

Full Direct AR-BFと同じAR(3)、strong prior、`mean_top3`、最大600点windowを使い、仮説の制約以外の条件を合わせる。

## 合成検証

- 無変化: `log BF < 0`
- persistent mean shift: `log BF > 10`
- 平均を変えない純粋なAR係数変化: `log BF < 0`
- 平均を変えない純粋なinnovation variance変化: `log BF < 0`

これは「intercept変化だけを検出する」という実装意図のsanity checkであり、RCA性能を示す結果ではない。

## 実行方法

375ケースの実行:

```bash
./scripts/run_ablation.sh --intercept-shift-ar-bayes-factor
```

既存のRaw+BF、暫定最終候補、Full Direct AR-BFとの対応比較:

```bash
PYTHONPATH=src:. venv/bin/python \
  scripts/analyze_intercept_shift_ar_bayes_factor.py
```

分析結果は `results/analysis/intercept_shift_ar_bayes_factor/` に保存する。

## 現在の状態

実装、synthetic test、RCAEval 1ケースのsmokeのみ完了した。正式375ケースは未実行である。
