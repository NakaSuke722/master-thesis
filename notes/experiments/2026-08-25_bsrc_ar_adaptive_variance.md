# BSRC-AR adaptive variance integration validation

実装日: 2026-08-25

## 事実: component ablationから生じた問題

入力列修正後のcomponent ablationでは、variance spike-and-slabの有無を固定したままquadratureをq4からq8へ増やすと、RCAEvalの順位性能が大きく低下した。一方、同じqでslabとspike-and-slabを比較した差はほぼなかった。

case artifactを調べると、root service metricのMAP variance ratioが最大quadrature nodeに集中していた。固定Gauss--Hermiteの最大比はq4で約5.1、q8で約18.2であり、q4は積分近似というより暗黙のvariance-ratio capとして働いていた可能性が高い。この状態ではq4の高スコアを「積分が収束したBSRC-ARの性能」と解釈できない。

## 解釈: 何を修正するか

分散比を

\[
\eta=\log r=\log\frac{\sigma^2_{post}}{\sigma^2_{pre}},
\qquad
\eta\sim\mathcal N(0,s_\eta^2)
\]

とする。coefficient-change maskを \(\gamma\) とすると、必要なslab marginalは、

\[
p(D\mid\gamma,\text{variance slab})
=\int p(D\mid\gamma,\eta)p(\eta)\,d\eta
\]

である。固定GHはprior中心0の周囲にnodeを置くため、尤度がtail側に強く移動した場合に粗くなる。新variantではlog integrand、

\[
f_\gamma(\eta)
=\log p(D\mid\gamma,\eta)+\log p(\eta)
\]

の事後mode \(\hat\eta_\gamma\) と負の二階微分、

\[
h_\gamma=-f_\gamma''(\hat\eta_\gamma)
\]

を求め、各maskに対して、

\[
\eta_i=\hat\eta_\gamma+\sqrt{\frac{2}{h_\gamma}}x_i
\]

へGauss--Hermite nodeを移動・拡縮する。これはAdaptive Gauss--Hermite quadratureであり、固定最大nodeによる分散比の上限をモデルへ入れない。

固定GHでは各nodeを別candidateとしていたが、adaptive版ではvariance slabを積分した結果を各coefficient maskにつき1 candidateとする。variance spikeは従来どおり \(r=1\) の離散candidateである。AR(3)・spike-and-slabでは、15個のcoefficient-only candidateと16個の積分済みvariance-slab candidate、合計31 candidateになる。

## 実装した検証順序

### 1. Synthetic variance-ratio validation

真のvariance ratioを1, 1.5, 2, 4, 8, 16と変え、fixed q4/q8とadaptive q7/q11/q15を比較する。

```bash
PYTHONPATH=src:. venv/bin/python \
  scripts/validate_bsrc_variance_integration.py
```

出力:

```text
results/analysis/bsrc_variance_integration_synthetic/
├── replicates.csv
├── summary.json
└── summary.md
```

主な収束診断は、同じsynthetic sampleにおけるadaptive q11とq15のlog BF差、およびposterior mean variance ratio差である。q11とq15が一致しない領域では、RCAEvalへ進む前に積分を再検討する。

### 2. Normal-only pseudo-fault

正常windowを40/60、50/50、60/40に分割し、後半をpseudo-faultとして偽のregime-change evidenceを測る。

```bash
PYTHONPATH=src:. venv/bin/python \
  scripts/analyze_bsrc_adaptive_pseudo_fault.py \
  --workers 4
```

出力:

```text
results/analysis/bsrc_variance_integration_pseudo_fault/
├── case_calibration.csv
├── summary.json
└── summary.md
```

ここではcaseごとの最大service log BF、positive/strong service率、variance-change posterior probabilityを確認する。syntheticで収束してもnormal-onlyで偽陽性が多ければ、variance priorまたはservice aggregationの校正が必要である。

### 3. Stratified RCAEval subset

3 dataset × 5 fault type × 5 casesの75ケースを、固定seedで層化抽出する。

```bash
PYTHONPATH=src:. venv/bin/python \
  scripts/run_bsrc_adaptive_subset.py \
  --workers 4
```

出力:

```text
results/sensitivity/rcaeval_re1/bsrc_ar_adaptive_subset/
├── selection_manifest.json
├── summary_service.json
└── service/<dataset>/*.json
```

subsetは正式375ケースとは別categoryへ保存する。manifestにcase_id、dataset、fault type、root serviceを残し、dataset/faultの偏りを防ぐ。

### 4. Full RCAEval 375 cases

上記3診断が安定した場合だけ実行する。

```bash
./scripts/run_ablation.sh --bsrc-ar-adaptive
```

出力:

```text
results/ablation/rcaeval_re1/bsrc_ar_adaptive_variance/
```

## 判断規則

この変更の目的はq4のスコアを再現することではなく、分散積分が数値的に収束したBSRC-ARを評価可能にすることである。次の順で判断する。

1. adaptive q11とq15がsyntheticの全variance-ratio領域で概ね一致するか。
2. normal-only pseudo-faultで、固定q8より極端な偽陽性が増えていないか。
3. 層化subsetで特定dataset/faultだけが大きく悪化していないか。
4. 以上を満たした場合のみ375ケースの順位性能を比較する。

q4よりRCAEvalスコアが低くても、q11/q15が収束していれば、q4の高さは暗黙capによる正則化効果だったと解釈する。その場合は積分点を戻すのではなく、説明可能なlog-variance priorのscaleまたはrobust innovation modelを別の感度分析軸として検証する。

## 実装時smoke

正式成果物を作らない一時ディレクトリで、次を確認した。

- syntheticを各ratio 2反復で実行し、adaptive q11/q15のlog BF差は最大でも約 \(7\times10^{-13}\) だった。
- 実processed dataから3 dataset × 5 fault type × 5件、合計75件が重複なく選択された。
- `re1_ob__cartservice_cpu__3` を設定読込からAMBER、artifact保存まで実行し、46 metricsを約1.1秒で処理してroot serviceをTop-1とした。これは精度評価ではなく実行経路の確認である。
- normal-only pseudo-faultを1ケースだけsmokeしたところ、最大service log BFは大きかった。この1ケースから偽陽性率は結論しないが、pseudo-fault全件を省略して375-case評価へ進むべきではない。

正式synthetic、normal-only全件、75-case subset、375-case ablationはいずれも未実行である。
