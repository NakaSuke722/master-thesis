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

この時点では、正式synthetic、normal-only全件、75-case subset、375-case ablationはいずれも未実行だった。その後の実行結果を以下に追記する。

## 2026-08-27 実験結果

### 目的

固定Gauss--Hermite quadratureのnode上限に依存しないAdaptive BSRC-ARについて、次の3点を順番に確認した。

1. 分散比積分が数値的に収束するか。
2. 障害が存在しないnormal-only分割で、regime changeを過剰検出しないか。
3. RCAEvalのroot-cause service順位で、正式375ケースを実行する価値がある程度の性能を示すか。

3番目は正式評価ではなく、3 dataset × 5 fault type × 5 casesの層化subsetによる早期判断である。計算量を抑えつつ、datasetまたはfault typeの偏りで判断しないことを目的とした。

### 試したモデル

Adaptive BSRC-ARは、正常期間と障害期間について、

\[
H_0:\text{同一のAR生成regime}
\]

と、

\[
H_1:\text{intercept、AR係数、innovation varianceの疎なregime change}
\]

をBayes Factorで比較する。AR(3)のinterceptと3つのlag係数にはspike-and-slab型のchange indicatorを置き、post/pre innovation variance ratioにはlog-normal slabを置いた。

初期BSRC-ARはvariance ratioを固定GH nodeで近似したが、q4が約5.1、q8が約18.2という異なる最大nodeを持ち、root metricのMAPが最大nodeへ集中した。そのため、本実験では各coefficient-change maskの事後modeと曲率に合わせるAdaptive Gauss--Hermite quadratureへ置き換えた。目的はスコアを上げることではなく、任意の固定node上限に依存しない連続的なvariance-ratio marginalizationを実現することだった。

### 1. Synthetic variance-ratio validation

真のvariance ratioを1、1.5、2、4、8、16と変え、adaptive q11とq15を比較した。

全ratioでq11/q15のlog BF差は最大でも約 \(3.4\times10^{-13}\)、posterior mean variance ratio差は最大でも約 \(3.3\times10^{-12}\) だった。したがって、少なくともこのsynthetic設定ではAdaptive Gauss--Hermite積分は十分に収束した。

出力:

```text
results/analysis/bsrc_variance_integration_synthetic/
```

これは「Adaptive BSRC-ARがRCAに有効」という結果ではなく、「固定q4/q8のnode選択問題を数値積分として解消できた」という結果である。

### 2. Normal-only pseudo-fault

375ケースの正常windowを40/60、50/50、60/40に分割し、後半を偽の障害期間として評価した。各分割で375ケース、合計1,125 case-conditionを評価した。

| Fit fraction | Median case max log BF | P90 | Positive services | Strong services |
|---:|---:|---:|---:|---:|
| 0.4 | 444.0819 | 833.1564 | 91.8% | 87.6% |
| 0.5 | 427.9077 | 809.0514 | 91.4% | 87.0% |
| 0.6 | 407.7942 | 748.4107 | 91.5% | 86.3% |

正常データだけで約9割のserviceが正のlog BFを持ち、約86--88%がstrong evidenceになった。特に `re1_tt` ではpositive service率が約98%、strong service率が約95--96%だった。また、metric-level計算失敗が合計33,526件あった。

出力:

```text
results/analysis/bsrc_variance_integration_pseudo_fault/
```

この結果は、数値積分が収束しても、現在のBSRC仮説がRCAEvalの正常window内変動に対して十分に校正されていないことを示す。正常window自体の非定常性、単一の定常AR(3)では表現できない時間変化、Gaussian innovation仮定の破れなどを、障害regime changeとして検出している可能性がある。

### 3. Stratified RCAEval 75-case subset

実行コマンド:

```bash
PYTHONPATH=src:. venv/bin/python \
  scripts/run_bsrc_adaptive_subset.py \
  --workers 4
```

選択条件:

- `re1_ob`、`re1_ss`、`re1_tt`
- `cpu`、`mem`、`disk`、`delay`、`loss`
- 各dataset/fault typeセルから固定seedで5ケース
- 3 × 5 × 5 = 75ケース
- service-level、`mean_top3`
- normal/abnormalはそれぞれ最大600点

manifestとresult artifactを照合し、15セルすべてが5ケース、結果JSONが75件存在することを確認した。

| Dataset | AC@1 | AC@3 | AC@5 | Avg@5 |
|---|---:|---:|---:|---:|
| re1_ob | 0.2000 | 0.4800 | 0.8400 | 0.5200 |
| re1_ss | 0.5600 | 0.7600 | 0.8400 | 0.7280 |
| re1_tt | 0.0400 | 0.3200 | 0.4800 | 0.3120 |
| macro | 0.2667 | 0.5200 | 0.7200 | 0.5200 |

実行時間:

- wall-clock: 153.1秒
- 4 workerのpure Python時間合計: 584.38秒

出力:

```text
results/sensitivity/rcaeval_re1/bsrc_ar_adaptive_subset/
├── selection_manifest.json
├── summary_service.json
└── service/<dataset>/*.json
```

### 結果の解釈

この75ケースは正式375ケースの間引きであり、正式スコアそのものではない。そのため、75ケース値を既存375ケース値と厳密なpaired comparisonとして扱うことはできない。

ただし、結果は採用判断用の早期検証として十分に厳しい。特に `re1_tt` のAC@1は0.04で、25ケース中root serviceを1位にできたのは1ケースだけである。macro AC@1も0.2667にとどまり、Adaptive BSRC-ARが少数のdatasetだけでなく、RCAEval全体で安定したroot-cause rankingを提供する兆候は得られなかった。

syntheticで積分は収束した一方、normal-onlyでは強い過剰検出があり、実障害subsetではroot-cause rankingも低かった。したがって、今回の低下を単なるquadrature実装問題として説明することはできない。現在の問題は、より上位のモデル仮定、すなわち「正常期間を単一の定常AR regimeで表し、postとの差をparameter changeとして測る」という仮説とRCAEvalデータの不一致にある可能性が高い。

## 結論と今後

Adaptive BSRC-ARについて、正式375ケースは実行しない。75ケースは正式値ではないものの、次の3点が揃ったためである。

1. 数値積分自体は収束しており、追加のquadrature調整で解決する問題ではない。
2. normal-only pseudo-faultでregime change evidenceが著しく過剰だった。
3. 層化75ケースでroot-cause rankingが低く、特にTrain Ticketで不安定だった。

BSRC-ARはAMBERの最終暫定方式には採用せず、生成過程そのものをBayesian hypothesisへ組み込む理論的拡張と、その限界を示した探索結果として位置づける。今回の修士論文の範囲では、これ以上モデルを追加して追試しない。

将来再検討する場合の候補は、robust innovation、time-varying AR、normal regime自体のmixture、service-level hierarchical shrinkageなどである。ただし、これらは新しいモデル研究に相当する。今回の結果を受けた直接のnext actionはBSRCをさらに太らせることではなく、既存のCounterfactual AR + Bayes Factorを中心にAMBERの主張、アブレーション、限界を整理し、修士論文執筆へ移行することである。
