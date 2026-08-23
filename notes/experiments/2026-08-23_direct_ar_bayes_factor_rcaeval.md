# Direct AR Bayes Factor RCAEval 375-case protocol

実装日: 2026-08-23

## 目的

ARを残差化の前処理として使うのではなく、`H0: pre/postで同じAR過程` と `H1: pre/postで異なるAR過程` の直接Bayes Factorをroot-cause rankingに用いられるか検証する。第1段階のsynthetic検出性と第2段階のnormal-only校正に続く、第3段階の実際のRCA性能評価である。

## 固定条件

- データ: RCAEval RE1 Zenodo v2、`re1_ob/re1_ss/re1_tt`、各125ケース
- normal/abnormal window: 最大600点
- AR次数: 3
- service aggregation: `mean_top3`
- prior: normal-only pseudo-faultの3候補中で最も保守的だったstrong profile
- winsorization: なし（AR likelihoodをclipしない）
- AMBERの既存スコアリングは変更しない

strong priorはintercept precision 0.1、lag precision 10.0、InvGamma `alpha=5, beta=4`である。これは「校正済み」ではなく、試した3候補の中で最も偽証拠が少ない候補である。

## 実装

- 正式config: `configs/ablation/rcaeval_re1_zenodo_v2/direct_ar_bayes_factor.yaml`
- AMBER mode: `residualization=ar_model`, `scoring=ar_bayes_factor`
- 実行: `./scripts/run_ablation.sh --direct-ar-bayes-factor`
- 対応比較: `scripts/analyze_direct_ar_bayes_factor.py`
- 分析出力予定: `results/analysis/direct_ar_bayes_factor/`

AMBERのDirect modeはmetricごとにpre/postのconditional AR designを作り、shared modelとseparate modelの解析的marginal likelihood差をservice内の上佉3 metricで平均する。診断artifactにはpre/post/sharedの係数posterior mean、innovation variance、spectral radius、long-run meanを保存する。

## 事前に定める比較

1. Direct AR-BF vs Raw+BF
2. Direct AR-BF vs `Stationary Counterfactual AR + horizon-aware uncertainty`
3. overall、dataset別、fault type別のAC@1/3/5とAvg@5
4. paired bootstrap 95% CI、AC@1のexact McNemar test、root-service rankとTop-1の得失
5. Direct AR-BFのroot service log BFと最上位競合serviceとのmargin

## 判断規則

- Direct AR-BFが高性能でも、normal-onlyで大きな偽証拠が確認されたため、RCAEvalの結果だけで絶対BFの校正性を主張しない。
- 暫定最終候補の置き換えは、対応のあるTop-1改善だけでなく、dataset/faultの一貫性、順位深度、normal-onlyの校正問題を合わせて判断する。
- Train Ticketでの改善は、normal-onlyの非定常性をroot-cause signalと取り違えた可能性を優先診断する。

## 現在の状態

実装と検証手順は完了。正式375ケースの実行と結果記録は未実施である。
