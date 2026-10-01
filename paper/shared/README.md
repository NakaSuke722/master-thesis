# 共通資料

`amber_notation.tex` は、修論本文・図・発表スライドで参照するAMBERの記号早見表です。
本文から独立したA4・2ページの資料で、`jlreq`、`newtxtext` / `newtxmath`、
指示関数 `\ind` の定義は既存の論文スタイルに合わせています。
共通プリアンブルは未整備のため、本資料内で必要最小限を定義しています。

- 1ページ目：入力系列、正常期間だけでの標準化、AR予測、残差、不確実性補正
- 2ページ目：ベイズ因子、サービス集約・順位、AC@K・Avg@Kとマクロ平均

会話で整理した記法に基づき、切片だけはケース添字 `c` との衝突を避け `c_0` としています。
予測誤差の式は `counterfactual_ar`、予測先ごとの不確実性補正（`diagonal`）、
サービス集約は `mean_top3` を対象とします。数値的な例外処理の全仕様ではありません。
記号の確認先は `src/models/amber.py`、`src/evaluation/metrics.py`、
`configs/main/rcaeval_re1_zenodo_v2.yaml` です。

## コンパイル

TeX Live（upLaTeX、dvipdfmx、latexmk、原ノ味フォント）を使用し、リポジトリルートで実行します。

```bash
latexmk -pdfdvi \
  -e '$latex=q/uplatex %O -interaction=nonstopmode -halt-on-error %S/' \
  -e '$dvipdf=q/dvipdfmx %O -o %D %S/' \
  -outdir=paper/shared/build paper/shared/amber_notation.tex
```

出力は `paper/shared/build/amber_notation.pdf`。PDFと中間ファイルは既存の `.gitignore` により管理対象外です。
`ref.bib` は既存の共有文献ファイルです。
