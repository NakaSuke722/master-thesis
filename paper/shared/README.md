# 共通資料

`amber_notation.tex` は、修論本文・図・発表スライドで参照するAMBERの記号と計算を説明する補足資料です。
本文から独立したA4・7ページの資料で、`jlreq`、`newtxtext` / `newtxmath`、
指示関数 `\ind` の定義は既存の論文スタイルに合わせています。
共通プリアンブルは未整備のため、本資料内で必要最小限を定義しています。

- 1〜2ページ目：入力の前提、メトリクスとサービス、観測値と系列の次元、期間分割と添字の対応例
- 3ページ目：正常期間だけでの標準化、1期先予測と反実仮想予測の計算例
- 4ページ目：不確実性補正、標準化誤差、似た記号の読み分け
- 5ページ目：ベイズ因子、サービス集約・順位と集約例
- 6ページ目：正解順位、AC@K・Avg@Kの比較表
- 7ページ目：ケース平均とマクロ平均、その違いの数値例

数値例はすべて説明用の仮想例であり、実験結果ではありません。

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
