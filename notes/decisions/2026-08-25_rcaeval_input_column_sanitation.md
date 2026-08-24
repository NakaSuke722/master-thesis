# RCAEval入力列の妥当性修正

作成日: 2026-08-25

## 背景

BSRC-ARのcase-level diagnosticsを調べたところ、Online Boutiqueの一部raw CSVには`time`ヘッダが2本含まれていた。`pandas.read_csv`は2本目を`time.1`へ自動改名するが、従来の前処理は`time`だけを削除していた。このため、重複timestampがservice metricとして推論へ混入していた。

また、normal/abnormalの全入力点で同一値を取るメトリクスにもBSRC-ARが大きな分散変化Bayes Factorを与えるケースが確認された。このような列は障害前後を区別する情報を持たず、RCAの候補に含めるべきではない。

## 決定

RCAEval RE1 Zenodo v2の共通前処理へ、モデルやデータセット固有ではない次の規則を追加する。

1. `time.1`、`time.2`など、CSVの重複`time`ヘッダに対してpandasが生成するaliasを除外する。
2. case全期間を通して値が1種類しかないメトリクスを除外する。
3. 正式なnormal末尾最大600点とabnormal先頭最大600点を選んだ後、その結合区間で値が1種類しかないメトリクスも除外する。
4. normal区間では定数でもabnormal区間で変化するメトリクスは保持する。
5. 除外後に有効なメトリクスが1本も残らないcaseは、推論を続けず明示的なエラーとする。

特定の`PassthroughCluster`などを名前で除外する処理は導入しない。完全定数かどうかという情報量だけで判定する。

## 影響

raw 375ケースを読み取り専用で監査した結果、修正後も全caseでground-truth serviceに属するメトリクスが1本以上残った。

既存processedデータはこの修正前に生成されているため、自動的には更新されない。今後の正式比較では、まず次を実行して375ケースを再前処理する必要がある。

```bash
PYTHONPATH=src:. venv/bin/python \
  src/prepare_rcaeval_re1.py \
  --config configs/main/rcaeval_re1_zenodo_v2.yaml
```

この前処理は全方式の共通入力を変更する。したがって、修正前のmain・Raw+BF・Counterfactual AR・Adaptive Direct・BSRC-ARなどの数値と、修正後の数値を同一条件の正式比較として混在させない。少なくとも最終採用候補と比較baselineは、同じ再生成済みprocessedデータで再実行する。

## 検証方針

- synthetic DataFrameで`time.N`と完全定数列が除外されること
- normalでは定数でも障害後に変化する列が保持されること
- 選択window外だけで変化する列が、実際の推論windowでは除外されること
- 正式raw 375ケースでground-truth serviceのメトリクスが消えないこと

