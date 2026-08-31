# RUN集計処理の軽量化と比較手法の結果確認

## 目的

RUNの`re1_tt 125/125`表示後にsummaryが出ないという報告を受け、
推論を再実行せずに集計できるようにする。AMBER、RUNを含む各手法の
スコアリング、学習条件、service集約方法は変更しない。

## 確認できた事実

- 対象checkout: `/Users/naka_suke_722/学業/大学院/研究/master-thesis`、branch `main`。
- 集計のheader-only readerは`amber_diagnostics`だけに対応していた。
  RUNの`run_diagnostics.attention_by_target`はmetric数に対して二次の大きさを持ち、
  summaryに不要だが従来は全JSONを読み込んでいた。
- 元の`Progress 125/125`は完了数ではなくcase番号で、しかもJSON保存前の表示だった。
  並列実行ではその表示だけで全caseの保存完了とは判断できない。
- 調査時にこのcheckoutで確認できたRUN結果は130件（OB 125、SS 5、TT 0）、計45,496,114 bytes。
  実行中のrunner・aggregateプロセスは確認できなかった。
  ユーザーが報告した375件終了時の状態とは一致せず、別checkout・実行環境の可能性は未確認。
  **報告された停止そのものを再現・解消できたとは判断しない。**
- 保存済み130件の集計を変更前のコードと比較した。辞書全体が完全一致し、
  読込・集計時間は約0.362秒→0.005秒だった。1回のlocal測定であり、
  キャッシュやデータ規模の影響を含む。375件の性能保証ではない。

## 修正

1. header-only readerをRUNと他のbaselineの末尾diagnosticsにも適用。
   main.pyのindent=4・diagnostics末尾形式を対象とする。
   headerと小さな末尾部分だけを読み、通常の書込中断では全読込へfallbackしてエラーにする。
   これは診断JSON内部全体の検証ではないため、resumeの完了判定には転用しない。
2. `./scripts/run_baselines.sh --run --aggregate-only`を追加。
   runnerを起動せず保存済み結果だけを集計する。
3. baseline集計では`--require-complete`を有効化。
   前処理済みcase IDとの厳密な照合、実験identity、case IDとファイル名、必須評価指標を確認し、
   不足caseがあればsummaryを出力・上書きせず停止する。
4. caseの表示を保存後に移動し、完了数と紛らわしい`Progress`を`Case`へ変更。
   scriptにも推論終了→集計開始を表示し、集計中はdatasetごとの読込件数を表示する。
5. 集計だけの再実行では、identityの一致する既存summaryの実行時間を保持する。
   既存summaryもwall-clock指定もなければ、従来のcase時間合計へのfallbackを維持する。
   これは並列実行時のwall-clock時間ではない。

## 他の手法の結果

以下はユーザー提示値と保存済みsummaryの一致を確認し、各375件のcase JSONからの
再集計でも同じ数値になることを確認した。推論は再実行していない。

| 手法 | Dataset | AC@1 | AC@3 | AC@5 | Avg@5 |
|---|---|---:|---:|---:|---:|
| ε-Diagnosis | re1_ob | 0.032 | 0.240 | 0.376 | 0.2240 |
| ε-Diagnosis | re1_ss | 0.000 | 0.008 | 0.008 | 0.0048 |
| ε-Diagnosis | re1_tt | 0.000 | 0.032 | 0.032 | 0.0240 |
| ε-Diagnosis | macro | 0.0107 | 0.0933 | 0.1387 | 0.0843 |
| CIRCA adapter | re1_ob | 0.552 | 0.864 | 0.936 | 0.8096 |
| CIRCA adapter | re1_ss | 0.544 | 0.832 | 0.952 | 0.7840 |
| CIRCA adapter | re1_tt | 0.424 | 0.664 | 0.736 | 0.6224 |
| CIRCA adapter | macro | 0.5067 | 0.7867 | 0.8747 | 0.7387 |

### 解釈と次アクション

- CIRCAは3 datasetで候補を比較的上位に含められている。ただしPCのmetric選択・条件集合制約を
  導入したadapter版であり、論文実装そのままの結果として表記しない。
- ε-Diagnosisの特にSS・TTは低スコアだが、これだけで手法自体の無効性とは結論しない。
  今後比較表を確定する前に、候補を出せたcaseの比率、空ランキング、正常・異常区間での
  metric変動や判定の内訳を点検する余地がある。今回その分析・手法変更は行っていない。
- RUNの375件がある実際の保存先を確認してから、上記aggregate-onlyを実行する。
  このcheckoutでは130件のため正式summaryは作成していない。既存case JSONも変更していない。
- 再実行が必要と確認された場合のみ、`./scripts/run_baselines.sh --run 4`で未完了caseを再開する。
  まず保存先の相違を確認し、不必要に学習をやり直さない。

## 検証

- pytest: 大きいRUN diagnosticsの読込量上限、全読込との集計一致、各baseline marker、
  chunk境界、旧形式fallback、書込途中JSON、case欠落・誤identity時の停止、
  既存summary保持、aggregate-onlyがrunnerを起動しないことをテスト。
- 8 MiBのsynthetic diagnosticsでもheader readerの読込量は最大65 KiB。
- `venv/bin/python -m pytest -q`: **180 passed**。
- `zsh -n scripts/run_baselines.sh`、`git diff --check`: 成功。
- 実際のaggregate-onlyコマンドは約0.63秒で不足245件を検出し、summaryを書かず終了した。
  正式375件の推論は実行していない。
