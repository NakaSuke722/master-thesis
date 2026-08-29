# metric-based RCA比較手法の実装・評価プロトコル

## 目的

最終AMBERを既存のmetric-based RCA手法と公正に比較するため、
ε-Diagnosis、RCD、CIRCA、RUNを同一のRCAEval RE1条件で実行できるようにする。
`data/raw/baro` という旧データセット名、モデルとしてのBARO、今回追加する4手法は
別物として扱う。

## 全手法で固定する事実

- データ源: Zenodo record `14590730`、RCAEval v2
- 対象: `re1_ob`、`re1_ss`、`re1_tt`、各125ケース、合計375ケース
- 入力window: 正常・異常とも利用可能な最大600点
- 障害開始時刻: 既知。processed dataのnormal/abnormal分割をそのまま使う
- 正式評価: service-level、AC@1/3/5およびAvg@1/3/5
- metric順位からservice順位への変換: metric順位を上から走査し、service名の初出順に重複除去
- AMBERの`mean_top3`はAMBER固有の集約であり、既存手法へ強制しない
- GT service・fault typeは推論入力へ渡さない
- 空の候補集合は例外や任意順位へ置換せず、全Top-k不正解として評価する
- 各ケースJSONへfull ranking、モデルparameter、手法別diagnosticsを保存する

この固定により、「同じデータと評価尺度」を保証する。一方、各論文固有の前処理・
グラフ学習・スコア定義までAMBERへ揃えることは、手法そのものを変えてしまうため行わない。

## 手法別の実装事実

### ε-Diagnosis

- 参照: Salesforce PyRCA commit
  `411310d589fac5cb8e7bdce67d33eadb091a1083` とRCAEval adapter commit
  `526cdd5818ea9d8c2a34e869ebd637bc6b4fa4b8`
- 正常・異常segmentを同じ長さにし、正常側は末尾、異常側は先頭を使用する
- PyRCAと同じbootstrap相関統計量、`alpha=0.01`、bootstrap 200回を用いる
- 再現性のため乱数seedを固定する
- AC@5を定義できるよう、返却上限をPyRCA既定の3ではなく5に明示する

最後の変更は検出統計量を変えず、評価する候補数だけをAC@5へ合わせるadapter変更である。

### RCD

- 参照: Salesforce PyRCAとRCAEvalのlocalized multi-phase RCD
- 正常・異常を表すF-nodeを追加する
- `bins=5`のk-means離散化、`gamma=5`のランダムchunk、localized PCを用いる
- phase 1は`alpha=0.01`から、phase 2は`alpha=0.001`から0.1刻みで探索する
- chunk順序をseedで固定し、候補上限を5にする
- 標準`causal-learn`に存在しないlocalized skeleton部分は、PyRCA同梱版の探索手順を
  現行`causal-learn`のchi-square CI testへ接続した

### CIRCA

- 参照: NetManAIOps/CIRCA commit
  `0215e1880096aa02a305c697f1c23cac4600ebd2`、論文artifact commit
  `1522ddd7efd16db55e9f351fd70324501ce9134e`、RCAEval adapter
- RCAEvalにはCIRCAが本来要求する運用上の構造グラフがないため、RCAEval adapterと同様、
  全case時系列からPCでmetric graphを学習する
- PCは`alpha=0.05`、stable skeleton discoveryを用いる
- graphの各nodeを親metricで線形回帰し、正常時の回帰残差に対する異常時残差の
  最大絶対z-score（RHT）で順位付けする
- RCAEval adapterと同じ`lookup_window=120`、`detect_window=10`、
  障害後300点時点のwindowを使用する。短いcaseでは存在する最後の点へ縮める

PC graphが正常・異常の双方を使う点は、AMBERの学習規則とは異なる。ただしこれは
RCAEval上で構造グラフを代替するための既存adapter仕様であり、diagnosticsへ明記する。

### RUN

- 参照: zmlin1998/RUN commit
  `e2c97d7b5796455a1acbd66ab10c5bbb88eaccc8` とRCAEval adapter
- DLinearのtrend/seasonal分解、metric attention、target metricごとのforecast学習、
  attentionからのGranger graph、相関によるcycle除去、sink-personalized PageRankを維持する
- 公式コードの固定CUDA、train batch 128/test batch 1を前提とするreshapeは、
  CPUを含む任意device・可変batchで動くよう修正する
- 公式の全系列75/25分割は、正常・異常が連結されたRCAEval caseでは障害後データを
  学習へ混入させる。そのため比較用adapterでは既知onsetを使い、標準化・DLinear学習を
  正常segmentだけへ限定する
- `seq_len=32`、hidden 128、moving average kernel 25、pretrain 1 epoch、
  forecast 1 epoch、Adam `lr=0.001`はRCAEval adapterに合わせる

RUNの正常限定学習は、データ漏洩を防ぐための重要な変更である。したがって論文では
「公式RUNそのもの」ではなく「公式architectureを用いたknown-onset RCAEval adapter」
と記述し、元論文の報告値との直接比較は行わない。

## 解釈上の注意

- CIRCAとRUNはmetric数に対する計算量が大きい。実行時間も研究結果の一部として保存する。
- 特にRUNはtarget metricごとにニューラルモデルを学習するため、375ケース一括実行は非常に重い。
- `workers`を増やすとケース並列になるが、1ケース内部のモデルを軽量化するわけではない。
  CIRCA/RUNはまず`workers=1`で1ケースの時間とメモリを確認する。
- 失敗時に入力列順やランダム順位へfallbackするとAC@kを人工的に押し上げるため、行わない。
- 4手法の結果が出るまで、性能についての結論は記録しない。

## 実行方法と出力先

追加依存を導入する。

```bash
venv/bin/python -m pip install -r requirements-baselines.txt
```

個別実行例:

```bash
./scripts/run_baselines.sh --epsilon-diagnosis 4
./scripts/run_baselines.sh --rcd 4
./scripts/run_baselines.sh --circa 1
./scripts/run_baselines.sh --run 1
```

4手法を順に実行する場合:

```bash
./scripts/run_baselines.sh --all 1
```

結果は次へ保存される。

- `results/baselines/rcaeval_re1/epsilon_diagnosis_known_onset/`
- `results/baselines/rcaeval_re1/rcd_localized_known_onset/`
- `results/baselines/rcaeval_re1/circa_pc_rht/`
- `results/baselines/rcaeval_re1/run_neural_granger_known_onset/`

## 結果取得後の次アクション

1. 各手法が375ケースを完了したか、失敗・空候補数、実行時間を確認する。
2. dataset別AC/Avgだけでなく、fault type別とcase-level paired差を集計する。
3. PC/RUNが計算不能な場合は、結果を見てから事前に定義したsubset評価へ切り替える。
   スコアを見ながらfeature screening規則を追加して正式結果と混在させない。
4. 修論では公式commit、adapter変更、既知onset、window、service変換を表で明記する。
