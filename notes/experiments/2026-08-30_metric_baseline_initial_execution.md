# metric-based RCA baseline初回実行と安定化

## 目的

ε-Diagnosis、RCD、CIRCA、RUNがRCAEval RE1 Zenodo v2の375ケースで
比較可能か、精度だけでなく警告・失敗・実行時間を確認する。

## 観測事実

### ε-Diagnosis

4 workersで実行中、NumPy `corrcoef`から`invalid value encountered in divide`が
多数表示された。これは非定数metricから作ったbootstrap標本が偶然定数になり、
標準偏差0で相関がNaNになる場合である。実装は直後にNaNを0へ変換していたため、
保存済みスコアの定義は変わらない。`np.errstate`でこの既知の警告だけを抑制した。

### RCD

375ケースを完了した。

| Dataset | AC@1 | AC@3 | AC@5 | Avg@5 |
|---|---:|---:|---:|---:|
| re1_ob | 0.504 | 0.680 | 0.688 | 0.6400 |
| re1_ss | 0.208 | 0.312 | 0.320 | 0.2848 |
| re1_tt | 0.072 | 0.104 | 0.112 | 0.0992 |

RCDはOpenBenchmarkでは一定の性能を示す一方、Sock ShopとTrain Ticketで大きく低下した。
ただしこの時点では結果の改変やparameter tuningをせず、正式baseline結果として保持する。

### CIRCA

RE1-OB 125ケースは完了したが、最初のRE1-SS caseでFisher-Z CI testが
`Data correlation matrix is singular`として停止した。当該caseは275 metrics中rank 237で、
完全相関metric pairが多数存在した。また重複除去後のfull PCも計算量が大きく、
そのままでは正式375ケースに使用できなかった。

対策として、以下を事前固定した一般規則として実装した。

1. 絶対相関0.999999999999以上の後続metricをPC入力から除外する。
2. serviceごとのround-robinでPC入力を最大60 metricsにする。
3. PC-stableの条件集合次数を最大1にする。
4. PC入力外metricも削除せず、親なしRHTでスコアを計算する。
5. GT、fault type、異常スコアはfeature選択に使わない。

方式が変わったため、旧`circa_pc_rht`の125件と混在させず、結果名を
`circa_stratified_adaptive_pc_rht`へ変更した。60 metrics以下ではfull PCを維持し、
screeningが必要なcaseだけbounded PC(1)へ切り替える。RE1-SSの再現失敗caseは約5.7秒で完了した。
代表RE1-TT caseは802 metricsを60 metricsのPC入力へ縮約し、約33.7秒で完了した。

### RUN

エラー・警告はなかったが、RE1-OBで1ケース約20--27秒を要した。これはtarget metricごとに
DLinearを学習するRUNのモデル構造による比重が大きく、単純な例外ではない。7ケースで中断した。

## 再開方法

baseline scriptは既存case JSONをskipする`--resume`を内部で使うようにした。
したがってε-DiagnosisとRUNは同じコマンドで未完了caseから再開できる。

```bash
./scripts/run_baselines.sh --epsilon-diagnosis 4
./scripts/run_baselines.sh --run 4
```

CIRCAは方式名を変更したため、修正版375ケースを新しい出力先へ最初から実行する。

```bash
./scripts/run_baselines.sh --circa 4
```

RUNの`workers`はcase並列数である。まず2 workersでメモリを確認し、余裕があれば4へ増やす。
並列化しても1ケース内部の計算量は変わらない。
