# RUNの定義を維持した実行高速化

## 目的・固定条件

RUNの正式375ケースが遅いため、学習条件を削らず、同じ計算の実装上の無駄を減らす。
正常限定学習、全入力metric、targetごとの独立モデル、seed + target、batch順、
seq_len=32、hidden=128、pretrain=1、forecast=1、Adam lr=0.001、CPUを固定する。
AMBERおよび他baselineのスコア計算は変更しない。

## 採用した修正

### 1. 独立channel層のまとめ計算

入力metricごとのLinear層を、channel軸を持つweight・biasのbankへ格納する。
各channelの重みを共有するのではなく、元の独立したパラメータをそのまま保持する。
参照版の初期化結果をコピーするため、初期化の乱数消費順序も維持する。
Linearのbias加算を含めた計算には`torch.baddbmm`を用いる。

参照した一次資料:
[PyTorch baddbmm](https://docs.pytorch.org/docs/2.12/generated/torch.baddbmm.html)、
[PyTorch Adam](https://docs.pytorch.org/docs/2.12/generated/torch.optim.Adam.html)。
同じbank内のパラメータは全て同じ学習stepを受けるため、要素ごとのAdam状態・更新を
bankへまとめても数学上の更新則は変わらない。CPU float32での完全bit一致は一般には保証しない。

### 2. 前処理結果の重複計算と不要メモリの削減

pretrainingのprojected/unprojected viewは、同じencoder・入力を使い、dropout等もない。
encoder出力を1回だけ計算して再利用し、teacher側はdetachする。
forecastで参照されないprojectorとそのAdam状態はpretraining後に解放する。
encoderのAdam状態はリセットしない。

### 3. 循環除去の同値な高速化

元の実装は、グラフがDAGになるまで、全残存辺の相関を毎回計算し直し、
最小相関の辺を1本削除していた。相関は辺の削除で変わらないため、これは冗長である。

新実装では、元の辺順序を保持したstable sortで相関の昇順に辺を並べる。
元実装のargminが同点で最初の辺を選ぶことと一致する。
この順序の先頭からk本削除したグラフがDAGか、という条件はkについて単調なので、
最初にDAGになるkを二分探索し、そのprefixを削除する。
相関計算は各辺1回、DAG判定は概ねlog2(E)回になる（Eは元の辺数）。

**循環に含まれる辺だけを削る新しい規則には変えていない。**
元のグラフ推定・PageRank・service集約方法を保つためである。

## 採用しなかった修正

- 比較学習の類似度行列のうち未使用の3ブロックを計算しない案。
  数学上は同じだが、TTのfloat32 GEMMの計算順序が変わり、学習後attentionに
  最大約1.6e-5、親候補の列挙順に差が出たため不採用。元の行列形を保持した。
- 自動的なGPU/MPS切替。TTの1 targetで試したが、この環境ではCPU約3.92秒に対し
  MPS約7.45秒と遅く、attention差も約5.6e-5だった。CPU設定を維持する。
- epoch削減、hidden縮小、feature screening、target間のモデル共有。
  これらは比較手法の定義・学習条件を変えるので行わない。

## 設定と互換性

- `configs/baselines/run.yaml`: `execution_backend: vectorized`を明示。
  `torch_num_threads: 1`は従来の実行上の既定値を明示しただけである。
- `execution_backend: reference`で元のPython層ループを使える。
- 正式結果の保存先は従来と同じ。新規diagnosticsにはbackendとstage別時間を残す。
- 既存の正式case JSONは変更しない。起動済みworkerへ修正は反映されない。
- CPU float32の端数差は残る。代表確認だけで375件全ての順位一致を保証しない。
- 時間比較を統一する場合は、元の結果をskipするresumeではなく全件再実行する。
  スコア比較と実行時間比較を混同しない。

## 残る制約・別途確認すべき事項

targetごとに全入力metricを扱う独立モデルを学習するため、TTの学習量はなお大きい。
高速化したことを「375件が短時間で終わる」と解釈しない。
代表OBケースでは、参照版・高速版の両方で循環除去後の辺が0本だった。
これは今回の高速化が生んだ差ではなく、元の親選択・循環除去規則の検証課題である。
正式な精度を解釈する際は、空グラフ率・同点順位を別途確認する必要がある。
