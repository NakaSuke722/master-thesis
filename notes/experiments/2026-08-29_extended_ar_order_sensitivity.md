# Counterfactual AR次数の拡張感度分析

実施日: 2026-08-29

## 目的

2026-08-28の感度分析ではAR(1)/AR(3)/AR(5)だけを比較し、簡潔性、実行時間、normal-only pseudo-faultの一部指標を理由にAR(3)を最終候補とした。しかし、正式RCAEvalスコアではAR(5)がAR(3)を上回る指標があり、AR(3)を積極的に選ぶ根拠は弱かった。

そこで、他の構成要素を固定したままAR次数だけを7/9/11/13へ拡張し、正式375ケースの感度とAR(7)の正常時挙動を確認した。

## 共通条件

- RCAEval RE1 Zenodo v2、375ケース
- normal/abnormalとも最大600点
- service-level、`mean_top3`
- normal-only metric standardization
- Stationary Counterfactual AR、root projection半径0.98
- clipなし
- diagonal horizon-aware uncertainty
- NIG Bayes Factor score
- 比較するのは `ar_order`だけ

## 正式375ケースの事実

| AR次数 | macro AC@1 | AC@3 | AC@5 | Avg@5 | wall time (s) | pure Python (s) |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | .7227 | .9280 | **.9680** | .8880 | 108 | 416.06 |
| 3 | .7440 | .9280 | **.9680** | .8891 | **94** | **364.66** |
| 5 | .7493 | .9333 | .9653 | .8928 | 126 | 485.98 |
| **7** | .7493 | .9333 | .9653 | **.8939** | 103 | 398.01 |
| 9 | .7493 | **.9360** | .9600 | .8933 | 108 | 416.60 |
| 11 | **.7520** | .9280 | .9600 | .8917 | 117 | 453.42 |
| 13 | .7440 | .9280 | .9627 | .8917 | 125 | 480.33 |

単一の次数がすべての指標で最良ではない。AC@1はAR(11)、AC@3はAR(9)、AC@5はAR(1)/AR(3)、Avg@5はAR(7)が最高だった。AR(7)はAR(5)とmacro AC@1/3/5が同値で、Avg@5だけを.0011上回った。AR(13)はAR(7)に全macro指標で劣り、これ以上の高次化を追加探索する根拠はない。

## dataset別の事実

AC@1は次数に対してシステム別に逆方向へ動いた。

| AR次数 | re1_ob | re1_ss | re1_tt |
|---:|---:|---:|---:|
| 3 | .752 | .840 | **.640** |
| 5 | .768 | .840 | **.640** |
| 7 | .800 | .848 | .600 |
| 9 | .808 | .840 | .600 |
| 11 | .816 | .848 | .592 |
| 13 | **.824** | .840 | .568 |

- Online Boutiqueは高次化にともない一貫して改善した。
- Sock Shopはほぼ不変だった。
- Train TicketはAR(5)を超えると一貫して悪化した。

これは、単一の「真の最適次数」よりも、システムごとに時間依存長や高次ARの影響が異なる可能性を示す。Online Boutiqueではより長い自己依存を拾う一方、metric数とservice数の多いTrain Ticketでは余分な依存や再帰予測誤差を拾う可能性があるが、現時点では仮説である。

## fault type別AC@1

| AR次数 | CPU | MEM | DISK | DELAY | LOSS |
|---:|---:|---:|---:|---:|---:|
| 3 | .773 | **.960** | .880 | .613 | .493 |
| 5 | **.787** | **.960** | .880 | .653 | .467 |
| 7 | .747 | .933 | .880 | .693 | .493 |
| 9 | .747 | .920 | .880 | .693 | **.507** |
| 11 | .747 | .947 | .880 | **.720** | .467 |
| 13 | .747 | .893 | .880 | **.720** | .480 |

高次化はDELAYを改善する一方、CPU/MEMを悪化させる傾向がある。DISKは全次数で同じである。fault type既知で次数を切り替えると事後適合になるため、そのようなルールは導入しない。

## case-level監査

保存済みfull rankingを`case_id`で突合した。

| 比較 | AR(7)で改善 | 同一 | AR(7)で悪化 | Top-1獲得/喪失 | exact McNemar |
|---|---:|---:|---:|---:|---:|
| AR(7) vs AR(3) | 18 | 341 | 16 | 10/8 | .8145 |
| AR(7) vs AR(5) | 13 | 351 | 11 | 8/8 | 1.0000 |
| AR(7) vs AR(9) | 12 | 353 | 10 | 4/4 | 1.0000 |
| AR(7) vs AR(11) | 23 | 334 | 18 | 10/11 | 1.0000 |
| AR(7) vs AR(13) | 28 | 325 | 22 | 15/13 | .8506 |

AR(7)の差は少数caseに限られ、Top-1の統計的な優越は確認できない。この結果はAR(7)の最適性を証明するものではない。

## AR(7) normal-only pseudo-fault

正常windowの前半50%でARを推定し、後半50%を障害のないpseudo-abnormalとして評価した。表のBFは実装上はservice集約後のlog BF scoreであり、校正済み検出閾値ではない。

| 次数 | Median max log BF | P90 max log BF | Median service log BF | Positive service fraction |
|---:|---:|---:|---:|---:|
| 3 | 582.76 | 1083.88 | 156.08 | 89.65% |
| 5 | 585.66 | 1149.75 | **155.81** | 89.92% |
| 7 | 600.12 | 1221.99 | 157.19 | **89.00%** |

AR(7)はAR(3)比でmedian maxが約3.0%、P90 maxが約12.7%増加した。Positive service fractionは約0.64 percentage point低下したため、すべての正常時指標が悪化したわけではない。

P90 maxの増加は特にre1_ttで大きく、AR(3)の1349.67からAR(7)の1799.62へ約33.3%増加した。これは正式AC@1がre1_ttで.640から.600へ下がったことと方向が一致する。ただし、pseudo-fault log BFは絶対校正されておらず、この指標単独でAR(7)を棄却できない。

## 解釈

- AR(7)はmacro Avg@5が最高で、AC@1/3も最高群にあるため、RCAEval上の経験的総合候補である。
- 一方、その改善はOnline BoutiqueとDELAYに寄り、Train Ticketとnormal-only tailは悪化した。
- AR(11)のAC@1最高はAR(7)より全375ケースで実質1件分であり、その他のmacro指標を下げてまで採用する根拠にはならない。
- 次数を複数試した後にRCAEvalスコアで最高候補を選ぶことは、RE1へのhyperparameter tuningである。修論では探索的感度分析であることを明示する。
- AR次数を障害種別やdataset別に切り替える追加ロジックは、モデルを再度太らせるため導入しない。

## 成果物

- 正式感度分析: `results/sensitivity/rcaeval_re1/unit_invariant_r0_98_p{7,9,11,13}/`
- AR(7) pseudo-fault: `results/analysis/ar_final_sensitivity_pseudo_fault/unit_invariant_r0_98_p7/`
- 設定: `configs/sensitivity/rcaeval_re1_zenodo_v2/unit_invariant_r0_98_p{7,9,11,13}.yaml`

## 次アクション

1. AR次数の追加探索は13で打ち切り、AR(15)以上は試さない。
2. AR(7)をRCAEval上の経験的第一候補とするが、正式mainはまだAR(3)のまま保持する。
3. 最終次数を選ぶ場合は、normal-only blocked multi-horizon forecast validationでlabelを使わずに選ぶか、AR(7)を探索的に選択したと明記するかを決める。
4. 最終次数をfreezeした後にだけmain YAMLを更新し、375ケースを正式再実行する。
5. 外部比較はmetrics-only条件でRandom、epsilon-Diagnosis、BARO、RCD、CIRCA、RUNを優先する。Raw+BFやObserved-lag AR+BFは外部baselineと分けてinternal ablationとして掲載する。
