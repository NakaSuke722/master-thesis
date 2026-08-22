# RCAEval RE1 Zenodo v2 正式アブレーション（service-level）

実施日: 2026-08-22

## 事実

同一のZenodo v2 RE1全375ケース、最大600点window、service-level、`mean_top3`、同一AR次数・ridge・NIG priorで比較した。変更した軸は residualization（AR / Raw）と scoring（Bayes Factor / GLRT）のみである。

| Method | AC@1 macro | AC@3 macro | AC@5 macro | Avg@5 macro |
|---|---:|---:|---:|---:|
| Full AMBER (AR+BF) | 0.6133 | 0.8587 | 0.9227 | 0.8187 |
| no_ar (Raw+BF) | **0.6400** | 0.8480 | **0.9280** | **0.8187** |
| no_bayes (AR+GLRT) | 0.4667 | 0.7733 | 0.8507 | 0.7131 |
| no_ar_no_bayes (Raw+GLRT) | 0.5120 | 0.7707 | 0.8400 | 0.7184 |

Dataset別の値（AC@1 / AC@3 / AC@5 / Avg@5）は以下である。

| Method | re1_ob | re1_ss | re1_tt |
|---|---|---|---|
| AR+BF | .448/.856/.944/.7808 | .848/.936/.992/.9312 | .544/.784/.832/.7440 |
| Raw+BF | .496/.848/.944/.7888 | .832/.952/1.000/.9376 | .592/.744/.840/.7296 |
| AR+GLRT | .280/.792/.936/.6800 | .688/.928/.984/.8864 | .432/.600/.632/.5728 |
| Raw+GLRT | .296/.792/.936/.6928 | .680/.920/.968/.8656 | .560/.600/.616/.5968 |

## 解釈

BFをGLRTへ置換すると、AR・Rawのいずれでもmacro AC@1とAvg@5が大きく下がる。したがって、この実験範囲ではNIG marginal likelihoodによるBayes FactorはAMBERの中核的寄与として強く支持される。

一方、observed-lag ARはRaw+BFに対しmacro AC@1で劣る（0.6133対0.6400）。ただしAC@3はAR+BFがわずかに高く（0.8587対0.8480）、Avg@5 macroは同じ0.8187である。ARはroot serviceを候補集合から除くより、上位候補内の順位を変えている可能性がある。このため「AR自体が無価値」とは結論しない。現時点で支持されないのは、現行のobserved-lag AR residualizationをそのまま最終方式として採用することである。

持続的なlevel shiftを `X_t^F = X_t^0 + Delta` とすると、異常期間の観測値をlagへ投入する現行AR(P)はshiftを予測側へ吸収しうる。その場合の残差は概ね `r_t ≈ (1 - sum phi_p) Delta` となり、root-cause signalが減衰する可能性がある。これは仮説であり、この集計結果だけからの因果的結論ではない。

## 次アクション

Raw+BFとAR+BFをcase_idで厳密に対応付け、各caseのroot service順位を比較する。`delta = raw_rank - ar_rank` とし、正ならARで順位改善、負なら悪化と定義する。全体・dataset別・fault type別にmovementとTop-1遷移を集計し、ARが改善／悪化する障害条件を特定する。

実装: `scripts/analyze_ar_rank_movement.py`。完全順位は結果artifactの`predicted_ranking`、または既存AMBER artifactの`amber_diagnostics.services`からのみ取得し、`predicted_top_5`だけのartifactから順位を推測しない。

### Persistent shift吸収の直接診断

`scripts/analyze_ar_signal_attenuation.py`を追加した。既存のRaw+BFとAR+BF artifactをcase IDで結合し、root metricごとにRaw中央値shift、shiftの前半・後半持続性、AR係数和、AR残差に残ったshift、初期AR-order windowから後期への残差減衰を測定する。さらに、`mean_top3`に実際に関与するroot metric群をcase-levelへ集約し、root-service順位差とのPearson・Spearman相関を出力する。

主要な検証量は `signal_retention_ratio = |AR residual median shift| / |Raw median shift|` である。1未満は残差信号の減衰を表す。ただし、この診断は関連を測るものであり、減衰が順位変化を引き起こしたという因果関係を単独で証明するものではない。

#### 診断結果

375ケース、root metric 7,464本を処理した。このうちRawまたはARのservice-level `mean_top3`に関与するunionは1,277本であり、case-level集計はこのunion内の中央値を使用した。

| ARによる順位移動 | Cases | median sum(phi) | median signal retention |
|---|---:|---:|---:|
| improved | 44 | 0.4683 | 0.5387 |
| same | 274 | 0.1328 | 0.8058 |
| worsened | 57 | **0.9400** | **0.0607** |

`rank_delta`（正ならARで改善）とのcase-level Spearman相関は、`sum_phi`が-0.2491、`signal_retention_ratio`が+0.2568だった。すなわち、AR係数和が大きいcaseほど順位が悪化し、Raw shiftがAR残差に多く保持されるcaseほど順位が改善する方向であり、persistent shift吸収仮説の予測と整合する。

一方、異常期間の前半から後半への残差減衰率と順位差のSpearman相関は-0.0316で、ほぼ関係がなかった。observed-lag ARによる吸収はAR次数ぶんの短い初期区間で完了しうること、障害影響の観測開始がinject timeから遅れることから、単純な前半・後半比較は主要な検証量として弱い。したがって現時点では、係数和とsignal retentionの結果は仮説を支持する記述的証拠だが、因果的な確証ではない。

成果物: `results/analysis/ar_signal_attenuation/`。
