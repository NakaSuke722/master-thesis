# Adaptive Direct AR-BF rollback ablation protocol

作成日: 2026-08-24

## 目的

Adaptive intervention-response Direct AR-BFはRCAEval 375ケースでmacro `AC@1=0.6880`、`AC@3=0.8800`、`AC@5=0.9520`、`Avg@5=0.8576`を得た。Full Direct AR-BFとIntercept-shift AR-BFは大きく上回ったが、応答形状、onset周辺化、normal-only pseudo-fault校正、per-row正規化を同時に変更したため、改善の寄与要因は未分離である。

本実験では統合candidateを固定し、構成要素を1つずつ元に戻すrollback ablationによって局所的な寄与を測る。AMBER本体のスコアリングロジックは変更しない。

## 正式条件

- RCAEval RE1 Zenodo record `14590730` v2
- `re1_ob` / `re1_ss` / `re1_tt`、各125ケース、合計375ケース
- normal / abnormal windowはそれぞれ最大600点
- service-level、aggregationは `mean_top3`
- AR(3)、strong AR prior、winsorizationなし
- 上記以外の条件は統合candidateと同一

## Rollback variants

| Variant | 統合candidateから戻す軸 | 固定する条件 |
|---|---|---|
| `adaptive_direct_no_null_calibration` | pseudo-fault baselineの差し引きを外す | per-row正規化は維持 |
| `adaptive_direct_fixed_onset` | onset周辺化を外す | onsetをfault boundary直後の`0`に固定 |
| `adaptive_direct_step_only` | 複数応答形状を外す | `step`だけを使う |
| `adaptive_direct_no_step_ramp` | 2係数modelだけを外す | 4つの単一係数形状は維持 |
| `adaptive_direct_no_per_row_normalization` | BFの行数正規化を外す | 生log BF同士のpseudo-fault校正は維持 |

`no_null_calibration`は `ar_null_calibration_fractions=[]` と `per_row_excess` の組合せである。これにより校正baselineは0になるが、観測log BFをconditional-AR行数で割る処理は残る。

`no_per_row_normalization`は `ar_null_calibration_mode=subtract` とし、観測とpseudo-faultの生log BFを直接差し引く。これによりpseudo-fault校正自体は残る。

## 実行

5 variantの一括実行:

```bash
./scripts/run_ablation.sh --adaptive-direct-rollback
```

統合candidateとのcase-level対応比較:

```bash
PYTHONPATH=src:. venv/bin/python \
  scripts/analyze_adaptive_direct_rollback.py
```

実験結果は次に保存される。

- 各variant: `results/ablation/rcaeval_re1/<variant>/`
- 対応分析: `results/analysis/adaptive_direct_rollback/`
- 分析生成物: `case_ranks.csv`、`summary.json`、`summary.md`

## 差の符号と解釈

分析スクリプトは、

\[
\Delta = \text{rollback} - \text{full adaptive}
\]

と定義する。

- `delta < 0`: 外した要素が統合candidateの性能に寄与した
- `delta = 0`: その評価指標で影響を確認できない
- `delta > 0`: 外した要素が性能を悪化させていた可能性がある

全体、dataset別、fault type別にAC@1/3/5とAvg@5を対応比較し、paired bootstrap 95%区間とAC@1のexact McNemar検定を出力する。

## 解釈上の制約

この実験が測るのは、統合candidate周辺で各1要素を外したときの局所的寄与である。要素間にinteractionがある場合、各差を足し合わせて統合candidate全体の改善量を説明することはできない。

## 現在の状態

設定、実行入口、対応分析の実装まで完了。正式375ケースは未実行である。
