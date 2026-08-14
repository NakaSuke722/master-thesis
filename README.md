# AMBER: Bayesian Residual Change Detection for Root Cause Analysis

AMBER は、マイクロサービスの時系列メトリクスを対象とした根本原因分析（Root Cause Analysis; RCA）手法です。正常期間で学習した自己回帰（AR）モデルの予測残差を用い、障害前後で残差分布が変化したかをベイズモデル比較によって評価します。得られたスコアから、根本原因候補をメトリクス単位またはサービス単位で順位付けします。

## 手法の概要

AMBER の処理は次の流れです。

1. 既知の障害発生時刻より前を正常期間、以後を異常期間として分割する。
2. 各メトリクスについて、正常期間だけを使って AR モデルを学習する。
3. 正常モデルによる予測値と観測値との差から残差を計算し、正常残差を基準に標準化する。
4. 障害前後で同一の残差分布を仮定するモデルと、異なる分布を仮定するモデルの周辺尤度を比較する。
5. 対数 Bayes Factor を異常スコアとして、メトリクスまたはサービスを順位付けする。

本研究では障害発生時刻は既知であると仮定します。障害検知や変化点推定そのものは対象外です。

## 対応データセット

現在の実験構成では、次のマイクロサービス・データセットを対象とします。

- Online Boutique
- Sock Shop
- Train Ticket

データセット固有のファイル名、障害時刻、正解ラベルなどは、各設定ファイルおよびデータ読み込み処理を参照してください。

## ディレクトリ構成

```text
master-thesis/
├── configs/
│   ├── amber.yaml                 # AMBER の基本設定
│   ├── ablation/                  # アブレーション実験の設定
│   ├── baselines/                 # 比較手法の設定
│   └── sensitivity/               # 感度分析の設定
├── data/
│   ├── raw/                       # 元データ（Git 管理対象外）
│   └── processed/                 # 前処理済みデータ（Git 管理対象外）
├── src/
│   ├── models/
│   │   ├── amber.py               # AMBER の実装
│   │   ├── ablations/             # AMBER の変種
│   │   └── baselines/             # 比較手法
│   └── evaluation/                # 評価指標・集計処理
├── scripts/
│   ├── run_main.sh                # AMBER の主要実験
│   ├── run_ablation.sh            # アブレーション実験
│   ├── run_baselines.sh           # 比較手法の実験
│   ├── run_sensitivity.sh         # 感度分析
│   └── make_figures.sh            # 図の生成
├── results/
│   ├── main/amber/                # AMBER の主要結果
│   ├── ablation/                  # アブレーション結果
│   ├── baselines/                 # 比較手法の結果
│   ├── sensitivity/               # 感度分析結果
│   ├── analysis/                  # 条件別分析・失敗事例
│   ├── figures/                   # 論文・発表用の図
│   └── debug/                     # デバッグ出力
├── tests/                         # テスト
├── notes/                         # 研究メモ
├── paper/                         # 修士論文・発表資料
├── README.md
└── requirements.txt
```

`results/metrics/` と `results/final_summary.json` は旧構成です。新しい実験では、後述する `results/main/`、`results/ablation/`、`results/baselines/`、`results/sensitivity/` を使用します。

## セットアップ

Python 3.10 以降を推奨します。リポジトリのルートで仮想環境を作成し、依存パッケージをインストールしてください。

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Windows PowerShell では、仮想環境の有効化に次を使用します。

```powershell
venv\Scripts\Activate.ps1
```

## データの配置

元データは `data/raw/` に配置し、必要に応じて前処理済みデータを `data/processed/` に生成します。大容量データは `.gitignore` により Git 管理から除外します。

実験前に、設定ファイルが参照するデータパスと、障害発生時刻・対象データセット・評価粒度が正しいことを確認してください。

## 設定

AMBER の標準設定は `configs/amber.yaml` に集約します。実験条件を変更する場合は、コード内の既定値を直接変更せず、目的別の YAML ファイルを編集してください。

- `configs/amber.yaml`: 主要実験
- `configs/ablation/`: AR 処理やベイズ比較などの構成要素を除いた実験
- `configs/baselines/`: 比較手法
- `configs/sensitivity/`: AR 次数、事前分布、分析窓などの感度分析

再現性のため、結果とともに使用した設定ファイル、データセット名、評価粒度、乱数シードを記録してください。

## 実験の実行

以下のコマンドはリポジトリのルートで実行します。

### AMBER の主要実験

```bash
bash scripts/run_main.sh
```

このスクリプトでは `configs/amber.yaml` を基準に、対象データセットと評価粒度（`metric` / `service`）ごとの AMBER 実験を実行します。個別条件だけを実行したい場合は、`scripts/run_main.sh` 内の実行例と、実行プログラムの `--help` を確認してください。

### アブレーション実験

```bash
bash scripts/run_ablation.sh
```

設定は `configs/ablation/` に置き、結果は `results/ablation/<variant>/<level>/` に保存します。代表的な変種は `no_ar` と `no_bayes` です。

### 比較手法

```bash
bash scripts/run_baselines.sh
```

比較手法ごとの設定は `configs/baselines/`、結果は `results/baselines/` に保存します。

### 感度分析

```bash
bash scripts/run_sensitivity.sh
```

感度分析の設定は `configs/sensitivity/` に置きます。主な分析軸は次のとおりです。

- `ar_order`: AR 次数
- `prior`: ベイズモデルの事前分布
- `window`: 正常・異常期間の分析窓

### 図の生成

```bash
bash scripts/make_figures.sh
```

集計済みの実験結果から、主要結果、アブレーション、比較、感度分析、追加分析の図を `results/figures/` 以下に生成します。

## 結果の格納先

```text
results/
├── main/amber/
│   ├── metric/                    # メトリクス単位の AMBER 結果
│   └── service/                   # サービス単位の AMBER 結果
├── ablation/
│   ├── no_ar/{metric,service}/
│   └── no_bayes/{metric,service}/
├── baselines/                     # 比較手法の結果
├── sensitivity/
│   ├── ar_order/
│   ├── prior/
│   └── window/
├── analysis/
│   ├── by_fault_type/             # 障害種別の分析
│   ├── by_dataset/                # データセット別の分析
│   └── failure_cases/             # 失敗事例
├── figures/
│   ├── main/
│   ├── ablation/
│   ├── comparison/
│   ├── sensitivity/
│   └── analysis/
└── debug/
```

`metric` は個々のメトリクスを、`service` は同一サービスに属する複数メトリクスを集約した順位付けを表します。実験出力の JSON は原則として再生成可能な成果物として扱い、論文で使用する確定済みの図だけを必要に応じて Git 管理します。

## 推奨する実験フロー

1. `data/raw/` に元データを配置する。
2. `configs/amber.yaml` のデータパスと実験条件を確認する。
3. `scripts/run_main.sh` で主要実験を実行する。
4. `scripts/run_baselines.sh` で比較手法を実行する。
5. `scripts/run_ablation.sh` と `scripts/run_sensitivity.sh` で手法の構成要素とハイパーパラメータを検証する。
6. `scripts/make_figures.sh` で結果を可視化する。
7. `results/analysis/` でデータセット別・障害種別の傾向と失敗事例を整理する。

## 開発時の注意

- 学習や標準化に異常期間の情報を混入させないでください。
- 実験条件は YAML に残し、結果ファイル名またはメタデータから追跡できるようにしてください。
- `src/models/amber.py` を AMBER の正式な実装名とし、旧名 `bayesian_residual_rca` を新しいコードや文書で使用しないでください。
- 大容量データ、実験途中の JSON、デバッグ出力はコミットしないでください。
- 複数条件を比較するときは、データ分割、評価粒度、Top-K、乱数シードを揃えてください。

## ライセンス

ライセンスを定める場合は、リポジトリ直下に `LICENSE` を追加し、本節を更新してください。
