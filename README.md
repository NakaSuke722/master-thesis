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

## まとめ

結論として、 `master-thesis` は、単なる「RCAモデルを実装して実験するコード置き場」から、
データ→実験設定→モデル→評価→実験結果→解析・図表→修論
という**研究プロセス全体を再現可能に管理するリポジトリ**へ変わりました。

特に重要なのは、**AMBER本体・アブレーション・ベースライン比較・感度分析を明確に分離したこと**です。これから実験が増えても、「この結果は何の実験だったのか」が迷子になりにくくなっています。

### 現在の全体像

以下、それぞれの役割を整理します。

#### 1. `configs/` ―― 「どんな条件で実験したか」

```text
configs/
├── amber.yaml
├── ablation/
├── sensitivity/
└── baselines/
```

ここは**実験条件をコードから分離する場所**です。

重要なのは、

> モデルの実装を変更して実験条件を変える

のではなく、

> YAMLを変更して実験条件を変える

という思想です。

#### `configs/amber.yaml`

AMBER本体の標準設定です。例えば概念的には、

```yaml
model:
  target: amber
  preprocess_strategy: default

evaluation:
  granularity: service
  k_values: [1, 3, 5]

datasets:
  - online_boutique
  - sock_shop
  - train_ticket
```

のような設定を保持します。つまり、AMBERの標準実験条件です。
将来的には論文で

> Unless otherwise specified, we used the configuration in `configs/amber.yaml`.

と言えるような位置付けになります。

---

#### `configs/ablation/`

**アブレーション実験**用です。例えば、

```text
configs/ablation/
├── no_ar.yaml
└── no_bayes.yaml
```

とします。目的は、「AMBERのどの構成要素が性能に寄与しているか？」を検証することです。
例えば、

| Method | AR | Bayesian model comparison |
|---|:---:|:---:|
| AMBER | ✓ | ✓ |
| w/o AR | × | ✓ |
| w/o Bayes | ✓ | × |

という比較を作るための設定です。

---

#### `configs/sensitivity/`

**感度分析**用です。

```text
configs/sensitivity/
├── ar_order.yaml
├── prior.yaml
└── window.yaml
```

アブレーションとの違いが重要です。アブレーションは、「ARあり vs. ARなし」のように**構成要素の必要性**を調べます。

感度分析は、p=1,2,3,5,10 のように、

> ハイパーパラメータを多少変えても結論が変わらないか？

を調べます。したがって、「robustnessの検証」に近い役割です。

---

#### `configs/baselines/`

BAROなどの**比較手法**の設定です。
目的は、「AMBERは既存手法に対して競争力があるか？」を調べること。
つまり、

- `ablation/` → **AMBER内部の分析**
- `baselines/` → **AMBERと他手法の比較**

です。

---

### 2. `data/` ―― 「何を観測したか」

ここは基本的に従来通りです。

```text
data/
├── raw/
└── processed/
```

#### `data/raw/`

BAROで公開されているOnline Boutique、Sock Shop、Train Ticketなどの**元データ**です。
ここは原則としてimmutable、つまり、研究コードから書き換えない方針にします。

---

#### `data/processed/`

AMBERへ入力できる形へ前処理したデータです。
例えば、

```text
data/processed/
└── default/
    └── sock_shop/
        └── catalogue_cpu/
            └── 1/
                ├── normal_data.csv
                ├── abnormal_data.csv
                └── graph.json
```

のような構造です。AMBERが実際に使う中心的な情報は、

\[
X^{(N)} = \{X(t):t<t_F\}
\]

と

\[
X^{(A)} = \{X(t):t\geq t_F\}
\]

です。

`graph.json` については、これまで決めた通り、**因果グラフ・コールグラフをAMBERのRCA推論に入力しない**という研究上の制約があります。

---

### 3. `src/` ―― 「研究手法そのもの」

ここが研究コードの中心です。

```text
src/
├── models/
│   ├── amber.py
│   ├── ablations/
│   └── baselines/
├── evaluation/
└── ...
```

#### `src/models/amber.py`

**提案手法AMBERそのもの**です。ここは非常に重要で、「修論の「提案手法」節と対応するコード」になります。

今後は**AMBER本体の理論を変更したときだけここを変更する**くらいの感覚がよいでしょう。

---

#### `src/models/ablations/`

AMBERの構成要素を除去・置換するための実装を置く場所です。
ただし、前にも触れた通り、

```text
amber_no_ar.py
amber_no_bayes.py
amber_no_xxx.py
```

を大量に作るのは避けたいです。
可能なら `amber.py` の共通処理を再利用し、

```python
use_ar=False
```

などで切り替える設計にします。
ここには、どうしても別実装が必要なものだけ置くのがよいです。

---

#### `src/models/baselines/`

BAROなどの**既存手法・比較手法**を実装する場所です。AMBERとは明確に分離します。

---

### 4. `src/evaluation/` ―― 「良いRCAとは何か」

ここはモデルではなく**評価方法**を担当します。
AMBERが、

```text
cartservice_cpu
paymentservice_cpu
...
```

というランキングを返したら、AC@1,\ AC@3,\ AC@5 や Avg@1,\ Avg@3,\ Avg@5 を計算するのはこちら側です。

非常に重要なのは、Model≠Evaluation として分離することです。

これによりBAROを実装しても、

```text
AMBER ranking
      ↓
同じevaluation

BARO ranking
      ↓
同じevaluation
```

とできます。つまり**公平な比較**になります。

---

### 5. `scripts/` ―― 「実験開始ボタン」

```text
scripts/
├── run_main.sh
├── run_ablation.sh
├── run_baselines.sh
├── run_sensitivity.sh
└── make_figures.sh
```

ここは研究者が実験するときの**インターフェース**です。
理想的には、毎回

```bash
python src/main.py --foo ... --bar ...
```

という長いコマンドを覚える必要をなくします。

#### `run_main.sh`

AMBERのMain Experiment。

```bash
./scripts/run_main.sh
```
↓
Online Boutique / Sock Shop / Train Ticket
↓
service / metric
↓
AC@K / Avg@K

という流れです。

---

#### `run_ablation.sh`

アブレーションを一括実行します。

最終的には、

```bash
./scripts/run_ablation.sh
``` 
だけで、

```text
AMBER
w/o AR
w/o Bayes
```
を全部回せると理想的です。

---

#### `run_sensitivity.sh`

例えば、

\[
p\in\{1,2,3,5,10\}
\]

を自動的に回します。

---

#### `make_figures.sh`

実験結果から修論用Figureを生成します。

したがって、

```text
run_main.sh
     ↓
JSON
     ↓
make_figures.sh
     ↓
PDF/PNG
```

という関係です。

---

### 6. `results/` ―― 今回最も大きく変わったところ

以前は、

```text
results/metrics/
```

を中心としていました。
これだと実験が増えたとき、

> これは本実験？
> アブレーション？
> prior変えたやつ？

が分からなくなります。
そこで**実験の目的別**に分けました。

```text
results/
├── main/
├── ablation/
├── baselines/
├── sensitivity/
├── analysis/
├── figures/
└── debug/
```

---

### 7. `results/main/amber/` ―― AMBERの正式結果

```text
results/main/
└── amber/
    ├── metric/
    └── service/
```

ここが、

\[
\boxed{\text{AMBERのMain Result}}
\]

です。

`.gitignore` により、

```text
*.json
```

はGit管理しません。

一方、

```text
.gitkeep
```

だけGit管理しています。

だからcloneした直後でも、

```text
results/main/amber/metric/
results/main/amber/service/
```

という構造は残ります。

---

### 8. `results/ablation/`

例えば、

```text
results/ablation/
├── no_ar/
└── no_bayes/
```

です。

ここには、

> AMBERから特定要素を除いたらどうなったか

という結果だけを置きます。Main Resultと混ぜません。

---

### 9. `results/baselines/`

BAROなどとの比較結果。

最終的には修論の、

**Comparison with Existing Methods**

に対応します。

---

### 10. `results/sensitivity/`

```text
results/sensitivity/
├── ar_order/
├── prior/
└── window/
```

例えば `ar_order/` なら、

\[
p=1,2,3,5,10
\]

について、

\[
p \mapsto Avg@K
\]

を保存します。

修論では、

> AMBERの性能が特定の \(p=3\) だけに依存しているわけではない

という主張の根拠になります。

---

### 11. `results/analysis/`

ここは**単純な性能評価より一段深い考察**です。

```text
results/analysis/
├── by_fault_type/
├── by_dataset/
└── failure_cases/
```

例えば、

\[
\mathrm{Avg@5}_{CPU}
\]

\[
\mathrm{Avg@5}_{memory}
\]

\[
\mathrm{Avg@5}_{delay}
\]

\[
\mathrm{Avg@5}_{loss}
\]

を比較できます。

そして、

> AMBERはCPU/memoryには強いが、delayでは下流サービスへの異常伝播との区別が難しい

などの考察につなげます。

`failure_cases/` は特に重要で、

> **なぜAMBERが間違えるのか？**

を分析する場所です。

---

### 12. `results/figures/`

```text
results/figures/
├── main/
├── ablation/
├── comparison/
├── sensitivity/
└── analysis/
```

ここは**論文に載せられる完成図**です。

例えば、

```text
main/overall_performance.pdf
ablation/ablation_avg5.pdf
sensitivity/ar_order.pdf
analysis/by_fault_type.pdf
```

などです。

生のJSONとFigureを分離したことで、

\[
\text{raw results}
\rightarrow
\text{analysis}
\rightarrow
\text{publication figure}
\]

という流れが明確になりました。

---

### 13. `results/debug/`

これは研究成果ではありません。

例えば、

- AR residualのプロット
- Bayes Factorの途中値
- 正常/異常分布の確認
- ACF
- 特定ケースのデバッグ

などを一時的に出します。

つまり、「研究中には必要だが論文には載せないもの」です。

---

### 14. `tests/`

ここは今後かなり重要になります。

AMBERの性能が高くなってくるほど、

> 「コードのバグで高くなってない？」

という疑いを潰す必要があります。

例えば、

```text
tests/
├── test_amber.py
├── test_evaluation.py
└── test_data_loader.py
```

として、

- 正常/異常分割が正しい
- ranking順が正しい
- Bayes Factor計算が解析解と一致
- service/metric変換が正しい
- ground truth leakageがない

などを検証できます。

---

### 15. `notes/`

これは**研究者としての自分向け**です。

例えば、

```text
notes/
├── research_log.md
├── amber_theory.md
├── experiment_notes.md
└── meeting_notes.md
```

など。

特に、

> 8/14：ARなし実験。Online Boutiqueで性能低下。
> 仮説：raw系列の自己相関によりiid仮定が崩れるため。

のような記録を残しておくと、数か月後にDiscussionを書くとき非常に助かります。

---

### 16. `paper/` ―― 研究成果を文章にする場所

```text
paper/
├── shared/
├── thesis/
└── slide/
```

#### `paper/thesis/`

修士論文本体。

現在の

```text
sections/
├── 01_intro.tex
├── 02_background.tex
├── 03_method.tex
├── 04_experiment.tex
...
```

という構成は非常に自然です。

```text
src/models/amber.py
        ↓
03_method.tex

results/main/
results/ablation/
results/baselines/
results/sensitivity/
        ↓
04_experiment.tex

results/analysis/
        ↓
05_discussion.tex
```

という**コードと論文の対応関係**ができました。

---

### 17. `paper/shared/`

例えば、

```text
ref.bib
```

など、修論と発表スライドで共有するものを置けます。

同じ文献情報を、

```text
thesis/ref.bib
slide/ref.bib
```

と二重管理するのを避けられます。

---

### 18. `paper/slide/`

研究室発表・学会発表・修論審査用のBeamerなどです。

つまり、

```text
paper/thesis/
```

が文章、

```text
paper/slide/
```

がプレゼンです。

---

### 19. `.gitignore` の役割も以前より重要になった

今回、

```gitignore
results/main/**/*.json
results/ablation/**/*.json
...
```

としたことで、

**実験結果そのものはGitHubへ大量にpushしない**設計になっています。

一方、

```text
.gitkeep
```

は追跡します。
つまりGitHubには、

```text
results/
└── main/
    └── amber/
        ├── metric/
        └── service/
```

という**設計図だけ残る**。

ローカルで実験すると、

```text
metric/
└── online_boutique/
    ├── cartservice_cpu_run1.json
    ├── ...
```

が生成されますが、Gitには載りません。

---

### 20. `requirements.txt`

これは、「この研究コードを動かすためのPython依存関係」です。

最終的には、

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

だけで別PCでも環境を再現できる状態を目指します。

---

### 21. `README.md`

READMEはこのリポジトリ全体の**取扱説明書**です。

最終的には第三者がREADMEだけ読んで、

```text
1. これは何の研究？
        ↓
2. AMBERとは？
        ↓
3. データはどこ？
        ↓
4. 環境構築は？
        ↓
5. Main experimentはどう実行する？
        ↓
6. Ablationは？
        ↓
7. 結果はどこ？
        ↓
8. 修論はどこ？
```

を理解できる状態にします。

---

### 全体を「研究の流れ」として見ると

今回の構造変更の本質はここです。

```text
                  ┌─────────────────┐
                  │      data/      │
                  │ 観測データ       │
                  └────────┬────────┘
                           ↓
                  ┌─────────────────┐
                  │    configs/     │
                  │ 実験条件         │
                  └────────┬────────┘
                           ↓
                  ┌─────────────────┐
                  │ src/models/     │
                  │ AMBER / baseline│
                  └────────┬────────┘
                           ↓
                  ┌─────────────────┐
                  │ src/evaluation/ │
                  │ AC@K / Avg@K    │
                  └────────┬────────┘
                           ↓
                  ┌─────────────────┐
                  │    results/     │
                  │ 実験結果         │
                  └────────┬────────┘
                           ↓
                ┌──────────┴──────────┐
                ↓                     ↓
       results/analysis/       results/figures/
          統計的解析              可視化
                └──────────┬──────────┘
                           ↓
                  ┌─────────────────┐
                  │     paper/      │
                  │ 修論・発表       │
                  └─────────────────┘
```

そして横から、

```text
scripts/
```

がこれらを実行し、

```text
tests/
```

が正しさを保証し、

```text
notes/
```

が研究過程を記録する、という構造です。