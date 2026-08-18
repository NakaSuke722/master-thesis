# ベンチマーク戦略の見直し：BAROデータセットからRCAEvalへ

作成日: 2026-08-19

作成者: 中川 浩輔

---

# 1. 背景

## 1.1 初期評価環境

AMBERの初期実験では、BARO論文で使用されている
マイクロサービス障害データセットを利用して評価を行った。

対象システム：

- Online Boutique
- Sock Shop
- Train Ticket

対象障害：

- CPU負荷
- Memory負荷
- Delay
- Loss

である。

初期実験では、以下のようなBARO由来のデータ構造を利用した。

dataset/
 └ fault_type/
    └ run/
       ├ normal_data.csv
       ├ abnormal_data.csv
       └ graph.json


各ケースについて、

- 障害発生前の正常時メトリクス
- 障害発生後の異常時メトリクス
- 障害注入サービス
- 障害種類

を利用し、AMBERによるRoot Cause Analysis (RCA)を実施した。

---

# 2. BAROデータセットでの予備実験

## 2.1 AMBERの評価方法

AMBERでは、各監視メトリクスについてBayesian model comparisonに基づくスコアを計算する。

具体的には、

- metric-level scoreを計算
- 同一サービスに属するmetricを集約
- service-level rankingを生成

という流れでroot cause serviceを推定した。

---

## 2.2 得られた結果

予備実験では、特にSock Shopにおいて高いRCA性能が得られた。

例：

- AC@1 約0.9
- AC@3 約1.0
- AC@5 約1.0

など、非常に高い性能を確認した。

一方で、この結果がAMBERの本質的な性能を示しているのか、
それともベンチマーク自体の性質によるものなのかを検討する必要が生じた。

---

# 3. metric-level RCA評価に関する問題点

## 3.1 metric-level ground truthの不足

当初、AMBERではservice-level RCAだけでなく、
metric-level RCAについても評価を行った。

metric-level RCAでは、

例：
- cartservice_cpu
- payment_latency
- frontend_error


のような個別メトリクスをroot cause候補として順位付けする。

しかし、正確な評価には、

「本当にroot causeであるmetricはどれか」

というground truthが必要である。

BAROデータセットでは、
- 障害注入サービス
- 障害種類


は与えられる。しかし、root cause metric

が明示的にannotationされているわけではない。

---

## 3.2 障害注入箇所と観測metricは一致しない可能性がある

マイクロサービス障害では、障害原因と観測される異常は異なる。

例：
cartservice CPU障害
    ↓
checkoutservice latency増加
    ↓
frontend latency増加
の場合、

実際に障害を注入した箇所は、

cartservice_cpu

である。

しかし観測上最も大きく変化するmetricは、

frontend_latency

かもしれない。
したがって、

「最も異常度が高いmetric」
=
「root cause metric」

とは限らない。

このため、metric-level RCAを正式評価するには、
明示的なroot cause metric annotationを持つデータセットが必要である。

---

# 4. ベンチマーク変更の検討

## 4.1 RCAEvalの調査

既存研究で広く利用されているRCA benchmarkとして、
RCAEvalを調査した。

RCAEvalでは、

- Online Boutique
- Sock Shop
- Train Ticket

を含む複数のマイクロサービスシステムについて、
多数の障害ケースが整理されている。

また、

root_cause_service
fault
inject_time

などのメタデータが提供される。

これにより、

「障害注入サービスを正解としたservice-level RCA」

を明確なground truthのもとで評価できる。

---

## 4.2 RCAEvalを主benchmarkへ採用する理由

BAROデータセットからRCAEvalへ移行する理由は、
metric-level annotationの改善ではない。

主な目的は以下である。

### 1. 評価の標準化

既存RCA研究と比較可能なbenchmarkを利用する。

### 2. Ground truthの明確化

service-level RCAについて、

root_cause_service

を正式な正解ラベルとして利用できる。

### 3. 再現性向上

dataset構造やmetadataが整理されており、
実験条件を明確に記述できる。

---

# 5. 今後の研究評価方針

今後は、評価粒度ごとに利用するbenchmarkを分離する。

---

## 5.1 Main Experiment：Service-level RCA

Benchmark:

RCAEval RE1


目的：

AMBERがroot cause serviceを正しく特定できるか評価する。

評価対象：

root_cause_service


評価指標：

- AC@1
- AC@3
- AC@5
- Avg@K

---

## 5.2 Fine-grained Experiment：Metric-level RCA

Metric-level RCAについては、
RCAEvalではなく、root cause metricを明示的に持つ
benchmarkを利用する。

候補：

- CausalRCA
- LatentScope Dataset

目的：

AMBERがmetric-level root cause localizationにも利用可能か検証する。

---

## 5.3 Hard Benchmark Evaluation

さらに、既存benchmarkが容易すぎる可能性を検証するため、
fault propagationを考慮した新しいbenchmarkについても調査する。

目的：

単純な異常度ランキングでは困難な状況でも、
AMBERが有効であるかを検証する。

候補：

- FSE 2026 Fault-Propagation-Aware Benchmark

---

# 6. 最終的な研究設計

AMBERの評価構造を以下へ変更する。

             AMBER

               |
               |
    Bayesian metric scoring

               |
    -------------------------
    |                       |
service aggregation       metric ranking
    |                       |
RCAEval RE1              CausalRCA等
 service RCA              metric RCA


---

# 7. 移行計画

## Phase 0：BARO実験の凍結

現在までのBAROデータセットによる実験結果を保存する。

目的：

- 予備実験結果として保持
- RCAEval移行後との比較
- 再現可能性確保

Git tag:
v0.1-baro-pilot


---

## Phase 1：Benchmark abstraction layerの導入

現在のコードは、
dataset/fault/run

というBARO固有形式に依存している。

今後は、

benchmark/case


という形式へ変更する。

各benchmarkについて、

- case ID
- root cause service
- fault type
- injection time
- data path

を共通形式で管理する。

---

## Phase 2：RCAEval RE1対応

実装項目：

- RCAEval loader作成
- metrics.parquet読み込み
- normal/abnormal split
- case_info.json生成
- service-level evaluation対応

---

## Phase 3：追加benchmark対応

RCAEval対応後、

- FSE 2026 benchmark
- CausalRCA

への対応を進める。

---

# 8. 研究上の意義

初期実験では、

「AMBERが障害注入箇所を当てられるか」

を主に評価していた。

しかし、benchmarkの性質やground truthの問題を検討した結果、
今後は、

「標準化されたRCA benchmark上で、
AMBERがroot cause serviceを特定できるか」

を主研究課題とする。

また、metric-level RCAについては、
適切なground truthを持つbenchmarkを利用することで、
より厳密に評価する。

この変更により、

- 評価の公平性
- 再現性
- 既存研究との比較可能性

を向上させる。v
