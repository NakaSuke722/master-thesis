# Decision: ARの採否前に順位移動を診断する

## 決定

RCAEval RE1 Zenodo v2正式アブレーションではBayes Factorを維持する。observed-lag ARは直ちに削除せず、Raw+BF対AR+BFのroot-service順位移動を診断してから再設計の要否を判断する。

## 根拠となる事実

macro AC@1はAR+BF 0.6133、Raw+BF 0.6400であり、現行ARはTop-1で優位でない。一方、macro AC@3はAR+BF 0.8587、Raw+BF 0.8480、Avg@5は両者0.8187である。Bayes FactorからGLRTへの置換は、いずれのresidualizationでも大幅な性能低下を伴った。

## 解釈上の制約

この結果はARというモデル化の発想全体を否定しない。異常観測値をlagに使う現行方式がpersistent shiftを吸収している可能性を検証する必要がある。

## 次の判定材料

case-levelで `delta = raw_rank - ar_rank` を測り、dataset・fault typeごとの改善／悪化とTop-1遷移を確認する。完全順位が保存されていない旧artifactは解析対象にせず、推測による順位補完を禁止する。

## 追記: Counterfactual ARの検証

persistent shift吸収の直接診断を受け、異常観測値をlagへ戻さない`counterfactual_ar`を正式アブレーションvariantとして追加する。長期再帰時の非定常AR発散を防ぐため、counterfactual baselineは正常windowのmin/maxへ制限し、clip率を診断値として保存する。このboundは結果へ影響しうるため、observed-lagとの差を純粋なlag入力差だけとして解釈しない。
