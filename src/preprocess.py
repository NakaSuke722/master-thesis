import os
import glob
import argparse
import pandas as pd

# 1. 各前処理モジュールのインポート
# （事前に src/preprocessors/ ディレクトリを作成し、これらのファイルを用意しておく必要があります）
from preprocessors.default import apply as apply_default
from preprocessors.standardized import apply as apply_standardized
from preprocessors.min_max import apply as apply_min_max

# 2. 戦略(Strategy)の登録辞書
STRATEGIES = {
    "default": apply_default,
    "standardized": apply_standardized,
    "min_max": apply_min_max
}

def get_causal_graph(dataset_name, variables):
    """
    アーキテクチャ図から完全にトレースしたコールグラフとドメイン知識を用いて、
    メトリクス間の完全な因果グラフ（隣接リスト）を動的に生成する。
    """
    causal_graph = {var: [] for var in variables}
    call_graph = {}
    
    # ==========================================
    # 1. 画像から完全トレースしたコールグラフの定義
    # ==========================================
    if dataset_name == "online_boutique":
        # 画像1: Online Boutique (全サービスとRedisを網羅)
        call_graph = {
            "frontend-external": ["frontend"],
            "frontend": ["adservice", "cartservice", "checkoutservice", "currencyservice", 
                         "productcatalogservice", "recommendationservice", "shippingservice"],
            "checkoutservice": ["cartservice", "currencyservice", "emailservice", 
                                "paymentservice", "productcatalogservice", "shippingservice"],
            "recommendationservice": ["productcatalogservice"],
            "cartservice": ["redis"],
            # 末端ノード（呼び出し先を持たないもの）も明記
            "adservice": [], "currencyservice": [], "emailservice": [], 
            "paymentservice": [], "productcatalogservice": [], "shippingservice": [], "redis": []
        }
        
    elif dataset_name == "sock_shop":
        # 画像2: Sock Shop (DB群およびRabbitMQの経路を完全網羅)
        # ※データセットの命名規則（複数形など）を吸収できるよう配慮
        call_graph = {
            "front-end": ["orders", "payment", "user", "catalogue", "cart"],
            "orders": ["shipping", "payment", "user", "cart", "orders-db"], # Mongoへ
            "user": ["user-db"],             # Mongoへ
            "catalogue": ["catalogue-db"],   # Mongoへ
            "cart": ["carts-db"],            # MySQLへ
            "shipping": ["rabbitmq"],        # Queue(RabbitMQ)へ
            "rabbitmq": ["queue-master"],    # QueueからMasterへ
            # 末端ノード
            "payment": [], "queue-master": [], "orders-db": [], 
            "user-db": [], "catalogue-db": [], "carts-db": []
        }
        
    elif dataset_name == "train_ticket":
        # 画像3: Train Ticket (図中の全ノードと矢印を完全トレース)
        call_graph = {
            "ts-ui-dashboard": ["ts-gateway-service"],
            "ts-gateway-service": [
                "ts-auth-service", "ts-verification-code-service", "ts-ticket-office-service",
                "ts-avatar-service", "ts-news-service", "ts-user-service", "ts-food-service",
                "ts-security-service", "ts-consign-service", "ts-contacts-service",
                "ts-assurance-service", "ts-wait-order-service", "ts-preserve-service",
                "ts-preserve-other-service", "ts-rebook-service", "ts-payment-service",
                "ts-inside-payment-service", "ts-notification-service", "ts-delivery-service",
                "ts-execute-service", "ts-cancel-service", "ts-order-service",
                "ts-order-other-service", "ts-travel-service", "ts-travel2-service",
                "ts-seat-service", "ts-basic-service", "ts-route-service", "ts-station-service",
                "ts-train-service", "ts-admin-user-service", "ts-admin-route-service",
                "ts-admin-travel-service", "ts-admin-basic-info-service", "ts-admin-order-service",
                "ts-config-service", "ts-voucher-service"
            ],
            "ts-food-service": ["ts-station-food-service", "ts-train-food-service", "ts-food-delivery-service"],
            "ts-consign-service": ["ts-consign-price-service"],
            "ts-wait-order-service": ["ts-preserve-service", "ts-preserve-other-service"],
            "ts-rebook-service": ["ts-payment-service", "ts-inside-payment-service"],
            "ts-preserve-service": ["ts-security-service", "ts-contacts-service", "ts-assurance-service", "ts-seat-service", "ts-travel-service", "ts-station-service", "ts-user-service"],
            "ts-preserve-other-service": ["ts-security-service", "ts-contacts-service", "ts-assurance-service", "ts-seat-service", "ts-travel2-service", "ts-station-service", "ts-user-service"],
            "ts-execute-service": ["ts-order-service", "ts-order-other-service"],
            "ts-cancel-service": ["ts-order-service", "ts-order-other-service"],
            "ts-order-service": ["ts-voucher-service"],
            "ts-order-other-service": ["ts-voucher-service"],
            "ts-travel-service": ["ts-route-service", "ts-station-service", "ts-train-service", "ts-route-plan-service", "ts-seat-service", "ts-order-service", "ts-basic-service"],
            "ts-travel2-service": ["ts-route-service", "ts-station-service", "ts-train-service", "ts-route-plan-service", "ts-seat-service", "ts-order-other-service", "ts-basic-service"],
            "ts-basic-service": ["ts-route-service", "ts-station-service", "ts-train-service"],
            "ts-travel-plan-service": ["ts-route-plan-service"],
            "ts-admin-route-service": ["ts-route-service"],
            "ts-admin-travel-service": ["ts-travel-service", "ts-travel2-service"],
            "ts-admin-basic-info-service": ["ts-price-service", "ts-route-service", "ts-station-service", "ts-train-service", "ts-basic-service", "ts-route-plan-service"],
            "ts-seat-service": ["ts-config-service"],
            "ts-admin-order-service": ["ts-order-service", "ts-order-other-service"]
        }

    # ==========================================
    # 2. 変数間の波及ルール適用 (ドメイン知識)
    # ==========================================
    for var in variables:
        if "_" not in var and "-" not in var:
            continue
            
        # "cartservice_cpu" -> svc="cartservice", metric="cpu" に分割
        parts = var.rsplit('_', 1) 
        if len(parts) != 2:
            continue
        svc, metric = parts[0], parts[1]

        # 表現の揺れ（orderとorders等）を吸収するためのヘルパー
        # サービス名が call_graph のキーや値に部分一致するかを確認する
        matched_svc = svc
        for defined_svc in call_graph.keys():
            if svc in defined_svc or defined_svc in svc:
                matched_svc = defined_svc
                break

        # --- ルールA: サービス内部の因果 (Workload -> CPU/Mem -> Latency/Error) ---
        if metric in ["cpu", "mem"]:
            workload_var = f"{svc}_workload"
            if workload_var in variables:
                causal_graph[var].append(workload_var)
        
        if metric.startswith("latency") or metric == "error":
            for parent_metric in ["cpu", "mem", "workload"]:
                parent_var = f"{svc}_{parent_metric}"
                if parent_var in variables:
                    causal_graph[var].append(parent_var)

        # --- ルールB: サービス間の因果 ---
        # 1. 順伝播 (CallerのWorkload -> CalleeのWorkload)
        if metric == "workload":
            for caller, callees in call_graph.items():
                if matched_svc in callees:
                    # callerの実際のメトリクス名を探す
                    caller_workload = f"{caller}_workload"
                    # 変数リストにあるプレフィックスと一致するものを追加
                    for v in variables:
                        if v.endswith("_workload") and (caller in v or v.replace("_workload", "") in caller):
                            if v not in causal_graph[var]:
                                causal_graph[var].append(v)

        # 2. 逆流伝播 (CalleeのLatency/Error -> CallerのLatency/Error)
        if metric.startswith("latency") or metric == "error":
            if matched_svc in call_graph:
                for callee in call_graph[matched_svc]:
                    # calleeの実際のメトリクス名を探す
                    for v in variables:
                        if (v.endswith(metric)) and (callee in v or v.replace(f"_{metric}", "") in callee):
                            if v not in causal_graph[var]:
                                causal_graph[var].append(v)

    return causal_graph

def process_dataset(strategy_name):
    # 入力された戦略が存在するかチェック
    if strategy_name != "all" and strategy_name not in STRATEGIES:
        raise ValueError(f"Unknown strategy: {strategy_name}")

    base_raw_dir = "data/raw"
    search_pattern = os.path.join(base_raw_dir, "*", "*", "*", "simple_data.csv")
    
    # 実行する戦略のリストを決定（"all"なら全て、それ以外は指定された1つだけ）
    targets = STRATEGIES.keys() if strategy_name == "all" else [strategy_name]
    
    for filepath in glob.glob(search_pattern):
        parts = filepath.split(os.sep)
        dataset = parts[-4]
        fault_type = parts[-3]
        run_id = parts[-2]
        base_dir = os.path.dirname(filepath)
        
        # --- ここから共通の前処理 ---
        
        # A. 異常発生時刻 (t_F) の読み込み
        inject_time_file = os.path.join(base_dir, "inject_time.txt")
        if not os.path.exists(inject_time_file): 
            continue
            
        with open(inject_time_file, "r") as f:
            inject_time = int(f.read().strip())

        # B. データの読み込み
        df = pd.read_csv(filepath)
        if 'time' not in df.columns: 
            continue
            
        # C. 異常発生時刻(t_F)のインデックス特定
        abnormal_indices = df[df['time'] >= inject_time].index
        if len(abnormal_indices) == 0: 
            continue
            
        t_f = abnormal_indices[0]
        if t_f == 0: 
            continue

        # D. 変数のクレンジング
        df = df.drop(columns=['time'])
        df = df.loc[:, df.std() > 0]  # 分散ゼロの変数を削除
        df = df.ffill().fillna(0)     # 欠損値の補完
        
        # E. 正常データと異常データの分割
        df_normal = df.iloc[:t_f].copy()
        df_abnormal = df.iloc[t_f:].copy()
        
        # F. 因果グラフの取得
        causal_graph = get_causal_graph(dataset, list(df.columns))
        
        # G. メタデータの構築（後続のモデルへ渡す情報）
        graph_info = {
            "variables": list(df.columns),
            "t_f_index": int(t_f),
            "t_f_timestamp": inject_time,
            "normal_samples": len(df_normal),
            "abnormal_samples": len(df_abnormal),
            "causal_graph": causal_graph
        }
        
        # --- 共通処理ここまで ---

        # 3. 登録された各モジュール(Strategy)へ処理を委譲
        for current_strategy in targets:
            # 出力先のディレクトリパスを動的に生成
            output_dir = os.path.join("data/processed", current_strategy, dataset, fault_type, run_id)
            
            # 実行すべき関数（apply）を取得
            processor_func = STRATEGIES[current_strategy]
            
            # 個別ロジック（保存処理）の実行
            processor_func(df_normal, df_abnormal, df.columns, output_dir, graph_info)

    print(f"Data processing complete. (Generated strategy: {strategy_name})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess dataset with specific strategy")
    parser.add_argument(
        "--strategy", 
        type=str, 
        default="all", 
        help="Specify strategy to generate (e.g., 'default', 'standardized', 'min_max', or 'all')"
    )
    args = parser.parse_args()
    
    process_dataset(args.strategy)