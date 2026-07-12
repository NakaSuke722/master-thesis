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
    アーキテクチャのコールグラフと動的フォールバックを用いて、
    メトリクス間の完全な有向非巡回グラフ（DAG）を生成する。
    """
    causal_graph = {var: [] for var in variables}
    call_graph = {}
    
    # ==========================================
    # 1. コールグラフの定義（main をフロントエンドの呼び出し先として追加）
    # ==========================================
    if dataset_name == "online_boutique":
        call_graph = {
            "frontend-external": ["frontend", "main"], # main(物理ホスト)を全体のエントリポイントに紐付け
            "frontend": ["adservice", "cartservice", "checkoutservice", "currencyservice", 
                         "productcatalogservice", "recommendationservice", "shippingservice"],
            "checkoutservice": ["cartservice", "currencyservice", "emailservice", 
                                "paymentservice", "productcatalogservice", "shippingservice"],
            "recommendationservice": ["productcatalogservice"],
            "cartservice": ["redis"],
            "adservice": [], "currencyservice": [], "emailservice": [], 
            "paymentservice": [], "productcatalogservice": [], "shippingservice": [], "redis": [], "main": []
        }
    elif dataset_name == "sock_shop":
        call_graph = {
            "front-end": ["orders", "payment", "user", "catalogue", "cart", "main"],
            "orders": ["shipping", "payment", "user", "cart", "orders-db"],
            "user": ["user-db"], "catalogue": ["catalogue-db"], "cart": ["carts-db"],
            "shipping": ["rabbitmq"], "rabbitmq": ["queue-master"],
            "payment": [], "queue-master": [], "orders-db": [], "user-db": [], "catalogue-db": [], "carts-db": [], "main": []
        }
    elif dataset_name == "train_ticket":
        call_graph = {
            "ts-ui-dashboard": ["ts-gateway-service", "main"],
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
            "ts-admin-order-service": ["ts-order-service", "ts-order-other-service"],
            "main": []
        }

    # ==========================================
    # 2. 波及ルールと動的フォールバックの適用
    # ==========================================
    for var in variables:
        if "_" not in var and "-" not in var:
            continue
            
        parts = var.rsplit('_', 1) 
        if len(parts) != 2:
            continue
        svc, metric = parts[0], parts[1]

        matched_svc = svc
        for defined_svc in call_graph.keys():
            if svc in defined_svc or defined_svc in svc:
                matched_svc = defined_svc
                break

        # --- A. サービス内部の因果 ---
        has_workload = False
        if metric in ["cpu", "mem"]:
            workload_var = f"{svc}_workload"
            if workload_var in variables:
                causal_graph[var].append(workload_var)
                has_workload = True
            
            # 【動的フォールバック1】自身にWorkload指標がない場合（Redis, Main等）
            # Caller(呼び出し元)のWorkloadを直接の親とする
            if not has_workload:
                for caller, callees in call_graph.items():
                    if matched_svc in callees:
                        for v in variables:
                            if v.endswith(f"{caller}_workload") and v not in causal_graph[var]:
                                causal_graph[var].append(v)
        
        has_latency = False
        if metric.startswith("latency") or metric == "error":
            for parent_metric in ["cpu", "mem", "workload"]:
                parent_var = f"{svc}_{parent_metric}"
                if parent_var in variables:
                    causal_graph[var].append(parent_var)
                    has_latency = True

        # --- B. サービス間の因果 ---
        if metric == "workload":
            for caller, callees in call_graph.items():
                if matched_svc in callees:
                    for v in variables:
                        if v.endswith(f"{caller}_workload") and v not in causal_graph[var]:
                            causal_graph[var].append(v)

        if metric.startswith("latency") or metric == "error":
            if matched_svc in call_graph:
                for callee in call_graph[matched_svc]:
                    callee_has_latency = False
                    for v in variables:
                        if (v.endswith(metric)) and (callee in v or v.replace(f"_{metric}", "") in callee):
                            if v not in causal_graph[var]:
                                causal_graph[var].append(v)
                                callee_has_latency = True
                    
                    # 【動的フォールバック2】CalleeにLatency指標がない場合（Redis等）
                    # CalleeのCPU/Mem悪化が、直接CallerのLatencyに波及するとみなす
                    if not callee_has_latency:
                        for v in variables:
                            if (v.endswith("_cpu") or v.endswith("_mem")) and (callee in v or v.replace("_cpu", "").replace("_mem", "") in callee):
                                if v not in causal_graph[var]:
                                    causal_graph[var].append(v)

    # ==========================================
    # 3. 孤立ノードの最終救済処置 (Safety Net)
    # ==========================================
    # 全ルールを適用してもなお孤立しているノードは、システム全体の入り口に強制接続する
    for var in variables:
        is_child = len(causal_graph[var]) > 0
        is_parent = any(var in parents for parents in causal_graph.values())
        
        if not is_child and not is_parent:
            for v in variables:
                if (v == "frontend_workload" or v == "frontend-external_workload") and v != var:
                    causal_graph[var].append(v)

    # ==========================================
    # 4. バグフリーの強制DAG化アルゴリズム
    # ==========================================
    cleaned_graph = {var: [] for var in variables}
    
    def is_ancestor(potential_ancestor, node):
        """nodeから親を辿ってpotential_ancestorに到達できるか（真なら閉路ができる）"""
        visited = set()
        stack = [node]
        while stack:
            curr = stack.pop()
            if curr == potential_ancestor:
                return True
            if curr not in visited:
                visited.add(curr)
                for p in cleaned_graph.get(curr, []):
                    stack.append(p)
        return False

    for child, parents in causal_graph.items():
        unique_parents = list(set(parents))
        for parent in unique_parents:
            if child == parent:
                continue # 自己ループの排除
            if not is_ancestor(child, parent):
                cleaned_graph[child].append(parent)

    return cleaned_graph

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