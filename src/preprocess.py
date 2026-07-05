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
    データセット名に応じて、既知の因果グラフ（親ノードのリスト）を返す。
    本来は外部の topology.json 等から読み込む実装が望ましいが、ここではハードコーディングで例示する。
    """
    if dataset_name == "online_boutique":
        # 例：各メトリクスの親ノードを定義（実態に合わせて修正してください）
        return {
            "frontend_cpu": [],
            "cartservice_cpu": ["frontend_cpu"],
            "checkoutservice_cpu": ["frontend_cpu", "cartservice_cpu"],
            # ... 他の変数も同様に定義 ...
        }
    elif dataset_name == "sock_shop":
        # sock_shop 用のグラフ構造
        pass
    
    # グラフ情報が定義されていない場合は、すべてのノードを独立（親なし）として扱う安全装置
    return {var: [] for var in variables}

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