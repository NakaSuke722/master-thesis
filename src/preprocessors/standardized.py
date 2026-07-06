# src/preprocessors/standardized.py
import os
import json
import pandas as pd
from sklearn.preprocessing import StandardScaler

def apply(df_normal, df_abnormal, columns, output_dir, graph_info):
    """
    正常データ(T_normal)の分布を基準として標準化を行い、
    モデルの最適化およびWasserstein距離の比較基盤を構築する。
    """
    scaler = StandardScaler()
    
    # 警告：数理モデルの前提に従い、必ず正常データのみで分布パラメータを学習(fit)する
    scaler.fit(df_normal)
    
    # 学習した正常時の基準を用いて、両方のデータを変換(transform)する
    df_normal_scaled = pd.DataFrame(scaler.transform(df_normal), columns=columns)
    df_abnormal_scaled = pd.DataFrame(scaler.transform(df_abnormal), columns=columns)
    
    # 保存先ディレクトリの自動生成
    os.makedirs(output_dir, exist_ok=True)
    
    # データの保存
    df_normal_scaled.to_csv(os.path.join(output_dir, "normal_data.csv"), index=False)
    df_abnormal_scaled.to_csv(os.path.join(output_dir, "abnormal_data.csv"), index=False)
    
    # 動的因果グラフ(PA_i)を含むメタデータの保存
    with open(os.path.join(output_dir, "graph_info.json"), "w") as f:
        json.dump(graph_info, f, indent=4)