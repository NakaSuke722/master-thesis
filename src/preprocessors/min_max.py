import os
import json
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def apply(df_normal, df_abnormal, columns, output_dir, graph_info):
    """Min-Maxスケーリングを適用して保存する"""
    scaler = MinMaxScaler()
    scaler.fit(df_normal)
    
    df_normal_scaled = pd.DataFrame(scaler.transform(df_normal), columns=columns)
    df_abnormal_scaled = pd.DataFrame(scaler.transform(df_abnormal), columns=columns)
    
    os.makedirs(output_dir, exist_ok=True)
    df_normal_scaled.to_csv(os.path.join(output_dir, "normal_data.csv"), index=False)
    df_abnormal_scaled.to_csv(os.path.join(output_dir, "abnormal_data.csv"), index=False)
    with open(os.path.join(output_dir, "graph_info.json"), "w") as f:
        json.dump(graph_info, f, indent=4)