import os
import json
import pandas as pd

def apply(df_normal, df_abnormal, columns, output_dir, graph_info):
    """スケーリングなしでそのまま保存する"""
    os.makedirs(output_dir, exist_ok=True)
    df_normal.to_csv(os.path.join(output_dir, "normal_data.csv"), index=False)
    df_abnormal.to_csv(os.path.join(output_dir, "abnormal_data.csv"), index=False)
    with open(os.path.join(output_dir, "graph_info.json"), "w") as f:
        json.dump(graph_info, f, indent=4)