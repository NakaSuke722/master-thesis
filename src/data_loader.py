# src/data_loader.py
import os
import pandas as pd
import numpy as np

def load_timeseries_data(
        dataset: str,
        fault_type: str, 
        run_id: int, 
        base_path: str = "data/raw"):
    """
    指定された条件のデータを読み込み、前処理を行う関数。
    戻り値: 前処理済みデータフレーム, 正解ラベル(文字列)
    """
    file_path = os.path.join(base_path, dataset, fault_type, str(run_id), "simple_data.csv")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data not found: {file_path}")
        
    df = pd.read_csv(file_path)
    
    # 1. 因果に関与しない時間情報の削除
    if 'time' in df.columns:
        df = df.drop(columns=['time'])
        
    # 2. 分散がゼロ（値が変動しない）の無効な変数を削除
    df = df.loc[:, df.std() > 0]
    
    # 3. 欠損値の補間（直前の値で埋める等）
    df = df.ffill().fillna(0)
    
    # 正解ラベルの抽出（フォルダ名 'cartservice_cpu' などがそのまま原因ノード名となる）
    ground_truth = fault_type
    
    return df, ground_truth