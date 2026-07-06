# src/models/trainer.py
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from models.nonlinear_anm import ANMSystem

class Phase1Trainer:
    def __init__(self, data_dir: str, epochs: int = 200, lr: float = 1e-3):
        self.data_dir = data_dir
        self.epochs = epochs
        self.lr = lr
        
        # M3 Mac (MPS) または GPU の自動割り当て
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
            
        # print(f"Using device: {self.device}")

    def load_data_and_graph(self):
        """前処理済みの正常データとメタデータを読み込む"""
        normal_path = os.path.join(self.data_dir, "normal_data.csv")
        info_path = os.path.join(self.data_dir, "graph_info.json")
        
        df_normal = pd.read_csv(normal_path)
        with open(info_path, "r") as f:
            graph_info = json.load(f)
            
        return df_normal, graph_info

    def train(self):
        """フェーズ1: 正常データを用いた全変数の非線形回帰学習"""
        # 1. データの準備
        df_normal, graph_info = self.load_data_and_graph()
        variables = graph_info["variables"]
        causal_graph = graph_info["causal_graph"]
        
        # テンソルへの変換 (Time_steps, Features)
        data_tensor = torch.tensor(df_normal.values, dtype=torch.float32).to(self.device)
        var_to_idx = {var: i for i, var in enumerate(variables)}
        
        # 2. モデルの初期化
        system = ANMSystem(variables, causal_graph).to(self.device)
        
        # 変数ごとに独立したオプティマイザを設定 (一括でも可能だが、管理上独立させる)
        optimizers = {
            var: optim.Adam(system[var].parameters(), lr=self.lr)
            for var in variables
        }
        criterion = nn.MSELoss()
        
        print(f"Starting Phase 1 Training on {len(variables)} variables...")
        system.train()
        
        # 3. 学習ループ
        for epoch in range(1, self.epochs + 1):
            total_loss = 0.0
            
            for var in variables:
                parents = causal_graph.get(var, [])
                
                # 目的変数 (Target: X_i)
                target_idx = var_to_idx[var]
                y_true = data_tensor[:, target_idx].unsqueeze(1) # (Batch, 1)
                
                # 入力変数 (Inputs: PA_i)
                if len(parents) > 0:
                    parent_indices = [var_to_idx[p] for p in parents]
                    x_input = data_tensor[:, parent_indices] # (Batch, num_parents)
                else:
                    # 親がない場合はダミー入力 (サイズ合わせ)
                    x_input = torch.zeros(data_tensor.size(0), 1).to(self.device)
                
                # フォワードパスとロス計算
                optimizers[var].zero_grad()
                y_pred = system[var](x_input)
                loss = criterion(y_pred, y_true)
                
                # バックプロパゲーションとパラメータ更新
                loss.backward()
                optimizers[var].step()
                
                total_loss += loss.item()
                
            # 進捗の表示 (50エポックごと)
            if epoch % 50 == 0 or epoch == 1:
                print(f"Epoch [{epoch}/{self.epochs}] | Total MSE Loss: {total_loss:.4f}")
                
        print("Phase 1 Training Completed.")
        return system

# 単体テスト用のエントリーポイント
if __name__ == "__main__":
    # 例として1つのデータセットでテスト実行
    test_dir = "data/processed/standardized/online_boutique/cartservice_cpu/1"
    trainer = Phase1Trainer(data_dir=test_dir)
    trained_model = trainer.train()