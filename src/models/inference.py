# src/models/inference.py
import os
import json
import torch
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
from models.nonlinear_anm import ANMSystem

class RCAInference:
    def __init__(self, processed_dir: str, gamma: float = 1.0, mc_samples: int = 30):
        self.processed_dir = processed_dir
        self.gamma = gamma
        self.mc_samples = mc_samples
        
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

    def load_processed_data(self):
        """前処理済みの正常データ、異常データ、およびメタデータを読み込む"""
        df_normal = pd.read_csv(os.path.join(self.processed_dir, "normal_data.csv"))
        df_abnormal = pd.read_csv(os.path.join(self.processed_dir, "abnormal_data.csv"))
        
        with open(os.path.join(self.processed_dir, "graph_info.json"), "r") as f:
            graph_info = json.load(f)
            
        return df_normal, df_abnormal, graph_info

    def compute_rca_scores(self, system: ANMSystem) -> list:
        """
        フェーズ2およびフェーズ3を包括的に実行し、補正異常スコアのランキングを返す
        """
        # 1. データの準備
        df_normal, df_abnormal, graph_info = self.load_processed_data()
        variables = graph_info["variables"]
        causal_graph = graph_info["causal_graph"]
        var_to_idx = {var: i for i, var in enumerate(variables)}
        
        # テンソル化
        normal_tensor = torch.tensor(df_normal.values, dtype=torch.float32).to(self.device)
        abnormal_tensor = torch.tensor(df_abnormal.values, dtype=torch.float32).to(self.device)
        
        # 全変数分のスコアを格納する辞書
        final_scores = {}
        
        # モデルを評価・推論状態（MC Dropoutのため内部でtrainモードを制御）へ
        system.to(self.device)
        system.eval()
        
        # 2. 変数ごとに独立して残差抽出と不確実性の定量化を実行
        for var in variables:
            parents = causal_graph.get(var, [])
            target_idx = var_to_idx[var]
            
            # --- 正常期間の決定論的残差抽出 (フェーズ1の経験分布用) ---
            y_normal_true = normal_tensor[:, target_idx].unsqueeze(1)
            if len(parents) > 0:
                p_indices = [var_to_idx[p] for p in parents]
                x_normal_input = normal_tensor[:, p_indices]
            else:
                x_normal_input = torch.zeros(normal_tensor.size(0), 1).to(self.device)
                
            with torch.no_grad():
                # 正常時はDropoutを無効にした決定論的予測
                system[var].eval()
                y_normal_pred = system[var](x_normal_input)
                eps_normal = (y_normal_true - y_normal_pred).detach().cpu().numpy().flatten()
            
            # --- 異常期間のMC Dropout推論 (フェーズ2 & 3) ---
            y_abnormal_true = abnormal_tensor[:, target_idx].unsqueeze(1)
            if len(parents) > 0:
                p_indices = [var_to_idx[p] for p in parents]
                x_abnormal_input = abnormal_tensor[:, p_indices]
            else:
                x_abnormal_input = torch.zeros(abnormal_tensor.size(0), 1).to(self.device)
                
            # MC推定の実行 (Dropout層を強制的に訓練モードにして複数回フォワードパス)
            y_abnormal_pred_mean, var_mc = system[var].mc_dropout_predict(
                x_abnormal_input, num_samples=self.mc_samples
            )
            
            # 決定論的残差の抽出
            eps_abnormal = (y_abnormal_true - y_abnormal_pred_mean).detach().cpu().numpy().flatten()
            
            # 認識論的不確実性 U_i の時間平均算出
            u_i = torch.mean(var_mc).item()
            
            # --- 1-Wasserstein距離の計算 (フェーズ3.1) ---
            # 正常時ノイズ分布 P_i と異常時ノイズ分布 P~_i の幾何学的距離
            s_i = wasserstein_distance(eps_normal, eps_abnormal)
            
            # --- スコアの割引補正 (フェーズ3.3) ---
            s_i_corrected = s_i / (1.0 + self.gamma * u_i)
            
            final_scores[var] = s_i_corrected

        # 3. 補正スコアに基づいて降順ソート（ランキング化）
        sorted_ranking = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
        predicted_ranking = [var for var, score in sorted_ranking]
        
        # --- デバッグ用出力（上位5件の生スコアを確認） ---
        print("\n[Debug] Top 5 Variables and Scores:")
        for i in range(5):
            var_name = sorted_ranking[i][0]
            var_score = sorted_ranking[i][1]
            print(f"  Rank {i+1}: {var_name} (Score: {var_score:.4f})")
        
        return predicted_ranking