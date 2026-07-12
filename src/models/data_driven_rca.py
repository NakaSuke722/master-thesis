import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import lingam

# ==========================================
# フェーズ3用: GNNオートエンコーダの定義
# ==========================================
class DenseGNNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.relu = nn.ReLU()

    def forward(self, x, adj):
        out = self.linear(x)
        out = torch.matmul(adj, out)
        return self.relu(out)

class RCA_GNN_Autoencoder(nn.Module):
    def __init__(self, hidden_dim=16):
        super().__init__()
        self.enc_layer1 = DenseGNNLayer(1, hidden_dim)
        self.enc_layer2 = DenseGNNLayer(hidden_dim, 1)
        self.dec_layer1 = DenseGNNLayer(1, hidden_dim)
        self.dec_layer2 = DenseGNNLayer(hidden_dim, 1)

    def forward(self, x, adj_fwd, adj_rev):
        h_enc = self.enc_layer1(x, adj_rev)
        S = self.enc_layer2(h_enc, adj_rev)
        h_dec = self.dec_layer1(S, adj_fwd)
        x_hat = self.dec_layer2(h_dec, adj_fwd)
        return S, x_hat

# ==========================================
# 統合モデル: データ駆動型 RCA パイプライン
# ==========================================
class DataDrivenRCA:
    def __init__(self, lambda_reg=0.1, epochs=300, lr=0.01):
        self.lambda_reg = lambda_reg
        self.epochs = epochs
        self.lr = lr
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

    def fit_predict(self, df_normal: pd.DataFrame, df_abnormal: pd.DataFrame):
        # 内部計算の警告を非表示にする
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        warnings.filterwarnings("ignore", category=UserWarning)

        variables = df_normal.columns.tolist()
        num_nodes = len(variables)
        
        # データのコピーを作成
        norm_values = df_normal.values.copy()
        abn_values = df_abnormal.values.copy()
        
        # --- データクリーニング: 分散ゼロ（定数列）への微小ジターの付加 ---
        # 正常データにおいて標準偏差が0の列を検知
        stds = np.std(norm_values, axis=0)
        constant_indices = np.where(stds == 0)[0]
        
        if len(constant_indices) > 0:
            np.random.seed(42)
            for idx in constant_indices:
                # 計算の破綻を防ぐため、10^-6スケールの極小ガウスノイズを注入
                jitter_norm = np.random.normal(0, 1e-6, norm_values.shape[0])
                jitter_abn = np.random.normal(0, 1e-6, abn_values.shape[0])
                norm_values[:, idx] += jitter_norm
                abn_values[:, idx] += jitter_abn

        # フェーズ1: LiNGAMによる因果構造の探索
        model_lingam = lingam.DirectLiNGAM()
        model_lingam.fit(norm_values)
        B_hat = model_lingam.adjacency_matrix_
        
        # フェーズ2: 異常残差の抽出
        # e_abn = X_abn - X_abn @ B_hat.T
        residuals = abn_values - np.dot(abn_values, B_hat.T)
        X_in = np.mean(np.abs(residuals), axis=0).reshape(-1, 1)

        # フェーズ3: GNNによる波及効果の吸収とスコアリング
        I = np.eye(num_nodes)
        adj_fwd = torch.tensor(B_hat + I, dtype=torch.float32).to(self.device)
        adj_rev = torch.tensor(B_hat.T + I, dtype=torch.float32).to(self.device)
        x_tensor = torch.tensor(X_in, dtype=torch.float32).to(self.device)

        model = RCA_GNN_Autoencoder().to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        model.train()
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            S, x_hat = model(x_tensor, adj_fwd, adj_rev)
            
            recon_loss = criterion(x_hat, x_tensor)
            l1_loss = torch.sum(torch.abs(S))
            loss = recon_loss + self.lambda_reg * l1_loss
            
            loss.backward()
            optimizer.step()

        # 4. 最終スコアリングとランキングの構築
        model.eval()
        with torch.no_grad():
            final_S, _ = model(x_tensor, adj_fwd, adj_rev)
            scores = final_S.cpu().numpy().flatten()

        ranking = []
        # print("\n--- Final Root Cause Ranking ---")
        sorted_indices = np.argsort(scores)[::-1]
        for rank, idx in enumerate(sorted_indices):
            var_name = variables[idx]
            score = scores[idx]
            ranking.append(var_name)
            # if rank < 10:
            #     print(f"Rank {rank+1}: {var_name} (Score: {score:.4f})")

        return ranking