import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import lingam
from collections import defaultdict

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

    def _build_prior_knowledge(self, variables, dataset_name):
        """
        アーキテクチャ(コールグラフ)に基づき、LiNGAM用の事前知識行列(Prior Knowledge)を生成する。
        """
        call_graph = {}
        if dataset_name == "online_boutique":
            call_graph = {
                "frontend-external": ["frontend", "main"],
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

        adj_services = defaultdict(set)
        for caller, callees in call_graph.items():
            for callee in callees:
                adj_services[caller].add(callee)
                adj_services[callee].add(caller)

        num_vars = len(variables)
        prior_knowledge = np.zeros((num_vars, num_vars), dtype=int)

        def get_service(var_name):
            if "_" not in var_name and "-" not in var_name:
                return var_name
            parts = var_name.rsplit('_', 1)
            if len(parts) != 2:
                return var_name
            svc = parts[0]
            matched_svc = svc
            for defined_svc in call_graph.keys():
                if svc in defined_svc or defined_svc in svc:
                    matched_svc = defined_svc
                    break
            return matched_svc

        for i, var_i in enumerate(variables):
            svc_i = get_service(var_i)
            for j, var_j in enumerate(variables):
                if i == j:
                    prior_knowledge[i, j] = -1 
                    continue
                svc_j = get_service(var_j)
                
                if svc_i and svc_j and svc_i != svc_j:
                    if svc_j not in adj_services.get(svc_i, set()):
                        prior_knowledge[i, j] = -1

        return prior_knowledge

    def fit_predict(self, df_normal: pd.DataFrame, df_abnormal: pd.DataFrame, dataset_name: str):
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        warnings.filterwarnings("ignore", category=UserWarning)

        # 1. 定数メトリクスの完全除外（微小ジターの廃止）
        stds = df_normal.std(axis=0)
        # 正常時において標準偏差がゼロ(分散なし)の変数を学習から完全に除外する
        valid_cols = stds[stds > 1e-6].index.tolist()
        
        df_norm_valid = df_normal[valid_cols].copy()
        df_abn_valid = df_abnormal[valid_cols].copy()
        
        variables = valid_cols
        num_nodes = len(variables)
        
        norm_values = df_norm_valid.values
        abn_values = df_abn_valid.values

        # 事前知識行列の構築（有効な変数のみで構築）
        prior_knowledge = self._build_prior_knowledge(variables, dataset_name)

        # フェーズ1: LiNGAMによる因果構造の探索
        model_lingam = lingam.DirectLiNGAM(prior_knowledge=prior_knowledge)
        model_lingam.fit(norm_values)
        B_hat = model_lingam.adjacency_matrix_
        
        # フェーズ2: 異常残差の抽出とスパイクの強調
        residuals = abn_values - np.dot(abn_values, B_hat.T)
        # 2. 平均値による希釈を防ぎ、異常の極大エネルギー(99パーセンタイル)を抽出
        X_in = np.percentile(np.abs(residuals), 99, axis=0).reshape(-1, 1)

        # 3. GNN入力の正規化（L1正則化の効きを安定させるため、最大値で割って0〜1にスケール）
        max_val = np.max(X_in)
        if max_val > 0:
            X_in = X_in / max_val

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
        sorted_indices = np.argsort(scores)[::-1]
        for idx in sorted_indices:
            ranking.append(variables[idx])

        # 除外された変数をスコア0としてランキングの末尾に追加（評価エラー防止）
        excluded_cols = [c for c in df_normal.columns if c not in valid_cols]
        ranking.extend(excluded_cols)

        return ranking