import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.special import logsumexp
import lingam
import warnings

class BayesianRCA:
    """
    ベイズ推論による逐次更新型 根本原因分析モデル
    """
    def __init__(self, tau_sq=50.0, theta=0.99, max_steps=None):
        self.tau_sq = tau_sq      # 異常エネルギーの想定分散
        self.theta = theta        # 確信度閾値
        self.max_steps = max_steps # 最大監視ウィンドウ幅

    def fit_predict(self, df_normal: pd.DataFrame, df_abnormal: pd.DataFrame, dataset_name: str = None):
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        warnings.filterwarnings("ignore", category=UserWarning)

        # 0. 定数メトリクスの除外（次元削減によるLiNGAMの破綻防止）
        stds = df_normal.std(axis=0)
        valid_cols = stds[stds > 1e-6].index.tolist()
        
        df_norm_valid = df_normal[valid_cols].copy()
        df_abn_valid = df_abnormal[valid_cols].copy()
        
        variables = valid_cols
        N = len(variables)
        
        norm_values = df_norm_valid.values
        abn_values = df_abn_valid.values

        # ==========================================
        # フェーズ1：オフライン学習（正常状態の因果メカニズム同定）
        # ==========================================
        # 事前知識を一切与えず、データ駆動のみで因果構造を学習
        model_lingam = lingam.DirectLiNGAM()
        model_lingam.fit(norm_values)
        B_hat = model_lingam.adjacency_matrix_

        # 正常残差の抽出
        norm_residuals = norm_values - np.dot(norm_values, B_hat.T)
        
        # 正常残差の分布パラメータ算出
        mu = np.mean(norm_residuals, axis=0)
        sigma_sq = np.var(norm_residuals, axis=0, ddof=1)
        sigma_sq = np.clip(sigma_sq, 1e-6, None) # ゼロ除算防止

        # ==========================================
        # フェーズ2：オンライン推論（異常波及の相殺とベイズ更新）
        # ==========================================
        T_abn = abn_values.shape[0]
        max_t = self.max_steps if self.max_steps is not None else T_abn
        max_t = min(max_t, T_abn)

        # 事前確率の初期化 (アンダーフローを防ぐため対数空間で保持)
        log_P = np.full(N, -np.log(N))

        for t in range(max_t):
            x_t = abn_values[t, :]
            
            # Step 2.2 残差の抽出（波及キャンセレーション）
            e_t = x_t - np.dot(B_hat, x_t)
            
            # Step 2.3 各仮説の周辺尤度の計算
            # ベースライン尤度: 仮に全ノードが正常ノイズに従った場合の対数尤度
            base_log_lik = norm.logpdf(e_t, loc=mu, scale=np.sqrt(sigma_sq))
            sum_base_log_lik = np.sum(base_log_lik)
            
            log_L = np.zeros(N)
            for k in range(N):
                # 仮説 M_k: ノードkのみ、分散が sigma_k^2 + tau^2 に変質したと仮定
                abn_log_lik_k = norm.logpdf(e_t[k], loc=mu[k], scale=np.sqrt(sigma_sq[k] + self.tau_sq))
                # 該当ノードkだけを異常尤度に差し替える
                log_L[k] = sum_base_log_lik - base_log_lik[k] + abn_log_lik_k
            
            # Step 2.4 ベイズの定理による信念の更新
            log_unnormalized_P = log_L + log_P
            # Log-Sum-Expトリックによる正規化係数の算出
            log_Z = logsumexp(log_unnormalized_P)
            log_P = log_unnormalized_P - log_Z
            
            # Step 3.1 確信度閾値によるリアルタイム判定
            P_t = np.exp(log_P)
            if np.any(P_t > self.theta):
                break

        # ==========================================
        # フェーズ3：終了判定と根本原因の出力
        # ==========================================
        # 最終時刻における事後確率分布（MAP推定）
        P_final = np.exp(log_P)
        
        ranking = []
        sorted_indices = np.argsort(P_final)[::-1]
        for idx in sorted_indices:
            ranking.append(variables[idx])

        # 除外された定数メトリクスをランキングの末尾に結合
        excluded_cols = [c for c in df_normal.columns if c not in valid_cols]
        ranking.extend(excluded_cols)

        return ranking