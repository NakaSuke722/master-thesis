# src/models/nonlinear_anm.py
import torch
import torch.nn as nn

class MechanismNetwork(nn.Module):
    """
    1つの変数 X_i に対する波及関数 f_i を学習するニューラルネットワーク（MLP）。
    親ノード PA_i の値を受け取り、X_i の値を予測する。
    """
    def __init__(self, input_dim: int, hidden_dim: int = 32, dropout_rate: float = 0.2):
        super(MechanismNetwork, self).__init__()
        
        # 親を持たない変数（独立変数）の場合は、定数を学習するダミーパラメータを用意する
        self.is_independent = (input_dim == 0)
        
        if self.is_independent:
            self.bias = nn.Parameter(torch.zeros(1))
        else:
            self.network = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(p=dropout_rate),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(p=dropout_rate),
                nn.Linear(hidden_dim, 1)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        順伝播処理
        x: 親ノード群のテンソル (バッチサイズ, PA_iの次元数)
        """
        if self.is_independent:
            # 親がない場合は、学習されたバイアス定数を返す
            return self.bias.expand(x.size(0), 1)
        else:
            return self.network(x)
            
    def mc_dropout_predict(self, x: torch.Tensor, num_samples: int = 30) -> tuple:
        """
        MC Dropoutを用いた推論（フェーズ2, 3で使用）。
        Dropoutを有効にした状態で複数回推論を行い、予測の平均(予測値)と分散(不確実性)を返す。
        """
        if self.is_independent:
            # 独立変数の場合、推論のブレ（不確実性）はゼロとする
            pred = self.bias.expand(x.size(0), 1)
            return pred, torch.zeros_like(pred)
            
        # 推論時でもDropout層を有効化する（訓練モードへ強制移行）
        self.train()
        
        predictions = []
        with torch.no_grad():
            for _ in range(num_samples):
                predictions.append(self.network(x))
                
        # テンソルをスタックして統計量を計算
        predictions_stack = torch.stack(predictions, dim=0) # (num_samples, batch, 1)
        
        mean_prediction = torch.mean(predictions_stack, dim=0)
        variance = torch.var(predictions_stack, dim=0) # これが認識論的不確実性 (Epistemic Uncertainty)
        
        return mean_prediction, variance


class ANMSystem(nn.ModuleDict):
    """
    システム全体の因果グラフを管理し、全変数分の MechanismNetwork を保持するコンテナクラス。
    """
    def __init__(self, variables: list, causal_graph: dict, hidden_dim: int = 32, dropout_rate: float = 0.2):
        super(ANMSystem, self).__init__()
        self.variables = variables
        self.causal_graph = causal_graph
        
        # 因果グラフに基づいて、各変数用のネットワークを動的に生成し辞書に登録
        for var in variables:
            parents = causal_graph.get(var, [])
            input_dim = len(parents)
            self[var] = MechanismNetwork(input_dim, hidden_dim, dropout_rate)