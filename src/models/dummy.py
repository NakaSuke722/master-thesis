# src/models/dummy.py
import random
from typing import List

def run_random_rca(variables: List[str], seed: int = 42) -> List[str]:
    """
    入力された変数のリストを無作為に並び替え、ダミーの原因ランキングを生成する。
    
    Parameters:
        variables (List[str]): 因果探索の対象となる変数（ノード）のリスト
        seed (int): 乱数シード（実験の再現性を担保するため固定）
        
    Returns:
        List[str]: 原因である確率が高いと予測された順番に並んだ変数のリスト
    """
    random.seed(seed)
    predicted_ranking = variables.copy()
    random.shuffle(predicted_ranking)
    
    return predicted_ranking