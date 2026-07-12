import os
import json
from pyvis.network import Network

def visualize_causal_graph_interactive(json_path: str):
    """
    PyVisを用いて、ブラウザ上で操作可能なインタラクティブな因果グラフを生成する。
    """
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"File not found: {json_path}")

    with open(json_path, 'r') as f:
        data = json.load(f)
    
    causal_graph = data.get("causal_graph", {})
    if not causal_graph:
        return

    # PyVisネットワークの初期化（階層的レイアウトを有効化）
    net = Network(height="800px", width="100%", bgcolor="#ffffff", font_color="black", directed=True)
    
    # ノードの色分け定義
    colors = {
        "workload": "#a8e6cf",  # 緑
        "resource": "#dcedc1",  # 黄緑
        "performance": "#ff8b94", # 赤
        "default": "#d3d3d3"
    }

    # ノードの追加
    for node in data.get("variables", []):
        if "workload" in node:
            color = colors["workload"]
        elif "cpu" in node or "mem" in node:
            color = colors["resource"]
        elif "latency" in node or "error" in node:
            color = colors["performance"]
        else:
            color = colors["default"]
            
        net.add_node(node, label=node, color=color, shape="dot", size=15)

    # エッジの追加
    for child, parents in causal_graph.items():
        for parent in parents:
            # PyVisは From(親) -> To(子) の順で追加する
            net.add_edge(parent, child, color="gray")

    # 物理演算（反発力）の設定で見やすくする
    net.force_atlas_2based(gravity=-50, central_gravity=0.01, spring_length=100, spring_strength=0.08, damping=0.4)
    
    # HTMLファイルとして出力してブラウザで開く
    output_html = "causal_graph_interactive.html"
    net.show(output_html, notebook=False)
    print(f"Graph generated: Open {output_html} in your web browser.")

if __name__ == "__main__":
    target_json = "data/processed/standardized/online_boutique/cartservice_cpu/1/graph_info.json"
    visualize_causal_graph_interactive(target_json)