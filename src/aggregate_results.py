# src/aggregate_results.py
import os
import glob
import json
import argparse
from collections import defaultdict

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--total-time", type=float, default=0.0)
    args = parser.parse_args()

    results_dir = "results/metrics"
    output_file = "results/final_summary.json"

    all_results = []
    pure_execution_time = 0.0
    model_used = "unknown"

    for filepath in glob.glob(os.path.join(results_dir, "*", "*.json")):
        with open(filepath, "r") as f:
            data = json.load(f)
            all_results.append(data)
            pure_execution_time += data.get("execution_time_sec", 0.0)
            
            # 個別データから使用されたモデル名を取得（最初の1つで上書き）
            if model_used == "unknown" and "model_used" in data:
                model_used = data["model_used"]

    if not all_results: 
        return

    dataset_metrics = defaultdict(lambda: defaultdict(list))
    for res in all_results:
        for k, v in res["metrics"].items():
            dataset_metrics[res["dataset"]][k].append(v)

    summary = {ds: {k: round(sum(v)/len(v), 4) for k, v in metrics.items()} 
               for ds, metrics in dataset_metrics.items()}

    # ターミナルのサマリ表示にもモデル名を記載
    print(f"\n=== Evaluation Summary (Model: {model_used}) ===")
    print(json.dumps(summary, indent=4))
    
    final_time = args.total_time if args.total_time > 0 else pure_execution_time
    print(f"\nTotal Execution Time: {round(final_time, 1)} seconds")

    # JSONの先頭に model_used を追加
    with open(output_file, "w") as f:
        json.dump({
            "model_used": model_used,
            "total_execution_time_sec": round(final_time, 1), 
            "pure_python_execution_time_sec": round(pure_execution_time, 2),
            "summary": summary, 
            "details": all_results
        }, f, indent=4)

if __name__ == "__main__":
    main()