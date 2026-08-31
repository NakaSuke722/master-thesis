#!/usr/bin/env python3
"""Bounded RUN timing/parity check; never writes formal case results.

Default: compare three target fits from one case. --full-case explicitly
compares a whole case including the original (potentially very slow) pruning.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import networkx as nx
import numpy as np

from data_loader import list_benchmark_processed_cases, load_benchmark_processed_case
from experiments.config import load_config
from models.baselines.common import prepare_paired_metric_data
from models.baselines.run import RUNScorer, _torch_modules


def reference_prune_cycles(graph, data):
    """Pre-optimization implementation, used only for bounded comparisons."""
    removed = 0
    while not nx.is_directed_acyclic_graph(graph):
        edges = list(graph.edges)
        correlations = []
        for source, target in edges:
            value = data[source].corr(data[target])
            correlations.append(float(value) if np.isfinite(value) else 0.0)
        graph.remove_edge(*edges[int(np.argmin(correlations))])
        removed += 1
    return removed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/baselines/run.yaml")
    parser.add_argument("--dataset", choices=["re1_ob", "re1_ss", "re1_tt"], required=True)
    parser.add_argument("--case-id", help="Default: first case in sorted processed manifest")
    parser.add_argument("--targets", type=int, default=3)
    parser.add_argument("--full-case", action="store_true")
    parser.add_argument("--output", type=Path, help="Optional NEW diagnostic JSON; refuses overwrite")
    args = parser.parse_args()
    if args.targets <= 0:
        parser.error("--targets must be positive")
    if args.output and args.output.exists():
        parser.error(f"Output already exists: {args.output}")
    config = load_config(args.config)
    params = config["model"].get("params", {})
    if config["model"]["target"] != "run":
        parser.error("--config must select RUN")
    loader_args = {
        "benchmark": config["benchmark"]["name"],
        "dataset": args.dataset,
        "strategy": config["model"].get("preprocess_strategy", "default"),
        "processed_root": config["paths"]["processed_data_dir"],
    }
    cases = list_benchmark_processed_cases(**loader_args)
    if args.case_id:
        cases = [case for case in cases if case.case_id == args.case_id]
    if not cases:
        parser.error("No matching processed case")
    case = cases[0]
    normal, abnormal, _ = load_benchmark_processed_case(case_id=case.case_id, **loader_args)
    paired = prepare_paired_metric_data(normal, abnormal, drop_segment_constants=False)
    values = ((paired.normal - paired.normal.mean()) / paired.normal.std(ddof=0).replace(0, 1)).to_numpy()
    scorer_keys = ("seq_len", "hidden_size", "moving_average_kernel", "pretrain_epochs",
                   "epochs", "learning_rate", "batch_size", "device", "torch_num_threads", "seed")
    options = {key: params[key] for key in scorer_keys if key in params}
    torch, nn, _ = _torch_modules()
    # Pay one-off optimizer imports before either timing. Target fits re-seed.
    torch.optim.Adam([nn.Parameter(torch.ones(1))])
    report = {"case_id": case.case_id, "dataset": args.dataset,
              "mode": "full_case" if args.full_case else "target_subset",
              "normal_samples": len(values), "input_metrics": values.shape[1],
              "parameters": options, "torch_version": torch.__version__, "backends": {}}
    outputs = {}
    for backend in ("reference", "vectorized"):
        model = RUNScorer(**options, execution_backend=backend)
        device = model._resolve_device()
        print(f"{case.case_id}: {backend} ({report['mode']})", flush=True)
        start = time.perf_counter()
        if args.full_case:
            if backend == "reference":
                model._prune_cycles = reference_prune_cycles
            ranking = model.predict(normal, abnormal)
            diagnostic = model.diagnostics_
            outputs[backend] = np.array([list(row.values()) for row in diagnostic["attention_by_target"].values()])
            details = {"ranking": ranking, "graph_edges": diagnostic["graph_edges"],
                       "stage_time_sec": diagnostic["stage_time_sec"]}
        else:
            x, y = model._windows(values)
            outputs[backend] = np.array([
                model._fit_target(x, y, target, device)
                for target in range(min(args.targets, values.shape[1]))
            ])
            details = {"target_count": len(outputs[backend])}
        elapsed = time.perf_counter() - start
        report["backends"][backend] = {"seconds": elapsed, **details}
        print(f"  {elapsed:.3f} sec", flush=True)
    old, new = outputs["reference"], outputs["vectorized"]
    report["maximum_attention_absolute_difference"] = float(np.max(np.abs(old - new)))
    report["attention_close_at_1e_6"] = bool(np.allclose(old, new, rtol=0, atol=1e-6))
    report["parent_sets_equal"] = all(
        set(RUNScorer._potential_parents(a)) == set(RUNScorer._potential_parents(b))
        for a, b in zip(old, new)
    )
    if args.full_case:
        reference, vectorized = report["backends"].values()
        report["service_ranking_equal"] = reference["ranking"] == vectorized["ranking"]
        report["graph_edges_equal"] = reference["graph_edges"] == vectorized["graph_edges"]
    # Attention arrays and formal RCA results are intentionally not written.
    text = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as file:
            file.write(text + "\n")
        print(f"Diagnostic saved: {args.output}", flush=True)
    else:
        print(text)
    if not report["attention_close_at_1e_6"] or not report["parent_sets_equal"] or (
        args.full_case and not (report["service_ranking_equal"] and report["graph_edges_equal"])
    ):
        raise SystemExit("Parity check failed; inspect the diagnostic before resuming the benchmark")


if __name__ == "__main__":
    main()
