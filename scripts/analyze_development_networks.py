"""E6 development diagnostics with common nodes and source-bootstrap edge stability."""
import argparse
import json
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

from omnianchor.analysis import bh_fdr
from omnianchor.campaign import append_event, create_run
from omnianchor.evaluation import compare_networks
from omnianchor.io import load_samples, read_json, write_json
from omnianchor.provenance import file_hash


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--dev", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frozen-evaluation", type=Path)
    args = parser.parse_args()
    samples = load_samples(args.dev / "samples.json")
    contract = read_json(args.frozen_evaluation) if args.frozen_evaluation else None
    if contract:
        if contract["status"] != "frozen" or file_hash(Path(__file__)) != contract["analysis_source_sha256"]:
            raise ValueError("Network analysis is not the frozen implementation.")
        for name, digest in contract["statistics_source_sha256"].items():
            if file_hash(Path(__file__).resolve().parents[1]/name) != digest:
                raise ValueError("Network statistics changed after freeze.")
        for name, digest in contract["evaluation_files"].items():
            if file_hash(args.dev/name) != digest:
                raise ValueError("Network evaluation population changed.")
        if read_json(args.features/"manifest.json")["contract_sha256"] != contract["scoring_contract_sha256"]:
            raise ValueError("Network features were not produced by the frozen scoring evaluation.")
        events = [line for line in (args.features/"events.jsonl").read_text().splitlines() if line]
        if not events or json.loads(events[-1])["status"] != "completed":
            raise ValueError("Final feature evaluation is incomplete.")
    role = "test" if contract else "dev"
    if any(s.metadata["split"] != role for s in samples):
        raise ValueError("Evaluation split does not match the admitted analysis role.")
    ids = [s.id for s in samples]
    labels = pd.read_csv(args.dev / "labels.csv", index_col="sample_id")
    labels.index = labels.index.astype(str)
    if len(ids) != len(set(ids)) or labels.index.duplicated().any() or set(labels.index) != set(ids):
        raise ValueError("Labels must exactly cover the evaluation samples.")
    columns = labels.columns.tolist()
    values = {"human": labels.loc[ids].to_numpy(dtype=float)}
    inputs = {}
    feature_info = (contract["feature_info"] if contract else read_json(args.features / "features.json"))
    for name, metadata in feature_info.items():
        if not metadata["direct_anchor_scores"]:
            continue
        path = args.features / (f"{name}-features.npz" if contract else f"{name}.npz")
        with np.load(path, allow_pickle=False) as data:
            row_ids, col_ids = data["sample_ids"].tolist(), data["column_ids"].tolist()
            if (len(row_ids) != len(set(row_ids)) or len(col_ids) != len(set(col_ids))
                    or set(col_ids) != set(columns) or not set(ids) <= set(row_ids)):
                raise ValueError("Invalid feature coordinate identity.")
            values[name] = data["values"][[row_ids.index(i) for i in ids]][:, [col_ids.index(c) for c in columns]]
        inputs[name] = file_hash(path)
    if any(v.shape != (len(ids), len(columns)) or not np.isfinite(v).all() for v in values.values()):
        raise ValueError("Network inputs must be complete finite aligned matrices.")
    keep = np.logical_and.reduce([np.ptp(v, axis=0)>0 for v in values.values()])
    nodes = [c for c, retained in zip(columns, keep) if retained]
    if len(nodes) < 3:
        raise ValueError("Insufficient common nonconstant nodes.")
    all_values = values
    values = {name: v[:, keep] for name, v in values.items()}
    create_run(args.output, {"purpose": "E6 frozen final network evaluation" if contract else "E6 development diagnostics, no test or graph-parameter selection",
        "test_used": bool(contract), "analysis_contract_sha256": file_hash(args.frozen_evaluation) if contract else None,
        "nodes": nodes, "removed_constant_nodes": [c for c, k in zip(columns, keep) if not k],
        "sample_ids": ids, "input_hashes": inputs, "labels_sha256": file_hash(args.dev / "labels.csv"),
        "seed": 42, "bootstrap": 1000, "confidence": .95, "display_top_k": 20,
        "definition": "full signed Pearson; top20 absolute nonzero edges only for display/stability",
        "undefined_policy": "Preserve full original node matrices with NaN for undefined correlations; comparisons use common nonconstant nodes only"})
    for name, matrix in all_values.items():
        with np.errstate(invalid="ignore", divide="ignore"):
            full = np.corrcoef(matrix, rowvar=False)
        pd.DataFrame(full, index=columns, columns=columns).to_csv(args.output/f"{name}-all-nodes-adjacency.csv")
    groups = np.array([s.group_id or s.id for s in samples])
    blocks = [np.flatnonzero(groups==g) for g in np.unique(groups)]
    if len(blocks) < 2:
        raise ValueError("Need independent source groups for bootstrap.")
    rng = np.random.default_rng(42)
    draws = [np.concatenate([blocks[j] for j in rng.integers(0, len(blocks), len(blocks))]) for _ in range(1000)]
    upper = np.triu_indices(len(nodes), 1)
    gold = np.corrcoef(values["human"], rowvar=False)
    pd.DataFrame(gold, index=nodes, columns=nodes).to_csv(args.output / "human-full-adjacency.csv")
    results, bootstrap_rhos = {}, {}
    try:
        for name, matrix in values.items():
            if name == "human":
                continue
            folder = args.output / name
            folder.mkdir()
            adjacency = np.corrcoef(matrix, rowvar=False)
            pd.DataFrame(adjacency, index=nodes, columns=nodes).to_csv(folder / "full-adjacency.csv")
            point = compare_networks(adjacency, gold, top_k=20)
            shuffled = matrix.copy()
            control_rng = np.random.default_rng(42)
            for j in range(len(nodes)):
                shuffled[:, j] = shuffled[control_rng.permutation(len(matrix)), j]
            control = compare_networks(np.corrcoef(shuffled, rowvar=False), gold, top_k=20)
            common_row = matrix[np.random.default_rng(42).permutation(len(matrix))]
            invariance = float(np.max(np.abs(np.corrcoef(common_row, rowvar=False)-adjacency)))
            if invariance > 1e-12:
                raise ValueError("Common-row permutation failed Pearson invariance.")
            weights, rhos, draw_ids, frequency = [], [], [], np.zeros(len(upper[0]), dtype=int)
            for draw_id, index in enumerate(draws):
                if np.any(np.ptp(matrix[index], axis=0)==0) or np.any(np.ptp(values["human"][index], axis=0)==0):
                    continue
                a, g = np.corrcoef(matrix[index], rowvar=False), np.corrcoef(values["human"][index], rowvar=False)
                rho = compare_networks(a, g, top_k=20)["edge_weight_spearman"]
                if rho is None:
                    continue
                weights.append(a[upper])
                rhos.append(rho)
                draw_ids.append(draw_id)
                ranked = [i for i in np.argsort(-np.abs(a[upper]), kind="stable") if a[upper][i] != 0][:20]
                frequency[ranked] += 1
            if len(weights) < 800:
                raise ValueError("Fewer than80% valid source-bootstrap draws; uncertainty is not identifiable.")
            low, high = np.quantile(weights, [.025, .975], axis=0)
            edges = pd.DataFrame({"source": [nodes[i] for i in upper[0]], "target": [nodes[j] for j in upper[1]],
                "weight": adjacency[upper], "ci_lower": low, "ci_upper": high,
                "top20_bootstrap_frequency": frequency/len(weights)})
            edges.to_csv(folder / "edges.csv", index=False)
            graph = nx.Graph()
            graph.add_nodes_from(nodes)
            for row in edges.to_dict("records"):
                graph.add_edge(row.pop("source"), row.pop("target"), **row)
            nx.write_graphml(graph, folder / "full.graphml")
            results[name] = {"point": point, "edge_weight_spearman_ci": np.quantile(rhos, [.025, .975]).tolist(),
                "valid_bootstraps": len(weights), "invalid_bootstraps": 1000-len(weights),
                "independent_source_groups": len(blocks), "independent_column_shuffle_control": control,
                "common_row_permutation_max_error": invariance, "control_is_significance_test": False}
            write_json(folder / "summary.json", results[name])
            bootstrap_rhos[name] = dict(zip(draw_ids, rhos))
        paired = []
        native = "native-raw_logp"
        if native in results:
            for baseline in ["e5-anchor", "qwen-embedding-anchor", "qwen-reranker-anchor"]:
                if baseline not in results:
                    continue
                common = sorted(set(bootstrap_rhos[native]) & set(bootstrap_rhos[baseline]))
                if len(common) < 800:
                    raise ValueError("Fewer than80% common valid paired network bootstrap draws.")
                delta = results[native]["point"]["edge_weight_spearman"] - results[baseline]["point"]["edge_weight_spearman"]
                differences = np.array([bootstrap_rhos[native][i]-bootstrap_rhos[baseline][i] for i in common])
                low, high = np.quantile(differences, [.025, .975])
                p = (1+np.count_nonzero(np.abs(differences-delta) >= abs(delta)))/(len(common)+1)
                paired.append(dict(native=native, baseline=baseline, difference=delta, lower=low, upper=high,
                    valid_paired_resamples=len(common), centered_bootstrap_p=float(p)))
        if paired:
            q = bh_fdr([r["centered_bootstrap_p"] for r in paired])
            for row, adjusted in zip(paired, q):
                row["bh_q"] = float(adjusted)
                row["family"] = "frozen3anchor_baselines" if contract else "exploratory3anchor_baselines"
            pd.DataFrame(paired).to_csv(args.output/"paired-network-differences.csv", index=False)
        write_json(args.output/"bootstrap-edge-correlations.json", bootstrap_rhos)
        write_json(args.output / "summary.json", results)
        append_event(args.output, "completed")
    except BaseException as exc:
        append_event(args.output, "failed", error=repr(exc))
        raise


if __name__ == "__main__":
    main()
