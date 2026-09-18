"""Paired VIVA candidate ranking; image controls keep the recipient action and gold."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from omnianchor import to_matrix
from omnianchor.analysis import bh_fdr
from omnianchor.campaign import create_run, append_event
from omnianchor.evaluation import evaluate_candidates
from omnianchor.io import load_samples, load_scores, read_json, write_json
from omnianchor.provenance import file_hash


def aligned_scores(folder, native, expected):
    rows = {}
    paths = sorted(folder.glob("part-*.parquet" if native else "part-*.npz"))
    for path in paths:
        if native:
            matrix = to_matrix(load_scores(path), variant="raw_logp", missing="error")
            if len(matrix.sample_ids) != 1:
                raise ValueError("Expected one VIVA recipient per part.")
            identifier, columns, values = matrix.sample_ids[0], matrix.anchor_ids, matrix.values[0]
        else:
            with np.load(path, allow_pickle=False) as data:
                identifier, columns, values = str(data["sample_id"]), data["anchor_ids"].tolist(), data["scores"].copy()
        if identifier in rows or identifier not in expected:
            raise ValueError("Duplicate or unexpected recipient.")
        if len(columns) != len(set(columns)) or set(columns) != set(expected[identifier]):
            raise ValueError("Candidate inventory mismatch.")
        rows[identifier] = values[[columns.index(c) for c in expected[identifier]]]
    if set(rows) != set(expected):
        raise ValueError("Incomplete recipient coverage; no implicit case deletion.")
    return rows, {str(p): file_hash(p) for p in paths}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--evaluation-samples", type=Path)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--labels", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frozen-evaluation", type=Path)
    args = parser.parse_args()
    samples = load_samples(args.samples)
    final = read_json(args.frozen_evaluation) if args.frozen_evaluation else None
    if final:
        if final["status"] != "frozen" or file_hash(args.samples) != final["samples_sha256"]:
            raise ValueError("Final VIVA inputs differ from freeze.")
        if [s.id for s in samples] != final["sample_ids"] or any(s.metadata["split"] != "test" for s in samples):
            raise ValueError("Final VIVA population/roles differ.")
        if args.evaluation_samples or len(args.labels) != 1 or file_hash(args.labels[0]) != final["labels_sha256"]:
            raise ValueError("Final analysis must use all recipients and exact frozen gold.")
        if file_hash(args.candidates) != final["viva_input_sha256"]["candidates.json"]:
            raise ValueError("Final candidate inventory differs.")
    elif any(s.metadata["split"] not in {"train", "dev"} for s in samples):
        raise ValueError("Development analyzer rejects protected test.")
    ids = [s.id for s in samples]
    evaluation = load_samples(args.evaluation_samples) if args.evaluation_samples else samples
    if not final and any(s.metadata["split"] not in {"train", "dev"} for s in evaluation):
        raise ValueError("Development evaluation subset rejects test.")
    evaluation_ids = [s.id for s in evaluation]
    if len(evaluation_ids) != len(set(evaluation_ids)) or not set(evaluation_ids) <= set(ids):
        raise ValueError("Invalid evaluation subset.")
    candidates = read_json(args.candidates)
    expected = {i: [a["id"] for a in candidates[i]] for i in ids}
    labels = pd.concat([pd.read_csv(p) for p in args.labels]).set_index(["sample_id", "anchor_id"])
    if labels.index.duplicated().any():
        raise ValueError("Duplicate recipient/candidate gold.")
    gold = {i: labels.loc[[(i, a) for a in expected[i]], "relevant"].to_numpy(int) for i in ids}
    conditions = ["image_action", "action_only", "mismatched_image_action"]
    methods = ["native", "qwen-embedding", "qwen-reranker"]
    scores, hashes, costs, collected, identities = {}, {}, {}, {}, {}
    for root in args.runs:
        if (root/"exit_code.txt").read_text().strip() != "0":
            raise ValueError("Require completed raw runs.")
        for method in methods:
            folder = root/method
            if not folder.exists():
                continue
            manifest = read_json(folder/"manifest.json")
            if final and manifest.get("frozen_evaluation", {}).get("sha256") != file_hash(args.frozen_evaluation):
                raise ValueError("Scoring did not use this final VIVA protocol.")
            shard_ids = manifest["sample_ids"]
            if len(shard_ids) != len(set(shard_ids)) or not set(shard_ids) <= set(ids) or manifest["method"] != method:
                raise ValueError("Scoring population differs from protocol.")
            identity = {k: manifest[k] for k in ["spec", "vision_reuse", "source_sha256", "conditions"]}
            if method in identities and identities[method] != identity:
                raise ValueError("VIVA shard instrument identities differ.")
            identities[method] = identity
            costs.setdefault(method, []).append({"run": str(root), **read_json(folder/"cost.json")})
            for condition in conditions:
                rows, sources = aligned_scores(folder/condition, method == "native", {i: expected[i] for i in shard_ids})
                hashes.update(sources)
                dest = collected.setdefault((method, condition), {})
                if set(dest) & set(rows):
                    raise ValueError("Duplicate VIVA recipient across shards.")
                dest.update(rows)
    for key, rows in collected.items():
        if set(rows) != set(ids):
            raise ValueError("Full scoring population incomplete.")
        metrics = evaluate_candidates({i: gold[i] for i in evaluation_ids}, {i: rows[i] for i in evaluation_ids})
        if metrics["excluded_samples"]:
            raise ValueError("Frozen valid population should contain positive and negative candidates.")
        ap = np.array([metrics["per_instance_ap"][i] for i in evaluation_ids])
        rr = np.array([evaluate_candidates({i: gold[i]}, {i: rows[i]})["mrr"] for i in evaluation_ids])
        scores[key] = np.column_stack([ap, rr])
    if len(scores) != 9:
        raise ValueError("Require all three methods and three paired conditions.")
    ids = evaluation_ids
    groups = np.array([s.group_id or s.id for s in evaluation])
    blocks = [np.flatnonzero(groups == g) for g in np.unique(groups)]
    rng = np.random.default_rng(42)
    draws = [np.concatenate([blocks[j] for j in rng.integers(len(blocks), size=len(blocks))]) for _ in range(1000)]
    create_run(args.output, {"purpose": "VIVA frozen final paired modality evidence" if final else "VIVA development paired modality evidence", "test_used": bool(final),
        "frozen_evaluation_sha256": file_hash(args.frozen_evaluation) if final else None,
        "samples": ids, "groups": len(blocks), "seed": 42,
        "bootstrap": "1000 recipient image-group draws conditional on fixed donor assignment and dataset",
        "mrr_ties": "stable frozen public candidate order", "annotation_origin": "model-assisted annotations with human verification; not new human-only ratings",
        "source_sha256": hashes, "script_sha256": file_hash(Path(__file__)),
        "input_sha256": {str(p): file_hash(p) for p in [args.samples, args.candidates, *args.labels,
            *([args.evaluation_samples] if args.evaluation_samples else [])]},
        "inference_status": "frozen final protocol" if final else "developmental; not confirmatory final test"})
    summaries, differences, per = [], [], []
    for (method, condition), values in scores.items():
        for j, metric in enumerate(["AP", "MRR"]):
            boot = np.array([values[d, j].mean() for d in draws])
            lo, hi = np.quantile(boot, [.025, .975])
            summaries.append(dict(method=method, condition=condition, metric=metric,
                value=values[:, j].mean(), ci_lower=lo, ci_upper=hi, n=len(ids)))
            per.extend(dict(sample_id=i, group_id=groups[k], method=method, condition=condition,
                metric=metric, value=values[k, j]) for k, i in enumerate(ids))
    comparisons = [("native", c) for c in conditions[1:]]
    comparisons += [(m, "image_action") for m in methods[1:]]
    for method, condition in comparisons:
        delta = scores[("native", "image_action")] - scores[(method, condition)]
        for j, metric in enumerate(["AP", "MRR"]):
            point = delta[:, j].mean()
            boot = np.array([delta[d, j].mean() for d in draws])
            lo, hi = np.quantile(boot, [.025, .975])
            p = (1 + (np.abs(boot-point) >= abs(point)).sum()) / 1001
            differences.append(dict(comparator=method, condition=condition, metric=metric,
                difference=point, ci_lower=lo, ci_upper=hi, centered_bootstrap_p=p))
    table = pd.DataFrame(differences)
    mask = table.metric == "AP"
    table.loc[mask, "primary_family_bh_q" if final else "exploratory_family_bh_q"] = bh_fdr(table.loc[mask, "centered_bootstrap_p"].to_numpy())
    pd.DataFrame(summaries).to_csv(args.output/"metrics.csv", index=False)
    table.to_csv(args.output/"paired-differences.csv", index=False)
    pd.DataFrame(per).to_csv(args.output/"per-recipient.csv", index=False)
    write_json(args.output/"costs.json", costs)
    append_event(args.output, "completed")


if __name__ == "__main__":
    main()
