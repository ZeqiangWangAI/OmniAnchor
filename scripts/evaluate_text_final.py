"""Apply frozen reference transforms and fitted probes to an admitted protected evaluation."""
import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from vlanchor import to_matrix, transform
from vlanchor.analysis import bh_fdr
from vlanchor.campaign import create_run, append_event, merge_score_shards
from vlanchor.evaluation import evaluate_candidates, evaluate_multilabel, evaluate_vad
from vlanchor.io import load_samples, load_scores, load_calibration, read_json, write_json
from vlanchor.provenance import file_hash


def baseline_methods(contract):
    methods = contract.get("baseline_methods", ["e5", "qwen-embedding", "qwen-reranker"])
    if methods not in (["e5", "qwen-embedding", "qwen-reranker"], ["qwen-embedding", "qwen-reranker"]):
        raise ValueError("Require the complete declared text or image baseline set.")
    if set(contract["methods"]) != {"native", *methods}:
        raise ValueError("Baseline methods disagree with the scoring admission.")
    return methods


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["contract", "data", "native-run", "baseline-run", "output"]:
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    contract = read_json(args.contract)
    if contract["status"] != "frozen":
        raise ValueError("Require frozen final protocol.")
    baselines = baseline_methods(contract)
    for name, digest in contract.get("analysis_source_sha256", {}).items():
        if file_hash(Path(__file__).resolve().parents[1]/name) != digest:
            raise ValueError("Final analysis changed after freeze.")
    for name, digest in contract["frozen_artifacts"].items():
        if file_hash(Path(name)) != digest:
            raise ValueError(f"Fitted artifact changed: {name}")
    for name, key in [("samples.json", "samples_sha256"), ("labels.csv", "labels_sha256")]:
        if file_hash(args.data/name) != contract[key]:
            raise ValueError("Final population/gold bytes changed.")
    samples = load_samples(args.data/"samples.json")
    ids = [s.id for s in samples]
    if ids != contract["sample_ids"] or any(s.metadata["split"] != "test" for s in samples):
        raise ValueError("Final population/roles differ.")
    for root in [args.native_run, args.baseline_run]:
        if (root/"exit_code.txt").read_text().strip() != "0":
            raise ValueError("Failed/incomplete model evaluation.")
    folders = [args.native_run/"measurement"] + [args.baseline_run/m for m in baselines]
    for folder in folders:
        if read_json(folder/"manifest.json")["frozen_evaluation"]["sha256"] != file_hash(args.contract):
            raise ValueError("Scoring did not use the frozen protocol.")
    columns = contract["label_columns"]
    labels = pd.read_csv(args.data/"labels.csv", index_col="sample_id")
    labels.index = labels.index.astype(str)
    if labels.index.duplicated().any() or set(labels.index) != set(ids) or set(labels.columns) != set(columns):
        raise ValueError("Gold coordinates differ.")
    gold = labels.loc[ids, columns].to_numpy(float)
    parts = sorted((args.native_run/"measurement").glob("part-*.parquet"))
    raw = merge_score_shards([load_scores(p) for p in parts], ids)
    feature_root, probe_root = Path(contract["probe_features"]), Path(contract["probes"])
    calibrated = transform(raw, load_calibration(feature_root/"calibration.json"))
    features = {}
    for variant in ["raw_logp", "reference_log_ratio", "reference_z"]:
        name = "native-"+variant
        matrix = to_matrix(calibrated, variant=variant, missing="error")
        features[name] = matrix.values[[matrix.sample_ids.index(i) for i in ids]][:,
            [matrix.anchor_ids.index(c) for c in contract["feature_columns"][name]]]
    source_hashes = {str(p): file_hash(p) for p in parts}
    for method in baselines:
        folder = args.baseline_run/method
        current, previous = read_json(folder/"manifest.json"), read_json(feature_root/f"{method}-identity.json")
        for key in ["method", "model_id", "revision", "spec_sha256", "instruction", "precision", "preprocessing", "vision_reuse"]:
            if current.get(key) != previous.get(key):
                raise ValueError(f"Baseline feature identity changed: {method}/{key}")
        with np.load(folder/"matrix.npz", allow_pickle=False) as data:
            row_ids, anchor_ids = data["sample_ids"].tolist(), data["anchor_ids"].tolist()
            if len(row_ids) != len(set(row_ids)) or set(row_ids) != set(ids):
                raise ValueError("Baseline coverage mismatch.")
            index = [row_ids.index(i) for i in ids]
            anchor_columns = contract["feature_columns"][method+"-anchor"]
            if len(anchor_ids) != len(set(anchor_ids)) or set(anchor_ids) != set(anchor_columns):
                raise ValueError("Baseline anchors mismatch.")
            features[method+"-anchor"] = data["scores"][index][:, [anchor_ids.index(c) for c in anchor_columns]]
            if method+"-original" in contract["feature_columns"]:
                features[method+"-original"] = data["features"][index]
        source_hashes[str(folder/"matrix.npz")] = file_hash(folder/"matrix.npz")
    if set(features) != set(contract["feature_columns"]):
        raise ValueError("Final method coverage mismatch.")
    for name, values in features.items():
        if values.shape != (len(ids), len(contract["feature_columns"][name])) or not np.isfinite(values).all():
            raise ValueError("Malformed final features.")
    create_run(args.output, {"purpose": "Frozen final direct/probe evaluation; no fit or selection", "test_used": True,
        "task": contract["task"], "contract_sha256": file_hash(args.contract), "sample_ids": ids,
        "source_sha256": source_hashes, "script_sha256": file_hash(Path(__file__)),
        "conditioning": contract["conditioning"], "source_group_crossings": contract["source_group_crossings"]})
    feature_info = read_json(feature_root/"features.json")
    predictions, direct, summaries = {}, {}, {}
    for name, values in features.items():
        np.savez_compressed(args.output/f"{name}-features.npz", values=values, sample_ids=np.array(ids),
            column_ids=np.array(contract["feature_columns"][name]))
        probe = joblib.load(probe_root/name/"probe.joblib")
        predictions[name] = probe.predict_scores(values)
        if not np.isfinite(predictions[name]).all():
            raise ValueError("Nonfinite probe prediction.")
        metric = evaluate_multilabel if contract["task"] == "multilabel" else evaluate_vad
        summaries[name] = metric(gold, predictions[name], columns)
        np.savez_compressed(args.output/f"{name}-predictions.npz", values=predictions[name], sample_ids=np.array(ids), label_columns=np.array(columns))
        if contract["task"] == "multilabel" and feature_info[name]["direct_anchor_scores"]:
            coords = contract["feature_columns"][name]
            direct[name] = values[:, [coords.index(c) for c in columns]]
    groups = np.array([s.group_id or s.id for s in samples])
    blocks = [np.flatnonzero(groups == g) for g in np.unique(groups)]
    if len(blocks) < 2:
        raise ValueError("Insufficient independent source groups.")
    rng = np.random.default_rng(42)
    draws = [np.concatenate([blocks[j] for j in rng.integers(len(blocks), size=len(blocks))]) for _ in range(1000)]
    rows, differences = [], []

    def statistics(values, kind):
        def metric(y, p):
            if contract["task"] == "multilabel":
                return np.array([evaluate_multilabel(y, p, columns)["macro_ap"]])
            return np.array([spearmanr(y[:, j], p[:, j]).statistic for j in range(len(columns))])
        names = ["macro_ap"] if contract["task"] == "multilabel" else ["spearman_"+c for c in columns]
        results = {}
        for name, prediction in values.items():
            point = metric(gold, prediction)
            boot = np.stack([metric(gold[d], prediction[d]) for d in draws])
            results[name] = (point, boot)
            for j, label in enumerate(names):
                valid = np.isfinite(boot[:, j])
                if valid.sum() < 800 or not np.isfinite(point[j]):
                    raise ValueError("Insufficient defined statistical draws; do not impute.")
                lo, hi = np.quantile(boot[valid, j], [.025, .975])
                rows.append(dict(kind=kind, method=name, metric=label, value=point[j], ci_lower=lo, ci_upper=hi, n=len(ids)))
        p, b = results["native-raw_logp"]
        comparators = contract["primary_comparators"] if kind == "probe" else ["e5-anchor", "qwen-embedding-anchor", "qwen-reranker-anchor"]
        family = []
        for name in comparators:
            q, c = results[name]
            for j, label in enumerate(names):
                delta = (b-c)[:, j]
                delta = delta[np.isfinite(delta)]
                if len(delta) < 800:
                    raise ValueError("Insufficient paired draws.")
                point = (p-q)[j]
                lo, hi = np.quantile(delta, [.025, .975])
                pv = (1+(np.abs(delta-point) >= abs(point)).sum())/(len(delta)+1)
                family.append(dict(kind=kind, comparator=name, metric=label, difference=point,
                    ci_lower=lo, ci_upper=hi, centered_bootstrap_p=pv))
        for row, q in zip(family, bh_fdr(np.array([r["centered_bootstrap_p"] for r in family]))):
            differences.append({**row, "family_bh_q": q})

    statistics(predictions, "probe")
    if direct:
        statistics(direct, "direct")
    shuffle = np.random.default_rng(42).permutation(len(ids))
    write_json(args.output/"direct.json", {k: {"multilabel": evaluate_multilabel(gold, v, columns),
        "within_sample": evaluate_candidates(dict(zip(ids, gold)), dict(zip(ids, v))),
        "label_shuffle_diagnostic": evaluate_multilabel(gold[shuffle], v, columns)} for k, v in direct.items()})
    write_json(args.output/"probe-metrics.json", summaries)
    pd.DataFrame(rows).to_csv(args.output/"metrics.csv", index=False)
    pd.DataFrame(differences).to_csv(args.output/"paired-differences.csv", index=False)
    append_event(args.output, "completed")


if __name__ == "__main__":
    main()
