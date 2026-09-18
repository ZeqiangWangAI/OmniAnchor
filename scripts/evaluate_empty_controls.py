"""Protected no-content and label-shuffle controls using previously fitted predictions."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from vlanchor.analysis import bh_fdr
from vlanchor.campaign import append_event, create_run
from vlanchor.evaluation import evaluate_candidates, evaluate_multilabel, evaluate_vad
from vlanchor.io import load_samples, read_json, write_json
from vlanchor.provenance import file_hash


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["contract", "data", "empty", "final-results", "output"]:
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    control = read_json(args.contract)
    if control["status"] != "frozen" or file_hash(Path(__file__)) != control["analyzer_sha256"]:
        raise ValueError("No-content analysis changed after freeze.")
    root = Path(__file__).resolve().parents[1]
    cpath = root/control["scoring_contract"]
    if file_hash(cpath) != control["scoring_contract_sha256"]:
        raise ValueError("Original final scoring protocol changed.")
    c = read_json(cpath)
    for path, digest in control["statistics_source_sha256"].items():
        if file_hash(root/path) != digest:
            raise ValueError("Control statistics changed after freeze.")
    for name, digest in control["empty_files_sha256"].items():
        if file_hash(args.empty/name) != digest:
            raise ValueError("Frozen no-content predictions changed.")
    for name, key in [("samples.json", "samples_sha256"), ("labels.csv", "labels_sha256")]:
        if file_hash(args.data/name) != c[key]:
            raise ValueError("Protected samples or labels changed.")
    m = read_json(args.final_results/"manifest.json")
    if m["contract_sha256"] != file_hash(cpath) or m["script_sha256"] != control["final_evaluator_sha256"]:
        raise ValueError("Original final evaluation does not match frozen scoring and analysis.")
    events = (args.final_results/"events.jsonl").read_text().splitlines()
    if not events or json.loads(events[-1])["status"] != "completed":
        raise ValueError("Final evaluation did not complete.")
    samples = load_samples(args.data/"samples.json")
    ids, columns = [s.id for s in samples], c["label_columns"]
    if ids != c["sample_ids"] or any(s.metadata["split"] != "test" for s in samples):
        raise ValueError("Protected population mismatch.")
    labels = pd.read_csv(args.data/"labels.csv", index_col="sample_id")
    labels.index = labels.index.astype(str)
    gold = labels.loc[ids, columns].to_numpy(float)
    groups = np.array([s.group_id or s.id for s in samples])
    blocks = [np.flatnonzero(groups == g) for g in np.unique(groups)]
    if len(blocks) < 2:
        raise ValueError("Control uncertainty needs at least two independent source groups.")
    rng = np.random.default_rng(42)
    draws = [np.concatenate([blocks[i] for i in rng.integers(len(blocks), size=len(blocks))]) for _ in range(1000)]
    shuffled = gold[np.random.default_rng(42).permutation(len(gold))]
    info = read_json(args.empty/"features.json")
    create_run(args.output, {"purpose": "Frozen final no-content and shuffled-label diagnostics",
        "control_contract_sha256": file_hash(args.contract), "original_final_manifest_sha256": file_hash(args.final_results/"manifest.json"),
        "sample_ids": ids, "source_groups": len(blocks), "test_used": True,
        "null_prediction": "One frozen no-content prediction repeated identically; no new fitting",
        "correlations": "Undefined constant-predictor Spearman is reported undefined, never zero; regression contrasts use MSE reduction",
        "shuffle": "One fixed seed42 row shuffle, descriptive only; not an exchangeable source-group significance test",
        "inference": "1000 paired source-group bootstrap,95%intervals,centered approximate bootstrap p; one8method probe family perstudy, separate6method direct family forValueEval"})
    summaries, rows, hashes = {}, [], {}
    try:
        for name, metadata in info.items():
            with np.load(args.empty/f"{name}.npz", allow_pickle=False) as data:
                empty_prediction = np.repeat(data["predictions"].astype(float), len(ids), axis=0)
                empty_features = np.repeat(data["features"].astype(float), len(ids), axis=0)
                empty_columns = data["column_ids"].tolist()
            path = args.final_results/f"{name}-predictions.npz"
            with np.load(path, allow_pickle=False) as data:
                if data["sample_ids"].tolist() != ids or data["label_columns"].tolist() != columns:
                    raise ValueError("Final prediction identity mismatch.")
                actual = data["values"].astype(float)
            hashes[str(path)] = file_hash(path)
            metric = evaluate_multilabel if c["task"] == "multilabel" else evaluate_vad
            summaries[name] = {"actual_probe": metric(gold, actual, columns),
                "no_content_probe": metric(gold, empty_prediction, columns),
                "shuffled_labels_probe_diagnostic": metric(shuffled, actual, columns)}
            comparisons = [("probe", actual, empty_prediction)]
            if c["task"] == "multilabel" and metadata["direct_anchor_scores"]:
                path = args.final_results/f"{name}-features.npz"
                with np.load(path, allow_pickle=False) as data:
                    if data["sample_ids"].tolist() != ids or data["column_ids"].tolist() != empty_columns:
                        raise ValueError("Final direct coordinate identity mismatch.")
                    current = data["values"][:, [empty_columns.index(a) for a in columns]].astype(float)
                empty_direct = empty_features[:, [empty_columns.index(a) for a in columns]]
                summaries[name]["no_content_direct"] = evaluate_multilabel(gold, empty_direct, columns)
                summaries[name]["no_content_within_sample"] = evaluate_candidates(dict(zip(ids, gold)), dict(zip(ids, empty_direct)))
                comparisons.append(("direct", current, empty_direct))
                hashes[str(path)] = file_hash(path)
            for kind, current, empty in comparisons:
                if current.shape != gold.shape or empty.shape != gold.shape or not np.isfinite(current).all() or not np.isfinite(empty).all():
                    raise ValueError("Invalid complete paired control predictions.")
                if c["task"] == "multilabel":
                    def statistic(index):
                        return evaluate_multilabel(gold[index], current[index], columns)["macro_ap"] - evaluate_multilabel(gold[index], empty[index], columns)["macro_ap"]
                    metric_name = "macro_ap_gain_over_no_content"
                else:
                    loss_gain = ((gold-empty)**2-(gold-current)**2).mean(axis=1)
                    def statistic(index):
                        return loss_gain[index].mean()
                    metric_name = "mse_reduction_from_no_content"
                    summaries[name]["actual_mse"] = float(np.mean((gold-current)**2))
                    summaries[name]["no_content_mse"] = float(np.mean((gold-empty)**2))
                point = statistic(np.arange(len(ids)))
                boot = np.array([statistic(index) for index in draws])
                if not np.isfinite(boot).all():
                    raise ValueError("Undefined paired control bootstrap statistic.")
                low, high = np.quantile(boot, [.025, .975])
                p = (1+np.count_nonzero(np.abs(boot-point) >= abs(point)))/(len(boot)+1)
                rows.append(dict(method=name, kind=kind, metric=metric_name, gain=float(point), lower=low, upper=high,
                    centered_bootstrap_p=float(p), positive_gain="content improves over fixed no-content prediction"))
        frame = pd.DataFrame(rows)
        for kind in frame.kind.unique():
            mask = frame.kind == kind
            frame.loc[mask, "bh_q"] = bh_fdr(frame.loc[mask, "centered_bootstrap_p"].to_numpy())
        frame.to_csv(args.output/"paired-control-differences.csv", index=False)
        write_json(args.output/"metrics.json", summaries)
        write_json(args.output/"final-input-hashes.json", hashes)
        append_event(args.output, "completed")
    except BaseException as exc:
        append_event(args.output, "failed", error=repr(exc))
        raise


if __name__ == "__main__":
    main()
