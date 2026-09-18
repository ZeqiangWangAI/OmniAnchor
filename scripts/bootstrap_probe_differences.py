"""Paired source-bootstrap differences for frozen development probe predictions."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from vlanchor.campaign import create_run, append_event
from vlanchor.evaluation import evaluate_multilabel
from vlanchor.io import load_samples, read_json, write_json
from vlanchor.provenance import file_hash


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probes", type=Path, required=True)
    parser.add_argument("--dev", type=Path, required=True)
    parser.add_argument("--primary", default="native-raw_logp")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    samples = load_samples(args.dev / "samples.json")
    if any(s.metadata["split"] != "dev" for s in samples):
        raise ValueError("Development diagnostic only; rejects test.")
    ids = [s.id for s in samples]
    labels = pd.read_csv(args.dev / "labels.csv", index_col="sample_id")
    labels.index = labels.index.astype(str)
    columns = labels.columns.tolist()
    gold = labels.loc[ids].to_numpy(float)
    task = read_json(args.probes / "manifest.json")["task"]
    predictions, hashes = {}, {}
    for name in read_json(args.probes / "selection.json"):
        path = args.probes / name / "dev-predictions.npz"
        with np.load(path, allow_pickle=False) as data:
            rows, cols = data["sample_ids"].tolist(), data["label_columns"].tolist()
            if len(rows) != len(set(rows)) or set(rows) != set(ids) or set(cols) != set(columns):
                raise ValueError("Predictions do not exactly match frozen evaluation coordinates.")
            predictions[name] = data["values"][[rows.index(i) for i in ids]][:, [cols.index(c) for c in columns]]
        hashes[name] = file_hash(path)
    if args.primary not in predictions or not all(np.isfinite(v).all() for v in predictions.values()):
        raise ValueError("Missing primary or nonfinite predictions.")
    groups = np.array([s.group_id or s.id for s in samples])
    blocks = [np.flatnonzero(groups == g) for g in np.unique(groups)]
    if len(blocks) < 2:
        raise ValueError("Need at least two independent source groups.")
    rng = np.random.default_rng(42)
    draws = [np.concatenate([blocks[j] for j in rng.integers(len(blocks), size=len(blocks))]) for _ in range(1000)]
    metric_names = (["spearman_"+c for c in columns]+["mean_spearman", "mse"]
                    if task == "regression" else ["macro_ap"])

    def metric(y, p):
        if task == "multilabel":
            return np.array([evaluate_multilabel(y, p, columns)["macro_ap"]])
        rho = [float(spearmanr(y[:, j], p[:, j]).statistic) for j in range(y.shape[1])]
        return np.array(rho+[float(np.mean(rho)), float(np.mean((y-p)**2))])

    create_run(args.output, {"purpose": "paired dev diagnostic; no final-test or post-selection coverage claim",
        "primary": args.primary, "task": task, "seed": 42, "bootstrap_replicates": 1000,
        "confidence": .95, "sampling_unit": "source group", "groups": len(blocks), "sample_ids": ids,
        "prediction_sha256": hashes, "label_sha256": file_hash(args.dev / "labels.csv"),
        "source_sha256": file_hash(Path(__file__)),
        "estimand": "row-weighted metric difference, primary minus comparator; same group draws for every method",
        "conditioning": "fixed fitted probes and dev-selected hyperparameters; no training/refitting uncertainty"})
    rows = []
    try:
        base = predictions[args.primary]
        point_base = metric(gold, base)
        bootstrap_base = np.stack([metric(gold[i], base[i]) for i in draws])
        for name, values in predictions.items():
            if name == args.primary:
                continue
            point = point_base - metric(gold, values)
            differences = bootstrap_base - np.stack([metric(gold[i], values[i]) for i in draws])
            np.savez_compressed(args.output / f"{name}-draws.npz", differences=differences,
                                metric_names=np.array(metric_names))
            for j, metric_name in enumerate(metric_names):
                valid = differences[:, j][np.isfinite(differences[:, j])]
                if len(valid) < 800:
                    raise ValueError("Fewer than80% defined bootstrap draws; do not silently omit.")
                lo, hi = np.quantile(valid, [.025, .975])
                rows.append({"primary": args.primary, "comparator": name, "metric": metric_name,
                    "difference": float(point[j]), "ci_lower": float(lo), "ci_upper": float(hi),
                    "valid_replicates": len(valid), "positive_favors_primary": metric_name != "mse"})
        pd.DataFrame(rows).to_csv(args.output / "paired-differences.csv", index=False)
        write_json(args.output / "summary.json", rows)
        append_event(args.output, "completed")
    except BaseException as exc:
        append_event(args.output, "failed", error=repr(exc))
        raise


if __name__ == "__main__":
    main()
