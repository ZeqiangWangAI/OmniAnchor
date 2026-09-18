"""Select frozen linear probes on train/dev, preserving models for one later protected evaluation."""
import argparse
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits

from vlanchor.campaign import append_event, create_run
from vlanchor.evaluation import evaluate_candidates, evaluate_multilabel, evaluate_vad, fit_linear_probe
from vlanchor.io import load_samples, read_json, write_json
from vlanchor.provenance import file_hash, runtime_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--dev", type=Path, required=True)
    parser.add_argument("--task", choices=["multilabel", "regression"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    train, dev = [load_samples(folder / "samples.json") for folder in [args.train, args.dev]]
    if any(s.metadata["split"] != "train" for s in train) or any(s.metadata["split"] != "dev" for s in dev):
        raise ValueError("Probe fitting requires explicit train and dev roles.")
    train_ids, dev_ids = [s.id for s in train], [s.id for s in dev]
    if set(train_ids) & set(dev_ids):
        raise ValueError("Probe row IDs overlap.")
    labels = []
    for folder, ids in [(args.train, train_ids), (args.dev, dev_ids)]:
        frame = pd.read_csv(folder / "labels.csv", index_col="sample_id")
        frame.index = frame.index.astype(str)
        if frame.index.duplicated().any() or set(frame.index) != set(ids):
            raise ValueError("Labels do not exactly cover their sample role.")
        labels.append(frame.loc[ids])
    columns = labels[0].columns.tolist()
    if set(columns) != set(labels[1].columns):
        raise ValueError("Train/dev label columns differ.")
    yt, yd = labels[0].to_numpy(), labels[1][columns].to_numpy()
    feature_manifest = read_json(args.features / "features.json")
    create_run(args.output, {"purpose": "dev hyperparameter selection; all reported dev metrics are selection/descriptive, not final generalization",
        "task": args.task, "seed": 42, "train_ids": train_ids, "dev_ids": dev_ids, "label_columns": columns,
        "train_dev_group_crossings": sorted({s.group_id for s in train}&{s.group_id for s in dev}),
        "runtime": runtime_manifest(), "cpu_role": "explicit statistical model fitting; neural inference remains Slurm CUDA",
        "source_hashes": {str(p): file_hash(p) for folder in [args.train, args.dev] for p in folder.glob("*.csv")},
        "feature_hashes": {name: file_hash(args.features / f"{name}.npz") for name in feature_manifest},
        "source_code_sha256": file_hash(Path(__file__)), "test_used": False})
    selection, direct = {}, {}
    permutation = np.random.default_rng(42).permutation(len(dev))
    try:
        with threadpool_limits(limits=1):
            for name, metadata in feature_manifest.items():
                with np.load(args.features / f"{name}.npz", allow_pickle=False) as data:
                    ids, feature_columns = data["sample_ids"].tolist(), data["column_ids"].tolist()
                    if len(ids) != len(set(ids)) or set(ids) != set(train_ids+dev_ids):
                        raise ValueError("Feature coverage differs; no implicit row intersection.")
                    lookup = {identifier: i for i, identifier in enumerate(ids)}
                    xt = data["values"][[lookup[i] for i in train_ids]]
                    xd = data["values"][[lookup[i] for i in dev_ids]]
                folder = args.output / name
                folder.mkdir()
                if args.task == "multilabel" and metadata["direct_anchor_scores"]:
                    if set(feature_columns) != set(columns):
                        raise ValueError("Direct scores are not the declared label concepts.")
                    values = xd[:, [feature_columns.index(c) for c in columns]]
                    direct[name] = {"multilabel": evaluate_multilabel(yd, values, columns),
                        "within_sample_candidates": evaluate_candidates(dict(zip(dev_ids, yd)), dict(zip(dev_ids, values))),
                        "shuffled_labels": evaluate_multilabel(yd[permutation], values, columns)}
                    write_json(folder / "direct-dev.json", direct[name])
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always", ConvergenceWarning)
                    probe = fit_linear_probe(xt, yt, xd, yd, task=args.task, seed=42)
                convergence = [str(w.message) for w in caught if issubclass(w.category, ConvergenceWarning)]
                if convergence:
                    write_json(folder / "convergence-failure.json", convergence)
                    raise RuntimeError("Probe convergence failed; do not accept an unconverged grid.")
                predictions = probe.predict_scores(xd)
                metric = evaluate_multilabel(yd, predictions, columns) if args.task == "multilabel" else evaluate_vad(yd, predictions, columns)
                selection[name] = {"best_parameter": probe.best_parameter, "selection_score": probe.dev_score,
                    "trials": probe.trials, "dev_metric": metric, "feature_columns": feature_columns,
                    "fitted_on": "train only; dev selects hyperparameter; no refit on dev"}
                joblib.dump(probe, folder / "probe.joblib")
                np.savez_compressed(folder / "dev-predictions.npz", sample_ids=np.array(dev_ids),
                                    label_columns=np.array(columns), values=predictions)
                write_json(folder / "selection.json", selection[name])
                append_event(args.output, "method_selected", method=name, parameter=probe.best_parameter)
                print(name, probe.best_parameter, probe.dev_score, flush=True)
        write_json(args.output / "selection.json", selection)
        write_json(args.output / "direct-dev.json", direct)
        write_json(args.output / "shuffled-label-control.json", {"seed": 42, "row_permutation": permutation.tolist(),
            "interpretation": "label-shuffle diagnostic; not a source-exchangeable significance test"})
        append_event(args.output, "completed")
    except BaseException as exc:
        append_event(args.output, "failed", error=repr(exc))
        raise


if __name__ == "__main__":
    main()
