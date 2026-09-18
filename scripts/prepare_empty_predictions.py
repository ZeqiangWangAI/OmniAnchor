"""Freeze no-content features and existing-probe predictions without reading final labels."""
import argparse
from pathlib import Path

import joblib
import numpy as np

from omnianchor import to_matrix, transform
from omnianchor.campaign import create_run, append_event, merge_score_shards
from omnianchor.io import load_calibration, load_scores, read_json, write_json
from omnianchor.provenance import file_hash


def empty_paths(study, native_run, baseline_run):
    if study is not None:
        if native_run is not None or baseline_run is not None:
            raise ValueError("Choose one empty-run layout, not both.")
        roots, native, baselines = [study.parent], study/"native", study
    else:
        if native_run is None or baseline_run is None:
            raise ValueError("Both separate no-content runs are required.")
        roots, native, baselines = [native_run, baseline_run], native_run/"measurement", baseline_run
    if any((root/"exit_code.txt").read_text().strip() != "0" for root in roots):
        raise ValueError("No-content scoring is incomplete or failed.")
    return native, baselines


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--empty-study", type=Path)
    parser.add_argument("--empty-native-run", type=Path)
    parser.add_argument("--empty-baseline-run", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    c = read_json(args.contract)
    if c["status"] != "frozen":
        raise ValueError("Require frozen fitted probes and a completed no-content run.")
    for path, digest in c["frozen_artifacts"].items():
        if file_hash(path) != digest:
            raise ValueError("Fitted artifact changed after freeze.")
    native_folder, baseline_root = empty_paths(args.empty_study, args.empty_native_run, args.empty_baseline_run)
    reference, probes = Path(c["probe_features"]), Path(c["probes"])
    paths = sorted(native_folder.glob("part-*.parquet"))
    raw = merge_score_shards([load_scores(p) for p in paths], ["control:empty"])
    source_sample = raw.manifest["samples"][0]
    if any(p.get("path") or p.get("text") for p in source_sample["parts"]):
        raise ValueError("The purported no-content input contains material.")
    calibrated = transform(raw, load_calibration(reference/"calibration.json"))
    features = {}
    for variant in ["raw_logp", "reference_log_ratio", "reference_z"]:
        matrix = to_matrix(calibrated, variant=variant, missing="error")
        columns = c["feature_columns"]["native-"+variant]
        features["native-"+variant] = matrix.values[:, [matrix.anchor_ids.index(i) for i in columns]]
    for method in c.get("baseline_methods", ["e5", "qwen-embedding", "qwen-reranker"]):
        folder = baseline_root/method
        current, previous = read_json(folder/"manifest.json"), read_json(reference/f"{method}-identity.json")
        for key in ["method", "model_id", "revision", "spec_sha256", "instruction", "precision", "preprocessing", "vision_reuse"]:
            if current.get(key) != previous.get(key):
                raise ValueError(f"No-content feature identity changed: {method}/{key}")
        path = folder/"matrix.npz"
        paths.extend([path, folder/"manifest.json"])
        with np.load(path, allow_pickle=False) as data:
            if data["sample_ids"].tolist() != ["control:empty"]:
                raise ValueError("Require one actual no-content input.")
            anchors = data["anchor_ids"].tolist()
            columns = c["feature_columns"][method+"-anchor"]
            features[method+"-anchor"] = data["scores"][:, [anchors.index(a) for a in columns]]
            if method+"-original" in c["feature_columns"]:
                features[method+"-original"] = data["features"].copy()
    if set(features) != set(c["feature_columns"]):
        raise ValueError("Missing no-content methods.")
    create_run(args.output, {"purpose": "Frozen no-content controls from completed scoring, no final labels read",
        "scoring_contract_sha256": file_hash(args.contract), "empty_native": str(native_folder), "empty_baselines": str(baseline_root),
        "source_sha256": {str(p): file_hash(p) for p in paths},
        "script_sha256": file_hash(Path(__file__)), "test_labels_read": False,
        "prediction": "Apply already fitted train-only scaler/probe to one empty feature row, never refit. Repeat that one prediction for evaluation so a constant predictor cannot acquire rounding-induced correlations."})
    info = read_json(reference/"features.json")
    for name, values in features.items():
        if values.shape != (1, len(c["feature_columns"][name])) or not np.isfinite(values).all():
            raise ValueError("Invalid no-content feature vector.")
        prediction = joblib.load(probes/name/"probe.joblib").predict_scores(values)
        if prediction.shape != (1, len(c["label_columns"])) or not np.isfinite(prediction).all():
            raise ValueError("Invalid no-content fitted prediction.")
        np.savez_compressed(args.output/f"{name}.npz", features=values, predictions=prediction,
            column_ids=np.array(c["feature_columns"][name]), label_columns=np.array(c["label_columns"]))
    write_json(args.output/"features.json", info)
    append_event(args.output, "completed")


if __name__ == "__main__":
    main()
