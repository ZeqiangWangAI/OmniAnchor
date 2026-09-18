"""Recompute final AP or affect correlations in R (legacy multilabel filename)."""
import argparse
import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd

from vlanchor.campaign import append_event, create_run
from vlanchor.io import read_json, write_json
from vlanchor.provenance import file_hash


def read_frozen_feature_info(contract, mirror=None):
    original_info = Path(contract["probe_features"])/"features.json"
    info_path = mirror/"features.json" if mirror else original_info
    if file_hash(info_path) != contract["frozen_artifacts"][str(original_info)]:
        raise ValueError("Frozen feature metadata changed; local mirrors must preserve bytes.")
    return read_json(info_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["analysis", "data", "contract", "output"]:
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--rscript", default="Rscript")
    parser.add_argument("--features", type=Path, help="Local mirror of frozen feature metadata; original hash is required")
    args = parser.parse_args()
    contract = read_json(args.contract)
    manifest = read_json(args.analysis/"manifest.json")
    events = (args.analysis/"events.jsonl").read_text().splitlines()
    if (contract["status"] != "frozen" or contract["task"] not in {"multilabel", "regression"}
            or manifest["contract_sha256"] != file_hash(args.contract)
            or not events or json.loads(events[-1])["status"] != "completed"):
        raise ValueError("Require a completed frozen final analysis.")
    if file_hash(args.data/"labels.csv") != contract["labels_sha256"]:
        raise ValueError("Original final labels changed.")
    columns = contract["label_columns"]
    metrics = pd.read_csv(args.analysis/"metrics.csv")
    info = read_frozen_feature_info(contract, args.features)
    expected = {("probe", name) for name in info}
    metric_names = ["spearman_"+c for c in columns]
    if contract["task"] == "multilabel":
        expected |= {("direct", name) for name, definition in info.items() if definition["direct_anchor_scores"]}
        metric_names = ["macro_ap"]
    expected_rows = {(kind, name, metric) for kind, name in expected for metric in metric_names}
    if set(zip(metrics.kind, metrics.method, metrics.metric)) != expected_rows or len(metrics) != len(expected_rows):
        raise ValueError("Require all declared direct and predictor methods.")
    rsource = Path(__file__).with_suffix(".R")
    create_run(args.output, {"purpose": "Independent base-R outcome metrics/ID/tie handling, without model inference or probe refitting",
        "analysis_metrics_sha256": file_hash(args.analysis/"metrics.csv"),
        "contract_sha256": file_hash(args.contract), "labels_sha256": file_hash(args.data/"labels.csv"),
        "R_source_sha256": file_hash(rsource), "exporter_sha256": file_hash(Path(__file__)),
        "transfer": "IEEE754 little-endian float64 row-major; no decimal score roundtrip",
        "tolerance": 1e-12})
    try:
        (args.output/"column-ids.txt").write_text("\n".join(columns)+"\n")
        labels = pd.read_csv(args.data/"labels.csv", index_col="sample_id")
        labels.index = labels.index.astype(str)
        if labels.index.duplicated().any() or set(labels.index) != set(contract["sample_ids"]) or set(labels.columns) != set(columns):
            raise ValueError("Canonical label population differs.")
        labels[columns].to_numpy(dtype="<f8").tofile(args.output/"labels.f64")
        pd.DataFrame({"sample_id": labels.index}).to_csv(args.output/"labels.index.csv", index=False)
        hashes = {}
        for row in metrics.drop_duplicates(["kind", "method"]).itertuples():
            suffix = "predictions" if row.kind == "probe" else "features"
            path = args.analysis/f"{row.method}-{suffix}.npz"
            with np.load(path, allow_pickle=False) as data:
                coords = data["label_columns" if suffix == "predictions" else "column_ids"].tolist()
                ids = data["sample_ids"].tolist()
                if (len(ids) != len(set(ids)) or set(ids) != set(contract["sample_ids"])
                        or len(coords) != len(set(coords)) or set(coords) != set(columns)):
                    raise ValueError("Score population or coordinate coverage differs.")
                values = data["values"][:, [coords.index(c) for c in columns]].astype("<f8")
                if values.shape != (len(ids), len(columns)) or not np.isfinite(values).all():
                    raise ValueError("Malformed scores; no silent exclusions.")
                values.tofile(args.output/f"{row.kind}--{row.method}.f64")
                pd.DataFrame({"sample_id": ids}).to_csv(args.output/f"{row.kind}--{row.method}.index.csv", index=False)
            hashes[str(path)] = file_hash(path)
        write_json(args.output/"input-hashes.json", hashes)
        result = subprocess.run([args.rscript, str(rsource), str(args.output), str(args.data/"labels.csv"),
            str(args.analysis/"metrics.csv"), "binary64"], capture_output=True, text=True)
        (args.output/"R-stdout.txt").write_text(result.stdout)
        (args.output/"R-stderr.txt").write_text(result.stderr)
        result.check_returncode()
        verified = pd.read_csv(args.output/"verification.csv")
        if len(verified) != len(expected_rows) or verified.max_error.max() >= 1e-12:
            raise ValueError("Independent verification coverage or tolerance failed.")
        append_event(args.output, "completed", comparisons=len(verified), max_error=float(verified.max_error.max()))
    except BaseException as exc:
        append_event(args.output, "failed", error=repr(exc))
        raise


if __name__ == "__main__":
    main()
