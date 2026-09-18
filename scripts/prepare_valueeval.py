"""Convert the official pinned HF release via the existing adapter and audit all splits."""
from __future__ import annotations

import argparse
import ast
import hashlib
import re
from pathlib import Path

import numpy as np
import pandas as pd

from vlanchor.campaign import save_bundle
from vlanchor.datasets.common import DatasetBundle, build_manifest
from vlanchor.datasets.text import VALUEEVAL_LABELS, load_valueeval
from vlanchor.io import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    matched = re.search(r"^\s*labels = (\[.*\])", (args.raw / "README.md").read_text(), re.M)
    if matched is None or tuple(ast.literal_eval(matched.group(1))) != VALUEEVAL_LABELS:
        raise ValueError("Cannot verify official binary label column order.")
    converted = args.output / "converted"
    converted.mkdir()
    bundles, sources = [], []
    for name, split in [("train", "train"), ("validation", "dev"), ("test", "test")]:
        path = args.raw / f"{name}.parquet"
        frame = pd.read_parquet(path)
        labels = np.stack(frame["Labels"])
        if labels.shape != (len(frame), 20) or not np.isin(labels, [0, 1]).all():
            raise ValueError("Invalid official label matrix.")
        argument_file, label_file = converted / f"arguments-{name}.tsv", converted / f"labels-{name}.tsv"
        frame.drop(columns="Labels").to_csv(argument_file, sep="\t", index=False)
        gold = pd.DataFrame(labels, columns=VALUEEVAL_LABELS)
        gold.insert(0, "Argument ID", frame["Argument ID"])
        gold.to_csv(label_file, sep="\t", index=False)
        bundles.append(load_valueeval(argument_file, label_file, split=split))
        sources.append(path)
    samples = [s for b in bundles for s in b.samples]
    labels = pd.concat([b.labels for b in bundles])
    manifest = build_manifest("valueeval", sources, samples,
                              source_url="https://huggingface.co/datasets/webis/Touche23-ValueEval/tree/1259eea6eb37980172311bc02ac5ea88e82b9e42",
                              license_note="HF card CC-BY-4.0; original release license discrepancy unresolved; no raw redistribution",
                              official_splits={s.id: s.metadata["split"] for s in samples},
                              label_columns=list(VALUEEVAL_LABELS), source_format="official HF converted parquet",
                              original_tsv_retrieval_status="Zenodo HTTP504; not byte-identical to originals",
                              adapter_manifests=[b.manifest for b in bundles])
    bundle = DatasetBundle(samples, labels, manifest)
    save_bundle(bundle, args.output / "official")
    groups = {}
    for sample in samples:
        if sample.metadata["split"] == "train":
            groups.setdefault(sample.group_id, []).append(sample)
    order = sorted(groups, key=lambda g: hashlib.sha256(f"42\0{g}".encode()).hexdigest())
    remaining = iter(order)
    roles = {}
    for role, count in [("reference", 64), ("search", 256)]:
        selected = []
        while len(selected) < count:
            selected.extend(groups[next(remaining)])
        roles[role] = selected
    roles["probe_train"] = [s for g in remaining for s in groups[g]]
    for role, subset in roles.items():
        folder = args.output / role
        folder.mkdir()
        write_json(folder / "samples.json", subset)
        labels.loc[[s.id for s in subset]].to_csv(folder / "labels.csv")
    summary = {"split_counts": {k: sum(s.metadata["split"] == k for s in samples) for k in ["train", "dev", "test"]},
               "official_crossing_groups": len(manifest["groups_crossing_official_splits"]),
               "train_role_counts": {k: len(v) for k, v in roles.items()},
               "test_evaluations": 0}
    write_json(args.output / "summary.json", summary)
    print(summary)


if __name__ == "__main__":
    main()
