"""Freeze grouped train-only reference and remaining official train/dev emotion inputs."""
import argparse
import hashlib
import os
import shutil
from collections import defaultdict
from pathlib import Path

import pandas as pd

from omnianchor.io import load_samples, write_json
from omnianchor.campaign import select_smoke
from omnianchor.provenance import file_hash


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--copy-media", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    origins = {}

    def save_samples(path, samples):
        portable = []
        for sample in samples:
            parts = []
            for part in sample.parts:
                if part.path and args.copy_media:
                    source = Path(part.path)
                    digest = file_hash(source)
                    destination = args.output / "media" / (digest + source.suffix)
                    if not destination.exists():
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copyfile(source, destination)
                    origins[str(source)] = {"sha256": digest, "copied_path": str(destination.relative_to(args.output))}
                    part = part.model_copy(update={"path": os.path.relpath(destination, path.parent)})
                parts.append(part)
            portable.append(sample.model_copy(update={"parts": tuple(parts)}))
        write_json(path, portable)
    train, dev, test = [load_samples(args.data / split / "samples.json") for split in ["train", "dev", "test"]]
    protected_groups = {s.group_id or s.id for s in dev+test}
    grouped = defaultdict(list)
    for sample in train:
        if (sample.group_id or sample.id) not in protected_groups:
            grouped[sample.group_id or sample.id].append(sample)
    reference, selected_groups = [], set()
    for group in sorted(grouped, key=lambda g: hashlib.sha256(f"42\0{g}".encode()).hexdigest()):
        reference.extend(grouped[group])
        selected_groups.add(group)
        if len(reference) >= 64:
            break
    if len(reference) < 64:
        raise ValueError("Insufficient train-only reference groups.")
    probe = [s for s in train if (s.group_id or s.id) not in selected_groups]
    for role, samples, label_source in [("reference", reference, "train"), ("probe_train", probe, "train"), ("dev", dev, "dev")]:
        folder = args.output / role
        folder.mkdir()
        save_samples(folder / "samples.json", samples)
        labels = pd.read_csv(args.data / label_source / "labels.csv", index_col="sample_id")
        labels.index = labels.index.astype(str)
        labels.loc[[s.id for s in samples]].to_csv(folder / "labels.csv")
    samples = probe + dev
    files = []
    for offset in range(0, len(samples), 1000):
        path = args.output / "shards" / f"samples-{offset//1000:05d}.json"
        save_samples(path, samples[offset:offset+1000])
        files.append({"file": path.name, "sha256": file_hash(path), "sample_ids": [s.id for s in samples[offset:offset+1000]]})
    write_json(args.output / "shards/manifest.json", {"files": files, "train_samples": len(probe), "dev_samples": len(dev), "test_samples": 0})
    save_samples(args.output / "smoke32.json", select_smoke(probe+dev, 16))
    write_json(args.output / "media-origins.json", origins)
    write_json(args.output / "manifest.json", {"source_manifest_sha256": file_hash(args.data / "manifest.json") if (args.data / "manifest.json").exists() else None,
        "source_split_sha256": {split: file_hash(args.data / split / "samples.json") for split in ["train", "dev", "test"]},
        "reference_selection": "train source groups absent from dev/test; sorted by SHA25642 group; whole groups until >=64 rows",
        "reference_count": len(reference), "probe_train_count": len(probe), "dev_count": len(dev),
        "reference_group_ids": sorted(selected_groups), "search": "PMPO is evaluated on predefined ValueEval; no emotion-label bridge search",
        "probe_train_dev_group_crossings": sorted({s.group_id for s in probe}&{s.group_id for s in dev}),
        "scientific_role": "affect12 features predict VA/VAD via prespecified Ridge; not direct VAD axes",
        "test": "original official/local split unchanged and unscored"})
    print(len(reference), len(probe), len(dev), len(files))


if __name__ == "__main__":
    main()
