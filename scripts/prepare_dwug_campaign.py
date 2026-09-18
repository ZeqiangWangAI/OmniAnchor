"""Freeze official DWUG3 usages, target-group splits and explicit source document IDs."""
import argparse
import hashlib
import re
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

from omnianchor.datasets import load_dwug
from omnianchor.io import write_json
from omnianchor.provenance import file_hash


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    samples, labels, manifests = [], [], []
    for path in sorted((args.data / "data").glob("*/uses.csv")):
        bundle = load_dwug(path, path.with_name("judgments.csv"))
        for sample in bundle.samples:
            match = re.fullmatch(r"(.+\.txt)-\d+-\d+", sample.id)
            # Document filenames are encoded in published identifiers; no invented unit.
            unit = match.group(1) if match else None
            samples.append(sample.model_copy(update={"metadata": {**sample.metadata,
                "sampling_unit": unit, "sampling_unit_rule": "published identifier before .txt-position-token suffix"}}))
        labels.append(bundle.labels)
        manifests.append(bundle.manifest)
    if len({s.id for s in samples}) != len(samples):
        raise ValueError("Usage IDs repeat across lemmas; require an explicit namespaced join.")
    pairs = pd.concat(labels, ignore_index=True)
    id_split = {s.id: s.metadata["split"] for s in samples}
    if any(id_split[a] != id_split[b] for a, b in pairs[["identifier1", "identifier2"]].itertuples(index=False, name=None)):
        raise ValueError("A judgment pair crosses target-group splits.")
    stats_path = args.data / "stats/opt/stats_groupings.csv"
    stats = pd.read_csv(stats_path, sep="\t")
    for split in ["train", "dev", "test"]:
        folder = args.output / split
        folder.mkdir()
        subset = [s for s in samples if s.metadata["split"] == split]
        ids = {s.id for s in subset}
        write_json(folder / "samples.json", subset)
        pairs[pairs.identifier1.isin(ids)].to_csv(folder / "judgments.csv", index=False)
        stats[stats.lemma.isin({s.group_id for s in subset})].to_csv(folder / "change-labels.csv", index=False)
    smoke = []
    for split in ["train", "dev"]:
        candidates = sorted([s for s in samples if s.metadata["split"] == split],
            key=lambda s: hashlib.sha256(f"42\0{s.id}".encode()).hexdigest())
        # Round-robin over target/time cells, without reading human judgments.
        cells = defaultdict(list)
        for sample in candidates:
            cells[(sample.group_id, sample.time)].append(sample)
        chosen = []
        while len(chosen) < 16:
            for key in sorted(cells):
                if cells[key] and len(chosen) < 16:
                    chosen.append(cells[key].pop(0))
            if not any(cells.values()) and len(chosen) < 16:
                raise ValueError("Insufficient train/dev usages.")
        smoke.extend(chosen)
    write_json(args.output / "smoke32.json", smoke)
    source_splits = defaultdict(set)
    for sample in samples:
        if sample.metadata["sampling_unit"]:
            source_splits[sample.metadata["sampling_unit"]].add(sample.metadata["split"])
    write_json(args.output / "manifest.json", {"dataset": "DWUG_EN", "version": "3.0.0",
        "split_strategy": "HANDOFF no-official-split grouped SHA25642 60/20/20 by lemma",
        "sample_counts": dict(Counter(s.metadata["split"] for s in samples)),
        "target_counts": {split: len({s.group_id for s in samples if s.metadata["split"] == split}) for split in ["train", "dev", "test"]},
        "source_documents_crossing_target_splits": sorted(k for k, v in source_splits.items() if len(v)>1),
        "unknown_sampling_unit_ids": [s.id for s in samples if s.metadata["sampling_unit"] is None],
        "adapters": manifests, "change_labels_sha256": file_hash(stats_path),
        "smoke_selection": "16train16dev; round-robin target/time cells, within-cell hash42 usage ID",
        "time_in_prompt": False, "labels_used_for_selection": False,
        "anchors_reference": "WiC train frozen independently; no DWUG label-based anchor selection"})
    print(Counter(s.metadata["split"] for s in samples))


if __name__ == "__main__":
    main()
