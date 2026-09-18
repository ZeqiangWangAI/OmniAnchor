"""Resolve exact image duplicates before VIVA inference, preserving the earlier split artifact."""
import argparse
from collections import defaultdict
from pathlib import Path

import pandas as pd

from omnianchor.datasets.common import grouped_hash_splits
from omnianchor.io import load_samples, read_json, write_json
from omnianchor.provenance import file_hash


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = read_json(args.source / "manifest.json")
    samples, candidates, frames = [], {}, []
    for split in ["train", "dev", "test"]:
        samples.extend(load_samples(args.source / split / "samples.json"))
        candidates.update(read_json(args.source / split / "candidates.json"))
        frames.append(pd.read_csv(args.source / split / "labels.csv"))
    labels = pd.concat(frames, ignore_index=True)
    hashes = manifest["media"]["media_sha256"]
    groups = defaultdict(list)
    for sample in samples:
        groups[hashes[sample.group_id]].append(sample.group_id)
    canonical = {name: min(names) for names in groups.values() for name in names}
    updated = [s.model_copy(update={"group_id": canonical[s.group_id], "metadata": {**s.metadata,
        "original_image_filename_group": s.group_id, "image_sha256": hashes[s.group_id]}}) for s in samples]
    splits = grouped_hash_splits(updated, seed=42)
    changes = [{"id": s.id, "old_split": s.metadata["split"], "new_split": splits[s.id]}
               for s in samples if s.metadata["split"] != splits[s.id]]
    updated = [s.model_copy(update={"metadata": {**s.metadata, "split": splits[s.id]}}) for s in updated]
    args.output.mkdir(parents=True, exist_ok=False)
    for split in ["train", "dev", "test"]:
        subset = [s for s in updated if splits[s.id] == split]
        folder = args.output / split
        folder.mkdir()
        write_json(folder / "samples.json", subset)
        write_json(folder / "candidates.json", {s.id: candidates[s.id] for s in subset})
        labels[labels.sample_id.isin({s.id for s in subset})].to_csv(folder / "labels.csv", index=False)
    manifest.update(splits=splits, groups_crossing_official_splits=[],
        parent_prepared_manifest_sha256=file_hash(args.source / "manifest.json"),
        split_strategy="grouped_sha256_60_20_20_exact_image_duplicates_canonical_filename",
        exact_duplicate_groups=[names for names in groups.values() if len(names)>1],
        split_changes_before_any_viva_inference=changes,
        selection_rule="decodable image; valid annotated correct action; unique positive/negative candidates; no model-based filtering")
    write_json(args.output / "manifest.json", manifest)
    write_json(args.output / "summary.json", {"retained": len(updated), "content_groups": len(groups),
        "split_counts": {split: sum(v==split for v in splits.values()) for split in ["train", "dev", "test"]},
        "split_changes": changes, "test_scores_seen": False})
    print(changes)


if __name__ == "__main__":
    main()
