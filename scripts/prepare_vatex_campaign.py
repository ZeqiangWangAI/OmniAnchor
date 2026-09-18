"""Freeze first500 available VATEX clips and source-disjoint dev/evaluation roles."""
import argparse
import hashlib
import os
from pathlib import Path

from vlanchor.datasets import load_vatex
from vlanchor.io import read_json, write_json
from vlanchor.provenance import file_hash


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--acquisition", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    selected = read_json(args.acquisition / "selected.json")
    if len(selected) != 500 or len({r["videoID"] for r in selected}) != 500:
        raise ValueError("Require500 unique frozen available clips.")
    annotations = {r["videoID"]: r for r in read_json(args.annotations)}
    args.output.mkdir(parents=True, exist_ok=False)
    media = args.output / "media"
    media.mkdir()
    roles = {}
    for row in selected:
        if file_hash(row["path"]) != row["sha256"]:
            raise ValueError("Acquired clip bytes changed.")
        os.link(row["path"], media / (row["videoID"] + ".mp4"))
        source = row["videoID"].rsplit("_", 2)[0]
        roles[row["videoID"]] = "dev" if int.from_bytes(hashlib.sha256(f"42\0{source}".encode()).digest()[:8], "big")/2**64 < .2 else "test"
    converted = args.output / "selected-annotations.json"
    write_json(converted, [annotations[r["videoID"]] for r in selected])
    manifests = {}
    for language in ["en", "zh"]:
        bundle = load_vatex(converted, media, language=language, split="validation")
        for split in ["dev", "test"]:
            folder = args.output / language / split
            folder.mkdir(parents=True)
            samples = []
            for sample in bundle.samples:
                clip = (sample.pair_id or sample.id).split(":")[1]
                if roles[clip] != split:
                    continue
                parts = tuple(p.model_copy(update={"path": f"../../media/{clip}.mp4"}) if p.type == "video" else p for p in sample.parts)
                samples.append(sample.model_copy(update={"parts": parts, "metadata": {**sample.metadata,
                    "split": split, "upstream_split": "validation", "videoID": clip}}))
            write_json(folder / "samples.json", samples)
            ids = {s.id for s in samples}
            bundle.labels[bundle.labels.video_id.isin(ids)].to_csv(folder / "relevance.csv", index=False)
        manifests[language] = bundle.manifest
    write_json(args.output / "manifest.json", {"dataset": "VATEX", "upstream_split": "validation",
        "selection": "first500 available by videoID; acquired507 after640 attempts;7overshoot unselected",
        "roles": roles, "dev_videos": sum(v=="dev" for v in roles.values()), "evaluation_videos": sum(v=="test" for v in roles.values()),
        "source_role_rule": "SHA256(42 NUL youtubeID) first64bits /2**64 <.2 dev else protected final evaluation",
        "input_rule": "videos have no captions; English/Chinese caption inputs are separate retrieval queries",
        "annotations_sha256": file_hash(args.annotations), "availability_sha256": file_hash(args.acquisition / "availability.json"),
        "selected_sha256": file_hash(args.acquisition / "selected.json"), "adapter_manifests": manifests,
        "video_sha256": {r["videoID"]: r["sha256"] for r in selected}, "frames": 8,
        "upstream_validation_role_override": "explicit source-disjoint local dev/test replaces adapter validation metadata"})
    print({s: sum(v==s for v in roles.values()) for s in ["dev", "test"]})


if __name__ == "__main__":
    main()
