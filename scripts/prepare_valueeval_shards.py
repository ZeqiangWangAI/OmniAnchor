"""Freeze fixed-bridge probe-train/dev inputs into 1000-material shards; no test inputs."""
import argparse
from pathlib import Path

from vlanchor.io import load_samples, write_json
from vlanchor.provenance import file_hash


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    train = load_samples(args.data / "probe_train/samples.json")
    dev = load_samples(args.data / "official/dev/samples.json")
    if {s.group_id for s in train} & {s.group_id for s in dev}:
        raise ValueError("Probe train/dev source groups overlap.")
    samples = train + dev
    if len({s.id for s in samples}) != len(samples) or any(s.metadata["split"] not in {"train", "dev"} for s in samples):
        raise ValueError("Duplicate or protected samples in shard preparation.")
    files = []
    for offset in range(0, len(samples), 1000):
        subset = samples[offset:offset + 1000]
        path = args.output / f"samples-{offset // 1000:05d}.json"
        write_json(path, subset)
        files.append({"file": path.name, "sha256": file_hash(path), "sample_ids": [s.id for s in subset]})
    write_json(args.output / "manifest.json", {"train_samples": len(train), "dev_samples": len(dev),
        "test_samples": 0, "files": files, "role": "fixed3 probe features; reference/search scored separately",
        "source_hashes": {str(p): file_hash(p) for p in [args.data / "probe_train/samples.json", args.data / "official/dev/samples.json"]}})
    print(len(samples), len(files))


if __name__ == "__main__":
    main()
