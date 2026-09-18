"""Prepare explicit occurrence-marked DWUG inputs and train-only WiC reference."""
import argparse
import hashlib
from pathlib import Path

from vlanchor.campaign import mark_target
from vlanchor.datasets import load_wic
from vlanchor.io import load_samples, write_json
from vlanchor.provenance import file_hash


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dwug", type=Path, required=True)
    parser.add_argument("--wic-train", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    for split in ["train", "dev", "test"]:
        write_json(args.output / f"{split}.json", [mark_target(s) for s in load_samples(args.dwug / split / "samples.json")])
    write_json(args.output / "smoke32.json", [mark_target(s) for s in load_samples(args.dwug / "smoke32.json")])
    wic = load_wic(args.wic_train, split="train")
    ordered = sorted(wic.samples, key=lambda s: hashlib.sha256(f"42\0{s.id}".encode()).hexdigest())
    groups, reference = set(), []
    for sample in ordered:
        if sample.group_id not in groups:
            reference.append(mark_target(sample))
            groups.add(sample.group_id)
        if len(reference) == 64:
            break
    if len(reference) != 64:
        raise ValueError("Require64 train target groups in reference.")
    write_json(args.output / "reference64.json", reference)
    write_json(args.output / "manifest.json", {"rendering": "insert literal <target> and </target> around original occurrence; preserve raw text/span in metadata",
        "time_in_prompt": False, "reference_source": str(args.wic_train), "reference_source_sha256": file_hash(args.wic_train),
        "reference_selection": "first64 unique train lemmas by hash42 usage ID; no gold labels read",
        "dwug_manifest_sha256": file_hash(args.dwug / "manifest.json"), "test_role": "protected; no scoring yet"})


if __name__ == "__main__":
    main()
