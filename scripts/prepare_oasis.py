"""Explicitly reconcile two trailing-space filenames in the published OASIS archive."""
import argparse
from pathlib import Path

from vlanchor.campaign import save_bundle
from vlanchor.datasets import load_oasis
from vlanchor.datasets.common import read_table
from vlanchor.io import write_json
from vlanchor.provenance import file_hash


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    data = read_table(args.raw / "OASIS.csv")
    corrections = []
    filenames = []
    for theme in data["Theme"]:
        original = f"{theme}.jpg"
        resolved = original
        if not (args.raw / "images" / original).is_file():
            resolved = f"{theme.rstrip()}.jpg"
            if resolved == original or not (args.raw / "images" / resolved).is_file():
                raise FileNotFoundError(original)
            corrections.append({"original": original, "resolved": resolved, "reason": "trailing whitespace in published Theme"})
        filenames.append(resolved)
    data["filename"] = filenames
    # Keep the published item number, not an inferred zero-based CSV row index.
    data["Item"] = data["Unnamed: 0"]
    converted = args.output / "OASIS-with-filenames.csv"
    data.to_csv(converted, index=False)
    bundle = load_oasis(converted, args.raw / "images", filename_column="filename")
    hashes = {s.id: file_hash(s.parts[0].path) for s in bundle.samples}
    if len(set(hashes.values())) != len(hashes):
        raise ValueError("Duplicate images need a group audit before split freezing.")
    bundle.manifest.update(original_metadata_sha256=file_hash(args.raw / "OASIS.csv"),
                           filename_corrections=corrections, published_item_id_column="Unnamed: 0")
    bundle.manifest["media"]["media_sha256"] = hashes
    save_bundle(bundle, args.output / "prepared")
    write_json(args.output / "summary.json", {"samples": len(bundle.samples), "unique_image_hashes": len(set(hashes.values())),
                                               "filename_corrections": corrections})
    print(len(bundle.samples), corrections)


if __name__ == "__main__":
    main()
