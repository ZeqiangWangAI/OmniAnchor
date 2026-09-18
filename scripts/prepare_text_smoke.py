"""Freeze complete affect splits and select exactly 32 label-blind train/dev materials."""
from __future__ import annotations

import argparse
from pathlib import Path

from vlanchor.campaign import save_bundle, select_smoke
from vlanchor.datasets import load_chinese_emobank, load_emobank
from vlanchor.io import write_json
from vlanchor.provenance import file_hash


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    en = load_emobank(args.raw / "emobank/emobank.csv", rating_perspective="reader",
                      individual_ratings_path=args.raw / "emobank/individual_reader_ratings.csv")
    zh = load_chinese_emobank(args.raw / "chinese_emobank/CVAS_all.csv", text_column="Text")
    summary = {}
    for name, bundle in [("emobank-reader", en), ("chinese-emobank-sentence", zh)]:
        save_bundle(bundle, args.output / name)
        smoke = select_smoke(bundle.samples)
        write_json(args.output / name / "smoke-samples.json", smoke)
        bundle.labels.loc[[s.id for s in smoke]].to_csv(args.output / name / "smoke-labels.csv")
        summary[name] = {"n": len(bundle.samples), "smoke_n": len(smoke),
                         "split_counts": {k: sum(s.metadata["split"] == k for s in bundle.samples)
                                          for k in ["train", "dev", "test"]},
                         "groups_crossing_official_splits": len(bundle.manifest["groups_crossing_official_splits"]),
                         "smoke_sha256": file_hash(args.output / name / "smoke-samples.json")}
    write_json(args.output / "summary.json", summary)
    print(summary)


if __name__ == "__main__":
    main()
