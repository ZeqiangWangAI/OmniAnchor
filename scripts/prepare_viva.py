"""Freeze available VIVA rows; explicitly exclude ambiguous candidates before scoring."""
import argparse
import json
from collections import Counter
from pathlib import Path

from omnianchor.datasets import load_viva
from omnianchor.io import write_json
from omnianchor.provenance import file_hash


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--availability", type=Path, required=True)
    parser.add_argument("--media-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    rows = json.loads(args.annotations.read_text())
    available = {r["image_file"]: r for r in json.loads(args.availability.read_text()) if r["status"] == "available"}
    retained, excluded = [], []
    for row in rows:
        reasons = []
        if row["image_file"] not in available:
            reasons.append("image_unavailable")
        actions = row["action_list"]
        by_letter = {str(a).split(".", 1)[0].strip(): str(a).split(".", 1)[1].strip()
                     for a in actions if "." in str(a)}
        if len(by_letter) != len(actions) or str(row["answer"]).strip() not in by_letter:
            reasons.append("missing_or_malformed_correct_action")
        if not row["values"]["positive"] or not row["values"]["negative"]:
            reasons.append("no_positive_or_no_negative_candidate")
        surfaces = [str(s).split(":", 1)[0].strip().casefold()
                    for kind in ["positive", "negative"] for s in row["values"][kind]]
        duplicates = sorted(s for s, count in Counter(surfaces).items() if count > 1)
        if duplicates:
            reasons.append("duplicate_or_conflicting_canonical_candidates")
        if reasons:
            excluded.append({"index": row["index"], "reasons": reasons, "duplicate_surfaces": duplicates})
        else:
            retained.append(row)
    filtered = args.output / "available-unambiguous-annotations.json"
    write_json(filtered, retained)
    bundle = load_viva(filtered, args.media_root, require_media=True)
    bundle.manifest.update(original_annotations_sha256=file_hash(args.annotations),
                           availability_sha256=file_hash(args.availability), exclusions=excluded,
                           selection_rule="decodable image; valid annotated correct action; unique candidates with positive and negative labels; no score-based exclusion")
    bundle.manifest["media"]["media_sha256"] = {r["image_file"]: r["sha256"] for r in available.values()}
    write_json(args.output / "manifest.json", bundle.manifest)
    for split in ["train", "dev", "test"]:
        subset = [s for s in bundle.samples if s.metadata["split"] == split]
        folder = args.output / split
        folder.mkdir()
        write_json(folder / "samples.json", subset)
        write_json(folder / "candidates.json", {s.id: bundle.candidates[s.id] for s in subset})
        bundle.labels.loc[[s.id for s in subset]].to_csv(folder / "labels.csv")
    summary = {"original": len(rows), "retained": len(retained), "excluded": len(excluded),
               "exclusion_reasons": dict(Counter(reason for r in excluded for reason in r["reasons"])),
               "split_counts": {k: sum(s.metadata["split"] == k for s in bundle.samples) for k in ["train", "dev", "test"]}}
    write_json(args.output / "summary.json", summary)
    print(summary)


if __name__ == "__main__":
    main()
