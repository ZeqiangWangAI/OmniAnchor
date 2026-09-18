"""Stage four OASIS train/dev images and four hash-reserved VATEX dev clips, without labels."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

from vlanchor.campaign import select_smoke
from vlanchor.io import load_samples, write_json
from vlanchor.provenance import file_hash
from vlanchor.types import Part, Sample


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oasis", type=Path, required=True)
    parser.add_argument("--vatex-availability", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    images = select_smoke(load_samples(args.oasis / "train/samples.json") +
                          load_samples(args.oasis / "dev/samples.json"), 2)
    availability_bytes = args.vatex_availability.read_bytes()
    clips = json.loads(availability_bytes)
    (args.output / "availability-snapshot.json").write_bytes(availability_bytes)
    # Freeze this source-level role rule before any video model calls or caption inspection.
    videos = [r for r in clips if r["status"] == "available" and
              int.from_bytes(hashlib.sha256(f"42\0{r['videoID'].rsplit('_', 2)[0]}".encode()).digest()[:8], "big") / 2**64 < .2][:4]
    if len(videos) < 4:
        raise ValueError("Insufficient preassigned dev clips; do not borrow evaluation videos.")
    media = args.output / "media"
    media.mkdir()
    staged = []
    origins = []
    for sample in images:
        source = Path(sample.parts[0].path)
        name = f"{sample.id.replace(':', '-')}.jpg"
        shutil.copyfile(source, media / name)
        staged.append(sample.model_copy(update={"parts": (Part(type="image", path=f"media/{name}"),)}))
        origins.append({"id": sample.id, "original": str(source), "sha256": file_hash(source)})
    for record in videos:
        source = Path(record["path"])
        name = f"{record['videoID']}.mp4"
        shutil.copyfile(source, media / name)
        staged.append(Sample(id=f"vatex:{record['videoID']}", source="vatex", group_id=record["videoID"].rsplit("_", 2)[0],
                             parts=(Part(type="video", path=f"media/{name}"),), metadata={"split": "dev"}))
        origins.append({"id": staged[-1].id, "original": str(source), "sha256": file_hash(source)})
    write_json(args.output / "samples.json", staged)
    write_json(args.output / "manifest.json", {"purpose": "real_media_smoke_not_scientific_validity", "origins": origins,
        "vatex_source_role_rule": "SHA256(42 NUL youtubeID) first64bits /2**64 <.2 =>dev; remainder protected evaluation",
        "vatex_used_dev_ids": [r["videoID"] for r in videos], "all_vatex_evaluation_must_exclude_these_sources": True,
        "availability_snapshot_sha256": hashlib.sha256(availability_bytes).hexdigest(), "frames": 8, "labels_read": False})
    print([s.id for s in staged])


if __name__ == "__main__":
    main()
