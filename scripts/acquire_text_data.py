"""Acquire pinned public text releases; refuse to replace an acquisition directory."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

from vlanchor.io import write_json


def fetch(url: str, path: Path) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(Request(url, headers={"User-Agent": "VLanchor-research/0.1"}), timeout=45) as response:
        with path.open("xb") as output:
            while block := response.read(1024 * 1024):
                output.write(block)
    return {"url": url, "path": str(path), "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = {"started_utc": datetime.now(timezone.utc).isoformat(),
                "files": [], "failures": [], "status": "acquiring"}
    sources = []
    eb = "248ce2a43e165a66d31aeaed83cff9641d6654e0"
    cb = "eaf0ca0fd6ec9e48e701231eee490d58bd074c12"
    for name in ["emobank.csv", "individual_reader_ratings.csv", "reader.csv", "meta.tsv"]:
        sources.append((f"https://raw.githubusercontent.com/JULIELab/EmoBank/{eb}/corpus/{name}", f"emobank/{name}"))
    sources.append((f"https://raw.githubusercontent.com/JULIELab/EmoBank/{eb}/README.md", "emobank/README.md"))
    for folder, name in [("CVAS_SD", "CVAS_all.csv"), ("CVAT_SD", "CVAT_all_SD.csv")]:
        sources.append((f"https://raw.githubusercontent.com/NYCU-NLP/Chinese-EmoBank/{cb}/ChineseEmoBank/{folder}/{name}", f"chinese_emobank/{name}"))
    sources.append((f"https://raw.githubusercontent.com/NYCU-NLP/Chinese-EmoBank/{cb}/README.md", "chinese_emobank/README.md"))
    # Versioned record used by the official Hugging Face dataset loader.
    for split in ["training", "validation", "test"]:
        for kind in ["arguments", "labels"]:
            name = f"{kind}-{split}.tsv"
            sources.append((f"https://zenodo.org/records/7879430/files/{name}?download=1", f"valueeval/{name}"))
    sources.append(("https://zenodo.org/api/records/7879430", "valueeval/release.json"))
    for url, name in sources:
        try:
            record = fetch(url, args.output / name)
            manifest["files"].append(record)
            print(json.dumps({"acquired": name, "bytes": record["bytes"]}), flush=True)
        except Exception as exc:
            manifest["failures"].append({"url": url, "path": name, "error": repr(exc)})
            print(json.dumps(manifest["failures"][-1]), flush=True)
        write_json(args.output / "acquisition.json", manifest)
    manifest["status"] = "complete" if not manifest["failures"] else "incomplete"
    manifest["completed_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(args.output / "acquisition.json", manifest)
    if manifest["failures"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
