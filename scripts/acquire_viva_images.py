"""Fetch all published VIVA image URLs before model-independent availability freezing."""
import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from PIL import Image

from acquire_text_data import fetch
from vlanchor.datasets.common import local_media
from vlanchor.io import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    data = json.loads(args.annotations.read_text())
    unique = {}
    for row in data:
        name = row["image_file"]
        if name in unique and unique[name] != row["image_url"]:
            raise ValueError("Same image filename has different source URLs.")
        unique[name] = row["image_url"]

    def acquire(item):
        name, url = item
        try:
            path = local_media(args.output / "images", name)
            result = fetch(url, path)
            with Image.open(path) as im:
                im.verify()
            return {"image_file": name, "status": "available", **result}
        except Exception as exc:
            return {"image_file": name, "url": url, "status": "failed", "error": repr(exc)}

    records = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for future in as_completed([pool.submit(acquire, item) for item in sorted(unique.items())]):
            records.append(future.result())
            write_json(args.output / "availability.partial.json", sorted(records, key=lambda r: r["image_file"]))
            print(len(records), len(unique), records[-1]["status"], flush=True)
    write_json(args.output / "availability.json", sorted(records, key=lambda r: r["image_file"]))


if __name__ == "__main__":
    main()
