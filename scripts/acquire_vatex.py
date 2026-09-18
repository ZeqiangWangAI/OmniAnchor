"""Acquire up to 500 validation clips in fixed ID order, keeping failures and source metadata."""
import argparse
import json
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from omnianchor.io import write_json
from omnianchor.provenance import file_hash


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target", type=int, default=500)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg required; no feature-only substitution.")
    records = sorted(json.loads(args.annotations.read_text()), key=lambda r: r["videoID"])
    write_json(args.output / "contract.json", {
        "annotations_sha256": file_hash(args.annotations), "candidate_order": [r["videoID"] for r in records],
        "target_available": args.target, "workers": 4, "per_download_timeout_seconds": 240,
        "selection": "first target successful clips by fixed videoID order, independent of captions/labels/model",
        "recipe": "yt-dlp bestvideo/best; official time range; force keyframes at cuts; remux mp4; no resolution/fps limit",
        "storage_floor_bytes": 50 * 1024 ** 3, "no_model_inference": True})

    def download(row):
        video_id = row["videoID"]
        youtube_id, start, end = video_id.rsplit("_", 2)
        folder = args.output / "attempts" / video_id
        folder.mkdir(parents=True, exist_ok=False)
        command = [sys.executable, "-m", "yt_dlp", "--no-playlist", "--socket-timeout", "20", "--retries", "0",
                   "--fragment-retries", "0", "--format", "bestvideo/best", "--write-info-json",
                   "--download-sections", f"*{int(start)}-{int(end)}", "--force-keyframes-at-cuts",
                   "--remux-video", "mp4", "--output", str(folder / "clip.%(ext)s"),
                   "https://www.youtube.com/watch?v=" + youtube_id]
        write_json(folder / "command.json", command)
        try:
            with (folder / "stdout.log").open("w") as out, (folder / "stderr.log").open("w") as err:
                process = subprocess.run(command, stdout=out, stderr=err, timeout=240, check=False)
            clip = folder / "clip.mp4"
            if process.returncode or not clip.is_file():
                return {"videoID": video_id, "status": "failed", "exit_code": process.returncode}
            import av
            with av.open(str(clip)) as container:
                stream = container.streams.video[0]
                fps = float(stream.average_rate or 0)
                if fps <= 0:
                    raise ValueError("No valid decoded frame rate.")
                first_frame = next(container.decode(stream))
                shape = [first_frame.width, first_frame.height]
            return {"videoID": video_id, "status": "available", "path": str(clip.resolve()),
                    "sha256": file_hash(clip), "bytes": clip.stat().st_size,
                    "fps": fps, "resolution": shape, "source_start": int(start), "source_end": int(end)}
        except Exception as exc:
            return {"videoID": video_id, "status": "failed", "error": repr(exc)}

    results = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for offset in range(0, len(records), 16):
            if shutil.disk_usage(args.output).free < 50 * 1024 ** 3:
                raise RuntimeError("Storage floor reached; retain all attempts and stop acquisition.")
            results.extend(pool.map(download, records[offset:offset + 16]))
            write_json(args.output / "availability.partial.json", results)
            available = [r for r in results if r["status"] == "available"]
            print(json.dumps({"attempted": len(results), "available": len(available)}), flush=True)
            if len(available) >= args.target:
                break
    selected = [r for r in results if r["status"] == "available"][:args.target]
    write_json(args.output / "availability.json", results)
    write_json(args.output / "selected.json", selected)
    if len(selected) < args.target:
        raise SystemExit("Fewer available videos than requested; no replacement dataset used.")


if __name__ == "__main__":
    main()
