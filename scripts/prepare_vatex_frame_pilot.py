"""Freeze matched32development-input VATEX frame-count pilots before full allocation."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import shutil

from omnianchor.io import load_samples, load_spec, write_json
from omnianchor.provenance import file_hash


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.contract.exists():
        raise FileExistsError("Frame pilot inputs and contracts are immutable.")
    root = Path(__file__).resolve().parents[1]
    original = root/"data/prepared/vatex-scoring-20260910-01/smoke32.json"
    spec = load_spec(root/"configs/studies/vatex_general128.json")
    samples = load_samples(original)
    if (len(samples) != 32 or any(s.metadata["split"] not in {"train", "dev"} for s in samples)
            or sum(s.parts[0].type == "video" for s in samples) != 16):
        raise ValueError("Require the original16videos+16captions technical cohort.")
    if args.output.resolve().parent != original.parent.parent:
        raise ValueError("Keep the same prepared-data parent so relative media paths remain unchanged.")
    args.output.mkdir(parents=True)
    shutil.copyfile(original, args.output/"samples.json")
    rows = []
    for index, count in enumerate([1, 8, 16]):
        resources = spec.resources.model_copy(update={"video": spec.resources.video.model_copy(update={"sampled_frames": count})})
        variant = spec.model_copy(update={"resources": resources})
        path = args.output/f"spec-{index:05d}.json"
        write_json(path, variant)
        rows.append({"index": index, "frames": count, "config": str(path), "config_sha256": file_hash(path)})
    write_json(args.contract, {"status": "frozen", "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Matched-hardware 1/8/16frame technical/cost verification, not retrieval validity",
        "samples": str(args.output/"samples.json"), "samples_sha256": file_hash(args.output/"samples.json"),
        "source_samples_sha256": file_hash(original), "sample_ids": [s.id for s in samples], "variants": rows,
        "media_sha256": {s.id: file_hash(s.parts[0].path) for s in samples if s.parts[0].type == "video"},
        "methods": ["native", "qwen-embedding", "qwen-reranker"],
        "hardware": "Slurm gpu_5000_ada, allthree frame counts matched; no model/precision/pixel-budget change",
        "controls": "16captioninputs are identical acrossframe conditions; report numerical differences before attempting any text/reference reuse",
        "acceptance": "All32inputs complete, native numerical checks pass, requested1/8/16frames evidenced; no implicit truncation or reduced pixel budget",
        "budget": {"native_tasks": 3, "two_baseline_tasks": 3, "max_wall_hours_each": 3,
            "concurrency_per_array": 1, "full_launch": "Only after measuredpilotcost, memory/storage and full frozen contracts"},
        "shared_prefill": False, "media_preprocessing_reuse": False,
        "vision_reuse_gate": 44786, "test_used": False,
        "timing": "Part of original E4frameablation; no VATEX retrievalmetric has been viewed or used for selection"})


if __name__ == "__main__":
    main()
