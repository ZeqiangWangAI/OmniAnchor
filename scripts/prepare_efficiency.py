"""Prepare the two missing N=16/64 points of the fixed 16/64/128/256 cost sweep."""
import argparse
from pathlib import Path

from vlanchor.io import load_samples, load_spec, read_json, write_json
from vlanchor.provenance import file_hash


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sensitivity", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    parent = read_json(args.sensitivity/"manifest.json")
    for name in ["dwug/base.json", "dwug/smoke32.json", "dwug/general256.json"]:
        if file_hash(args.sensitivity/name) != parent["files_sha256"][name]:
            raise ValueError("E3 inputs or fixed anchor bank changed.")
    spec = load_spec(args.sensitivity/"dwug/base.json")
    full = load_spec(args.sensitivity/"dwug/general256.json")
    samples = load_samples(args.sensitivity/"dwug/smoke32.json")
    if len(samples) != 32 or any(s.metadata["split"] not in {"train", "dev"} for s in samples):
        raise ValueError("Cost estimation requires the same32development materials.")
    if spec.anchors != full.anchors[:128]:
        raise ValueError("General banks are not nested.")
    args.output.mkdir(parents=True, exist_ok=False)
    rows = []
    for i, n in enumerate([16, 64]):
        config, inputs = args.output/f"spec-{i:05d}.json", args.output/f"samples-{i:05d}.json"
        write_json(config, spec.model_copy(update={"anchors": full.anchors[:n]}))
        write_json(inputs, samples)
        rows.append({"index": i, "anchors": n, "spec_sha256": file_hash(config), "samples_sha256": file_hash(inputs)})
    write_json(args.output/"manifest.json", {"status": "frozen", "rows": rows,
        "source_sha256": file_hash(Path(__file__)), "parent_sha256": file_hash(args.sensitivity/"manifest.json"),
        "reuse_other_points": {"128": {"e3_array": 45994, "task_index": 6}, "256": {"e3_array": 45994, "task_index": 10}},
        "purpose": "Prespecified N16/64/128/256 efficiency sweep only; no scientific label reads or method selection",
        "hardware": "gpu_5000_ada; same Qwen35 BF16,32materials,3bridges,resources asE3",
        "budget": {"new_tasks": 2, "max_concurrent": 1, "wall_hours_per_task": 1},
        "report": "Measured preprocessing/forward/total seconds,allocated/reserved GPU bytes,items and token lengths; allocated Slurm cost separately. One observed32sample run perN, not a timing confidence interval. Nested anchors vary in token lengths, so not an isolated causal N effect."})


if __name__ == "__main__":
    main()
