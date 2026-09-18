"""Freeze all train-selected bridge sets on the original ValueEval protected population."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import shutil

from omnianchor.io import load_samples, load_spec, read_json, write_json
from omnianchor.provenance import file_hash


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.contract.exists():
        raise FileExistsError("Final evaluation artifacts are immutable.")
    root = Path(__file__).resolve().parents[1]
    bank = root/"data/prepared/selected-bridges-dev-20260910-01"
    data = root/"data/prepared/valueeval-final-20260910-01"
    selection = root/"runs/bridge-search-44688/search/selected.json"
    original = root/"research/contracts/valueeval-final-20260910-01.json"
    c = read_json(original)
    selected = read_json(selection)
    if selected["status"] != "ok" or selected["manifest"]["split"] != "train":
        raise ValueError("Only the original train-only bridge search is eligible.")
    samples = load_samples(data/"samples.json")
    if len(samples) != 1576 or any(s.metadata["split"] != "test" for s in samples):
        raise ValueError("Require complete original ValueEval test population.")
    for name, key in [("samples.json", "samples_sha256"), ("labels.csv", "labels_sha256")]:
        if file_hash(data/name) != c[key]:
            raise ValueError("Original protected data changed.")
    specs = sorted(bank.glob("spec-*.json"))
    if len(specs) != 6 or any(len(load_spec(p).bridges) != 1 for p in specs):
        raise ValueError("Require the six already-frozen new bridge coordinates.")
    args.output.mkdir(parents=True)
    for name in ["samples.json", "labels.csv"]:
        shutil.copyfile(data/name, args.output/name)
    source_names = list(c["scoring_source_sha256"])
    admissions = [file_hash(original)]
    for i, spec in enumerate(specs):
        path = args.output/f"contract-{i:05d}.json"
        write_json(path, {"status": "frozen", "frozen_utc": datetime.now(timezone.utc).isoformat(),
            "purpose": "Full ValueEval final scores for one train-selected bridge coordinate; no new search",
            "methods": ["native"], "samples_sha256": file_hash(args.output/"samples.json"),
            "sample_ids": [s.id for s in samples], "media_sha256": {s.id: [] for s in samples},
            "config_sha256": file_hash(spec), "scoring_source_sha256": {p: file_hash(root/p) for p in source_names},
            "verification_samples": c["verification_samples"], "verification_samples_sha256": c["verification_samples_sha256"],
            "hardware": "Slurm gpu_a5000, matched existing fixed3 native final45631",
            "bridge_bank_sha256": file_hash(bank/"manifest.json"), "shared_prefill": False,
            "budget": {"wall_hours": 2, "gpu": 1}, "selection": "All frozen train-selected objectives, no dev feedback"})
        admissions.append(file_hash(path))
    telemetry = read_json(root/"runs/cost-snapshot-20260910-01/telemetry.json")
    cost_record = next(r for r in telemetry if r["path"] == "runs/development-45631/measurement/cost.json")
    observed = cost_record["value"]
    if (observed["samples"], observed["anchors"], observed["bridges"]) != (1576, 20, 3):
        raise ValueError("Measured cost basis differs from the stated fixed3 experiment.")
    write_json(args.contract, {"status": "frozen", "frozen_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "E3 complete held-out train-selected bridge comparison; all six sets, no test tuning",
        "analysis_source_sha256": file_hash(root/"scripts/analyze_selected_bridges.py"),
        "statistics_source_sha256": {p: file_hash(root/p) for p in ["src/omnianchor/evaluation.py", "src/omnianchor/analysis.py", "src/omnianchor/calibration.py", "src/omnianchor/reliability.py", "src/omnianchor/campaign.py"]},
        "evaluation_files": {p: file_hash(args.output/p) for p in ["samples.json", "labels.csv"]},
        "bank_files": {str(p.relative_to(bank)): file_hash(p) for p in bank.rglob("*") if p.is_file()},
        "selection_sha256": file_hash(selection), "admitted_scoring_contract_sha256": admissions,
        "existing_fixed3_run": 45631, "new_bridge_count": 6, "sample_count": 1576,
        "primary": "Full20anchor macroAP of reference_z; 5predefined sets vs fixed3 in one BHq.05 family, shared1000source bootstrap draws and centered approximate bootstrap p",
        "secondary": "Raw5contrastfamily separate; raw/logratio AP invariance verified; train-eligible16anchor AP, within-recipient AP, template Spearman/alpha reported without new selection",
        "cost_basis": cost_record, "point_incremental_gpu_hours": observed["measurement_seconds"]*2/3600,
        "budget": {"tasks": 6, "wall_hours_per_task": 2, "concurrency": 1},
        "timing": "Train-selected sets were frozen before dev scoring. Final analysis frozen before observing any selected-bridge full-dev or final metric. Existing final fixed3 timing alone informs resource budget.",
        "semantic_limit": read_json(bank/"manifest.json")["semantic_review"],
        "reuse": "Use existing final45631 fixed3 raw outputs; do not rescore or choose among repeated baseline test runs."})
    print("Frozen six new bridge coordinates on1576protected materials; existing fixed3 reused.", flush=True)


if __name__ == "__main__":
    main()
