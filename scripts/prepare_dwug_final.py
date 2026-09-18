"""Freeze full protected DWUG targets and shard admissions without method selection."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import shutil

from vlanchor.io import load_samples, write_json
from vlanchor.provenance import file_hash


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.contract.exists():
        raise FileExistsError("Final protocols and inputs are immutable.")
    root = Path(__file__).resolve().parents[1]
    source = root/"data/prepared/dwug-target-inputs-20260910-01"
    labels = root/"data/prepared/dwug-en3-20260910-02/test"
    config = root/"configs/studies/dwug_general128.json"
    verification = source/"smoke32.json"
    reference = root/"runs/development-44792"
    samples = load_samples(source/"test.json")
    development = load_samples(source/"train.json")+load_samples(source/"dev.json")
    if len(samples) != 1789 or len({s.target.lemma for s in samples}) != 9 or any(s.metadata["split"] != "test" for s in samples):
        raise ValueError("Require all original1789usages from9protected targets.")
    if {s.target.lemma for s in samples} & {s.target.lemma for s in development}:
        raise ValueError("Target lemmas cross protected and development roles.")
    if len(load_samples(verification)) != 32 or any(s.metadata["split"] not in {"train", "dev"} for s in load_samples(verification)):
        raise ValueError("Require frozen32development verification inputs.")
    if (reference/"exit_code.txt").read_text().strip() != "0":
        raise ValueError("Original train reference did not complete.")
    sources = ["src/vlanchor/engine.py", "src/vlanchor/backends/hf.py", "src/vlanchor/backends/media.py",
        "src/vlanchor/backends/vision_reuse.py", "src/vlanchor/official_baselines.py",
        "scripts/run_measurement_shard.py", "scripts/run_baseline_shard.py", "scripts/frozen_evaluation.py",
        "configs/models-20260910.json"]
    args.output.mkdir(parents=True)
    shutil.copyfile(source/"test.json", args.output/"samples.json")
    for name in ["change-labels.csv", "judgments.csv"]:
        shutil.copyfile(labels/name, args.output/name)
    files, contracts = [], {}
    for i, offset in enumerate(range(0, len(samples), 400)):
        selected = samples[offset:offset+400]
        path = args.output/"shards"/f"samples-{i:05d}.json"
        write_json(path, selected)
        files.append({"file": path.name, "sha256": file_hash(path), "sample_ids": [s.id for s in selected]})
        admission = args.output/"shards"/f"contract-{i:05d}.json"
        write_json(admission, {"status": "frozen", "frozen_utc": datetime.now(timezone.utc).isoformat(),
            "purpose": "One shard of complete9target DWUG final evaluation; no target selection",
            "methods": ["native", "e5", "qwen-embedding", "qwen-reranker"],
            "samples_sha256": file_hash(path), "sample_ids": [s.id for s in selected],
            "media_sha256": {s.id: [] for s in selected}, "config_sha256": file_hash(config),
            "scoring_source_sha256": {p: file_hash(root/p) for p in sources},
            "verification_samples": str(verification.relative_to(root)), "verification_samples_sha256": file_hash(verification),
            "hardware": "Slurm gpu_a5000; matched native train/dev/reference hardware",
            "selection": "All test rows, original ordering; shard boundary only for resource allocation",
            "budgets": {"native_wall_hours": 4, "baseline_wall_hours": 1, "gpu_per_job": 1},
            "shared_prefill": False, "model_adaptation": "none; fixed128WiCtrainanchors and means_in_context bridges"})
        contracts[str(admission.relative_to(args.output))] = file_hash(admission)
    write_json(args.output/"shards/manifest.json", {"files": files, "train_samples": 0, "dev_samples": 0,
        "test_samples": len(samples), "parent_samples_sha256": file_hash(args.output/"samples.json")})
    analysis_sources = ["scripts/analyze_dwug.py", "scripts/dwug_statistics.py", "src/vlanchor/evaluation.py", "src/vlanchor/analysis.py"]
    write_json(args.contract, {"status": "frozen", "frozen_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Full held-out DWUG semantic change and usage-relatedness, no model/geometry selection",
        "samples_sha256": file_hash(args.output/"samples.json"), "sample_ids": [s.id for s in samples],
        "targets": sorted({s.target.lemma for s in samples}), "shards_manifest_sha256": file_hash(args.output/"shards/manifest.json"),
        "labels_sha256": {p: file_hash(args.output/p) for p in ["change-labels.csv", "judgments.csv"]},
        "scoring_contract_sha256": contracts, "baseline_methods": ["e5", "qwen-embedding", "qwen-reranker"],
        "feature_assembly_sha256": file_hash(root/"scripts/build_development_features.py"),
        "analysis_source_sha256": {p: file_hash(root/p) for p in analysis_sources},
        "reference_score_sha256": {str(p.relative_to(reference)): file_hash(p) for p in sorted((reference/"measurement").glob("part-*"))},
        "geometry": "All8fixedgeometries; primary native-raw energy distance vs graded human change Spearman; centroid secondary; no selection from dev changes",
        "inference": "1000 within-target period-stratified source-document bootstrap; 1000 genre-stratified whole-document period permutations (conditional exchangeability); 1000 target-cluster bootstrap across9targets. Primary3comparator energy-rho difference family BHq.05. Within-method target permutation families separately corrected; no posthoc expansion.",
        "controls": "100 row-count-balanced draws,100within-periodsourcehalfsplits; raw/logratio Euclidean invariance",
        "usage_pairs": "Exclude undecidable0, mean ratings per unorderedpair; pooled Spearman with whole-target bootstrap, report conditional-corpus/shared-document/shared-annotator and9target limitations",
        "source_documents_crossing_development": sorted({s.metadata['sampling_unit'] for s in samples} & {s.metadata['sampling_unit'] for s in development}),
        "cost_basis": "Completed400native shards44793/44794/44919/44924/45184/45186 approximately2h46-2h55; full400baseline shards about22min; no pilot-based model/input reduction",
        "budget": {"native_shards": len(files), "native_wall_hours_each": 4, "baseline_shards": len(files), "baseline_wall_hours_each": 1, "native_concurrency": 2, "baseline_concurrency": 1},
        "timing": "Scoring/analysis frozen before viewing any fulldevDWUG change/pair result or protectedDWUG score. Earlier FMAT/OASIS results known; no prospective registration claim.",
        "legal": "Modified marked DWUG context redistribution restricted by original license; provide acquisition/marking scripts and numerical outputs rather than republishing source contexts."})
    print(f"Frozen {len(samples)} usages,9targets,{len(files)} scoring shards.", flush=True)


if __name__ == "__main__":
    main()
