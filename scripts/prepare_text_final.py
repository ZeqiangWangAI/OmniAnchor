"""Freeze text or the prescribed OASIS affect12 test and already-fitted probes."""
import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from omnianchor.io import load_samples, read_json, write_json
from omnianchor.provenance import file_hash


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["test", "config", "verification", "features", "probes", "output", "contract"]:
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--oasis-affect12", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.contract.exists() or args.output.exists():
        raise FileExistsError("Final protocols are immutable.")
    samples = load_samples(args.test/"samples.json")
    if not samples or any(s.metadata["split"] != "test" for s in samples):
        raise ValueError("Require the frozen test population.")
    baselines = ["e5", "qwen-embedding", "qwen-reranker"]
    timing = "Existing text study, no protected-score method selection"
    if args.oasis_affect12:
        pilot = read_json(root/"research/contracts/oasis-affect12-pilot-20260910-01.json")
        if (args.test.resolve() != root/"data/prepared/oasis-20260910-02/prepared/test"
                or len(samples) != 201 or file_hash(args.config) != pilot["spec_sha256"]
                or file_hash(args.verification) != pilot["samples_sha256"]
                or any(len(s.parts) != 1 or s.parts[0].type != "image" for s in samples)):
            raise ValueError("OASIS extension requires the unchanged original201images and affect12 instrument.")
        baselines = ["qwen-embedding", "qwen-reranker"]
        timing = pilot["timing_disclosure"]
    elif any(p.type != "text" for s in samples for p in s.parts):
        raise ValueError("Media require the explicitly prescribed OASIS affect12 mode.")
    verification = load_samples(args.verification)
    if any(s.metadata["split"] not in {"train", "dev"} for s in verification):
        raise ValueError("Numerical gate must use development inputs.")
    for folder in [args.features, args.probes]:
        events = (folder/"events.jsonl").read_text().splitlines()
        if not events or json.loads(events[-1])["status"] != "completed":
            raise ValueError("Feature assembly and probe selection must both complete before final freeze.")
    manifest = read_json(args.probes/"manifest.json")
    development = load_samples(args.test.parent/"train/samples.json") + load_samples(args.test.parent/"dev/samples.json")
    if set(s.id for s in samples) & set(manifest["train_ids"]+manifest["dev_ids"]):
        raise ValueError("Protected row overlap with probe selection.")
    selection = read_json(args.probes/"selection.json")
    feature_info = read_json(args.features/"features.json")
    if set(selection) != set(feature_info):
        raise ValueError("Missing fitted methods.")
    if args.oasis_affect12 and (manifest["task"] != "regression" or manifest["label_columns"] != ["V", "A"]
            or set(feature_info) != {"native-raw_logp", "native-reference_log_ratio", "native-reference_z",
                "qwen-embedding-anchor", "qwen-embedding-original", "qwen-reranker-anchor"}):
        raise ValueError("Require all six fixed affect12 geometries and separate VA prediction.")
    args.output.mkdir(parents=True)
    media = {s.id: [] for s in samples}
    if args.oasis_affect12:
        staged = []
        for sample in samples:
            source = Path(sample.parts[0].path)
            digest = file_hash(source)
            target = args.output/"media"/(digest+source.suffix)
            target.parent.mkdir(exist_ok=True)
            if not target.exists():
                shutil.copyfile(source, target)
            part = sample.parts[0].model_copy(update={"path": str(target.relative_to(args.output))})
            staged.append(sample.model_copy(update={"parts": (part,)}))
            media[sample.id] = [{"part_index": 0, "sha256": digest}]
        write_json(args.output/"samples.json", staged)
    else:
        shutil.copyfile(args.test/"samples.json", args.output/"samples.json")
    shutil.copyfile(args.test/"labels.csv", args.output/"labels.csv")
    sources = ["src/omnianchor/engine.py", "src/omnianchor/backends/hf.py", "src/omnianchor/backends/media.py",
        "src/omnianchor/backends/vision_reuse.py", "src/omnianchor/official_baselines.py",
        "scripts/run_measurement_shard.py", "scripts/run_baseline_shard.py", "scripts/frozen_evaluation.py",
        "configs/models-20260910.json"]
    artifacts = [args.features/"calibration.json", args.features/"features.json", args.probes/"selection.json",
        args.probes/"manifest.json", *[args.probes/name/"probe.joblib" for name in selection]]
    for method in baselines:
        artifacts.append(args.features/f"{method}-identity.json")
    write_json(args.contract, {"status": "frozen", "frozen_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "OASIS affect12 train-fitted predictive validity, separate from direct VA" if args.oasis_affect12 else "One final fixed3 text evaluation; direct ValueEval and fixed train-fitted predictive validity",
        "methods": ["native", *baselines], "baseline_methods": baselines,
        "samples_sha256": file_hash(args.output/"samples.json"), "sample_ids": [s.id for s in samples],
        "media_sha256": media, "config_sha256": file_hash(args.config),
        "labels_sha256": file_hash(args.output/"labels.csv"),
        "scoring_source_sha256": {s: file_hash(root/s) for s in sources},
        "verification_samples": str(args.verification), "verification_samples_sha256": file_hash(args.verification),
        "frozen_artifacts": {str(p): file_hash(p) for p in artifacts},
        "task": manifest["task"], "label_columns": manifest["label_columns"],
        "probe_features": str(args.features), "probes": str(args.probes),
        "feature_columns": {k: v["feature_columns"] for k, v in selection.items()},
        "primary": "macroAP for ValueEval; dimension-wise Spearman for affect; direct and probe results separate",
        "primary_comparators": (["e5-original"] if "e5" in baselines else [])+["qwen-embedding-original", "qwen-reranker-anchor"],
        "inference": "1000 paired source-group draws seed42,95%CI; native raw probe minus declared comparators across dimensions is one BHq.05 family per dataset; ValueEval direct raw minus3anchor baselines is a separate family",
        "controls": "fixed seed42 label shuffle diagnostic, not an exchangeable permutation test; no-content from saved train/development empty controls; undefined correlation stays undefined",
        "conditioning": "fixed train-fitted probes and reference; no dev refit; inference conditional on existing source splits, with crossings reported",
        "source_group_crossings": sorted({s.group_id for s in samples} & {s.group_id for s in development}),
        "method_adaptation": "none using protected scores; no adaptation from OASIS final into text studies",
        "timing_disclosure": timing,
        "analysis_source_sha256": {p: file_hash(root/p) for p in ["scripts/evaluate_text_final.py",
            "src/omnianchor/calibration.py", "src/omnianchor/evaluation.py", "src/omnianchor/analysis.py"]},
        "budget": {"native_max_wall_hours": 3, "baseline_max_wall_hours": 3, "gpu_per_job": 1},
        "run_policy": "one completed evaluation per method; preserve failures and disclose identical-protocol retries"})


if __name__ == "__main__":
    main()
