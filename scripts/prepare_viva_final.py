"""Freeze the original216 VIVA test recipients and their same-split image controls."""
import argparse
import shutil
from datetime import datetime, timezone
from pathlib import Path

from vlanchor.io import load_samples, read_json, write_json
from vlanchor.provenance import file_hash


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["data", "labels", "output", "contract"]:
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.contract.exists():
        raise FileExistsError("Frozen protocols must not be overwritten.")
    root = Path(__file__).resolve().parents[1]
    samples = load_samples(args.data/"test.json")
    development = load_samples(args.data/"development.json")
    if len(samples) != 216 or any(s.metadata["split"] != "test" for s in samples):
        raise ValueError("Require exact original216test cohort.")
    if {s.group_id for s in samples} & {s.group_id for s in development}:
        raise ValueError("Image-group leakage.")
    donors = read_json(args.data/"donors.json")
    ids = {s.id: s for s in samples}
    for sample in samples:
        if donors[sample.id] not in ids or ids[donors[sample.id]].group_id == sample.group_id:
            raise ValueError("Invalid frozen same-split donor.")
    args.output.mkdir(parents=True)
    staged, media = [], {}
    for sample in samples:
        parts, records = [], []
        for i, part in enumerate(sample.parts):
            if part.path:
                path = Path(part.path)
                records.append({"part_index": i, "sha256": file_hash(path)})
                part = part.model_copy(update={"path": "../"+str(path.relative_to(args.output.parent.resolve()))})
            parts.append(part)
        staged.append(sample.model_copy(update={"parts": tuple(parts)}))
        media[sample.id] = records
    write_json(args.output/"samples.json", staged)
    shutil.copyfile(args.labels, args.output/"labels.csv")
    config = root/"configs/studies/valueeval_fixed3.json"
    verification = args.data/"smoke32.json"
    sources = ["src/vlanchor/engine.py", "src/vlanchor/backends/hf.py", "src/vlanchor/backends/media.py",
        "src/vlanchor/backends/vision_reuse.py", "src/vlanchor/official_baselines.py",
        "scripts/run_viva_shard.py", "scripts/run_measurement_shard.py", "scripts/frozen_evaluation.py",
        "scripts/verify_native.py", "scripts/vision_reuse_admission.py", "configs/models-20260910.json"]
    write_json(args.contract, {"status": "frozen", "frozen_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Final VIVA action-conditioned value ranking, unchanged predefined methods and image controls",
        "methods": ["native", "qwen-embedding", "qwen-reranker"],
        "samples_sha256": file_hash(args.output/"samples.json"), "sample_ids": [s.id for s in samples],
        "media_sha256": media, "config_sha256": file_hash(config), "labels_sha256": file_hash(args.output/"labels.csv"),
        "scoring_source_sha256": {p: file_hash(root/p) for p in sources},
        "verification_samples": str(verification), "verification_samples_sha256": file_hash(verification),
        "viva_input_sha256": {p: file_hash(args.data/p) for p in ["candidates.json", "donors.json"]},
        "conditions": ["image_action", "action_only", "mismatched_image_action"],
        "primary": "mean per-recipient AP; MRRsecondary withstablepubliccandidateorderties",
        "inference": "1000recipientimagegroupbootstrap seed42,95%CI conditional onfrozendonors;4primaryAPcontrasts(nativeimage minusnativeactiononly,nativewrongimage,embeddingimage,rerankerimage),centeredpairedbootstrap,BHq.05",
        "annotation_origin": "published model-assisted annotations verified/revised by humans, not newly collected human-onlyratings",
        "method_adaptation": "none afterpilotnegativeimageincrement; full850devresultsnotyetavailable; all3methods/all3conditionsreported",
        "hardware_constraint": "gpu_5000_ada", "gpu_name_contains": "5000 Ada",
        "acceleration": "only exact independentvisionreuse admitted44786 onAda;noKV/prefill/no media-preprocessingreuse",
        "budget": {"tasks": 3, "wall_hours_per_task": 3, "max_concurrent": 1},
        "run_policy": "one completed run per method; failed attempts preserved and identical-protocol repairs disclosed"})


if __name__ == "__main__":
    main()
