"""Freeze a disjoint final OASIS evaluation with unchanged direct VA definitions."""
import argparse
import shutil
from datetime import datetime,timezone
from pathlib import Path

from omnianchor.io import load_samples,write_json
from omnianchor.provenance import file_hash


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--contract",type=Path,required=True)
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    args.output.mkdir(parents=True,exist_ok=False)
    if args.contract.exists():
        raise FileExistsError("Never overwrite a frozen final protocol.")
    samples=load_samples(args.data/"test/samples.json")
    if len(samples)!=201 or any(s.metadata["split"]!="test" for s in samples):
        raise ValueError("Require original201testimage cohort.")
    development=load_samples(args.data/"train/samples.json")+load_samples(args.data/"dev/samples.json")
    if {s.group_id for s in samples}&{s.group_id for s in development}:
        raise ValueError("Source groups cross final/development roles.")
    staged,media=[],{}
    for sample in samples:
        parts=[]
        records=[]
        for i,part in enumerate(sample.parts):
            if part.path:
                source=Path(part.path)
                digest=file_hash(source)
                target=args.output/"media"/(digest+source.suffix)
                target.parent.mkdir(parents=True,exist_ok=True)
                if not target.exists():
                    shutil.copyfile(source,target)
                part=part.model_copy(update={"path":str(target.relative_to(args.output))})
                records.append({"part_index":i,"sha256":digest})
            parts.append(part)
        staged.append(sample.model_copy(update={"parts":tuple(parts)}))
        media[sample.id]=records
    write_json(args.output/"samples.json",staged)
    shutil.copyfile(args.data/"test/labels.csv",args.output/"labels.csv")
    config=root/"configs/studies/oasis_direct_va.json"
    verification="data/prepared/oasis-affect-20260910-01/smoke32.json"
    sources=["src/omnianchor/engine.py","src/omnianchor/backends/hf.py","src/omnianchor/backends/media.py",
        "src/omnianchor/backends/vision_reuse.py","src/omnianchor/official_baselines.py",
        "scripts/run_measurement_shard.py","scripts/run_baseline_shard.py","scripts/frozen_evaluation.py",
        "scripts/vision_reuse_admission.py","configs/models-20260910.json"]
    write_json(args.contract,{"status":"frozen","frozen_utc":datetime.now(timezone.utc).isoformat(),
        "purpose":"Independent final201image direct VA validity; no prediction layer or method search",
        "methods":["native","qwen-embedding","qwen-reranker"],
        "samples_sha256":file_hash(args.output/"samples.json"),"sample_ids":[s.id for s in samples],
        "media_sha256":media,"config_sha256":file_hash(config),
        "labels_sha256":file_hash(args.output/"labels.csv"),"scoring_source_sha256":{s:file_hash(root/s) for s in sources},
        "verification_samples":verification,"verification_samples_sha256":file_hash(root/verification),
        "criterion":"published image-level VA human ratings, signed Pearson primary; Spearman secondary",
        "contrasts":{"V":"mean_bridge(logp(pleasant)-logp(unpleasant))","A":"mean_bridge(logp(aroused)-logp(calm))"},
        "baseline_contrasts":"samefouranchor positive-minusnegative cosine/relevance scores",
        "inference":"1000pairedimagebootstrap,95%CI;4primarynative-minusbaselinePearsondifferences,familyBHq.05",
        "method_adaptation":"none after32pilot; testneverusedtochoose anchors, signs, bridges, weights, or model",
        "development_full_results":"not yet available at freeze; raw directmethodsneed no training",
        "run_policy":"one completed evaluation per method; failed attempts remain immutable and retries must retain identical protocol",
        "budget":{"native_wall_hours":2,"both_baselines_wall_hours":2,"gpus_per_job":1},
        "scope":"OASIS directVA only; other studies retain separate protected protocols"})


if __name__=="__main__":
    main()
