"""Action-conditioned VIVA ranking under paired image-removal and image-swap controls."""
import argparse
from contextlib import nullcontext
import os
from pathlib import Path
from time import perf_counter

import numpy as np

from omnianchor import measure
from omnianchor.backends.vision_reuse import reuse_vision_outputs
from omnianchor.campaign import create_run, append_event
from omnianchor.io import load_samples,load_spec,read_json,save_scores,write_json
from omnianchor.official_baselines import QwenRetrieval
from omnianchor.provenance import file_hash,runtime_manifest
from omnianchor.types import Anchor,Sample,Part
from run_measurement_shard import TimedBackend
from verify_native import verify_native
from vision_reuse_admission import admit_vision_reuse
from frozen_evaluation import validate_frozen_evaluation
from viva_conditions import condition_sample


class ConditionBackend(TimedBackend):
    condition="image_action"

    @property
    def identity(self):
        return {**super().identity,"viva_condition":self.condition}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data",type=Path,required=True)
    parser.add_argument("--samples",type=Path,required=True)
    parser.add_argument("--method",choices=["native","qwen-embedding","qwen-reranker"],required=True)
    parser.add_argument("--vision-reuse-gate",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--frozen-evaluation",type=Path)
    parser.add_argument("--interaction-controls",action="store_true")
    args=parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Require Slurm CUDA.")
    samples=load_samples(args.samples)
    root=Path(__file__).resolve().parents[1]
    config=root/"configs/studies/valueeval_fixed3.json"
    final=(validate_frozen_evaluation(args.frozen_evaluation,args.samples,config,args.method)
           if args.frozen_evaluation else None)
    if not final and any(s.metadata["split"] not in {"train","dev"} for s in samples):
        raise ValueError("Development driver rejects test.")
    if final:
        for name,digest in final["protocol"]["viva_input_sha256"].items():
            if file_hash(args.data/name)!=digest:
                raise ValueError("VIVA candidate/donor inventory changed after freeze.")
    inventory={s.id:s for s in (samples if final else load_samples(args.data/"development.json"))}
    donors=read_json(args.data/"donors.json")
    candidates=read_json(args.data/"candidates.json")
    spec=load_spec(config)
    conditions=(["image_only","empty"] if args.interaction_controls else
                ["image_action","action_only","mismatched_image_action"])
    if final and conditions!=final["protocol"]["conditions"]:
        raise ValueError("Conditions differ from the frozen scoring protocol.")
    create_run(args.output,{"purpose":"VIVA direct candidate ranking, no probe or common concept matrix",
        "method":args.method,"conditions":conditions,"sample_ids":[s.id for s in samples],
        "vision_reuse":admit_vision_reuse(args.vision_reuse_gate),"spec":spec.model_dump(),
        "input_sha256":{str(p):file_hash(p) for p in [args.samples,args.data/"candidates.json",args.data/"donors.json"]},
        "source_sha256":file_hash(Path(__file__)),"slurm_job_id":os.environ["SLURM_JOB_ID"],
        "known_input":"recipient correct action removed only in named image_only/empty controls; never reason or value answer explanation", "test_used":bool(final),
        "condition_source_sha256":file_hash(Path(__file__).with_name("viva_conditions.py")),
        "frozen_evaluation":final})
    import torch
    if final and final["protocol"]["gpu_name_contains"] not in torch.cuda.get_device_name():
        raise RuntimeError("Allocated GPU differs from the frozen hardware requirement.")
    torch.manual_seed(42)
    if args.method=="native":
        backend=ConditionBackend(spec.model,spec.resources,shared_prefill=False,reuse_vision_features=True)
        backend._ensure_loaded()
        verification_samples=load_samples(root/final["protocol"]["verification_samples"]) if final else samples
        if any(s.metadata["split"] not in {"train","dev"} for s in verification_samples):
            raise ValueError("Native verification must use development inputs.")
        check=verify_native(backend,output=args.output/"native-verification.json",image_sample=verification_samples[0],
            anchors=[Anchor(id="m1",surface="a strong emotion"),Anchor(id="m2",surface="an emotional response")])
        if check["status"]!="passed":
            raise RuntimeError("Native verification failed.")
    else:
        model_id="Qwen/Qwen3-VL-Embedding-2B" if args.method=="qwen-embedding" else "Qwen/Qwen3-VL-Reranker-2B"
        revisions=read_json(root/"configs/models-20260910.json")["models"]
        backend=QwenRetrieval(model_id,revisions[model_id],root/"research/upstream/qwen3-vl-embedding-393e297",spec.resources)
    write_json(args.output/"model-identity.json",backend.identity if args.method=="native" else {
        "model_id":model_id,"revision":revisions[model_id],"resources":spec.resources.model_dump(),
        "runtime":runtime_manifest(),"precision":"BF16;FP32pooling/normalization",
        "preprocessing":"strictbudget-controlledofficialrecipe,noNULLfallback/noimplicittruncation",
        "source_hashes":{str(p.relative_to(root)):file_hash(p) for p in
            [root/"src/omnianchor/official_baselines.py",root/"src/omnianchor/backends/vision_reuse.py"]}})
    started=perf_counter()
    costs={c:0. for c in conditions}
    try:
        for i,sample in enumerate(samples):
            donor=inventory[donors[sample.id]] if "mismatched_image_action" in conditions else None
            anchors=tuple(Anchor.model_validate(a) for a in candidates[sample.id])
            anchor_features=None
            if args.method=="qwen-embedding":
                anchor_features=backend.encode([Sample(id=a.id,parts=(Part(type="text",text=a.surface),)) for a in anchors])
            for condition in conditions:
                item=condition_sample(sample,condition,donor)
                before=perf_counter()
                path=args.output/condition/f"part-{i:05d}"
                if args.method=="native":
                    backend.condition=condition
                    table=measure([item],spec.model_copy(update={"anchors":anchors}),backend=backend)
                    save_scores(table,path.with_suffix(".parquet"))
                    if table.manifest["execution"]["failed_items"]:
                        raise RuntimeError("Native failure; preserve complete paired denominator.")
                else:
                    if args.method=="qwen-embedding":
                        values=backend.encode([item])[0]@anchor_features.T
                    else:
                        with reuse_vision_outputs(backend.model) if args.vision_reuse_gate else nullcontext():
                            values=backend.score(item,[a.surface for a in anchors],"Retrieve materials that express the human value described by the query.")
                    if not np.isfinite(values).all():
                        raise ValueError("Nonfinite candidate scores.")
                    path.parent.mkdir(parents=True,exist_ok=True)
                    np.savez_compressed(path.with_suffix(".npz"),sample_id=sample.id,anchor_ids=np.array([a.id for a in anchors]),scores=values)
                torch.cuda.synchronize()
                costs[condition]+=perf_counter()-before
            append_event(args.output,"sample_completed",sample_id=sample.id)
            print(i+1,len(samples),sample.id,flush=True)
        write_json(args.output/"cost.json",{"measurement_seconds":perf_counter()-started,"condition_seconds":costs,
            "samples":len(samples),"gpu":torch.cuda.get_device_name(),"peak_allocated_bytes":torch.cuda.max_memory_allocated()})
        append_event(args.output,"completed")
    except BaseException as exc:
        append_event(args.output,"failed",error=repr(exc))
        raise


if __name__=="__main__":
    main()
