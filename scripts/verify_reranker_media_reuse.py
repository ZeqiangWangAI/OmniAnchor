"""Gate once-per-query-list media decoding against both current and prior strict reranker scores."""
import argparse
from copy import deepcopy
import os
from pathlib import Path
from time import perf_counter

import numpy as np

from vlanchor.backends.vision_reuse import reuse_vision_outputs
from vlanchor.campaign import create_run,append_event
from vlanchor.io import load_samples,load_spec,read_json,write_json
from vlanchor.official_baselines import QwenRetrieval
from vlanchor.provenance import file_hash
from vlanchor.types import Anchor
from vision_reuse_admission import admit_vision_reuse


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prior-gate",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Require Slurm CUDA allocation.")
    prior=admit_vision_reuse(args.prior_gate/"summary.json")
    root=Path(__file__).resolve().parents[1]
    samples=[]
    for language in ["emobank-reader","chinese-emobank-sentence"]:
        samples.extend(load_samples(root/"data/prepared/text-smoke-20260910-02"/language/"smoke-samples.json"))
    samples.extend(load_samples(root/"data/prepared/multimodal-smoke-20260910-02/samples.json"))
    if len(samples)!=40 or any(s.metadata["split"] not in {"train","dev"} for s in samples):
        raise ValueError("Require original40development gate inputs.")
    spec=load_spec(root/"configs/smoke/media-qwen35.json")
    anchors=(*spec.anchors[:14],Anchor(id="multi_en",surface="a strong emotion"),Anchor(id="multi_zh",surface="自由與關懷",language="zh"))
    model_id="Qwen/Qwen3-VL-Reranker-2B"
    revision=read_json(root/"configs/models-20260910.json")["models"][model_id]
    create_run(args.output,{"purpose":"Exact reuse of frozen media preprocessing within one query list; no changed pixels/frames/tokens/model",
        "prior_gate":prior,"model_id":model_id,"revision":revision,"samples":[s.id for s in samples],
        "source_sha256":{str(p.relative_to(root)):file_hash(p) for p in
            [Path(__file__),root/"src/vlanchor/official_baselines.py",root/"src/vlanchor/backends/vision_reuse.py"]},
        "probability_tolerance":1e-5,"slurm_job_id":os.environ["SLURM_JOB_ID"],"test_used":False})
    import torch
    torch.manual_seed(42)
    backend=QwenRetrieval(model_id,revision,root/"research/upstream/qwen3-vl-embedding-393e297",spec.resources)
    original_inputs=backend._inputs
    input_evidence=[]

    def inputs(*a,**kw):
        result=original_inputs(*a,**kw)
        input_evidence.append(deepcopy(backend.last_input_evidence))
        return result

    backend._inputs=inputs
    records=[]
    try:
        for i,sample in enumerate(samples):
            results={}
            for enabled in [False,True]:
                backend.reuse_media_preprocessing=enabled
                input_evidence.clear()
                torch.cuda.synchronize()
                start=perf_counter()
                with reuse_vision_outputs(backend.model):
                    values=backend.score(sample,[a.surface for a in anchors],"Retrieve materials related to the concept described by the query.")
                torch.cuda.synchronize()
                results[enabled]=(values,perf_counter()-start,deepcopy(input_evidence))
            if results[False][2]!=results[True][2] or len(results[True][2])!=16:
                raise ValueError("Per-query token/media evidence differs after preprocessing reuse.")
            old=np.load(args.prior_gate/f"reranker-{i:05d}.npz")["original"]
            delta=float(np.max(np.abs(results[False][0]-results[True][0])))
            prior_delta=float(np.max(np.abs(old-results[True][0])))
            row={"sample_id":sample.id,"modality":sample.parts[0].type,"max_abs_current_probability_error":delta,
                "max_abs_original_strict_probability_error":prior_delta,"reference_seconds":results[False][1],
                "reuse_seconds":results[True][1],"passed":max(delta,prior_delta)<=1e-5}
            records.append(row)
            np.savez_compressed(args.output/f"scores-{i:05d}.npz",reference=results[False][0],reused=results[True][0],original_strict=old)
            write_json(args.output/f"inputs-{i:05d}.json",results[True][2])
            append_event(args.output,"comparison",**row)
            print(row,flush=True)
        passed=all(r["passed"] for r in records)
        write_json(args.output/"summary.json",{"status":"passed" if passed else "failed","comparisons":records})
        if not passed:
            raise RuntimeError("Media preprocessing reuse rejected; do not enable production.")
        append_event(args.output,"completed")
    except BaseException as exc:
        append_event(args.output,"failed",error=repr(exc))
        raise


if __name__=="__main__":
    main()
